import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { AnalyzeEventsRequest, FrameData, MatchEvent, TeamId } from "@/lib/types";

const EVENT_MODEL = process.env.ANTHROPIC_EVENT_MODEL ?? process.env.ANTHROPIC_SUMMARY_MODEL ?? "claude-sonnet-4-6";

interface RawEvent {
  frameIndex: number;
  type: MatchEvent["type"];
  team?: TeamId;
  description: string;
  confidence?: number;
  position?: { x: number; y: number };
  semantic_label?: string | null;
  evidence_used?: string[];
  conflicts?: string[];
}

interface RawFrameUpdate {
  frameIndex: number;
  possession?: TeamId | "contested";
  events?: RawEvent[];
}

interface RawEventReview {
  frames?: RawFrameUpdate[];
}

function cleanJson(text: string) {
  return text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
}

function eventId(frameIndex: number, eventIndex: number, type: string) {
  return `f${frameIndex}-review-${eventIndex}-${type}`;
}

function isKeyMoment(type: MatchEvent["type"]) {
  return ["goal", "shot", "save", "corner", "freekick", "foul", "offside", "card_yellow", "card_red", "card_unknown"].includes(type);
}

function normalizeEventType(type: string): MatchEvent["type"] | null {
  const map: Record<string, MatchEvent["type"]> = {
    goal: "goal",
    shot: "shot",
    shot_saved: "save",
    save: "save",
    shot_off_target: "shot",
    corner: "corner",
    goal_kick: "goal-kick",
    "goal-kick": "goal-kick",
    card_yellow: "card_yellow",
    yellow_card: "card_yellow",
    card_red: "card_red",
    red_card: "card_red",
    card_unknown: "card_unknown",
    foul: "foul",
    dangerous_foul: "foul",
    freekick: "freekick",
    free_kick: "freekick",
    offside: "offside",
    pass: "pass",
    tackle: "tackle",
    "throw-in": "throw-in",
    throw_in: "throw-in",
    dribble: "dribble",
  };
  return map[type] ?? null;
}

function mergeReviewedFrames(frames: FrameData[], review: RawEventReview): FrameData[] {
  const updates = new Map((review.frames ?? []).map((frame) => [frame.frameIndex, frame]));

  return frames.map((frame) => {
    const update = updates.get(frame.frameIndex);
    if (!update) return frame;

    const reviewedEvents: MatchEvent[] = (update.events ?? []).flatMap((event, i) => {
      const type = normalizeEventType(event.type);
      if (!type) return [];
      return [{
        id: eventId(frame.frameIndex, i, type),
        timestamp: frame.timestamp,
        type,
        team: event.team,
        description: event.description,
        confidence: event.confidence ?? 0.75,
        position: event.position,
        isKeyMoment: isKeyMoment(type),
        semanticLabel: event.semantic_label ?? undefined,
        evidenceUsed: event.evidence_used,
        conflicts: event.conflicts,
      }];
    });

    return {
      ...frame,
      possession: update.possession ?? frame.possession,
      events: [...frame.events, ...reviewedEvents],
    };
  });
}

function nearestPlayers(frame: FrameData, count = 4) {
  if (!frame.ballPosition) return [];
  return frame.players
    .map((p) => ({
      id: p.id,
      team: p.team,
      role: p.role,
      position: p.position,
      distance: Math.hypot(
        p.position.x - frame.ballPosition!.x,
        p.position.y - frame.ballPosition!.y
      ),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, count);
}

function candidateWindows(frames: FrameData[]) {
  const candidates: Array<{
    candidate_type: string;
    frameIndex: number;
    window: { start_timestamp: number; end_timestamp: number };
    signals: Record<string, unknown>;
  }> = [];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    const prev = frames[i - 1];
    const next = frames[i + 1];
    const ball = frame.ballPosition;
    const nearest = nearestPlayers(frame, 3);
    const window = {
      start_timestamp: Math.max(0, prev?.timestamp ?? frame.timestamp - 4),
      end_timestamp: next?.timestamp ?? frame.timestamp + 4,
    };

    if (ball) {
      const inGoalZone = ball.x <= 8 || ball.x >= 92;
      const inShotZone = ball.x <= 22 || ball.x >= 78;
      const nearCorner = (ball.x <= 10 || ball.x >= 90) && (ball.y <= 12 || ball.y >= 88);

      if (inGoalZone) {
        candidates.push({
          candidate_type: "possible_goal_or_save",
          frameIndex: frame.frameIndex,
          window,
          signals: {
            ball_entered_goal_zone: { detected: true, side: ball.x <= 8 ? "left" : "right" },
            possession_team: frame.possession,
            nearest_players: nearest,
          },
        });
      } else if (inShotZone) {
        candidates.push({
          candidate_type: "possible_shot",
          frameIndex: frame.frameIndex,
          window,
          signals: {
            ball_in_attacking_third: true,
            possession_team: frame.possession,
            nearest_players: nearest,
          },
        });
      }

      if (nearCorner) {
        candidates.push({
          candidate_type: "possible_corner_or_goal_kick",
          frameIndex: frame.frameIndex,
          window,
          signals: {
            ball_near_corner_or_byline: true,
            possession_team: frame.possession,
            nearest_players: nearest,
          },
        });
      }
    }

    if (prev && prev.possession !== "contested" && frame.possession !== "contested" && prev.possession !== frame.possession) {
      candidates.push({
        candidate_type: "possible_turnover_or_tackle",
        frameIndex: frame.frameIndex,
        window,
        signals: {
          possession_changed_from: prev.possession,
          possession_changed_to: frame.possession,
          nearest_players: nearest,
        },
      });
    }
  }

  return candidates.slice(0, 12);
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "ANTHROPIC_API_KEY not configured." }, { status: 500 });
    }

    const body = (await req.json()) as AnalyzeEventsRequest;
    const { frames, images } = body;
    if (!frames?.length || !images?.length) {
      return NextResponse.json({ error: "Frames and images are required." }, { status: 400 });
    }

    const compactFrames = frames.map((frame) => ({
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      possession: frame.possession,
      ballPosition: frame.ballPosition,
      playerCount: frame.players.length,
      homePlayers: frame.players.filter((p) => p.team === "home").length,
      awayPlayers: frame.players.filter((p) => p.team === "away").length,
      playersNearBall: nearestPlayers(frame),
    }));
    const candidates = candidateWindows(frames);

    type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
    type TextBlock = { type: "text"; text: string };
    const content: Array<ImageBlock | TextBlock> = [
      {
        type: "text",
        text: `You are a soccer match analysis verifier for fixed wide-angle tactical camera footage.
You operate as a SPARSE VERIFIER and SEMANTIC LABELER only.

Upstream already ran:
- YOLO football detection for players and ball
- clip-level jersey color clustering for team assignment
- nearest-player possession estimate per sampled frame
- simple deterministic candidate generation from ball location and possession changes

Your job:
- Verify candidate windows using the images and YOLO metadata.
- Correct possession only when the visual evidence is clearer than YOLO's nearest-player estimate.
- Label sparse high-value events for the timeline.
- Do NOT enumerate dense events from scratch.
- Low confidence beats a wrong confident answer.

Return ONLY valid JSON:
{
  "frames": [
    {
      "frameIndex": 0,
      "possession": "home",
      "events": [
        {
          "frameIndex": 0,
          "type": "shot",
          "team": "home",
          "description": "Home attacker shoots from the edge of the box",
          "confidence": 0.82,
          "position": { "x": 78, "y": 48 },
          "semantic_label": "direct_play",
          "evidence_used": ["ball in attacking third", "nearest home player at ball"],
          "conflicts": []
        }
      ]
    }
  ]
}

Rules:
- possession must be "home", "away", or "contested".
- Allowed event types: goal, shot, shot_saved, shot_off_target, corner, goal_kick, card_yellow, card_red, card_unknown, foul, freekick, offside, pass, tackle, throw-in, dribble.
- Only report events visible or strongly implied inside candidate windows.
- Do not create an event for every frame.
- If no candidate is convincing, return an empty events array for that frame.

Goal rules:
- Do NOT confirm a goal from one weak signal.
- Strong confirmation: visible ball in net, keeper retrieving from net, clear scoreboard score change, or obvious non-half-boundary kickoff aftermath.
- Supporting signals: ball in goal zone, ball disappears in goalmouth, celebration cluster near goal, defending shape collapse.
- Rejection signals: keeper catches/parries, ball visible wide/above goal, restart appears to be corner/goal kick, no score/restart evidence.
- Never confirm a goal solely from celebration or pressure near the box.
- If ambiguous, label shot or no event with lower confidence rather than goal.

Semantic labels after confirmation:
- counterattack_goal
- set_piece_goal
- sustained_buildup
- high_press_turnover
- direct_play
- null when unsupported

YOLO metadata:
${JSON.stringify(compactFrames)}

Candidate windows:
${JSON.stringify(candidates)}`,
      },
    ];

    for (const image of images.slice(0, frames.length)) {
      content.push({
        type: "image",
        source: {
          type: "base64",
          media_type: "image/jpeg",
          data: image.base64,
        },
      });
      content.push({
        type: "text",
        text: `Frame timestamp: ${image.timestamp.toFixed(1)}s`,
      });
    }

    const client = new Anthropic({ apiKey });
    const response = await client.messages.create({
      model: EVENT_MODEL,
      max_tokens: 3500,
      messages: [{ role: "user", content }],
    });

    const raw = response.content[0].type === "text" ? response.content[0].text : "{}";
    const parsed = JSON.parse(cleanJson(raw)) as RawEventReview;
    return NextResponse.json({ frames: mergeReviewedFrames(frames, parsed) });
  } catch (err) {
    console.error("[/api/analyze/events]", err);
    return NextResponse.json({ error: "Event review failed." }, { status: 500 });
  }
}
