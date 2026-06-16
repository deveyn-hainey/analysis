import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { AnalyzeEventsRequest, FrameData, MatchEvent, TeamId } from "@/lib/types";

const EVENT_MODEL = process.env.ANTHROPIC_EVENT_MODEL ?? process.env.ANTHROPIC_SUMMARY_MODEL ?? "claude-sonnet-4-6";

interface RawEvent {
  frameIndex: number;
  timestamp?: number;
  type: string;
  team?: TeamId;
  description: string;
  confidence?: number;
  position?: { x: number; y: number };
  semantic_label?: string | null;
  evidence_used?: string[];
  conflicts?: string[];
  pipeline_flag?: MatchEvent["pipelineFlag"];
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

function eventId(frameIndex: number, eventIndex: number, type: string, timestamp: number) {
  return `f${frameIndex}-review-${eventIndex}-${Math.round(timestamp * 10)}-${type}`;
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
  const frameTimestamps = frames.map((frame) => frame.timestamp);

  return frames.map((frame) => {
    const update = updates.get(frame.frameIndex);
    if (!update) return frame;

    const reviewedEvents: MatchEvent[] = (update.events ?? []).flatMap((event, i) => {
      const type = normalizeEventType(event.type);
      if (!type) return [];
      const rawTimestamp =
        typeof event.timestamp === "number" && Number.isFinite(event.timestamp)
          ? event.timestamp
          : frame.timestamp;
      const closestTimestamp = frameTimestamps.reduce((best, timestamp) =>
        Math.abs(timestamp - rawTimestamp) < Math.abs(best - rawTimestamp) ? timestamp : best
      , frame.timestamp);
      const timestamp = Math.abs(rawTimestamp - closestTimestamp) <= 4 ? closestTimestamp : frame.timestamp;
      return [{
        id: eventId(frame.frameIndex, i, type, timestamp),
        timestamp,
        type,
        team: event.team,
        description: event.description,
        confidence: event.confidence ?? 0.75,
        position: event.position,
        isKeyMoment: isKeyMoment(type),
        semanticLabel: event.semantic_label ?? undefined,
        evidenceUsed: event.evidence_used,
        conflicts: event.conflicts,
        pipelineFlag: event.pipeline_flag,
      }];
    });

    return {
      ...frame,
      possession: update.possession ?? frame.possession,
      events: [...frame.events, ...reviewedEvents],
    };
  });
}

function eventKey(event: MatchEvent) {
  return `${Math.round(event.timestamp)}-${event.team ?? "none"}-${event.type}`;
}

function synthesizeMovementEvents(frames: FrameData[]): FrameData[] {
  const existing = new Set(frames.flatMap((f) => f.events.map(eventKey)));

  return frames.map((frame, i) => {
    const prev = frames[i - 1];
    const additions: MatchEvent[] = [];

    if (prev && prev.ballPosition && frame.ballPosition) {
      const distance = Math.hypot(
        frame.ballPosition.x - prev.ballPosition.x,
        frame.ballPosition.y - prev.ballPosition.y
      );
      const sameTeamPossession =
        frame.possession !== "contested" &&
        prev.possession === frame.possession;
      const possessionTurnover =
        prev.possession !== "contested" &&
        frame.possession !== "contested" &&
        prev.possession !== frame.possession;

      if (sameTeamPossession && distance >= 10) {
        const type: MatchEvent["type"] = distance >= 22 ? "pass" : "dribble";
        const team = frame.possession as TeamId;
        const event: MatchEvent = {
          id: `f${frame.frameIndex}-synth-${type}`,
          timestamp: frame.timestamp,
          type,
          team,
          description:
            type === "pass"
              ? `${team === "home" ? "Home" : "Away"} ball progression detected from YOLO ball movement`
              : `${team === "home" ? "Home" : "Away"} controlled carry detected from YOLO ball movement`,
          confidence: 0.45,
          isKeyMoment: false,
          position: frame.ballPosition,
          evidenceUsed: [`ball moved ${distance.toFixed(1)} pitch units while possession stayed ${frame.possession}`],
          conflicts: ["synthetic event from tracking, not visually verified by Claude"],
        };
        if (!existing.has(eventKey(event))) additions.push(event);
      }

      if (possessionTurnover) {
        const team = frame.possession as TeamId;
        const event: MatchEvent = {
          id: `f${frame.frameIndex}-synth-turnover`,
          timestamp: frame.timestamp,
          type: "tackle",
          team,
          description: `${team === "home" ? "Home" : "Away"} regain detected from possession change`,
          confidence: 0.42,
          isKeyMoment: false,
          position: frame.ballPosition,
          semanticLabel: "high_press_turnover",
          evidenceUsed: [`possession changed from ${prev.possession} to ${frame.possession}`],
          conflicts: ["classified as tackle/turnover from tracking signal"],
        };
        if (!existing.has(eventKey(event))) additions.push(event);
      }
    }

    return additions.length
      ? { ...frame, events: [...frame.events, ...additions] }
      : frame;
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

      if (prev?.ballPosition && next?.ballPosition) {
        const v1 = {
          x: ball.x - prev.ballPosition.x,
          y: ball.y - prev.ballPosition.y,
        };
        const v2 = {
          x: next.ballPosition.x - ball.x,
          y: next.ballPosition.y - ball.y,
        };
        const dot = v1.x * v2.x + v1.y * v2.y;
        const nearBox = ball.x <= 28 || ball.x >= 72;
        if (nearBox && dot < -40) {
          candidates.push({
            candidate_type: "possible_shot_save_or_deflection",
            frameIndex: frame.frameIndex,
            window,
            signals: {
              sudden_ball_direction_reversal_near_box: true,
              possession_team: frame.possession,
              nearest_players: nearest,
            },
          });
        }
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

    if (!ball && frame.players.length > 0) {
      candidates.push({
        candidate_type: "possible_open_play_ball_not_detected",
        frameIndex: frame.frameIndex,
        window,
        signals: {
          ball_detected: false,
          yolo_player_count: frame.players.length,
          yolo_possession_estimate: frame.possession,
          instruction: "Use the image to correct possession and label only obvious high-value events.",
        },
      });
    }

    if (prev) {
      const playerCountDelta = Math.abs(frame.players.length - prev.players.length);
      const centroid = frame.players.length
        ? {
            x: frame.players.reduce((sum, player) => sum + player.position.x, 0) / frame.players.length,
            y: frame.players.reduce((sum, player) => sum + player.position.y, 0) / frame.players.length,
          }
        : null;
      const prevCentroid = prev.players.length
        ? {
            x: prev.players.reduce((sum, player) => sum + player.position.x, 0) / prev.players.length,
            y: prev.players.reduce((sum, player) => sum + player.position.y, 0) / prev.players.length,
          }
        : null;
      const centroidJump = centroid && prevCentroid
        ? Math.hypot(centroid.x - prevCentroid.x, centroid.y - prevCentroid.y)
        : 0;
      if (playerCountDelta >= 6 || centroidJump >= 35) {
        candidates.push({
          candidate_type: "possible_camera_cut_replay_or_goal_aftermath",
          frameIndex: frame.frameIndex,
          window,
          signals: {
            camera_cut_or_replay_angle_possible: true,
            player_count_delta: playerCountDelta,
            centroid_jump: +centroidJump.toFixed(1),
            instruction: "Check for replay/angle change before counting a new event.",
          },
        });
      }
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
- If YOLO missed the ball, use the frame image to set possession when one team is clearly in controlled possession.
- If no major event is visible, still return possession corrections for frames where possession is clear.

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
          "timestamp": 24.0,
          "confidence": 0.82,
          "position": { "x": 78, "y": 48 },
          "semantic_label": "direct_play",
          "evidence_used": ["ball in attacking third", "nearest home player at ball"],
          "conflicts": [],
          "pipeline_flag": null
        }
      ]
    }
  ]
}

Rules:
- possession must be "home", "away", or "contested".
- Allowed event types: goal, shot, shot_saved, shot_off_target, corner, goal_kick, card_yellow, card_red, card_unknown, foul, freekick, offside, pass, tackle, throw-in, dribble.
- Every event MUST include a numeric timestamp copied from one of the supplied frame timestamps. Do not infer time from frame order alone.
- Only report events visible or strongly implied inside candidate windows.
- Do not create an event for every frame.
- If no candidate is convincing, return an empty events array for that frame.
- Pass/dribble/tackle events may be lower confidence when inferred from YOLO ball movement plus visible player context.

Goal rules:
- Do NOT confirm a goal from one weak signal.
- Strong confirmation: visible ball in net, keeper retrieving from net, clear scoreboard score change, or obvious non-half-boundary kickoff aftermath.
- Supporting signals: ball in goal zone, ball disappears in goalmouth, celebration cluster near goal, defending shape collapse.
- Rejection signals: keeper catches/parries, ball visible wide/above goal, restart appears to be corner/goal kick, no score/restart evidence.
- Goal recall triggers include ball direction reversal near the box and possible camera cut/replay/angle change after an attacking moment. These triggers should cause verification, not automatic confirmation.
- Never confirm a goal solely from celebration or pressure near the box.
- If ambiguous, label shot or no event with lower confidence rather than goal.
- Replays and camera cuts are a known limitation. If a frame appears to be a replay or angle cut, add "replay_suspected" to pipeline_flag and do not count a fresh live event unless the timestamp window clearly shows live play.
- If evidence conflicts, keep the event only when confidence remains justified and put the disagreement in conflicts. Use pipeline_flag "verifier_conflict" or "scoreboard_conflict" when appropriate.

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
    const reviewedFrames = mergeReviewedFrames(frames, parsed);
    return NextResponse.json({ frames: synthesizeMovementEvents(reviewedFrames) });
  } catch (err) {
    console.error("[/api/analyze/events]", err);
    return NextResponse.json({ error: "Event review failed." }, { status: 500 });
  }
}
