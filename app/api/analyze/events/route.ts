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
  return ["goal", "shot", "save", "corner", "freekick", "foul", "offside"].includes(type);
}

function mergeReviewedFrames(frames: FrameData[], review: RawEventReview): FrameData[] {
  const updates = new Map((review.frames ?? []).map((frame) => [frame.frameIndex, frame]));

  return frames.map((frame) => {
    const update = updates.get(frame.frameIndex);
    if (!update) return frame;

    const reviewedEvents: MatchEvent[] = (update.events ?? []).map((event, i) => ({
      id: eventId(frame.frameIndex, i, event.type),
      timestamp: frame.timestamp,
      type: event.type,
      team: event.team,
      description: event.description,
      confidence: event.confidence ?? 0.75,
      position: event.position,
      isKeyMoment: isKeyMoment(event.type),
    }));

    return {
      ...frame,
      possession: update.possession ?? frame.possession,
      events: [...frame.events, ...reviewedEvents],
    };
  });
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
      playersNearBall: frame.ballPosition
        ? frame.players
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
            .slice(0, 4)
        : [],
    }));

    type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
    type TextBlock = { type: "text"; text: string };
    const content: Array<ImageBlock | TextBlock> = [
      {
        type: "text",
        text: `You are reviewing sampled tactical-view soccer frames.

YOLO has already detected player and ball positions. Your job is NOT dense tracking.
Use the images plus the YOLO metadata to correct possession and identify timeline events.

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
          "position": { "x": 78, "y": 48 }
        }
      ]
    }
  ]
}

Rules:
- possession must be "home", "away", or "contested".
- Only report events visible or strongly implied in the current sampled frame.
- Report goals only for clear evidence: scoreboard score change, ball in/behind goal, net/goalmouth celebration, goalkeeper retrieving from net.
- Do not infer a goal from pressure near the box.
- Include timeline events for passes, shots, tackles, saves, corners, freekicks, fouls, offsides, throw-ins, dribbles, and goals when visible.
- Use sparse high-value events; avoid inventing an event for every frame.
- If unsure, leave events empty and set possession conservatively.

YOLO metadata:
${JSON.stringify(compactFrames)}`,
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
