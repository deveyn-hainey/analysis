import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { FrameData, MatchEvent, Player, AnalyzeFrameRequest } from "@/lib/types";

const FRAME_PROMPT = `You are a professional soccer video analysis system. Analyze this single frame from a match.

Return ONLY valid JSON — no markdown, no code fences, no explanation:
{
  "players": [
    { "id": "h1", "number": 7, "team": "home", "role": "fwd", "position": { "x": 65, "y": 30 }, "action": "running" }
  ],
  "ballPosition": { "x": 50, "y": 45 },
  "possession": "home",
  "possessingPlayerId": "h7",
  "events": [
    { "type": "pass", "team": "home", "description": "...", "confidence": 0.85 }
  ]
}

── PLAYERS ──
- position x/y: 0–100 percent of field (0,0 = top-left, 100,100 = bottom-right)
- team: pick one jersey colour as "home", the other as "away". Be CONSISTENT for every player.
- role: "gk" (near own goal), "def", "mid", "fwd"
- action: "running" | "standing" | "passing" | "shooting" | "tackling" | "jumping" | "goalkeeping" | "dribbling"
- possessingPlayerId: id of the player whose foot is touching or closest to the ball

── EVENTS — be GENEROUS, report everything you can infer ──
- Any foot near/touching ball in a kicking motion → "pass" (or "shot" if toward goal)
- Two players both reaching for the same ball → "tackle"
- Ball near corner flag → "corner"
- Players forming a wall or referee visible with whistle → "freekick"
- Player diving/diving save → "save"
- If you set a player's action to "passing", "shooting", or "tackling" you MUST include a matching event.

── GOAL DETECTION — check ALL of these ──
Report a "goal" event if ANY of the following are visible, even if you missed the exact crossing moment:
1. Ball is visually inside or touching the net/goal frame
2. Attacking players are celebrating — arms raised, jumping, embracing teammates
3. Goalkeeper is on the ground, looking defeated, or retrieving ball from net
4. Scoreboard in frame shows a recent score change
5. Players surrounding the goal in a post-goal cluster

── PASS COUNTING ──
A pass is a deliberate ball transfer from one player to a teammate. If you can see the ball has just left a player's foot toward a teammate, that is one pass. Do not count the same pass multiple times.`;

interface RawFrameEvent {
  type: string;
  team?: string;
  description: string;
  confidence: number;
}

interface RawFrameResult {
  players?: Player[];
  ballPosition?: { x: number; y: number };
  possession?: string;
  possessingPlayerId?: string;
  events?: RawFrameEvent[];
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "ANTHROPIC_API_KEY not configured. Add it in Vercel → Settings → Environment Variables." },
        { status: 500 }
      );
    }

    const body = (await req.json()) as AnalyzeFrameRequest;
    const { base64, timestamp, frameIndex } = body;

    const client = new Anthropic({ apiKey });

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1200,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: { type: "base64", media_type: "image/jpeg", data: base64 },
            },
            { type: "text", text: FRAME_PROMPT },
          ],
        },
      ],
    });

    const raw = response.content[0].type === "text" ? response.content[0].text.trim() : "{}";
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

    let parsed: RawFrameResult = {};
    try {
      parsed = JSON.parse(cleaned) as RawFrameResult;
    } catch {
      parsed = {};
    }

    const players: Player[] = parsed.players ?? [];

    const explicitEvents: MatchEvent[] = (parsed.events ?? []).map((e, i) => ({
      id: `f${frameIndex}-e${i}`,
      timestamp,
      type: e.type as MatchEvent["type"],
      team: e.team as MatchEvent["team"],
      description: e.description,
      confidence: e.confidence ?? 0.8,
      isKeyMoment: ["goal", "shot", "save", "corner", "freekick", "foul"].includes(e.type),
    }));

    // Synthesize events from player actions when Claude omits them
    const ACTION_TO_EVENT: Partial<Record<string, MatchEvent["type"]>> = {
      passing: "pass",
      shooting: "shot",
      tackling: "tackle",
    };
    const explicitKeys = new Set(explicitEvents.map((e) => `${e.team}-${e.type}`));
    const synthesized: MatchEvent[] = [];
    for (const player of players) {
      const evType = ACTION_TO_EVENT[player.action];
      if (!evType) continue;
      const key = `${player.team}-${evType}`;
      if (explicitKeys.has(key)) continue;
      synthesized.push({
        id: `f${frameIndex}-synth-${player.id}`,
        timestamp,
        type: evType,
        team: player.team,
        description: `${player.team === "home" ? "Home" : "Away"} #${player.number} ${player.action}`,
        confidence: 0.6,
        isKeyMoment: evType === "shot",
        position: player.position,
      });
      explicitKeys.add(key);
    }

    // Resolve possessing player
    const possPlayerId = parsed.possessingPlayerId;
    const possPlayer = players.find((p) => p.id === possPlayerId);
    const possessingPlayer = possPlayer
      ? { team: possPlayer.team, playerId: possPlayer.id }
      : undefined;

    const frame: FrameData = {
      frameIndex,
      timestamp,
      players,
      ballPosition: parsed.ballPosition,
      possession: (parsed.possession as FrameData["possession"]) ?? "contested",
      possessingPlayer,
      events: [...explicitEvents, ...synthesized],
    };

    return NextResponse.json(frame);
  } catch (err) {
    console.error("[/api/analyze/frame]", err);
    return NextResponse.json({ error: "Frame analysis failed." }, { status: 500 });
  }
}
