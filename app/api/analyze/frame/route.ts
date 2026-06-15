import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { FrameData, MatchEvent, Player, AnalyzeFrameRequest } from "@/lib/types";

const FRAME_PROMPT = `You are a professional soccer video analysis system analyzing broadcast TV footage.

CONTEXT: Broadcast soccer is filmed from high wide angles. Players are SMALL figures on the pitch — don't expect close-ups. The ball is a small white/yellow dot. Jersey colors distinguish the two teams.

Return ONLY valid JSON — no markdown, no code fences:
{
  "players": [
    { "id": "h1", "number": 7, "team": "home", "role": "fwd", "position": { "x": 65, "y": 30 }, "action": "running" }
  ],
  "ballPosition": { "x": 50, "y": 45 },
  "possession": "home",
  "possessingPlayerId": "h7",
  "events": [
    { "type": "pass", "team": "home", "description": "precise plain-English description", "confidence": 0.85 }
  ]
}

── STEP 1: SCOREBOARD SCAN (do this before anything else) ──
Scan every corner and edge of the image for score overlays, tickers, or graphics.
Common broadcast placements: top-left, top-right, bottom strip.
Look for: "1-0", "2-1", team abbreviations with numbers, half-time scores.
If ANY score is visible:
  - Emit a "goal" event per goal shown (not just the change — total goals scored)
  - description: "Scoreboard reads [score]"
  - confidence: 0.98
  - team: whichever team has the higher score, or the team that just scored

── STEP 2: GOAL VISUAL CUES ──
Even without a scoreboard, report "goal" if you see:
- Ball inside or touching the net/goal frame
- Multiple attacking players celebrating (arms up, jumping, running together)
- Goalkeeper on ground looking dejected or retrieving ball from net
- Players mobbing one player in a huddle near the goal

── STEP 3: MOTION CONTEXT (if a previous frame was provided) ──
Compare positions between the two frames to detect:
- A player who was winding up → now shot/pass has occurred
- Ball trajectory change → tackle or interception
- Players converging → set piece or challenge
- Player sprinting toward goal and now missing from the area → possible shot

── STEP 4: OTHER EVENTS ──
- Player foot contacting ball toward teammate → "pass"
- Player foot contacting ball toward goal → "shot"
- Two players both reaching for same ball → "tackle"
- Ball at corner flag → "corner"
- Players in a wall, referee visible, or arm gesture → "freekick"
- Goalkeeper diving/catching → "save"
- Player on ground after contact → "foul"
RULE: If you assign "passing", "shooting", or "tackling" as a player action, you MUST include the matching event.

── PLAYERS (EXHAUSTIVE — BOTH TEAMS) ──
SCAN ZONE BY ZONE: left third → centre → right third → near each goal → midfield.
A full-pitch broadcast typically shows 14–22 figures: ~11 per team + 1–2 officials.

CRITICAL: Include every player from BOTH teams. Do NOT skip the away team because they wear a
darker jersey colour (red, navy, black, dark blue are all common away colours). If you only list
one team's players, the analysis is wrong. Aim: ≥10 home players AND ≥10 away players visible.

- id: "h1"…"h11" for home team, "a1"…"a11" for away team.
- Referees/officials (black kit or fluorescent bib) → EXCLUDE from the player list entirely.
- x/y: 0–100 percent of pitch width/height (0,0 = top-left, 100,100 = bottom-right)
- team: label the lighter/more numerous jersey group "home", the other "away". BE CONSISTENT.
- role: "gk" (nearest own goal line), "def", "mid", "fwd"
- action: "running"|"standing"|"passing"|"shooting"|"tackling"|"jumping"|"goalkeeping"|"dribbling"
- possessingPlayerId: id of the player whose foot is touching or nearest to the ball
- number: shirt number if legible, else 0

── PASS COUNTING ──
One pass = ball visibly leaving a player's foot toward a teammate. Never double-count the same pass.`;


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
    const { base64, timestamp, frameIndex, prevBase64, prevTimestamp } = body;

    const client = new Anthropic({ apiKey });

    // Build message content — include previous frame first when available so the
    // model can reason about motion and continuity between the two frames.
    type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
    type TextBlock = { type: "text"; text: string };
    const content: Array<ImageBlock | TextBlock> = [];

    if (prevBase64) {
      content.push({
        type: "image",
        source: { type: "base64", media_type: "image/jpeg", data: prevBase64 },
      });
      content.push({
        type: "text",
        text: `PREVIOUS FRAME (${prevTimestamp?.toFixed(1)}s) — for motion context only, do NOT report events from this frame:`,
      });
    }

    content.push({
      type: "image",
      source: { type: "base64", media_type: "image/jpeg", data: base64 },
    });
    content.push({
      type: "text",
      text: `CURRENT FRAME (${timestamp.toFixed(1)}s) — analyse this frame and report your JSON:\n\n${FRAME_PROMPT}`,
    });

    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1500,
      messages: [{ role: "user", content }],
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
