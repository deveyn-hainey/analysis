import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type {
  MatchAnalysis,
  FrameData,
  MatchEvent,
  Player,
  TeamAnalysis,
  CoachingInsight,
  AnalyzeRequest,
} from "@/lib/types";
import { SAMPLE_ANALYSIS } from "@/lib/sampleData";

// Claude prompt for per-frame analysis
const FRAME_PROMPT = `You are a professional soccer video analysis system. Analyze this frame from a soccer match footage.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:
{
  "players": [
    {
      "id": "h1",
      "number": 1,
      "team": "home",
      "role": "gk",
      "position": { "x": 5, "y": 50 },
      "action": "goalkeeping"
    }
  ],
  "ballPosition": { "x": 50, "y": 50 },
  "possession": "home",
  "events": [
    {
      "type": "pass",
      "team": "home",
      "description": "...",
      "confidence": 0.85
    }
  ]
}

Rules:
- x/y are 0–100 percentages of field width/height (0,0 = top-left, 100,100 = bottom-right)
- team: "home" = lighter/left side, "away" = darker/right side
- role: "gk", "def", "mid", or "fwd"
- action: "running", "standing", "passing", "shooting", "tackling", "jumping", "goalkeeping", "dribbling"
- event types: "pass", "shot", "tackle", "goal", "save", "corner", "freekick", "foul", "offside", "throw-in", "dribble"
- Include only clearly visible players. Estimate positions as carefully as possible.
- If no event is happening, events = []`;

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
  events?: RawFrameEvent[];
}

async function analyzeFrame(
  client: Anthropic,
  base64: string,
  timestamp: number,
  frameIndex: number
): Promise<FrameData> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: base64,
            },
          },
          { type: "text", text: FRAME_PROMPT },
        ],
      },
    ],
  });

  const text = response.content[0].type === "text" ? response.content[0].text : "{}";

  let parsed: RawFrameResult = {};
  try {
    parsed = JSON.parse(text) as RawFrameResult;
  } catch {
    // If Claude returns non-JSON for a difficult frame, use empty data
    parsed = {};
  }

  const events: MatchEvent[] = (parsed.events ?? []).map((e, i) => ({
    id: `f${frameIndex}-e${i}`,
    timestamp,
    type: e.type as MatchEvent["type"],
    team: e.team as MatchEvent["team"],
    description: e.description,
    confidence: e.confidence ?? 0.8,
    isKeyMoment: ["goal", "shot", "save", "corner", "freekick", "foul"].includes(e.type),
  }));

  return {
    frameIndex,
    timestamp,
    players: parsed.players ?? [],
    ballPosition: parsed.ballPosition,
    possession: (parsed.possession as FrameData["possession"]) ?? "contested",
    events,
  };
}

function buildHeatmap(teamId: "home" | "away", allFrames: FrameData[]): number[][] {
  const grid: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
  for (const frame of allFrames) {
    for (const p of frame.players) {
      if (p.team !== teamId) continue;
      const col = Math.min(9, Math.floor(p.position.x / 10));
      const row = Math.min(9, Math.floor(p.position.y / 10));
      grid[row][col]++;
    }
  }
  const max = Math.max(...grid.flat(), 1);
  return grid.map((row) => row.map((v) => +(v / max).toFixed(2)));
}

function countEventType(events: MatchEvent[], type: string, team?: "home" | "away") {
  return events.filter((e) => e.type === type && (!team || e.team === team)).length;
}

function buildTeamAnalysis(
  id: "home" | "away",
  name: string,
  color: string,
  formation: TeamAnalysis["formation"],
  frames: FrameData[],
  allEvents: MatchEvent[]
): TeamAnalysis {
  const teamEvents = allEvents.filter((e) => e.team === id);
  const passes = countEventType(teamEvents, "pass");
  const totalAttempts = passes + countEventType(teamEvents, "shot");
  const homePossessionFrames = frames.filter((f) => f.possession === id).length;
  const possession = Math.round((homePossessionFrames / Math.max(frames.length, 1)) * 100);

  const positions = frames.flatMap((f) => f.players.filter((p) => p.team === id).map((p) => p.position));
  const avgX = positions.length ? positions.reduce((s, p) => s + p.x, 0) / positions.length : 50;
  const avgY = positions.length ? positions.reduce((s, p) => s + p.y, 0) / positions.length : 50;

  return {
    id,
    name,
    color,
    formation,
    averagePosition: { x: +avgX.toFixed(1), y: +avgY.toFixed(1) },
    stats: {
      possession,
      passes: passes + totalAttempts,
      passAccuracy: passes > 0 ? Math.min(95, 65 + passes) : 70,
      shots: countEventType(teamEvents, "shot"),
      shotsOnTarget: Math.ceil(countEventType(teamEvents, "shot") * 0.6),
      tackles: countEventType(teamEvents, "tackle"),
      fouls: countEventType(teamEvents, "foul"),
      corners: countEventType(teamEvents, "corner"),
      goals: countEventType(teamEvents, "goal"),
    },
    heatmap: buildHeatmap(id, frames),
  };
}

async function generateInsights(
  client: Anthropic,
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis
): Promise<CoachingInsight[]> {
  const prompt = `You are an elite soccer performance analyst. Based on this match data, generate 4–5 actionable coaching insights.

Match stats:
HOME (${homeTeam.name}): possession=${homeTeam.stats.possession}%, passes=${homeTeam.stats.passes}, pass accuracy=${homeTeam.stats.passAccuracy}%, shots=${homeTeam.stats.shots}, tackles=${homeTeam.stats.tackles}, fouls=${homeTeam.stats.fouls}
AWAY (${awayTeam.name}): possession=${awayTeam.stats.possession}%, passes=${awayTeam.stats.passes}, pass accuracy=${awayTeam.stats.passAccuracy}%, shots=${awayTeam.stats.shots}, tackles=${awayTeam.stats.tackles}, fouls=${awayTeam.stats.fouls}

Return ONLY valid JSON array:
[{
  "id": "i1",
  "category": "attacking",
  "priority": "high",
  "title": "...",
  "observation": "...",
  "recommendation": "...",
  "affectedTeam": "home"
}]

category options: attacking, defensive, possession, tactical, physical
priority options: critical, high, medium, low
affectedTeam options: home, away, both`;

  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    messages: [{ role: "user", content: prompt }],
  });

  const text = response.content[0].type === "text" ? response.content[0].text : "[]";
  try {
    return JSON.parse(text) as CoachingInsight[];
  } catch {
    return [];
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as AnalyzeRequest;

    if (body.demo) {
      return NextResponse.json(SAMPLE_ANALYSIS);
    }

    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "ANTHROPIC_API_KEY is not configured. Set it in .env.local or use demo mode." },
        { status: 500 }
      );
    }

    if (!body.frames || body.frames.length === 0) {
      return NextResponse.json({ error: "No frames provided." }, { status: 400 });
    }

    const client = new Anthropic({ apiKey });

    // Analyze frames sequentially to respect rate limits
    const analyzedFrames: FrameData[] = [];
    for (let i = 0; i < body.frames.length; i++) {
      const { base64, timestamp } = body.frames[i];
      const frameData = await analyzeFrame(client, base64, timestamp, i);
      analyzedFrames.push(frameData);
    }

    const allEvents = analyzedFrames.flatMap((f) => f.events);

    const homeTeam = buildTeamAnalysis("home", "Home Team", "#3b82f6", "4-3-3", analyzedFrames, allEvents);
    const awayTeam = buildTeamAnalysis("away", "Away Team", "#ef4444", "4-5-1", analyzedFrames, allEvents);

    const insights = await generateInsights(client, homeTeam, awayTeam);

    const analysis: MatchAnalysis = {
      id: `match-${Date.now()}`,
      processedAt: new Date().toISOString(),
      videoDuration: body.frames[body.frames.length - 1]?.timestamp ?? 0,
      framesAnalyzed: analyzedFrames.length,
      homeTeam,
      awayTeam,
      frames: analyzedFrames,
      keyEvents: allEvents.filter((e) => e.isKeyMoment),
      insights,
      score: {
        home: countEventType(allEvents, "goal", "home"),
        away: countEventType(allEvents, "goal", "away"),
      },
      processingMethod: "ai",
    };

    return NextResponse.json(analysis);
  } catch (err) {
    console.error("[/api/analyze]", err);
    return NextResponse.json({ error: "Internal server error during analysis." }, { status: 500 });
  }
}
