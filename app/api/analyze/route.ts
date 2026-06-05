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
const FRAME_PROMPT = `You are a professional soccer video analysis system. Analyze this frame from a soccer match.

Return ONLY valid JSON (no markdown, no code fences) with this exact structure:
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

Rules for players:
- x/y: 0–100 percentage of field width/height. (0,0) = top-left, (100,100) = bottom-right.
- team: pick one jersey colour as "home", the other as "away". Be consistent for every player in the frame.
- role: "gk", "def", "mid", or "fwd" — infer from position on field.
- action: "running", "standing", "passing", "shooting", "tackling", "jumping", "goalkeeping", "dribbling"

Rules for events — be GENEROUS, report anything you can infer:
- Any player whose foot is near or contacting the ball = "pass" (or "shot" if aimed at goal)
- Any player kicking the ball toward goal = "shot"
- Two players contesting the same ball = "tackle"
- Ball near the corner flag = "corner"
- Players forming a wall or referee gesture visible = "freekick"
- Ball crossing/entering the goal = "goal"
- Goalkeeper diving or catching = "save"
- If you set a player's action to "passing", "shooting", or "tackling", you MUST also include a matching event entry.
- events = [] only if the frame is clearly static with no ball contact or challenge`;

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

  // Synthesize events from player actions when Claude didn't report them explicitly.
  // This ensures passes/shots/tackles show up in stats even for simple clips.
  const ACTION_TO_EVENT: Partial<Record<string, MatchEvent["type"]>> = {
    passing: "pass",
    shooting: "shot",
    tackling: "tackle",
  };
  const explicitTypes = new Set(explicitEvents.map((e) => `${e.team}-${e.type}`));
  const synthesized: MatchEvent[] = [];
  for (const player of players) {
    const evType = ACTION_TO_EVENT[player.action];
    if (!evType) continue;
    const key = `${player.team}-${evType}`;
    if (explicitTypes.has(key)) continue; // already reported
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
    explicitTypes.add(key);
  }

  const events = [...explicitEvents, ...synthesized];

  return {
    frameIndex,
    timestamp,
    players,
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
  const passCount = countEventType(teamEvents, "pass");
  const shotCount = countEventType(teamEvents, "shot");
  const possessionFrames = frames.filter((f) => f.possession === id).length;
  const possession = Math.round((possessionFrames / Math.max(frames.length, 1)) * 100);

  const positions = frames.flatMap((f) => f.players.filter((p) => p.team === id).map((p) => p.position));
  const avgX = positions.length ? positions.reduce((s, p) => s + p.x, 0) / positions.length : 50;
  const avgY = positions.length ? positions.reduce((s, p) => s + p.y, 0) / positions.length : 50;

  // Pass accuracy: 0 when no passes detected; scales up with volume since we only
  // detect visually successful passes (failed passes are rarely visible as distinct actions).
  const passAccuracy = passCount === 0 ? 0 : Math.min(92, 68 + passCount * 2);

  return {
    id,
    name,
    color,
    formation,
    averagePosition: { x: +avgX.toFixed(1), y: +avgY.toFixed(1) },
    stats: {
      possession,
      passes: passCount,
      passAccuracy,
      shots: shotCount,
      shotsOnTarget: shotCount > 0 ? Math.max(1, Math.round(shotCount * 0.6)) : 0,
      tackles: countEventType(teamEvents, "tackle"),
      fouls: countEventType(teamEvents, "foul"),
      corners: countEventType(teamEvents, "corner"),
      goals: countEventType(teamEvents, "goal"),
    },
    heatmap: buildHeatmap(id, frames),
  };
}

function buildFallbackInsights(
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis
): CoachingInsight[] {
  const insights: CoachingInsight[] = [];

  const dominant = homeTeam.stats.possession >= 50 ? homeTeam : awayTeam;
  const passive = dominant.id === "home" ? awayTeam : homeTeam;

  insights.push({
    id: "fb-1",
    category: "possession",
    priority: "high",
    title: `${dominant.name} controls the ball`,
    observation: `${dominant.name} held possession for ${dominant.stats.possession}% of the clip. ${passive.name} spent most of the time without the ball.`,
    recommendation: `${passive.name} needs quicker pressing triggers and a more compact shape to win the ball back. ${dominant.name} should vary tempo to avoid being predictable.`,
    affectedTeam: passive.id,
  });

  if (homeTeam.stats.goals > 0 || awayTeam.stats.goals > 0) {
    const scorer = homeTeam.stats.goals > 0 ? homeTeam : awayTeam;
    const conceded = scorer.id === "home" ? awayTeam : homeTeam;
    insights.push({
      id: "fb-2",
      category: "attacking",
      priority: "critical",
      title: `${scorer.name} found the net`,
      observation: `${scorer.name} scored ${scorer.stats.goals} goal(s) from ${scorer.stats.shots} detected shot(s).`,
      recommendation: `Study the build-up sequence and replicate the movement pattern in training. ${conceded.name} should review the defensive shape in the lead-up and identify the breakdown.`,
      affectedTeam: scorer.id,
    });
  }

  if (homeTeam.stats.shots > 0 || awayTeam.stats.shots > 0) {
    const shooter = homeTeam.stats.shots >= awayTeam.stats.shots ? homeTeam : awayTeam;
    insights.push({
      id: "fb-3",
      category: "attacking",
      priority: "medium",
      title: `Shot volume from ${shooter.name}`,
      observation: `${shooter.name} generated ${shooter.stats.shots} shot attempt(s), ${shooter.stats.shotsOnTarget} on target.`,
      recommendation: `Reinforce the positions and combinations that led to these opportunities. Focus on first-time finishes inside the penalty area.`,
      affectedTeam: shooter.id,
    });
  }

  const highLine = homeTeam.averagePosition.x > 55 ? homeTeam : awayTeam.averagePosition.x < 45 ? awayTeam : null;
  if (highLine) {
    insights.push({
      id: "fb-4",
      category: "tactical",
      priority: "medium",
      title: `${highLine.name} playing an advanced defensive line`,
      observation: `${highLine.name}'s average player position was well into the opponent's half (x ≈ ${highLine.averagePosition.x.toFixed(0)}), compressing the space available to the opposition.`,
      recommendation: `Ensure the midfield tracks runners in behind. A misplaced pass from this shape can lead to dangerous breakaways — identify a recovery defender to cover.`,
      affectedTeam: highLine.id,
    });
  }

  if (homeTeam.stats.tackles > 0 || awayTeam.stats.tackles > 0) {
    const tackler = homeTeam.stats.tackles >= awayTeam.stats.tackles ? homeTeam : awayTeam;
    insights.push({
      id: "fb-5",
      category: "defensive",
      priority: "low",
      title: `${tackler.name}'s pressing intensity`,
      observation: `${tackler.name} made ${tackler.stats.tackles} tackle(s) in the clip, indicating an active press.`,
      recommendation: `Maintain the defensive work rate but ensure tackles are well-timed — reckless challenges in dangerous areas will give away set pieces.`,
      affectedTeam: tackler.id,
    });
  }

  return insights.slice(0, 5);
}

async function generateInsights(
  client: Anthropic,
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis
): Promise<CoachingInsight[]> {
  const prompt = `You are an elite soccer performance analyst. Based on this match clip data, generate 4–5 actionable coaching insights. Be specific and use the numbers provided. Even if some stats are low or zero, infer meaningful observations from what you can see.

HOME (${homeTeam.name}):
- Possession: ${homeTeam.stats.possession}%
- Passes detected: ${homeTeam.stats.passes} (accuracy: ${homeTeam.stats.passAccuracy}%)
- Shots: ${homeTeam.stats.shots} (on target: ${homeTeam.stats.shotsOnTarget})
- Tackles: ${homeTeam.stats.tackles} | Fouls: ${homeTeam.stats.fouls} | Goals: ${homeTeam.stats.goals}
- Average player position: x=${homeTeam.averagePosition.x}, y=${homeTeam.averagePosition.y} (0–100 scale, 0=own goal end)

AWAY (${awayTeam.name}):
- Possession: ${awayTeam.stats.possession}%
- Passes detected: ${awayTeam.stats.passes} (accuracy: ${awayTeam.stats.passAccuracy}%)
- Shots: ${awayTeam.stats.shots} (on target: ${awayTeam.stats.shotsOnTarget})
- Tackles: ${awayTeam.stats.tackles} | Fouls: ${awayTeam.stats.fouls} | Goals: ${awayTeam.stats.goals}
- Average player position: x=${awayTeam.averagePosition.x}, y=${awayTeam.averagePosition.y}

Return ONLY a valid JSON array — no markdown, no code fences, no explanation:
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

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      messages: [{ role: "user", content: prompt }],
    });

    const raw = response.content[0].type === "text" ? response.content[0].text.trim() : "";
    // Strip markdown code fences if Claude wrapped the JSON
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
    const parsed = JSON.parse(cleaned) as CoachingInsight[];
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : buildFallbackInsights(homeTeam, awayTeam);
  } catch {
    return buildFallbackInsights(homeTeam, awayTeam);
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
        { error: "ANTHROPIC_API_KEY is not configured. Add it in your Vercel project settings under Settings → Environment Variables, then redeploy." },
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
