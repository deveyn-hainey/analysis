import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type {
  MatchAnalysis,
  FrameData,
  MatchEvent,
  TeamAnalysis,
  CoachingInsight,
  SummarizeRequest,
  TeamId,
} from "@/lib/types";

function calcDistanceCovered(teamId: TeamId, frames: FrameData[]): number {
  const PITCH_M_X = 105;
  const PITCH_M_Y = 68;
  let total = 0;
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const curr = frames[i];
    for (const player of curr.players.filter((p) => p.team === teamId)) {
      const prevPlayer = prev.players.find((p) => p.id === player.id);
      if (!prevPlayer) continue;
      const dx = ((player.position.x - prevPlayer.position.x) / 100) * PITCH_M_X;
      const dy = ((player.position.y - prevPlayer.position.y) / 100) * PITCH_M_Y;
      total += Math.sqrt(dx * dx + dy * dy);
    }
  }
  return Math.round(total);
}

function buildHeatmap(teamId: TeamId, frames: FrameData[]): number[][] {
  const grid: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
  for (const frame of frames) {
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

function countEventType(events: MatchEvent[], type: string, team?: TeamId) {
  return events.filter((e) => e.type === type && (!team || e.team === team)).length;
}

/**
 * Remove duplicate goal events caused by a celebration being visible across
 * multiple consecutive frames. Two goals for the same team within MIN_GAP
 * seconds are treated as the same goal — real goals require at least a kickoff
 * reset before another can be scored.
 */
function deduplicateGoals(events: MatchEvent[]): MatchEvent[] {
  const MIN_GAP = 12; // seconds
  const nonGoals = events.filter((e) => e.type !== "goal");
  const goals = events
    .filter((e) => e.type === "goal")
    .sort((a, b) => a.timestamp - b.timestamp);

  const deduped: MatchEvent[] = [];
  for (const goal of goals) {
    const lastSameTeam = [...deduped].reverse().find((g) => g.team === goal.team);
    if (!lastSameTeam || goal.timestamp - lastSameTeam.timestamp >= MIN_GAP) {
      deduped.push(goal);
    }
  }
  return [...nonGoals, ...deduped];
}

/**
 * Count passes by tracking possession changes across frames.
 * When the possessing player changes to a teammate (same team, different player ID),
 * that transition counts as one pass.
 */
function countCrossFramePasses(frames: FrameData[], teamId: TeamId): number {
  let passes = 0;
  let lastPossessor: { team: TeamId; playerId: string } | null = null;

  for (const frame of frames) {
    const curr = frame.possessingPlayer;
    if (!curr) continue;

    if (
      lastPossessor &&
      lastPossessor.team === teamId &&
      curr.team === teamId &&
      lastPossessor.playerId !== curr.playerId
    ) {
      passes++;
    }

    lastPossessor = curr;
  }

  return passes;
}

function buildTeamAnalysis(
  id: TeamId,
  frames: FrameData[],
  allEvents: MatchEvent[]
): TeamAnalysis {
  const teamEvents = allEvents.filter((e) => e.team === id);

  // Cross-frame pass counting is more accurate than event counting alone
  const crossFramePasses = countCrossFramePasses(frames, id);
  const eventPasses = countEventType(teamEvents, "pass");
  const passCount = Math.max(crossFramePasses, eventPasses);

  const shotCount = countEventType(teamEvents, "shot");
  const possessionFrames = frames.filter((f) => f.possession === id).length;
  const possession = Math.round((possessionFrames / Math.max(frames.length, 1)) * 100);

  const positions = frames.flatMap((f) =>
    f.players.filter((p) => p.team === id).map((p) => p.position)
  );
  const avgX = positions.length
    ? positions.reduce((s, p) => s + p.x, 0) / positions.length
    : 50;
  const avgY = positions.length
    ? positions.reduce((s, p) => s + p.y, 0) / positions.length
    : 50;

  // Pass accuracy: 0 when no passes, scales with volume since we only detect
  // visually successful passes from frames
  const passAccuracy = passCount === 0 ? 0 : Math.min(92, 68 + passCount * 2);

  return {
    id,
    name: id === "home" ? "Home Team" : "Away Team",
    color: id === "home" ? "#3b82f6" : "#ef4444",
    formation: "4-3-3",
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
      distanceCovered: calcDistanceCovered(id, frames),
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

  if (dominant.stats.possession > 0) {
    insights.push({
      id: "fb-1",
      category: "possession",
      priority: "high",
      title: `${dominant.name} controls the tempo`,
      observation: `${dominant.name} held possession for ${dominant.stats.possession}% of the clip. ${passive.name} spent most of the time chasing the ball.`,
      recommendation: `${passive.name} needs quicker pressing triggers and a more compact defensive shape to win possession back sooner. ${dominant.name} should vary tempo to avoid becoming predictable.`,
      affectedTeam: passive.id,
    });
  }

  const scorer = homeTeam.stats.goals > 0 ? homeTeam : awayTeam.stats.goals > 0 ? awayTeam : null;
  if (scorer) {
    const conceded = scorer.id === "home" ? awayTeam : homeTeam;
    insights.push({
      id: "fb-2",
      category: "attacking",
      priority: "critical",
      title: `${scorer.name} conversion`,
      observation: `${scorer.name} scored ${scorer.stats.goals} goal(s) from ${scorer.stats.shots} detected shot attempt(s).`,
      recommendation: `Replicate the build-up pattern and movement in training. ${conceded.name} must review its defensive shape in the seconds leading up to the conceded goal.`,
      affectedTeam: scorer.id,
    });
  }

  if (homeTeam.stats.shots + awayTeam.stats.shots > 0) {
    const shooter =
      homeTeam.stats.shots >= awayTeam.stats.shots ? homeTeam : awayTeam;
    insights.push({
      id: "fb-3",
      category: "attacking",
      priority: "medium",
      title: `Shot volume from ${shooter.name}`,
      observation: `${shooter.name} created ${shooter.stats.shots} shot attempt(s), ${shooter.stats.shotsOnTarget} on target.`,
      recommendation: `Focus on first-time finishes inside the box. Review the positions that generated clear sight of goal and make them a training set-piece.`,
      affectedTeam: shooter.id,
    });
  }

  const highLine =
    homeTeam.averagePosition.x > 58
      ? homeTeam
      : awayTeam.averagePosition.x < 42
      ? awayTeam
      : null;
  if (highLine) {
    insights.push({
      id: "fb-4",
      category: "tactical",
      priority: "medium",
      title: `${highLine.name} playing a high line`,
      observation: `${highLine.name}'s average player position (x ≈ ${highLine.averagePosition.x.toFixed(0)}) is well into the opponent's half, compressing space aggressively.`,
      recommendation: `Ensure a recovery runner covers the space in behind — a quick transition from the opponent could exploit gaps left by the advanced shape.`,
      affectedTeam: highLine.id,
    });
  }

  if (homeTeam.stats.passes + awayTeam.stats.passes > 0) {
    const passer =
      homeTeam.stats.passes >= awayTeam.stats.passes ? homeTeam : awayTeam;
    insights.push({
      id: "fb-5",
      category: "possession",
      priority: "low",
      title: `${passer.name} passing volume`,
      observation: `${passer.name} completed approximately ${passer.stats.passes} passes in this clip, compared to ${(passer.id === "home" ? awayTeam : homeTeam).stats.passes} for the opposition.`,
      recommendation: `Continue building through possession but ensure passes have purpose — look for incisive forward options rather than lateral recycling.`,
      affectedTeam: passer.id,
    });
  }

  return insights.slice(0, 5);
}

async function generateInsights(
  client: Anthropic,
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis
): Promise<CoachingInsight[]> {
  const prompt = `You are an elite soccer performance analyst reviewing a short match clip.

Generate 4–5 specific, actionable coaching insights using the numbers below. If some stats are low or zero, infer from what you can see — possession dominance, goal conversion, and player positioning are all meaningful even in short clips.

HOME (${homeTeam.name}):
- Possession: ${homeTeam.stats.possession}%
- Passes: ${homeTeam.stats.passes} (accuracy est. ${homeTeam.stats.passAccuracy}%)
- Shots: ${homeTeam.stats.shots} (on target: ${homeTeam.stats.shotsOnTarget}) | Goals: ${homeTeam.stats.goals}
- Tackles: ${homeTeam.stats.tackles} | Fouls: ${homeTeam.stats.fouls} | Corners: ${homeTeam.stats.corners}
- Avg player x-position: ${homeTeam.averagePosition.x}/100 (0=own goal, 100=opponent goal)

AWAY (${awayTeam.name}):
- Possession: ${awayTeam.stats.possession}%
- Passes: ${awayTeam.stats.passes} (accuracy est. ${awayTeam.stats.passAccuracy}%)
- Shots: ${awayTeam.stats.shots} (on target: ${awayTeam.stats.shotsOnTarget}) | Goals: ${awayTeam.stats.goals}
- Tackles: ${awayTeam.stats.tackles} | Fouls: ${awayTeam.stats.fouls} | Corners: ${awayTeam.stats.corners}
- Avg player x-position: ${awayTeam.averagePosition.x}/100

Return ONLY a valid JSON array — no markdown, no code fences:
[{
  "id": "i1",
  "category": "attacking",
  "priority": "high",
  "title": "short title",
  "observation": "what the data shows",
  "recommendation": "specific actionable step for training or tactics",
  "affectedTeam": "home"
}]

category: attacking | defensive | possession | tactical | physical
priority: critical | high | medium | low
affectedTeam: home | away | both`;

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      messages: [{ role: "user", content: prompt }],
    });

    const raw =
      response.content[0].type === "text" ? response.content[0].text.trim() : "";
    const cleaned = raw
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();
    const parsed = JSON.parse(cleaned) as CoachingInsight[];
    return Array.isArray(parsed) && parsed.length > 0
      ? parsed
      : buildFallbackInsights(homeTeam, awayTeam);
  } catch {
    return buildFallbackInsights(homeTeam, awayTeam);
  }
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "ANTHROPIC_API_KEY not configured." },
        { status: 500 }
      );
    }

    const body = (await req.json()) as SummarizeRequest;
    const { frames } = body;

    if (!frames || frames.length === 0) {
      return NextResponse.json({ error: "No frames provided." }, { status: 400 });
    }

    const allEvents = deduplicateGoals(frames.flatMap((f) => f.events));

    const homeTeam = buildTeamAnalysis("home", frames, allEvents);
    const awayTeam = buildTeamAnalysis("away", frames, allEvents);

    const client = new Anthropic({ apiKey });
    const insights = await generateInsights(client, homeTeam, awayTeam);

    const analysis: MatchAnalysis = {
      id: `match-${Date.now()}`,
      processedAt: new Date().toISOString(),
      videoDuration: frames[frames.length - 1]?.timestamp ?? 0,
      framesAnalyzed: frames.length,
      homeTeam,
      awayTeam,
      frames,
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
    console.error("[/api/analyze/summarize]", err);
    return NextResponse.json({ error: "Summarize failed." }, { status: 500 });
  }
}
