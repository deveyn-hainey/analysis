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
import {
  distanceMeters,
  estimateShotXg,
  isStablePlayerId,
  isVerifiedEvent,
  shotLikeEvents,
  stableTrackingCoverage,
  teamExpectedGoals,
} from "@/lib/visionMetrics";

const SUMMARY_MODEL = process.env.ANTHROPIC_SUMMARY_MODEL ?? "claude-sonnet-4-6";

function calcDistanceCovered(teamId: TeamId, frames: FrameData[]): number {
  let total = 0;
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const curr = frames[i];
    for (const player of curr.players.filter((p) => p.team === teamId)) {
      if (!isStablePlayerId(player.id) || player.number <= 0) continue;
      const prevPlayer = prev.players.find((p) => p.id === player.id);
      if (!prevPlayer) continue;
      const step = distanceMeters(player.position, prevPlayer.position);
      if (step <= 9) total += step;
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
  return events.filter((e) => e.type === type && (!team || e.team === team) && isVerifiedEvent(e)).length;
}

function duplicateWindowSeconds(type: MatchEvent["type"]) {
  if (type === "goal") return 12;
  if (["shot", "save", "corner", "goal-kick", "freekick", "foul", "offside", "card_yellow", "card_red", "card_unknown"].includes(type)) {
    return 6;
  }
  if (["pass", "dribble", "tackle"].includes(type)) return 5;
  return 3;
}

function mergeDuplicateEvents(events: MatchEvent[]): MatchEvent[] {
  const sorted = [...events]
    .filter((event) => event.pipelineFlag !== "replay_suspected")
    .sort((a, b) => a.timestamp - b.timestamp || b.confidence - a.confidence);

  const merged: MatchEvent[] = [];
  for (const event of sorted) {
    const duplicate = merged.find((existing) => {
      if (existing.type !== event.type || existing.team !== event.team) return false;
      if (Math.abs(existing.timestamp - event.timestamp) > duplicateWindowSeconds(event.type)) return false;
      if (event.position && existing.position) {
        const d = Math.hypot(event.position.x - existing.position.x, event.position.y - existing.position.y);
        if (["pass", "dribble", "tackle"].includes(event.type)) return d <= 18;
        return d <= 26;
      }
      return true;
    });

    if (!duplicate) {
      merged.push(event);
      continue;
    }

    const duplicateVerified = isVerifiedEvent(duplicate);
    const eventVerified = isVerifiedEvent(event);
    const replaceDetails = (eventVerified && !duplicateVerified) || event.confidence > duplicate.confidence;
    duplicate.confidence = Math.max(duplicate.confidence, event.confidence);
    duplicate.evidenceUsed = [...new Set([...(duplicate.evidenceUsed ?? []), ...(event.evidenceUsed ?? [])])];
    duplicate.conflicts = [...new Set([...(duplicate.conflicts ?? []), ...(event.conflicts ?? [])])];
    if (!duplicate.semanticLabel && event.semanticLabel) duplicate.semanticLabel = event.semanticLabel;
    if (!duplicate.pipelineFlag && event.pipelineFlag) duplicate.pipelineFlag = event.pipelineFlag;
    if (replaceDetails) {
      duplicate.description = event.description;
      duplicate.position = event.position ?? duplicate.position;
    }
  }

  return merged;
}

function buildEventConflicts(events: MatchEvent[]): MatchAnalysis["eventConflicts"] {
  return events
    .filter((event) => (event.conflicts?.length ?? 0) > 0 || event.pipelineFlag)
    .map((event) => ({
      timestamp: event.timestamp,
      type: event.type,
      team: event.team,
      description: event.description,
      conflicts: event.conflicts ?? [],
      evidenceUsed: event.evidenceUsed,
      pipelineFlag: event.pipelineFlag,
    }));
}

function scoreboardTeamName(teamId: TeamId, frames: FrameData[]) {
  const counts = new Map<string, number>();
  const key = teamId === "home" ? "homeLabel" : "awayLabel";

  for (const frame of frames) {
    const label = frame.scoreboard?.[key];
    if (!label) continue;

    const cleaned = label.trim().replace(/\s+/g, " ");
    if (cleaned.length < 2 || cleaned.length > 20) continue;
    if (/^\d+$/.test(cleaned)) continue;

    counts.set(cleaned, (counts.get(cleaned) ?? 0) + 1);
  }

  const [best] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0] ?? [];
  return best ?? (teamId === "home" ? "Home Team" : "Away Team");
}

function buildTeamAnalysis(
  id: TeamId,
  frames: FrameData[],
  allEvents: MatchEvent[]
): TeamAnalysis {
  const teamEvents = allEvents.filter((e) => e.team === id);

  // Player IDs are assigned per frame by the vision model, so possession changes
  // across frames are not stable enough to infer passes reliably.
  const passCount = countEventType(teamEvents, "pass");
  const shotEvents = shotLikeEvents(teamEvents);
  const shotCount = shotEvents.length;
  const possessionSampleFrames = frames.filter((f) => f.possession === "home" || f.possession === "away");
  const possessionFrames = possessionSampleFrames.filter((f) => f.possession === id).length;
  const possession =
    possessionSampleFrames.length > 0
      ? Math.round((possessionFrames / possessionSampleFrames.length) * 100)
      : 50;

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
  const completedPasses = passCount;
  const turnoverLike = countEventType(teamEvents, "tackle") + countEventType(teamEvents, "dribble");
  const passAccuracy = completedPasses === 0
    ? 0
    : Math.round((completedPasses / Math.max(completedPasses + turnoverLike * 0.35, completedPasses)) * 100);
  const trackingCoverage = stableTrackingCoverage(frames, id);
  const expectedGoals = teamExpectedGoals(allEvents, id);
  const verifiedEventShare = teamEvents.length
    ? teamEvents.filter(isVerifiedEvent).length / teamEvents.length
    : 0.5;

  return {
    id,
    name: scoreboardTeamName(id, frames),
    color: id === "home" ? "#3b82f6" : "#ef4444",
    formation: "4-3-3",
    averagePosition: { x: +avgX.toFixed(1), y: +avgY.toFixed(1) },
    stats: {
      possession,
      passes: passCount,
      passAccuracy,
      shots: shotCount,
      shotsOnTarget: shotLikeEvents(teamEvents).filter((event) => event.type === "goal" || event.type === "save").length,
      tackles: countEventType(teamEvents, "tackle"),
      fouls: countEventType(teamEvents, "foul"),
      corners: countEventType(teamEvents, "corner"),
      goals: countEventType(teamEvents, "goal"),
      distanceCovered: calcDistanceCovered(id, frames),
      expectedGoals,
      metricConfidence: {
        possession: +(Math.min(0.88, Math.max(0.35, possessionSampleFrames.length / Math.max(frames.length, 1))).toFixed(2)),
        passes: +(Math.min(0.82, 0.35 + verifiedEventShare * 0.42 + trackingCoverage * 0.12).toFixed(2)),
        shots: +(Math.min(0.9, 0.42 + shotEvents.reduce((s, e) => s + e.confidence, 0) / Math.max(shotEvents.length, 1) * 0.45).toFixed(2)),
        xg: +(Math.min(0.86, 0.38 + shotEvents.filter((event) => event.position).length / Math.max(shotEvents.length, 1) * 0.28 + verifiedEventShare * 0.2).toFixed(2)),
        distance: +(Math.min(0.88, Math.max(0.2, trackingCoverage)).toFixed(2)),
      },
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
  awayTeam: TeamAnalysis,
  eventConflicts: MatchAnalysis["eventConflicts"] = []
): Promise<CoachingInsight[]> {
  const prompt = `You are an elite soccer performance analyst reviewing a short match clip.

Generate 4–5 specific, actionable coaching insights using the numbers below. If some stats are low or zero, infer from what you can see — possession dominance, goal conversion, and player positioning are all meaningful even in short clips.

HOME (${homeTeam.name}):
- Possession: ${homeTeam.stats.possession}%
- Passes: ${homeTeam.stats.passes} (accuracy est. ${homeTeam.stats.passAccuracy}%)
- Shots: ${homeTeam.stats.shots} (on target: ${homeTeam.stats.shotsOnTarget}) | Goals: ${homeTeam.stats.goals} | xG: ${homeTeam.stats.expectedGoals?.toFixed(2) ?? "n/a"}
- Metric confidence: possession ${homeTeam.stats.metricConfidence?.possession ?? 0}, passes ${homeTeam.stats.metricConfidence?.passes ?? 0}, shots ${homeTeam.stats.metricConfidence?.shots ?? 0}, distance ${homeTeam.stats.metricConfidence?.distance ?? 0}
- Tackles: ${homeTeam.stats.tackles} | Fouls: ${homeTeam.stats.fouls} | Corners: ${homeTeam.stats.corners}
- Avg player x-position: ${homeTeam.averagePosition.x}/100 (0=own goal, 100=opponent goal)

AWAY (${awayTeam.name}):
- Possession: ${awayTeam.stats.possession}%
- Passes: ${awayTeam.stats.passes} (accuracy est. ${awayTeam.stats.passAccuracy}%)
- Shots: ${awayTeam.stats.shots} (on target: ${awayTeam.stats.shotsOnTarget}) | Goals: ${awayTeam.stats.goals} | xG: ${awayTeam.stats.expectedGoals?.toFixed(2) ?? "n/a"}
- Metric confidence: possession ${awayTeam.stats.metricConfidence?.possession ?? 0}, passes ${awayTeam.stats.metricConfidence?.passes ?? 0}, shots ${awayTeam.stats.metricConfidence?.shots ?? 0}, distance ${awayTeam.stats.metricConfidence?.distance ?? 0}
- Tackles: ${awayTeam.stats.tackles} | Fouls: ${awayTeam.stats.fouls} | Corners: ${awayTeam.stats.corners}
- Avg player x-position: ${awayTeam.averagePosition.x}/100

EVENT CONFLICTS / REVIEW FLAGS:
${eventConflicts.length > 0
  ? eventConflicts.slice(0, 6).map((event) =>
      `- ${event.timestamp.toFixed(1)}s ${event.type}: ${(event.conflicts ?? []).join("; ") || event.pipelineFlag || "review flag"}`
    ).join("\n")
  : "- None"}

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
      model: SUMMARY_MODEL,
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
    const { frames, eventReviewWarnings } = body;

    if (!frames || frames.length === 0) {
      return NextResponse.json({ error: "No frames provided." }, { status: 400 });
    }

    const allEvents: MatchEvent[] = mergeDuplicateEvents(frames.flatMap((f) => f.events)).map((event): MatchEvent => {
      const source: MatchEvent["source"] = event.id.includes("-scoreboard-goal-")
        ? "scoreboard"
        : event.pipelineFlag === "low_confidence"
        ? "heuristic"
        : "llm";
      return ["shot", "goal", "save"].includes(event.type)
        ? { ...event, xg: estimateShotXg(event), source }
        : { ...event, source };
    });
    const eventConflicts = buildEventConflicts(allEvents);

    const homeTeam = buildTeamAnalysis("home", frames, allEvents);
    const awayTeam = buildTeamAnalysis("away", frames, allEvents);

    const client = new Anthropic({ apiKey });
    const insights = await generateInsights(client, homeTeam, awayTeam, eventConflicts);

    const analysis: MatchAnalysis = {
      id: `match-${Date.now()}`,
      processedAt: new Date().toISOString(),
      videoDuration: frames[frames.length - 1]?.timestamp ?? 0,
      framesAnalyzed: frames.length,
      homeTeam,
      awayTeam,
      frames,
      keyEvents: allEvents,
      eventConflicts,
      analysisWarnings: [
        "Replay and broadcast angle changes are flagged when detected, but may still require coach review.",
        "Possession is sampled from frame-level visual evidence, not counted as timeline events.",
        ...(eventReviewWarnings ?? []),
      ],
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
