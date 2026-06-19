import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type {
  MatchAnalysis,
  FrameData,
  MatchEvent,
  OutcomeProjection,
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
import { fieldPosition } from "@/lib/pitchMapping";
import { runVisionSynthesis, shotKey, type SynthesisKeyFrame, type VisionGoal } from "@/lib/visionSynthesis";

function playerFieldPosition(frame: FrameData, player: FrameData["players"][number]) {
  return fieldPosition(player.position, player.pitchPosition, frame.pitchView);
}

function calcDistanceCovered(teamId: TeamId, frames: FrameData[]): number {
  let total = 0;
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const curr = frames[i];
    for (const player of curr.players.filter((p) => p.team === teamId)) {
      if (!isStablePlayerId(player.id) || player.number <= 0) continue;
      const prevPlayer = prev.players.find((p) => p.id === player.id);
      if (!prevPlayer) continue;
      const step = distanceMeters(playerFieldPosition(curr, player), playerFieldPosition(prev, prevPlayer));
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
      const position = playerFieldPosition(frame, p);
      const col = Math.min(9, Math.max(0, Math.floor(position.x / 10)));
      const row = Math.min(9, Math.max(0, Math.floor(position.y / 10)));
      grid[row][col]++;
    }
  }
  const max = Math.max(...grid.flat(), 1);
  return grid.map((row) => row.map((v) => +(v / max).toFixed(2)));
}

function countEventType(events: MatchEvent[], type: string, team?: TeamId) {
  return events.filter((e) => e.type === type && (!team || e.team === team) && isVerifiedEvent(e)).length;
}

function scoreFromScoreboard(frames: FrameData[]): { home: number; away: number } | null {
  let hasReading = false;
  let home = 0;
  let away = 0;
  for (const frame of [...frames].sort((a, b) => a.timestamp - b.timestamp)) {
    const board = frame.scoreboard;
    if (
      !board ||
      !Number.isFinite(board.home) ||
      !Number.isFinite(board.away) ||
      board.home < 0 ||
      board.away < 0 ||
      board.home > 20 ||
      board.away > 20
    ) {
      continue;
    }
    hasReading = true;
    home = Math.max(home, board.home);
    away = Math.max(away, board.away);
  }
  return hasReading ? { home, away } : null;
}

// Goals scored *within this clip* = scoreboard delta from the first valid reading
// to the last. Robust to camera cuts (the scoreboard is a stable overlay, not
// affected by close-ups or which side players are on) and excludes goals already
// on the board before the upload started. Falls back to counted goal events when
// no scoreboard is legible.
function clipScopedGoals(
  frames: FrameData[],
  goalEvents: { home: number; away: number }
): { home: number; away: number } {
  const valid = [...frames]
    .sort((a, b) => a.timestamp - b.timestamp)
    .map((f) => f.scoreboard)
    .filter(
      (b): b is NonNullable<FrameData["scoreboard"]> =>
        !!b &&
        Number.isFinite(b.home) &&
        Number.isFinite(b.away) &&
        b.home >= 0 &&
        b.away >= 0 &&
        b.home <= 20 &&
        b.away <= 20
    );

  if (valid.length === 0) return goalEvents;

  const first = valid[0];
  const last = valid[valid.length - 1];
  return {
    home: Math.max(0, Math.min(20, last.home - first.home)),
    away: Math.max(0, Math.min(20, last.away - first.away)),
  };
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
  allEvents: MatchEvent[],
  scoreboardScore?: { home: number; away: number } | null
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
    f.players.filter((p) => p.team === id).map((p) => playerFieldPosition(f, p))
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
      goals: scoreboardScore ? scoreboardScore[id] : countEventType(teamEvents, "goal"),
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

  return insights.slice(0, 5).map((insight) => enrichInsightEvidence(insight, homeTeam, awayTeam, "fallback"));
}

function insightTeamNames(insight: CoachingInsight, homeTeam: TeamAnalysis, awayTeam: TeamAnalysis) {
  if (insight.affectedTeam === "home") return [homeTeam.name];
  if (insight.affectedTeam === "away") return [awayTeam.name];
  return [homeTeam.name, awayTeam.name];
}

function enrichInsightEvidence(
  insight: CoachingInsight,
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis,
  source: NonNullable<CoachingInsight["source"]>
): CoachingInsight {
  const teamNames = insightTeamNames(insight, homeTeam, awayTeam).join(" / ");
  const evidence = new Set<string>(insight.evidenceUsed ?? []);

  evidence.add(source === "claude" ? "Claude summary model interpreted CV/statistical metrics" : "deterministic fallback insight template");
  evidence.add(`clip-scoped scoreboard: ${homeTeam.name} ${homeTeam.stats.goals}-${awayTeam.stats.goals} ${awayTeam.name}`);

  if (insight.category === "possession") {
    evidence.add(`sampled possession: ${homeTeam.name} ${homeTeam.stats.possession}%, ${awayTeam.name} ${awayTeam.stats.possession}%`);
    evidence.add(`pass estimates: ${homeTeam.name} ${homeTeam.stats.passes}, ${awayTeam.name} ${awayTeam.stats.passes}`);
  } else if (insight.category === "attacking") {
    evidence.add(`shot/xG stats: ${homeTeam.name} ${homeTeam.stats.shots} shots/${homeTeam.stats.expectedGoals?.toFixed(2) ?? "n/a"} xG, ${awayTeam.name} ${awayTeam.stats.shots} shots/${awayTeam.stats.expectedGoals?.toFixed(2) ?? "n/a"} xG`);
  } else if (insight.category === "defensive") {
    evidence.add(`defensive events: ${homeTeam.name} ${homeTeam.stats.tackles} tackles, ${awayTeam.name} ${awayTeam.stats.tackles} tackles`);
    evidence.add(`shots conceded/on target context: ${homeTeam.name} ${homeTeam.stats.shotsOnTarget} SOT, ${awayTeam.name} ${awayTeam.stats.shotsOnTarget} SOT`);
  } else if (insight.category === "tactical") {
    evidence.add(`field-space average positions: ${homeTeam.name} x=${homeTeam.averagePosition.x}, ${awayTeam.name} x=${awayTeam.averagePosition.x}`);
  } else if (insight.category === "physical") {
    evidence.add(`stable-ID distance estimate: ${homeTeam.name} ${homeTeam.stats.distanceCovered}m, ${awayTeam.name} ${awayTeam.stats.distanceCovered}m`);
  }

  evidence.add(`affected team scope: ${teamNames}`);

  return {
    ...insight,
    source,
    evidenceUsed: [...evidence].slice(0, 5),
  };
}

// Deterministic outcome projection used when the vision synthesis call fails or
// returns nothing. Mirrors the old client-side heuristic (score gap + possession
// gap) so the dashboard always has a populated outcome model.
function buildFallbackOutcome(homeTeam: TeamAnalysis, awayTeam: TeamAnalysis): OutcomeProjection {
  const scoreGap = homeTeam.stats.goals - awayTeam.stats.goals;
  const possGap = homeTeam.stats.possession - awayTeam.stats.possession;
  const leadPct = Math.min(88, Math.max(52, 50 + Math.abs(scoreGap) * 14 + Math.abs(possGap) * 0.35));
  const draw = Math.max(6, Math.round((100 - leadPct) * 0.62));
  const other = Math.max(4, 100 - Math.round(leadPct) - draw);
  const homeLeads = scoreGap > 0 || (scoreGap === 0 && possGap >= 0);
  return {
    homeWin: homeLeads ? Math.round(leadPct) : other,
    draw,
    awayWin: homeLeads ? other : Math.round(leadPct),
    reasoning: "Heuristic projection from score and possession gap (vision synthesis unavailable).",
    source: "fallback",
  };
}

interface SynthesisOutput {
  insights: CoachingInsight[];
  outcome: OutcomeProjection;
  summary: string;
  goals: VisionGoal[];
  shotXg: Map<string, number>;
}

// Turn vision-detected goals into goal events, skipping any that duplicate a goal
// already on the timeline (e.g. one synthesized from the scoreboard) for the same
// team within 10s. Gives the timeline a goal with a real timestamp + team even
// when the scoreboard is illegible.
function buildVisionGoalEvents(goals: VisionGoal[], existing: MatchEvent[]): MatchEvent[] {
  const events: MatchEvent[] = [];
  for (const goal of goals) {
    const duplicate = [...existing, ...events].some(
      (e) => e.type === "goal" && e.team === goal.team && Math.abs(e.timestamp - goal.timestamp) <= 10
    );
    if (duplicate) continue;
    const event: MatchEvent = {
      id: `vision-goal-${goal.timestamp.toFixed(1)}-${goal.team}`,
      timestamp: goal.timestamp,
      type: "goal",
      team: goal.team,
      description: goal.description,
      confidence: 0.82,
      isKeyMoment: true,
      source: "llm",
      evidenceUsed: ["vision model confirmed ball crossing the line / goal restart"],
    };
    events.push({ ...event, xg: estimateShotXg(event), xgSource: "vision" });
  }
  return events;
}

// Drop near-duplicate insights (same title, or same category targeting the same
// team with near-identical observations) so the list doesn't repeat the same
// point. Keeps the first occurrence.
function dedupeInsights(insights: CoachingInsight[]): CoachingInsight[] {
  const seenTitles = new Set<string>();
  const seenKeys = new Set<string>();
  const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
  const result: CoachingInsight[] = [];
  for (const insight of insights) {
    const title = norm(insight.title);
    const key = `${insight.category}|${insight.affectedTeam}|${norm(insight.observation).slice(0, 60)}`;
    if (seenTitles.has(title) || seenKeys.has(key)) continue;
    seenTitles.add(title);
    seenKeys.add(key);
    result.push(insight);
  }
  return result;
}

// Runs the single vision-grounded synthesis call (Opus 4.8, multi-image) and
// adapts its output to the shapes the rest of the route expects. Falls back to
// deterministic insights/outcome on any failure so a flaky model call never
// breaks the analysis.
async function generateSynthesis(
  client: Anthropic,
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis,
  shotEvents: MatchEvent[],
  eventConflicts: NonNullable<MatchAnalysis["eventConflicts"]>,
  keyFrames: SynthesisKeyFrame[]
): Promise<SynthesisOutput> {
  try {
    const result = await runVisionSynthesis(client, {
      homeTeam,
      awayTeam,
      shotEvents,
      eventConflicts,
      keyFrames,
    });

    const insights = dedupeInsights(
      result.insights.length > 0
        ? result.insights.map((insight) => enrichInsightEvidence(insight, homeTeam, awayTeam, "claude"))
        : buildFallbackInsights(homeTeam, awayTeam)
    );

    const outcome: OutcomeProjection = result.outcome
      ? { ...result.outcome, source: "vision" }
      : buildFallbackOutcome(homeTeam, awayTeam);

    return { insights, outcome, summary: result.summary, goals: result.goals, shotXg: result.shotXg };
  } catch {
    return {
      insights: dedupeInsights(buildFallbackInsights(homeTeam, awayTeam)),
      outcome: buildFallbackOutcome(homeTeam, awayTeam),
      summary: "",
      goals: [],
      shotXg: new Map(),
    };
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
    const { frames, eventReviewWarnings, keyFrames } = body;

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
    const scoreboardScore = scoreFromScoreboard(frames);

    // Team analysis built first to give the synthesis model numeric context.
    // Rebuilt below once vision xG is merged so the displayed team xG reflects it.
    const homeTeamPre = buildTeamAnalysis("home", frames, allEvents, scoreboardScore);
    const awayTeamPre = buildTeamAnalysis("away", frames, allEvents, scoreboardScore);
    const shotEvents = shotLikeEvents(allEvents);

    const client = new Anthropic({ apiKey });
    const synthesis = await generateSynthesis(
      client,
      homeTeamPre,
      awayTeamPre,
      shotEvents,
      eventConflicts ?? [],
      (keyFrames ?? []) as SynthesisKeyFrame[]
    );

    // Merge per-shot vision xG onto matching events; tag every shot-like event
    // with whether its xG came from the model looking at the frame or the formula.
    const keyEvents: MatchEvent[] = allEvents.map((event) => {
      if (!["shot", "goal", "save"].includes(event.type)) return event;
      const visionXg = synthesis.shotXg.get(shotKey(event.timestamp));
      return visionXg != null
        ? { ...event, xg: visionXg, xgSource: "vision" as const }
        : { ...event, xgSource: "positional" as const };
    });

    // Add goals the vision model confirmed from the frames (ball crossing the
    // line), skipping any that duplicate an existing goal on the timeline.
    keyEvents.push(...buildVisionGoalEvents(synthesis.goals, keyEvents));
    keyEvents.sort((a, b) => a.timestamp - b.timestamp);

    const homeTeam = buildTeamAnalysis("home", frames, keyEvents, scoreboardScore);
    const awayTeam = buildTeamAnalysis("away", frames, keyEvents, scoreboardScore);

    const analysis: MatchAnalysis = {
      id: `match-${Date.now()}`,
      processedAt: new Date().toISOString(),
      videoDuration: frames[frames.length - 1]?.timestamp ?? 0,
      framesAnalyzed: frames.length,
      homeTeam,
      awayTeam,
      frames,
      keyEvents,
      eventConflicts,
      analysisWarnings: [
        "Replay and broadcast angle changes are flagged when detected, but may still require coach review.",
        "Possession is sampled from frame-level visual evidence, not counted as timeline events.",
        "Pass accuracy and distance covered are low-confidence: player IDs are assigned per frame, so cross-frame passing and movement can't be tracked reliably from broadcast angle.",
        ...(scoreboardScore ? ["Final score is taken from scoreboard reads; goal timeline only includes score changes observed after the clip baseline."] : []),
        ...(eventReviewWarnings ?? []),
      ],
      insights: synthesis.insights,
      outcome: synthesis.outcome,
      clipSummary: synthesis.summary || undefined,
      score: {
        home: scoreboardScore?.home ?? countEventType(keyEvents, "goal", "home"),
        away: scoreboardScore?.away ?? countEventType(keyEvents, "goal", "away"),
      },
      clipGoals: clipScopedGoals(frames, {
        home: countEventType(keyEvents, "goal", "home"),
        away: countEventType(keyEvents, "goal", "away"),
      }),
      processingMethod: "ai",
    };

    return NextResponse.json(analysis);
  } catch (err) {
    console.error("[/api/analyze/summarize]", err);
    return NextResponse.json({ error: "Summarize failed." }, { status: 500 });
  }
}
