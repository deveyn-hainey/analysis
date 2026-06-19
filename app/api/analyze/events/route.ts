import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { AnalyzeEventsRequest, FrameData, MatchEvent, TeamId } from "@/lib/types";
import { fieldPosition, mapImagePositionToPitch } from "@/lib/pitchMapping";

const EVENT_MODEL = process.env.ANTHROPIC_EVENT_MODEL ?? process.env.ANTHROPIC_SUMMARY_MODEL ?? "claude-sonnet-4-6";
const EVENT_REVIEW_BATCH_SIZE = 4;
const EVENT_REVIEW_MAX_ATTEMPTS = 3;
const MAX_CANDIDATE_WINDOWS = 40;
const DRIBBLE_DISTANCE_THRESHOLD = 6;
const PASS_DISTANCE_THRESHOLD = 16;

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
  scoreboard?: { home: number; away: number; homeLabel?: string; awayLabel?: string } | null;
}

interface RawEventReview {
  frames?: RawFrameUpdate[];
}

type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
type TextBlock = { type: "text"; text: string; cache_control?: { type: "ephemeral" } };
type ReviewContentBlock = ImageBlock | TextBlock;

function cleanJson(text: string) {
  const trimmed = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  const firstObject = trimmed.indexOf("{");
  const lastObject = trimmed.lastIndexOf("}");
  if (firstObject >= 0 && lastObject > firstObject) {
    return trimmed.slice(firstObject, lastObject + 1);
  }
  return trimmed;
}

// Closes unclosed strings, arrays, and objects left open by a token-limit truncation.
// Does not fix unescaped quotes mid-string — those are rarer and harder to repair safely.
function repairTruncatedJson(text: string): string {
  let inString = false;
  let escape = false;
  const stack: string[] = [];

  for (const c of text) {
    if (escape) { escape = false; continue; }
    if (c === "\\" && inString) { escape = true; continue; }
    if (c === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (c === "{") stack.push("}");
    else if (c === "[") stack.push("]");
    else if (c === "}" || c === "]") stack.pop();
  }

  let repaired = text.trimEnd();
  if (inString) repaired += '"';
  for (let i = stack.length - 1; i >= 0; i--) repaired += stack[i];
  return repaired;
}

// Extracts whatever complete frame objects exist from a (possibly truncated) response.
function recoverPartialFrames(text: string): RawFrameUpdate[] {
  const cleaned = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  const match = cleaned.match(/"frames"\s*:\s*\[/);
  if (!match || match.index === undefined) return [];

  const arrayStart = match.index + match[0].length;
  const frames: RawFrameUpdate[] = [];
  let i = arrayStart;

  while (i < cleaned.length) {
    while (i < cleaned.length && /[\s,]/.test(cleaned[i])) i++;
    if (i >= cleaned.length || cleaned[i] === "]") break;
    if (cleaned[i] !== "{") break;

    let depth = 0;
    let inStr = false;
    let esc = false;
    const start = i;

    for (; i < cleaned.length; i++) {
      const c = cleaned[i];
      if (esc) { esc = false; continue; }
      if (c === "\\" && inStr) { esc = true; continue; }
      if (c === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (c === "{") depth++;
      else if (c === "}") {
        depth--;
        if (depth === 0) {
          try { frames.push(JSON.parse(cleaned.slice(start, i + 1)) as RawFrameUpdate); } catch { /* malformed object, skip */ }
          i++;
          break;
        }
      }
    }
  }

  return frames;
}

function errorMessage(err: unknown) {
  if (err instanceof Error) return err.message;
  return String(err);
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  const updates = new Map<number, RawFrameUpdate>();
  for (const frameUpdate of review.frames ?? []) {
    const existing = updates.get(frameUpdate.frameIndex);
    updates.set(frameUpdate.frameIndex, {
      frameIndex: frameUpdate.frameIndex,
      possession: frameUpdate.possession ?? existing?.possession,
      events: [...(existing?.events ?? []), ...(frameUpdate.events ?? [])],
      scoreboard: frameUpdate.scoreboard ?? existing?.scoreboard,
    });
  }
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
      scoreboard: update.scoreboard !== undefined ? update.scoreboard : frame.scoreboard,
    };
  });
}

function eventKey(event: MatchEvent) {
  return `${Math.round(event.timestamp)}-${event.team ?? "none"}-${event.type}`;
}

function tooCloseToExisting(event: MatchEvent, events: MatchEvent[], windowSeconds: number) {
  return events.some((existing) => {
    if (existing.type !== event.type || existing.team !== event.team) return false;
    if (Math.abs(existing.timestamp - event.timestamp) > windowSeconds) return false;
    if (event.position && existing.position) {
      const d = Math.hypot(event.position.x - existing.position.x, event.position.y - existing.position.y);
      return d <= 14;
    }
    return true;
  });
}

// Confirms goals from scoreboard reads alone, comparing every frame's scoreboard
// across the whole clip — not just within one Claude review batch. This is what
// catches a goal whose live action was never confirmed (candidate heuristics missed
// it, or that batch's Claude review call failed/timed out/was skipped) but whose
// aftermath frame still legibly shows the new score. A scoreboard digit read is a
// much stronger, near-deterministic signal than ball-near-goal-zone proximity, so
// this runs independent of pipelineFlag/low_confidence and isn't gated on any
// particular batch having succeeded.
function synthesizeGoalsFromScoreboard(frames: FrameData[]): FrameData[] {
  const sorted = [...frames].sort((a, b) => a.timestamp - b.timestamp);
  const runningMax: Record<TeamId, number> = { home: 0, away: 0 };
  let hasBaseline = false;
  const additions = new Map<number, MatchEvent[]>();

  for (const frame of sorted) {
    if (!frame.scoreboard) continue;

    if (!hasBaseline) {
      runningMax.home = Math.max(0, frame.scoreboard.home);
      runningMax.away = Math.max(0, frame.scoreboard.away);
      hasBaseline = true;
      continue;
    }

    for (const team of ["home", "away"] as TeamId[]) {
      const seen = frame.scoreboard[team];
      // Use the running max (not just the previous frame) as the baseline so a
      // single misread frame that shows a lower score after the real goal doesn't
      // get treated as a second event, and a goal already counted doesn't repeat.
      if (typeof seen === "number" && Number.isFinite(seen) && seen > runningMax[team]) {
        const previous = runningMax[team];
        const event: MatchEvent = {
          id: `f${frame.frameIndex}-scoreboard-goal-${team}-${seen}`,
          timestamp: frame.timestamp,
          type: "goal",
          team,
          description: `Scoreboard read ${frame.scoreboard.homeLabel ?? "Home"} ${frame.scoreboard.home}-${frame.scoreboard.away} ${frame.scoreboard.awayLabel ?? "Away"} — ${team === "home" ? frame.scoreboard.homeLabel ?? "Home" : frame.scoreboard.awayLabel ?? "Away"} goal confirmed from score overlay`,
          confidence: 0.92,
          isKeyMoment: true,
          evidenceUsed: [`scoreboard read ${seen} for ${team}, up from ${previous} previously observed after the clip baseline`],
          source: "scoreboard",
        };
        additions.set(frame.frameIndex, [...(additions.get(frame.frameIndex) ?? []), event]);
        runningMax[team] = seen;
      }
    }
  }

  if (additions.size === 0) return frames;
  return frames.map((frame) =>
    additions.has(frame.frameIndex)
      ? { ...frame, events: [...frame.events, ...additions.get(frame.frameIndex)!] }
      : frame
  );
}

function synthesizeMovementEvents(frames: FrameData[]): FrameData[] {
  const existing = new Set(frames.flatMap((f) => f.events.map(eventKey)));
  const allEvents = frames.flatMap((f) => f.events);
  const lastRoutine = new Map<string, MatchEvent>();

  return frames.map((frame, i) => {
    const prev = frames[i - 1];
    const additions: MatchEvent[] = [];
    const prevBall = prev ? fieldBallPosition(prev) : undefined;
    const currentBall = fieldBallPosition(frame);

    if (prev && prevBall && currentBall) {
      const distance = Math.hypot(
        currentBall.x - prevBall.x,
        currentBall.y - prevBall.y
      );
      const sameTeamPossession =
        frame.possession !== "contested" &&
        prev.possession === frame.possession;
      const possessionTurnover =
        prev.possession !== "contested" &&
        frame.possession !== "contested" &&
        prev.possession !== frame.possession;

      if (sameTeamPossession && distance >= DRIBBLE_DISTANCE_THRESHOLD) {
        const samePlayer =
          prev.possessingPlayer?.team === frame.possessingPlayer?.team &&
          prev.possessingPlayer?.playerId === frame.possessingPlayer?.playerId;
        const type: MatchEvent["type"] = distance >= PASS_DISTANCE_THRESHOLD ? "pass" : "dribble";
        if (type === "pass" && samePlayer && distance < PASS_DISTANCE_THRESHOLD * 1.5) return { ...frame, events: frame.events };
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
          position: currentBall,
          evidenceUsed: [`ball moved ${distance.toFixed(1)} pitch units while possession stayed ${frame.possession}`],
          conflicts: ["synthetic event from tracking, not visually verified by Claude"],
          pipelineFlag: "low_confidence",
        };
        const routineKey = `${team}-${type}`;
        const previousRoutine = lastRoutine.get(routineKey);
        if (
          !existing.has(eventKey(event)) &&
          !tooCloseToExisting(event, allEvents, type === "pass" ? 5 : 4) &&
          (!previousRoutine || Math.abs(event.timestamp - previousRoutine.timestamp) > (type === "pass" ? 4 : 3))
        ) {
          additions.push(event);
          allEvents.push(event);
          lastRoutine.set(routineKey, event);
        }
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
          position: currentBall,
          semanticLabel: "high_press_turnover",
          evidenceUsed: [`possession changed from ${prev.possession} to ${frame.possession}`],
          conflicts: ["classified as tackle/turnover from tracking signal"],
          pipelineFlag: "low_confidence",
        };
        const routineKey = `${team}-tackle`;
        const previousRoutine = lastRoutine.get(routineKey);
        if (
          !existing.has(eventKey(event)) &&
          !tooCloseToExisting(event, allEvents, 5) &&
          (!previousRoutine || Math.abs(event.timestamp - previousRoutine.timestamp) > 5)
        ) {
          additions.push(event);
          allEvents.push(event);
          lastRoutine.set(routineKey, event);
        }
      }
    }

    return additions.length
      ? { ...frame, events: [...frame.events, ...additions] }
      : frame;
  });
}

function fieldBallPosition(frame: FrameData) {
  return frame.pitchBall ?? (frame.ballPosition ? mapImagePositionToPitch(frame.ballPosition, frame.pitchView) : undefined);
}

function fieldPlayerPosition(player: FrameData["players"][number], frame: FrameData) {
  return fieldPosition(player.position, player.pitchPosition, frame.pitchView);
}

function nearestPlayers(frame: FrameData, count = 4) {
  const ball = fieldBallPosition(frame);
  if (!ball) return [];
  return frame.players
    .map((p) => {
      const position = fieldPlayerPosition(p, frame);
      return {
        id: p.id,
        team: p.team,
        role: p.role,
        position,
        distance: Math.hypot(position.x - ball.x, position.y - ball.y),
      };
    })
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
    const ball = fieldBallPosition(frame);
    const nearest = nearestPlayers(frame, 3);
    const window = {
      start_timestamp: Math.max(0, prev?.timestamp ?? frame.timestamp - 4),
      end_timestamp: next?.timestamp ?? frame.timestamp + 4,
    };

    if (ball) {
      // Goal-mouth width only — without a y-bound this fired for every ordinary
      // byline touch (including corners), eating most of the candidate budget
      // with "possible goal" noise instead of routine play.
      const inGoalZone = (ball.x <= 8 || ball.x >= 92) && ball.y >= 32 && ball.y <= 68;
      const inShotZone = ball.x <= 22 || ball.x >= 78;
      const nearCorner = (ball.x <= 10 || ball.x >= 90) && (ball.y <= 12 || ball.y >= 88);
      const nearestOpponent = nearest.find((player) =>
        nearest[0] && player.team !== nearest[0].team
      );

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

      const prevBall = prev ? fieldBallPosition(prev) : undefined;
      const nextBall = next ? fieldBallPosition(next) : undefined;

      if (prevBall && nextBall) {
        const v1 = {
          x: ball.x - prevBall.x,
          y: ball.y - prevBall.y,
        };
        const v2 = {
          x: nextBall.x - ball.x,
          y: nextBall.y - ball.y,
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

      if (prevBall && frame.possession !== "contested") {
        const distance = Math.hypot(ball.x - prevBall.x, ball.y - prevBall.y);
        if (distance >= DRIBBLE_DISTANCE_THRESHOLD) {
          candidates.push({
            candidate_type: distance >= PASS_DISTANCE_THRESHOLD ? "possible_pass" : "possible_dribble",
            frameIndex: frame.frameIndex,
            window,
            signals: {
              ball_progression_distance: +distance.toFixed(1),
              possession_team: frame.possession,
              nearest_players: nearest,
            },
          });
        }
      }

      if (nearest[0] && nearestOpponent && nearest[0].distance <= 12 && nearestOpponent.distance <= 14) {
        candidates.push({
          candidate_type: "possible_pressure_or_challenge",
          frameIndex: frame.frameIndex,
          window,
          signals: {
            nearest_ball_player: nearest[0],
            nearest_opponent: nearestOpponent,
            possession_team: frame.possession,
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

    if (!ball && frame.players.length > 0) {
      candidates.push({
        candidate_type: "possible_open_play_ball_not_detected",
        frameIndex: frame.frameIndex,
        window,
        signals: {
          ball_detected: false,
          yolo_player_count: frame.players.length,
          yolo_possession_estimate: frame.possession,
          instruction: "Use the image to correct possession. Still report routine passes/dribbles/challenges you can see even without a YOLO ball position — prefer a lower-confidence event over no event.",
        },
      });
    }

    if (prev) {
      // Celebration cluster: a sudden surge of players near either goal mouth is a
      // strong indicator of a goal even when there's no scoreboard and YOLO never
      // placed the ball in the net (common if the goal frame was not sampled or the
      // ball was occluded). We flag it as a candidate so Claude can visually confirm.
      const nearGoal = (p: FrameData["players"][number], side: "left" | "right") => {
        const position = fieldPlayerPosition(p, frame);
        return (side === "left" ? position.x <= 22 : position.x >= 78) &&
          position.y >= 28 && position.y <= 72;
      };

      for (const side of ["left", "right"] as const) {
        const currCount = frame.players.filter((p) => nearGoal(p, side)).length;
        const prevCount = prev.players.filter((p) => nearGoal(p, side)).length;
        if (currCount >= 4 && currCount > prevCount + 1) {
          candidates.push({
            candidate_type: "possible_goal_celebration_cluster",
            frameIndex: frame.frameIndex,
            window,
            signals: {
              player_surge_near_goal: true,
              side,
              current_player_count_near_goal: currCount,
              previous_player_count_near_goal: prevCount,
              possession_team: frame.possession,
              nearest_players: nearest,
              instruction:
                "Multiple players are suddenly clustered near this goal. Check the image for: ball in net, keeper retrieving ball, player celebrations (arms up, jumping, mobbing). If any are visible, report a goal event. If no celebration cues are present, report no event.",
            },
          });
          break; // only one side per frame to avoid duplicates
        }
      }
    }

    if (prev) {
      const playerCountDelta = Math.abs(frame.players.length - prev.players.length);
      const centroid = frame.players.length
        ? {
            x: frame.players.reduce((sum, player) => sum + fieldPlayerPosition(player, frame).x, 0) / frame.players.length,
            y: frame.players.reduce((sum, player) => sum + fieldPlayerPosition(player, frame).y, 0) / frame.players.length,
          }
        : null;
      const prevCentroid = prev.players.length
        ? {
            x: prev.players.reduce((sum, player) => sum + fieldPlayerPosition(player, prev).x, 0) / prev.players.length,
            y: prev.players.reduce((sum, player) => sum + fieldPlayerPosition(player, prev).y, 0) / prev.players.length,
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

  return candidates.slice(0, MAX_CANDIDATE_WINDOWS);
}

type CandidateWindow = ReturnType<typeof candidateWindows>[number];

function fallbackEventForCandidate(
  frame: FrameData,
  candidate: CandidateWindow
): RawEvent | null {
  const team = frame.possession === "home" || frame.possession === "away"
    ? frame.possession
    : undefined;
  const ball = fieldBallPosition(frame);

  if (candidate.candidate_type === "possible_goal_or_save") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "shot",
      team,
      description: "Unverified shot/save candidate from YOLO ball location near the goal zone",
      confidence: 0.38,
      position: ball,
      evidence_used: ["YOLO placed the ball in the goal zone", "Claude event review was unavailable for this window"],
      conflicts: ["not confirmed by Claude vision review"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_shot" || candidate.candidate_type === "possible_shot_save_or_deflection") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "shot",
      team,
      description: "Unverified shot candidate from YOLO ball movement in the attacking third",
      confidence: 0.35,
      position: ball,
      evidence_used: ["YOLO attacking-third ball position or direction change", "Claude event review was unavailable for this window"],
      conflicts: ["not confirmed by Claude vision review"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_corner_or_goal_kick") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "corner",
      team,
      description: "Unverified corner/byline restart candidate from YOLO ball location",
      confidence: 0.32,
      position: ball,
      evidence_used: ["YOLO placed the ball near the byline/corner area", "Claude event review was unavailable for this window"],
      conflicts: ["could be corner, goal kick, or open play"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_pass" || candidate.candidate_type === "possible_dribble") {
    const type = candidate.candidate_type === "possible_pass" ? "pass" : "dribble";
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type,
      team,
      description: `Unverified ${type} candidate from YOLO ball progression`,
      confidence: type === "pass" ? 0.42 : 0.4,
      position: ball,
      evidence_used: ["YOLO tracked ball progression while one team retained possession", "Claude event review may not have confirmed this window"],
      conflicts: ["tracking-derived event"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_pressure_or_challenge") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "tackle",
      team,
      description: "Unverified pressure/challenge candidate from players converging near the ball",
      confidence: 0.36,
      position: ball,
      evidence_used: ["YOLO detected nearby opponents around the ball", "Claude event review may not have confirmed this window"],
      conflicts: ["could be pressure, tackle attempt, or normal marking"],
      pipeline_flag: "low_confidence",
    };
  }

  // YOLO frequently misses the ball in wide broadcast shots (small, fast-moving object
  // on a generic, non-soccer-trained model). Without this case, those frames produced
  // no fallback at all when Claude's review batch failed — a major source of the
  // "no events" feeling, since ball-not-detected candidates are common.
  if (candidate.candidate_type === "possible_open_play_ball_not_detected" && team) {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "dribble",
      team,
      description: "Unverified ball-progression candidate; YOLO did not detect the ball this frame",
      confidence: 0.3,
      evidence_used: ["YOLO possession estimate from nearest players, ball position missing", "Claude event review was unavailable for this window"],
      conflicts: ["ball position not detected by YOLO", "not confirmed by Claude vision review"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_goal_celebration_cluster") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "shot",
      team,
      description: "Unverified possible goal: players clustered near goal zone but no visual confirmation from Claude",
      confidence: 0.4,
      position: ball,
      evidence_used: ["YOLO detected sudden player surge near goal area", "Claude event review was unavailable for this window"],
      conflicts: ["could be a corner, free kick wall, or crowded box — not visually confirmed"],
      pipeline_flag: "low_confidence",
    };
  }

  if (candidate.candidate_type === "possible_turnover_or_tackle") {
    return {
      frameIndex: frame.frameIndex,
      timestamp: frame.timestamp,
      type: "tackle",
      team,
      description: "Unverified possession turnover candidate from YOLO tracking",
      confidence: 0.34,
      position: ball,
      semantic_label: "high_press_turnover",
      evidence_used: ["YOLO possession estimate changed between sampled frames", "Claude event review was unavailable for this window"],
      conflicts: ["classified as tackle/turnover from tracking signal only"],
      pipeline_flag: "low_confidence",
    };
  }

  return null;
}

function fallbackUpdatesForFrames(
  frames: FrameData[],
  candidates: ReturnType<typeof candidateWindows>
): RawFrameUpdate[] {
  const framesByIndex = new Map(frames.map((frame) => [frame.frameIndex, frame]));
  const updates = new Map<number, RawFrameUpdate>();

  for (const candidate of candidates) {
    const frame = framesByIndex.get(candidate.frameIndex);
    if (!frame) continue;

    const update = updates.get(frame.frameIndex) ?? {
      frameIndex: frame.frameIndex,
      possession: frame.possession,
      events: [],
    };
    const fallback = fallbackEventForCandidate(frame, candidate);
    if (fallback) {
      update.events = [...(update.events ?? []), fallback];
      updates.set(frame.frameIndex, update);
    }
  }

  return [...updates.values()];
}

function buildReviewContent(
  images: AnalyzeEventsRequest["images"],
  compactFrames: Array<Record<string, unknown>>,
  candidates: ReturnType<typeof candidateWindows>
): ReviewContentBlock[] {
  const content: ReviewContentBlock[] = [
    {
      type: "text",
      text: `You are a soccer match analysis verifier for fixed wide-angle tactical camera footage.
You operate as an EVENT-RICH SOCCER TIMELINE REVIEWER.

Upstream already ran:
- YOLO football detection for players and ball
- clip-level jersey color clustering for team assignment
- nearest-player possession estimate per sampled frame
- simple deterministic candidate generation from ball location and possession changes

Your job:
- For every supplied frame image, scan for a scoreboard/score overlay and report it (see SCOREBOARD READING below) — do this independent of whether you confirm any event for that frame.
- Verify candidate windows using the images and YOLO metadata.
- Correct possession only when the visual evidence is clearer than YOLO's nearest-player estimate.
- Build a useful event timeline, not only rare highlights.
- Report meaningful passes, dribbles/carries, pressure/challenges, turnovers, shots, saves, restarts, and goals when visible or strongly implied.
- Use lower confidence for tracking-implied routine events instead of omitting them.
- Low confidence beats a blank timeline.
- If YOLO missed the ball, use the frame image to set possession when one team is clearly in controlled possession.
- If no major event is visible, still return possession corrections and routine ball-progression events where the metadata and image support them.

SCOREBOARD READING (do this for every frame, separate from event confirmation):
- Scan every corner/edge of the image for a score overlay, ticker, or graphic (e.g. "1-0", "USA 1 PAR 0").
- If legible, report the exact numeric score as {"home": <int>, "away": <int>} using the same team labels as possession.
- If the overlay includes team/country/club labels or abbreviations (e.g. "USA", "PAR", "STO", "ARS"), include them as "homeLabel" and "awayLabel". Use the exact visible label, trimmed to 2-20 characters. If labels are not legible, omit the label fields.
- If no scoreboard is visible or you can't read it confidently, report it as null. Do not guess.
- You do NOT need to compare this to other batches or remember prior scores — just report what this single frame shows. Score increases are detected deterministically downstream by comparing your readings across every frame in the clip, including frames reviewed in other batches. This means you don't need certainty about whether a score "changed" to report it — an accurate reading of a frame that already shows the post-goal score is exactly what's needed, even if you have no visual record of the goal itself.

Return ONLY valid JSON:
{
  "frames": [
    {
      "frameIndex": 0,
      "possession": "home",
      "scoreboard": { "home": 1, "away": 0, "homeLabel": "USA", "awayLabel": "PAR" },
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
"scoreboard" must be present on every frame entry — use null when not legible.

Rules:
- possession must be "home", "away", or "contested".
- Allowed event types: goal, shot, shot_saved, shot_off_target, corner, goal_kick, card_yellow, card_red, card_unknown, foul, freekick, offside, pass, tackle, throw-in, dribble.
- Every event MUST include a numeric timestamp copied from one of the supplied frame timestamps. Do not infer time from frame order alone.
- Only report events visible or strongly implied inside candidate windows.
- Aim for a useful timeline: 1–3 events per active candidate window is acceptable when play is moving.
- Do not create duplicate events for the same action across adjacent frames.
- If no candidate is convincing, return an empty events array for that frame, but do not be overly conservative for routine passes/dribbles.
- Pass/dribble/tackle events may be lower confidence when inferred from YOLO ball movement plus visible player context.

Routine event rules:
- pass: ball moves meaningfully between teammates or advances quickly while the same team keeps possession.
- dribble: ball carrier advances or carries under control while possession stays with the same team.
- tackle: opponent pressure/challenge, ball contest, or possession regain; use "tackle" for pressure/regain even when exact contact is uncertain.
- shot: ball is directed toward goal, enters the box/goal channel, forces goalkeeper reaction, or has a visible shooting body shape.
- save: goalkeeper block/catch/parry or shot stopped near goal.
- corner/goal_kick/throw-in: ball near boundary with restart shape; use lower confidence when uncertain.

Goal rules:
- Do NOT confirm a goal from one weak signal.
- Strong confirmation: visible ball in net, keeper retrieving from net, or obvious non-half-boundary kickoff aftermath, all within this batch's own images.
- Do NOT emit a goal event yourself just because this frame's scoreboard shows a goal already happened — you only see this batch, not the whole clip, so you can't tell if that score is new or was already there before this batch started. Just report it accurately in the "scoreboard" field above; an increase is detected automatically by comparing your readings across the whole clip, including frames outside this batch.
- Supporting signals: ball in goal zone, ball disappears in goalmouth, celebration cluster near goal, defending shape collapse.
- Rejection signals: keeper catches/parries, ball visible wide/above goal, restart appears to be corner/goal kick, no score/restart evidence.
- SCOREBOARD-FREE GOAL DETECTION: When no scoreboard is visible, rely on visual cues alone:
  * If a "possible_goal_celebration_cluster" candidate is present, carefully scan the image for celebrating players (arms raised, jumping, mobbing, running together), ball in/near net, or goalkeeper dejection. These are sufficient to confirm a goal without a scoreboard.
  * A clear celebration cluster (4+ players celebrating near a goal) combined with the attacking team in possession moments before is strong enough to confirm a goal at confidence 0.80–0.88.
  * Absence of celebration in a "possible_goal_celebration_cluster" candidate means the cluster was likely a corner, free kick, or congested box — do not emit a goal.
- Goal recall triggers include ball direction reversal near the box and possible camera cut/replay/angle change after an attacking moment. These triggers should cause verification, not automatic confirmation.
- Never confirm a goal solely from a scoreboard when you cannot see live action.
- If ambiguous, label shot or no event with lower confidence rather than goal.
- Replays and camera cuts are a known limitation. If a frame appears to be a replay or angle cut, add "replay_suspected" to pipeline_flag and do not count a fresh live event unless the timestamp window clearly shows live play.
- If evidence conflicts, keep the event only when confidence remains justified and put the disagreement in conflicts. Use pipeline_flag "verifier_conflict" or "scoreboard_conflict" when appropriate.

Semantic labels after confirmation:
- counterattack_goal
- set_piece_goal
- sustained_buildup
- high_press_turnover
- direct_play
- null when unsupported`,
      cache_control: { type: "ephemeral" },
    },
    {
      type: "text",
      text: `YOLO metadata:
${JSON.stringify(compactFrames)}

Candidate windows:
${JSON.stringify(candidates)}`,
    },
  ];

  for (const image of images) {
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

  return content;
}

async function reviewFrameBatch(
  client: Anthropic,
  images: AnalyzeEventsRequest["images"],
  frames: FrameData[],
  candidates: ReturnType<typeof candidateWindows>
): Promise<RawEventReview> {
  const frameIndexes = new Set(frames.map((frame) => frame.frameIndex));
  const compactFrames = frames.map((frame) => ({
    frameIndex: frame.frameIndex,
    timestamp: frame.timestamp,
    possession: frame.possession,
    ballPosition: frame.ballPosition,
    fieldBallPosition: fieldBallPosition(frame),
    hasPitchCalibration: Boolean(frame.pitchBall || frame.players.some((p) => p.pitchPosition)),
    playerCount: frame.players.length,
    homePlayers: frame.players.filter((p) => p.team === "home").length,
    awayPlayers: frame.players.filter((p) => p.team === "away").length,
    playersNearBall: nearestPlayers(frame),
  }));
  const batchCandidates = candidates.filter((candidate) =>
    frameIndexes.has(candidate.frameIndex)
  );

  let lastError: unknown;
  for (let attempt = 1; attempt <= EVENT_REVIEW_MAX_ATTEMPTS; attempt++) {
    try {
      const response = await client.messages.create({
        model: EVENT_MODEL,
        // 4 frames per batch × ~3 events × ~80 tokens each ≈ 1000 tokens output.
        // 2500 gives 2.5× headroom for busy passages without the 50-60s response
        // times we saw at 5000 tokens with 8-frame batches.
        max_tokens: 2500,
        messages: [{ role: "user", content: buildReviewContent(images, compactFrames, batchCandidates) }],
      });

      const raw = response.content[0].type === "text" ? response.content[0].text : "{}";

      // Try clean parse → repaired parse → partial frame extraction
      try {
        return JSON.parse(cleanJson(raw)) as RawEventReview;
      } catch {
        try {
          return JSON.parse(cleanJson(repairTruncatedJson(raw))) as RawEventReview;
        } catch {
          const recovered = recoverPartialFrames(raw);
          if (recovered.length > 0) return { frames: recovered };
          throw new Error("JSON parse failed after repair and partial recovery");
        }
      }
    } catch (err) {
      lastError = err;
      if (attempt < EVENT_REVIEW_MAX_ATTEMPTS) {
        await sleep(700 * attempt);
      }
    }
  }

  throw lastError;
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

    const candidates = candidateWindows(frames);
    const client = new Anthropic({ apiKey });
    const reviewedUpdates: RawFrameUpdate[] = [];
    const reviewWarnings: string[] = [];

    for (let start = 0; start < frames.length; start += EVENT_REVIEW_BATCH_SIZE) {
      const batchFrames = frames.slice(start, start + EVENT_REVIEW_BATCH_SIZE);
      const batchImages = images.slice(start, start + EVENT_REVIEW_BATCH_SIZE);
      try {
        const review = await reviewFrameBatch(client, batchImages, batchFrames, candidates);
        reviewedUpdates.push(...(review.frames ?? []));
      } catch (err) {
        const warning = `Claude event review skipped for frames ${batchFrames[0]?.frameIndex}-${batchFrames.at(-1)?.frameIndex}: ${errorMessage(err)}`;
        console.error("[/api/analyze/events] batch failed", warning);
        reviewWarnings.push(warning);
      }
    }

    const framesWithReviewedEvents = new Set(
      reviewedUpdates
        .filter((update) => (update.events?.length ?? 0) > 0)
        .map((update) => update.frameIndex)
    );
    const fallbackUpdates = fallbackUpdatesForFrames(
      frames,
      candidates.filter((candidate) => !framesWithReviewedEvents.has(candidate.frameIndex))
    );
    const reviewedFrames = mergeReviewedFrames(frames, {
      frames: [...fallbackUpdates, ...reviewedUpdates],
    });

    // Scoreboard synthesis and goal deduplication are intentionally NOT run here.
    // The client batches requests (one per 8 frames) and each batch would start
    // runningMax at 0, causing every batch that contains a "1-0" scoreboard frame
    // to independently fire a goal — producing duplicates (e.g. three "1-0" goals
    // instead of one). The client merges all batch results and runs synthesis once
    // across the full clip where runningMax correctly tracks across all frames.
    return NextResponse.json({
      frames: synthesizeMovementEvents(reviewedFrames),
      warnings: reviewWarnings,
    });
  } catch (err) {
    console.error("[/api/analyze/events]", err);
    return NextResponse.json({ error: `Event review failed: ${errorMessage(err)}` }, { status: 500 });
  }
}
