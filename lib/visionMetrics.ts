import type { FrameData, MatchEvent, Position, TeamId } from "@/lib/types";
import { fieldPosition } from "@/lib/pitchMapping";

const PITCH_M_X = 105;
const PITCH_M_Y = 68;

export function isVerifiedEvent(event: MatchEvent) {
  return event.pipelineFlag !== "low_confidence" && event.pipelineFlag !== "replay_suspected";
}

export function goalX(team: TeamId) {
  return team === "home" ? 100 : 0;
}

export function estimateShotXg(event: Pick<MatchEvent, "type" | "team" | "position" | "confidence" | "semanticLabel">) {
  if (event.type === "goal") return Math.max(0.18, Math.min(0.95, (event.confidence || 0.8) * 0.62));
  if (!event.team || !event.position) return event.type === "save" ? 0.12 : 0.07;

  const gx = goalX(event.team);
  const dx = Math.abs(gx - event.position.x);
  const dy = Math.abs(50 - event.position.y);
  const distanceMeters = Math.hypot((dx / 100) * PITCH_M_X, (dy / 100) * PITCH_M_Y);
  const angleQuality = Math.max(0, 1 - dy / 42);
  const distanceQuality = 1 / (1 + Math.exp((distanceMeters - 16) / 4.5));
  const centralBoost = 0.06 * angleQuality;
  const targetBoost = event.type === "save" ? 0.05 : 0;
  const labelBoost =
    event.semanticLabel === "set_piece_goal" ? -0.02 :
    event.semanticLabel === "counterattack_goal" ? 0.03 :
    0;

  const xg = 0.025 + 0.34 * distanceQuality + centralBoost + targetBoost + labelBoost;
  return +Math.max(0.02, Math.min(0.78, xg)).toFixed(3);
}

export function shotLikeEvents(events: MatchEvent[], team?: TeamId) {
  return events.filter((event) =>
    ["shot", "goal", "save"].includes(event.type) &&
    (!team || event.team === team) &&
    isVerifiedEvent(event)
  );
}

export function teamExpectedGoals(events: MatchEvent[], team: TeamId) {
  // Prefer the per-shot xG already on the event (vision-estimated when available)
  // so the team total matches the shot map, finishing panel, and cumulative curve.
  // Fall back to the positional estimate only when an event has no xG yet.
  return +shotLikeEvents(events, team)
    .reduce((sum, event) => sum + (event.xg ?? estimateShotXg(event)), 0)
    .toFixed(2);
}

export function distanceMeters(a: Position, b: Position) {
  const dx = ((a.x - b.x) / 100) * PITCH_M_X;
  const dy = ((a.y - b.y) / 100) * PITCH_M_Y;
  return Math.hypot(dx, dy);
}

export function isStablePlayerId(id: string) {
  return /^[ha]\d+$/.test(id);
}

export function stableTrackingCoverage(frames: FrameData[], team?: TeamId) {
  const players = frames.flatMap((frame) => frame.players.filter((player) => !team || player.team === team));
  if (!players.length) return 0;
  return players.filter((player) => isStablePlayerId(player.id) && player.number > 0).length / players.length;
}

// Count passes from the per-frame possession chain instead of the throttled
// `pass` event timeline (which de-duplicates to ~1 per 4-5s and badly undercounts).
// A completed pass = the ball moving to a different teammate; a loss = possession
// handed to the other team. Pass accuracy = completed / (completed + lost), which
// naturally tracks possession dominance. Best run over dense (5fps) frames.
export function countPossessionPasses(
  frames: FrameData[]
): Record<TeamId, { completed: number; lost: number }> {
  const result: Record<TeamId, { completed: number; lost: number }> = {
    home: { completed: 0, lost: 0 },
    away: { completed: 0, lost: 0 },
  };
  const sorted = [...frames].sort((a, b) => a.timestamp - b.timestamp);
  // Require a new holder to persist this many frames before counting a change —
  // cuts per-frame ID/assignment flicker. Relaxed for sparse review frames.
  const minHold = sorted.length >= 60 ? 2 : 1;
  const keyOf = (p: { team: TeamId; playerId: string }) => `${p.team}:${p.playerId}`;

  let confirmed: { team: TeamId; playerId: string } | null = null;
  let pending: { team: TeamId; playerId: string } | null = null;
  let pendingCount = 0;

  for (const frame of sorted) {
    const p = frame.possessingPlayer;
    if (!p || frame.possession === "contested") continue; // hold last state through contested blips
    const holder = { team: p.team, playerId: p.playerId };
    if (confirmed && keyOf(holder) === keyOf(confirmed)) {
      pending = null;
      pendingCount = 0;
      continue;
    }
    if (pending && keyOf(holder) === keyOf(pending)) pendingCount += 1;
    else {
      pending = holder;
      pendingCount = 1;
    }
    if (pendingCount >= minHold) {
      if (confirmed) {
        if (confirmed.team === holder.team && confirmed.playerId !== holder.playerId) {
          result[holder.team].completed += 1;
        } else if (confirmed.team !== holder.team) {
          result[confirmed.team].lost += 1;
        }
      }
      confirmed = holder;
      pending = null;
      pendingCount = 0;
    }
  }
  return result;
}

export function buildPassNetwork(frames: FrameData[], team: TeamId) {
  const nodes = new Map<string, { id: string; number: number; team: TeamId; position: Position; touches: number }>();
  const links = new Map<string, { from: string; to: string; count: number }>();
  let previous: FrameData["possessingPlayer"] | undefined;

  for (const frame of [...frames].sort((a, b) => a.timestamp - b.timestamp)) {
    const touch = frame.possessingPlayer;
    if (!touch || touch.team !== team) {
      previous = touch;
      continue;
    }

    const player = frame.players.find((p) => p.id === touch.playerId && p.team === team);
    if (player) {
      const position = fieldPosition(player.position, player.pitchPosition, frame.pitchView);
      const current = nodes.get(player.id);
      nodes.set(player.id, {
        id: player.id,
        number: player.number,
        team,
        position: current
          ? {
              x: (current.position.x * current.touches + position.x) / (current.touches + 1),
              y: (current.position.y * current.touches + position.y) / (current.touches + 1),
            }
          : position,
        touches: (current?.touches ?? 0) + 1,
      });
    }

    if (previous?.team === team && previous.playerId !== touch.playerId) {
      const key = `${previous.playerId}->${touch.playerId}`;
      const existing = links.get(key);
      links.set(key, {
        from: previous.playerId,
        to: touch.playerId,
        count: (existing?.count ?? 0) + 1,
      });
    }

    previous = touch;
  }

  return {
    nodes: [...nodes.values()],
    links: [...links.values()].filter((link) => nodes.has(link.from) && nodes.has(link.to)),
  };
}
