import type { FrameData, MatchEvent, Position, TeamId } from "@/lib/types";

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
  return +shotLikeEvents(events, team)
    .reduce((sum, event) => sum + estimateShotXg(event), 0)
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
      const current = nodes.get(player.id);
      nodes.set(player.id, {
        id: player.id,
        number: player.number,
        team,
        position: current
          ? {
              x: (current.position.x * current.touches + player.position.x) / (current.touches + 1),
              y: (current.position.y * current.touches + player.position.y) / (current.touches + 1),
            }
          : player.position,
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
