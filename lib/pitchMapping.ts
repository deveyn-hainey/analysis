import type { PitchView, Position } from "@/lib/types";

function clamp(n: number, lo = 0, hi = 100) {
  return Math.min(hi, Math.max(lo, n));
}

export function hasPitchCalibration(frame: {
  pitchBall?: Position;
  players: Array<{ pitchPosition?: Position }>;
}) {
  return Boolean(frame.pitchBall || frame.players.some((player) => player.pitchPosition));
}

export function mapImagePositionToPitch(position: Position, view?: PitchView): Position {
  if (!view) return position;
  const x = view.lengthMin + (position.x / 100) * (view.lengthMax - view.lengthMin);
  const fieldTop = clamp(view.topImageY, 0, 90);
  const y = ((position.y - fieldTop) / Math.max(1, 100 - fieldTop)) * 100;
  return { x: clamp(x), y: clamp(y) };
}

export function fieldPosition(
  imagePosition: Position,
  pitchPosition?: Position,
  pitchView?: PitchView
): Position {
  return pitchPosition ?? mapImagePositionToPitch(imagePosition, pitchView);
}
