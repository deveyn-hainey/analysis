"use client";

import { useState } from "react";
import type { FrameData, Player } from "@/lib/types";
import { fieldPosition, hasPitchCalibration, mapImagePositionToPitch } from "@/lib/pitchMapping";

interface SoccerFieldProps {
  frame: FrameData;
}

const W = 700;
const H = 454;

const px = (x: number) => (x / 100) * W;
const py = (y: number) => (y / 100) * H;

const ROLE_LABELS: Record<Player["role"], string> = {
  gk: "GK",
  def: "DEF",
  mid: "MID",
  fwd: "FWD",
};

const ACTION_COLORS: Partial<Record<Player["action"], string>> = {
  shooting: "#f59e0b",
  tackling: "#8b5cf6",
  passing: "#06b6d4",
};

export default function SoccerField({ frame }: SoccerFieldProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  const hoveredPlayer = frame.players.find((p) => p.id === hovered);
  const calibrated = hasPitchCalibration(frame);
  const ball = frame.pitchBall ?? (frame.ballPosition ? mapImagePositionToPitch(frame.ballPosition, frame.pitchView) : null);
  const referees = frame.pitchReferees ?? frame.referees?.map((pos) => mapImagePositionToPitch(pos, frame.pitchView));

  return (
    <div className="relative w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-xl overflow-hidden"
        style={{ maxHeight: 454 }}
      >
        {/* Pitch stripes */}
        {Array.from({ length: 7 }).map((_, i) => (
          <rect key={i} x={i * 100} y={0} width={100} height={H} fill={i % 2 === 0 ? "#14532d" : "#166534"} />
        ))}

        {/* Boundary */}
        <rect x={8} y={8} width={W - 16} height={H - 16} fill="none" stroke="white" strokeWidth={2} opacity={0.7} />

        {/* Center line */}
        <line x1={W / 2} y1={8} x2={W / 2} y2={H - 8} stroke="white" strokeWidth={1.5} opacity={0.7} />

        {/* Center circle */}
        <circle cx={W / 2} cy={H / 2} r={60} fill="none" stroke="white" strokeWidth={1.5} opacity={0.7} />
        <circle cx={W / 2} cy={H / 2} r={3} fill="white" opacity={0.7} />

        {/* Left penalty area */}
        <rect x={8} y={H / 2 - 85} width={108} height={170} fill="none" stroke="white" strokeWidth={1.5} opacity={0.7} />
        <rect x={8} y={H / 2 - 42} width={38} height={84} fill="none" stroke="white" strokeWidth={1.5} opacity={0.7} />
        <rect x={2} y={H / 2 - 28} width={6} height={56} fill="none" stroke="white" strokeWidth={2} opacity={0.7} />
        <circle cx={80} cy={H / 2} r={3} fill="white" opacity={0.6} />

        {/* Right penalty area */}
        <rect x={W - 116} y={H / 2 - 85} width={108} height={170} fill="none" stroke="white" strokeWidth={1.5} opacity={0.7} />
        <rect x={W - 46} y={H / 2 - 42} width={38} height={84} fill="none" stroke="white" strokeWidth={1.5} opacity={0.7} />
        <rect x={W - 8} y={H / 2 - 28} width={6} height={56} fill="none" stroke="white" strokeWidth={2} opacity={0.7} />
        <circle cx={W - 80} cy={H / 2} r={3} fill="white" opacity={0.6} />

        {/* Corner arcs */}
        <path d={`M 8,8 A 16,16 0 0,1 24,8`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.5} />
        <path d={`M 8,${H - 8} A 16,16 0 0,0 24,${H - 8}`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.5} />
        <path d={`M ${W - 8},8 A 16,16 0 0,0 ${W - 24},8`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.5} />
        <path d={`M ${W - 8},${H - 8} A 16,16 0 0,1 ${W - 24},${H - 8}`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.5} />

        {/* Referees */}
        {referees?.map((pos, i) => {
          return (
          <circle
            key={`ref-${i}`}
            cx={px(pos.x)}
            cy={py(pos.y)}
            r={6}
            fill="#1a1a1a"
            stroke="#fbbf24"
            strokeWidth={1.5}
          />
          );
        })}

        {/* Ball */}
        {ball && (
          <g>
            <circle cx={px(ball.x)} cy={py(ball.y)} r={5.5} fill="white" stroke="#d97706" strokeWidth={1.75} />
            <circle cx={px(ball.x)} cy={py(ball.y)} r={2.25} fill="#d97706" />
          </g>
        )}

        {/* Players */}
        {frame.players.map((player) => {
          const pos = fieldPosition(player.position, player.pitchPosition, frame.pitchView);
          const x = px(pos.x);
          const y = py(pos.y);
          const color = player.team === "home" ? "#ef4444" : "#3b82f6";
          const accent = ACTION_COLORS[player.action];
          const isHovered = hovered === player.id;

          return (
            <g
              key={player.id}
              onMouseEnter={() => setHovered(player.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {isHovered && <circle cx={x} cy={y} r={12} fill={color} opacity={0.15} />}
              {accent && <circle cx={x} cy={y} r={9} fill="none" stroke={accent} strokeWidth={1.5} opacity={0.8} />}
              <circle cx={x} cy={y} r={6.5} fill={color} stroke="white" strokeWidth={1.25} />
            </g>
          );
        })}
      </svg>

      {/* Hover tooltip */}
      {hoveredPlayer && (
        <div className="absolute bottom-4 left-4 bg-[#070e07]/95 border border-[#1c3020] rounded-lg p-3 text-xs pointer-events-none z-10">
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: hoveredPlayer.team === "home" ? "#ef4444" : "#3b82f6" }}
            />
            <span className="font-semibold text-[#f0fdf4]">
              #{hoveredPlayer.number} · {ROLE_LABELS[hoveredPlayer.role]}
            </span>
          </div>
          <div className="text-[#6b9e6b] capitalize">{hoveredPlayer.action}</div>
          <div className="text-[#6b9e6b]">
            Pos: ({hoveredPlayer.position.x.toFixed(0)}, {hoveredPlayer.position.y.toFixed(0)})
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 flex items-center gap-4 bg-[#070e07]/80 border border-[#1c3020] rounded-lg px-3 py-2 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-[#6b9e6b]">Home</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-[#6b9e6b]">Away</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-white border-2 border-amber-500" />
          <span className="text-[#6b9e6b]">Ball</span>
        </div>
        {frame.referees && frame.referees.length > 0 && (
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-black border-2 border-yellow-400" />
            <span className="text-[#6b9e6b]">Ref</span>
          </div>
        )}
        {!calibrated && frame.pitchView && (
          <div className="hidden sm:flex items-center gap-1.5 text-[#829086]">
            <span>View map</span>
          </div>
        )}
      </div>
    </div>
  );
}
