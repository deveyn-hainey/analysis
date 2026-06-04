"use client";

import { useState } from "react";
import type { FrameData, Player } from "@/lib/types";

interface SoccerFieldProps {
  frame: FrameData;
}

const W = 700;
const H = 454;

// Convert 0–100 pct to SVG coords
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

  return (
    <div className="relative w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-xl overflow-hidden"
        style={{ maxHeight: 454 }}
      >
        {/* Field */}
        <rect width={W} height={H} fill="#166534" />
        {/* Pitch stripes */}
        {Array.from({ length: 7 }).map((_, i) => (
          <rect key={i} x={i * 100} y={0} width={100} height={H} fill={i % 2 === 0 ? "#15803d" : "#166534"} />
        ))}

        {/* Boundary */}
        <rect x={8} y={8} width={W - 16} height={H - 16} fill="none" stroke="white" strokeWidth={2} opacity={0.8} />

        {/* Center line */}
        <line x1={W / 2} y1={8} x2={W / 2} y2={H - 8} stroke="white" strokeWidth={1.5} opacity={0.8} />

        {/* Center circle */}
        <circle cx={W / 2} cy={H / 2} r={60} fill="none" stroke="white" strokeWidth={1.5} opacity={0.8} />
        <circle cx={W / 2} cy={H / 2} r={3} fill="white" opacity={0.8} />

        {/* Left penalty area */}
        <rect x={8} y={H / 2 - 85} width={108} height={170} fill="none" stroke="white" strokeWidth={1.5} opacity={0.8} />
        {/* Left goal area */}
        <rect x={8} y={H / 2 - 42} width={38} height={84} fill="none" stroke="white" strokeWidth={1.5} opacity={0.8} />
        {/* Left goal */}
        <rect x={2} y={H / 2 - 28} width={6} height={56} fill="none" stroke="white" strokeWidth={2} opacity={0.8} />
        {/* Left penalty spot */}
        <circle cx={80} cy={H / 2} r={3} fill="white" opacity={0.7} />

        {/* Right penalty area */}
        <rect x={W - 116} y={H / 2 - 85} width={108} height={170} fill="none" stroke="white" strokeWidth={1.5} opacity={0.8} />
        {/* Right goal area */}
        <rect x={W - 46} y={H / 2 - 42} width={38} height={84} fill="none" stroke="white" strokeWidth={1.5} opacity={0.8} />
        {/* Right goal */}
        <rect x={W - 8} y={H / 2 - 28} width={6} height={56} fill="none" stroke="white" strokeWidth={2} opacity={0.8} />
        {/* Right penalty spot */}
        <circle cx={W - 80} cy={H / 2} r={3} fill="white" opacity={0.7} />

        {/* Corner arcs */}
        <path d={`M 8,8 A 16,16 0 0,1 24,8`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.6} />
        <path d={`M 8,${H - 8} A 16,16 0 0,0 24,${H - 8}`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.6} />
        <path d={`M ${W - 8},8 A 16,16 0 0,0 ${W - 24},8`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.6} />
        <path d={`M ${W - 8},${H - 8} A 16,16 0 0,1 ${W - 24},${H - 8}`} fill="none" stroke="white" strokeWidth={1.5} opacity={0.6} />

        {/* Ball */}
        {frame.ballPosition && (
          <g>
            <circle
              cx={px(frame.ballPosition.x)}
              cy={py(frame.ballPosition.y)}
              r={8}
              fill="white"
              stroke="#d97706"
              strokeWidth={2}
            />
            <circle cx={px(frame.ballPosition.x)} cy={py(frame.ballPosition.y)} r={3} fill="#d97706" />
          </g>
        )}

        {/* Players */}
        {frame.players.map((player) => {
          const x = px(player.position.x);
          const y = py(player.position.y);
          const color = player.team === "home" ? "#3b82f6" : "#ef4444";
          const accent = ACTION_COLORS[player.action];
          const isHovered = hovered === player.id;

          return (
            <g
              key={player.id}
              onMouseEnter={() => setHovered(player.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {/* Glow ring on hover */}
              {isHovered && (
                <circle cx={x} cy={y} r={18} fill={color} opacity={0.15} />
              )}
              {/* Action ring */}
              {accent && (
                <circle cx={x} cy={y} r={14} fill="none" stroke={accent} strokeWidth={2} opacity={0.8} />
              )}
              {/* Player dot */}
              <circle cx={x} cy={y} r={11} fill={color} stroke="white" strokeWidth={1.5} />
              {/* Jersey number */}
              <text
                x={x}
                y={y}
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize={8}
                fontWeight="bold"
                fontFamily="system-ui"
              >
                {player.number}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Hover tooltip */}
      {hoveredPlayer && (
        <div className="absolute bottom-4 left-4 bg-[#0d1117]/95 border border-[#30363d] rounded-lg p-3 text-xs pointer-events-none z-10">
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: hoveredPlayer.team === "home" ? "#3b82f6" : "#ef4444" }}
            />
            <span className="font-semibold text-[#e6edf3]">
              #{hoveredPlayer.number} · {ROLE_LABELS[hoveredPlayer.role]}
            </span>
          </div>
          <div className="text-[#8b949e] capitalize">{hoveredPlayer.action}</div>
          <div className="text-[#8b949e]">
            Pos: ({hoveredPlayer.position.x.toFixed(0)}, {hoveredPlayer.position.y.toFixed(0)})
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 right-4 flex items-center gap-4 bg-[#0d1117]/80 border border-[#30363d] rounded-lg px-3 py-2 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-[#8b949e]">Home</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-[#8b949e]">Away</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-white border-2 border-amber-500" />
          <span className="text-[#8b949e]">Ball</span>
        </div>
      </div>
    </div>
  );
}
