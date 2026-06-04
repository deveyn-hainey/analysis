"use client";

import type { TeamAnalysis } from "@/lib/types";

interface HeatmapProps {
  team: TeamAnalysis;
}

function getHeatColor(value: number): string {
  if (value === 0) return "rgba(255,255,255,0.02)";
  if (value < 0.2) return "rgba(59,130,246,0.15)";
  if (value < 0.4) return "rgba(59,130,246,0.30)";
  if (value < 0.6) return "rgba(16,185,129,0.45)";
  if (value < 0.8) return "rgba(245,158,11,0.55)";
  return "rgba(239,68,68,0.70)";
}

export default function Heatmap({ team }: HeatmapProps) {
  return (
    <div className="relative">
      {/* Field background */}
      <div
        className="rounded-lg overflow-hidden border border-[#30363d]"
        style={{ aspectRatio: "700/454", background: "#166534" }}
      >
        {/* Heatmap grid */}
        <div className="w-full h-full grid grid-cols-10 grid-rows-10">
          {team.heatmap.flatMap((row, r) =>
            row.map((val, c) => (
              <div
                key={`${r}-${c}`}
                style={{ backgroundColor: getHeatColor(val) }}
                title={`Density: ${(val * 100).toFixed(0)}%`}
              />
            ))
          )}
        </div>

        {/* Field lines overlay */}
        <div className="absolute inset-0">
          <svg viewBox="0 0 700 454" className="w-full h-full">
            <rect x={8} y={8} width={684} height={438} fill="none" stroke="white" strokeWidth={1} opacity={0.3} />
            <line x1={350} y1={8} x2={350} y2={446} stroke="white" strokeWidth={1} opacity={0.3} />
            <circle cx={350} cy={227} r={60} fill="none" stroke="white" strokeWidth={1} opacity={0.3} />
            <rect x={8} y={142} width={108} height={170} fill="none" stroke="white" strokeWidth={1} opacity={0.3} />
            <rect x={584} y={142} width={108} height={170} fill="none" stroke="white" strokeWidth={1} opacity={0.3} />
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-2 justify-center">
        <span className="text-xs text-[#8b949e]">Low</span>
        <div className="flex gap-0.5">
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => (
            <div
              key={v}
              className="w-5 h-3 rounded-sm"
              style={{ backgroundColor: getHeatColor(v) === "rgba(255,255,255,0.02)" ? "#30363d" : getHeatColor(v) }}
            />
          ))}
        </div>
        <span className="text-xs text-[#8b949e]">High</span>
      </div>
    </div>
  );
}
