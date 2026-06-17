"use client";

import type { TeamAnalysis } from "@/lib/types";

interface HeatmapProps {
  team: TeamAnalysis;
}

function getHeatColor(value: number): string {
  if (value === 0) return "rgba(20,83,45,0.55)";
  if (value < 0.2) return "rgba(22,101,52,0.68)";
  if (value < 0.4) return "rgba(34,151,76,0.72)";
  if (value < 0.6) return "rgba(49,185,96,0.78)";
  if (value < 0.8) return "rgba(234,204,21,0.82)";
  return "rgba(215,86,58,0.86)";
}

function sampledHeatValue(team: TeamAnalysis, row: number, col: number) {
  const sourceRows = team.heatmap.length;
  const sourceCols = team.heatmap[0]?.length ?? 1;
  const sourceRow = Math.min(sourceRows - 1, Math.floor((row / 10) * sourceRows));
  const sourceCol = Math.min(sourceCols - 1, Math.floor((col / 14) * sourceCols));
  const base = team.heatmap[sourceRow]?.[sourceCol] ?? 0;
  const left = team.heatmap[sourceRow]?.[Math.max(0, sourceCol - 1)] ?? base;
  const right = team.heatmap[sourceRow]?.[Math.min(sourceCols - 1, sourceCol + 1)] ?? base;
  return Math.min(1, base * 0.72 + left * 0.14 + right * 0.14);
}

export default function Heatmap({ team }: HeatmapProps) {
  const rows = 10;
  const cols = 14;

  return (
    <div className="relative">
      <div
        className="relative overflow-hidden rounded-lg bg-[#08110b]"
        style={{ aspectRatio: "14/8.6" }}
      >
        <div
          className="absolute inset-0 grid gap-1.5 p-1"
          style={{
            gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
          }}
        >
          {Array.from({ length: rows }).flatMap((_, r) =>
            Array.from({ length: cols }).map((_, c) => {
              const val = sampledHeatValue(team, r, c);
              return (
                <div
                  key={`${r}-${c}`}
                  className="rounded"
                  style={{ backgroundColor: getHeatColor(val) }}
                  title={`Density: ${(val * 100).toFixed(0)}%`}
                />
              );
            })
          )}
        </div>

        <div className="absolute inset-0 p-5">
          <svg viewBox="0 0 700 454" className="w-full h-full">
            <rect x={8} y={8} width={684} height={438} rx={8} fill="none" stroke="#d1fae5" strokeWidth={1.6} opacity={0.26} />
            <line x1={350} y1={8} x2={350} y2={446} stroke="#d1fae5" strokeWidth={1.4} opacity={0.16} />
            <circle cx={350} cy={227} r={62} fill="none" stroke="#d1fae5" strokeWidth={1.4} opacity={0.18} />
            <circle cx={350} cy={227} r={3} fill="#d1fae5" opacity={0.22} />
            <rect x={8} y={139} width={108} height={176} fill="none" stroke="#d1fae5" strokeWidth={1.4} opacity={0.22} />
            <rect x={8} y={188} width={40} height={78} fill="none" stroke="#d1fae5" strokeWidth={1.4} opacity={0.18} />
            <rect x={584} y={139} width={108} height={176} fill="none" stroke="#d1fae5" strokeWidth={1.4} opacity={0.22} />
            <rect x={652} y={188} width={40} height={78} fill="none" stroke="#d1fae5" strokeWidth={1.4} opacity={0.18} />
          </svg>
        </div>
      </div>

      <div className="mt-5 flex items-center gap-5 text-xs font-mono text-[#6b7c70]">
        <span>low</span>
        <div className="h-2 w-52 rounded-full bg-gradient-to-r from-[#123820] via-[#31d473] to-[#fde047]" />
        <span>high</span>
        <span className="text-orange-300">
          <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full bg-orange-400" />
          pressing hotspot
        </span>
      </div>
    </div>
  );
}
