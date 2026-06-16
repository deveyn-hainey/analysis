"use client";

import type { Player, Position } from "@/lib/types";

interface Props {
  base64: string;
  players: Player[];
  ballPosition?: Position;
  referees?: Position[];
}

const DOT_COLOR: Record<string, string> = {
  home: "#4ade80",
  away: "#ef4444",
};

export default function FrameOverlay({ base64, players, ballPosition, referees }: Props) {
  const homeCount = players.filter((p) => p.team === "home").length;
  const awayCount = players.filter((p) => p.team === "away").length;

  return (
    <div>
      <div className="relative rounded-lg overflow-hidden bg-black">
        <img
          src={`data:image/jpeg;base64,${base64}`}
          alt="Extracted match frame"
          className="w-full block"
        />
        {/* SVG overlay — viewBox matches the 1280×720 extracted frame dimensions */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 1280 720"
          preserveAspectRatio="none"
        >
          {players.map((p) => {
            const cx = (p.position.x / 100) * 1280;
            const cy = (p.position.y / 100) * 720;
            const fill = DOT_COLOR[p.team] ?? "#ffffff";
            return (
              <g key={p.id}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={10}
                  fill={fill}
                  fillOpacity={0.82}
                  stroke="#000"
                  strokeWidth={1.5}
                />
                {p.number > 0 && (
                  <text
                    x={cx}
                    y={cy + 3.5}
                    textAnchor="middle"
                    fontSize={8}
                    fontWeight="700"
                    fill="#000"
                    fontFamily="monospace"
                  >
                    {p.number}
                  </text>
                )}
              </g>
            );
          })}
          {referees?.map((pos, i) => (
            <circle
              key={`ref-${i}`}
              cx={(pos.x / 100) * 1280}
              cy={(pos.y / 100) * 720}
              r={8}
              fill="#1a1a1a"
              fillOpacity={0.85}
              stroke="#fbbf24"
              strokeWidth={2}
            />
          ))}
          {ballPosition && (
            <circle
              cx={(ballPosition.x / 100) * 1280}
              cy={(ballPosition.y / 100) * 720}
              r={6}
              fill="#fbbf24"
              stroke="#000"
              strokeWidth={1.5}
            />
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 text-xs text-[#6b9e6b]">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-green-400 inline-block" />
          Home ({homeCount})
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
          Away ({awayCount})
        </span>
        {referees && referees.length > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-black border border-yellow-400 inline-block" />
            Referee ({referees.length})
          </span>
        )}
        {ballPosition && (
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block" />
            Ball
          </span>
        )}
        <span className="ml-auto text-[#3d5c40] italic">AI-detected positions (approximate)</span>
      </div>
    </div>
  );
}
