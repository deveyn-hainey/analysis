"use client";

import type { TeamAnalysis } from "@/lib/types";

interface TeamComparisonProps {
  homeTeam: TeamAnalysis;
  awayTeam: TeamAnalysis;
}

interface StatRow {
  label: string;
  home: number;
  away: number;
  suffix?: string;
  higherIsBetter?: boolean;
}

function StatBar({ label, home, away, suffix = "", higherIsBetter = true }: StatRow) {
  const total = home + away || 1;
  const homePct = (home / total) * 100;
  const awayPct = (away / total) * 100;
  const homeWins = higherIsBetter ? home >= away : home <= away;

  return (
    <div className="py-2">
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm font-semibold ${homeWins ? "text-blue-400" : "text-[#e6edf3]"}`}>
          {home}{suffix}
        </span>
        <span className="text-xs text-[#8b949e] text-center flex-1 mx-3">{label}</span>
        <span className={`text-sm font-semibold ${!homeWins ? "text-red-400" : "text-[#e6edf3]"}`}>
          {away}{suffix}
        </span>
      </div>
      <div className="flex gap-0.5 h-1.5 rounded-full overflow-hidden bg-[#30363d]">
        <div
          className="h-full rounded-l-full bg-blue-500 transition-all"
          style={{ width: `${homePct}%` }}
        />
        <div
          className="h-full rounded-r-full bg-red-500 transition-all"
          style={{ width: `${awayPct}%` }}
        />
      </div>
    </div>
  );
}

export default function TeamComparison({ homeTeam, awayTeam }: TeamComparisonProps) {
  const rows: StatRow[] = [
    { label: "Possession", home: homeTeam.stats.possession, away: awayTeam.stats.possession, suffix: "%" },
    { label: "Pass Accuracy", home: homeTeam.stats.passAccuracy, away: awayTeam.stats.passAccuracy, suffix: "%" },
    { label: "Passes", home: homeTeam.stats.passes, away: awayTeam.stats.passes },
    { label: "Shots", home: homeTeam.stats.shots, away: awayTeam.stats.shots },
    { label: "Shots on Target", home: homeTeam.stats.shotsOnTarget, away: awayTeam.stats.shotsOnTarget },
    { label: "Tackles", home: homeTeam.stats.tackles, away: awayTeam.stats.tackles },
    { label: "Fouls", home: homeTeam.stats.fouls, away: awayTeam.stats.fouls, higherIsBetter: false },
  ];

  return (
    <div>
      {/* Team headers */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-sm font-semibold text-[#e6edf3]">{homeTeam.name}</span>
          <span className="text-xs bg-[#21262d] px-2 py-0.5 rounded text-[#8b949e]">
            {homeTeam.formation}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs bg-[#21262d] px-2 py-0.5 rounded text-[#8b949e]">
            {awayTeam.formation}
          </span>
          <span className="text-sm font-semibold text-[#e6edf3]">{awayTeam.name}</span>
          <div className="w-3 h-3 rounded-full bg-red-500" />
        </div>
      </div>

      <div className="divide-y divide-[#30363d]">
        {rows.map((row) => (
          <StatBar key={row.label} {...row} />
        ))}
      </div>
    </div>
  );
}
