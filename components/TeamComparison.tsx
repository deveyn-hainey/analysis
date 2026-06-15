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
  const homeWins = higherIsBetter ? home >= away : home <= away;

  return (
    <div className="py-2.5">
      <div className="flex justify-between items-center mb-2">
        <span className={`text-sm font-bold tabular-nums ${homeWins ? "text-green-400" : "text-[#f0fdf4]"}`}>
          {home}{suffix}
        </span>
        <span className="text-xs text-[#6b9e6b] text-center flex-1 mx-3">{label}</span>
        <span className={`text-sm font-bold tabular-nums ${!homeWins ? "text-green-400" : "text-[#f0fdf4]"}`}>
          {away}{suffix}
        </span>
      </div>
      <div className="flex h-1.5 rounded-full overflow-hidden bg-[#1c3020]">
        <div
          className={`h-full rounded-l-full transition-all ${homeWins ? "bg-green-400" : "bg-[#2d4a30]"}`}
          style={{ width: `${homePct}%` }}
        />
        <div
          className={`h-full rounded-r-full transition-all ${!homeWins ? "bg-green-400" : "bg-[#2d4a30]"}`}
          style={{ width: `${100 - homePct}%` }}
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
    { label: "Distance (m)", home: homeTeam.stats.distanceCovered, away: awayTeam.stats.distanceCovered },
  ];

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
          <span className="text-sm font-semibold text-[#f0fdf4]">{homeTeam.name}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-[#f0fdf4]">{awayTeam.name}</span>
          <div className="w-2.5 h-2.5 rounded-full bg-gray-400" />
        </div>
      </div>

      <div className="divide-y divide-[#1c3020]">
        {rows.map((row) => (
          <StatBar key={row.label} {...row} />
        ))}
      </div>
    </div>
  );
}
