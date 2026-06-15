"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TeamAnalysis } from "@/lib/types";

interface StatsChartProps {
  homeTeam: TeamAnalysis;
  awayTeam: TeamAnalysis;
}

export default function StatsChart({ homeTeam, awayTeam }: StatsChartProps) {
  const total =
    homeTeam.stats.passes + homeTeam.stats.shots + homeTeam.stats.tackles +
    awayTeam.stats.passes + awayTeam.stats.shots + awayTeam.stats.tackles;

  if (total === 0) {
    return (
      <div className="flex items-center justify-center h-[260px] text-[#6b9e6b] text-sm">
        No action events detected. Try uploading a longer clip with visible ball activity.
      </div>
    );
  }

  const data = [
    { name: "Passes", [homeTeam.name]: homeTeam.stats.passes, [awayTeam.name]: awayTeam.stats.passes },
    { name: "Shots", [homeTeam.name]: homeTeam.stats.shots, [awayTeam.name]: awayTeam.stats.shots },
    { name: "On Target", [homeTeam.name]: homeTeam.stats.shotsOnTarget, [awayTeam.name]: awayTeam.stats.shotsOnTarget },
    { name: "Tackles", [homeTeam.name]: homeTeam.stats.tackles, [awayTeam.name]: awayTeam.stats.tackles },
    { name: "Fouls", [homeTeam.name]: homeTeam.stats.fouls, [awayTeam.name]: awayTeam.stats.fouls },
    { name: "Corners", [homeTeam.name]: homeTeam.stats.corners, [awayTeam.name]: awayTeam.stats.corners },
  ];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1c3020" />
        <XAxis dataKey="name" tick={{ fill: "#6b9e6b", fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: "#6b9e6b", fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0d1a0d",
            border: "1px solid #1c3020",
            borderRadius: "8px",
            color: "#f0fdf4",
            fontSize: "12px",
          }}
          cursor={{ fill: "rgba(74,222,128,0.05)" }}
        />
        <Legend wrapperStyle={{ fontSize: "12px", color: "#6b9e6b", paddingTop: "8px" }} />
        <Bar dataKey={homeTeam.name} fill="#4ade80" radius={[3, 3, 0, 0]} maxBarSize={28} />
        <Bar dataKey={awayTeam.name} fill="#6b7280" radius={[3, 3, 0, 0]} maxBarSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}
