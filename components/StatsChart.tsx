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
  const data = [
    {
      name: "Passes",
      [homeTeam.name]: homeTeam.stats.passes,
      [awayTeam.name]: awayTeam.stats.passes,
    },
    {
      name: "Shots",
      [homeTeam.name]: homeTeam.stats.shots,
      [awayTeam.name]: awayTeam.stats.shots,
    },
    {
      name: "On Target",
      [homeTeam.name]: homeTeam.stats.shotsOnTarget,
      [awayTeam.name]: awayTeam.stats.shotsOnTarget,
    },
    {
      name: "Tackles",
      [homeTeam.name]: homeTeam.stats.tackles,
      [awayTeam.name]: awayTeam.stats.tackles,
    },
    {
      name: "Fouls",
      [homeTeam.name]: homeTeam.stats.fouls,
      [awayTeam.name]: awayTeam.stats.fouls,
    },
    {
      name: "Corners",
      [homeTeam.name]: homeTeam.stats.corners,
      [awayTeam.name]: awayTeam.stats.corners,
    },
  ];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
        <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "8px",
            color: "#e6edf3",
            fontSize: "12px",
          }}
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
        />
        <Legend
          wrapperStyle={{ fontSize: "12px", color: "#8b949e", paddingTop: "8px" }}
        />
        <Bar dataKey={homeTeam.name} fill="#3b82f6" radius={[3, 3, 0, 0]} maxBarSize={28} />
        <Bar dataKey={awayTeam.name} fill="#ef4444" radius={[3, 3, 0, 0]} maxBarSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}
