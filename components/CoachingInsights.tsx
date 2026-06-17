"use client";

import type { CoachingInsight } from "@/lib/types";
import { TrendingUp, Shield, Activity, Compass, Zap } from "lucide-react";

interface CoachingInsightsProps {
  insights: CoachingInsight[];
  homeTeamName: string;
  awayTeamName: string;
}

const CATEGORY_CONFIG = {
  attacking: { icon: TrendingUp, color: "text-orange-400", bg: "bg-orange-400/10", border: "border-orange-400/20" },
  defensive: { icon: Shield, color: "text-blue-400", bg: "bg-blue-400/10", border: "border-blue-400/20" },
  possession: { icon: Activity, color: "text-cyan-400", bg: "bg-cyan-400/10", border: "border-cyan-400/20" },
  tactical: { icon: Compass, color: "text-violet-400", bg: "bg-violet-400/10", border: "border-violet-400/20" },
  physical: { icon: Zap, color: "text-yellow-400", bg: "bg-yellow-400/10", border: "border-yellow-400/20" },
};

const PRIORITY_CONFIG = {
  critical: { label: "Critical", color: "text-red-400", bg: "bg-red-400/10" },
  high: { label: "High", color: "text-orange-400", bg: "bg-orange-400/10" },
  medium: { label: "Medium", color: "text-yellow-400", bg: "bg-yellow-400/10" },
  low: { label: "Low", color: "text-[#6b9e6b]", bg: "bg-[#142014]" },
};

export default function CoachingInsights({ insights, homeTeamName, awayTeamName }: CoachingInsightsProps) {
  if (!insights.length) {
    return (
      <div className="text-center text-[#6b9e6b] text-sm py-8">
        No insights generated yet.
      </div>
    );
  }

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      {insights.map((insight) => {
        const cat = CATEGORY_CONFIG[insight.category] ?? CATEGORY_CONFIG.tactical;
        const pri = PRIORITY_CONFIG[insight.priority] ?? PRIORITY_CONFIG.medium;
        const Icon = cat.icon;

        return (
          <div
            key={insight.id}
            className={`border rounded-lg p-4 ${cat.border} bg-[#07100a]`}
          >
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="flex items-start gap-3">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${cat.bg}`}>
                  <Icon className={`w-3.5 h-3.5 ${cat.color}`} />
                </div>
                <div>
                  <h4 className="text-base font-black text-[#f0fdf4] leading-tight">{insight.title}</h4>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className={`text-xs capitalize ${cat.color}`}>{insight.category}</span>
                    <span className="text-[#1c3020]">·</span>
                    <span className={`text-xs px-2 py-0.5 rounded-md ${pri.bg} ${pri.color}`}>
                      {pri.label}
                    </span>
                  </div>
                </div>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded-md border border-[#1c3020] bg-[#101812] flex-shrink-0 ${
                  insight.affectedTeam === "home"
                    ? "text-green-400"
                    : insight.affectedTeam === "away"
                    ? "text-gray-300"
                    : "text-green-300"
                }`}
              >
                {insight.affectedTeam === "home"
                  ? homeTeamName
                  : insight.affectedTeam === "away"
                  ? awayTeamName
                  : "Both teams"}
              </span>
            </div>

            <div className="mb-4">
              <div className="text-[10px] font-mono text-[#617169] uppercase tracking-[0.22em] mb-2">Observation</div>
              <p className="text-sm text-[#c8d2ca] leading-6">{insight.observation}</p>
            </div>

            <div className="bg-green-400/10 rounded-lg p-3 border border-green-400/20">
              <div className="text-[10px] font-mono text-green-400 uppercase tracking-[0.22em] mb-2">Recommendation</div>
              <p className="text-sm text-[#d1fae5] leading-6">{insight.recommendation}</p>
            </div>

            <div className="mt-4 grid grid-cols-[116px_1fr_38px] items-center gap-3">
              <span className="text-[10px] font-mono text-[#617169]">model confidence</span>
              <div className="h-1.5 rounded-full bg-[#142014] overflow-hidden">
                <div
                  className="h-full rounded-full bg-green-400"
                  style={{ width: `${insight.priority === "critical" ? 93 : insight.priority === "high" ? 86 : insight.priority === "medium" ? 78 : 68}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-green-300 text-right">
                {insight.priority === "critical" ? "0.93" : insight.priority === "high" ? "0.86" : insight.priority === "medium" ? "0.78" : "0.68"}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
