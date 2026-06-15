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
    <div className="flex flex-col gap-4">
      {insights.map((insight) => {
        const cat = CATEGORY_CONFIG[insight.category] ?? CATEGORY_CONFIG.tactical;
        const pri = PRIORITY_CONFIG[insight.priority] ?? PRIORITY_CONFIG.medium;
        const Icon = cat.icon;

        return (
          <div
            key={insight.id}
            className={`border rounded-xl p-4 ${cat.border} bg-[#0d1a0d]`}
          >
            <div className="flex items-start justify-between gap-3 mb-3">
              <div className="flex items-center gap-2.5">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${cat.bg}`}>
                  <Icon className={`w-4 h-4 ${cat.color}`} />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-[#f0fdf4] leading-tight">{insight.title}</h4>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className={`text-xs capitalize ${cat.color}`}>{insight.category}</span>
                    <span className="text-[#1c3020]">·</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded-full ${pri.bg} ${pri.color}`}>
                      {pri.label}
                    </span>
                  </div>
                </div>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded-full flex-shrink-0 ${
                  insight.affectedTeam === "home"
                    ? "bg-green-400/15 text-green-400"
                    : insight.affectedTeam === "away"
                    ? "bg-gray-400/15 text-gray-300"
                    : "bg-green-400/10 text-green-300"
                }`}
              >
                {insight.affectedTeam === "home"
                  ? homeTeamName
                  : insight.affectedTeam === "away"
                  ? awayTeamName
                  : "Both teams"}
              </span>
            </div>

            <div className="mb-2">
              <div className="text-xs font-medium text-[#6b9e6b] uppercase tracking-wide mb-1">Observation</div>
              <p className="text-sm text-[#d1fae5] leading-relaxed">{insight.observation}</p>
            </div>

            <div className="bg-[#070e07] rounded-lg p-3 border border-[#1c3020]">
              <div className="text-xs font-medium text-green-400 uppercase tracking-wide mb-1">Recommendation</div>
              <p className="text-sm text-[#d1fae5] leading-relaxed">{insight.recommendation}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
