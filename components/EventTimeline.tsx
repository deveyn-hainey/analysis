"use client";

import type { MatchEvent } from "@/lib/types";
import {
  Flag,
  Target,
  Shield,
  AlertCircle,
  CornerDownRight,
  Wind,
  Trophy,
  CheckCircle,
  ArrowUpRight,
} from "lucide-react";

interface EventTimelineProps {
  events: MatchEvent[];
  selectedTimestamp?: number;
  onSelect?: (timestamp: number) => void;
}

const EVENT_CONFIG: Record<
  string,
  { icon: React.ElementType; color: string; bgColor: string }
> = {
  goal: { icon: Trophy, color: "text-yellow-400", bgColor: "bg-yellow-400/10" },
  shot: { icon: Target, color: "text-orange-400", bgColor: "bg-orange-400/10" },
  save: { icon: CheckCircle, color: "text-blue-400", bgColor: "bg-blue-400/10" },
  tackle: { icon: Shield, color: "text-violet-400", bgColor: "bg-violet-400/10" },
  foul: { icon: AlertCircle, color: "text-red-400", bgColor: "bg-red-400/10" },
  corner: { icon: CornerDownRight, color: "text-cyan-400", bgColor: "bg-cyan-400/10" },
  freekick: { icon: Wind, color: "text-emerald-400", bgColor: "bg-emerald-400/10" },
  pass: { icon: ArrowUpRight, color: "text-slate-400", bgColor: "bg-slate-400/10" },
  offside: { icon: Flag, color: "text-amber-400", bgColor: "bg-amber-400/10" },
  dribble: { icon: ArrowUpRight, color: "text-purple-400", bgColor: "bg-purple-400/10" },
  "throw-in": { icon: Flag, color: "text-slate-400", bgColor: "bg-slate-400/10" },
};

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function EventTimeline({ events, selectedTimestamp, onSelect }: EventTimelineProps) {
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);

  if (sorted.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-[#8b949e] text-sm">
        No events detected
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 overflow-y-auto max-h-[400px] pr-1">
      {sorted.map((event) => {
        const cfg = EVENT_CONFIG[event.type] ?? EVENT_CONFIG.pass;
        const Icon = cfg.icon;
        const isSelected = selectedTimestamp !== undefined && Math.abs(event.timestamp - selectedTimestamp) < 10;

        return (
          <button
            key={event.id}
            onClick={() => onSelect?.(event.timestamp)}
            className={`w-full text-left flex items-start gap-3 p-3 rounded-lg transition-colors ${
              isSelected
                ? "bg-emerald-500/10 border border-emerald-500/30"
                : "hover:bg-[#21262d] border border-transparent"
            }`}
          >
            {/* Time */}
            <div className="text-xs font-mono text-[#8b949e] w-10 pt-0.5 flex-shrink-0">
              {formatTime(event.timestamp)}
            </div>

            {/* Icon */}
            <div
              className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${cfg.bgColor}`}
            >
              <Icon className={`w-3.5 h-3.5 ${cfg.color}`} />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className={`text-xs font-semibold uppercase tracking-wide ${cfg.color}`}>
                  {event.type.replace("-", " ")}
                </span>
                {event.team && (
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded-full ${
                      event.team === "home"
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {event.team === "home" ? "Eagles FC" : "City United"}
                  </span>
                )}
              </div>
              <p className="text-xs text-[#8b949e] leading-relaxed">{event.description}</p>
            </div>

            {/* Confidence */}
            <div className="flex-shrink-0 text-right">
              <div className="text-xs text-[#8b949e]">{Math.round(event.confidence * 100)}%</div>
              {event.isKeyMoment && (
                <div className="text-xs text-emerald-400 mt-0.5">Key</div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
