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
  homeTeamName: string;
  awayTeamName: string;
}

const EVENT_CONFIG: Record<string, { icon: React.ElementType; color: string; bgColor: string }> = {
  goal: { icon: Trophy, color: "text-yellow-400", bgColor: "bg-yellow-400/10" },
  shot: { icon: Target, color: "text-orange-400", bgColor: "bg-orange-400/10" },
  save: { icon: CheckCircle, color: "text-green-400", bgColor: "bg-green-400/10" },
  tackle: { icon: Shield, color: "text-violet-400", bgColor: "bg-violet-400/10" },
  foul: { icon: AlertCircle, color: "text-red-400", bgColor: "bg-red-400/10" },
  corner: { icon: CornerDownRight, color: "text-cyan-400", bgColor: "bg-cyan-400/10" },
  "goal-kick": { icon: Flag, color: "text-cyan-300", bgColor: "bg-cyan-300/10" },
  freekick: { icon: Wind, color: "text-green-400", bgColor: "bg-green-400/10" },
  card_yellow: { icon: AlertCircle, color: "text-yellow-400", bgColor: "bg-yellow-400/10" },
  card_red: { icon: AlertCircle, color: "text-red-500", bgColor: "bg-red-500/10" },
  card_unknown: { icon: AlertCircle, color: "text-amber-300", bgColor: "bg-amber-300/10" },
  pass: { icon: ArrowUpRight, color: "text-[#6b9e6b]", bgColor: "bg-[#142014]" },
  offside: { icon: Flag, color: "text-amber-400", bgColor: "bg-amber-400/10" },
  dribble: { icon: ArrowUpRight, color: "text-purple-400", bgColor: "bg-purple-400/10" },
  "throw-in": { icon: Flag, color: "text-[#6b9e6b]", bgColor: "bg-[#142014]" },
};

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function EventTimeline({
  events,
  selectedTimestamp,
  onSelect,
  homeTeamName,
  awayTeamName,
}: EventTimelineProps) {
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);

  if (sorted.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-[#6b9e6b] text-sm">
        No events detected
      </div>
    );
  }

  return (
    <div className="relative overflow-y-auto max-h-[640px] pr-1">
      {sorted.map((event) => {
        const cfg = EVENT_CONFIG[event.type] ?? EVENT_CONFIG.pass;
        const isSelected =
          selectedTimestamp !== undefined &&
          Math.abs(event.timestamp - selectedTimestamp) < 10;

        const teamLabel =
          event.team === "home"
            ? homeTeamName
            : event.team === "away"
            ? awayTeamName
            : null;

        return (
          <button
            key={event.id}
            onClick={() => onSelect?.(event.timestamp)}
            className={`relative w-full text-left grid grid-cols-[18px_58px_1fr_auto] gap-3 px-1 py-3 transition-colors ${
              isSelected
                ? "rounded-lg bg-green-400/10"
                : "hover:bg-[#101a12]"
            }`}
          >
            <div className="relative flex justify-center">
              <span className={`mt-1 h-3 w-3 rounded-full ${event.type === "goal" ? "bg-yellow-300" : event.team === "home" ? "bg-green-400" : "bg-[#7c8a82]"}`} />
              <span className="absolute top-5 bottom-[-18px] w-px bg-[#1c3020]" />
            </div>

            <div className="pt-0.5 text-sm font-black font-mono text-[#c8d2ca]">
              {formatTime(event.timestamp)}
            </div>

            <div className="flex-1 min-w-0">
              <p className="text-sm text-[#c8d2ca] leading-relaxed">{event.description}</p>
              {teamLabel && <p className="mt-1 text-xs text-[#6f8175]">{teamLabel}</p>}
              {event.semanticLabel && (
                <p className="text-xs text-green-400/80 mt-1">{event.semanticLabel.replaceAll("_", " ")}</p>
              )}
              {event.evidenceUsed && event.evidenceUsed.length > 0 && (
                <p className="text-xs font-mono text-[#617169] mt-1">
                  vision conf · {event.confidence.toFixed(2)}
                </p>
              )}
              {((event.conflicts?.length ?? 0) > 0 || event.pipelineFlag) && (
                <p className="text-xs text-amber-300/90 mt-1">
                  Review: {event.conflicts?.slice(0, 2).join("; ") || event.pipelineFlag?.replaceAll("_", " ")}
                </p>
              )}
            </div>

            <span className={`mt-0.5 rounded-md px-2 py-1 text-[10px] font-mono uppercase ${cfg.bgColor} ${cfg.color}`}>
              {event.type.replace("-", " ")}
            </span>
          </button>
        );
      })}
    </div>
  );
}
