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
    <div className="flex flex-col gap-1 overflow-y-auto max-h-[520px] pr-1">
      {sorted.map((event) => {
        const cfg = EVENT_CONFIG[event.type] ?? EVENT_CONFIG.pass;
        const Icon = cfg.icon;
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
            className={`w-full text-left flex items-start gap-3 p-3 rounded-lg transition-colors ${
              isSelected
                ? "bg-green-400/10 border border-green-400/30"
                : "hover:bg-[#142014] border border-transparent"
            }`}
          >
            <div className="text-xs font-mono text-[#6b9e6b] w-10 pt-0.5 flex-shrink-0">
              {formatTime(event.timestamp)}
            </div>

            <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${cfg.bgColor}`}>
              <Icon className={`w-3.5 h-3.5 ${cfg.color}`} />
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className={`text-xs font-semibold uppercase tracking-wide ${cfg.color}`}>
                  {event.type.replace("-", " ")}
                </span>
                {teamLabel && (
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded-full ${
                      event.team === "home"
                        ? "bg-green-400/15 text-green-400"
                        : "bg-gray-400/15 text-gray-300"
                    }`}
                  >
                    {teamLabel}
                  </span>
                )}
              </div>
              <p className="text-xs text-[#6b9e6b] leading-relaxed">{event.description}</p>
              {event.semanticLabel && (
                <p className="text-xs text-green-400/80 mt-1">{event.semanticLabel.replaceAll("_", " ")}</p>
              )}
              {event.evidenceUsed && event.evidenceUsed.length > 0 && (
                <p className="text-xs text-[#6b9e6b]/70 mt-1">
                  Evidence: {event.evidenceUsed.slice(0, 2).join("; ")}
                </p>
              )}
              {((event.conflicts?.length ?? 0) > 0 || event.pipelineFlag) && (
                <p className="text-xs text-amber-300/90 mt-1">
                  Review: {event.conflicts?.slice(0, 2).join("; ") || event.pipelineFlag?.replaceAll("_", " ")}
                </p>
              )}
            </div>

            <div className="flex-shrink-0 text-right">
              <div className="text-xs text-[#6b9e6b]">{Math.round(event.confidence * 100)}%</div>
              {event.isKeyMoment && (
                <div className="text-xs text-green-400 mt-0.5">Key</div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
