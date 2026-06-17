"use client";

import { useMemo, useState } from "react";
import type { ElementType } from "react";
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
  selectedEventId?: string;
  selectedTimestamp?: number;
  onSelect?: (event: MatchEvent) => void;
  homeTeamName: string;
  awayTeamName: string;
  groupWindowSeconds?: number;
}

const EVENT_CONFIG: Record<string, { icon: ElementType; color: string; bgColor: string }> = {
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

function formatRange(start: number, end: number) {
  return `${formatTime(start)}-${formatTime(end)}`;
}

function eventKey(event: MatchEvent) {
  return [
    event.type,
    event.team ?? "neutral",
    event.description.toLowerCase().replace(/[^a-z0-9 ]/g, "").slice(0, 60),
  ].join("|");
}

function collapseNearDuplicates(events: MatchEvent[]) {
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp || b.confidence - a.confidence);
  const kept: MatchEvent[] = [];
  const lastByKey = new Map<string, MatchEvent>();

  for (const event of sorted) {
    const key = eventKey(event);
    const previous = lastByKey.get(key);
    if (previous && Math.abs(event.timestamp - previous.timestamp) <= 12) {
      if (event.confidence > previous.confidence) {
        const idx = kept.findIndex((candidate) => candidate.id === previous.id);
        if (idx >= 0) kept[idx] = event;
        lastByKey.set(key, event);
      }
      continue;
    }

    kept.push(event);
    lastByKey.set(key, event);
  }

  return kept.sort((a, b) => a.timestamp - b.timestamp);
}

function defaultWindow(events: MatchEvent[]) {
  const last = events.at(-1)?.timestamp ?? 0;
  if (last >= 3600) return 600;
  if (last >= 900) return 300;
  return 60;
}

export default function EventTimeline({
  events,
  selectedEventId,
  selectedTimestamp,
  onSelect,
  homeTeamName,
  awayTeamName,
  groupWindowSeconds,
}: EventTimelineProps) {
  const sorted = useMemo(() => collapseNearDuplicates(events), [events]);
  const windowSize = groupWindowSeconds ?? defaultWindow(sorted);
  const [openGroups, setOpenGroups] = useState<Set<number>>(() => new Set([0]));

  const groups = useMemo(() => {
    const grouped = new Map<number, MatchEvent[]>();
    for (const event of sorted) {
      const key = Math.floor(event.timestamp / windowSize);
      grouped.set(key, [...(grouped.get(key) ?? []), event]);
    }
    return [...grouped.entries()].map(([key, items]) => ({
      key,
      start: key * windowSize,
      end: (key + 1) * windowSize - 1,
      events: items,
      keyMoments: items.filter((event) => event.isKeyMoment || event.type === "goal").length,
    }));
  }, [sorted, windowSize]);

  const selectedGroupKey =
    selectedEventId
      ? groups.find((group) => group.events.some((event) => event.id === selectedEventId))?.key
      : selectedTimestamp !== undefined
      ? Math.floor(selectedTimestamp / windowSize)
      : undefined;

  if (sorted.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-[#6b9e6b] text-sm">
        No events detected
      </div>
    );
  }

  return (
    <div className="relative h-full max-h-[760px] overflow-y-auto pr-1">
      {groups.map((group) => {
        const isOpen = openGroups.has(group.key) || group.key === selectedGroupKey;
        return (
          <section key={group.key} className="border-b border-[#132018] last:border-b-0 py-2">
            <button
              onClick={() => {
                const next = new Set(openGroups);
                if (next.has(group.key)) next.delete(group.key);
                else next.add(group.key);
                setOpenGroups(next);
              }}
              className="mb-1 flex w-full items-center justify-between rounded-lg px-2 py-2 text-left hover:bg-[#101a12]"
            >
              <div>
                <div className="text-xs font-mono uppercase tracking-[0.18em] text-[#617169]">
                  {formatRange(group.start, group.end)}
                </div>
                <div className="mt-1 text-sm font-bold text-[#c8d2ca]">
                  {group.events.length} event{group.events.length === 1 ? "" : "s"}
                  {group.keyMoments > 0 && <span className="ml-2 text-yellow-300">{group.keyMoments} key</span>}
                </div>
              </div>
              <span className="rounded-md border border-[#1c3020] px-2 py-1 text-xs font-mono text-[#829086]">
                {isOpen ? "hide" : "show"}
              </span>
            </button>

            {isOpen && group.events.map((event, index) => {
              const cfg = EVENT_CONFIG[event.type] ?? EVENT_CONFIG.pass;
              const isSelected = selectedEventId === event.id;

              const teamLabel =
                event.team === "home"
                  ? homeTeamName
                  : event.team === "away"
                  ? awayTeamName
                  : null;

              return (
                <button
                  key={event.id}
                  onClick={() => onSelect?.(event)}
                  className={`relative grid w-full grid-cols-[18px_58px_minmax(0,1fr)_92px] items-start gap-3 rounded-lg px-2 py-3 text-left transition-colors ${
                    isSelected
                      ? "bg-green-400/12 ring-1 ring-green-400/30"
                      : "hover:bg-[#101a12]"
                  }`}
                >
                  <div className="relative flex justify-center">
                    <span className={`mt-1 h-3 w-3 rounded-full ${event.type === "goal" ? "bg-yellow-300" : event.team === "home" ? "bg-green-400" : "bg-[#7c8a82]"}`} />
                    {index < group.events.length - 1 && <span className="absolute top-5 bottom-[-18px] w-px bg-[#1c3020]" />}
                  </div>

                  <div className="pt-0.5 text-sm font-black font-mono text-[#c8d2ca]">
                    {formatTime(event.timestamp)}
                  </div>

                  <div className="min-w-0">
                    <p className="text-sm text-[#c8d2ca] leading-relaxed">{event.description}</p>
                    {teamLabel && <p className="mt-1 text-xs text-[#6f8175]">{teamLabel}</p>}
                    {event.semanticLabel && (
                      <p className="text-xs text-green-400/80 mt-1">{event.semanticLabel.replaceAll("_", " ")}</p>
                    )}
                    <p className="text-xs font-mono text-[#617169] mt-1">
                      vision conf · {event.confidence.toFixed(2)}
                    </p>
                    {((event.conflicts?.length ?? 0) > 0 || event.pipelineFlag) && (
                      <p className="text-xs text-amber-300/90 mt-1">
                        Review: {event.conflicts?.slice(0, 2).join("; ") || event.pipelineFlag?.replaceAll("_", " ")}
                      </p>
                    )}
                  </div>

                  <span className={`justify-self-end whitespace-nowrap rounded-md px-2 py-1 text-center text-[10px] font-mono uppercase leading-none ${cfg.bgColor} ${cfg.color}`}>
                    {event.type.replace("-", " ")}
                  </span>
                </button>
              );
            })}
          </section>
        );
      })}
    </div>
  );
}
