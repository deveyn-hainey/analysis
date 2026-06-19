"use client";

import { useEffect, useState, useRef, useMemo, Suspense } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  Film,
  ChevronLeft,
  ChevronDown,
  Download,
  Cpu,
  AlertTriangle,
  Check,
} from "lucide-react";
import type { MatchAnalysis, FrameData, MatchEvent, Player, PitchView, Position } from "@/lib/types";
import SoccerField from "@/components/SoccerField";
import FrameOverlay from "@/components/FrameOverlay";
import EventTimeline from "@/components/EventTimeline";
import StatsChart from "@/components/StatsChart";
import TeamComparison from "@/components/TeamComparison";
import CoachingInsights from "@/components/CoachingInsights";
import Heatmap from "@/components/Heatmap";
import { frameImageStore } from "@/lib/frameImageStore";
import { denseFrameStore } from "@/lib/denseFrameStore";
import { matchLibrary } from "@/lib/matchLibrary";
import { buildPassNetwork, countPossessionPasses, estimateShotXg, teamExpectedGoals } from "@/lib/visionMetrics";

const PANEL = "rounded-lg border border-[#1c3020] bg-[#0b130d] shadow-[0_0_40px_rgba(74,222,128,0.03)]";
const EYEBROW = "text-[11px] uppercase tracking-[0.28em] text-[#5f7567] font-mono";

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "TM";
}

function estimateXg(stats: MatchAnalysis["homeTeam"]["stats"]) {
  if (typeof stats.expectedGoals === "number") return stats.expectedGoals;
  const value = stats.shots * 0.09 + stats.shotsOnTarget * 0.18 + stats.goals * 0.32 + stats.corners * 0.04;
  return Math.max(stats.goals * 0.65, value);
}

// For the events timeline only: when several events land in the same second
// (e.g. a regain + carry + speculative shot read off one moment), show just the
// single most trustworthy one — a goal always wins, otherwise highest confidence.
// keyEvents stays untouched so team stats/counts are unaffected.
function collapseConcurrentEvents(events: MatchEvent[]): MatchEvent[] {
  const groups = new Map<number, MatchEvent[]>();
  for (const event of events) {
    const bucket = Math.floor(event.timestamp);
    const list = groups.get(bucket);
    if (list) list.push(event);
    else groups.set(bucket, [event]);
  }
  const pickBest = (group: MatchEvent[]) => {
    const goals = group.filter((e) => e.type === "goal");
    const pool = goals.length ? goals : group;
    return pool.reduce((best, e) => (e.confidence > best.confidence ? e : best));
  };
  return [...groups.values()].map(pickBest).sort((a, b) => a.timestamp - b.timestamp);
}

function bestScoreboardLabel(frames: FrameData[], side: "home" | "away") {
  const key = side === "home" ? "homeLabel" : "awayLabel";
  const counts = new Map<string, number>();

  for (const frame of frames) {
    const label = frame.scoreboard?.[key];
    if (!label) continue;
    const cleaned = label.trim().replace(/\s+/g, " ");
    if (cleaned.length < 2 || cleaned.length > 20 || /^\d+$/.test(cleaned)) continue;
    counts.set(cleaned, (counts.get(cleaned) ?? 0) + 1);
  }

  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

function withScoreboardTeamNames(analysis: MatchAnalysis): MatchAnalysis {
  const homeLabel = bestScoreboardLabel(analysis.frames, "home");
  const awayLabel = bestScoreboardLabel(analysis.frames, "away");
  const homeIsGeneric = /^(home team|home)$/i.test(analysis.homeTeam.name);
  const awayIsGeneric = /^(away team|away)$/i.test(analysis.awayTeam.name);

  if ((!homeLabel || !homeIsGeneric) && (!awayLabel || !awayIsGeneric)) return analysis;

  return {
    ...analysis,
    homeTeam: {
      ...analysis.homeTeam,
      name: homeLabel && homeIsGeneric ? homeLabel : analysis.homeTeam.name,
    },
    awayTeam: {
      ...analysis.awayTeam,
      name: awayLabel && awayIsGeneric ? awayLabel : analysis.awayTeam.name,
    },
  };
}

function pct(value: number, total: number) {
  return Math.max(4, Math.min(100, (value / Math.max(total, 1)) * 100));
}

function renderablePlayers(players: Player[], maxPerTeam = 11) {
  return (["home", "away"] as const).flatMap((team) => {
    const best = new Map<string, Player>();
    for (const player of players.filter((p) => p.team === team)) {
      if (!best.has(player.id)) best.set(player.id, player);
    }
    return [...best.values()].slice(0, maxPerTeam);
  });
}

function renderableFrame(frame: FrameData): FrameData {
  return {
    ...frame,
    players: renderablePlayers(frame.players),
  };
}

// Linear interpolation between sampled frames for smooth overlay motion.
// IDs from the YOLO worker are currently per-frame, so players are also matched
// by nearest same-team position to avoid visual jumps when detection order changes.
function lerpPosition(a: Position, b: Position, alpha: number): Position {
  return {
    x: +(a.x + (b.x - a.x) * alpha).toFixed(2),
    y: +(a.y + (b.y - a.y) * alpha).toFixed(2),
  };
}

function lerpPitchView(a: PitchView, b: PitchView, alpha: number): PitchView {
  return {
    lengthMin: +(a.lengthMin + (b.lengthMin - a.lengthMin) * alpha).toFixed(2),
    lengthMax: +(a.lengthMax + (b.lengthMax - a.lengthMax) * alpha).toFixed(2),
    topImageY: +(a.topImageY + (b.topImageY - a.topImageY) * alpha).toFixed(2),
    confidence:
      typeof a.confidence === "number" && typeof b.confidence === "number"
        ? +(a.confidence + (b.confidence - a.confidence) * alpha).toFixed(2)
        : a.confidence ?? b.confidence,
  };
}

function pitchViewDelta(a: PitchView, b: PitchView) {
  return Math.max(
    Math.abs(a.lengthMin - b.lengthMin),
    Math.abs(a.lengthMax - b.lengthMax),
    Math.abs(a.topImageY - b.topImageY)
  );
}

function smoothPitchView(previous: PitchView, target: PitchView): PitchView {
  return pitchViewDelta(previous, target) > 8
    ? target
    : lerpPitchView(previous, target, 0.65);
}

function distance(a: Position, b: Position) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function matchPlayersByTeamAndDistance(aPlayers: Player[], bPlayers: Player[], maxDistance = 16) {
  const used = new Set<string>();

  return aPlayers.map((player) => {
    // ID-first: if the same stable ID exists in the target frame, use it directly.
    const idMatch = bPlayers.find((p) => p.id === player.id && !used.has(p.id));
    if (idMatch) {
      used.add(idMatch.id);
      return { player, match: idMatch };
    }

    let best: Player | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const candidate of bPlayers) {
      if (used.has(candidate.id) || candidate.team !== player.team) continue;

      const d = distance(player.position, candidate.position);
      const numberMatches = player.number > 0 && candidate.number > 0 && player.number === candidate.number;
      const allowedDistance = numberMatches ? maxDistance * 1.6 : maxDistance;

      if (d <= allowedDistance && d < bestDistance) {
        best = candidate;
        bestDistance = d;
      }
    }

    if (best) used.add(best.id);
    return { player, match: best };
  });
}

function interpolateFrame(a: FrameData, b: FrameData, alpha: number): FrameData {
  // ID-first: stable IDs from the worker should now match directly.
  // Fall back to nearest-same-team distance only when an ID has dropped out.
  const bById = new Map(b.players.map((p) => [p.id, p]));
  const players = a.players.map((player) => {
    const match = bById.get(player.id) ??
      b.players.find((p) => p.team === player.team && player.number > 0 && p.number === player.number);
    if (!match) return player;
    return {
      ...player,
      position: lerpPosition(player.position, match.position, alpha),
      pitchPosition:
        player.pitchPosition && match.pitchPosition
          ? lerpPosition(player.pitchPosition, match.pitchPosition, alpha)
          : player.pitchPosition,
    };
  });

  const ballPosition =
    a.ballPosition && b.ballPosition
      ? {
          x: a.ballPosition.x + (b.ballPosition.x - a.ballPosition.x) * alpha,
          y: a.ballPosition.y + (b.ballPosition.y - a.ballPosition.y) * alpha,
        }
      : a.ballPosition;
  const pitchBall =
    a.pitchBall && b.pitchBall ? lerpPosition(a.pitchBall, b.pitchBall, alpha) : a.pitchBall;
  const pitchView =
    a.pitchView && b.pitchView ? lerpPitchView(a.pitchView, b.pitchView, alpha) : a.pitchView ?? b.pitchView;

  return { ...a, players, ballPosition, pitchBall, pitchView };
}

function interpolateDenseFrame(frames: FrameData[], timestamp: number): FrameData {
  if (frames.length === 1 || timestamp <= frames[0].timestamp) return frames[0];
  const last = frames[frames.length - 1];
  if (timestamp >= last.timestamp) return last;

  let nextIdx = frames.findIndex((frame) => frame.timestamp >= timestamp);
  if (nextIdx <= 0) nextIdx = 1;

  const prev = frames[nextIdx - 1];
  const next = frames[nextIdx];
  const span = Math.max(0.001, next.timestamp - prev.timestamp);
  const alpha = Math.min(1, Math.max(0, (timestamp - prev.timestamp) / span));
  if (prev.isPitchView === false || next.isPitchView === false) {
    return alpha < 0.5 ? prev : next;
  }

  return interpolateFrame(prev, next, alpha);
}

function easeDisplayedFrame(previous: FrameData | null, target: FrameData): FrameData {
  if (!previous || Math.abs(previous.timestamp - target.timestamp) > 1) return target;
  const canCoastMissingPlayers = target.isPitchView !== false && target.players.length >= 6;

  // Match purely by ID — stable IDs from the worker make this reliable and avoid
  // the contradictory distance thresholds that caused visual jitter before.
  const prevById = new Map(previous.players.map((p) => [p.id, p]));
  const targetIds = new Set(target.players.map((p) => p.id));
  const smoothedPlayers = target.players.map((player) => {
    const prev = prevById.get(player.id);
    if (!prev) return player;
    return {
      ...player,
      position: lerpPosition(prev.position, player.position, 0.28),
      pitchPosition:
        prev.pitchPosition && player.pitchPosition
          ? lerpPosition(prev.pitchPosition, player.pitchPosition, 0.28)
          : player.pitchPosition,
    };
  });
  const coastedPlayers = canCoastMissingPlayers
    ? previous.players
        .filter((player) => !targetIds.has(player.id) && /^([ha])\d+$/.test(player.id))
        .slice(0, 4)
        .map((player) => ({ ...player, action: "standing" as const }))
    : [];

  const ballPosition =
    previous.ballPosition && target.ballPosition
      ? lerpPosition(previous.ballPosition, target.ballPosition, distance(previous.ballPosition, target.ballPosition) > 6 ? 0.9 : 0.65)
      : target.ballPosition;
  const pitchBall =
    previous.pitchBall && target.pitchBall
      ? lerpPosition(previous.pitchBall, target.pitchBall, distance(previous.pitchBall, target.pitchBall) > 6 ? 0.9 : 0.65)
      : target.pitchBall;
  const pitchView =
    previous.pitchView && target.pitchView
      ? smoothPitchView(previous.pitchView, target.pitchView)
      : target.pitchView;

  return { ...target, players: [...smoothedPlayers, ...coastedPlayers], ballPosition, pitchBall, pitchView };
}

type FieldOrientation = "broadcast" | "mirrored";

function orientPosition(position: Position, orientation: FieldOrientation): Position {
  return orientation === "mirrored"
    ? { x: 100 - position.x, y: position.y }
    : position;
}

function orientFrameForField(frame: FrameData, orientation: FieldOrientation): FrameData {
  if (orientation === "broadcast") return frame;
  return {
    ...frame,
    players: frame.players.map((player) => ({
      ...player,
      position: orientPosition(player.position, orientation),
      pitchPosition: player.pitchPosition ? orientPosition(player.pitchPosition, orientation) : undefined,
    })),
    ballPosition: frame.ballPosition ? orientPosition(frame.ballPosition, orientation) : undefined,
    pitchBall: frame.pitchBall ? orientPosition(frame.pitchBall, orientation) : undefined,
    referees: frame.referees?.map((position) => orientPosition(position, orientation)),
    pitchReferees: frame.pitchReferees?.map((position) => orientPosition(position, orientation)),
  };
}

// SVG overlay rendered on top of the video element — draws team-coloured rings
// at players' feet, matching the lightweight tracking style from the model repo.
function TrackingOverlaySvg({ frame }: { frame: FrameData }) {
  const HOME = "#ef4444";
  const AWAY = "#3b82f6";

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 1280 720"
      preserveAspectRatio="none"
    >
      {renderablePlayers(frame.players).map((player, i) => {
        const cx = (player.position.x / 100) * 1280;
        const cy = (player.position.y / 100) * 720;
        const color = player.team === "home" ? HOME : AWAY;
        return (
          <g key={`${player.id}-${i}`}>
            <path
              d={`M ${cx - 28} ${cy + 20} A 28 9 0 0 0 ${cx + 28} ${cy + 20}`}
              fill="none"
              stroke={color}
              strokeWidth={3}
              strokeOpacity={0.95}
              strokeLinecap="round"
            />
          </g>
        );
      })}
      {/* Referees */}
      {frame.referees?.map((pos, i) => (
        <path
          key={`ref-${i}`}
          d={`M ${(pos.x / 100) * 1280 - 26} ${(pos.y / 100) * 720 + 20} A 26 8 0 0 0 ${
            (pos.x / 100) * 1280 + 26
          } ${(pos.y / 100) * 720 + 20}`}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
      ))}
      {/* Ball */}
      {frame.ballPosition && (
        <g>
          <circle
            cx={(frame.ballPosition.x / 100) * 1280}
            cy={(frame.ballPosition.y / 100) * 720}
            r={8}
            fill="#fbbf24"
            stroke="#000"
            strokeWidth={1.5}
          />
        </g>
      )}
    </svg>
  );
}

function SystemHeader({
  analysis,
  denseStatus,
  denseFrames,
  onExport,
}: {
  analysis: MatchAnalysis;
  denseStatus: "idle" | "loading" | "ready" | "error";
  denseFrames: FrameData[];
  onExport: () => void;
}) {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-green-400/80 text-black flex items-center justify-center shadow-[0_0_24px_rgba(74,222,128,0.35)]">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="text-xl font-black tracking-tight text-[#f0fdf4]">
              SOCCER<span className="text-yellow-300">VISION</span>
            </div>
            <div className={EYEBROW}>computer vision match analytics</div>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <span className="rounded-full border border-green-400/25 bg-green-400/10 px-4 py-2 text-xs font-mono uppercase tracking-[0.18em] text-green-300">
            <span className="mr-2 inline-block h-2 w-2 rounded-full bg-green-400" />
            {denseStatus === "loading" ? "dense tracking loading" : "analysis complete"}
          </span>
          <span className="rounded-lg border border-[#1c3020] px-4 py-2 text-xs font-mono text-[#6f8175]">
            model: pitch-vision v3.2
          </span>
          <button
            onClick={onExport}
            className="rounded-lg bg-yellow-300 px-5 py-2 text-sm font-bold text-black hover:bg-yellow-200 transition-colors"
          >
            Export Report
          </button>
        </div>
      </div>

      <div className={`${PANEL} p-6`}>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-lg border border-green-400/35 bg-green-400/10 flex items-center justify-center text-xl font-black text-green-300">
              {initials(analysis.homeTeam.name)}
            </div>
            <div>
              <div className="text-2xl font-black text-[#f0fdf4]">{analysis.homeTeam.name}</div>
              <div className={EYEBROW}>home</div>
            </div>
          </div>

          <div className="text-center min-w-[240px]">
            <div className={EYEBROW}>clip end</div>
            <div className="font-mono text-6xl font-black leading-none text-[#f0fdf4]">
              <span className="text-green-300">{analysis.score.home}</span>
              <span className="px-6 text-[#5f7567]">-</span>
              <span>{analysis.score.away}</span>
            </div>
          </div>

          <div className="flex items-center justify-end gap-4">
            <div className="text-right">
              <div className="text-2xl font-black text-[#f0fdf4]">{analysis.awayTeam.name}</div>
              <div className={EYEBROW}>away</div>
            </div>
            <div className="w-16 h-16 rounded-lg border border-gray-400/25 bg-gray-400/10 flex items-center justify-center text-xl font-black text-gray-300">
              {initials(analysis.awayTeam.name)}
            </div>
          </div>
        </div>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-1">
        {[
          `${formatDuration(analysis.videoDuration)} analyzed`,
          `${analysis.framesAnalyzed.toLocaleString()} sampled frames`,
          `${denseFrames.length || analysis.framesAnalyzed} tracked frames`,
          `avg confidence ${Math.round((analysis.keyEvents.reduce((s, e) => s + e.confidence, 0) / Math.max(analysis.keyEvents.length, 1)) * 100)}%`,
          denseStatus === "ready" ? "ByteTrack dense active" : "sparse tracking fallback",
        ].map((item) => (
          <span key={item} className="shrink-0 rounded-lg border border-[#1c3020] bg-[#0b130d] px-4 py-2 text-xs font-mono text-[#829086]">
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function SummaryPanel({ analysis }: { analysis: MatchAnalysis }) {
  const home = analysis.homeTeam;
  const away = analysis.awayTeam;
  const homeXg = estimateXg(home.stats);
  const awayXg = estimateXg(away.stats);
  const homeWin = analysis.score.home >= analysis.score.away;
  const leading = homeWin ? home : away;
  const trailing = homeWin ? away : home;
  const scoreLabel = `${home.name} ${analysis.score.home}-${analysis.score.away} ${away.name}`;

  // Prefer the vision synthesis model's projection; fall back to the legacy
  // score/possession heuristic for analyses produced before the model ran.
  const fallbackLead = Math.min(88, Math.max(52, 50 + Math.abs(analysis.score.home - analysis.score.away) * 14 + Math.abs(home.stats.possession - away.stats.possession) * 0.35));
  const fallbackDraw = Math.max(6, Math.round((100 - fallbackLead) * 0.62));
  const fallbackOther = Math.max(4, 100 - Math.round(fallbackLead) - fallbackDraw);
  const outcome = analysis.outcome ?? {
    homeWin: homeWin ? Math.round(fallbackLead) : fallbackOther,
    draw: fallbackDraw,
    awayWin: homeWin ? fallbackOther : Math.round(fallbackLead),
    reasoning: "",
    source: "fallback" as const,
  };
  const outcomeRows = [
    { label: `${home.name} win`, value: outcome.homeWin },
    { label: "Draw", value: outcome.draw },
    { label: `${away.name} win`, value: outcome.awayWin },
  ];
  const outcomeMax = Math.max(...outcomeRows.map((r) => r.value));

  const chips = [
    { code: "CTL", text: `${leading.name} controlled ${leading.stats.possession}% possession.`, tone: "green" },
    { code: "xG", text: `${homeXg >= awayXg ? home.name : away.name} xG edge ${Math.abs(homeXg - awayXg).toFixed(2)}.`, tone: "green" },
    { code: "SET", text: `${home.stats.corners + away.stats.corners} corner situations detected.`, tone: "yellow" },
    { code: "RSK", text: `${trailing.name} conceded ${trailing.stats.shotsOnTarget} shots on target.`, tone: "orange" },
  ];

  return (
    <div className={`${PANEL} border-yellow-300/20 p-6`}>
      <div className="grid lg:grid-cols-[1.6fr_1fr] gap-8">
        <div>
          <div className="text-xs uppercase tracking-[0.28em] text-yellow-300 font-mono mb-5">
            <span className="mr-2 inline-block h-2 w-2 rounded-full bg-yellow-300/70" />
            vision ai - match summary
          </div>
          <h1 className="max-w-4xl text-3xl font-black leading-tight text-[#f0fdf4]">
            Uploaded clip ends {scoreLabel}.
          </h1>
          <p className="mt-5 max-w-4xl text-base leading-8 text-[#aeb8b0]">
            {analysis.clipSummary?.trim() ||
              "The model tracked possession, shot creation, event timing, and player movement inside this uploaded video segment. The outcome profile below describes this clip window only, not a full-match forecast."}
          </p>

          <div className="mt-7 grid md:grid-cols-2 gap-3">
            {chips.map((chip) => (
              <div key={chip.code} className="rounded-lg border border-[#1c3020] bg-[#07100a] p-4 flex items-center gap-4">
                <div className={`w-11 h-11 rounded-lg flex items-center justify-center text-xs font-black ${
                  chip.tone === "yellow" ? "bg-yellow-300/10 text-yellow-300" : chip.tone === "orange" ? "bg-orange-400/10 text-orange-300" : "bg-green-400/10 text-green-300"
                }`}>
                  {chip.code}
                </div>
                <p className="text-sm leading-6 text-[#c8d2ca]">{chip.text}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="border-l border-[#1c3020] pl-8">
          <div className={EYEBROW}>outcome model</div>
          <div className="mt-2 text-sm text-[#829086]">
            {outcome.source === "vision" ? "vision-grounded projection for this clip window" : "heuristic projection (vision model unavailable)"}
          </div>
          {outcomeRows.map((row) => (
            <div key={row.label} className="mt-7">
              <div className="flex justify-between text-sm text-[#c8d2ca]">
                <span>{row.label}</span>
                <span className="font-black">{row.value}%</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-[#142014] overflow-hidden">
                <div
                  className={`h-full rounded-full ${row.value === outcomeMax ? "bg-green-400" : "bg-[#9aa5a0]"}`}
                  style={{ width: `${row.value}%` }}
                />
              </div>
            </div>
          ))}
          <div className="mt-10 rounded-lg border border-green-400/30 bg-green-400/10 p-4 text-sm leading-6 text-green-100 flex items-start gap-3">
            <Check className="w-5 h-5 shrink-0 mt-0.5" />
            <span>{outcome.reasoning?.trim() || (homeXg === awayXg ? "Balanced clip profile" : "Clip profile advantage - review with event flags")}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function VisionMetricStrip({ analysis }: { analysis: MatchAnalysis }) {
  const home = analysis.homeTeam.stats;
  const away = analysis.awayTeam.stats;
  const homeName = analysis.homeTeam.name;
  const awayName = analysis.awayTeam.name;
  const metrics = [
    { label: "expected goals", value: estimateXg(home).toFixed(2), suffix: "xG", sub: `conf ${Math.round((home.metricConfidence?.xg ?? 0.45) * 100)}% · vs ${estimateXg(away).toFixed(2)}`, fill: pct(estimateXg(home), estimateXg(home) + estimateXg(away)) },
    { label: "possession", value: home.possession.toString(), suffix: "%", sub: `conf ${Math.round((home.metricConfidence?.possession ?? 0.5) * 100)}% · vs ${away.possession}%`, fill: home.possession },
    { label: "pass accuracy", value: home.passAccuracy.toString(), suffix: "%", sub: `inferred · vs ${away.passAccuracy}%`, fill: home.passAccuracy },
    { label: "shots on target", value: home.shotsOnTarget.toString(), suffix: "", sub: `verified · vs ${away.shotsOnTarget}`, fill: pct(home.shotsOnTarget, home.shotsOnTarget + away.shotsOnTarget) },
    { label: "distance covered", value: (home.distanceCovered / 1000).toFixed(1), suffix: "km", sub: `stable IDs ${Math.round((home.metricConfidence?.distance ?? 0) * 100)}%`, fill: pct(home.distanceCovered, home.distanceCovered + away.distanceCovered) },
    { label: "key events", value: analysis.keyEvents.length.toString(), suffix: "", sub: `${analysis.keyEvents.filter((e) => e.isKeyMoment).length} key moments`, fill: Math.min(100, analysis.keyEvents.length * 9) },
  ];

  return (
    <div>
      <div className="mb-4 flex items-center gap-2 text-xs font-mono">
        <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
        <span className="uppercase tracking-[0.18em] text-green-300">{homeName}</span>
        <span className="text-[#617169]">headline figures — each tile shows {homeName}, with “vs {awayName}” for comparison</span>
      </div>
      <div className="grid md:grid-cols-3 xl:grid-cols-6 gap-4">
      {metrics.map((metric) => (
        <div key={metric.label} className={`${PANEL} p-5`}>
          <div className={EYEBROW}>{metric.label}</div>
          <div className="mt-8 flex items-end gap-2">
            <span className="text-4xl font-black tabular-nums text-[#f0fdf4]">{metric.value}</span>
            {metric.suffix && <span className="pb-1 text-sm text-[#829086]">{metric.suffix}</span>}
          </div>
          <div className="mt-5 h-1.5 rounded-full bg-[#233128] overflow-hidden">
            <div className="h-full rounded-full bg-green-400" style={{ width: `${metric.fill}%` }} />
          </div>
          <div className="mt-3 text-xs font-mono text-[#617169]">{metric.sub}</div>
        </div>
      ))}
      </div>
    </div>
  );
}

function XgMomentumPanel({ analysis }: { analysis: MatchAnalysis }) {
  const homeEvents = analysis.keyEvents.filter((event) => event.team === "home" && ["shot", "goal", "save"].includes(event.type));
  const awayEvents = analysis.keyEvents.filter((event) => event.team === "away" && ["shot", "goal", "save"].includes(event.type));
  const duration = Math.max(analysis.videoDuration, 1);

  // Adaptive time axis: the curve is already scaled to the real clip length, so
  // the labels under it must match. Short clips read in seconds / mm:ss instead
  // of a fixed 0–90' match scale, which would otherwise mislabel a 2-minute clip
  // as a full match.
  const useSeconds = duration < 300; // under 5 minutes
  const fmtTime = (s: number) => {
    if (duration < 90) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  };
  const AXIS_TICKS = 6;
  const axisTicks = Array.from({ length: AXIS_TICKS + 1 }, (_, i) => (duration * i) / AXIS_TICKS);

  const buildPath = (events: typeof analysis.keyEvents, totalXg: number) => {
    const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);
    let current = 0;
    const parts = [`M 40 290`];
    sorted.forEach((event, index) => {
      const share = event.xg ?? estimateShotXg(event);
      const x = 40 + (event.timestamp / duration) * 620;
      const yPrev = 290 - Math.min(250, current * 92);
      current += share;
      const yNext = 290 - Math.min(250, current * 92);
      parts.push(`L ${x.toFixed(1)} ${yPrev.toFixed(1)} L ${x.toFixed(1)} ${yNext.toFixed(1)}`);
      if (index === sorted.length - 1) parts.push(`L 660 ${yNext.toFixed(1)}`);
    });
    if (!sorted.length) parts.push("L 660 290");
    return parts.join(" ");
  };

  const homeEventXg = teamExpectedGoals(analysis.keyEvents, "home");
  const awayEventXg = teamExpectedGoals(analysis.keyEvents, "away");
  const homeXg = analysis.homeTeam.stats.expectedGoals ?? (homeEventXg || estimateXg(analysis.homeTeam.stats));
  const awayXg = analysis.awayTeam.stats.expectedGoals ?? (awayEventXg || estimateXg(analysis.awayTeam.stats));
  const goalEvents = analysis.keyEvents.filter((event) => event.type === "goal");

  return (
    <div className={`${PANEL} p-6`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className={EYEBROW}>expected goals momentum</div>
          <h2 className="mt-3 text-xl font-black text-[#f0fdf4]">Cumulative xG by {useSeconds ? "second" : "minute"}</h2>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-green-300">● {analysis.homeTeam.name} {homeXg.toFixed(2)}</span>
          <span className="text-slate-400">● {analysis.awayTeam.name} {awayXg.toFixed(2)}</span>
          <span className="text-yellow-300">▲ goal</span>
        </div>
      </div>
      <svg viewBox="0 0 700 330" className="mt-4 h-[330px] w-full">
        {[0, 1, 2, 3, 4].map((i) => (
          <line key={`h-${i}`} x1={40} x2={660} y1={290 - i * 58} y2={290 - i * 58} stroke="#1c3020" strokeWidth={1} />
        ))}
        {[0, 1, 2, 3, 4].map((i) => (
          <line key={`v-${i}`} y1={35} y2={290} x1={40 + i * 155} x2={40 + i * 155} stroke="#132018" strokeWidth={1} />
        ))}
        <path d={`${buildPath(homeEvents, homeXg)} L 660 290 L 40 290 Z`} fill="rgba(74,222,128,0.12)" />
        <path d={buildPath(homeEvents, homeXg)} fill="none" stroke="#6ee787" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        <path d={buildPath(awayEvents, awayXg)} fill="none" stroke="#94a3a0" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        {goalEvents.map((event, i) => {
          const x = 40 + (event.timestamp / duration) * 620;
          return (
            <g key={event.id}>
              <line x1={x} x2={x} y1={38} y2={290} stroke="#fde047" strokeWidth={1} strokeDasharray="3 5" opacity={0.5} />
              <path d={`M ${x - 6} 28 L ${x + 6} 28 L ${x} 44 Z`} fill="#fde047" />
              <text x={x + 8} y={i % 2 ? 70 : 54} fill="#fde047" fontSize={10} fontFamily="monospace">
                {fmtTime(event.timestamp)}
              </text>
            </g>
          );
        })}
        {axisTicks.map((t, i) => (
          <text key={i} x={40 + (t / duration) * 620} y={315} fill="#617169" fontSize={12} textAnchor="middle" fontFamily="monospace">
            {fmtTime(t)}
          </text>
        ))}
      </svg>
    </div>
  );
}

function FinishingPanel({ analysis }: { analysis: MatchAnalysis }) {
  // Use clip-scoped goals (scored within the upload) — not the full-match scoreboard —
  // so goals stay consistent with the in-clip shots, xG, and conversion below.
  const clipGoals = analysis.clipGoals ?? analysis.score;
  const rows = [
    ["Total shots", analysis.homeTeam.stats.shots, analysis.awayTeam.stats.shots],
    ["On target", analysis.homeTeam.stats.shotsOnTarget, analysis.awayTeam.stats.shotsOnTarget],
    ["Corners", analysis.homeTeam.stats.corners, analysis.awayTeam.stats.corners],
    ["Conversion", Math.min(100, Math.round((clipGoals.home / Math.max(analysis.homeTeam.stats.shots, 1)) * 100)), Math.min(100, Math.round((clipGoals.away / Math.max(analysis.awayTeam.stats.shots, 1)) * 100)), "%"],
    ["Goals / xG", clipGoals.home, clipGoals.away],
  ] as const;

  return (
    <div className={`${PANEL} p-6`}>
      <div className={EYEBROW}>finishing quality</div>
      <div className="mt-2 text-sm text-[#829086]">
        within this clip · full-match scoreboard: {analysis.homeTeam.name} {analysis.score.home}–{analysis.score.away} {analysis.awayTeam.name}
      </div>
      <div className="mt-7 divide-y divide-[#1c3020]">
        {rows.map((row) => (
          <div key={row[0]} className="grid grid-cols-[80px_1fr_80px] items-center py-4 text-sm">
            <span className="font-black text-green-300">{row[0] === "Goals / xG" ? `${row[1]} / ${estimateXg(analysis.homeTeam.stats).toFixed(2)}` : `${row[1]}${row[3] ?? ""}`}</span>
            <span className="text-center text-[#829086]">{row[0]}</span>
            <span className="text-right font-black text-slate-300">{row[0] === "Goals / xG" ? `${row[2]} / ${estimateXg(analysis.awayTeam.stats).toFixed(2)}` : `${row[2]}${row[3] ?? ""}`}</span>
          </div>
        ))}
      </div>
      <div className="mt-5 rounded-lg border border-yellow-300/20 bg-yellow-300/10 p-4">
        <div className="text-xs uppercase tracking-[0.22em] font-mono text-yellow-300">model read</div>
        <p className="mt-3 text-sm leading-6 text-[#c8d2ca]">
          {analysis.homeTeam.name} generated {estimateXg(analysis.homeTeam.stats).toFixed(2)} xG while restricting {analysis.awayTeam.name} to {estimateXg(analysis.awayTeam.stats).toFixed(2)}. Dominance is measured through volume, field tilt, and chance quality.
        </p>
      </div>
    </div>
  );
}

function PitchLines() {
  return (
    <svg viewBox="0 0 700 454" className="absolute inset-0 h-full w-full">
      <rect x={20} y={24} width={660} height={406} rx={6} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.18} />
      <line x1={350} y1={24} x2={350} y2={430} stroke="#d1fae5" strokeWidth={1.5} opacity={0.14} />
      <circle cx={350} cy={227} r={62} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.12} />
      <rect x={20} y={132} width={104} height={190} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.14} />
      <rect x={20} y={189} width={40} height={76} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.14} />
      <rect x={576} y={132} width={104} height={190} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.14} />
      <rect x={640} y={189} width={40} height={76} fill="none" stroke="#d1fae5" strokeWidth={1.5} opacity={0.14} />
    </svg>
  );
}

function PassNetworkPanel({ frames, currentFrame, homeTeamName, awayTeamName }: { frames: FrameData[]; currentFrame: FrameData; homeTeamName: string; awayTeamName: string }) {
  const [selectedTeam, setSelectedTeam] = useState<"home" | "away">("home");
  const teamName = selectedTeam === "home" ? homeTeamName : awayTeamName;

  // Use all frames up to the current playback position — stable full-match window
  // prevents the network from resetting as the user scrubs through the video.
  const networkFrames = frames.filter((frame) => frame.timestamp <= currentFrame.timestamp);
  const network = buildPassNetwork(networkFrames.length ? networkFrames : [currentFrame], selectedTeam);
  const currentPlayers = currentFrame.players.filter((player) => player.team === selectedTeam).slice(0, 11);
  const nodes = (network.nodes.length
    ? network.nodes
    : currentPlayers.map((player) => ({
        id: player.id,
        number: player.number,
        team: player.team,
        position: player.position,
        touches: 1,
      }))
  ).slice(0, 11);
  const maxTouches = Math.max(1, ...nodes.map((node) => node.touches));
  const realLinks = network.links.filter((link) =>
    nodes.some((node) => node.id === link.from) && nodes.some((node) => node.id === link.to)
  );
  const estimatedLinks = nodes
    .flatMap((node, i) =>
      nodes.slice(i + 1).map((other) => ({
        from: node.id,
        to: other.id,
        count: 1,
        distance: Math.hypot(node.position.x - other.position.x, node.position.y - other.position.y),
      }))
    )
    .sort((a, b) => a.distance - b.distance)
    .filter((link) => link.distance <= 36)
    .slice(0, Math.min(10, Math.max(0, nodes.length - 2)));
  const links = realLinks.length >= 3 ? realLinks : estimatedLinks;
  const linksAreEstimated = realLinks.length < 3;
  const nodeColor = selectedTeam === "home" ? "#ef4444" : "#3b82f6";
  const nodeStroke = selectedTeam === "home" ? "#f87171" : "#60a5fa";
  const linkColor = selectedTeam === "home" ? "#f87171" : "#60a5fa";

  return (
    <div className={`${PANEL} p-6`}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className={EYEBROW}>pass network - {teamName}</div>
          <h2 className="mt-3 text-xl font-black text-[#f0fdf4]">Average Positions & Links</h2>
        </div>
        <div className="flex rounded-lg overflow-hidden border border-[#1c3020] text-xs">
          <button
            onClick={() => setSelectedTeam("home")}
            className={`px-3 py-1.5 transition-colors ${selectedTeam === "home" ? "bg-green-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"}`}
          >
            {homeTeamName}
          </button>
          <button
            onClick={() => setSelectedTeam("away")}
            className={`px-3 py-1.5 transition-colors ${selectedTeam === "away" ? "bg-red-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"}`}
          >
            {awayTeamName}
          </button>
        </div>
      </div>
      <div className="relative mx-auto mt-8 aspect-[700/454] max-w-4xl rounded-lg bg-[#09110c]">
        <PitchLines />
        <svg viewBox="0 0 700 454" className="absolute inset-0 h-full w-full">
          {links.map((link, i) => {
            const node = nodes.find((candidate) => candidate.id === link.from);
            const next = nodes.find((candidate) => candidate.id === link.to);
            if (!node || !next) return null;
            return (
              <line
                key={`${link.from}-${link.to}-${i}`}
                x1={node.position.x * 7}
                y1={node.position.y * 4.54}
                x2={next.position.x * 7}
                y2={next.position.y * 4.54}
                stroke={linkColor}
                strokeWidth={linksAreEstimated ? 3 : 1.5 + Math.min(5, link.count)}
                opacity={linksAreEstimated ? 0.42 : 0.32}
              />
            );
          })}
          {nodes.map((node, i) => (
            <g key={`${node.id}-${i}`}>
              <circle cx={node.position.x * 7} cy={node.position.y * 4.54} r={7 + (node.touches / maxTouches) * 7} fill={nodeColor} fillOpacity={0.72} stroke={nodeStroke} strokeWidth={1.5} />
              <text x={node.position.x * 7} y={node.position.y * 4.54 + 3.5} textAnchor="middle" fill="#061008" fontSize={9} fontWeight={900}>
                {node.number || i + 1}
              </text>
            </g>
          ))}
          {currentPlayers.map((player, i) => (
            <circle
              key={`live-${player.id}-${i}`}
              cx={player.position.x * 7}
              cy={player.position.y * 4.54}
              r={3.5}
              fill="#f0fdf4"
              fillOpacity={0.9}
              stroke="#061008"
              strokeWidth={1.5}
            />
          ))}
        </svg>
      </div>
      <div className="mt-5 text-xs font-mono text-[#829086]">
        {nodes.length
          ? linksAreEstimated
            ? "● node size = touches - connectors = estimated team shape - white dot = live position"
            : "● node size = touches - line weight = pass volume - white dot = live position"
          : "No stable possession-player transitions available"}
      </div>
    </div>
  );
}

function PipelinePanel({ denseStatus, denseFrames }: { denseStatus: string; denseFrames: FrameData[] }) {
  const items = [
    "Detection - YOLOv11",
    "Tracking - ByteTrack",
    "Calibration - team color centroids",
    `Dense frames - ${denseFrames.length || "pending"}`,
    denseStatus === "ready" ? "Status - live dense active" : `Status - ${denseStatus}`,
  ];
  return (
    <div className={`${PANEL} p-6`}>
      <div className={EYEBROW}>inference pipeline</div>
      <div className="mt-5 flex gap-3 flex-wrap">
        {items.map((item) => (
          <span key={item} className="rounded-lg border border-[#1c3020] bg-[#08110b] px-4 py-2 text-xs font-mono text-[#829086]">
            <span className="mr-2 inline-block h-2 w-2 rounded-full bg-green-400" />
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function DashboardContent() {
  const router = useRouter();
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<FrameData | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | undefined>(undefined);
  // Smoothly interpolated frame — updated at ~30fps via RAF for fluid overlay motion
  const [liveFrame, setLiveFrame] = useState<FrameData | null>(null);
  const [activeHeatmapTeam, setActiveHeatmapTeam] = useState<"home" | "away">("home");
  const [showReviewFlags, setShowReviewFlags] = useState(false);
  const [pitchView, setPitchView] = useState<"frame" | "tactical">("tactical");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [denseFrames, setDenseFrames] = useState<import("@/lib/types").FrameData[]>([]);
  const [denseStatus, setDenseStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
  const liveFrameRef = useRef<FrameData | null>(null);
  const loadedIdRef = useRef<string | null>(null);

  // Load the active match from the in-session library, and re-sync when the user
  // switches matches. Analysis/video/frames reset only on an actual match change
  // (tracked via loadedIdRef) so background dense-tracking updates don't reset playback.
  useEffect(() => {
    const sync = () => {
      const entry = matchLibrary.active();
      if (!entry) {
        router.push("/");
        return;
      }
      if (loadedIdRef.current !== entry.id) {
        loadedIdRef.current = entry.id;
        const data = withScoreboardTeamNames(entry.analysis);
        setAnalysis(data);
        setVideoUrl(entry.videoUrl);
        if (data.frames.length > 0) {
          setSelectedFrame(data.frames[0]);
          setLiveFrame(data.frames[0]);
        }
      }
    };
    sync();
    return matchLibrary.subscribe(sync);
  }, [router]);

  useEffect(() => {
    liveFrameRef.current = liveFrame;
  }, [liveFrame]);

  // Subscribe to dense frame store — fires whenever dense tracking status changes.
  useEffect(() => {
    const sync = () => {
      setDenseFrames(denseFrameStore.getFrames());
      setDenseStatus(denseFrameStore.getStatus());
    };
    sync(); // read current state immediately
    return denseFrameStore.subscribe(sync);
  }, []);

  // RAF loop: reads video.currentTime at ~30fps and drives both the SVG overlay and SoccerField.
  // Dense frames are interpolated between real 5fps detections, then eased lightly
  // to reduce detector jitter. Sparse analysis frames use the same interpolation fallback.
  useEffect(() => {
    const video = videoRef.current;
    // Only drive playback-synced frames in the tactical/video view. In Frame view
    // the user picks frames manually, so the RAF loop must not overwrite selection.
    if (!video || !analysis || !videoUrl || pitchView !== "tactical") return;

    const sparseSorted = [...analysis.frames].sort((a, b) => a.timestamp - b.timestamp);
    let rafId: number;
    let lastWall = 0;

    const tick = () => {
      const now = performance.now();
      if (now - lastWall >= 33) { // ~30fps wall-clock gate
        lastWall = now;
        const t = video.currentTime;

        const dense = denseFrameStore.getFrames();

        if (dense.length > 0) {
          const target = interpolateDenseFrame(dense, t);
          const smoothed = easeDisplayedFrame(liveFrameRef.current, target);
          liveFrameRef.current = smoothed;
          setSelectedFrame(target);
          setLiveFrame(smoothed);
        } else {
          // Sparse fallback: linear interpolation between the two surrounding keyframes.
          let prevIdx = 0;
          for (let i = 0; i < sparseSorted.length - 1; i++) {
            if (sparseSorted[i].timestamp <= t) prevIdx = i;
            else break;
          }
          const prev = sparseSorted[prevIdx];
          const next = sparseSorted[Math.min(prevIdx + 1, sparseSorted.length - 1)];
          setSelectedFrame(prev);
          if (prev === next || next.timestamp <= prev.timestamp) {
            liveFrameRef.current = prev;
            setLiveFrame(prev);
          } else {
            const alpha = Math.min(1, (t - prev.timestamp) / (next.timestamp - prev.timestamp));
            const target = interpolateFrame(prev, next, alpha);
            liveFrameRef.current = target;
            setLiveFrame(target);
          }
        }
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [analysis, videoUrl, denseFrames, pitchView]);

  // Count passes from the possession chain over the densest frames available,
  // rather than the throttled `pass` event timeline (which undercounts badly).
  const passStats = useMemo(
    () => countPossessionPasses(denseFrames.length ? denseFrames : analysis?.frames ?? []),
    [denseFrames, analysis]
  );

  if (!analysis) {
    return (
      <div className="min-h-screen bg-[#070e07] flex items-center justify-center">
        <div className="flex items-center gap-3 text-[#6b9e6b]">
          <Cpu className="w-5 h-5 animate-pulse text-green-400" />
          <span>Loading analysis…</span>
        </div>
      </div>
    );
  }

  const currentFrame = renderableFrame(selectedFrame ?? analysis.frames[0]);
  const displayFrame = renderableFrame(liveFrame ?? currentFrame);
  const fieldFrame = orientFrameForField(displayFrame, "broadcast");

  // Override the throttled-event pass count / accuracy with the possession-chain
  // tally so the displayed passes reflect every detected pass and accuracy tracks
  // possession. Other stats are unchanged.
  const withPossessionPasses = (team: MatchAnalysis["homeTeam"]) => {
    const ps = passStats[team.id];
    const total = ps.completed + ps.lost;
    return {
      ...team,
      stats: {
        ...team.stats,
        passes: ps.completed,
        passAccuracy: total > 0 ? Math.round((ps.completed / total) * 100) : team.stats.passAccuracy,
      },
    };
  };
  const displayHome = withPossessionPasses(analysis.homeTeam);
  const displayAway = withPossessionPasses(analysis.awayTeam);
  const displayAnalysis = { ...analysis, homeTeam: displayHome, awayTeam: displayAway };
  const passNetworkFrames = (denseFrames.length ? denseFrames : analysis.frames).map((frame) =>
    orientFrameForField(frame, "broadcast")
  );
  const frameImage = currentFrame ? frameImageStore.get(currentFrame.frameIndex) : null;

  const seekTo = (timestamp: number) => {
    const closest = analysis.frames.reduce((best, f) =>
      Math.abs(f.timestamp - timestamp) < Math.abs(best.timestamp - timestamp) ? f : best
    );
    setSelectedFrame(closest);
    setLiveFrame(closest);
    if (videoRef.current && videoUrl) {
      videoRef.current.currentTime = timestamp;
    }
  };

  const seekToEvent = (event: MatchEvent) => {
    setSelectedEventId(event.id);
    seekTo(event.timestamp);
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `match-analysis-${analysis.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Shared tracking panel — Live Tracking keeps the video + overlay as the primary
  // surface. The tactical field is rendered as its own report panel below.
  const trackingPanel = (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div>
          <h2 className="text-sm font-semibold text-[#f0fdf4]">
            {pitchView === "tactical" ? "Live Tracking" : "Frame View"}
          </h2>
          {pitchView === "tactical" && videoUrl && (
            <p className="text-xs text-[#6b9e6b] mt-0.5 flex items-center gap-2">
              <span>Player rings and field sync to playback in real time</span>
              {denseStatus === "loading" && (
                <span className="rounded-full border border-yellow-400/30 bg-yellow-400/10 px-2 py-0.5 text-yellow-300 animate-pulse">
                  Dense tracking loading
                </span>
              )}
              {denseStatus === "ready" && (
                <span className="rounded-full border border-green-400/30 bg-green-400/10 px-2 py-0.5 text-green-300">
                  {denseFrames.length} dense frames active
                </span>
              )}
            </p>
          )}
          {pitchView === "frame" && currentFrame && (
            <p className="text-xs text-[#6b9e6b] mt-0.5">
              {currentFrame.players.filter((p) => p.team === "home").length} home ·{" "}
              {currentFrame.players.filter((p) => p.team === "away").length} away detected
            </p>
          )}
        </div>
        <div className="flex rounded-lg border border-[#1c3020] overflow-hidden text-xs">
          <button
            onClick={() => setPitchView("tactical")}
            className={`px-2.5 py-1 transition-colors ${pitchView === "tactical" ? "bg-green-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"}`}
          >
            Tactical
          </button>
          <button
            onClick={() => setPitchView("frame")}
            className={`px-2.5 py-1 transition-colors ${pitchView === "frame" ? "bg-green-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"}`}
          >
            Frame
          </button>
        </div>
      </div>

      {pitchView === "tactical" && (
        videoUrl ? (
          <div className="relative rounded-lg overflow-hidden bg-black">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full block aspect-video object-contain"
            />
            <TrackingOverlaySvg frame={displayFrame} />
          </div>
        ) : (
          <div className="border border-dashed border-[#1c3020] rounded-lg flex items-center justify-center gap-3 text-[#6b9e6b] text-xs py-16">
            <Film className="w-4 h-4" />
            Upload a clip to enable live tracking overlay
          </div>
        )
      )}

      {pitchView === "frame" && (
        <div className="space-y-3">
          <div className="flex items-center gap-1 flex-wrap">
            {analysis.frames.map((f) => (
              <button
                key={f.frameIndex}
                onClick={() => setSelectedFrame(f)}
                className={`text-xs px-2 py-1 rounded transition-colors ${
                  selectedFrame?.frameIndex === f.frameIndex
                    ? "bg-green-400 text-black font-medium"
                    : "bg-[#142014] text-[#6b9e6b] hover:bg-[#1c3020]"
                }`}
              >
                {Math.floor(f.timestamp / 60)}:{String(Math.floor(f.timestamp % 60)).padStart(2, "0")}
              </button>
            ))}
          </div>
          {currentFrame && (
            frameImage
              ? <FrameOverlay base64={frameImage} players={currentFrame.players} ballPosition={currentFrame.ballPosition} referees={currentFrame.referees} />
              : <SoccerField frame={currentFrame} />
          )}
          {!frameImage && (
            <p className="text-xs text-[#3d5c40] mt-2 text-center italic">
              Frame images available after uploading a video
            </p>
          )}
        </div>
      )}
    </div>
  );

  const tacticalFramePanel = (
    <div className={`${PANEL} p-6`}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className={EYEBROW}>tactical frame</div>
          <h2 className="mt-3 text-xl font-black text-[#f0fdf4]">Current Shape</h2>
        </div>
        <span className="rounded-lg border border-[#1c3020] px-3 py-1.5 text-xs font-mono text-[#829086]">
          {formatDuration(displayFrame.timestamp)}
        </span>
      </div>
      <div className="mt-6 mx-auto max-w-4xl">
        <SoccerField frame={fieldFrame} />
      </div>
    </div>
  );

  const spatialOccupancyPanel = (
    <div className={`${PANEL} p-6`}>
      <div className="flex items-center justify-between gap-4 mb-6">
        <div>
          <div className={EYEBROW}>player movement heatmap</div>
          <h2 className="mt-3 text-2xl font-black text-[#f0fdf4]">Spatial Occupancy</h2>
        </div>
        <div className="flex rounded-lg overflow-hidden border border-[#1c3020] bg-[#07100a] p-1">
          <button
            onClick={() => setActiveHeatmapTeam("home")}
            className={`rounded-md px-4 py-2 text-xs font-bold transition-colors ${
              activeHeatmapTeam === "home" ? "bg-green-400 text-black" : "text-[#829086] hover:text-[#f0fdf4]"
            }`}
          >
            {analysis.homeTeam.name}
          </button>
          <button
            onClick={() => setActiveHeatmapTeam("away")}
            className={`rounded-md px-4 py-2 text-xs font-bold transition-colors ${
              activeHeatmapTeam === "away" ? "bg-green-400 text-black" : "text-[#829086] hover:text-[#f0fdf4]"
            }`}
          >
            {analysis.awayTeam.name}
          </button>
        </div>
      </div>
      <div className="mx-auto max-w-2xl">
        <Heatmap team={activeHeatmapTeam === "home" ? analysis.homeTeam : analysis.awayTeam} />
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#050a07]">
      {/* Nav */}
      <nav className="border-b border-[#132018] px-6 py-3 sticky top-0 z-10 bg-[#050a07]/95 backdrop-blur-sm">
        <div className="max-w-[1500px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 text-[#6b9e6b] hover:text-[#f0fdf4] transition-colors text-sm"
            >
              <ChevronLeft className="w-4 h-4" />
              New analysis
            </button>
            <div className="h-4 w-px bg-[#1c3020]" />
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-green-400/10 flex items-center justify-center">
                <Activity className="w-3 h-3 text-green-400" />
              </div>
              <span className="font-semibold text-sm text-[#f0fdf4]">SoccerVision Report</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {matchLibrary.list().length > 1 ? (
              <div className="flex items-center gap-1 rounded-lg border border-[#1c3020] p-1">
                {matchLibrary.list().map((m) => (
                  <button
                    key={m.id}
                    onClick={() => matchLibrary.setActive(m.id)}
                    title={m.title}
                    className={`max-w-[140px] truncate rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      matchLibrary.activeId() === m.id ? "bg-green-400 text-black" : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                    }`}
                  >
                    {m.title}
                  </button>
                ))}
              </div>
            ) : (
              <span className="rounded-lg border border-[#1c3020] px-3 py-1.5 text-xs font-medium text-[#6b9e6b]">
                Insights
              </span>
            )}

            {analysis.processingMethod === "demo" && (
              <span className="text-xs bg-yellow-400/10 text-yellow-400 px-2 py-1 rounded-full border border-yellow-400/20">
                Demo
              </span>
            )}
            {analysis.processingMethod === "ai" && (
              <span className="text-xs bg-green-400/10 text-green-400 px-2 py-1 rounded-full border border-green-400/20">
                AI processed
              </span>
            )}

            <button
              onClick={exportJson}
              className="flex items-center gap-1.5 text-xs text-[#6b9e6b] hover:text-[#f0fdf4] border border-[#1c3020] hover:border-[#2d4a30] transition-colors px-3 py-1.5 rounded-lg"
            >
              <Download className="w-3.5 h-3.5" />
              Export JSON
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-[1500px] mx-auto px-6 py-6 space-y-6">
        <SystemHeader analysis={analysis} denseStatus={denseStatus} denseFrames={denseFrames} onExport={exportJson} />

        {((analysis.analysisWarnings?.length ?? 0) > 0 || (analysis.eventConflicts?.length ?? 0) > 0) && (
          <div className="border border-amber-400/25 bg-amber-400/10 rounded-lg">
            <button
              onClick={() => setShowReviewFlags((v) => !v)}
              className="flex w-full items-center gap-3 p-3 text-left"
            >
              <AlertTriangle className="w-4 h-4 text-amber-300 flex-shrink-0" />
              <span className="flex-1 text-xs font-semibold text-amber-200">
                Review flags · {(analysis.eventConflicts?.length ?? 0)} event conflict(s)
                {(analysis.analysisWarnings?.length ?? 0) > 0 ? `, ${analysis.analysisWarnings?.length} note(s)` : ""}
              </span>
              <ChevronDown className={`w-4 h-4 text-amber-300 transition-transform ${showReviewFlags ? "rotate-180" : ""}`} />
            </button>
            {showReviewFlags && (
              <div className="space-y-1 px-3 pb-3 pl-10 text-xs leading-relaxed text-amber-100/90">
                <div>{(analysis.eventConflicts?.length ?? 0)} event conflict(s).</div>
                {analysis.analysisWarnings?.map((warning, i) => (
                  <div key={i}>{warning}</div>
                ))}
              </div>
            )}
          </div>
        )}

        <>
            <SummaryPanel analysis={analysis} />
            <VisionMetricStrip analysis={displayAnalysis} />

            <div className="grid lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8">{trackingPanel}</div>
              <div className="lg:col-span-4 relative">
                <div className={`${PANEL} p-5 flex flex-col lg:absolute lg:inset-0`}>
                  <div className={EYEBROW}>match events</div>
                  <h2 className="mt-2 text-xl font-black text-[#f0fdf4]">Auto-extracted from video</h2>
                  <div className="mt-5 flex-1 min-h-0">
                    <EventTimeline
                      events={collapseConcurrentEvents(analysis.keyEvents)}
                      selectedEventId={selectedEventId}
                      selectedTimestamp={selectedFrame?.timestamp}
                      onSelect={seekToEvent}
                      homeTeamName={analysis.homeTeam.name}
                      awayTeamName={analysis.awayTeam.name}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-4">
              {tacticalFramePanel}
              <PassNetworkPanel frames={passNetworkFrames} currentFrame={fieldFrame} homeTeamName={analysis.homeTeam.name} awayTeamName={analysis.awayTeam.name} />
            </div>

            <div className="grid lg:grid-cols-2 gap-4">
              <div className={`${PANEL} p-6`}>
                <div className={EYEBROW}>head to head</div>
                <h2 className="mt-3 text-xl font-black text-[#f0fdf4]">Team Comparison</h2>
                <div className="mt-5">
                  <TeamComparison homeTeam={displayHome} awayTeam={displayAway} />
                </div>
              </div>
              <FinishingPanel analysis={analysis} />
            </div>

            <div className="grid lg:grid-cols-3 gap-4">
              <XgMomentumPanel analysis={analysis} />
              <div className={`${PANEL} p-6`}>
                <div className={EYEBROW}>action statistics</div>
                <h2 className="mt-3 text-xl font-black text-[#f0fdf4]">Duels & Set Actions</h2>
                <div className="mt-5">
                  <StatsChart homeTeam={analysis.homeTeam} awayTeam={analysis.awayTeam} />
                </div>
              </div>
              {spatialOccupancyPanel}
            </div>

            <div className={`${PANEL} p-6`}>
              <div className="flex items-center gap-3 mb-7">
                <div className="w-12 h-12 rounded-lg bg-green-400/10 border border-green-400/25 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-black text-[#f0fdf4]">Coaching Insights</h2>
                  <p className="text-sm text-[#829086]">AI-generated performance analysis and recommendations</p>
                </div>
              </div>
              <CoachingInsights
                insights={analysis.insights}
                homeTeamName={analysis.homeTeam.name}
                awayTeamName={analysis.awayTeam.name}
              />
            </div>
            <PipelinePanel denseStatus={denseStatus} denseFrames={denseFrames} />
          </>
      </main>

      <footer className="border-t border-[#1c3020] py-5 text-center text-[#6b9e6b] text-xs mt-8">
        SoccerVision Analytics · {analysis.id} · {new Date(analysis.processedAt).toLocaleString()}
      </footer>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#070e07] flex items-center justify-center">
          <Cpu className="w-6 h-6 animate-pulse text-green-400" />
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
