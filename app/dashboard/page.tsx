"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Activity,
  Film,
  Trophy,
  ChevronLeft,
  Clock,
  Download,
  Cpu,
  Route,
  Target,
  AlertTriangle,
} from "lucide-react";
import type { MatchAnalysis, FrameData, Player, Position } from "@/lib/types";
import SoccerField from "@/components/SoccerField";
import FrameOverlay from "@/components/FrameOverlay";
import EventTimeline from "@/components/EventTimeline";
import StatsChart from "@/components/StatsChart";
import TeamComparison from "@/components/TeamComparison";
import CoachingInsights from "@/components/CoachingInsights";
import Heatmap from "@/components/Heatmap";
import MetricCard from "@/components/MetricCard";
import { videoStore } from "@/lib/videoStore";
import { frameImageStore } from "@/lib/frameImageStore";
import { denseFrameStore } from "@/lib/denseFrameStore";

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
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

function distance(a: Position, b: Position) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function matchPlayersByTeamAndDistance(aPlayers: Player[], bPlayers: Player[], maxDistance = 16) {
  const used = new Set<string>();

  return aPlayers.map((player) => {
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
  const matches = matchPlayersByTeamAndDistance(a.players, b.players, 26);
  const players = a.players.map((player) => {
    const match =
      b.players.find((p) => p.team === player.team && player.number > 0 && p.number === player.number) ??
      matches.find((entry) => entry.player.id === player.id)?.match;
    if (!match) return player;
    return {
      ...player,
      position: lerpPosition(player.position, match.position, alpha),
    };
  });

  const ballPosition =
    a.ballPosition && b.ballPosition
      ? {
          x: a.ballPosition.x + (b.ballPosition.x - a.ballPosition.x) * alpha,
          y: a.ballPosition.y + (b.ballPosition.y - a.ballPosition.y) * alpha,
        }
      : a.ballPosition;

  return { ...a, players, ballPosition };
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

  return interpolateFrame(prev, next, alpha);
}

function easeDisplayedFrame(previous: FrameData | null, target: FrameData): FrameData {
  if (!previous || Math.abs(previous.timestamp - target.timestamp) > 1) return target;

  const matches = matchPlayersByTeamAndDistance(previous.players, target.players, 12);
  const smoothedPlayers = target.players.map((player) => {
    const prev = matches.find((entry) => entry.match?.id === player.id)?.player;
    if (!prev) return player;

    return {
      ...player,
      position: lerpPosition(prev.position, player.position, 0.42),
    };
  });

  const ballPosition =
    previous.ballPosition && target.ballPosition
      ? lerpPosition(previous.ballPosition, target.ballPosition, 0.55)
      : target.ballPosition;

  return { ...target, players: smoothedPlayers, ballPosition };
}

type ViewMode = "coach" | "player";

// SVG overlay rendered on top of the video element — draws team-coloured rings
// at players' feet, matching the lightweight tracking style from the model repo.
function TrackingOverlaySvg({ frame }: { frame: FrameData }) {
  const HOME = "#3b82f6";
  const AWAY = "#ef4444";

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 1280 720"
      preserveAspectRatio="none"
    >
      {frame.players.map((player) => {
        const cx = (player.position.x / 100) * 1280;
        const cy = (player.position.y / 100) * 720;
        const color = player.team === "home" ? HOME : AWAY;
        return (
          <g key={player.id}>
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

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<FrameData | null>(null);
  // Smoothly interpolated frame — updated at ~30fps via RAF for fluid overlay motion
  const [liveFrame, setLiveFrame] = useState<FrameData | null>(null);
  const [activeHeatmapTeam, setActiveHeatmapTeam] = useState<"home" | "away">("home");
  const [viewMode, setViewMode] = useState<ViewMode>("coach");
  const [pitchView, setPitchView] = useState<"frame" | "tactical">("tactical");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [denseFrames, setDenseFrames] = useState<import("@/lib/types").FrameData[]>([]);
  const [denseStatus, setDenseStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
  const liveFrameRef = useRef<FrameData | null>(null);

  useEffect(() => {
    const url = videoStore.get();
    setVideoUrl(url);
  }, []);

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
    if (!video || !analysis || !videoUrl) return;

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
  }, [analysis, videoUrl, denseFrames]);

  useEffect(() => {
    const stored = sessionStorage.getItem("matchAnalysis");
    if (stored) {
      try {
        const data = JSON.parse(stored) as MatchAnalysis;
        setAnalysis(data);
        if (data.frames.length > 0) {
          setSelectedFrame(data.frames[0]);
          setLiveFrame(data.frames[0]);
        }
      } catch {
        router.push("/");
      }
    } else {
      router.push("/");
    }
  }, [router, searchParams]);

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

  const currentFrame = selectedFrame ?? analysis.frames[0];
  const displayFrame = liveFrame ?? currentFrame;
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

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `match-analysis-${analysis.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Shared pitch panel — Tactical stacks video+overlay above the abstract field
  // (both driven by the interpolated liveFrame), Frame shows
  // the raw extracted image with dot overlay for precise frame inspection.
  const pitchPanel = (
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
          <div className="space-y-4">
            <div className="relative rounded-lg overflow-hidden bg-black">
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                className="w-full block aspect-video object-contain"
              />
              <TrackingOverlaySvg frame={displayFrame} />
            </div>
            <div className="rounded-lg border border-[#1c3020] bg-[#081208] p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-[#6b9e6b]">
                <span>Tactical frame</span>
                <span>{formatDuration(displayFrame.timestamp)}</span>
              </div>
              <div className="mx-auto max-w-4xl">
                <SoccerField frame={displayFrame} />
              </div>
            </div>
          </div>
        ) : (
          // No video — show placeholder + full-width tactical field
          <div className="space-y-3">
            <div className="border border-dashed border-[#1c3020] rounded-lg flex items-center justify-center gap-3 text-[#6b9e6b] text-xs py-5">
              <Film className="w-4 h-4" />
              Upload a clip to enable live tracking overlay
            </div>
            {currentFrame && <SoccerField frame={currentFrame} />}
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

  return (
    <div className="min-h-screen bg-[#070e07]">
      {/* Nav */}
      <nav className="border-b border-[#1c3020] px-6 py-3 sticky top-0 z-10 bg-[#070e07]/95 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
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
              <div className="w-6 h-6 rounded bg-green-400/10 flex items-center justify-center">
                <Activity className="w-3 h-3 text-green-400" />
              </div>
              <span className="font-semibold text-sm text-[#f0fdf4]">Match Analysis</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex rounded-lg border border-[#1c3020] overflow-hidden text-xs">
              <button
                onClick={() => setViewMode("coach")}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  viewMode === "coach" ? "bg-green-400 text-black" : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                }`}
              >
                Coach
              </button>
              <button
                onClick={() => setViewMode("player")}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  viewMode === "player" ? "bg-green-400 text-black" : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                }`}
              >
                Player
              </button>
            </div>

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

            {viewMode === "coach" && (
              <button
                onClick={exportJson}
                className="flex items-center gap-1.5 text-xs text-[#6b9e6b] hover:text-[#f0fdf4] border border-[#1c3020] hover:border-[#2d4a30] transition-colors px-3 py-1.5 rounded-lg"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON
              </button>
            )}
          </div>
        </div>
      </nav>

      <main className="max-w-[1500px] mx-auto px-6 py-6 space-y-6">
        {/* Match header */}
        <div className="card p-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-green-400/15 flex items-center justify-center">
                  <span className="text-green-400 font-bold text-xs">HM</span>
                </div>
                <div>
                  <div className="font-bold text-[#f0fdf4]">{analysis.homeTeam.name}</div>
                  <div className="text-xs text-[#6b9e6b]">Home</div>
                </div>
              </div>

              <div className="text-center">
                <div className="text-3xl font-bold text-[#f0fdf4] font-mono">
                  {analysis.score.home} — {analysis.score.away}
                </div>
                <div className="text-xs text-[#6b9e6b] mt-0.5">Final</div>
              </div>

              <div className="flex items-center gap-3">
                <div>
                  <div className="font-bold text-[#f0fdf4] text-right">{analysis.awayTeam.name}</div>
                  <div className="text-xs text-[#6b9e6b] text-right">Away</div>
                </div>
                <div className="w-10 h-10 rounded-xl bg-gray-400/15 flex items-center justify-center">
                  <span className="text-gray-300 font-bold text-xs">AW</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-[#6b9e6b]">
              <div className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                {formatDuration(analysis.videoDuration)} analysed
              </div>
              <div className="flex items-center gap-1.5">
                <Film className="w-3.5 h-3.5" />
                {analysis.framesAnalyzed} frames
              </div>
            </div>
          </div>
        </div>

        {((analysis.analysisWarnings?.length ?? 0) > 0 || (analysis.eventConflicts?.length ?? 0) > 0) && (
          <div className="border border-amber-400/25 bg-amber-400/10 rounded-lg p-3 flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 text-amber-300 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-amber-100/90 leading-relaxed">
              <div className="font-semibold text-amber-200">Review flags</div>
              <div>{(analysis.eventConflicts?.length ?? 0)} event conflict(s).</div>
              {analysis.analysisWarnings?.map((warning, i) => (
                <div key={i}>{warning}</div>
              ))}
            </div>
          </div>
        )}

        {/* ── PLAYER MODE ── */}
        {viewMode === "player" && (
          <>
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-5">
                <div className="w-8 h-8 rounded-lg bg-green-400/10 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-green-400" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[#f0fdf4]">Coaching Insights</h2>
                  <p className="text-xs text-[#6b9e6b]">Priority-ranked, plain-English recommendations</p>
                </div>
              </div>
              <CoachingInsights
                insights={analysis.insights}
                homeTeamName={analysis.homeTeam.name}
                awayTeamName={analysis.awayTeam.name}
              />
            </div>

            <div className="grid lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8">{pitchPanel}</div>
              <div className="lg:col-span-4 card p-4">
                <h2 className="text-sm font-semibold text-[#f0fdf4] mb-3">Event Timeline</h2>
                <EventTimeline
                  events={analysis.keyEvents}
                  selectedTimestamp={selectedFrame?.timestamp}
                  onSelect={seekTo}
                  homeTeamName={analysis.homeTeam.name}
                  awayTeamName={analysis.awayTeam.name}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <MetricCard
                icon={Trophy}
                label="Goals"
                value={`${analysis.score.home} – ${analysis.score.away}`}
                sub={`${analysis.keyEvents.filter((e) => e.type === "goal").length} detected`}
              />
              <MetricCard
                icon={Activity}
                label="Possession"
                value={`${analysis.homeTeam.stats.possession}%`}
                sub={`vs ${analysis.awayTeam.stats.possession}% away`}
              />
              <MetricCard
                icon={Target}
                label="Shots on target"
                value={`${analysis.homeTeam.stats.shotsOnTarget + analysis.awayTeam.stats.shotsOnTarget}`}
                sub={`${analysis.homeTeam.stats.shotsOnTarget} home · ${analysis.awayTeam.stats.shotsOnTarget} away`}
              />
            </div>
          </>
        )}

        {/* ── COACH MODE ── */}
        {viewMode === "coach" && (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                icon={Activity}
                label="Home possession"
                value={`${analysis.homeTeam.stats.possession}%`}
                sub={`vs ${analysis.awayTeam.stats.possession}% away`}
              />
              <MetricCard
                icon={Trophy}
                label="Key events"
                value={analysis.keyEvents.length}
                sub={`${analysis.keyEvents.filter((e) => e.type === "goal").length} goal(s) detected`}
              />
              <MetricCard
                icon={Route}
                label="Home distance"
                value={`${analysis.homeTeam.stats.distanceCovered}m`}
                sub={`Away: ${analysis.awayTeam.stats.distanceCovered}m`}
              />
              <MetricCard
                icon={Film}
                label="Frames analysed"
                value={analysis.framesAnalyzed}
                sub={`every ~${Math.round(analysis.videoDuration / Math.max(analysis.framesAnalyzed, 1))}s`}
              />
            </div>

            <div className="grid lg:grid-cols-12 gap-4">
              <div className="lg:col-span-8">{pitchPanel}</div>
              <div className="lg:col-span-4 card p-4">
                <h2 className="text-sm font-semibold text-[#f0fdf4] mb-3">Event Timeline</h2>
                <EventTimeline
                  events={analysis.keyEvents}
                  selectedTimestamp={selectedFrame?.timestamp}
                  onSelect={seekTo}
                  homeTeamName={analysis.homeTeam.name}
                  awayTeamName={analysis.awayTeam.name}
                />
              </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-4">
              <div className="card p-5">
                <h2 className="text-sm font-semibold text-[#f0fdf4] mb-4">Action Statistics</h2>
                <StatsChart homeTeam={analysis.homeTeam} awayTeam={analysis.awayTeam} />
              </div>
              <div className="card p-5">
                <h2 className="text-sm font-semibold text-[#f0fdf4] mb-4">Team Comparison</h2>
                <TeamComparison homeTeam={analysis.homeTeam} awayTeam={analysis.awayTeam} />
              </div>
            </div>

            <div className="card p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[#f0fdf4]">Player Movement Heatmap</h2>
                <div className="flex rounded-lg overflow-hidden border border-[#1c3020]">
                  <button
                    onClick={() => setActiveHeatmapTeam("home")}
                    className={`px-3 py-1.5 text-xs transition-colors ${
                      activeHeatmapTeam === "home" ? "bg-green-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                    }`}
                  >
                    {analysis.homeTeam.name}
                  </button>
                  <button
                    onClick={() => setActiveHeatmapTeam("away")}
                    className={`px-3 py-1.5 text-xs transition-colors ${
                      activeHeatmapTeam === "away" ? "bg-green-400 text-black font-medium" : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                    }`}
                  >
                    {analysis.awayTeam.name}
                  </button>
                </div>
              </div>
              <div className="max-w-2xl mx-auto">
                <Heatmap team={activeHeatmapTeam === "home" ? analysis.homeTeam : analysis.awayTeam} />
              </div>
            </div>

            <div className="card p-5">
              <div className="flex items-center gap-2 mb-5">
                <div className="w-8 h-8 rounded-lg bg-green-400/10 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-green-400" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[#f0fdf4]">Coaching Insights</h2>
                  <p className="text-xs text-[#6b9e6b]">AI-generated performance analysis and recommendations</p>
                </div>
              </div>
              <CoachingInsights
                insights={analysis.insights}
                homeTeamName={analysis.homeTeam.name}
                awayTeamName={analysis.awayTeam.name}
              />
            </div>
          </>
        )}
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
