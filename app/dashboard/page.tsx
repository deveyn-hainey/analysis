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
import type { MatchAnalysis, FrameData } from "@/lib/types";
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

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

type ViewMode = "coach" | "player";

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<FrameData | null>(null);
  const [activeHeatmapTeam, setActiveHeatmapTeam] = useState<"home" | "away">("home");
  const [viewMode, setViewMode] = useState<ViewMode>("coach");
  const [pitchView, setPitchView] = useState<"frame" | "tactical">("tactical");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const url = videoStore.get();
    setVideoUrl(url);
  }, []);

  // Keep the tactical field in sync with video playback — as the video plays,
  // find the nearest sampled frame and update the player/ball overlay.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !analysis) return;
    const onTimeUpdate = () => {
      const t = video.currentTime;
      const closest = analysis.frames.reduce((best, f) =>
        Math.abs(f.timestamp - t) < Math.abs(best.timestamp - t) ? f : best
      );
      setSelectedFrame(closest);
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    return () => video.removeEventListener("timeupdate", onTimeUpdate);
  }, [analysis]);

  useEffect(() => {
    const stored = sessionStorage.getItem("matchAnalysis");
    if (stored) {
      try {
        const data = JSON.parse(stored) as MatchAnalysis;
        setAnalysis(data);
        if (data.frames.length > 0) setSelectedFrame(data.frames[0]);
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
  const frameImage = currentFrame ? frameImageStore.get(currentFrame.frameIndex) : null;

  const seekTo = (timestamp: number) => {
    const closest = analysis.frames.reduce((best, f) =>
      Math.abs(f.timestamp - timestamp) < Math.abs(best.timestamp - timestamp) ? f : best
    );
    setSelectedFrame(closest);
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

  // Shared video+tactical / frame-by-frame panel used in both view modes.
  const pitchPanel = (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div>
          <h2 className="text-sm font-semibold text-[#f0fdf4]">
            {pitchView === "tactical" ? "Video + Tactical" : "Frame View"}
          </h2>
          {pitchView === "tactical" && videoUrl && (
            <p className="text-xs text-[#6b9e6b] mt-0.5">
              Tactical field syncs to video playback
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
        <div className="space-y-3">
          {videoUrl ? (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                className="w-full rounded-lg"
                style={{ maxHeight: 240 }}
              />
              <p className="text-xs text-[#6b9e6b]">
                Click an event in the timeline to jump · field updates as video plays
              </p>
            </>
          ) : (
            <div className="border border-dashed border-[#1c3020] rounded-lg flex items-center justify-center gap-3 text-[#6b9e6b] text-xs py-5">
              <Film className="w-4 h-4" />
              Upload a clip to enable video playback and live tracking
            </div>
          )}
          {currentFrame && <SoccerField frame={currentFrame} />}
        </div>
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
            {/* Coach / Player toggle */}
            <div className="flex rounded-lg border border-[#1c3020] overflow-hidden text-xs">
              <button
                onClick={() => setViewMode("coach")}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  viewMode === "coach"
                    ? "bg-green-400 text-black"
                    : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                }`}
              >
                Coach
              </button>
              <button
                onClick={() => setViewMode("player")}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  viewMode === "player"
                    ? "bg-green-400 text-black"
                    : "text-[#6b9e6b] hover:text-[#f0fdf4]"
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

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
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
            {/* Insights first and prominent */}
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

            {/* Video + Pitch + Timeline */}
            <div className="grid lg:grid-cols-5 gap-4">
              <div className="lg:col-span-3">
                {pitchPanel}
              </div>

              <div className="lg:col-span-2 card p-4">
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

            {/* Quick stats */}
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
            {/* KPI row */}
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

            {/* Video + Pitch + Timeline */}
            <div className="grid lg:grid-cols-5 gap-4">
              <div className="lg:col-span-3">
                {pitchPanel}
              </div>

              <div className="lg:col-span-2 card p-4">
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

            {/* Stats + Comparison */}
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

            {/* Heatmap */}
            <div className="card p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-[#f0fdf4]">Player Movement Heatmap</h2>
                <div className="flex rounded-lg overflow-hidden border border-[#1c3020]">
                  <button
                    onClick={() => setActiveHeatmapTeam("home")}
                    className={`px-3 py-1.5 text-xs transition-colors ${
                      activeHeatmapTeam === "home"
                        ? "bg-green-400 text-black font-medium"
                        : "text-[#6b9e6b] hover:text-[#f0fdf4]"
                    }`}
                  >
                    {analysis.homeTeam.name}
                  </button>
                  <button
                    onClick={() => setActiveHeatmapTeam("away")}
                    className={`px-3 py-1.5 text-xs transition-colors ${
                      activeHeatmapTeam === "away"
                        ? "bg-green-400 text-black font-medium"
                        : "text-[#6b9e6b] hover:text-[#f0fdf4]"
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

            {/* Coaching insights */}
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
