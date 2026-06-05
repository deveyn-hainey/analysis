"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Activity,
  Users,
  Film,
  Trophy,
  ChevronLeft,
  Clock,
  Download,
  Cpu,
} from "lucide-react";
import type { MatchAnalysis, FrameData } from "@/lib/types";
import SoccerField from "@/components/SoccerField";
import EventTimeline from "@/components/EventTimeline";
import StatsChart from "@/components/StatsChart";
import TeamComparison from "@/components/TeamComparison";
import CoachingInsights from "@/components/CoachingInsights";
import Heatmap from "@/components/Heatmap";
import MetricCard from "@/components/MetricCard";

function formatDuration(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<FrameData | null>(null);
  const [activeHeatmapTeam, setActiveHeatmapTeam] = useState<"home" | "away">("home");

  useEffect(() => {
    const stored = sessionStorage.getItem("matchAnalysis");
    if (stored) {
      try {
        const data = JSON.parse(stored) as MatchAnalysis;
        setAnalysis(data);
        if (data.frames.length > 0) {
          setSelectedFrame(data.frames[0]);
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
      <div className="min-h-screen bg-[#0d1117] flex items-center justify-center">
        <div className="flex items-center gap-3 text-[#8b949e]">
          <Cpu className="w-5 h-5 animate-pulse text-emerald-400" />
          <span>Loading analysis…</span>
        </div>
      </div>
    );
  }

  const currentFrame = selectedFrame ?? analysis.frames[0];

  const handleTimelineSelect = (timestamp: number) => {
    const closest = analysis.frames.reduce((best, f) =>
      Math.abs(f.timestamp - timestamp) < Math.abs(best.timestamp - timestamp) ? f : best
    );
    setSelectedFrame(closest);
  };

  return (
    <div className="min-h-screen bg-[#0d1117]">
      {/* Nav */}
      <nav className="border-b border-[#30363d] px-6 py-3 sticky top-0 z-10 bg-[#0d1117]/95 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 text-[#8b949e] hover:text-[#e6edf3] transition-colors text-sm"
            >
              <ChevronLeft className="w-4 h-4" />
              New analysis
            </button>
            <div className="h-4 w-px bg-[#30363d]" />
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-emerald-500/10 flex items-center justify-center">
                <Activity className="w-3 h-3 text-emerald-400" />
              </div>
              <span className="font-semibold text-sm text-[#e6edf3]">Match Analysis Dashboard</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {analysis.processingMethod === "demo" && (
              <span className="text-xs bg-yellow-400/10 text-yellow-400 px-2 py-1 rounded-full border border-yellow-400/20">
                Demo data
              </span>
            )}
            {analysis.processingMethod === "ai" && (
              <span className="text-xs bg-emerald-400/10 text-emerald-400 px-2 py-1 rounded-full border border-emerald-400/20">
                AI processed
              </span>
            )}
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `match-analysis-${analysis.id}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="flex items-center gap-1.5 text-xs text-[#8b949e] hover:text-[#e6edf3] border border-[#30363d] hover:border-[#8b949e] transition-colors px-3 py-1.5 rounded-lg"
            >
              <Download className="w-3.5 h-3.5" />
              Export JSON
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Match header */}
        <div className="card p-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-8">
              {/* Home team */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                  <span className="text-blue-400 font-bold text-sm">EFC</span>
                </div>
                <div>
                  <div className="font-bold text-[#e6edf3]">{analysis.homeTeam.name}</div>
                  <div className="text-xs text-[#8b949e]">Home · {analysis.homeTeam.formation}</div>
                </div>
              </div>

              {/* Score */}
              <div className="text-center">
                <div className="text-3xl font-bold text-[#e6edf3] font-mono">
                  {analysis.score.home} — {analysis.score.away}
                </div>
                <div className="text-xs text-[#8b949e] mt-0.5">Final</div>
              </div>

              {/* Away team */}
              <div className="flex items-center gap-3">
                <div>
                  <div className="font-bold text-[#e6edf3] text-right">{analysis.awayTeam.name}</div>
                  <div className="text-xs text-[#8b949e] text-right">Away · {analysis.awayTeam.formation}</div>
                </div>
                <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                  <span className="text-red-400 font-bold text-sm">CU</span>
                </div>
              </div>
            </div>

            {/* Meta */}
            <div className="flex items-center gap-4 text-xs text-[#8b949e]">
              <div className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                {formatDuration(analysis.videoDuration)} analysed
              </div>
              <div className="flex items-center gap-1.5">
                <Film className="w-3.5 h-3.5" />
                {analysis.framesAnalyzed} frames
              </div>
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" />
                {new Date(analysis.processedAt).toLocaleDateString()}
              </div>
            </div>
          </div>
        </div>

        {/* KPI row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            icon={Activity}
            label="Home possession"
            value={`${analysis.homeTeam.stats.possession}%`}
            sub="vs 33% away"
            accent="blue"
          />
          <MetricCard
            icon={Trophy}
            label="Key events"
            value={analysis.keyEvents.length}
            sub={`${analysis.keyEvents.filter((e) => e.type === "goal").length} goal(s) detected`}
            accent="yellow"
          />
          <MetricCard
            icon={Users}
            label="Players tracked"
            value={currentFrame?.players.length ?? 0}
            sub="across all frames"
            accent="emerald"
          />
          <MetricCard
            icon={Film}
            label="Frames analysed"
            value={analysis.framesAnalyzed}
            sub={`every ~${Math.round(analysis.videoDuration / Math.max(analysis.framesAnalyzed, 1))}s interval`}
            accent="emerald"
          />
        </div>

        {/* Field + Timeline row */}
        <div className="grid lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 card p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[#e6edf3]">Player Positions</h2>
              <div className="flex items-center gap-2">
                {analysis.frames.map((f, i) => (
                  <button
                    key={f.frameIndex}
                    onClick={() => setSelectedFrame(f)}
                    className={`text-xs px-2 py-1 rounded transition-colors ${
                      selectedFrame?.frameIndex === f.frameIndex
                        ? "bg-emerald-500 text-white"
                        : "bg-[#21262d] text-[#8b949e] hover:bg-[#30363d]"
                    }`}
                  >
                    {Math.floor(f.timestamp / 60)}:{String(Math.floor(f.timestamp % 60)).padStart(2, "0")}
                  </button>
                ))}
              </div>
            </div>
            {currentFrame && <SoccerField frame={currentFrame} />}
          </div>

          <div className="lg:col-span-2 card p-4">
            <h2 className="text-sm font-semibold text-[#e6edf3] mb-3">Event Timeline</h2>
            <EventTimeline
              events={analysis.keyEvents}
              selectedTimestamp={selectedFrame?.timestamp}
              onSelect={handleTimelineSelect}
            />
          </div>
        </div>

        {/* Stats + Comparison row */}
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="card p-5">
            <h2 className="text-sm font-semibold text-[#e6edf3] mb-4">Action Statistics</h2>
            <StatsChart homeTeam={analysis.homeTeam} awayTeam={analysis.awayTeam} />
          </div>
          <div className="card p-5">
            <h2 className="text-sm font-semibold text-[#e6edf3] mb-4">Team Comparison</h2>
            <TeamComparison homeTeam={analysis.homeTeam} awayTeam={analysis.awayTeam} />
          </div>
        </div>

        {/* Heatmaps row */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-[#e6edf3]">Player Movement Heatmap</h2>
            <div className="flex rounded-lg overflow-hidden border border-[#30363d]">
              <button
                onClick={() => setActiveHeatmapTeam("home")}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  activeHeatmapTeam === "home"
                    ? "bg-blue-500 text-white"
                    : "text-[#8b949e] hover:text-[#e6edf3]"
                }`}
              >
                {analysis.homeTeam.name}
              </button>
              <button
                onClick={() => setActiveHeatmapTeam("away")}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  activeHeatmapTeam === "away"
                    ? "bg-red-500 text-white"
                    : "text-[#8b949e] hover:text-[#e6edf3]"
                }`}
              >
                {analysis.awayTeam.name}
              </button>
            </div>
          </div>
          <div className="max-w-2xl mx-auto">
            <Heatmap
              team={activeHeatmapTeam === "home" ? analysis.homeTeam : analysis.awayTeam}
            />
          </div>
        </div>

        {/* Coaching insights */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-[#e6edf3]">Coaching Staff Insights</h2>
              <p className="text-xs text-[#8b949e]">AI-generated performance analysis and recommendations</p>
            </div>
          </div>
          <CoachingInsights
            insights={analysis.insights}
            homeTeamName={analysis.homeTeam.name}
            awayTeamName={analysis.awayTeam.name}
          />
        </div>
      </main>

      <footer className="border-t border-[#30363d] py-5 text-center text-[#8b949e] text-xs mt-8">
        SoccerVision Analytics · Analysis ID: {analysis.id} · Processed {new Date(analysis.processedAt).toLocaleString()}
      </footer>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#0d1117] flex items-center justify-center">
          <Cpu className="w-6 h-6 animate-pulse text-emerald-400" />
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
