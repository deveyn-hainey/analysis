"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import {
  Upload,
  Play,
  Activity,
  Users,
  Target,
  TrendingUp,
  ChevronRight,
  Film,
  Cpu,
  BarChart3,
} from "lucide-react";
import type { AnalyzeRequest, MatchAnalysis } from "@/lib/types";

const ACCEPTED_TYPES = ["video/mp4", "video/webm", "video/ogg", "video/quicktime"];

function extractFrames(
  file: File,
  intervalSeconds = 8,
  onProgress?: (pct: number) => void
): Promise<Array<{ base64: string; timestamp: number }>> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const frames: Array<{ base64: string; timestamp: number }> = [];

    video.preload = "metadata";
    video.muted = true;

    video.onloadedmetadata = () => {
      const duration = video.duration;
      const timestamps: number[] = [];
      for (let t = 0; t < duration; t += intervalSeconds) {
        timestamps.push(t);
      }

      canvas.width = 640;
      canvas.height = 360;

      let idx = 0;
      const seekNext = () => {
        if (idx >= timestamps.length) {
          URL.revokeObjectURL(video.src);
          resolve(frames);
          return;
        }
        video.currentTime = timestamps[idx];
      };

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0, 640, 360);
        const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
        frames.push({ base64, timestamp: timestamps[idx] });
        onProgress?.(Math.round(((idx + 1) / timestamps.length) * 50));
        idx++;
        seekNext();
      };

      video.onerror = () => reject(new Error("Failed to load video"));
      seekNext();
    };

    video.src = URL.createObjectURL(file);
  });
}

export default function HomePage() {
  const router = useRouter();
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<"idle" | "extracting" | "analyzing" | "done" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  const handleFile = useCallback(
    async (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setErrorMsg("Please upload an MP4, WebM, MOV, or OGG video file.");
        setStatus("error");
        return;
      }

      setStatus("extracting");
      setProgress(5);

      try {
        const frames = await extractFrames(file, 8, (pct) => setProgress(pct));
        setStatus("analyzing");
        setProgress(55);

        const payload: AnalyzeRequest = { frames };
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        setProgress(90);

        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: "Unknown error" }));
          throw new Error((err as { error?: string }).error ?? "Analysis failed");
        }

        const analysis = (await res.json()) as MatchAnalysis;
        sessionStorage.setItem("matchAnalysis", JSON.stringify(analysis));
        setProgress(100);
        setStatus("done");
        router.push("/dashboard");
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
        setStatus("error");
      }
    },
    [router]
  );

  const handleDemo = useCallback(async () => {
    setStatus("analyzing");
    setProgress(10);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ demo: true, frames: [] } satisfies AnalyzeRequest),
      });

      setProgress(90);

      if (!res.ok) throw new Error("Failed to load demo");
      const analysis = (await res.json()) as MatchAnalysis;
      sessionStorage.setItem("matchAnalysis", JSON.stringify(analysis));
      setProgress(100);
      setStatus("done");
      router.push("/dashboard");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("error");
    }
  }, [router]);

  const isProcessing = status === "extracting" || status === "analyzing";

  return (
    <div className="min-h-screen bg-[#0d1117]">
      {/* Nav */}
      <nav className="border-b border-[#30363d] px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <span className="font-semibold text-[#e6edf3]">SoccerVision</span>
            <span className="text-[#8b949e] text-sm hidden sm:block">Analytics</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#8b949e]">
            <span className="hidden sm:block">Powered by Claude Vision AI</span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-16 pb-8">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-4 py-1.5 text-emerald-400 text-sm mb-6">
            <Cpu className="w-3.5 h-3.5" />
            Computer Vision · Real-time AI Analysis
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-[#e6edf3] leading-tight mb-4">
            Turn match footage into
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-400">
              coaching intelligence
            </span>
          </h1>
          <p className="text-[#8b949e] text-lg max-w-2xl mx-auto">
            Upload a soccer match clip and our AI extracts player movements, detects key events, and delivers
            actionable performance insights in seconds.
          </p>
        </div>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {[
            { icon: Users, label: "Player tracking" },
            { icon: Target, label: "Shot detection" },
            { icon: BarChart3, label: "Possession stats" },
            { icon: TrendingUp, label: "Coaching insights" },
          ].map(({ icon: Icon, label }) => (
            <div
              key={label}
              className="flex items-center gap-2 bg-[#161b22] border border-[#30363d] rounded-full px-4 py-2 text-sm text-[#8b949e]"
            >
              <Icon className="w-3.5 h-3.5 text-emerald-400" />
              {label}
            </div>
          ))}
        </div>

        {/* Upload card */}
        <div className="max-w-2xl mx-auto">
          {status === "error" && (
            <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {errorMsg}
            </div>
          )}

          {isProcessing ? (
            <div className="card p-10 text-center">
              <div className="w-14 h-14 rounded-full bg-emerald-500/10 flex items-center justify-center mx-auto mb-5">
                <Cpu className="w-7 h-7 text-emerald-400 animate-pulse" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-[#e6edf3]">
                {status === "extracting" ? "Extracting frames…" : "Analysing with Claude Vision…"}
              </h3>
              <p className="text-[#8b949e] text-sm mb-6">
                {status === "extracting"
                  ? "Capturing keyframes from your video"
                  : "Detecting players, positions, and key events"}
              </p>
              <div className="h-2 bg-[#30363d] rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-[#8b949e] text-xs mt-2">{progress}%</p>
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files[0];
                if (file) handleFile(file);
              }}
              className={`card p-10 text-center cursor-pointer transition-colors ${
                dragOver ? "border-emerald-500/60 bg-emerald-500/5" : "hover:border-[#8b949e]"
              }`}
              onClick={() => document.getElementById("video-input")?.click()}
            >
              <input
                id="video-input"
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
              <div className="w-14 h-14 rounded-full bg-[#21262d] flex items-center justify-center mx-auto mb-5">
                <Upload className="w-6 h-6 text-[#8b949e]" />
              </div>
              <h3 className="text-lg font-semibold mb-1 text-[#e6edf3]">Drop your match video here</h3>
              <p className="text-[#8b949e] text-sm mb-6">MP4, WebM, MOV · 2–3 minute clip recommended</p>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-[#30363d]" />
                <span className="text-[#8b949e] text-xs">or</span>
                <div className="flex-1 h-px bg-[#30363d]" />
              </div>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDemo();
                }}
                className="mt-6 w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-400 transition-colors text-white font-medium py-3 px-6 rounded-xl"
              >
                <Play className="w-4 h-4" />
                View Live Demo
                <ChevronRight className="w-4 h-4" />
              </button>
              <p className="text-[#8b949e] text-xs mt-2">
                Loads pre-analysed match data — no API key required
              </p>
            </div>
          )}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 py-16 border-t border-[#30363d] mt-8">
        <h2 className="text-center text-2xl font-bold text-[#e6edf3] mb-10">How it works</h2>
        <div className="grid sm:grid-cols-3 gap-6">
          {[
            {
              step: "01",
              icon: Film,
              title: "Upload footage",
              desc: "Drop any 2–3 minute match clip. The browser extracts keyframes every 8 seconds for analysis.",
            },
            {
              step: "02",
              icon: Cpu,
              title: "AI vision analysis",
              desc: "Claude Vision identifies players, positions, and actions in each frame. Structured JSON is built from the results.",
            },
            {
              step: "03",
              icon: BarChart3,
              title: "Coaching dashboard",
              desc: "Insights are surfaced in an interactive dashboard — heatmaps, timelines, team stats, and prioritised recommendations.",
            },
          ].map(({ step, icon: Icon, title, desc }) => (
            <div key={step} className="card p-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center flex-shrink-0">
                  <Icon className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <div className="text-xs text-emerald-400 font-mono mb-1">{step}</div>
                  <h3 className="font-semibold text-[#e6edf3] mb-1">{title}</h3>
                  <p className="text-[#8b949e] text-sm leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Footer note about the field image */}
      <footer className="border-t border-[#30363d] py-6 text-center text-[#8b949e] text-xs">
        SoccerVision Analytics · Computer Vision POC · Built with Next.js & Claude AI
      </footer>
    </div>
  );
}
