"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
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
import type { AnalyzeFrameRequest, FrameData, MatchAnalysis } from "@/lib/types";
import { videoStore } from "@/lib/videoStore";

const ACCEPTED_TYPES = ["video/mp4", "video/webm", "video/ogg", "video/quicktime"];

function frameInterval(durationSeconds: number): number {
  return Math.max(2, Math.min(8, Math.round(durationSeconds / 16)));
}

function extractFrames(
  file: File,
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
      const interval = frameInterval(duration);
      const timestamps: number[] = [];
      for (let t = 0; t < duration; t += interval) {
        timestamps.push(+t.toFixed(1));
      }

      canvas.width = 640;
      canvas.height = 360;

      let idx = 0;
      const seekNext = () => {
        if (idx >= timestamps.length) {
          resolve(frames);
          return;
        }
        video.currentTime = timestamps[idx];
      };

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0, 640, 360);
        const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
        frames.push({ base64, timestamp: timestamps[idx] });
        onProgress?.(Math.round(((idx + 1) / timestamps.length) * 30));
        idx++;
        seekNext();
      };

      video.onerror = () => reject(new Error("Failed to load video"));
      seekNext();
    };

    // Store the object URL before setting src — dashboard will read it
    const objectUrl = URL.createObjectURL(file);
    videoStore.set(objectUrl);
    video.src = objectUrl;
  });
}

export default function HomePage() {
  const router = useRouter();
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<"idle" | "extracting" | "analyzing" | "summarizing" | "done" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [statusDetail, setStatusDetail] = useState("");
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
      setStatusDetail("Reading video…");

      try {
        const rawFrames = await extractFrames(file, (pct) => {
          setProgress(5 + pct);
          setStatusDetail("Capturing keyframes…");
        });

        setStatus("analyzing");
        setStatusDetail(`Sending ${rawFrames.length} frames to AI…`);
        setProgress(36);

        let completed = 0;
        const analyzedFrames = await Promise.all(
          rawFrames.map(async (rawFrame, i) => {
            const prev = i > 0 ? rawFrames[i - 1] : undefined;
            const payload: AnalyzeFrameRequest = {
              base64: rawFrame.base64,
              timestamp: rawFrame.timestamp,
              frameIndex: i,
              prevBase64: prev?.base64,
              prevTimestamp: prev?.timestamp,
            };
            try {
              const res = await fetch("/api/analyze/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              if (!res.ok) {
                const err = await res.json().catch(() => ({ error: "Unknown" }));
                throw new Error((err as { error?: string }).error ?? "Frame failed");
              }
              const data = (await res.json()) as FrameData;
              completed++;
              setProgress(36 + Math.round((completed / rawFrames.length) * 50));
              setStatusDetail(`Analysed ${completed} of ${rawFrames.length} frames…`);
              return data;
            } catch {
              completed++;
              setProgress(36 + Math.round((completed / rawFrames.length) * 50));
              setStatusDetail(`Analysed ${completed} of ${rawFrames.length} frames…`);
              return {
                frameIndex: i,
                timestamp: rawFrame.timestamp,
                players: [],
                events: [],
                possession: "contested" as const,
              } as FrameData;
            }
          })
        );

        setStatus("summarizing");
        setStatusDetail("Building match insights…");
        setProgress(88);

        const sumRes = await fetch("/api/analyze/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames: analyzedFrames }),
        });

        if (!sumRes.ok) {
          const err = await sumRes.json().catch(() => ({ error: "Unknown error" }));
          throw new Error((err as { error?: string }).error ?? "Summarize failed");
        }

        const analysis = (await sumRes.json()) as MatchAnalysis;
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
    videoStore.clear();
    setStatus("analyzing");
    setProgress(10);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ demo: true, frames: [] }),
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

  const isProcessing = status === "extracting" || status === "analyzing" || status === "summarizing";

  return (
    <div className="min-h-screen bg-[#070e07]">
      {/* Nav */}
      <nav className="border-b border-[#1c3020] px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-green-400/15 flex items-center justify-center">
              <Activity className="w-4 h-4 text-green-400" />
            </div>
            <span className="font-semibold text-[#f0fdf4]">SoccerVision</span>
            <span className="text-[#6b9e6b] text-sm hidden sm:block">Analytics</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#6b9e6b]">
            <span className="hidden sm:block">Powered by Claude AI</span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-16 pb-8">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 bg-green-400/10 border border-green-400/20 rounded-full px-4 py-1.5 text-green-400 text-sm mb-6">
            <Cpu className="w-3.5 h-3.5" />
            Computer Vision · AI Coaching Analysis
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-[#f0fdf4] leading-tight mb-4">
            Turn match footage into
            <br />
            <span className="text-green-400">
              coaching intelligence
            </span>
          </h1>
          <p className="text-[#6b9e6b] text-lg max-w-2xl mx-auto">
            Upload a soccer match clip. AI extracts player movements, detects key events, and delivers
            actionable coaching insights in plain English.
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
              className="flex items-center gap-2 bg-[#0d1a0d] border border-[#1c3020] rounded-full px-4 py-2 text-sm text-[#6b9e6b]"
            >
              <Icon className="w-3.5 h-3.5 text-green-400" />
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
              <div className="w-14 h-14 rounded-full bg-green-400/10 flex items-center justify-center mx-auto mb-5">
                <Cpu className="w-7 h-7 text-green-400 animate-pulse" />
              </div>
              <h3 className="text-lg font-semibold mb-1 text-[#f0fdf4]">
                {status === "extracting"
                  ? "Reading video…"
                  : status === "summarizing"
                  ? "Building insights…"
                  : "Analysing with AI Vision"}
              </h3>
              <p className="text-[#6b9e6b] text-sm mb-6 min-h-[20px]">{statusDetail}</p>
              <div className="h-2 bg-[#1c3020] rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-400 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-[#6b9e6b] text-xs mt-2">{progress}%</p>
            </div>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files[0];
                if (file) handleFile(file);
              }}
              className={`card p-10 text-center cursor-pointer transition-colors ${
                dragOver ? "border-green-400/60 bg-green-400/5" : "hover:border-[#2d4a30]"
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
              <div className="w-14 h-14 rounded-full bg-[#142014] flex items-center justify-center mx-auto mb-5">
                <Upload className="w-6 h-6 text-[#6b9e6b]" />
              </div>
              <h3 className="text-lg font-semibold mb-1 text-[#f0fdf4]">Drop your match video here</h3>
              <p className="text-[#6b9e6b] text-sm mb-6">MP4, WebM, MOV · 2–3 minute clip recommended</p>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-[#1c3020]" />
                <span className="text-[#6b9e6b] text-xs">or</span>
                <div className="flex-1 h-px bg-[#1c3020]" />
              </div>

              <button
                onClick={(e) => { e.stopPropagation(); handleDemo(); }}
                className="mt-6 w-full flex items-center justify-center gap-2 bg-green-400 hover:bg-green-300 transition-colors text-black font-semibold py-3 px-6 rounded-xl"
              >
                <Play className="w-4 h-4" />
                View Live Demo
                <ChevronRight className="w-4 h-4" />
              </button>
              <p className="text-[#6b9e6b] text-xs mt-2">
                Loads pre-analysed match data — no API key required
              </p>
            </div>
          )}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-6xl mx-auto px-6 py-16 border-t border-[#1c3020] mt-8">
        <h2 className="text-center text-2xl font-bold text-[#f0fdf4] mb-10">How it works</h2>
        <div className="grid sm:grid-cols-3 gap-6">
          {[
            {
              step: "01",
              icon: Film,
              title: "Upload footage",
              desc: "Drop any match clip. The browser automatically extracts keyframes — no full video ever leaves your device until analysis begins.",
            },
            {
              step: "02",
              icon: Cpu,
              title: "AI vision analysis",
              desc: "Each frame is analysed for player positions, ball location, and key events. Goals, shots, passes, and set pieces are all detected.",
            },
            {
              step: "03",
              icon: BarChart3,
              title: "Coaching dashboard",
              desc: "Switch between Coach and Analyst views. Click any event to jump to that moment in your video. Export structured data for downstream tools.",
            },
          ].map(({ step, icon: Icon, title, desc }) => (
            <div key={step} className="card p-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-green-400/10 flex items-center justify-center flex-shrink-0">
                  <Icon className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <div className="text-xs text-green-400 font-mono mb-1">{step}</div>
                  <h3 className="font-semibold text-[#f0fdf4] mb-1">{title}</h3>
                  <p className="text-[#6b9e6b] text-sm leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-[#1c3020] py-6 text-center text-[#6b9e6b] text-xs">
        SoccerVision Analytics · Computer Vision POC · Built with Next.js & Claude AI
      </footer>
    </div>
  );
}
