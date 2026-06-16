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
import type { AnalyzeEventsRequest, AnalyzeFrameRequest, FrameData, MatchAnalysis, MatchEvent } from "@/lib/types";
import { videoStore } from "@/lib/videoStore";
import { frameImageStore } from "@/lib/frameImageStore";

const ACCEPTED_TYPES = ["video/mp4", "video/webm", "video/ogg", "video/quicktime"];
const FRAME_ANALYSIS_CONCURRENCY = 4;
const MAX_FRAME_RETRIES = 1;
const MAX_FAILED_FRAME_RATIO = 0.3;
const SEND_PREVIOUS_FRAME_CONTEXT = false;
const VISION_WORKER_URL = process.env.NEXT_PUBLIC_VISION_WORKER_URL?.replace(/\/$/, "");
// Frames per /api/analyze/events request. Matches the server-side batch size so
// each client request maps to exactly one Claude call on the server side.
const EVENT_REVIEW_CLIENT_BATCH = 8;
// Two concurrent event-review requests roughly halves wall-clock time vs sequential.
const EVENT_REVIEW_CLIENT_CONCURRENCY = 2;

type RawFrame = { base64: string; timestamp: number };

function frameInterval(durationSeconds: number): number {
  // Capped at 8s before, which could step right over an entire goal sequence
  // (crossing the line + celebration + scoreboard update is often a 2-4s window)
  // between two samples. Tighter cap trades more frames/cost for not missing it.
  return Math.max(2, Math.min(4, Math.round(durationSeconds / 20)));
}

function extractFrames(
  file: File,
  onProgress?: (pct: number) => void
): Promise<RawFrame[]> {
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

      // 640×360 was throwing away exactly the resolution needed to see the ball:
      // a ball that's ~15px wide in a 1920×1080 broadcast source shrinks to ~5px,
      // well below what any detector (YOLO or Claude) can reliably pick out — and
      // the soccana model was trained at 1280px besides. Every "ball not detected"
      // review note traces back to this.
      canvas.width = 1280;
      canvas.height = 720;

      let idx = 0;
      const seekNext = () => {
        if (idx >= timestamps.length) {
          resolve(frames);
          return;
        }
        video.currentTime = timestamps[idx];
      };

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0, 1280, 720);
        const base64 = canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
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

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function emptyFrame(rawFrame: RawFrame, frameIndex: number): FrameData {
  return {
    frameIndex,
    timestamp: rawFrame.timestamp,
    players: [],
    events: [],
    possession: "contested",
  };
}

async function analyzeFrame(payload: AnalyzeFrameRequest): Promise<FrameData> {
  const res = await fetch("/api/analyze/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Frame analysis failed" }));
    throw new Error((err as { error?: string }).error ?? "Frame analysis failed");
  }

  return res.json() as Promise<FrameData>;
}

async function analyzeFrameWithRetry(payload: AnalyzeFrameRequest): Promise<FrameData> {
  for (let attempt = 0; attempt <= MAX_FRAME_RETRIES; attempt++) {
    try {
      return await analyzeFrame(payload);
    } catch (err) {
      if (attempt === MAX_FRAME_RETRIES) throw err;
      await sleep(600 * (attempt + 1));
    }
  }

  throw new Error("Frame analysis failed");
}

async function analyzeFrames(
  rawFrames: RawFrame[],
  onProgress: (completed: number, failed: number) => void
): Promise<{ frames: FrameData[]; failed: number }> {
  const results: Array<FrameData | undefined> = new Array(rawFrames.length);
  let nextIndex = 0;
  let completed = 0;
  let failed = 0;

  async function worker() {
    while (nextIndex < rawFrames.length) {
      const i = nextIndex;
      nextIndex++;

      const rawFrame = rawFrames[i];
      const prev = i > 0 ? rawFrames[i - 1] : undefined;
      const payload: AnalyzeFrameRequest = {
        base64: rawFrame.base64,
        timestamp: rawFrame.timestamp,
        frameIndex: i,
        prevBase64: SEND_PREVIOUS_FRAME_CONTEXT ? prev?.base64 : undefined,
        prevTimestamp: SEND_PREVIOUS_FRAME_CONTEXT ? prev?.timestamp : undefined,
      };

      try {
        results[i] = await analyzeFrameWithRetry(payload);
      } catch {
        failed++;
        results[i] = emptyFrame(rawFrame, i);
      } finally {
        completed++;
        onProgress(completed, failed);
      }
    }
  }

  const workerCount = Math.min(FRAME_ANALYSIS_CONCURRENCY, rawFrames.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  return {
    frames: results.map((frame, i) => frame ?? emptyFrame(rawFrames[i], i)),
    failed,
  };
}

async function analyzeFramesWithWorker(rawFrames: RawFrame[]): Promise<FrameData[]> {
  if (!VISION_WORKER_URL) {
    throw new Error("Vision worker is not configured");
  }

  const res = await fetch(`${VISION_WORKER_URL}/analyze-frames`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames: rawFrames }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Vision worker analysis failed" }));
    throw new Error((err as { error?: string }).error ?? "Vision worker analysis failed");
  }

  const data = (await res.json()) as { frames?: FrameData[] };
  if (!data.frames || data.frames.length === 0) {
    throw new Error("Vision worker returned no frames");
  }

  return data.frames;
}

// Sends scoreboard reads across every frame and synthesises goal events wherever
// the score increased — matches the server-side logic so cross-batch goals are
// caught after all client batches have been merged.
function synthesizeGoalsFromScoreboard(frames: FrameData[]): FrameData[] {
  const sorted = [...frames].sort((a, b) => a.timestamp - b.timestamp);
  const runningMax: Record<string, number> = { home: 0, away: 0 };
  const additions = new Map<number, MatchEvent[]>();

  for (const frame of sorted) {
    if (!frame.scoreboard) continue;
    for (const team of ["home", "away"] as const) {
      const seen = frame.scoreboard![team];
      if (typeof seen === "number" && Number.isFinite(seen) && seen > runningMax[team]) {
        const ev: MatchEvent = {
          id: `f${frame.frameIndex}-scoreboard-goal-${team}-${seen}`,
          timestamp: frame.timestamp,
          type: "goal",
          team,
          description: `Scoreboard read ${frame.scoreboard!.home}-${frame.scoreboard!.away} — ${team === "home" ? "Home" : "Away"} goal confirmed from score overlay`,
          confidence: 0.92,
          isKeyMoment: true,
          evidenceUsed: [`scoreboard read ${seen} for ${team}, up from ${runningMax[team]} previously observed`],
        };
        additions.set(frame.frameIndex, [...(additions.get(frame.frameIndex) ?? []), ev]);
        runningMax[team] = seen;
      }
    }
  }

  if (additions.size === 0) return frames;
  return frames.map((f) =>
    additions.has(f.frameIndex)
      ? { ...f, events: deduplicateEvents([...f.events, ...additions.get(f.frameIndex)!]) }
      : f
  );
}

function deduplicateEvents(events: MatchEvent[]): MatchEvent[] {
  const seen = new Set<string>();
  return events.filter((e) => {
    if (seen.has(e.id)) return false;
    seen.add(e.id);
    return true;
  });
}

// When a scoreboard is readable anywhere in the clip, it is the ground truth for
// goals. Visual/cluster-based goal detections are kept only when there is no
// scoreboard at all — otherwise they double-count (one from Claude review, one from
// scoreboard synthesis) and inflate the score.
function preferScoreboardGoals(frames: FrameData[]): FrameData[] {
  const hasScoreboard = frames.some((f) => f.scoreboard != null);
  if (!hasScoreboard) return frames; // no scoreboard anywhere — keep visual goals as-is

  return frames.map((f) => ({
    ...f,
    events: f.events.filter(
      (e) => e.type !== "goal" || e.id.includes("-scoreboard-goal-")
    ),
  }));
}

// Resize frames to a smaller resolution for event review. Claude doesn't need
// full 1280×720 to detect celebrations, ball-in-net, or player actions — and
// sending half-resolution images cuts Claude payload size by ~60%.
function resizeFramesForReview(
  frames: RawFrame[],
  width = 960,
  height = 540,
  quality = 0.78
): Promise<RawFrame[]> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;

  return Promise.all(
    frames.map(
      (f) =>
        new Promise<RawFrame>((resolve) => {
          const img = new Image();
          img.onload = () => {
            ctx.drawImage(img, 0, 0, width, height);
            resolve({
              base64: canvas.toDataURL("image/jpeg", quality).split(",")[1],
              timestamp: f.timestamp,
            });
          };
          img.src = `data:image/jpeg;base64,${f.base64}`;
        })
    )
  );
}

async function reviewEventsWithClaude(
  rawFrames: RawFrame[],
  frames: FrameData[],
  onProgress?: (completed: number, total: number) => void
): Promise<{ frames: FrameData[]; warnings: string[] }> {
  // Downscale images before sending — Claude needs enough detail to see
  // celebrations and ball-in-net, but not full 1280×720 YOLO resolution.
  // 960×540 at 0.78 quality cuts payload ~60% and speeds up each API call.
  const reviewImages = await resizeFramesForReview(rawFrames);

  const totalBatches = Math.ceil(reviewImages.length / EVENT_REVIEW_CLIENT_BATCH);
  const allWarnings: string[] = [];
  const resultFrames: FrameData[] = [...frames];
  let nextBatch = 0;

  async function worker() {
    while (nextBatch < totalBatches) {
      const batchIdx = nextBatch++;
      const start = batchIdx * EVENT_REVIEW_CLIENT_BATCH;
      const batchImages = reviewImages.slice(start, start + EVENT_REVIEW_CLIENT_BATCH);
      const batchFrames = frames.slice(start, start + EVENT_REVIEW_CLIENT_BATCH);

      const payload: AnalyzeEventsRequest = { images: batchImages, frames: batchFrames };

      try {
        const res = await fetch("/api/analyze/events", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data = (await res.json()) as { frames?: FrameData[]; warnings?: string[] };
          if (data.frames) {
            data.frames.forEach((frame, i) => { resultFrames[start + i] = frame; });
          }
          if (data.warnings) allWarnings.push(...data.warnings);
        } else {
          const err = await res.json().catch(() => ({ error: "Event review failed" }));
          allWarnings.push(`Batch ${batchIdx + 1}/${totalBatches} failed: ${(err as { error?: string }).error ?? "unknown error"}`);
        }
      } catch (err) {
        allWarnings.push(`Batch ${batchIdx + 1}/${totalBatches} failed: ${err instanceof Error ? err.message : "unknown"}`);
      }

      onProgress?.(batchIdx + 1, totalBatches);
    }
  }

  const concurrency = Math.min(EVENT_REVIEW_CLIENT_CONCURRENCY, totalBatches);
  await Promise.all(Array.from({ length: concurrency }, () => worker()));

  // Global scoreboard synthesis across all merged frames catches cross-batch goals
  // (e.g. goal happened in batch 1 frames but scoreboard only visible in batch 2).
  const withScoreboardGoals = synthesizeGoalsFromScoreboard(resultFrames);

  // If a scoreboard was readable anywhere, it is the ground truth — remove visual
  // goal events so they don't double-count alongside scoreboard-synthesized ones.
  return { frames: preferScoreboardGoals(withScoreboardGoals), warnings: allWarnings };
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

        if (rawFrames.length === 0) {
          throw new Error("No frames could be extracted from this video. Please try a longer clip.");
        }

        // Store images so the dashboard can render actual-frame overlays
        frameImageStore.clear();
        rawFrames.forEach((f, i) => frameImageStore.set(i, f.base64));

        setStatus("analyzing");
        setStatusDetail(
          VISION_WORKER_URL
            ? `Analysing ${rawFrames.length} frames with YOLO worker…`
            : `Analysing ${rawFrames.length} frames with AI…`
        );
        setProgress(36);

        let analyzedFrames: FrameData[];
        let eventReviewWarnings: string[] = [];
        if (VISION_WORKER_URL) {
          try {
            analyzedFrames = await analyzeFramesWithWorker(rawFrames);
            setProgress(72);
            setStatusDetail("Reviewing YOLO detections for match events…");
            const totalBatches = Math.ceil(rawFrames.length / EVENT_REVIEW_CLIENT_BATCH);
            const reviewed = await reviewEventsWithClaude(
              rawFrames,
              analyzedFrames,
              (completed, total) => {
                setProgress(72 + Math.round((completed / total) * 14));
                setStatusDetail(`Reviewing events: batch ${completed}/${total}…`);
              }
            );
            analyzedFrames = reviewed.frames;
            eventReviewWarnings = reviewed.warnings;
            setProgress(86);
            setStatusDetail(`Reviewed ${analyzedFrames.length} frames across ${totalBatches} batches…`);
          } catch (err) {
            const detail = err instanceof Error ? err.message : "Unknown worker error";
            throw new Error(
              `YOLO worker failed at ${VISION_WORKER_URL}. Make sure the worker terminal is still running, then retry. Details: ${detail}`
            );
          }
        } else {
          const claudeResult = await analyzeFrames(
            rawFrames,
            (completed, failedFrames) => {
              setProgress(36 + Math.round((completed / rawFrames.length) * 50));
              setStatusDetail(
                failedFrames > 0
                  ? `Analysed ${completed} of ${rawFrames.length} frames (${failedFrames} retried and skipped)…`
                  : `Analysed ${completed} of ${rawFrames.length} frames…`
              );
            }
          );

          if (claudeResult.failed / rawFrames.length > MAX_FAILED_FRAME_RATIO) {
            throw new Error(
              `AI analysis failed for ${claudeResult.failed} of ${rawFrames.length} frames. Please try again with a shorter or clearer clip.`
            );
          }
          analyzedFrames = claudeResult.frames;
        }

        setStatus("summarizing");
        setStatusDetail("Building match insights…");
        setProgress(88);

        const sumRes = await fetch("/api/analyze/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames: analyzedFrames, eventReviewWarnings }),
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
