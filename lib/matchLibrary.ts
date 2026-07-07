import type { FrameData, MatchAnalysis } from "@/lib/types";
import { videoStore } from "@/lib/videoStore";
import { frameImageStore } from "@/lib/frameImageStore";
import { denseFrameStore } from "@/lib/denseFrameStore";

export type DenseStatus = "idle" | "loading" | "ready" | "error";

// One analyzed match held in memory for the session. Each entry owns its own
// video URL, per-frame images, and dense tracking, so switching between matches
// restores the right data. (In-memory only — cleared on full page reload.)
export interface MatchEntry {
  id: string;
  title: string;
  analysis: MatchAnalysis;
  videoUrl: string | null;
  frameImages: Map<number, string>;
  denseFrames: FrameData[];
  denseStatus: DenseStatus;
  createdAt: number;
}

const _entries: MatchEntry[] = [];
let _activeId: string | null = null;
const _listeners = new Set<() => void>();
const notify = () => _listeners.forEach((fn) => fn());

function withNearestPitchViews(denseFrames: FrameData[], sparseFrames: FrameData[]): FrameData[] {
  const views = sparseFrames
    .filter((frame) => frame.pitchView)
    .sort((a, b) => a.timestamp - b.timestamp);
  if (views.length === 0) return denseFrames;

  return denseFrames.map((frame) => {
    if (frame.pitchView || frame.pitchBall || frame.players.some((player) => player.pitchPosition)) {
      return frame;
    }
    const nearest = views.reduce((best, candidate) =>
      Math.abs(candidate.timestamp - frame.timestamp) < Math.abs(best.timestamp - frame.timestamp)
        ? candidate
        : best
    );
    return nearest.pitchView ? { ...frame, pitchView: nearest.pitchView } : frame;
  });
}

// Load an entry into the shared "active view" buffers the dashboard reads from.
function hydrate(entry: MatchEntry) {
  videoStore.set(entry.videoUrl);
  frameImageStore.replaceAll(entry.frameImages);
  if (entry.denseStatus === "ready") denseFrameStore.setReady(entry.denseFrames);
  else if (entry.denseStatus === "loading") denseFrameStore.setLoading();
  else if (entry.denseStatus === "error") denseFrameStore.setError();
  else denseFrameStore.clear();
}

export const matchLibrary = {
  list: () => _entries.slice(),
  active: () => _entries.find((e) => e.id === _activeId) ?? null,
  activeId: () => _activeId,

  add(entry: MatchEntry) {
    _entries.push(entry);
    _activeId = entry.id;
    hydrate(entry);
    notify();
  },

  setActive(id: string) {
    const entry = _entries.find((e) => e.id === id);
    if (!entry || _activeId === id) return;
    _activeId = id;
    hydrate(entry);
    notify();
  },

  remove(id: string) {
    const i = _entries.findIndex((e) => e.id === id);
    if (i < 0) return;
    const [removed] = _entries.splice(i, 1);
    if (removed.videoUrl) URL.revokeObjectURL(removed.videoUrl);
    if (_activeId === id) {
      const next = _entries[i] ?? _entries[i - 1] ?? null;
      _activeId = next?.id ?? null;
      if (next) hydrate(next);
      else {
        videoStore.clear();
        frameImageStore.clear();
        denseFrameStore.clear();
      }
    }
    notify();
  },

  // Background dense-tracking result for a specific match. Only touches the shared
  // dense buffer when that match is the one currently being viewed.
  setDense(id: string, frames: FrameData[], status: DenseStatus) {
    const entry = _entries.find((e) => e.id === id);
    if (!entry) return;
    entry.denseFrames = status === "ready" ? withNearestPitchViews(frames, entry.analysis.frames) : frames;
    entry.denseStatus = status;
    if (_activeId === id) {
      if (status === "ready") denseFrameStore.setReady(entry.denseFrames);
      else if (status === "loading") denseFrameStore.setLoading();
      else if (status === "error") denseFrameStore.setError();
      else denseFrameStore.clear();
    }
    notify();
  },

  subscribe(fn: () => void) {
    _listeners.add(fn);
    return () => {
      _listeners.delete(fn);
    };
  },
};
