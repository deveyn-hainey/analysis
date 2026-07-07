import type { FrameData } from "@/lib/types";

type Status = "idle" | "loading" | "ready" | "error";

export interface DenseProgress {
  stage: string;
  framesDone: number;
  framesTotal: number;
  percent: number | null;
}

let _frames: FrameData[] = [];
let _status: Status = "idle";
let _progress: DenseProgress | null = null;
let _error: string | null = null;
const _listeners = new Set<() => void>();

function notify() {
  _listeners.forEach((fn) => fn());
}

export const denseFrameStore = {
  getFrames(): FrameData[] {
    return _frames;
  },
  getStatus(): Status {
    return _status;
  },
  getProgress(): DenseProgress | null {
    return _progress;
  },
  getError(): string | null {
    return _error;
  },
  setLoading(progress?: DenseProgress | null) {
    if (_status !== "loading") _frames = [];
    _status = "loading";
    _progress = progress ?? null;
    _error = null;
    notify();
  },
  setReady(frames: FrameData[]) {
    _frames = frames;
    _status = "ready";
    _progress = null;
    _error = null;
    notify();
  },
  setError(message?: string) {
    _status = "error";
    _progress = null;
    _error = message ?? null;
    notify();
  },
  clear() {
    _frames = [];
    _status = "idle";
    _progress = null;
    _error = null;
    notify();
  },
  subscribe(fn: () => void): () => void {
    _listeners.add(fn);
    return () => _listeners.delete(fn);
  },
};
