import type { FrameData } from "@/lib/types";

type Status = "idle" | "loading" | "ready" | "error";

let _frames: FrameData[] = [];
let _status: Status = "idle";
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
  setLoading() {
    _frames = [];
    _status = "loading";
    notify();
  },
  setReady(frames: FrameData[]) {
    _frames = frames;
    _status = "ready";
    notify();
  },
  setError() {
    _status = "error";
    notify();
  },
  clear() {
    _frames = [];
    _status = "idle";
    notify();
  },
  subscribe(fn: () => void): () => void {
    _listeners.add(fn);
    return () => _listeners.delete(fn);
  },
};
