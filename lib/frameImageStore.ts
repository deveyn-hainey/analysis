const _frames = new Map<number, string>();

export const frameImageStore = {
  set(index: number, base64: string) {
    _frames.set(index, base64);
  },
  get(index: number): string | null {
    return _frames.get(index) ?? null;
  },
  // Swap in a whole match's frame images at once (used when switching matches).
  replaceAll(map: Map<number, string>) {
    _frames.clear();
    for (const [index, base64] of map) _frames.set(index, base64);
  },
  clear() {
    _frames.clear();
  },
};
