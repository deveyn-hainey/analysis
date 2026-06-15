const _frames = new Map<number, string>();

export const frameImageStore = {
  set(index: number, base64: string) {
    _frames.set(index, base64);
  },
  get(index: number): string | null {
    return _frames.get(index) ?? null;
  },
  clear() {
    _frames.clear();
  },
};
