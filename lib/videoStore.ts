let _objectUrl: string | null = null;

export const videoStore = {
  set(url: string) {
    if (_objectUrl) URL.revokeObjectURL(_objectUrl);
    _objectUrl = url;
  },
  get(): string | null {
    return _objectUrl;
  },
  clear() {
    if (_objectUrl) URL.revokeObjectURL(_objectUrl);
    _objectUrl = null;
  },
};
