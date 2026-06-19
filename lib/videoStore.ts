let _objectUrl: string | null = null;

export const videoStore = {
  // Sets the active video URL without revoking the previous one — the match
  // library owns URLs across multiple entries and revokes them on removal.
  set(url: string | null) {
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
