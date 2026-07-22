"""SoccerVision worker package.

Structure mirrors the sibling Soccer_Analysis_Model repo:

- ``config``        — every env-tunable knob in one place
- ``models``        — detector loading (fine-tuned soccana by default)
- ``schemas``       — request/response models and shared types
- ``geometry``      — pure coordinate math helpers
- ``pitch_mask``    — green-pitch gating of person detections
- ``detection``     — YOLO inference → Detection lists
- ``teams``         — jersey-colour team assignment (clustering + kit anchoring)
- ``tracking``      — stable ID assignment, pruning, possession smoothing
- ``postprocess``   — clip-level interpolation/consolidation passes
- ``frames``        — per-frame analysis assembly
- ``pitch_homography`` — keypoint-model homography onto true pitch coords
- ``api``           — FastAPI app wiring
"""
