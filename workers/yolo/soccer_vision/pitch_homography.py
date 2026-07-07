"""Optional pitch homography.

Projects image-space positions onto true pitch coordinates using a YOLO field
*keypoint* (pose) model plus the same homography mapping as the sibling
``Soccer_Analysis_Model`` repo. This is what lets the tactical board place a
zoomed/half-field shot in the correct part of the pitch instead of stretching it
across the whole board.

Fully optional and additive: if the ``sports`` package or the keypoint model
isn't available, ``attach_pitch_positions`` is a no-op and callers keep their
image-space coordinates (the ring overlay never uses any of this).

Setup:
    pip install sports huggingface_hub      # plus ultralytics (already required)
The keypoint model auto-downloads from HuggingFace (``Adit-jain/Soccana_Keypoint``)
on first use, or set ``KEYPOINT_MODEL_PATH`` to a local ``best.pt``. Disable with
``ENABLE_PITCH_HOMOGRAPHY=0``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("yolo-worker.pitch")

_ENABLED = os.getenv("ENABLE_PITCH_HOMOGRAPHY", "auto").lower()
_MODEL_PATH = os.getenv("KEYPOINT_MODEL_PATH", "").strip()
_HF_REPO = os.getenv("KEYPOINT_HF_REPO", "Adit-jain/Soccana_Keypoint")
_HF_FILE = os.getenv("KEYPOINT_HF_FILENAME", "Model/weights/best.pt")
_CONF = float(os.getenv("KEYPOINT_CONFIDENCE", "0.5"))
# The keypoint model costs ~4x the detector per frame, but the camera barely
# moves between consecutive dense-tracking frames. Recompute the homography
# only every Nth call and reuse the cached transformer in between. Set to 1 to
# recalibrate every frame.
_RECALIBRATE_EVERY = max(1, int(os.getenv("KEYPOINT_EVERY_N_FRAMES", "3")))

# Maps our 29 field keypoints (keypoint_constants order) to the sports-lib pitch
# vertex indices. Copied verbatim from
# Soccer_Analysis_Model/tactical_analysis/homography.py so the correspondences
# match what the keypoint model was trained to emit.
_OUR_TO_SPORTS = np.array([
    0, 1, 9, 4, 12, 2, 6, 3, 7, 5, 32, 13, 16, 14, 15, 33,
    24, 25, 17, 28, 20, 26, 22, 27, 23, 29, 34, 30, 31,
])
# Three extra reference points appended after the standard pitch vertices
# (left penalty-arc apex, center point, right penalty-arc apex), in cm.
_EXTRA_PITCH_POINTS = np.array([[2932, 3500], [6000, 3500], [9069, 3500]], dtype=np.float32)


class _Projector:
    def __init__(self) -> None:
        self._tried = False
        self._model = None
        self._view_transformer_cls = None
        self._all_pitch_points: Optional[np.ndarray] = None
        self._pitch_length = 12000.0  # cm (sports SoccerPitchConfiguration)
        self._pitch_width = 7000.0
        self.available = False
        self._cached_transformer = None
        self._calls_since_calibration = 0

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        if _ENABLED in ("0", "false", "off", "no"):
            logger.info("pitch homography disabled via ENABLE_PITCH_HOMOGRAPHY")
            return
        try:
            from sports.common.view import ViewTransformer
            from sports.configs.soccer import SoccerPitchConfiguration
            from ultralytics import YOLO

            cfg = SoccerPitchConfiguration()
            vertices = np.array(cfg.vertices, dtype=np.float32)
            self._all_pitch_points = np.concatenate((vertices, _EXTRA_PITCH_POINTS), axis=0)
            self._pitch_length = float(getattr(cfg, "length", 12000))
            self._pitch_width = float(getattr(cfg, "width", 7000))
            self._view_transformer_cls = ViewTransformer

            model_path = _MODEL_PATH or self._download_model()
            self._model = YOLO(model_path)
            self.available = True
            logger.info("pitch homography enabled (keypoint model: %s)", model_path)
        except Exception as exc:  # missing `sports`, model, network, etc.
            logger.warning("pitch homography unavailable — keeping image coords. (%s)", exc)

    def _download_model(self) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)

    def project(self, image, points_px: np.ndarray) -> Optional[np.ndarray]:
        """Project (N,2) image-pixel points onto pitch coords normalized to 0..100
        (x = length, left goal line = 0; y = width, top touchline = 0). Returns
        ``None`` when calibration isn't possible for this frame."""
        self._ensure()
        if not self.available or points_px is None or len(points_px) == 0:
            return None
        try:
            transformer = self._get_transformer(image)
            if transformer is None:
                return None
            projected = transformer.transform_points(points=np.asarray(points_px, dtype=np.float32))
            if projected is None:
                return None

            out = np.empty_like(projected, dtype=np.float32)
            out[:, 0] = np.clip(projected[:, 0] / self._pitch_length * 100.0, 0, 100)
            out[:, 1] = np.clip(projected[:, 1] / self._pitch_width * 100.0, 0, 100)
            return out
        except Exception as exc:
            logger.debug("pitch projection failed for a frame: %s", exc)
            return None

    def _get_transformer(self, image):
        """Return a frame→pitch ViewTransformer, recalibrating every Nth call."""
        self._calls_since_calibration += 1
        if (
            self._cached_transformer is not None
            and self._calls_since_calibration < _RECALIBRATE_EVERY
        ):
            return self._cached_transformer
        self._calls_since_calibration = 0

        from . import models as _models
        result = self._model.predict(image, verbose=False, conf=0.1, device=_models.DEVICE)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            self._cached_transformer = None
            return None
        detected = keypoints.data.cpu().numpy()[0]  # (29, 3): x, y, confidence
        mask = detected[:, 2] > _CONF
        if int(mask.sum()) < 4:
            self._cached_transformer = None  # need 4+ landmarks for a homography
            return None

        frame_points = detected[mask, :2].astype(np.float32)
        pitch_points = self._all_pitch_points[_OUR_TO_SPORTS[mask]].astype(np.float32)
        self._cached_transformer = self._view_transformer_cls(source=frame_points, target=pitch_points)
        return self._cached_transformer


_projector = _Projector()


def attach_pitch_positions(frame: dict, image) -> None:
    """Add normalized (0-100) ``pitchPosition`` to each player and ``pitchBall`` /
    ``pitchReferees`` to the frame, when calibration is available. No-op otherwise
    — the ring overlay / live tracking always uses image-space ``position``."""
    players = frame.get("players") or []
    ball = frame.get("ballPosition")
    referees = frame.get("referees") or []
    if not players and not ball and not referees:
        return

    width, height = image.size
    points: list[list[float]] = []
    refs: list[tuple[str, int]] = []
    for i, player in enumerate(players):
        pos = player.get("position")
        if not pos:
            continue
        points.append([pos["x"] / 100.0 * width, pos["y"] / 100.0 * height])
        refs.append(("player", i))
    if ball:
        points.append([ball["x"] / 100.0 * width, ball["y"] / 100.0 * height])
        refs.append(("ball", -1))
    for i, ref in enumerate(referees):
        points.append([ref["x"] / 100.0 * width, ref["y"] / 100.0 * height])
        refs.append(("ref", i))
    if not points:
        return

    projected = _projector.project(image, np.asarray(points, dtype=np.float32))
    if projected is None:
        return

    pitch_referees: list[dict[str, float]] = []
    for (kind, idx), (x, y) in zip(refs, projected):
        coord = {"x": round(float(x), 1), "y": round(float(y), 1)}
        if kind == "player":
            players[idx]["pitchPosition"] = coord
        elif kind == "ball":
            frame["pitchBall"] = coord
        else:
            pitch_referees.append(coord)
    if pitch_referees:
        frame["pitchReferees"] = pitch_referees
