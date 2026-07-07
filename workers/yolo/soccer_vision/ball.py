"""Ball selection and recovery.

Two problems with naive per-frame max-confidence ball picking on broadcast
footage:

1. False positives — a white boot, a bald head, or a crowd blob can out-score
   the real ball for a frame, teleporting the ball across the pitch.
   ``BallTracker.select`` penalises candidates by distance from the last
   accepted position so the most *plausible* ball wins, not just the most
   confident box.
2. Misses — the ball is a handful of pixels in a wide shot and YOLO loses it
   during fast passes. ``BallTracker.recover`` re-runs the detector on a
   zoomed crop around the last known position, where the ball is many times
   larger relative to the input, and maps the hit back to full-frame coords.
"""
import os
from typing import Optional

import numpy as np
from PIL import Image

from . import config
from .config import logger
from .schemas import Detection

BALL_RECOVERY_ENABLED = os.getenv("YOLO_BALL_RECOVERY", "1") not in ("0", "false", "False")
# Crop side as a fraction of the frame's larger dimension.
BALL_RECOVERY_CROP_FRAC = float(os.getenv("YOLO_BALL_RECOVERY_CROP_FRAC", "0.25"))
# Stop attempting recovery after this many consecutive misses — the last known
# position is stale and the crop would just be a random patch of pitch.
BALL_RECOVERY_MAX_MISSES = int(os.getenv("YOLO_BALL_RECOVERY_MAX_MISSES", "8"))
# Distance penalty: a candidate loses this much confidence per full frame
# diagonal of distance from the last accepted ball position.
BALL_DISTANCE_PENALTY = float(os.getenv("YOLO_BALL_DISTANCE_PENALTY", "1.5"))
# A jump farther than this fraction of the frame diagonal in one step needs
# clearly higher confidence to be believed.
BALL_MAX_PLAUSIBLE_JUMP = float(os.getenv("YOLO_BALL_MAX_PLAUSIBLE_JUMP", "0.35"))


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BallTracker:
    """Per-clip stateful ball chooser. Create one per request/video."""

    def __init__(self) -> None:
        self.last_center: Optional[tuple[float, float]] = None
        self.misses = 0

    def select(
        self,
        ball_detections: list[Detection],
        width: int,
        height: int,
    ) -> Optional[Detection]:
        """Pick the most plausible ball detection, or None."""
        if not ball_detections:
            return None
        diagonal = float(np.hypot(width, height))
        if self.last_center is None:
            best = max(ball_detections, key=lambda d: d.confidence)
        else:
            def score(d: Detection) -> float:
                cx, cy = _center(d.xyxy)
                dist = float(np.hypot(cx - self.last_center[0], cy - self.last_center[1]))
                penalty = BALL_DISTANCE_PENALTY * (dist / diagonal)
                # Recent track → trust proximity more; stale track → trust confidence.
                staleness = min(1.0, self.misses / max(1, BALL_RECOVERY_MAX_MISSES))
                return d.confidence - penalty * (1.0 - staleness)

            best = max(ball_detections, key=score)
            cx, cy = _center(best.xyxy)
            jump = float(np.hypot(cx - self.last_center[0], cy - self.last_center[1]))
            if jump > BALL_MAX_PLAUSIBLE_JUMP * diagonal and best.confidence < 2 * config.BALL_CONFIDENCE:
                # Low-confidence teleport across the frame: more likely a false
                # positive than the real ball. Treat as a miss and let
                # interpolation / recovery handle it.
                return None

        self._accept(_center(best.xyxy))
        return best

    def recover(self, image: Image.Image) -> Optional[Detection]:
        """Re-detect the ball on a zoomed crop around the last known position."""
        if (
            not BALL_RECOVERY_ENABLED
            or self.last_center is None
            or self.misses > BALL_RECOVERY_MAX_MISSES
        ):
            self.misses += 1
            return None
        self.misses += 1

        from . import models
        if models.use_huggingface_yolov5():
            return None

        width, height = image.size
        side = max(64, int(max(width, height) * BALL_RECOVERY_CROP_FRAC))
        cx, cy = self.last_center
        x1 = int(max(0, min(width - side, cx - side / 2)))
        y1 = int(max(0, min(height - side, cy - side / 2)))
        crop = image.crop((x1, y1, x1 + side, y1 + side))

        try:
            result = models.model.predict(
                source=np.asarray(crop),
                conf=config.BALL_CONFIDENCE * 0.8,
                imgsz=640,
                device=models.DEVICE,
                verbose=False,
            )[0]
        except Exception as exc:
            logger.debug("ball recovery pass failed: %s", exc)
            return None

        names = result.names
        best: Optional[Detection] = None
        for box in result.boxes:
            cls_name = str(names.get(int(box.cls[0]), "")).lower()
            if cls_name not in config.BALL_CLASSES:
                continue
            confidence = float(box.conf[0])
            bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
            candidate = Detection(
                cls_name=cls_name,
                confidence=confidence,
                xyxy=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
            )
            if best is None or confidence > best.confidence:
                best = candidate

        if best is not None:
            self._accept(_center(best.xyxy))
        return best

    def _accept(self, center: tuple[float, float]) -> None:
        self.last_center = center
        self.misses = 0
