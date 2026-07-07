"""Green-pitch masking: reject person detections that aren't on the field."""
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from . import config
from .schemas import Detection


def compute_pitch_mask(image: Image.Image) -> Optional[np.ndarray]:
    """Return a filled binary mask of the playing surface, or None if no clear pitch.

    The pitch is the single largest contiguous green region. Filling its contour
    means players, lines and the centre circle (which are not green) still count as
    "on pitch", while the crowd, dugouts, sidelines and advertising boards — the
    main source of phantom player detections from a generic COCO model — fall
    outside it and get rejected downstream.
    """
    if not config.PITCH_FILTER_ENABLED:
        return None
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, config.PITCH_HSV_LOWER, config.PITCH_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < config.PITCH_MIN_AREA_FRAC * mask.size:
        return None

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)
    # Dilate so players standing right on the touchline aren't clipped out.
    return cv2.dilate(filled, kernel)


def foot_on_pitch(box: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    """True if the detection's foot point (bottom-centre of box) lands on the pitch."""
    x1, y1, x2, y2 = box
    fx = int((x1 + x2) / 2)
    fy = int(y2)
    h, w = mask.shape[:2]
    fx = max(0, min(w - 1, fx))
    fy = max(0, min(h - 1, fy))
    return bool(mask[fy, fx])


def filter_persons_to_pitch(
    image: Image.Image,
    detections: list[Detection],
    pitch_mask: Optional[np.ndarray] = None,
) -> list[Detection]:
    """Drop person detections whose feet aren't on the pitch; keep ball/others as-is.

    This is the main defence against over-tracking: a generic COCO model reports
    every spectator and bench player as a "person", and those flickering detections
    are what make ByteTrack's IDs churn and the overlay flash. Removing them leaves
    a stable ~22-player set that tracks cleanly.
    """
    mask = pitch_mask if pitch_mask is not None else compute_pitch_mask(image)
    if mask is None and not config.REQUIRE_PITCH_VIEW:
        return detections
    if mask is None:
        return [
            d for d in detections
            if d.cls_name not in config.PLAYER_CLASSES and d.cls_name not in config.REFEREE_CLASSES
        ]
    kept: list[Detection] = []
    for d in detections:
        if d.cls_name in config.PLAYER_CLASSES and d.cls_name not in config.BALL_CLASSES:
            if not foot_on_pitch(d.xyxy, mask):
                continue
        kept.append(d)
    return kept
