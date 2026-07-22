"""Jersey-colour team assignment.

Two strategies, in priority order:
1. Known-kit anchoring — when both kit colours are supplied (request field or
   YOLO_*_KIT_COLOR env), cluster then label clusters by distance to the
   canonical colour signatures so "home" is deterministic across clips.
2. Unsupervised K-means (k=2) on 48-dim jersey HSV histograms — with optional
   clip-global centroids computed during the calibration pass so a single noisy
   frame can't flip the split.
"""
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from . import config
from .config import logger
from .schemas import Detection, TeamId


def crop_jersey_features(image: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray:
    """Return a normalised HSV histogram of the upper-body (jersey) region.

    Using the upper 55% of the bounding box captures the shirt while excluding
    shorts and legs, whose colours are far more uniform and less discriminative.
    A 16-bin histogram per channel (H, S, V) gives a 48-dim feature vector that
    is robust to brightness variation — two differently-lit white jerseys will
    cluster together, unlike a single brightness mean which would drift.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    jersey_y2 = y1 + max(1, int((y2 - y1) * 0.55))
    crop_rgb = np.asarray(image.crop((x1, y1, x2, jersey_y2)))
    if crop_rgb.size == 0:
        return np.zeros(48, dtype=np.float32)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    total = hist.sum()
    return hist / (total + 1e-6)


KIT_COLOR_HSV: dict[str, tuple[float, float, float]] = {
    "red": (0.0, 0.9, 0.8),
    "blue": (112.0, 0.85, 0.75),
    "navy": (118.0, 0.75, 0.35),
    "white": (0.0, 0.08, 0.92),
    "black": (0.0, 0.08, 0.12),
    "yellow": (30.0, 0.85, 0.9),
    "orange": (18.0, 0.85, 0.85),
    "green": (60.0, 0.75, 0.55),
    "purple": (140.0, 0.7, 0.65),
}


def _validate_kit_config() -> None:
    configured = [color for color in (config.HOME_KIT_COLOR, config.AWAY_KIT_COLOR) if color]
    unsupported = [color for color in configured if color not in KIT_COLOR_HSV]
    if unsupported:
        logger.warning(
            "Unsupported YOLO_*_KIT_COLOR value(s) %s. Supported colors are %s. "
            "Team assignment will fall back to unsupervised clustering when either side is unsupported.",
            unsupported,
            sorted(KIT_COLOR_HSV.keys()),
        )
        return
    if config.HOME_KIT_COLOR and config.AWAY_KIT_COLOR:
        logger.info(
            "team-color: home kit=%s renders as home/red; away kit=%s renders as away/blue",
            config.HOME_KIT_COLOR,
            config.AWAY_KIT_COLOR,
        )
    elif config.HOME_KIT_COLOR or config.AWAY_KIT_COLOR:
        logger.warning(
            "Both YOLO_HOME_KIT_COLOR and YOLO_AWAY_KIT_COLOR are required for anchored team assignment. "
            "Received home=%r away=%r; falling back to unsupervised clustering.",
            config.HOME_KIT_COLOR,
            config.AWAY_KIT_COLOR,
        )


_validate_kit_config()


def hsv_to_circular_signature(hue: float, sat: float, val: float) -> np.ndarray:
    radians = (hue / 180.0) * 2.0 * np.pi
    return np.array([np.cos(radians) * sat, np.sin(radians) * sat, sat, val], dtype=np.float32)


def normalize_kit_color(color_name: Optional[str]) -> str:
    color = (color_name or "").strip().lower()
    return "" if color in ("", "auto", "none", "unknown") else color


def request_kit_colors(home_color: Optional[str] = None, away_color: Optional[str] = None) -> tuple[str, str]:
    home = normalize_kit_color(home_color) if home_color is not None else config.HOME_KIT_COLOR
    away = normalize_kit_color(away_color) if away_color is not None else config.AWAY_KIT_COLOR
    return home, away


def kit_color_signature(color_name: Optional[str]) -> Optional[np.ndarray]:
    hsv = KIT_COLOR_HSV.get(normalize_kit_color(color_name))
    if not hsv:
        return None
    return hsv_to_circular_signature(*hsv)


def crop_jersey_color_signature(image: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray:
    """Return a compact colour signature for known-kit anchoring.

    K-means can separate two kit clusters, but its labels are arbitrary: cluster 0
    may be USA in one clip and Paraguay in the next. This signature uses the same
    upper-body crop, removes obvious grass, and represents hue circularly so red
    near hue 0/180 does not split across the boundary.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    jersey_y2 = y1 + max(1, int((y2 - y1) * 0.60))
    crop_rgb = np.asarray(image.crop((x1, y1, x2, jersey_y2)))
    if crop_rgb.size == 0:
        return np.zeros(4, dtype=np.float32)

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    # Prefer saturated shirt pixels and exclude pitch-green spill/background.
    non_grass = ~((hue >= 25) & (hue <= 95) & (sat > 0.25))
    chroma = (sat > 0.18) & (val > 0.18) & non_grass
    mask = chroma if int(chroma.sum()) >= 12 else (val > 0.12)
    if int(mask.sum()) == 0:
        return np.zeros(4, dtype=np.float32)

    selected_hue = hue[mask]
    selected_sat = sat[mask]
    selected_val = val[mask]
    radians = (selected_hue / 180.0) * 2.0 * np.pi
    mean_sat = float(np.mean(selected_sat))
    return np.array(
        [
            float(np.mean(np.cos(radians) * selected_sat)),
            float(np.mean(np.sin(radians) * selected_sat)),
            mean_sat,
            float(np.mean(selected_val)),
        ],
        dtype=np.float32,
    )


def kmeans_two(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(features).astype(float)

    dists_from_first = np.linalg.norm(X - X[0], axis=1)
    seed1_idx = int(dists_from_first.argmax())
    centroids = np.stack([X[0], X[seed1_idx]])

    labels = np.zeros(len(features), dtype=int)
    for _ in range(10):
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centroids], axis=1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for label in (0, 1):
            members = X[labels == label]
            if len(members):
                centroids[label] = members.mean(axis=0)

    return labels, centroids


def assign_teams_by_known_kits(
    signatures: list[np.ndarray],
    home_kit_color: Optional[str] = None,
    away_kit_color: Optional[str] = None,
) -> Optional[list[TeamId]]:
    home_color, away_color = request_kit_colors(home_kit_color, away_kit_color)
    home_sig = kit_color_signature(home_color)
    away_sig = kit_color_signature(away_color)
    if home_sig is None or away_sig is None or not signatures:
        return None

    if len(signatures) >= 4:
        labels, centroids = kmeans_two(signatures)
        same_cost = float(np.linalg.norm(centroids[0] - home_sig) + np.linalg.norm(centroids[1] - away_sig))
        swap_cost = float(np.linalg.norm(centroids[0] - away_sig) + np.linalg.norm(centroids[1] - home_sig))
        label_to_team: dict[int, TeamId] = {0: "home", 1: "away"} if same_cost <= swap_cost else {0: "away", 1: "home"}
        teams = [label_to_team[int(label)] for label in labels]
        margin = abs(same_cost - swap_cost)
    else:
        teams = []
        margins: list[float] = []
        for sig in signatures:
            home_dist = float(np.linalg.norm(sig - home_sig))
            away_dist = float(np.linalg.norm(sig - away_sig))
            teams.append("home" if home_dist <= away_dist else "away")
            margins.append(abs(home_dist - away_dist))
        margin = float(np.mean(margins)) if margins else 0.0

    if config.TRACK_DIAGNOSTICS:
        logger.debug(
            "team-color: anchored home=%s away=%s count=%d avg_margin=%.3f",
            home_color,
            away_color,
            len(teams),
            margin,
        )

    return teams


def split_teams(features: list[np.ndarray]) -> list[TeamId]:
    """Assign team labels via K-means (k=2) on jersey HSV histogram features.

    Initialises the two centroids by picking the pair of detections that are
    furthest apart in feature space, which is more stable than a brightness-based
    seed when both teams wear similarly dark or light kits.
    """
    if not features:
        return []
    if len(features) == 1:
        return ["home"]

    labels, _ = kmeans_two(features)

    return ["home" if label == 0 else "away" for label in labels]


def assign_teams(
    image: Image.Image,
    person_detections: list[Detection],
    global_centroids: Optional[tuple[np.ndarray, np.ndarray]] = None,
    home_kit_color: Optional[str] = None,
    away_kit_color: Optional[str] = None,
) -> list[TeamId]:
    features = [crop_jersey_features(image, d.xyxy) for d in person_detections]
    home_color, away_color = request_kit_colors(home_kit_color, away_kit_color)

    if home_color and away_color:
        signatures = [crop_jersey_color_signature(image, d.xyxy) for d in person_detections]
        known_kit_teams = assign_teams_by_known_kits(signatures, home_color, away_color)
        if known_kit_teams is not None:
            return known_kit_teams

    if global_centroids and features:
        home_c, away_c = global_centroids
        return [
            "home" if np.linalg.norm(f - home_c) <= np.linalg.norm(f - away_c) else "away"
            for f in features
        ]

    return split_teams(features)
