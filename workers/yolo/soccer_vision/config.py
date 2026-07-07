"""All environment-tunable configuration for the vision worker.

Every knob lives here so behaviour differences between deployments are always
explained by this one file plus the environment, never by scattered getenv
calls.
"""
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("yolo_worker")
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

WORKER_DIR = Path(__file__).resolve().parent.parent

# ── Model selection ──────────────────────────────────────────────────────────
# Unset YOLO_MODEL_PATH now means "use the fine-tuned soccana model" (local
# checkpoint if present, HF Hub download otherwise) rather than generic COCO
# yolo11n — see models.resolve_default_model_path(). The COCO model has no
# ball/referee classes worth trusting on broadcast footage and was the main
# reason detection quality was poor out of the box.
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "").strip()
MODEL_BACKEND = os.getenv("YOLO_BACKEND", "auto").lower()
# For ultralytics checkpoints hosted on the HF Hub but not packaged for the
# `ultralyticsplus`/`from_pretrained` style, set YOLO_MODEL_PATH to the repo id
# and YOLO_HF_FILENAME to the .pt path inside that repo.
HF_FILENAME = os.getenv("YOLO_HF_FILENAME")
SOCCANA_HF_REPO = os.getenv("YOLO_DEFAULT_HF_REPO", "Adit-jain/soccana")
SOCCANA_HF_FILENAME = os.getenv("YOLO_DEFAULT_HF_FILENAME", "Model/weights/best.pt")

# ── Detection thresholds ─────────────────────────────────────────────────────
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
# The ball is a small, fast-moving object in wide broadcast shots; a missed ball
# starves almost every downstream signal, so it gets its own permissive floor.
BALL_CONFIDENCE = float(os.getenv("YOLO_BALL_CONFIDENCE", "0.1"))
# 1280 matches soccana's training imgsz; Ultralytics' 640 default silently
# downscales the already-tiny ball before the model ever sees it.
IMAGE_SIZE = int(os.getenv("YOLO_IMGSZ", "1280"))
# Inference device: "auto" picks cuda > mps (Apple Silicon GPU) > cpu.
DEVICE = os.getenv("YOLO_DEVICE", "auto").lower()

# ── Class name mapping ───────────────────────────────────────────────────────
PLAYER_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_PLAYER_CLASSES", "person,player,goalkeeper").split(",")}
BALL_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_BALL_CLASSES", "sports ball,ball").split(",")}
REFEREE_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_REFEREE_CLASSES", "referee").split(",")}

# ── Team assignment ──────────────────────────────────────────────────────────
HOME_KIT_COLOR = os.getenv("YOLO_HOME_KIT_COLOR", "").strip().lower()
AWAY_KIT_COLOR = os.getenv("YOLO_AWAY_KIT_COLOR", "").strip().lower()
MAX_PLAYERS_PER_TEAM = int(os.getenv("YOLO_MAX_PLAYERS_PER_TEAM", "11"))
# "siglip" trains a SigLIP+UMAP+KMeans classifier per clip (stronger, needs
# transformers/umap-learn); "hsv" is the lightweight histogram clustering.
TEAM_BACKEND = os.getenv("YOLO_TEAM_BACKEND", "siglip").lower()

# ── Tracking / smoothing ─────────────────────────────────────────────────────
DEFAULT_DENSE_FPS = float(os.getenv("YOLO_DENSE_FPS", "15"))
# BoT-SORT + ReID keeps player IDs stable through occlusions; set YOLO_TRACKER
# to bytetrack_soccer.yaml to fall back to the lighter IoU-only tracker.
TRACKER_CONFIG = os.getenv("YOLO_TRACKER", str(WORKER_DIR / "botsort_soccer.yaml"))
TRACK_SMOOTHING_ALPHA = float(os.getenv("YOLO_TRACK_SMOOTHING_ALPHA", "0.35"))
BALL_INTERPOLATION_LIMIT = int(os.getenv("YOLO_BALL_INTERPOLATION_LIMIT", "30"))
# Max consecutive frames a player can be missing before we stop coasting their
# position. ~0.7s at 15fps — long enough to bridge occlusion blips, short enough
# not to leave a ghost where a player has actually left the frame.
PLAYER_INTERPOLATION_LIMIT = int(os.getenv("YOLO_PLAYER_INTERPOLATION_LIMIT", "10"))
TRACK_DIAGNOSTICS = os.getenv("YOLO_TRACK_DIAGNOSTICS", "0") in ("1", "true", "True")
TRACK_DIAGNOSTIC_EVERY = max(1, int(os.getenv("YOLO_TRACK_DIAGNOSTIC_EVERY", "1")))

# ── Pitch masking ────────────────────────────────────────────────────────────
# Green-pitch HSV gate. OpenCV hue is 0-180; grass sits roughly 25-95 across
# sunlit/shadowed/yellowed pitches.
PITCH_HSV_LOWER = np.array(
    [int(v) for v in os.getenv("YOLO_PITCH_HSV_LOWER", "25,25,25").split(",")], dtype=np.uint8
)
PITCH_HSV_UPPER = np.array(
    [int(v) for v in os.getenv("YOLO_PITCH_HSV_UPPER", "95,255,255").split(",")], dtype=np.uint8
)
# If the largest green region covers less than this fraction of the frame we
# assume it isn't a wide pitch shot (replay closeup/cutaway).
PITCH_MIN_AREA_FRAC = float(os.getenv("YOLO_PITCH_MIN_AREA_FRAC", "0.10"))
PITCH_FILTER_ENABLED = os.getenv("YOLO_PITCH_FILTER", "1") not in ("0", "false", "False")
REQUIRE_PITCH_VIEW = os.getenv("YOLO_REQUIRE_PITCH_VIEW", "1") not in ("0", "false", "False")

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
