import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

logger = logging.getLogger("yolo_worker")
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
MODEL_BACKEND = os.getenv("YOLO_BACKEND", "auto").lower()
# For ultralytics checkpoints hosted on the HF Hub but not packaged for the
# `ultralyticsplus`/`from_pretrained` style (e.g. Adit-jain/soccana, a YOLOv11n model
# uploaded as a plain repo file rather than via Ultralytics' own HF integration), set
# YOLO_MODEL_PATH to the repo id and YOLO_HF_FILENAME to the path of the .pt file
# inside that repo. We resolve it to a local path via huggingface_hub before handing
# it to ultralytics.YOLO() — this avoids the `ultralyticsplus` package, which as of
# writing pins ultralytics==8.0.239 and silently fails on YOLO11 architectures.
HF_FILENAME = os.getenv("YOLO_HF_FILENAME")
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
# The ball is a small, fast-moving object in wide broadcast shots, and the default
# generic (non-soccer-trained) model misses it far more often than it misses players.
# A missed ball starves almost every downstream candidate/event signal, so it gets its
# own, more permissive threshold instead of sharing the player confidence cutoff.
BALL_CONFIDENCE = float(os.getenv("YOLO_BALL_CONFIDENCE", "0.1"))
# Inference resolution for the ultralytics backend. 1280 matches soccana's training
# imgsz; override down for speed on CPU if a frame is already small, or up for even
# tinier balls on high-res source video.
IMAGE_SIZE = int(os.getenv("YOLO_IMGSZ", "1280"))
PLAYER_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_PLAYER_CLASSES", "person,player,goalkeeper").split(",")}
BALL_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_BALL_CLASSES", "sports ball,ball").split(",")}
# NOTE: with the default COCO-pretrained model (yolo11n.pt) there is no "referee" class,
# so this filter is a no-op until YOLO_MODEL_PATH points at a soccer-fine-tuned model
# that actually emits one (see docs/vision-architecture.md). Referees currently fall
# into PLAYER_CLASSES as plain "person" detections and get fed into team clustering.
REFEREE_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_REFEREE_CLASSES", "referee").split(",")}
DEFAULT_DENSE_FPS = float(os.getenv("YOLO_DENSE_FPS", "15"))
TRACK_SMOOTHING_ALPHA = float(os.getenv("YOLO_TRACK_SMOOTHING_ALPHA", "0.35"))
BALL_INTERPOLATION_LIMIT = int(os.getenv("YOLO_BALL_INTERPOLATION_LIMIT", "30"))
# Max consecutive frames a player can be missing before we stop coasting their
# position. ~0.7s at 15fps — long enough to bridge occlusion blips, short enough
# not to leave a ghost where a player has actually left the frame.
PLAYER_INTERPOLATION_LIMIT = int(os.getenv("YOLO_PLAYER_INTERPOLATION_LIMIT", "10"))
MAX_PLAYERS_PER_TEAM = int(os.getenv("YOLO_MAX_PLAYERS_PER_TEAM", "11"))
TRACK_DIAGNOSTICS = os.getenv("YOLO_TRACK_DIAGNOSTICS", "0") in ("1", "true", "True")
TRACK_DIAGNOSTIC_EVERY = max(1, int(os.getenv("YOLO_TRACK_DIAGNOSTIC_EVERY", "1")))

app = FastAPI(title="SoccerVision YOLO Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def use_huggingface_yolov5() -> bool:
    if MODEL_BACKEND == "yolov5":
        return True
    if MODEL_BACKEND == "ultralytics":
        return False
    if HF_FILENAME:
        # YOLO_HF_FILENAME only makes sense for an ultralytics checkpoint resolved via
        # huggingface_hub — without this check the "repo/name" shape of MODEL_PATH would
        # otherwise trip the yolov5 auto-detect heuristic below.
        return False
    return "/" in MODEL_PATH and not MODEL_PATH.endswith(".pt")


def resolve_ultralytics_model_path() -> str:
    if not HF_FILENAME:
        return MODEL_PATH
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=MODEL_PATH, filename=HF_FILENAME)


if use_huggingface_yolov5():
    import yolov5

    model = yolov5.load(MODEL_PATH)
    model.conf = min(CONFIDENCE, BALL_CONFIDENCE)
else:
    model = YOLO(resolve_ultralytics_model_path())


def _validate_class_config() -> None:
    # YOLO_PLAYER_CLASSES/YOLO_BALL_CLASSES/YOLO_REFEREE_CLASSES are matched against the
    # model's own class names by exact string. A mismatch (e.g. configuring "ball" for a
    # model that actually calls it "football") doesn't error — it just silently produces
    # zero detections for that class, forever. Surface that loudly on startup instead of
    # letting it fail quietly for an entire session.
    model_names = {str(name).strip().lower() for name in model.names.values()}
    if not (BALL_CLASSES & model_names):
        logger.warning(
            "YOLO_BALL_CLASSES=%s has no overlap with this model's classes %s — "
            "ball detection will silently return nothing. Check the model's real "
            "class names and fix YOLO_BALL_CLASSES.",
            sorted(BALL_CLASSES), sorted(model_names),
        )
    if not (PLAYER_CLASSES & model_names):
        logger.warning(
            "YOLO_PLAYER_CLASSES=%s has no overlap with this model's classes %s — "
            "player detection will silently return nothing. Check the model's real "
            "class names and fix YOLO_PLAYER_CLASSES.",
            sorted(PLAYER_CLASSES), sorted(model_names),
        )
    if not (REFEREE_CLASSES & model_names):
        logger.info(
            "YOLO_REFEREE_CLASSES=%s has no overlap with this model's classes %s — "
            "referees (if any) will be treated as players for team clustering.",
            sorted(REFEREE_CLASSES), sorted(model_names),
        )


_validate_class_config()


class RawFrame(BaseModel):
    base64: str
    timestamp: float


class AnalyzeFramesRequest(BaseModel):
    frames: list[RawFrame]


TeamId = Literal["home", "away"]


@dataclass
class Detection:
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    tracker_id: Optional[int] = None


TrackState = dict[str, Any]


def decode_frame(raw: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def position_from_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    anchor: Literal["center", "bottom"] = "center",
) -> dict[str, float]:
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    y = y2 if anchor == "bottom" else (y1 + y2) / 2
    return {
        "x": round((x / width) * 100, 1),
        "y": round((y / height) * 100, 1),
    }


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


# Green-pitch HSV gate. OpenCV hue is 0-180; grass sits roughly 25-95 across
# sunlit/shadowed/yellowed pitches. Tunable via env without code changes.
PITCH_HSV_LOWER = np.array(
    [int(v) for v in os.getenv("YOLO_PITCH_HSV_LOWER", "25,25,25").split(",")], dtype=np.uint8
)
PITCH_HSV_UPPER = np.array(
    [int(v) for v in os.getenv("YOLO_PITCH_HSV_UPPER", "95,255,255").split(",")], dtype=np.uint8
)
# If the largest green region covers less than this fraction of the frame we
# assume it isn't a wide pitch shot (replay closeup, goalmouth scramble) and skip
# filtering entirely rather than risk dropping every real player.
PITCH_MIN_AREA_FRAC = float(os.getenv("YOLO_PITCH_MIN_AREA_FRAC", "0.10"))
PITCH_FILTER_ENABLED = os.getenv("YOLO_PITCH_FILTER", "1") not in ("0", "false", "False")


def compute_pitch_mask(image: Image.Image) -> Optional[np.ndarray]:
    """Return a filled binary mask of the playing surface, or None if no clear pitch.

    The pitch is the single largest contiguous green region. Filling its contour
    means players, lines and the centre circle (which are not green) still count as
    "on pitch", while the crowd, dugouts, sidelines and advertising boards — the
    main source of phantom player detections from a generic COCO model — fall
    outside it and get rejected downstream.
    """
    if not PITCH_FILTER_ENABLED:
        return None
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, PITCH_HSV_LOWER, PITCH_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < PITCH_MIN_AREA_FRAC * mask.size:
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


def filter_persons_to_pitch(image: Image.Image, detections: list[Detection]) -> list[Detection]:
    """Drop person detections whose feet aren't on the pitch; keep ball/others as-is.

    This is the main defence against over-tracking: a generic COCO model reports
    every spectator and bench player as a "person", and those flickering detections
    are what make ByteTrack's IDs churn and the overlay flash. Removing them leaves
    a stable ~22-player set that tracks cleanly.
    """
    mask = compute_pitch_mask(image)
    if mask is None:
        return detections
    kept: list[Detection] = []
    for d in detections:
        if d.cls_name in PLAYER_CLASSES and d.cls_name not in BALL_CLASSES:
            if not foot_on_pitch(d.xyxy, mask):
                continue
        kept.append(d)
    return kept


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

    X = np.stack(features).astype(float)

    # Seed: centroid 0 = first sample, centroid 1 = sample most distant from it.
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

    return ["home" if label == 0 else "away" for label in labels]


def player_role(position: dict[str, float], team: TeamId) -> str:
    x = position["x"]
    own_goal_x = 0 if team == "home" else 100
    distance_from_own_goal = abs(x - own_goal_x)
    if distance_from_own_goal < 12:
        return "gk"
    if distance_from_own_goal < 35:
        return "def"
    if distance_from_own_goal < 68:
        return "mid"
    return "fwd"


def pitch_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def prune_team_players(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one detection per track ID and cap each side to a plausible XI."""
    pruned: list[dict[str, Any]] = []

    for team in ("home", "away"):
        team_players = [p for p in players if p["team"] == team]
        best_by_id: dict[str, dict[str, Any]] = {}
        for player in team_players:
            existing = best_by_id.get(player["id"])
            player_score = float(player.get("detectionConfidence", 0.0)) * 1000 + float(player.get("boxArea", 0.0))
            existing_score = (
                float(existing.get("detectionConfidence", 0.0)) * 1000 + float(existing.get("boxArea", 0.0))
                if existing
                else -1
            )
            if existing is None or player_score > existing_score:
                best_by_id[player["id"]] = player

        kept = sorted(
            best_by_id.values(),
            key=lambda p: (float(p.get("detectionConfidence", 0.0)), float(p.get("boxArea", 0.0))),
            reverse=True,
        )[:MAX_PLAYERS_PER_TEAM]
        pruned.extend(sorted(kept, key=lambda p: p["id"]))

    return pruned


def stabilize_player_ids(
    frame: dict[str, Any],
    tracks: dict[TeamId, list[TrackState]],
    next_ids: dict[TeamId, int],
    max_jump: float = 20.0,
) -> dict[str, Any]:
    """Assign stable hN/aN IDs and smooth positions across tracker dropouts."""
    timestamp = float(frame["timestamp"])
    updated_players: list[dict[str, Any]] = []
    used_track_indices: dict[TeamId, set[int]] = {"home": set(), "away": set()}
    matched_existing = 0
    created_tracks = 0

    for team in ("home", "away"):
        team_players = [p for p in frame["players"] if p["team"] == team]

        for player in team_players:
            tracker_number = int(player["number"]) if player.get("number") else None
            canonical_team: TeamId = team
            if tracker_number is not None:
                for candidate_team in ("home", "away"):
                    if any(track.get("tracker_id") == tracker_number for track in tracks[candidate_team]):
                        canonical_team = candidate_team
                        break
                player["team"] = canonical_team

            team_tracks = tracks[canonical_team]

            best_idx: Optional[int] = None
            best_cost = max_jump
            for idx, track in enumerate(team_tracks):
                if idx in used_track_indices[canonical_team]:
                    continue
                if tracker_number is not None and track.get("tracker_id") == tracker_number:
                    best_idx = idx
                    best_cost = 0
                    break
                age = timestamp - float(track["last_seen"])
                if age > 10.0:
                    continue
                cost = pitch_distance(player["position"], track["position"]) + age * 1.5
                if cost < best_cost:
                    best_idx = idx
                    best_cost = cost

            if best_idx is None:
                if tracker_number is None:
                    next_ids[canonical_team] += 1
                    tracker_number = next_ids[canonical_team]
                else:
                    next_ids[canonical_team] = max(next_ids[canonical_team], tracker_number)
                stable_id = f"{'h' if canonical_team == 'home' else 'a'}{tracker_number}"
                team_tracks.append({
                    "id": stable_id,
                    "tracker_id": tracker_number,
                    "position": player["position"],
                    "smoothed_position": player["position"],
                    "last_seen": timestamp,
                })
                created_tracks += 1
            else:
                used_track_indices[canonical_team].add(best_idx)
                track = team_tracks[best_idx]
                stable_id = str(track["id"])
                previous_smoothed = track.get("smoothed_position", track["position"])
                smoothed = {
                    "x": round(previous_smoothed["x"] * (1 - TRACK_SMOOTHING_ALPHA) + player["position"]["x"] * TRACK_SMOOTHING_ALPHA, 1),
                    "y": round(previous_smoothed["y"] * (1 - TRACK_SMOOTHING_ALPHA) + player["position"]["y"] * TRACK_SMOOTHING_ALPHA, 1),
                }
                track["position"] = player["position"]
                track["smoothed_position"] = smoothed
                track["last_seen"] = timestamp
                matched_existing += 1

            player["id"] = stable_id
            try:
                player["number"] = int(stable_id[1:])
            except Exception:
                player["number"] = 0
            player["position"] = team_tracks[best_idx]["smoothed_position"] if best_idx is not None else player["position"]
            updated_players.append(player)

    raw_players = updated_players
    frame["players"] = prune_team_players(raw_players)
    if TRACK_DIAGNOSTICS:
        frame["_trackingDiagnostics"] = {
            "rawPlayers": len(raw_players),
            "postPrunePlayers": len(frame["players"]),
            "matchedExisting": matched_existing,
            "createdTracks": created_tracks,
            "droppedByPrune": max(0, len(raw_players) - len(frame["players"])),
            "homeCount": len([p for p in frame["players"] if p["team"] == "home"]),
            "awayCount": len([p for p in frame["players"] if p["team"] == "away"]),
            "stableIds": sorted(p["id"] for p in frame["players"]),
        }
    if frame.get("possessingPlayer"):
        poss = frame["possessingPlayer"]
        nearest = min(
            [p for p in frame["players"] if p["team"] == poss["team"]],
            key=lambda p: pitch_distance(p["position"], frame["ballPosition"]),
            default=None,
        )
        if nearest:
            frame["possessingPlayer"] = {"team": nearest["team"], "playerId": nearest["id"]}

    return frame


def smooth_possession(
    frame: dict[str, Any],
    previous_possession: Union[TeamId, Literal["contested"]],
    max_control_distance: float = 9.0,
) -> Union[TeamId, Literal["contested"]]:
    if not frame.get("ballPosition") or not frame["players"]:
        return previous_possession if previous_possession != "contested" else "contested"

    nearest = min(
        frame["players"],
        key=lambda p: pitch_distance(p["position"], frame["ballPosition"]),
    )
    distance = pitch_distance(nearest["position"], frame["ballPosition"])
    if distance > max_control_distance:
        return previous_possession if previous_possession != "contested" and distance <= max_control_distance * 1.6 else "contested"

    frame["possessingPlayer"] = {"team": nearest["team"], "playerId": nearest["id"]}
    return nearest["team"]


def interpolate_ball_positions(frames: list[dict[str, Any]], limit: int = BALL_INTERPOLATION_LIMIT) -> list[dict[str, Any]]:
    """Fill short gaps in ball detections, matching the cleaner offline demo behavior."""
    known = [(i, frame.get("ballPosition")) for i, frame in enumerate(frames) if frame.get("ballPosition")]
    if len(known) < 2:
        return frames

    for (start_idx, start_pos), (end_idx, end_pos) in zip(known, known[1:]):
        if start_pos is None or end_pos is None:
            continue
        gap = end_idx - start_idx - 1
        if gap <= 0 or gap > limit:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            frames[start_idx + offset]["ballPosition"] = {
                "x": round(start_pos["x"] * (1 - alpha) + end_pos["x"] * alpha, 1),
                "y": round(start_pos["y"] * (1 - alpha) + end_pos["y"] * alpha, 1),
            }
            frames[start_idx + offset]["ballInterpolated"] = True

    return frames


def consolidate_player_teams(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lock each stable player ID to a single team for the whole clip.

    Per-frame jersey clustering can occasionally flip a player home<->away for a
    single frame, which makes them jump across the tactical view. We tally a
    confidence-weighted vote per ID across every frame it appears in and rewrite
    the team (and role) to the majority, so a given player keeps one colour for
    the entire clip.
    """
    votes: dict[str, dict[TeamId, float]] = {}
    for frame in frames:
        for player in frame["players"]:
            weight = float(player.get("detectionConfidence", 0.0)) or 1.0
            tally = votes.setdefault(player["id"], {"home": 0.0, "away": 0.0})
            tally[player["team"]] += weight

    majority: dict[str, TeamId] = {
        pid: ("home" if tally["home"] >= tally["away"] else "away")
        for pid, tally in votes.items()
    }

    for frame in frames:
        for player in frame["players"]:
            team = majority.get(player["id"], player["team"])
            player["team"] = team
            player["role"] = player_role(player["position"], team)
        if frame.get("possessingPlayer"):
            pid = frame["possessingPlayer"].get("playerId")
            if pid in majority:
                frame["possessingPlayer"]["team"] = majority[pid]
    return frames


def interpolate_player_positions(
    frames: list[dict[str, Any]], limit: int = PLAYER_INTERPOLATION_LIMIT
) -> list[dict[str, Any]]:
    """Bridge short detection gaps per player so tracks don't flicker on/off.

    YOLO misses the odd frame when players overlap or blur; ByteTrack keeps the ID
    alive internally but emits no box, so the player vanishes for a frame or two and
    the overlay flashes. For each stable ID we linearly interpolate a position into
    the missing frames between two real sightings (gap <= limit), mark it inferred,
    and re-cap each frame to a plausible XI.
    """
    appearances: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, frame in enumerate(frames):
        for player in frame["players"]:
            appearances.setdefault(player["id"], []).append((idx, player))

    for pid, seen in appearances.items():
        for (start_idx, start_p), (end_idx, end_p) in zip(seen, seen[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > limit:
                continue
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1)
                position = {
                    "x": round(start_p["position"]["x"] * (1 - alpha) + end_p["position"]["x"] * alpha, 1),
                    "y": round(start_p["position"]["y"] * (1 - alpha) + end_p["position"]["y"] * alpha, 1),
                }
                frames[start_idx + offset]["players"].append({
                    **start_p,
                    "position": position,
                    "role": player_role(position, start_p["team"]),
                    # Slightly below a real detection so pruning prefers live boxes.
                    "detectionConfidence": round(float(start_p.get("detectionConfidence", 0.5)) * 0.9, 3),
                    "inferred": True,
                })

    for frame in frames:
        frame["players"] = prune_team_players(frame["players"])
    return frames


def annotate_tracking_quality(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify frames by whether they are suitable for tactical aggregation.

    Broadcast cutaways and closeups legitimately contain only one or two visible
    players. Those frames are useful for the video overlay, but if they feed the
    tactical shape/pass network they make the entire team vanish. Emit an explicit
    quality flag so the frontend can hold the last wide tactical state.
    """
    for frame in frames:
        players = frame.get("players", [])
        home = sum(1 for player in players if player.get("team") == "home")
        away = sum(1 for player in players if player.get("team") == "away")
        inferred = sum(1 for player in players if player.get("inferred"))
        total = len(players)

        if total >= 14 and home >= 5 and away >= 5:
            quality = "wide"
        elif total <= 6 or home == 0 or away == 0:
            quality = "closeup"
        else:
            quality = "low_confidence"

        frame["trackingQuality"] = quality
        frame["trackingCounts"] = {
            "players": total,
            "home": home,
            "away": away,
            "inferred": inferred,
        }

    return frames


def recompute_possession_for_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_possession: Union[TeamId, Literal["contested"]] = "contested"
    for frame in frames:
        frame["possession"] = smooth_possession(frame, previous_possession)
        previous_possession = frame["possession"]
    return frames


def passes_confidence(cls_name: str, confidence: float) -> bool:
    # Ball detections get the lower BALL_CONFIDENCE floor; everything else (mainly
    # players) still needs the stricter CONFIDENCE floor to avoid flooding team
    # clustering with low-quality boxes.
    threshold = BALL_CONFIDENCE if cls_name in BALL_CLASSES else CONFIDENCE
    return confidence >= threshold


def detections_for_image(image: Image.Image) -> list[Detection]:
    if use_huggingface_yolov5():
        result = model(image, size=960)
        names = result.names
        detections: list[Detection] = []
        for prediction in result.pred[0]:
            x1, y1, x2, y2, confidence, cls_id = prediction.tolist()
            cls_name = str(names[int(cls_id)]).lower()
            if not passes_confidence(cls_name, float(confidence)):
                continue
            detections.append(
                Detection(
                    cls_name=cls_name,
                    confidence=float(confidence),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    tracker_id=None,
                )
            )
        return detections

    # Run at the lower of the two floors so ball boxes below CONFIDENCE aren't
    # discarded by Ultralytics before we get a chance to apply the ball-specific
    # threshold below. imgsz matters a lot for ball recall: Ultralytics defaults to
    # 640, but soccana was trained at 1280 — running inference at the architecture's
    # default silently downscales the already-tiny ball further before the model
    # ever sees it.
    result = model.predict(
        source=np.asarray(image),
        conf=min(CONFIDENCE, BALL_CONFIDENCE),
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]
    names = result.names
    detections: list[Detection] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = str(names.get(cls_id, cls_id)).lower()
        confidence = float(box.conf[0])
        if not passes_confidence(cls_name, confidence):
            continue
        detections.append(
            Detection(
                cls_name=cls_name,
                confidence=confidence,
                xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                tracker_id=None,
            )
        )

    return detections


def detections_for_ultralytics_result(result: Any) -> list[Detection]:
    names = result.names
    detections: list[Detection] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = str(names.get(cls_id, cls_id)).lower()
        confidence = float(box.conf[0])
        if not passes_confidence(cls_name, confidence):
            continue

        tracker_id = None
        if getattr(box, "id", None) is not None:
            tracker_id = int(box.id[0])

        detections.append(
            Detection(
                cls_name=cls_name,
                confidence=confidence,
                xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                tracker_id=tracker_id,
            )
        )

    return detections


def analyze_single_frame(raw: RawFrame, frame_index: int) -> dict[str, Any]:
    image = decode_frame(raw.base64)
    width, height = image.size
    detections = filter_persons_to_pitch(image, detections_for_image(image))

    person_detections = [
        d for d in detections
        if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
    ]
    ball_detections = [d for d in detections if d.cls_name in BALL_CLASSES]
    features = [crop_jersey_features(image, d.xyxy) for d in person_detections]
    teams = split_teams(features)

    players: list[dict[str, Any]] = []
    team_counts: dict[TeamId, int] = {"home": 0, "away": 0}

    for detection, team in zip(person_detections, teams):
        team_counts[team] += 1
        position = position_from_box(detection.xyxy, width, height, anchor="bottom")
        tracker_number = detection.tracker_id if detection.tracker_id is not None else team_counts[team]
        player_id = f"{'h' if team == 'home' else 'a'}{tracker_number}"
        players.append(
            {
                "id": player_id,
                "number": detection.tracker_id or 0,
                "team": team,
                "role": player_role(position, team),
                "position": position,
                "action": "standing",
                "detectionConfidence": round(detection.confidence, 3),
                "boxArea": round(box_area(detection.xyxy), 1),
            }
        )

    ball_position = None
    if ball_detections:
        best_ball = max(ball_detections, key=lambda d: d.confidence)
        ball_position = position_from_box(best_ball.xyxy, width, height)

    possession: Union[TeamId, Literal["contested"]] = "contested"
    possessing_player = None
    if ball_position and players:
        nearest = min(
            players,
            key=lambda p: (p["position"]["x"] - ball_position["x"]) ** 2
            + (p["position"]["y"] - ball_position["y"]) ** 2,
        )
        possession = nearest["team"]
        possessing_player = {"team": nearest["team"], "playerId": nearest["id"]}

    frame: dict[str, Any] = {
        "frameIndex": frame_index,
        "timestamp": raw.timestamp,
        "players": players,
        "ballPosition": ball_position,
        "possession": possession,
        "events": [],
    }

    if possessing_player:
        frame["possessingPlayer"] = possessing_player

    return frame


def analyze_precomputed_frame(
    raw: RawFrame,
    frame_index: int,
    image: Image.Image,
    detections: list[Detection],
    teams: list[TeamId],
) -> dict[str, Any]:
    width, height = image.size
    person_detections = [
        d for d in detections
        if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
    ]
    ball_detections = [d for d in detections if d.cls_name in BALL_CLASSES]
    referee_detections = [d for d in detections if d.cls_name in REFEREE_CLASSES]

    players: list[dict[str, Any]] = []
    team_counts: dict[TeamId, int] = {"home": 0, "away": 0}

    for detection, team in zip(person_detections, teams):
        team_counts[team] += 1
        position = position_from_box(detection.xyxy, width, height, anchor="bottom")
        tracker_number = detection.tracker_id if detection.tracker_id is not None else team_counts[team]
        player_id = f"{'h' if team == 'home' else 'a'}{tracker_number}"
        players.append(
            {
                "id": player_id,
                "number": detection.tracker_id or 0,
                "team": team,
                "role": player_role(position, team),
                "position": position,
                "action": "standing",
                "detectionConfidence": round(detection.confidence, 3),
                "boxArea": round(box_area(detection.xyxy), 1),
            }
        )

    ball_position = None
    if ball_detections:
        best_ball = max(ball_detections, key=lambda d: d.confidence)
        ball_position = position_from_box(best_ball.xyxy, width, height)

    possession: Union[TeamId, Literal["contested"]] = "contested"
    possessing_player = None
    if ball_position and players:
        nearest = min(
            players,
            key=lambda p: (p["position"]["x"] - ball_position["x"]) ** 2
            + (p["position"]["y"] - ball_position["y"]) ** 2,
        )
        possession = nearest["team"]
        possessing_player = {"team": nearest["team"], "playerId": nearest["id"]}

    frame: dict[str, Any] = {
        "frameIndex": frame_index,
        "timestamp": raw.timestamp,
        "players": players,
        "ballPosition": ball_position,
        "possession": possession,
        "events": [],
    }

    if possessing_player:
        frame["possessingPlayer"] = possessing_player

    if referee_detections:
        frame["referees"] = [position_from_box(d.xyxy, width, height) for d in referee_detections]

    return frame


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/analyze-video")
def analyze_video_file(
    file: UploadFile = File(...),
    fps: float = Form(default=DEFAULT_DENSE_FPS),
) -> dict[str, Any]:
    """Accept a raw video file and return dense per-frame tracking data.

    Two-pass approach:
    1. Sample ~30 evenly-spaced frames to establish stable global team-colour
       centroids (avoids the per-frame flickering that happens when a single
       unbalanced frame shifts the brightness-split clustering).
    2. Extract frames at `fps` (default 5) and classify every player using the
       global centroids, returning the dense FrameData array.
    """
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "mp4")
    content = file.file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"error": "Could not open video", "frames": []}

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, round(video_fps / fps))

        logger.info(
            "analyze-video: %.1ffps video, target %.1ffps, step=%d, %d total → ~%d output frames",
            video_fps, fps, frame_step, total_frames, max(1, total_frames // frame_step),
        )

        # ── Pass 1: calibration — pool colours from ~30 spread-out frames ──────
        n_cal = 30
        cal_step = max(1, total_frames // n_cal)
        cal_indices = list(range(0, total_frames, cal_step))[:n_cal]
        all_cal_features: list[np.ndarray] = []

        for idx in cal_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            dets = filter_persons_to_pitch(img, detections_for_image(img))
            person_dets = [
                d for d in dets
                if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
            ]
            all_cal_features.extend(crop_jersey_features(img, d.xyxy) for d in person_dets)

        # Build fixed centroids; fall back to per-frame clustering if too few players found.
        global_centroids: Optional[tuple[np.ndarray, np.ndarray]] = None
        if len(all_cal_features) >= 4:
            cal_teams = split_teams(all_cal_features)
            home_feats = [f for f, t in zip(all_cal_features, cal_teams) if t == "home"]
            away_feats = [f for f, t in zip(all_cal_features, cal_teams) if t == "away"]
            if home_feats and away_feats:
                global_centroids = (
                    np.stack(home_feats).mean(axis=0),
                    np.stack(away_feats).mean(axis=0),
                )
                logger.info(
                    "analyze-video: global HSV centroids built — home n=%d, away n=%d",
                    len(home_feats), len(away_feats),
                )

        # ── Pass 2: dense tracking at target fps ────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        dense_frames: list[dict[str, Any]] = []
        tracks: dict[TeamId, list[TrackState]] = {"home": [], "away": []}
        next_ids: dict[TeamId, int] = {"home": 0, "away": 0}
        previous_possession: Union[TeamId, Literal["contested"]] = "contested"
        previous_diag_ids: set[str] = set()

        if not use_huggingface_yolov5():
            tracker_config = os.getenv("YOLO_TRACKER", "bytetrack_soccer.yaml")
            tracked_results = model.track(
                source=tmp_path,
                stream=True,
                persist=True,
                tracker=tracker_config,
                conf=min(CONFIDENCE, BALL_CONFIDENCE),
                imgsz=IMAGE_SIZE,
                vid_stride=frame_step,
                verbose=False,
            )

            for output_idx, result in enumerate(tracked_results):
                timestamp = round((output_idx * frame_step) / video_fps, 3)
                img = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
                dets = filter_persons_to_pitch(img, detections_for_ultralytics_result(result))
                person_dets = [
                    d for d in dets
                    if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
                ]
                tracker_ids = [d.tracker_id for d in person_dets if d.tracker_id is not None]
                frame_features = [crop_jersey_features(img, d.xyxy) for d in person_dets]

                if global_centroids and frame_features:
                    home_c, away_c = global_centroids
                    teams: list[TeamId] = [
                        "home" if np.linalg.norm(f - home_c) <= np.linalg.norm(f - away_c) else "away"
                        for f in frame_features
                    ]
                else:
                    teams = split_teams(frame_features)

                raw = RawFrame(base64="", timestamp=timestamp)
                frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams)
                frame = stabilize_player_ids(frame, tracks, next_ids)
                if TRACK_DIAGNOSTICS and output_idx % TRACK_DIAGNOSTIC_EVERY == 0:
                    diag = frame.get("_trackingDiagnostics", {})
                    current_ids = set(diag.get("stableIds", []))
                    logger.info(
                        "track-diag frame=%04d t=%.2fs persons=%d bytetrack_unique=%d matched=%d new=%d raw=%d kept=%d home=%d away=%d dropped=%d id_added=%d id_lost=%d",
                        output_idx,
                        timestamp,
                        len(person_dets),
                        len(set(tracker_ids)),
                        diag.get("matchedExisting", 0),
                        diag.get("createdTracks", 0),
                        diag.get("rawPlayers", 0),
                        diag.get("postPrunePlayers", 0),
                        diag.get("homeCount", 0),
                        diag.get("awayCount", 0),
                        diag.get("droppedByPrune", 0),
                        len(current_ids - previous_diag_ids),
                        len(previous_diag_ids - current_ids),
                    )
                    previous_diag_ids = current_ids
                frame.pop("_trackingDiagnostics", None)
                frame["possession"] = smooth_possession(frame, previous_possession)
                previous_possession = frame["possession"]
                dense_frames.append(frame)

                if (output_idx + 1) % 50 == 0:
                    logger.info(
                        "analyze-video: %d / ~%d frames processed",
                        output_idx + 1, max(1, total_frames // frame_step),
                    )
        else:
            frame_idx = 0
            output_idx = 0

            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp = round(frame_idx / video_fps, 3)
                    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    dets = filter_persons_to_pitch(img, detections_for_image(img))
                    person_dets = [
                        d for d in dets
                        if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
                    ]
                    colors = [crop_mean_color(img, d.xyxy) for d in person_dets]

                    if global_centroids and colors:
                        home_c, away_c = global_centroids
                        teams: list[TeamId] = [
                            "home" if np.linalg.norm(c - home_c) <= np.linalg.norm(c - away_c) else "away"
                            for c in colors
                        ]
                    else:
                        teams = split_teams(colors)

                    raw = RawFrame(base64="", timestamp=timestamp)
                    frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams)
                    frame = stabilize_player_ids(frame, tracks, next_ids)
                    frame["possession"] = smooth_possession(frame, previous_possession)
                    previous_possession = frame["possession"]
                    dense_frames.append(frame)
                    output_idx += 1

                    if output_idx % 50 == 0:
                        logger.info(
                            "analyze-video: %d / ~%d frames processed",
                            output_idx, max(1, total_frames // frame_step),
                        )

                frame_idx += 1

        cap.release()
        if TRACK_DIAGNOSTICS:
            pre_post_ids = [set(player["id"] for player in frame["players"]) for frame in dense_frames]
            avg_kept = sum(len(ids) for ids in pre_post_ids) / max(len(pre_post_ids), 1)
            avg_delta = sum(
                len(pre_post_ids[i] ^ pre_post_ids[i - 1])
                for i in range(1, len(pre_post_ids))
            ) / max(len(pre_post_ids) - 1, 1)
            logger.info(
                "track-diag pre-postprocess summary frames=%d avg_kept=%.2f avg_id_delta=%.2f unique_ids=%d",
                len(dense_frames),
                avg_kept,
                avg_delta,
                len(set().union(*pre_post_ids)) if pre_post_ids else 0,
            )
        dense_frames = consolidate_player_teams(dense_frames)
        dense_frames = interpolate_player_positions(dense_frames)
        dense_frames = recompute_possession_for_frames(interpolate_ball_positions(dense_frames))
        dense_frames = annotate_tracking_quality(dense_frames)
        if TRACK_DIAGNOSTICS:
            post_ids = [set(player["id"] for player in frame["players"]) for frame in dense_frames]
            avg_kept = sum(len(ids) for ids in post_ids) / max(len(post_ids), 1)
            avg_delta = sum(
                len(post_ids[i] ^ post_ids[i - 1])
                for i in range(1, len(post_ids))
            ) / max(len(post_ids) - 1, 1)
            logger.info(
                "track-diag postprocess summary frames=%d avg_kept=%.2f avg_id_delta=%.2f unique_ids=%d",
                len(dense_frames),
                avg_kept,
                avg_delta,
                len(set().union(*post_ids)) if post_ids else 0,
            )
        logger.info("analyze-video: complete — %d dense frames", len(dense_frames))
        return {"frames": dense_frames, "videoFps": video_fps, "targetFps": fps}

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/analyze-frames")
def analyze_frames(request: AnalyzeFramesRequest) -> dict[str, Any]:
    # Team color clustering runs per frame rather than pooled across the whole clip.
    # Pooling let a single noisy frame (lighting drift, a referee's kit color getting
    # swept into PLAYER_CLASSES, a frame with few visible players) shift the brightness
    # split for every frame in the match. Each broadcast frame has enough players from
    # both teams to cluster reliably on its own, and a bad frame then only costs that
    # one frame instead of corrupting team assignment for the whole video.
    frames = []
    tracks: dict[TeamId, list[TrackState]] = {"home": [], "away": []}
    next_ids: dict[TeamId, int] = {"home": 0, "away": 0}
    previous_possession: Union[TeamId, Literal["contested"]] = "contested"
    for i, raw_frame in enumerate(request.frames):
        image = decode_frame(raw_frame.base64)
        detections = filter_persons_to_pitch(image, detections_for_image(image))
        person_detections = [
            d for d in detections
            if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
        ]
        features = [crop_jersey_features(image, d.xyxy) for d in person_detections]
        teams = split_teams(features)
        frame = analyze_precomputed_frame(raw_frame, i, image, detections, teams)
        frame = stabilize_player_ids(frame, tracks, next_ids)
        frame["possession"] = smooth_possession(frame, previous_possession)
        previous_possession = frame["possession"]
        frames.append(frame)

    frames = consolidate_player_teams(frames)
    frames = interpolate_player_positions(frames)
    frames = recompute_possession_for_frames(interpolate_ball_positions(frames))
    frames = annotate_tracking_quality(frames)

    return {
        "processingMethod": "yolo-worker",
        "frames": frames,
    }
