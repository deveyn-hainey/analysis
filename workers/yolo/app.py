import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

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
    tracker_id: int | None = None


def decode_frame(raw: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def position_from_box(box: tuple[float, float, float, float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = box
    return {
        "x": round((((x1 + x2) / 2) / width) * 100, 1),
        "y": round((((y1 + y2) / 2) / height) * 100, 1),
    }


def crop_mean_color(image: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    upper_y2 = y1 + max(1, (y2 - y1) // 2)
    crop = np.asarray(image.crop((x1, y1, x2, upper_y2)))
    if crop.size == 0:
        return np.array([128.0, 128.0, 128.0])
    return crop.reshape(-1, 3).mean(axis=0)


def split_teams(colors: list[np.ndarray]) -> list[TeamId]:
    if not colors:
        return []
    if len(colors) == 1:
        return ["home"]

    brightness = np.array([color.mean() for color in colors])
    dark_idx = int(brightness.argmin())
    light_idx = int(brightness.argmax())
    centroids = np.stack([colors[light_idx], colors[dark_idx]]).astype(float)

    labels = np.zeros(len(colors), dtype=int)
    for _ in range(6):
        distances = np.stack([np.linalg.norm(np.stack(colors) - c, axis=1) for c in centroids], axis=1)
        labels = distances.argmin(axis=1)
        for label in (0, 1):
            members = [colors[i] for i, value in enumerate(labels) if value == label]
            if members:
                centroids[label] = np.stack(members).mean(axis=0)

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
    detections = detections_for_image(image)

    person_detections = [
        d for d in detections
        if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
    ]
    ball_detections = [d for d in detections if d.cls_name in BALL_CLASSES]
    colors = [crop_mean_color(image, d.xyxy) for d in person_detections]
    teams = split_teams(colors)

    players: list[dict[str, Any]] = []
    team_counts: dict[TeamId, int] = {"home": 0, "away": 0}

    for detection, team in zip(person_detections, teams):
        team_counts[team] += 1
        position = position_from_box(detection.xyxy, width, height)
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
            }
        )

    ball_position = None
    if ball_detections:
        best_ball = max(ball_detections, key=lambda d: d.confidence)
        ball_position = position_from_box(best_ball.xyxy, width, height)

    possession: TeamId | Literal["contested"] = "contested"
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
        position = position_from_box(detection.xyxy, width, height)
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
            }
        )

    ball_position = None
    if ball_detections:
        best_ball = max(ball_detections, key=lambda d: d.confidence)
        ball_position = position_from_box(best_ball.xyxy, width, height)

    possession: TeamId | Literal["contested"] = "contested"
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
    fps: float = Form(default=5.0),
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
        all_cal_colors: list[np.ndarray] = []

        for idx in cal_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            dets = detections_for_image(img)
            person_dets = [
                d for d in dets
                if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
            ]
            all_cal_colors.extend(crop_mean_color(img, d.xyxy) for d in person_dets)

        # Build fixed centroids; fall back to per-frame clustering if too few players found.
        global_centroids: tuple[np.ndarray, np.ndarray] | None = None
        if len(all_cal_colors) >= 4:
            cal_teams = split_teams(all_cal_colors)
            home_cols = [c for c, t in zip(all_cal_colors, cal_teams) if t == "home"]
            away_cols = [c for c, t in zip(all_cal_colors, cal_teams) if t == "away"]
            if home_cols and away_cols:
                global_centroids = (
                    np.stack(home_cols).mean(axis=0),
                    np.stack(away_cols).mean(axis=0),
                )
                logger.info(
                    "analyze-video: global centroids home=%s away=%s",
                    global_centroids[0].round(1).tolist(),
                    global_centroids[1].round(1).tolist(),
                )

        # ── Pass 2: dense tracking at target fps ────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        dense_frames: list[dict[str, Any]] = []

        if not use_huggingface_yolov5():
            tracker_config = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
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
                dets = detections_for_ultralytics_result(result)
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
                dense_frames.append(analyze_precomputed_frame(raw, output_idx, img, dets, teams))

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
                    dets = detections_for_image(img)
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
                    dense_frames.append(
                        analyze_precomputed_frame(raw, output_idx, img, dets, teams)
                    )
                    output_idx += 1

                    if output_idx % 50 == 0:
                        logger.info(
                            "analyze-video: %d / ~%d frames processed",
                            output_idx, max(1, total_frames // frame_step),
                        )

                frame_idx += 1

        cap.release()
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
    for i, raw_frame in enumerate(request.frames):
        image = decode_frame(raw_frame.base64)
        detections = detections_for_image(image)
        person_detections = [
            d for d in detections
            if d.cls_name in PLAYER_CLASSES and d.cls_name not in REFEREE_CLASSES
        ]
        colors = [crop_mean_color(image, d.xyxy) for d in person_detections]
        teams = split_teams(colors)
        frames.append(analyze_precomputed_frame(raw_frame, i, image, detections, teams))

    return {
        "processingMethod": "yolo-worker",
        "frames": frames,
    }
