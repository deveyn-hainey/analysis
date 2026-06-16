import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
MODEL_BACKEND = os.getenv("YOLO_BACKEND", "auto").lower()
CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
PLAYER_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_PLAYER_CLASSES", "person,player,goalkeeper").split(",")}
BALL_CLASSES = {name.strip().lower() for name in os.getenv("YOLO_BALL_CLASSES", "sports ball,ball").split(",")}
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
    return "/" in MODEL_PATH and not MODEL_PATH.endswith(".pt")


if use_huggingface_yolov5():
    import yolov5

    model = yolov5.load(MODEL_PATH)
    model.conf = CONFIDENCE
else:
    model = YOLO(MODEL_PATH)


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


def detections_for_image(image: Image.Image) -> list[Detection]:
    if use_huggingface_yolov5():
        result = model(image, size=960)
        names = result.names
        detections: list[Detection] = []
        for prediction in result.pred[0]:
            x1, y1, x2, y2, confidence, cls_id = prediction.tolist()
            cls_name = str(names[int(cls_id)]).lower()
            detections.append(
                Detection(
                    cls_name=cls_name,
                    confidence=float(confidence),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections

    result = model.predict(source=np.asarray(image), conf=CONFIDENCE, verbose=False)[0]
    names = result.names
    detections: list[Detection] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = str(names.get(cls_id, cls_id)).lower()
        detections.append(
            Detection(
                cls_name=cls_name,
                confidence=float(box.conf[0]),
                xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
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
        players.append(
            {
                "id": f"{'h' if team == 'home' else 'a'}{team_counts[team]}",
                "number": 0,
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/analyze-frames")
def analyze_frames(request: AnalyzeFramesRequest) -> dict[str, Any]:
    frames = [analyze_single_frame(frame, i) for i, frame in enumerate(request.frames)]
    return {
        "processingMethod": "yolo-worker",
        "frames": frames,
    }
