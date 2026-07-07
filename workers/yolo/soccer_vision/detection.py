"""YOLO inference → Detection lists, with per-class confidence floors."""
from typing import Any

import numpy as np
from PIL import Image

from . import config, models
from .schemas import Detection


def decode_frame(raw: str) -> Image.Image:
    import base64
    import io

    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def passes_confidence(cls_name: str, confidence: float) -> bool:
    # Ball detections get the lower BALL_CONFIDENCE floor; everything else (mainly
    # players) still needs the stricter CONFIDENCE floor to avoid flooding team
    # clustering with low-quality boxes.
    threshold = config.BALL_CONFIDENCE if cls_name in config.BALL_CLASSES else config.CONFIDENCE
    return confidence >= threshold


def detections_for_image(image: Image.Image) -> list[Detection]:
    if models.use_huggingface_yolov5():
        result = models.model(image, size=960)
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
    # threshold. imgsz matters a lot for ball recall: Ultralytics defaults to
    # 640, but soccana was trained at 1280 — running inference at the architecture's
    # default silently downscales the already-tiny ball further before the model
    # ever sees it.
    result = models.model.predict(
        source=np.asarray(image),
        conf=min(config.CONFIDENCE, config.BALL_CONFIDENCE),
        imgsz=config.IMAGE_SIZE,
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


def split_detections(detections: list[Detection]) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Split into (players, ball, referees) using the configured class-name sets."""
    person = [
        d for d in detections
        if d.cls_name in config.PLAYER_CLASSES and d.cls_name not in config.REFEREE_CLASSES
    ]
    ball = [d for d in detections if d.cls_name in config.BALL_CLASSES]
    referee = [d for d in detections if d.cls_name in config.REFEREE_CLASSES]
    return person, ball, referee
