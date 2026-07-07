"""Detector loading and class-name validation.

Default behaviour (no YOLO_MODEL_PATH set) is now the fine-tuned soccana
YOLOv11 model — Player/Ball/Referee classes trained on broadcast soccer at
imgsz 1280 — instead of the generic COCO yolo11n checkpoint. COCO has no
referee class and treats every spectator as "person", which is what made
out-of-the-box detection quality so poor.

Resolution order for the default:
1. a local ``soccana.pt`` next to the worker (already downloaded), else
2. HF Hub download of ``Adit-jain/soccana`` ``Model/weights/best.pt``, else
3. ``yolo11n.pt`` as a last-resort generic fallback (logged loudly).
"""
from . import config
from .config import logger


def resolve_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = resolve_device()
logger.info("inference device: %s", DEVICE)


def use_huggingface_yolov5() -> bool:
    if config.MODEL_BACKEND == "yolov5":
        return True
    if config.MODEL_BACKEND == "ultralytics":
        return False
    if config.HF_FILENAME:
        # YOLO_HF_FILENAME only makes sense for an ultralytics checkpoint resolved via
        # huggingface_hub — without this check the "repo/name" shape of MODEL_PATH would
        # otherwise trip the yolov5 auto-detect heuristic below.
        return False
    return "/" in config.MODEL_PATH and not config.MODEL_PATH.endswith(".pt")


def resolve_default_model_path() -> str:
    local_soccana = config.WORKER_DIR / "soccana.pt"
    if local_soccana.exists():
        logger.info("using local fine-tuned model %s", local_soccana)
        return str(local_soccana)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=config.SOCCANA_HF_REPO, filename=config.SOCCANA_HF_FILENAME)
        logger.info("using fine-tuned model from HF Hub: %s", config.SOCCANA_HF_REPO)
        return path
    except Exception as exc:
        logger.warning(
            "could not fetch fine-tuned soccana model (%s); falling back to generic "
            "yolo11n.pt — expect much worse ball/referee detection. Set "
            "YOLO_MODEL_PATH or restore network access to fix.",
            exc,
        )
        return "yolo11n.pt"


def resolve_ultralytics_model_path() -> str:
    if not config.MODEL_PATH:
        return resolve_default_model_path()
    if not config.HF_FILENAME:
        return config.MODEL_PATH
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=config.MODEL_PATH, filename=config.HF_FILENAME)


RESOLVED_MODEL_PATH: str = ""


def load_model():
    global RESOLVED_MODEL_PATH
    if use_huggingface_yolov5():
        import yolov5

        model = yolov5.load(config.MODEL_PATH)
        model.conf = min(config.CONFIDENCE, config.BALL_CONFIDENCE)
        return model

    from ultralytics import YOLO

    RESOLVED_MODEL_PATH = resolve_ultralytics_model_path()
    return YOLO(RESOLVED_MODEL_PATH)


model = load_model()


def new_model_instance():
    """A fresh YOLO instance sharing the same weights file.

    Ultralytics models are stateful: a streaming ``model.track()`` generator
    and any other ``predict()``/``track()`` call must never share one instance,
    or they corrupt each other's predictor state and can deadlock. Long-running
    jobs and auxiliary passes (ball recovery) therefore get their own instance;
    the weights file itself is cached on disk, so this is cheap.
    """
    from ultralytics import YOLO

    return YOLO(RESOLVED_MODEL_PATH or resolve_ultralytics_model_path())


def _validate_class_config() -> None:
    # YOLO_PLAYER_CLASSES/YOLO_BALL_CLASSES/YOLO_REFEREE_CLASSES are matched against the
    # model's own class names by exact string. A mismatch (e.g. configuring "ball" for a
    # model that actually calls it "football") doesn't error — it just silently produces
    # zero detections for that class, forever. Surface that loudly on startup instead of
    # letting it fail quietly for an entire session.
    model_names = {str(name).strip().lower() for name in model.names.values()}
    if not (config.BALL_CLASSES & model_names):
        logger.warning(
            "YOLO_BALL_CLASSES=%s has no overlap with this model's classes %s — "
            "ball detection will silently return nothing. Check the model's real "
            "class names and fix YOLO_BALL_CLASSES.",
            sorted(config.BALL_CLASSES), sorted(model_names),
        )
    if not (config.PLAYER_CLASSES & model_names):
        logger.warning(
            "YOLO_PLAYER_CLASSES=%s has no overlap with this model's classes %s — "
            "player detection will silently return nothing. Check the model's real "
            "class names and fix YOLO_PLAYER_CLASSES.",
            sorted(config.PLAYER_CLASSES), sorted(model_names),
        )
    if not (config.REFEREE_CLASSES & model_names):
        logger.info(
            "YOLO_REFEREE_CLASSES=%s has no overlap with this model's classes %s — "
            "referees (if any) will be treated as players for team clustering.",
            sorted(config.REFEREE_CLASSES), sorted(model_names),
        )


_validate_class_config()
