"""FastAPI app: HTTP wiring only — all vision logic lives in the sibling modules."""
import os
import tempfile
from typing import Any, Literal, Optional, Union

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import numpy as np

from . import config, models
from .config import logger
from .detection import decode_frame, detections_for_image, detections_for_ultralytics_result, split_detections
from .frames import analyze_precomputed_frame
from .pitch_mask import compute_pitch_mask, filter_persons_to_pitch
from .postprocess import run_postprocessing
from .schemas import AnalyzeFramesRequest, RawFrame, TeamId, TrackState
from .teams import assign_teams, crop_jersey_features, request_kit_colors, split_teams
from .tracking import smooth_possession, stabilize_player_ids

app = FastAPI(title="SoccerVision YOLO Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": config.MODEL_PATH or "soccana (default)"}


@app.post("/analyze-video")
def analyze_video_file(
    file: UploadFile = File(...),
    fps: float = Form(default=config.DEFAULT_DENSE_FPS),
    homeKitColor: Optional[str] = Form(default=None),
    awayKitColor: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    """Accept a raw video file and return dense per-frame tracking data.

    Two-pass approach:
    1. Sample ~30 evenly-spaced frames to establish stable global team-colour
       centroids (avoids the per-frame flickering that happens when a single
       unbalanced frame shifts the brightness-split clustering).
    2. Extract frames at `fps` and classify every player using the global
       centroids, returning the dense FrameData array.
    """
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "mp4")
    content = file.file.read()
    home_kit_color, away_kit_color = request_kit_colors(homeKitColor, awayKitColor)

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
            "analyze-video: %.1ffps video, target %.1ffps, step=%d, %d total → ~%d output frames, kits home=%s away=%s",
            video_fps, fps, frame_step, total_frames, max(1, total_frames // frame_step),
            home_kit_color or "auto", away_kit_color or "auto",
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
            pitch_mask = compute_pitch_mask(img)
            dets = filter_persons_to_pitch(img, detections_for_image(img), pitch_mask)
            person_dets, _, _ = split_detections(dets)
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

        if not models.use_huggingface_yolov5():
            tracked_results = models.model.track(
                source=tmp_path,
                stream=True,
                persist=True,
                tracker=config.TRACKER_CONFIG,
                conf=min(config.CONFIDENCE, config.BALL_CONFIDENCE),
                imgsz=config.IMAGE_SIZE,
                vid_stride=frame_step,
                verbose=False,
            )

            for output_idx, result in enumerate(tracked_results):
                timestamp = round((output_idx * frame_step) / video_fps, 3)
                img = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
                pitch_mask = compute_pitch_mask(img)
                dets = filter_persons_to_pitch(img, detections_for_ultralytics_result(result), pitch_mask)
                person_dets, _, _ = split_detections(dets)
                tracker_ids = [d.tracker_id for d in person_dets if d.tracker_id is not None]
                teams = assign_teams(img, person_dets, global_centroids, home_kit_color, away_kit_color)

                raw = RawFrame(base64="", timestamp=timestamp)
                frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams, pitch_mask is not None, pitch_mask)
                frame = stabilize_player_ids(frame, tracks, next_ids)
                if config.TRACK_DIAGNOSTICS and output_idx % config.TRACK_DIAGNOSTIC_EVERY == 0:
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
                    pitch_mask = compute_pitch_mask(img)
                    dets = filter_persons_to_pitch(img, detections_for_image(img), pitch_mask)
                    person_dets, _, _ = split_detections(dets)
                    teams = assign_teams(img, person_dets, global_centroids, home_kit_color, away_kit_color)

                    raw = RawFrame(base64="", timestamp=timestamp)
                    frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams, pitch_mask is not None, pitch_mask)
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
        if config.TRACK_DIAGNOSTICS:
            _log_id_stability("pre-postprocess", dense_frames)
        dense_frames = run_postprocessing(dense_frames)
        if config.TRACK_DIAGNOSTICS:
            _log_id_stability("postprocess", dense_frames)
        logger.info("analyze-video: complete — %d dense frames", len(dense_frames))
        return {"frames": dense_frames, "videoFps": video_fps, "targetFps": fps}

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _log_id_stability(stage: str, frames: list[dict[str, Any]]) -> None:
    id_sets = [set(player["id"] for player in frame["players"]) for frame in frames]
    avg_kept = sum(len(ids) for ids in id_sets) / max(len(id_sets), 1)
    avg_delta = sum(
        len(id_sets[i] ^ id_sets[i - 1])
        for i in range(1, len(id_sets))
    ) / max(len(id_sets) - 1, 1)
    logger.info(
        "track-diag %s summary frames=%d avg_kept=%.2f avg_id_delta=%.2f unique_ids=%d",
        stage,
        len(frames),
        avg_kept,
        avg_delta,
        len(set().union(*id_sets)) if id_sets else 0,
    )


@app.post("/analyze-frames")
def analyze_frames(request: AnalyzeFramesRequest) -> dict[str, Any]:
    # Team color clustering runs per frame rather than pooled across the whole clip.
    # Pooling let a single noisy frame (lighting drift, a referee's kit color getting
    # swept into PLAYER_CLASSES, a frame with few visible players) shift the brightness
    # split for every frame in the match. Each broadcast frame has enough players from
    # both teams to cluster reliably on its own, and a bad frame then only costs that
    # one frame instead of corrupting team assignment for the whole video.
    frames = []
    home_kit_color, away_kit_color = request_kit_colors(request.homeKitColor, request.awayKitColor)
    tracks: dict[TeamId, list[TrackState]] = {"home": [], "away": []}
    next_ids: dict[TeamId, int] = {"home": 0, "away": 0}
    previous_possession: Union[TeamId, Literal["contested"]] = "contested"
    for i, raw_frame in enumerate(request.frames):
        image = decode_frame(raw_frame.base64)
        pitch_mask = compute_pitch_mask(image)
        detections = filter_persons_to_pitch(image, detections_for_image(image), pitch_mask)
        person_detections, _, _ = split_detections(detections)
        teams = assign_teams(image, person_detections, None, home_kit_color, away_kit_color)
        frame = analyze_precomputed_frame(raw_frame, i, image, detections, teams, pitch_mask is not None, pitch_mask)
        frame = stabilize_player_ids(frame, tracks, next_ids)
        frame["possession"] = smooth_possession(frame, previous_possession)
        previous_possession = frame["possession"]
        frames.append(frame)

    frames = run_postprocessing(frames)

    return {
        "processingMethod": "yolo-worker",
        "frames": frames,
    }
