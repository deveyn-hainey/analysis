"""FastAPI app: HTTP wiring only — all vision logic lives in the sibling modules."""
import os
import tempfile
import threading
import time
import traceback
from typing import Any, Literal, Optional, Union

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import numpy as np

from . import config, models
from .ball import BallTracker
from .config import logger
from .detection import decode_frame, detections_for_image, detections_for_ultralytics_result, split_detections
from .frames import analyze_precomputed_frame
from .jobs import Job, create_job, get_job
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

# The dense pass is CPU/GPU-bound and stateful (one tracker per video); run one
# at a time so two uploads don't interleave and double the wall-clock of both.
_dense_semaphore = threading.Semaphore(1)


@app.get("/health")
def health() -> dict[str, Any]:
    from .pitch_homography import _projector

    _projector._ensure()
    return {
        "status": "ok",
        "model": config.MODEL_PATH or "soccana (default)",
        "device": models.DEVICE,
        "pitchHomography": _projector.available,
    }


def _run_dense_analysis(
    tmp_path: str,
    fps: float,
    home_kit_color: str,
    away_kit_color: str,
    job: Job,
) -> dict[str, Any]:
    """The dense two-pass video analysis. Reports progress via `job`."""
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, round(video_fps / fps))
        expected_output = max(1, total_frames // frame_step)

        logger.info(
            "job %s: %.1ffps video, target %.1ffps, step=%d, %d total → ~%d output frames, kits home=%s away=%s",
            job.id, video_fps, fps, frame_step, total_frames, expected_output,
            home_kit_color or "auto", away_kit_color or "auto",
        )

        # ── Pass 1: calibration — pool colours from ~30 spread-out frames ──────
        job.update(stage="calibrating", frames_done=0, frames_total=expected_output)
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
            job.update()  # heartbeat so a calibration stall is visible

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
                    "job %s: global HSV centroids built — home n=%d, away n=%d",
                    job.id, len(home_feats), len(away_feats),
                )

        # ── Pass 2: dense tracking at target fps ────────────────────────────────
        job.update(stage="tracking")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        dense_frames: list[dict[str, Any]] = []
        ball_tracker = BallTracker()
        tracks: dict[TeamId, list[TrackState]] = {"home": [], "away": []}
        next_ids: dict[TeamId, int] = {"home": 0, "away": 0}
        previous_possession: Union[TeamId, Literal["contested"]] = "contested"

        if not models.use_huggingface_yolov5():
            # Own instance: the request-serving endpoints (/analyze-frames, live
            # tracking) share the global model, and a streaming track() on it
            # would deadlock with their predict() calls.
            job_model = models.new_model_instance()
            tracked_results = job_model.track(
                source=tmp_path,
                stream=True,
                persist=True,
                tracker=config.TRACKER_CONFIG,
                conf=min(config.CONFIDENCE, config.BALL_CONFIDENCE),
                imgsz=config.IMAGE_SIZE,
                device=models.DEVICE,
                vid_stride=frame_step,
                verbose=False,
            )

            for output_idx, result in enumerate(tracked_results):
                timestamp = round((output_idx * frame_step) / video_fps, 3)
                img = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
                pitch_mask = compute_pitch_mask(img)
                dets = filter_persons_to_pitch(img, detections_for_ultralytics_result(result), pitch_mask)
                person_dets, _, _ = split_detections(dets)
                teams = assign_teams(img, person_dets, global_centroids, home_kit_color, away_kit_color)

                raw = RawFrame(base64="", timestamp=timestamp)
                frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams, pitch_mask is not None, pitch_mask, ball_tracker)
                frame = stabilize_player_ids(frame, tracks, next_ids)
                frame.pop("_trackingDiagnostics", None)
                frame["possession"] = smooth_possession(frame, previous_possession)
                previous_possession = frame["possession"]
                dense_frames.append(frame)
                job.update(frames_done=output_idx + 1)

                if (output_idx + 1) % 50 == 0:
                    logger.info("job %s: %d / ~%d frames processed", job.id, output_idx + 1, expected_output)
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
                    frame = analyze_precomputed_frame(raw, output_idx, img, dets, teams, pitch_mask is not None, pitch_mask, ball_tracker)
                    frame = stabilize_player_ids(frame, tracks, next_ids)
                    frame["possession"] = smooth_possession(frame, previous_possession)
                    previous_possession = frame["possession"]
                    dense_frames.append(frame)
                    output_idx += 1
                    job.update(frames_done=output_idx)

                    if output_idx % 50 == 0:
                        logger.info("job %s: %d / ~%d frames processed", job.id, output_idx, expected_output)

                frame_idx += 1

        cap.release()
        job.update(stage="postprocessing")
        dense_frames = run_postprocessing(dense_frames)
        logger.info("job %s: complete — %d dense frames", job.id, len(dense_frames))
        return {"frames": dense_frames, "videoFps": video_fps, "targetFps": fps}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _dense_worker(tmp_path: str, fps: float, home: str, away: str, job: Job) -> None:
    with _dense_semaphore:
        job.update(status="running", stage="starting")
        try:
            result = _run_dense_analysis(tmp_path, fps, home, away, job)
            job.update(status="done", stage="done", result=result, finished_at=time.time())
        except Exception as exc:
            logger.error("job %s failed: %s\n%s", job.id, exc, traceback.format_exc())
            job.update(status="error", stage="error", error=str(exc), finished_at=time.time())


@app.post("/analyze-video")
def analyze_video_file(
    file: UploadFile = File(...),
    fps: float = Form(default=config.DEFAULT_DENSE_FPS),
    homeKitColor: Optional[str] = Form(default=None),
    awayKitColor: Optional[str] = Form(default=None),
    sync: bool = Form(default=False),
) -> dict[str, Any]:
    """Kick off dense video analysis and return a job ID immediately.

    Poll GET /jobs/{jobId} for stage + frame progress, then fetch
    GET /jobs/{jobId}/result when status is "done". Pass sync=true to block
    until completion and get the result inline (old behaviour, useful for
    curl/scripts).
    """
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "mp4")
    content = file.file.read()
    home_kit_color, away_kit_color = request_kit_colors(homeKitColor, awayKitColor)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job = create_job()
    logger.info(
        "job %s: accepted %s (%.1f MB), fps=%.1f, sync=%s",
        job.id, file.filename, len(content) / 1e6, fps, sync,
    )

    if sync:
        _dense_worker(tmp_path, fps, home_kit_color, away_kit_color, job)
        if job.status == "error":
            return {"error": job.error, "frames": []}
        return job.result or {"frames": []}

    thread = threading.Thread(
        target=_dense_worker,
        args=(tmp_path, fps, home_kit_color, away_kit_color, job),
        daemon=True,
    )
    thread.start()
    return job.to_status_dict()


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job (worker restarted?) — re-upload the video")
    return job.to_status_dict()


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job (worker restarted?) — re-upload the video")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "analysis failed")
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"job not finished (status={job.status})")
    return job.result


@app.post("/analyze-frames")
def analyze_frames(request: AnalyzeFramesRequest) -> dict[str, Any]:
    # Team color clustering runs per frame rather than pooled across the whole clip.
    # Pooling let a single noisy frame (lighting drift, a referee's kit color getting
    # swept into PLAYER_CLASSES, a frame with few visible players) shift the brightness
    # split for every frame in the match. Each broadcast frame has enough players from
    # both teams to cluster reliably on its own, and a bad frame then only costs that
    # one frame instead of corrupting team assignment for the whole video.
    frames = []
    ball_tracker = BallTracker()
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
        frame = analyze_precomputed_frame(raw_frame, i, image, detections, teams, pitch_mask is not None, pitch_mask, ball_tracker)
        frame = stabilize_player_ids(frame, tracks, next_ids)
        frame["possession"] = smooth_possession(frame, previous_possession)
        previous_possession = frame["possession"]
        frames.append(frame)

    frames = run_postprocessing(frames)

    return {
        "processingMethod": "yolo-worker",
        "frames": frames,
    }
