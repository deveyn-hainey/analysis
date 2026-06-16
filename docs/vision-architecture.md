# Vision Architecture Recommendation

## Recommendation

Use Claude for interpretation and coaching language, not as the primary dense soccer tracker.

The best next step is a two-layer pipeline:

1. A computer-vision worker extracts structured tracking data from the video.
2. Claude reviews the structured data plus a few candidate key moments and generates coaching insights.

This keeps the app fast, reduces Claude cost, and makes overlays more accurate.

## Why Not Claude Only?

Claude can describe sampled frames, but tactical overlays need consistent geometry and identity over time:

- player and goalkeeper detection
- ball detection
- referee exclusion
- persistent track IDs
- team classification
- pitch calibration / homography
- event candidates such as shots, passes, crosses, saves, and goals

A general vision-language model is not the right tool for all of that dense frame-by-frame work.

## Why Not Batch API For Uploads?

Anthropic Message Batches are useful for cheap background work, not immediate uploads. Batches are asynchronous and can take much longer than a user expects to wait after dropping a 2-3 minute clip.

Use batches later for:

- reprocessing historical clips
- overnight analysis
- bulk evaluation
- cheaper non-interactive analysis

## Practical Starting Stack

Start with this local or cloud worker:

- Python 3.11+
- `ffmpeg` for video decoding
- Ultralytics YOLO for player/ball/referee detection
- BoT-SORT or ByteTrack for tracking
- OpenCV for geometry
- `supervision` for detection utilities
- optional GPU for speed

Minimum Python packages:

```txt
ultralytics
opencv-python
supervision
numpy
```

## Model Choice

Start with YOLO object detection and tracking before trying a larger soccer action model.

Recommended first version:

- YOLO detector trained/fine-tuned for `player`, `goalkeeper`, `referee`, `ball`
- BoT-SORT for moving broadcast cameras
- team color clustering from tracked player crops
- simple goal candidates from ball/goalmouth/scoreboard/celebration signals
- Claude review only for uncertain goal/shot moments

Recommended research-grade path:

- SoccerNet Game State Reconstruction style pipeline
- pitch line detection and camera calibration
- player re-identification
- jersey number recognition where visible
- SoccerNet action or ball-action spotting models for event timing

## Expected Costs

Current Claude-only path:

- Cheapest to set up because it only needs `ANTHROPIC_API_KEY`
- Slower because every sampled frame is a remote vision request
- Accuracy is limited for tracking and ball events

YOLO worker path:

- Local CPU: no cloud cost, but slower
- Local Apple Silicon or consumer GPU: low/no per-video cost, usually much faster
- Cloud GPU: pay for GPU runtime, but video analysis can be much faster and more accurate
- Claude cost drops because only summaries and key candidate moments go to Claude

## What To Set Up Next

For a real accuracy upgrade, set up one of these:

1. Local worker on your machine
   - Install Python, `ffmpeg`, and the packages above.
   - Good for development and cost control.

2. Cloud worker
   - Use RunPod, Modal, Replicate, AWS, GCP, or another GPU host.
   - Better for deployed users and faster turnaround.

3. Hybrid first
   - Keep the current Next/Vercel app.
   - Add a worker endpoint later.
   - Continue using Claude as a fallback while the CV model matures.

## Integration Shape

The eventual app flow should be:

1. User uploads video.
2. App sends video to a worker.
3. Worker returns tracking JSON:

```json
{
  "frames": [
    {
      "timestamp": 12.4,
      "players": [
        { "trackId": "p17", "team": "home", "bbox": [120, 80, 146, 132], "pitch": { "x": 52, "y": 34 } }
      ],
      "ball": { "bbox": [318, 146, 324, 152], "pitch": { "x": 61, "y": 48 } },
      "events": []
    }
  ],
  "candidates": [
    { "type": "goal", "timestamp": 84.2, "confidence": 0.72, "reason": "ball near goalmouth plus celebration" }
  ]
}
```

4. Claude reviews candidate moments and creates coaching insights.
5. Dashboard renders the worker's structured tracking data.

## Current Interim Setting

The current app remains Claude-only, but it now avoids sending previous-frame image context by default and uses bounded concurrency. That makes uploads faster while we prepare the YOLO worker path.
