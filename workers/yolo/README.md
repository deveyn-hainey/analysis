# YOLO Vision Worker

Optional local worker for faster, cheaper frame analysis before Claude summarization.

This worker uses open-source YOLO locally. It does not require another paid API. The first run may download YOLO weights.

## Setup

```bash
cd workers/yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

Then run the Next app with:

```bash
NEXT_PUBLIC_VISION_WORKER_URL=http://localhost:8001 npm run dev
```

## Cost

- Local CPU: free beyond electricity, but slower.
- Local GPU / Apple Silicon: free beyond electricity and much faster if supported by your Python/PyTorch install.
- Cloud GPU: paid by runtime. Keep Claude for final coaching summaries.

## Model

Default:

```bash
YOLO_MODEL_PATH=yolo11n.pt
```

This is a generic COCO model. It can detect people and sports balls, but it is not soccer-specialized. For better accuracy, replace it with a fine-tuned soccer detector that has classes such as:

- player
- goalkeeper
- referee
- ball

```bash
YOLO_MODEL_PATH=/path/to/soccer-best.pt uvicorn app:app --host 0.0.0.0 --port 8001
```

This worker is intentionally a starter path: it improves speed and gives the app a real CV layer, but production-quality soccer analytics still needs a soccer-trained detector, tracking, and pitch calibration.

## Use a pretrained football model from Hugging Face

For the interview demo, start here instead of training from scratch:

```bash
YOLO_BACKEND=yolov5 \
YOLO_MODEL_PATH=keremberke/yolov5m-football \
YOLO_PLAYER_CLASSES=player \
YOLO_BALL_CLASSES=football \
uvicorn app:app --host 0.0.0.0 --port 8001
```

That model is hosted on Hugging Face and loads through the `yolov5` package. It was trained on `keremberke/football-object-detection` and is intended for football object detection. The first run downloads the weights locally.

This model only has two classes — `football` (the ball) and `player` (everyone on the pitch, including goalkeepers and referees). There is no separate `referee` class, so `YOLO_REFEREE_CLASSES` has nothing to match against and is left unset above; referees are not excluded from team-color clustering with this model.

The worker logs a warning on startup if your configured `YOLO_PLAYER_CLASSES`/`YOLO_BALL_CLASSES` don't overlap with the model's actual class names at all — check that log line if ball or player detections seem to be silently missing. Confirm a model's real class names yourself with:

```python
import yolov5
print(yolov5.load("keremberke/yolov5m-football").names)
```

## Use a pretrained soccer-specific YOLOv11n model (player/ball/referee)

[Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) is a YOLOv11n model trained
specifically on players, the ball, and referees (3 classes), rather than a generic
COCO/football-vs-everything-else split. It's hosted on the HF Hub as a plain repo file,
not via Ultralytics' own `ultralyticsplus`/`from_pretrained` packaging, so don't install
`ultralyticsplus` for it — as of writing that package pins `ultralytics==8.0.239`, which
predates YOLO11 support and will downgrade the `ultralytics` install this worker already
depends on. Instead, point the worker at the repo + file path and let `huggingface_hub`
resolve it to a local path before `ultralytics.YOLO()` loads it:

```bash
YOLO_MODEL_PATH=Adit-jain/soccana \
YOLO_HF_FILENAME=Model/weights/best.pt \
YOLO_DENSE_FPS=15 \
uvicorn app:app --host 0.0.0.0 --port 8001
```

This model's real classes are `Player`, `Ball`, `Referee` (verified by loading the
checkpoint directly — the model card's casing differs slightly but matching here is
case-insensitive). They line up with this worker's defaults, so no
`YOLO_PLAYER_CLASSES`/`YOLO_BALL_CLASSES`/`YOLO_REFEREE_CLASSES` overrides are needed —
unlike the `keremberke/yolov5m-football` model above, this one also gives you a real
referee class, so officials get excluded from team-color clustering correctly.

For smoother dashboard overlays, the worker now mirrors the cleaner offline model
pipeline more closely:

- dense video tracking defaults to `YOLO_DENSE_FPS=15` instead of 5fps
- player positions use the bottom-center of the detection box, closer to the feet
- stable track IDs are smoothed over time to reduce detector jitter
- short ball-detection gaps are linearly interpolated with `YOLO_BALL_INTERPOLATION_LIMIT`
- tracker IDs keep their previous team assignment when jersey clustering flickers

The biggest remaining difference from `Soccer_Analysis_Model` is pitch homography:
that repo also uses a keypoint model to project image detections onto true pitch
coordinates. This worker still emits image-percent coordinates, so tactical views
will improve further once a keypoint model/homography pass is added.

## Fine-tune a soccer detector

Training requires labeled images in YOLO format. For a demo, label a small set of representative frames from the exact kind of match footage you will present:

- `player`
- `goalkeeper`
- `referee`
- `ball`

Recommended demo-size dataset:

- 100-300 labeled frames can make the demo visibly better than generic YOLO.
- Include wide tactical shots, close-ups, goalmouth moments, dark/light jerseys, and frames where the ball is small.
- Put 80% of frames in train and 20% in validation.

Dataset shape:

```txt
soccer-dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  dataset.yaml
```

Use `dataset.example.yaml` as the starting point.

Train locally:

```bash
python train.py \
  --data /absolute/path/to/soccer-dataset/dataset.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 960 \
  --batch 8
```

Run the worker with the trained weights:

```bash
YOLO_MODEL_PATH=runs/soccer/detector/weights/best.pt \
uvicorn app:app --host 0.0.0.0 --port 8001
```

For an interview demo, this local training path has no API cost. Training on CPU may be slow; if you have Apple Silicon or a GPU-backed Python/PyTorch install, it should be much faster.

## How YOLO and Claude work together

YOLO runs first and produces structured frame data:

- player locations
- ball location when visible
- rough team split from jersey colors
- rough possession from nearest player to ball

Claude runs after that to generate the coaching insights and can still be used as a fallback when the YOLO worker is not running.

The demo story is:

1. YOLO does the fast computer-vision extraction locally.
2. Claude turns the extracted match data into a readable coaching report.
3. The dashboard renders the resulting tactical overlay and event timeline.
