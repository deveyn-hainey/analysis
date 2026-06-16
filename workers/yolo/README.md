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
