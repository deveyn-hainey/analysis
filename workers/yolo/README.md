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
