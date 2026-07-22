# SoccerVision Analytics

Converts raw soccer match footage into structured tracking data and coaching insights. **Next.js 15** app + an optional local **YOLO computer-vision worker**, with **Claude** used for interpretation/coaching language rather than dense frame-by-frame tracking.

---

## Summary

Two independent pieces, talking over HTTP:

1. **`app/` (Next.js)** — upload UI, dashboard, and thin serverless API routes that call Claude for frame sampling and coaching summaries. This is what's deployed to Vercel.
2. **`workers/yolo/` (FastAPI + Ultralytics YOLO)** — the real computer-vision layer. Runs locally (or on any Python host), detects players/ball/referees, tracks them across frames, assigns teams, and projects positions onto true pitch coordinates.

**Why split this way:** Claude is good at describing a frame in language, but tactical overlays need consistent geometry and identity over time (same player, same ID, frame after frame) — that's a dense CV problem, not a vision-language one. Keeping it a separate worker also means the Next/Vercel deployment never has to run YOLO, which serverless platforms aren't built for anyway. See [docs/vision-architecture.md](docs/vision-architecture.md) for the original reasoning.

The worker returns a **job ID immediately** and reports live progress (stage, frame count, percent) rather than blocking the request — dense analysis on a full clip can take minutes, and silently hanging with no feedback was the biggest source of "is this stuck?" confusion during development.

---

## How it's built, and why

### `workers/yolo/soccer_vision/` — the vision pipeline

One module per concern, so a behavior change always has one obvious place to make it:

| Module | Responsibility | Why it's separate |
| --- | --- | --- |
| `config.py` | Every env-tunable knob | One file to check when behavior differs between environments |
| `models.py` | Detector loading, device selection | Isolates YOLO instance lifecycle — see ReID note below |
| `detection.py` | Inference → `Detection` objects | Per-class confidence floors (ball needs a lower bar than players) |
| `pitch_mask.py` | Green-pitch gating | Rejects crowd/bench "person" detections before they ever reach tracking |
| `teams.py` / `teams_siglip.py` | Team assignment | Two backends — see below |
| `ball.py` | Ball selection + recovery | Plausibility-gated picking + zoomed recovery pass for a tiny, fast object |
| `tracking.py` | Stable ID assignment, XI pruning, possession | Keeps IDs consistent across ByteTrack/BoT-SORT dropouts |
| `postprocess.py` | Clip-level interpolation/consolidation | Runs once, after all frames are known (needs the full clip) |
| `pitch_homography.py` | 29-keypoint → true pitch coordinates | Optional; falls back to image-space coordinates if unavailable |
| `jobs.py` | In-memory async job registry | See "Async jobs" below |
| `api.py` | FastAPI routes only | No vision logic lives here — just wiring |

**Team assignment has two backends** (`YOLO_TEAM_BACKEND`, default `siglip`):
- **SigLIP + UMAP + KMeans** (default) — visual embeddings of player crops, clustered per clip. More robust to similar kits and lighting than color alone. Ported from the sibling `Soccer_Analysis_Model` repo.
- **HSV histogram clustering** (`hsv`) — jersey-region color histograms. Lighter, no extra model download. Automatic fallback if SigLIP fails (too few players in calibration, missing deps, degenerate clusters) — it can never make things worse, only equal.

**Tracking is BoT-SORT + ReID** (`botsort_soccer.yaml`), not plain ByteTrack. ByteTrack matches boxes purely by IoU, so players swap IDs whenever they cross or the broadcast camera pans. BoT-SORT adds appearance re-identification and optical-flow camera-motion compensation, which directly fixes both. Requires `ultralytics >= 8.4` (upgraded from 8.3.x, which shipped `with_reid` as a documented no-op). Fall back with `YOLO_TRACKER=bytetrack_soccer.yaml`.

**Every concurrent YOLO use gets its own model instance** (`models.new_model_instance()`). Ultralytics predictors are not reentrant — an early version of this worker had the ball-recovery pass call `predict()` on the same instance that a streaming `track()` call was mid-way through, which deadlocked the whole job on the first missed ball. Instances share the same cached weights file, so this costs nothing but a bit of memory.

### Async jobs (`jobs.py` + `api.py`)

`POST /analyze-video` returns a job ID instantly; the dense pass runs on a background thread. `GET /jobs/{id}` reports `stage` (`calibrating` → `tracking` → `postprocessing` → `done`), `framesDone`/`framesTotal`, and heartbeats `updatedAt` on every processed frame.

**Why:** the original version was one long synchronous HTTP call with no visibility — if it hung, there was no way to tell whether it was slow or actually stuck, or which stage it died in. The frontend now polls every 2s and flags a stall if the heartbeat stops advancing for 90s, naming the exact stage and frame. This is what caught the ReID deadlock above in one shot instead of hours of guessing.

### `app/` (Next.js)

```
app/page.tsx                        Home/upload; client-side frame extraction (Canvas API)
app/dashboard/page.tsx              Interactive dashboard; polls the worker job for dense results
app/api/analyze/frame/route.ts      Serverless: Claude Vision on one sampled frame
app/api/analyze/summarize/route.ts  Aggregates frame data, dedupes events, generates insights
app/api/analyze/route.ts            Demo endpoint (precomputed sample data, no API key needed)

components/   SoccerField, EventTimeline, StatsChart, TeamComparison, Heatmap, CoachingInsights, MetricCard
lib/          types.ts (shared types), matchLibrary/denseFrameStore (in-memory session state), pitchMapping
```

No video is ever uploaded to the Next server — the browser extracts and compresses keyframes locally; only those (and, separately, the raw file to the YOLO worker) leave the client.

---

## Local setup

```bash
git clone <your-repo-url> && cd analysis && npm install
cp .env.example .env.local   # add ANTHROPIC_API_KEY
```

**Terminal 1 — vision worker:**
```bash
cd workers/yolo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

**Terminal 2 — app:**
```bash
NEXT_PUBLIC_VISION_WORKER_URL=http://localhost:8001 npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Or click **"View Live Demo"** for a precomputed sample with no key/worker needed.

Worker tuning knobs, model details, and tests: [workers/yolo/README.md](workers/yolo/README.md).

---

## Deploying today

- **Next.js app → Vercel**, as-is: `vercel --prod`, with `ANTHROPIC_API_KEY` set in project env vars. Demo mode works with no key.
- **YOLO worker → currently local-only.** It needs a real host to be usable by anyone but you locally. See below.

---

## Next step: productionizing the worker (GCP)

Right now the worker runs on your machine and uses your Mac's GPU (MPS) for inference. To make this usable beyond your laptop:

1. **Containerize it.** Add a `Dockerfile` in `workers/yolo/` (Python 3.11, `pip install -r requirements.txt`, `CMD uvicorn app:app`). The model weights auto-download from Hugging Face on first boot, so the image itself stays small — or bake `soccana.pt` in to skip that on cold start.
2. **Host on Cloud Run with GPU support**, not a CPU-only box. Cloud Run now supports attaching an NVIDIA L4 per instance, which is the closest managed equivalent to your local GPU setup and scales to zero when idle (important — this worker is bursty, not always-on). Alternative: **GKE Autopilot with a GPU node pool** if you want more control over concurrency/batching than Cloud Run's request model allows.
3. **Swap `YOLO_DEVICE`** from `mps` (Apple Silicon-only) to `cuda` — this is already env-driven (`config.DEVICE`, `models.resolve_device()`), so no code change, just redeploying with an NVIDIA host.
4. **The async job system is already cloud-ready** in shape but not in durability: jobs live in an in-memory dict (`jobs.py`), so a Cloud Run instance restart or scale-to-zero loses in-flight jobs. For a real deployment, back `jobs.py` with **Cloud Tasks + Firestore** (or Redis) instead of the in-process dict — same interface (`create_job`/`get_job`/`update`), different storage.
5. **Point `NEXT_PUBLIC_VISION_WORKER_URL`** at the Cloud Run URL and add CORS origin restriction (`CORS_ORIGINS` is already env-driven — currently `*`).

None of this needs a rewrite; it needs the four changes above plus a `Dockerfile`. This is the natural next task.

---

## Current bottleneck: the detector, not the pipeline

Tracking, team assignment, pitch calibration, and job visibility are all solid now. What's left is the actual model: **`soccana.pt` is a YOLOv11-nano — 5.6MB, the smallest architecture in its family.** It was chosen for inference speed, and it shows up as missed players and (especially) missed ball detections that no amount of tracking/interpolation logic can fully paper over — you can't track a detection that never happened.

Two ways to address it, in order of effort vs. payoff:

- **Fine-tune on your actual footage** (highest accuracy per hour spent). If there's a consistent style of clip you test with — a specific camera angle, phone footage vs. broadcast — labeling ~150–300 frames of *that* footage and fine-tuning on top of `soccana.pt` closes the domain gap directly. `workers/yolo/train.py` is already built for this.
- **Train a larger architecture on the existing SoccerNet data** (`yolo11s` or `yolo11m` at imgsz 1280, using the training pipeline in the sibling `Soccer_Analysis_Model` repo). Meaningfully better recall, especially for the ball. `s` is realistic overnight on a Mac; `m` wants a cheap cloud GPU (a few dollars on Colab/Lambda) — and would also justify moving inference off Apple MPS onto the GCP GPU path above, at which point it's cheap to serve.

Either path is additive: the pipeline already treats `YOLO_MODEL_PATH` as swappable, so a better checkpoint drops in with no code change.

---

## Notes

- **No server-side video storage.** The Next app only ever sees compressed keyframe JPEGs (<100KB each); the worker receives the raw file directly from the browser and discards it after analysis.
- **Rate limits**: Claude frame calls use bounded browser concurrency with one retry per failed frame.
- **Export**: the dashboard has a one-click JSON export of the full `MatchAnalysis` object.

## Tech stack

- [Next.js 15](https://nextjs.org) — App Router, serverless API routes
- [Anthropic Claude](https://anthropic.com) — frame interpretation, coaching summaries
- [Ultralytics YOLO](https://ultralytics.com) (BoT-SORT+ReID tracking) — player/ball/referee detection
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) + UMAP + KMeans — team classification
- [Recharts](https://recharts.org), [Tailwind CSS](https://tailwindcss.com), [Lucide React](https://lucide.dev)
