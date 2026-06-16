# SoccerVision Analytics

A computer vision proof-of-concept that converts raw soccer match footage into structured performance data and coaching insights, built with **Next.js 15** and **Claude Vision AI**.

---

## What it does

| Step | Detail |
|------|--------|
| **Upload** | Drop any 2–3 minute MP4/WebM/MOV match clip |
| **Extract** | The browser captures sampled keyframes using the Canvas API — no server upload of the full video |
| **Analyse** | Sampled frames are sent to Claude via the Anthropic API; the model returns player positions, detected actions, and key events as structured JSON |
| **Dashboard** | Results are rendered as an interactive coaching dashboard: field visualisation, event timeline, team stats, heatmaps, and prioritised coaching recommendations |

---

## Local setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd analysis
npm install
```

### 2. Configure environment

Create `.env.local` locally and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

Optional model overrides:

```
ANTHROPIC_FRAME_MODEL=claude-sonnet-4-6
ANTHROPIC_SUMMARY_MODEL=claude-sonnet-4-6
NEXT_PUBLIC_VISION_WORKER_URL=http://localhost:8001
```

Recommended defaults:

- Keep `ANTHROPIC_FRAME_MODEL` on Sonnet while this app still relies on Claude for frame-level vision.
- Use `ANTHROPIC_SUMMARY_MODEL` for the final coaching insight pass.
- Set `NEXT_PUBLIC_VISION_WORKER_URL` only when running the optional YOLO worker.
- Do not use Claude Batch API for the interactive upload path; batches are cheaper, but asynchronous and better suited to background re-analysis.

### Optional YOLO worker

For a free local computer-vision layer:

```bash
cd workers/yolo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

Then run the web app with `NEXT_PUBLIC_VISION_WORKER_URL=http://localhost:8001`.

This costs nothing beyond local compute/electricity. For deployment, the worker needs a CPU/GPU host; the Next/Vercel app itself should not run YOLO.

### 3. Run locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 4. Demo mode (no API key needed)

Click **"View Live Demo"** on the home page. This loads a pre-computed sample match analysis (Eagles FC 1–0 City United) directly from the server — no Anthropic key required.

---

## Deploy to Vercel

```bash
npm install -g vercel
vercel --prod
```

Set `ANTHROPIC_API_KEY` as an **Environment Variable** in the Vercel project settings (Settings → Environment Variables). The demo mode works without it.

Or use the Vercel dashboard:
1. Push this repo to GitHub
2. Import it in [vercel.com/new](https://vercel.com/new)
3. Add `ANTHROPIC_API_KEY` under Environment Variables
4. Deploy

---

## Architecture

```
app/
  page.tsx              — Home/upload page; client-side frame extraction
  dashboard/page.tsx    — Interactive analytics dashboard
  api/analyze/frame/route.ts
                        — Serverless function: calls Claude Vision for one sampled frame
  api/analyze/summarize/route.ts
                        — Aggregates frame data, deduplicates events, generates insights
  api/analyze/route.ts  — Demo endpoint for precomputed sample data
  workers/yolo/         — Optional local YOLO worker for free CV preprocessing

components/
  SoccerField.tsx       — SVG field with animated player dots
  EventTimeline.tsx     — Chronological key-event list
  StatsChart.tsx        — Recharts bar chart (passes, shots, tackles…)
  TeamComparison.tsx    — Side-by-side stat bars with possession split
  Heatmap.tsx           — 10×10 player density grid overlay on the field
  CoachingInsights.tsx  — Priority-ranked AI recommendations
  MetricCard.tsx        — KPI summary cards

lib/
  types.ts              — Shared TypeScript types (MatchAnalysis, FrameData, …)
  sampleData.ts         — Pre-computed demo match (8 frames, 22 players, 5 insights)
```

---

## Notes

- **Frame budget**: A 2–3 minute clip produces roughly 15–23 sampled frames. Each compressed JPEG is <100 KB.
- **Rate limits**: Frame calls use bounded browser concurrency and one retry per failed frame.
- **No video stored server-side**: Only base64 keyframe images are sent; the original video never leaves the browser.
- **Export**: The dashboard includes a one-click JSON export of the full `MatchAnalysis` object for downstream use in coaching tools.
- **Accuracy roadmap**: Claude is useful for interpretation, but not ideal as the dense tracker. The next production-grade step is a YOLO-based video worker for player/ball detection, tracking, team classification, and pitch calibration. See [docs/vision-architecture.md](docs/vision-architecture.md).

---

## Tech stack

- [Next.js 15](https://nextjs.org) — App Router, serverless API routes
- [Anthropic Claude Sonnet](https://anthropic.com) — Vision model for frame analysis
- [Recharts](https://recharts.org) — Data visualisation
- [Tailwind CSS](https://tailwindcss.com) — Styling
- [Lucide React](https://lucide.dev) — Icons
