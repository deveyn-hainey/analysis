# SoccerVision Analytics

A computer vision proof-of-concept that converts raw soccer match footage into structured performance data and coaching insights, built with **Next.js 15** and **Claude Vision AI**.

---

## What it does

| Step | Detail |
|------|--------|
| **Upload** | Drop any 2–3 minute MP4/WebM/MOV match clip |
| **Extract** | The browser captures keyframes every 8 s using the Canvas API — no server upload of the full video |
| **Analyse** | Each frame is sent to Claude Sonnet via the Anthropic API; the model returns player positions, detected actions, and key events as structured JSON |
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

```bash
cp .env.local.example .env.local
```

Open `.env.local` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

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
  api/analyze/route.ts  — Serverless function: calls Claude Vision per frame,
                          aggregates MatchAnalysis, generates coaching insights

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

- **Frame budget**: A 3-minute clip at 8 s intervals produces ~22 frames. Each compressed JPEG is <100 KB, well within Vercel's 4.5 MB request limit.
- **Rate limits**: Frames are analysed sequentially to stay within Claude API rate limits.
- **No video stored server-side**: Only base64 keyframe images are sent; the original video never leaves the browser.
- **Export**: The dashboard includes a one-click JSON export of the full `MatchAnalysis` object for downstream use in coaching tools.

---

## Tech stack

- [Next.js 15](https://nextjs.org) — App Router, serverless API routes
- [Anthropic Claude Sonnet](https://anthropic.com) — Vision model for frame analysis
- [Recharts](https://recharts.org) — Data visualisation
- [Tailwind CSS](https://tailwindcss.com) — Styling
- [Lucide React](https://lucide.dev) — Icons
