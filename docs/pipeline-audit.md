# SoccerVision metric and Claude audit

## Dashboard metrics

| Surface | Current source | Class | Confidence/warning to expose | Main limitations |
| --- | --- | --- | --- | --- |
| Score/goals | Scoreboard reads in `app/api/analyze/events/route.ts`, synthesized across all reviewed frames in `app/page.tsx`; fallback visual goals from Claude when no scoreboard exists | Real CV/LLM verified, scoreboard-first | Goal event confidence; `scoreboard_conflict`/review flags | Claude OCR can misread overlays; no dedicated OCR yet |
| Possession | YOLO nearest controlled player per frame in `workers/yolo/app.py`, optionally corrected by Claude event review; summarized by frame share | Heuristic CV-derived | `TeamStats.metricConfidence.possession` | Ball misses, occlusion, camera cuts, team-color swaps |
| Passes | Verified `pass` events from Claude review plus tracking candidates; summarized after dedupe | LLM-verified heuristic | `TeamStats.metricConfidence.passes`; low-confidence events flagged | Sparse frame sampling misses short passes; player IDs still approximate without tracker |
| Pass accuracy | Completed pass count divided by completed passes plus turnover-like pressure proxy | Heuristic/statistical estimate | Label as inferred | No failed-pass detector yet |
| Shots / on target | Verified `shot`, `save`, `goal` events; on-target is goal or save | LLM-verified CV candidate | `TeamStats.metricConfidence.shots`; event confidence | Needs soccer action classifier for body/ball trajectory |
| xG / cumulative xG | `lib/visionMetrics.ts` estimates each shot from location, angle, event type, confidence, and semantic context | Heuristic/statistical estimate | `TeamStats.metricConfidence.xg`; event `xg` | Needs calibrated historical shot dataset |
| Finishing quality | Goals, shots, on-target, and xG from summarized stats | Heuristic/statistical estimate | xG confidence and shot confidence | Conversion on short clips is noisy |
| Distance covered | Same stable player ID displacement only, capped per frame step | Real CV-derived when IDs stable; otherwise unreliable/low | `TeamStats.metricConfidence.distance` | Broadcast perspective is not pitch-calibrated homography; ID switches undercount/overcount |
| Spatial occupancy | Team player positions binned into 10x10 grid in summarize route | Real CV-derived | Possession/tracking warning | Depends on detector recall and camera calibration |
| Shot map | Verified shot/save/goal events with event positions only | LLM/CV candidate-derived | Empty-state when no located shots | Claude may place shot origin imprecisely |
| Pass network | Stable `possessingPlayer` transitions across frames via `buildPassNetwork` | Heuristic CV-derived | Empty-state when no stable transitions | Sparse sampling misses pass recipient sequence |
| Event timeline | Claude verified events plus low-confidence tracking fallbacks from event route | Mixed LLM/CV/heuristic | Per-event confidence, source, conflicts, pipeline flag | Duplicate/replay risk remains |
| Coaching insights | Claude summary route from deterministic stats and conflict flags; fallback deterministic insights | LLM-derived | Insight confidence should be treated as priority-derived, not model-calibrated | Hallucination risk if stats are sparse |
| Dense tracking status / tracked objects | `denseFrameStore` fed by YOLO worker `/analyze-video`; dashboard displays frame counts | Real CV-derived | Dense ready/error/loading status | ByteTrack only available through Ultralytics backend |

## Claude usage

| Route/component | Responsibility | Model/env | Payload/batching | Schema/parse/retry | Risks | Better constraint |
| --- | --- | --- | --- | --- | --- | --- |
| `app/api/analyze/frame/route.ts` | Fallback per-frame visual analysis when YOLO worker is absent: players, ball, possession, events | `ANTHROPIC_FRAME_MODEL` default `claude-sonnet-4-6` | One current image, optional previous image disabled in client | JSON parse only; no retry server-side, client retries once | High cost, hallucinated players/events, weak ID stability | Prefer YOLO first; use only when worker unavailable |
| `app/api/analyze/events/route.ts` | Event verification, possession correction, scoreboard reading | `ANTHROPIC_EVENT_MODEL` then `ANTHROPIC_SUMMARY_MODEL`, default `claude-sonnet-4-6` | Client batches 4 downscaled images; route also batches by 4; up to 40 candidate windows | Clean JSON, truncation repair, partial frame recovery; 3 attempts with backoff; fallback low-confidence candidates | Duplicate routine events, scoreboard OCR errors, event hallucination in busy windows | Smaller candidate windows, stricter schema validation, dedicated OCR, cache repeated frames |
| `app/api/analyze/summarize/route.ts` | Coaching insight generation from summarized stats/conflicts | `ANTHROPIC_SUMMARY_MODEL` default `claude-sonnet-4-6` | Text-only compact stats and up to 6 conflicts | JSON array parse; deterministic fallback insights | Can overstate weak metrics, confidence is not calibrated | Include metric confidences and warnings; keep stats deterministic |
| `app/page.tsx` | Client orchestration of worker, event review, summary, scoreboard synthesis | Calls internal routes, not Claude directly | Parallel frame/event requests with downscaled review images | Client retries per-frame once; records batch warnings | Parallel Claude calls can spike cost and latency | Gate Claude review to high-value candidates when YOLO confidence is high |

## Recommended ML work

1. Add a dedicated scoreboard OCR path and compare it against Claude reads before synthesizing goals.
2. Label 200-500 frames for ball/player/referee/team and evaluate detector recall by camera angle and resolution.
3. Add a small event benchmark: 30 clips with pass/shot/turnover timestamps and shot origin labels.
4. Calibrate xG on labeled shot locations, body part/context, and goalkeeper/defender pressure.
5. Add pitch homography calibration so distance and heatmaps use field coordinates instead of image-percent coordinates.
