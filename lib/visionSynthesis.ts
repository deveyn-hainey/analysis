import type Anthropic from "@anthropic-ai/sdk";
import type {
  CoachingInsight,
  MatchEvent,
  OutcomeProjection,
  TeamAnalysis,
} from "@/lib/types";

// One vision-grounded synthesis call. Unlike the per-frame detection model
// (Sonnet, high volume), this is a single call per analysis that actually looks
// at a curated set of key frames alongside the numeric metrics, so the outputs
// that are genuinely judgment calls — outcome projection, coaching insights,
// shot xG — are grounded in what's visible rather than reasoned from a stat line.
// Defaults to Opus 4.8 for stronger tactical reasoning; override per env.
const SYNTHESIS_MODEL = process.env.ANTHROPIC_SYNTHESIS_MODEL ?? "claude-opus-4-8";

type ImageBlock = {
  type: "image";
  source: { type: "base64"; media_type: "image/jpeg"; data: string };
};
type TextBlock = { type: "text"; text: string };

export interface SynthesisKeyFrame {
  timestamp: number;
  base64: string;
}

export interface VisionSynthesisResult {
  outcome: Omit<OutcomeProjection, "source"> | null;
  summary: string;
  insights: CoachingInsight[];
  // Per-shot xG keyed by rounded timestamp (seconds, 1 decimal) so the summarize
  // route can merge it onto the matching shot/goal/save event.
  shotXg: Map<string, number>;
  usedVision: boolean;
}

export function shotKey(timestamp: number): string {
  return timestamp.toFixed(1);
}

// Stable instruction prefix — cached so repeated analyses don't re-pay for it.
const SYSTEM_PROMPT = `You are an elite soccer performance analyst reviewing a short match clip.

You are given (1) computed metrics for both teams and (2) a curated set of still frames from the clip, each labelled with its timestamp. Study the frames: tactical shape, defensive line height, pressing, space, body positioning at shots, and goalkeeper/defender positions. Ground every judgment in what the frames and metrics actually show — do not invent events you cannot see.

Return ONLY a single valid JSON object — no markdown, no code fences — with this exact shape:
{
  "summary": "<2-3 sentence plain-English narrative of what actually happens in THIS clip — who controls play, the key chances/events, and how it's trending. Describe the clip, not a full match.>",
  "outcome": {
    "homeWin": <int 0-100>,
    "draw": <int 0-100>,
    "awayWin": <int 0-100>,
    "reasoning": "<1-2 sentences citing what you saw>"
  },
  "insights": [
    {
      "category": "attacking" | "defensive" | "possession" | "tactical" | "physical",
      "priority": "critical" | "high" | "medium" | "low",
      "title": "<short title>",
      "observation": "<what the frames/metrics show>",
      "recommendation": "<specific, actionable coaching step>",
      "affectedTeam": "home" | "away" | "both"
    }
  ],
  "shotXg": [
    { "timestamp": <number, matching a provided shot timestamp>, "xg": <number 0-1> }
  ]
}

Rules:
- outcome.homeWin + outcome.draw + outcome.awayWin MUST sum to 100. This is a projection for THIS passage of play, not a full-match prediction — reflect the momentum and chances visible in the clip.
- Produce 4-5 insights. Prefer observations only the frames reveal (shape, line height, finishing technique) over restating the numbers.
- For shotXg, only include timestamps present in the SHOTS list below; estimate xG from the visible chance quality (angle, distance, defenders, keeper position). Omit shots you cannot see clearly.`;

function buildContextText(
  homeTeam: TeamAnalysis,
  awayTeam: TeamAnalysis,
  shotEvents: MatchEvent[],
  eventConflicts: Array<{ timestamp: number; type: string; conflicts?: string[]; pipelineFlag?: string }>
): string {
  const team = (t: TeamAnalysis) =>
    `${t.name}:
- Possession: ${t.stats.possession}% | Passes: ${t.stats.passes} (acc est. ${t.stats.passAccuracy}%)
- Shots: ${t.stats.shots} (on target ${t.stats.shotsOnTarget}) | Goals: ${t.stats.goals} | xG: ${t.stats.expectedGoals?.toFixed(2) ?? "n/a"}
- Tackles: ${t.stats.tackles} | Fouls: ${t.stats.fouls} | Corners: ${t.stats.corners}
- Avg player x-position: ${t.averagePosition.x}/100 (0=own goal, 100=opponent goal)`;

  const shots = shotEvents.length
    ? shotEvents
        .map((e) => `- ${e.timestamp.toFixed(1)}s ${e.team ?? "?"} ${e.type}${e.position ? ` at (${e.position.x.toFixed(0)},${e.position.y.toFixed(0)})` : ""}`)
        .join("\n")
    : "- None detected";

  const conflicts = eventConflicts.length
    ? eventConflicts
        .slice(0, 6)
        .map((c) => `- ${c.timestamp.toFixed(1)}s ${c.type}: ${(c.conflicts ?? []).join("; ") || c.pipelineFlag || "review flag"}`)
        .join("\n")
    : "- None";

  return `HOME ${team(homeTeam)}

AWAY ${team(awayTeam)}

SHOTS (provide xG for these timestamps):
${shots}

EVENT CONFLICTS / REVIEW FLAGS:
${conflicts}`;
}

interface RawSynthesis {
  summary?: string;
  outcome?: { homeWin?: number; draw?: number; awayWin?: number; reasoning?: string };
  insights?: Array<Partial<CoachingInsight>>;
  shotXg?: Array<{ timestamp?: number; xg?: number }>;
}

function normalizeOutcome(raw: RawSynthesis["outcome"]): VisionSynthesisResult["outcome"] {
  if (!raw) return null;
  const home = Math.max(0, Math.round(raw.homeWin ?? 0));
  const draw = Math.max(0, Math.round(raw.draw ?? 0));
  const away = Math.max(0, Math.round(raw.awayWin ?? 0));
  const sum = home + draw + away;
  if (sum <= 0) return null;
  // Re-normalize to 100 in case the model's three values don't sum exactly.
  const homeWin = Math.round((home / sum) * 100);
  const awayWin = Math.round((away / sum) * 100);
  const drawPct = 100 - homeWin - awayWin;
  return {
    homeWin,
    draw: drawPct,
    awayWin,
    reasoning: (raw.reasoning ?? "").trim() || "Projection from observed play and metrics.",
  };
}

export async function runVisionSynthesis(
  client: Anthropic,
  args: {
    homeTeam: TeamAnalysis;
    awayTeam: TeamAnalysis;
    shotEvents: MatchEvent[];
    eventConflicts: Array<{ timestamp: number; type: string; conflicts?: string[]; pipelineFlag?: string }>;
    keyFrames: SynthesisKeyFrame[];
  }
): Promise<VisionSynthesisResult> {
  const { homeTeam, awayTeam, shotEvents, eventConflicts, keyFrames } = args;

  const content: Array<ImageBlock | TextBlock> = [];
  content.push({ type: "text", text: buildContextText(homeTeam, awayTeam, shotEvents, eventConflicts) });

  if (keyFrames.length > 0) {
    content.push({ type: "text", text: `\nKEY FRAMES (${keyFrames.length}), each labelled with its timestamp:` });
    for (const frame of keyFrames) {
      content.push({ type: "text", text: `Frame @ ${frame.timestamp.toFixed(1)}s:` });
      content.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: frame.base64 } });
    }
  }

  const response = await client.messages.create({
    model: SYNTHESIS_MODEL,
    max_tokens: 4096,
    system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
    messages: [{ role: "user", content }],
  });

  const rawText = response.content[0]?.type === "text" ? response.content[0].text.trim() : "";
  const cleaned = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

  const parsed = JSON.parse(cleaned) as RawSynthesis;

  const shotXg = new Map<string, number>();
  for (const entry of parsed.shotXg ?? []) {
    if (typeof entry.timestamp === "number" && typeof entry.xg === "number") {
      shotXg.set(shotKey(entry.timestamp), Math.min(1, Math.max(0, entry.xg)));
    }
  }

  const insights = (parsed.insights ?? [])
    .filter((i): i is CoachingInsight => Boolean(i.title && i.observation && i.recommendation && i.category && i.affectedTeam))
    .map((i, idx) => ({ ...i, id: i.id ?? `vi-${idx + 1}` }));

  return {
    outcome: normalizeOutcome(parsed.outcome),
    summary: (parsed.summary ?? "").trim(),
    insights,
    shotXg,
    usedVision: keyFrames.length > 0,
  };
}
