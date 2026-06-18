import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { PitchView } from "@/lib/types";

// Cheap, high-volume estimation — stays on Sonnet rather than the synthesis model.
const PITCH_MODEL = process.env.ANTHROPIC_PITCH_MODEL ?? "claude-sonnet-4-6";

interface PitchRequest {
  frames: Array<{ timestamp: number; base64: string }>;
}

type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
type TextBlock = { type: "text"; text: string };

const PROMPT = `You are calibrating a broadcast soccer camera to the pitch. For EACH numbered frame, estimate where the visible image maps onto the real pitch, using visible landmarks (halfway line, center circle, penalty boxes, goals, touchlines).

For each frame return:
- lengthMin: pitch length % at the LEFT edge of the image (0 = left goal line, 50 = halfway line, 100 = right goal line).
- lengthMax: pitch length % at the RIGHT edge of the image.
- topImageY: image vertical % (0 = top of image, 100 = bottom) where the PLAYING FIELD begins — the far touchline / grass horizon. Everything above it is crowd/stands.
- confidence: 0-1, how sure you are from the visible landmarks.

If the camera shows only the right half of the pitch, lengthMin/lengthMax might be ~50/100. If zoomed wide, ~0/100. lengthMin < lengthMax always (left edge is lower pitch-length than right edge for a standard broadcast).

Return ONLY a JSON array, one object per frame in order, no markdown:
[{ "lengthMin": 50, "lengthMax": 100, "topImageY": 35, "confidence": 0.7 }]`;

interface RawView {
  lengthMin?: number;
  lengthMax?: number;
  topImageY?: number;
  confidence?: number;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n));
}

function normalizeView(raw: RawView | undefined): PitchView | null {
  if (!raw || typeof raw.lengthMin !== "number" || typeof raw.lengthMax !== "number" || typeof raw.topImageY !== "number") {
    return null;
  }
  if ((raw.confidence ?? 1) < 0.35) return null;
  const lengthMin = clamp(raw.lengthMin, -20, 120);
  const lengthMax = clamp(raw.lengthMax, -20, 120);
  // Degenerate / inverted window — not usable.
  if (lengthMax - lengthMin < 5) return null;
  return {
    lengthMin,
    lengthMax,
    topImageY: clamp(raw.topImageY, 0, 90),
  };
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "ANTHROPIC_API_KEY not configured." }, { status: 500 });
    }

    const { frames } = (await req.json()) as PitchRequest;
    if (!frames || frames.length === 0) {
      return NextResponse.json({ views: [] });
    }

    const client = new Anthropic({ apiKey });
    const content: Array<ImageBlock | TextBlock> = [{ type: "text", text: PROMPT }];
    frames.forEach((frame, i) => {
      content.push({ type: "text", text: `Frame ${i + 1} (${frame.timestamp.toFixed(1)}s):` });
      content.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: frame.base64 } });
    });

    const response = await client.messages.create({
      model: PITCH_MODEL,
      max_tokens: 1500,
      messages: [{ role: "user", content }],
    });

    const raw = response.content[0]?.type === "text" ? response.content[0].text.trim() : "[]";
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

    let parsed: RawView[] = [];
    try {
      parsed = JSON.parse(cleaned) as RawView[];
    } catch {
      parsed = [];
    }

    const views = frames.map((frame, i) => ({
      timestamp: frame.timestamp,
      pitchView: normalizeView(parsed[i]),
    }));

    return NextResponse.json({ views });
  } catch (err) {
    console.error("[/api/analyze/pitch]", err);
    return NextResponse.json({ views: [] });
  }
}
