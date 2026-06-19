import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { PitchView } from "@/lib/types";

const PITCH_MODEL = process.env.ANTHROPIC_PITCH_MODEL ?? "claude-sonnet-4-6";

interface PitchRequest {
  frames: Array<{ timestamp: number; base64: string }>;
}

type ImageBlock = { type: "image"; source: { type: "base64"; media_type: "image/jpeg"; data: string } };
type TextBlock = { type: "text"; text: string };

const PROMPT = `You are calibrating a broadcast soccer camera to the pitch. For EACH numbered frame, estimate where the visible image maps onto the real pitch, using visible landmarks such as the halfway line, center circle, penalty boxes, goals, touchlines, and penalty arcs.

Return:
- lengthMin: pitch length % at the LEFT edge of the image (0 = left goal line, 50 = halfway line, 100 = right goal line).
- lengthMax: pitch length % at the RIGHT edge of the image.
- topImageY: image vertical % where the playable grass begins; crowd/stands above this should not map to pitch width.
- confidence: 0-1.

Examples:
- left penalty area visible: lengthMin around 0, lengthMax around 45.
- right penalty area visible: lengthMin around 55, lengthMax around 100.
- full-pitch wide shot: lengthMin around 0, lengthMax around 100.

Return ONLY a JSON array, one object per frame in order:
[{ "lengthMin": 0, "lengthMax": 45, "topImageY": 18, "confidence": 0.75 }]`;

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
  if (
    !raw ||
    typeof raw.lengthMin !== "number" ||
    typeof raw.lengthMax !== "number" ||
    typeof raw.topImageY !== "number"
  ) {
    return null;
  }
  if ((raw.confidence ?? 1) < 0.35) return null;
  const lengthMin = clamp(raw.lengthMin, 0, 100);
  const lengthMax = clamp(raw.lengthMax, 0, 100);
  if (lengthMax - lengthMin < 8) return null;
  return {
    lengthMin,
    lengthMax,
    topImageY: clamp(raw.topImageY, 0, 90),
    confidence: clamp(raw.confidence ?? 0.5, 0, 1),
  };
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) return NextResponse.json({ views: [] });

    const { frames } = (await req.json()) as PitchRequest;
    if (!frames?.length) return NextResponse.json({ views: [] });

    const client = new Anthropic({ apiKey });
    const content: Array<ImageBlock | TextBlock> = [{ type: "text", text: PROMPT }];
    frames.forEach((frame, i) => {
      content.push({ type: "text", text: `Frame ${i + 1} (${frame.timestamp.toFixed(1)}s):` });
      content.push({
        type: "image",
        source: { type: "base64", media_type: "image/jpeg", data: frame.base64 },
      });
    });

    const response = await client.messages.create({
      model: PITCH_MODEL,
      max_tokens: 1500,
      messages: [{ role: "user", content }],
    });

    const raw = response.content[0]?.type === "text" ? response.content[0].text.trim() : "[]";
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
    const parsed = JSON.parse(cleaned) as RawView[];

    return NextResponse.json({
      views: frames.map((frame, i) => ({
        timestamp: frame.timestamp,
        pitchView: normalizeView(parsed[i]),
      })),
    });
  } catch (err) {
    console.error("[/api/analyze/pitch]", err);
    return NextResponse.json({ views: [] });
  }
}
