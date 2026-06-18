export type TeamId = "home" | "away";

export type PlayerAction =
  | "running"
  | "standing"
  | "passing"
  | "shooting"
  | "tackling"
  | "jumping"
  | "goalkeeping"
  | "dribbling";

export type EventType =
  | "pass"
  | "shot"
  | "tackle"
  | "goal"
  | "save"
  | "corner"
  | "goal-kick"
  | "freekick"
  | "foul"
  | "card_yellow"
  | "card_red"
  | "card_unknown"
  | "offside"
  | "throw-in"
  | "dribble";

export type FormationType = "4-4-2" | "4-3-3" | "4-5-1" | "3-5-2" | "3-4-3";

export interface Position {
  x: number; // 0–100, left-to-right
  y: number; // 0–100, top-to-bottom
}

export interface Player {
  id: string;
  number: number;
  team: TeamId;
  position: Position;
  action: PlayerAction;
  role: "gk" | "def" | "mid" | "fwd";
}

export interface MatchEvent {
  id: string;
  timestamp: number; // seconds from video start
  type: EventType;
  team?: TeamId;
  playerId?: string;
  position?: Position;
  description: string;
  confidence: number;
  isKeyMoment: boolean;
  semanticLabel?: string;
  evidenceUsed?: string[];
  conflicts?: string[];
  pipelineFlag?: "missed_detection" | "scoreboard_conflict" | "replay_suspected" | "verifier_conflict" | "low_confidence";
  xg?: number;
  // How the xG figure was produced: "vision" = Claude looked at the shot frame
  // (keeper/defender positions, body shape, angle); "positional" = estimateShotXg
  // formula from event location only.
  xgSource?: "vision" | "positional";
  source?: "cv" | "heuristic" | "llm" | "scoreboard" | "fallback";
}

// Win/draw/loss style projection for the clip. Produced by the vision synthesis
// model from numeric metrics + key frames, not a hardcoded formula. Scoped to the
// passage of play in the clip, not a full-match prediction.
export interface OutcomeProjection {
  homeWin: number; // 0–100, sums to ~100 with draw + awayWin
  draw: number;
  awayWin: number;
  reasoning: string;
  source: "vision" | "fallback";
}

// Estimated mapping from this frame's broadcast image to true pitch coordinates,
// used only by the tactical board so a half-field camera shot lands in the right
// half of the pitch instead of being normalized to the middle. The ring/overlay
// tracking ignores this and keeps using raw image-space `position`.
//   lengthMin/lengthMax: pitch length % (0 = left goal line, 100 = right) visible
//     at the image's left / right edges.
//   topImageY: image y % (0 = top, 100 = bottom) where the playing field begins
//     (far touchline / horizon), excluding crowd and stands above it.
export interface PitchView {
  lengthMin: number;
  lengthMax: number;
  topImageY: number;
}

export interface FrameData {
  frameIndex: number;
  timestamp: number;
  pitchView?: PitchView;
  players: Player[];
  ballPosition?: Position;
  events: MatchEvent[];
  possession: TeamId | "contested";
  // Player with closest contact to the ball — used for cross-frame pass counting
  possessingPlayer?: { team: TeamId; playerId: string };
  // Match officials detected separately from players (currently only populated by
  // the YOLO worker when the loaded model has a distinct referee class, e.g. soccana).
  referees?: Position[];
  // Scoreboard overlay reading for this frame, when legible. Tracked per-frame
  // (rather than asking Claude to compare across batches) so a goal can be
  // confirmed deterministically from the score increasing between any two frames,
  // even when they land in different review batches or a batch's event
  // confirmation otherwise fails — see synthesizeGoalsFromScoreboard.
  scoreboard?: { home: number; away: number; homeLabel?: string; awayLabel?: string } | null;
}

export interface TeamStats {
  possession: number;
  passes: number;
  passAccuracy: number;
  shots: number;
  shotsOnTarget: number;
  tackles: number;
  fouls: number;
  corners: number;
  goals: number;
  distanceCovered: number; // approx total meters covered by all players this team
  expectedGoals?: number;
  metricConfidence?: Partial<Record<"possession" | "passes" | "shots" | "xg" | "distance", number>>;
}

export interface TeamAnalysis {
  id: TeamId;
  name: string;
  color: string;
  formation: FormationType;
  stats: TeamStats;
  heatmap: number[][]; // 10×10 grid, 0–1 density values
  averagePosition: Position;
}

export interface CoachingInsight {
  id: string;
  category: "attacking" | "defensive" | "possession" | "tactical" | "physical";
  priority: "critical" | "high" | "medium" | "low";
  title: string;
  observation: string;
  recommendation: string;
  affectedTeam: TeamId | "both";
  source?: "claude" | "fallback";
  evidenceUsed?: string[];
}

export interface MatchAnalysis {
  id: string;
  processedAt: string;
  videoDuration: number;
  framesAnalyzed: number;
  homeTeam: TeamAnalysis;
  awayTeam: TeamAnalysis;
  frames: FrameData[];
  keyEvents: MatchEvent[];
  eventConflicts?: Array<{
    timestamp: number;
    type: EventType;
    team?: TeamId;
    description: string;
    conflicts: string[];
    evidenceUsed?: string[];
    pipelineFlag?: MatchEvent["pipelineFlag"];
  }>;
  analysisWarnings?: string[];
  insights: CoachingInsight[];
  outcome?: OutcomeProjection;
  // `score` is the full-match scoreboard reading at the end of the clip (may carry
  // goals scored before the upload). `clipGoals` is only the goals scored within
  // the uploaded clip (scoreboard delta) — use this for finishing/conversion so
  // they stay consistent with in-clip shots and xG.
  score: { home: number; away: number };
  clipGoals: { home: number; away: number };
  processingMethod: "ai" | "demo";
}

// Payload sent from client to /api/analyze (demo only)
export interface AnalyzeRequest {
  frames: Array<{ base64: string; timestamp: number }>;
  demo?: boolean;
}

// Payload for per-frame analysis
export interface AnalyzeFrameRequest {
  base64: string;
  timestamp: number;
  frameIndex: number;
  prevBase64?: string;    // previous frame for motion context
  prevTimestamp?: number;
}

// Payload for final summarize step
export interface SummarizeRequest {
  frames: FrameData[];
  // Warnings from the /api/analyze/events review step (e.g. a batch's Claude call
  // failed and fell back to heuristic-only events) — surfaced to the coach instead
  // of being silently discarded, since it explains why some events lack full review.
  eventReviewWarnings?: string[];
  // Curated key frames (downsized JPEG base64) for the vision synthesis pass —
  // event frames plus evenly-sampled coverage. Lets the summary model actually
  // see the play instead of reasoning over numbers alone. Omitted → text-only
  // synthesis fallback.
  keyFrames?: Array<{ timestamp: number; base64: string }>;
}

export interface AnalyzeEventsRequest {
  frames: FrameData[];
  images: Array<{ base64: string; timestamp: number }>;
}
