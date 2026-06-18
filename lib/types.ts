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
  detectionConfidence?: number;
  inferred?: boolean;
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
  source?: "cv" | "heuristic" | "llm" | "scoreboard" | "fallback";
}

export interface FrameData {
  frameIndex: number;
  timestamp: number;
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
  // Worker-side tracking quality. Wide frames have enough stable players from both
  // teams for tactical shape/network panels; closeups should still drive the video
  // overlay but should not collapse the tactical state.
  trackingQuality?: "wide" | "low_confidence" | "closeup";
  trackingCounts?: { players: number; home: number; away: number; inferred: number };
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
  score: { home: number; away: number };
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
}

export interface AnalyzeEventsRequest {
  frames: FrameData[];
  images: Array<{ base64: string; timestamp: number }>;
}
