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
  | "freekick"
  | "foul"
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
}

export interface FrameData {
  frameIndex: number;
  timestamp: number;
  players: Player[];
  ballPosition?: Position;
  events: MatchEvent[];
  possession: TeamId | "contested";
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
  insights: CoachingInsight[];
  score: { home: number; away: number };
  processingMethod: "ai" | "demo";
}

// Payload sent from client to /api/analyze
export interface AnalyzeRequest {
  frames: Array<{ base64: string; timestamp: number }>;
  demo?: boolean;
}
