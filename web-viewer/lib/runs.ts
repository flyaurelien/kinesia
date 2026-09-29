/** Analysis records shared by the server routes and the browser. */

export type RunState =
  | "new"
  | "preparing"
  | "uploading"
  | "queued"
  | "running"
  | "downloading"
  | "building"
  | "done"
  | "failed"
  | "cancelled";

export const ACTIVE_STATES: RunState[] = ["preparing", "uploading", "queued", "running", "downloading", "building"];

export type RunStatus = {
  state: RunState;
  message?: string;
  stage?: string | null;
  progress?: { done: number; total: number } | null;
  job?: string;
  pools?: string[];
  pool?: string | null;
  node?: string | null;
  gpu?: string | null;
  error?: string | null;
  started_at?: string;
  submitted_at?: string;
  updated_at?: string;
  finished_at?: string;
  gpu_started_at?: string | null;
  gpu_finished_at?: string | null;
  receipts?: {
    gpu?: string;
    tracking?: { frames: number; seconds: number };
    camera?: { focal: number; source: string };
    bodies?: { person_frames: number; seconds: number };
  };
  history?: { t: string; state: RunState; message?: string }[];
  pid?: number;
};

export type VideoInfo = { width: number; height: number; fps: number; frames: number; duration: number };

export type RunSummary = {
  id: string;
  name: string;
  created_at: string;
  original_name?: string;
  video: VideoInfo | null;
  status: RunStatus;
  /** Whether the local orchestrator process is alive (only meaningful while active). */
  worker_alive: boolean;
  people: number | null;
  has_poster: boolean;
  has_scene: boolean;
};

export function isActive(state: RunState): boolean {
  return ACTIVE_STATES.includes(state);
}
