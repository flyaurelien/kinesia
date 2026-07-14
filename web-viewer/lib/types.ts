// Lightweight overview of a run for list/index views (no per-frame data).
// Which chosen subject a run reconstructs. Multi-subject selections spawn one
// run per subject; sibling runs share `trackFile` and the viewer reunites them
// in a single 3D scene, colouring each by its detect-step palette colour.
export type RunSubject = {
  index: number;
  id: string | null;
  label: string | null;
  color: string | null;
  trackFile: string | null;
};

export type RunSummary = {
  id: string;
  processedFrames: number;
  hasMeshes: boolean;
  fps: number | null;
  updatedAt: string | null;
  createdAt?: string | null;
  inferenceTarget?: "body" | "hand" | null;
  latestAnalysisId?: string | null;
  qaStatus?: string | null;
  subject?: RunSubject | null;
};

// Reference point used to ground the reconstructed scene in world space.
export type WorldAnchor = {
  floor_y: number;
  center_x: number;
  center_z: number;
};

// Camera/scene framing the viewer uses to position the 3D space.
export type SpaceViewInfo = {
  mode?: string;
  world_anchor?: WorldAnchor | null;
  view_state?: {
    scale: number;
    center_x: number;
    center_y: number;
  } | null;
};

// Per-frame reconstruction + tracking output consumed by the viewer.
export type RunFrame = {
  index: number;
  videoFrame: number;
  meshFile: string | null;
  meshUrl: string | null;
  subjectPresent?: boolean;
  inferenceStatus?: string | null;
  subjectTrackingStatus?: string | null;
  cameraComp: [number, number, number];
  jointsCam?: Array<[number, number, number]> | null;
  // Tracked subject box in original-video pixel coords [x1, y1, x2, y2] and the
  // per-frame pinhole focal length — used to overlay the tracking box + the
  // projected skeleton on the original video.
  bbox?: [number, number, number, number] | null;
  focalLength?: number | null;
  // Optional fields added for backward compatibility with older runs.
  // These are computed server-side (see lib/runs.ts) and consumed by the viewer.
  rootWorldRaw?: [number, number, number];
  rootWorldStabilized?: [number, number, number];
  footContact?: {
    left: boolean;
    right: boolean;
    support: "left" | "right" | "both" | "none";
  };
  // Subject-tracking confidence for this frame (0..1), null when no subject.
  trackingScore?: number | null;
  // Offline identity-resolution confidence (0..1) and ambiguity flag — drives
  // the human-review queue (flagged = a crossing/look-alike worth confirming).
  identityConfidence?: number | null;
  identityAmbiguous?: boolean;
};

// A named time series (one value per frame) plotted in the signal charts.
export type RunSignal = {
  id: string;
  label: string;
  unit: string;
  description: string;
  values: Array<number | null>;
};

// Full payload for a single run, including every frame and signal (viewer page).
export type RunDetail = {
  id: string;
  analysisId?: string | null;
  inferenceTarget?: "body" | "hand" | null;
  processedFrames: number;
  hasMeshes: boolean;
  fps: number;
  spaceView: SpaceViewInfo | null;
  videoWidth: number | null;
  videoHeight: number | null;
  inputVideoUrl: string | null;
  previewVideoUrl: string | null;
  previewVideoTimebase: "processed" | "source";
  subject?: RunSubject | null;
  signals: RunSignal[];
  frames: RunFrame[];
  analyses?: Array<{
    analysisId: string;
    preset: string;
    createdAt: string | null;
    qaStatus: string | null;
  }>;
  qa?: {
    status: string;
    needs_review: boolean;
    tracking_score: number | null;
    joint_visibility_ratio: number | null;
    critical_joint_visibility_ratio: number | null;
    camera_motion_severity: number | null;
    reasons: string[];
  } | null;
};
