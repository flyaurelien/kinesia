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

// Aggregate of one scalar over gait cycles/steps: mean, SD, sample count.
export type GaitStat = { mean: number | null; sd: number | null; n: number };

// One clinical angle resampled to 0-100% of the gait cycle (101 points),
// aggregated as mean +/- SD across all same-side strides.
export type GaitCycle = {
  n_cycles: number;
  mean: number[] | null;
  sd: number[] | null;
};

// A detected gait event (heel-strike / toe-off) on one side.
export type GaitEvent = {
  frame: number;
  time_s: number;
  side: "left" | "right";
  type: "heel_strike" | "toe_off";
};

// The clinical gait-analysis layer for a run (see src/.../gait.py).
export type RunGait = {
  params: { filter: string; order: number; cutoff_hz: number; zero_phase: boolean };
  neutralReference: {
    applied: boolean;
    method: string;
    staticFrames: number;
    staticDurationS: number;
    offsetsDeg: Record<string, number>;
    note: string;
  };
  events: GaitEvent[];
  spatiotemporal: {
    walkingDetected: boolean;
    cadenceStepsPerMin: number | null;
    stepTimeS: GaitStat;
    stepLengthM: GaitStat;
    strideTimeS: GaitStat;
    strideLengthM: GaitStat;
    walkingSpeedMS: GaitStat;
    stancePct: GaitStat;
    swingPct: GaitStat;
    doubleSupportPct: GaitStat;
  };
  // cycles[side][signalId] -> normalized cycle. signalId matches a RunSignal id.
  cycles: {
    left: Record<string, GaitCycle>;
    right: Record<string, GaitCycle>;
  };
};

// A static object reconstructed once for the whole clip and placed in the same
// metric world as the subject (see scripts/place_scene_object.py).
export type SceneObject = {
  name: string;
  meshUrl: string;
  // Metres per mesh unit — the object model outputs a normalised shape, so this
  // is solved against the subject's floor rather than taken from it.
  scale: number;
  upAxis: "X" | "Y" | "Z";
  centreWorld: [number, number, number];
  // Floor position and heading fitted so the mesh's projected silhouette
  // matches the observed mask — the mask's bbox centre is not the projection
  // of the 3D centre, and says nothing about which way the object faces.
  positionWorld: [number, number, number] | null;
  yawRad: number | null;
  fitIou: number | null;
  solved: { depthM: number; heightM: number; widthM: number; floorZ: number };
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
  gait?: RunGait | null;
  frames: RunFrame[];
  sceneObjects: SceneObject[];
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
