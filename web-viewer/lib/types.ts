// Lightweight overview of a run for list/index views (no per-frame data).
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
  fogDetected?: boolean;
  fogScore?: number | null;
  fogScoreSmooth?: number | null;
  fogComponents?: Record<string, number | null>;
  // Subject-tracking confidence for this frame (0..1), null when no subject.
  trackingScore?: number | null;
};

// A named time series (one value per frame) plotted in the signal charts.
export type RunSignal = {
  id: string;
  label: string;
  unit: string;
  description: string;
  values: Array<number | null>;
};

// A contiguous span of frames flagged as a freezing-of-gait (FoG) event.
export type FogSegment = {
  startFrameIndex: number;
  endFrameIndex: number;
  startVideoFrame: number;
  endVideoFrame: number;
  durationSec: number;
};

// Run-level FoG detection summary (threshold + the resulting segments).
export type FogSummary = {
  threshold: number;
  detectedFrameCount: number;
  detectedRatio: number;
  segments: FogSegment[];
};

// Detection metrics for a single run, scored against its ground-truth labels.
export type DatasetRunMetrics = {
  precision: number | null;
  recall: number | null;
  f1Event: number | null;
  matchedEvents: number | null;
  falsePositiveEvents: number | null;
  falseNegativeEvents: number | null;
  meanOnsetLatencyMs: number | null;
  falsePositiveDurationPerMinS: number | null;
  nonInterpretableRate: number | null;
};

// Metrics averaged across the runs in one dataset split (tuning or holdout).
export type DatasetAggregateMetrics = {
  count: number;
  precision: number | null;
  recall: number | null;
  f1Event: number | null;
  meanOnsetLatencyMs: number | null;
  falsePositiveDurationPerMinS: number | null;
  nonInterpretableRate: number | null;
};

// Full evaluation report for a dataset: per-split aggregates + per-run breakdown.
export type DatasetEvaluationSummary = {
  datasetId: string;
  preset: string;
  generatedAt: string | null;
  runsEvaluated: number;
  splits: {
    tuning: DatasetAggregateMetrics;
    holdout: DatasetAggregateMetrics;
  };
  perRun: Array<{
    runId: string;
    split: "tuning" | "holdout";
    metrics: DatasetRunMetrics;
    qaStatus: string | null;
    needsReview: boolean;
  }>;
};

// A single ground-truth labeled interval (e.g. an annotated FoG episode).
export type LabeledEpisode = {
  label: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  startFrameIndex: number | null;
  endFrameIndex: number | null;
};

// Association between a run and the dataset it belongs to (labels + metadata).
export type RunDatasetLink = {
  datasetId: string;
  split: "tuning" | "holdout";
  patientId: string | null;
  sessionId: string | null;
  videoPath: string | null;
  notes: string | null;
  labelEventsPath: string | null;
  labeledEpisodes: LabeledEpisode[];
  latestEvaluation: {
    preset: string;
    generatedAt: string | null;
    metrics: DatasetRunMetrics | null;
  } | null;
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
  fog: FogSummary | null;
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
    event_confidence: number | null;
    reasons: string[];
  } | null;
  datasets?: RunDatasetLink[];
};
