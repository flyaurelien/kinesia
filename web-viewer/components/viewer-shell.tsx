"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent,
} from "react";

import { apiFetch, apiUrl } from "../lib/api-client";
import type { RunDetail, RunSignal, RunSummary } from "../lib/types";
import { ThreeSpaceViewer, preloadRunAssets, type RunAssetPreloadProgress } from "./viewer-three";
import { KinematicsPlot } from "./kinematics-plot";
import { VideoTrackingOverlay } from "./video-overlay";
import { parseAnnotationsFile } from "../lib/export";
import {
  bundleBaseName,
  buildBundleFiles,
  downloadBundle,
  DEFAULT_CHANNEL_FORMATS,
  type ChannelFormats,
  type ExportInterval,
} from "../lib/export-bundle";
import { WizardPanel } from "./wizard/wizard-shell";
import { useWizard, type WizardActions } from "./wizard/state";
import "./wizard/wizard.css";

type RunsResponse = { runs: RunSummary[] };
type RunDetailResponse = { run: RunDetail };
type JobsResponse = { jobs: GenerationJob[] };
type CreateJobResponse = { job: GenerationJob };

type GenerationJobStatus = "queued" | "running" | "completed" | "failed" | "canceled";
type InferenceTarget = "body" | "hand";
type PatientDetectionMode = "manual" | "auto";
type PlotGroup = "angles" | "verticalSpeed" | "acceleration" | "position" | "distance" | "speed" | "rotation";
type SubjectBox = { x: number; y: number; width: number; height: number };
type TimelineRemovedSegment = { id: string; startSec: number; endSec: number };
type TimelineVisibleSegment = {
  id: string;
  index: number;
  startSec: number;
  endSec: number;
  displayStartSec: number;
  displayEndSec: number;
};
type SubjectPreview = {
  box: SubjectBox;
  frameSec: number;
  frameIndex: number;
  fps: number;
  source: string | null;
  candidateCount: number;
};
type CameraStatus = "idle" | "starting" | "ready" | "recording" | "error";
type VideoEditTool = "subject" | "crop";
type AnnotationDragMode = "create" | "move" | "resize-start" | "resize-end";
type AnnotationDragState = {
  mode: AnnotationDragMode;
  segmentIndex: number | null;
  startFrame: number;
  originalStart: number;
  originalEnd: number;
} | null;
type FogAnnotationSource = "auto_predicted" | "auto_corrected" | "manual";
type FogAnnotationSegment = {
  startFrameIndex: number;
  endFrameIndex: number;
  startVideoFrame: number;
  endVideoFrame: number;
  startSec: number;
  endSec: number;
  label: "fog";
  source: FogAnnotationSource;
  createdAt: string;
  // Original model-predicted frame range (post-processing applied) carried
  // through edits so we can render "what the model said" vs "what the human
  // changed". Undefined for segments created entirely by a human.
  predictedStartFrameIndex?: number;
  predictedEndFrameIndex?: number;
};
type GoldTruthSegment = {
  startFrameIndex: number;
  endFrameIndex: number;
  startSec: number;
  endSec: number;
  label: string;
  datasetId: string;
  split: "tuning" | "holdout";
  labelEventsPath: string | null;
  // Provenance carried over from an imported file (e.g. manual / auto_corrected
  // / auto_predicted, an ELAN tier id, or "csv"). Undefined for dataset gold.
  source?: string;
};
type BenchmarkMetrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
};
type DeletedFogPrediction = FogAnnotationSegment & {
  deletedAt: string;
  deletedAtFrameIndex: number;
  originalSource: Extract<FogAnnotationSource, "auto_predicted" | "auto_corrected">;
};
type DraftAnnotation = { start: number; end: number } | null;
type FogHistorySnapshot = { manual: FogAnnotationSegment[]; deleted: DeletedFogPrediction[] };

type GenerationJob = {
  id: string;
  runId: string;
  videoFileName: string;
  inferenceTarget?: InferenceTarget;
  sam3TextPrompts?: string[];
  status: GenerationJobStatus;
  paused?: boolean;
  createdAt: string;
  updatedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  processedFrames: number;
  totalFrames: number | null;
  progressPercent: number | null;
  error: string | null;
};
type RunLoadState = RunAssetPreloadProgress & {
  runId: string;
};

const PLOT_COLORS = ["#38bdf8", "#fb923c", "#22c55e", "#f43f5e", "#a78bfa", "#facc15", "#14b8a6", "#f97316"];
const MANUAL_FOG_STORAGE_KEY = "kinesia.manualFogAnnotations.v1";
const DELETED_AUTO_FOG_STORAGE_KEY = "kinesia.deletedAutoFogAnnotations.v1";
const FOG_THRESHOLD_STORAGE_KEY = "kinesia.fogThresholdByRun.v1";
const FOG_POSTPROCESS_STORAGE_KEY = "kinesia.fogPostProcessSettings.v1";
const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "m4v", "avi", "mkv", "webm"]);
const DEFAULT_FOG_MIN_DURATION_SEC = 0.4;
const DEFAULT_FOG_MERGE_GAP_SEC = 0.6;
const DEFAULT_FOG_THRESHOLD = 0.37;
const MIN_TRIM_GAP_SEC = 0.1;
const DIRECT_UPLOAD_LIMIT_BYTES = 16 * 1024 * 1024;
const UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024;
const PUBLIC_BASIC_UI = process.env.NEXT_PUBLIC_KINESIA_BASIC_UI === "1";
const DEFAULT_SUBJECT_PROMPT = "person";
// How many evenly-spaced timestamps the multi-frame auto-detect preview
// samples across the kept timeline. Enough to spot identity swaps without
// being slow.
const AUTO_PREVIEW_SAMPLE_COUNT = 4;

const GROUPS: Array<{ id: PlotGroup; label: string; unit: string; predicate: (signal: RunSignal) => boolean }> = [
  {
    id: "angles",
    label: "Angles",
    unit: "deg",
    predicate: (signal) => signal.unit === "deg",
  },
  {
    id: "verticalSpeed",
    label: "Vertical Speed",
    unit: "m/s",
    predicate: (signal) => signal.unit === "m/s" && /\.vz$/.test(signal.id),
  },
  {
    id: "acceleration",
    label: "Acceleration",
    unit: "m/s^2",
    predicate: (signal) => signal.unit === "m/s^2",
  },
  {
    id: "position",
    label: "Position",
    unit: "m",
    predicate: (signal) =>
      signal.unit === "m" &&
      !signal.id.includes("distance") &&
      !signal.id.includes("delta") &&
      !signal.id.endsWith(".speed"),
  },
  {
    id: "distance",
    label: "Distance",
    unit: "m",
    predicate: (signal) => signal.unit === "m" && signal.id.includes("distance"),
  },
  {
    id: "speed",
    label: "Speed",
    unit: "m/s",
    predicate: (signal) => signal.unit === "m/s" && !/\.vz$/.test(signal.id),
  },
  {
    id: "rotation",
    label: "Rotation",
    unit: "deg/s",
    predicate: (signal) => signal.unit === "deg/s",
  },
];

const DEFAULT_BY_GROUP: Record<PlotGroup, string[]> = {
  angles: [
    "joint.hip.left.angle_deg",
    "joint.hip.right.angle_deg",
    "joint.knee.left.angle_deg",
    "joint.knee.right.angle_deg",
    "joint.ankle.left.angle_deg",
    "joint.ankle.right.angle_deg",
  ],
  verticalSpeed: ["joint.knee.left.vz", "joint.knee.right.vz", "joint.ankle.left.vz", "joint.ankle.right.vz"],
  acceleration: ["joint.knee.left.acceleration", "joint.knee.right.acceleration"],
  position: ["joint.pelvis.z", "joint.knee.left.z", "joint.knee.right.z", "joint.ankle.left.z", "joint.ankle.right.z"],
  distance: ["joint.knee.left.distance", "joint.knee.right.distance", "joint.ankle.left.distance", "joint.ankle.right.distance"],
  speed: ["joint.com.speed", "joint.knee.left.speed", "joint.knee.right.speed", "joint.ankle.left.speed", "joint.ankle.right.speed"],
  rotation: [
    "joint.knee.left.angular_velocity_deg",
    "joint.knee.right.angular_velocity_deg",
    "joint.hip.left.angular_velocity_deg",
    "joint.hip.right.angular_velocity_deg",
    "turn.yaw_rate",
  ],
};

const PLOT_JOINT_OPTIONS: Array<{ index: number; label: string }> = [
  { index: 5, label: "Left shoulder" },
  { index: 6, label: "Right shoulder" },
  { index: 9, label: "Left hip" },
  { index: 10, label: "Right hip" },
  { index: 11, label: "Left knee" },
  { index: 12, label: "Right knee" },
  { index: 13, label: "Left ankle" },
  { index: 14, label: "Right ankle" },
  { index: 15, label: "Left toe" },
  { index: 18, label: "Right toe" },
  { index: 17, label: "Left heel" },
  { index: 20, label: "Right heel" },
];
// All selectable joints; the picker offers every one of these.
const PLOT_JOINT_INDEX_SET = new Set(PLOT_JOINT_OPTIONS.map((joint) => joint.index));
// Default selection — the lower-limb gait joints, so plots stay focused out of
// the box while still letting the user add any other joint.
const DEFAULT_ACTIVE_PLOT_JOINTS = [9, 10, 11, 12, 13, 14];

// Constrain a number to the inclusive [min, max] range.
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Compact action-button icons (Material-style paths, fill = currentColor).
const ACTION_ICON_PATHS: Record<string, string> = {
  undo: "M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z",
  redo: "M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z",
  restore: "M13 3a9 9 0 0 0-9 9H1l3.96 3.96L9 12H6a7 7 0 1 1 2.05 4.95l-1.42 1.42A9 9 0 1 0 13 3zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z",
  split: "M14 4l2.29 2.29-2.88 2.88 1.42 1.42 2.88-2.88L20 12V4zM10 4H2v8l2.29-2.29 4.71 4.7V20h2v-6.41l-5.29-5.3z",
  delete: "M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z",
  clear: "M15 16h4v2h-4zM15 8h7v2h-7zM15 12h6v2h-6zM3 18c0 1.1.9 2 2 2h6c1.1 0 2-.9 2-2V8H3v10zM14 5h-3l-1-1H6L5 5H2v2h12z",
  download: "M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z",
  restart: "M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z",
  pause: "M6 5h4v14H6zM14 5h4v14h-4z",
  play: "M8 5v14l11-7z",
  stop: "M6 6h12v12H6z",
  minus: "M5 11h14v2H5z",
};

// Render one of the compact action-button icons by name.
function ActionIcon({ name }: { name: keyof typeof ACTION_ICON_PATHS }) {
  return (
    // display:block removes the inline-SVG baseline gap so the glyph centres
    // exactly in grid/flex icon buttons.
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" style={{ display: "block" }}>
      <path fill="currentColor" d={ACTION_ICON_PATHS[name]} />
    </svg>
  );
}

// Format a numeric reading with a unit, using fewer decimals as magnitude grows.
function fmtValue(value: number | null | undefined, unit: string): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  const digits = Math.abs(value) >= 100 ? 1 : Math.abs(value) >= 10 ? 2 : 3;
  return `${value.toFixed(digits)} ${unit}`;
}

// Format seconds as M:SS.xx for the trim/timeline readouts.
function fmtTrimTime(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

// Clamp a removed-segment span into the video and reject spans too short to keep.
function normalizeCutSegment(startSec: number, endSec: number, durationSec: number): { startSec: number; endSec: number } | null {
  const start = clamp(Math.min(startSec, endSec), 0, durationSec);
  const end = clamp(Math.max(startSec, endSec), 0, durationSec);
  if (end - start < MIN_TRIM_GAP_SEC) {
    return null;
  }
  return { startSec: start, endSec: end };
}

// Sort, clamp and de-dupe cut points, dropping any closer than MIN_TRIM_GAP_SEC.
function normalizeTimelineCutPoints(points: number[], durationSec: number): number[] {
  const sorted = points
    .map((point) => clamp(point, 0, durationSec))
    .filter((point) => point >= MIN_TRIM_GAP_SEC && durationSec - point >= MIN_TRIM_GAP_SEC)
    .sort((a, b) => a - b);
  const normalized: number[] = [];
  for (const point of sorted) {
    const last = normalized.at(-1);
    if (last === undefined || point - last >= MIN_TRIM_GAP_SEC) {
      normalized.push(point);
    }
  }
  return normalized;
}

// Collapse overlapping or touching removed segments into a minimal set of spans.
function mergeRemovedSegments(segments: TimelineRemovedSegment[]): TimelineRemovedSegment[] {
  const sorted = segments
    .sort((a, b) => a.startSec - b.startSec);
  const merged: TimelineRemovedSegment[] = [];
  for (const segment of sorted) {
    const last = merged.at(-1);
    if (last && segment.startSec <= last.endSec + 0.01) {
      last.endSec = Math.max(last.endSec, segment.endSec);
    } else {
      merged.push({ ...segment });
    }
  }
  return merged;
}

// Compute the kept (visible) timeline pieces after trim bounds, cut points and
// removed spans are applied, assigning each a contiguous display offset so the
// edited video reads as one gap-free clip.
function buildTimelineVisibleSegments(
  trimStartSec: number,
  trimEndSec: number,
  cutPointsSec: number[],
  removedSegments: TimelineRemovedSegment[],
): TimelineVisibleSegment[] {
  const start = Math.min(trimStartSec, trimEndSec);
  const end = Math.max(trimStartSec, trimEndSec);
  if (end - start < MIN_TRIM_GAP_SEC) {
    return [];
  }
  const boundaries = [start, ...cutPointsSec.filter((point) => point > start + MIN_TRIM_GAP_SEC && point < end - MIN_TRIM_GAP_SEC), end];
  const visible: TimelineVisibleSegment[] = [];
  let displayCursor = 0;

  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const segmentStart = boundaries[index];
    const segmentEnd = boundaries[index + 1];
    let cursor = segmentStart;
    for (const removed of removedSegments) {
      if (removed.endSec <= cursor || removed.startSec >= segmentEnd) {
        continue;
      }
      const keptEnd = Math.min(removed.startSec, segmentEnd);
      if (keptEnd - cursor >= MIN_TRIM_GAP_SEC) {
        const duration = keptEnd - cursor;
        visible.push({
          id: `segment-${cursor.toFixed(3)}-${keptEnd.toFixed(3)}`,
          index: visible.length + 1,
          startSec: cursor,
          endSec: keptEnd,
          displayStartSec: displayCursor,
          displayEndSec: displayCursor + duration,
        });
        displayCursor += duration;
      }
      cursor = Math.max(cursor, Math.min(removed.endSec, segmentEnd));
    }
    if (segmentEnd - cursor >= MIN_TRIM_GAP_SEC) {
      const duration = segmentEnd - cursor;
      visible.push({
        id: `segment-${cursor.toFixed(3)}-${segmentEnd.toFixed(3)}`,
        index: visible.length + 1,
        startSec: cursor,
        endSec: segmentEnd,
        displayStartSec: displayCursor,
        displayEndSec: displayCursor + duration,
      });
      displayCursor += duration;
    }
  }

  return visible;
}

// Map an offset along the edited (gap-free) timeline back to a time in the
// original source video.
function originalTimeFromTimelineOffset(segments: TimelineVisibleSegment[], offsetSec: number): number {
  if (segments.length === 0) {
    return 0;
  }
  const last = segments[segments.length - 1];
  const safeOffset = clamp(offsetSec, 0, last.displayEndSec);
  for (const segment of segments) {
    if (safeOffset <= segment.displayEndSec + 0.001) {
      return clamp(segment.startSec + (safeOffset - segment.displayStartSec), segment.startSec, segment.endSec);
    }
  }
  return last.endSec;
}

// Inverse of originalTimeFromTimelineOffset: map a source-video time to its
// offset on the edited (gap-free) timeline.
function timelineOffsetFromOriginalTime(segments: TimelineVisibleSegment[], timeSec: number): number {
  if (segments.length === 0) {
    return 0;
  }
  const first = segments[0];
  if (timeSec <= first.startSec) {
    return 0;
  }
  for (const segment of segments) {
    if (timeSec <= segment.endSec) {
      return segment.displayStartSec + clamp(timeSec - segment.startSec, 0, segment.endSec - segment.startSec);
    }
  }
  return segments[segments.length - 1].displayEndSec;
}

// A job is "active" while it is still queued or running.
function isActiveJob(job: GenerationJob): boolean {
  return job.status === "queued" || job.status === "running";
}

// Human-readable label for a generation-job status.
function statusLabel(status: GenerationJobStatus): string {
  if (status === "queued") return "Queued";
  if (status === "running") return "Running";
  if (status === "completed") return "Done";
  if (status === "canceled") return "Canceled";
  return "Failed";
}

// Look up a plot group's config (label/unit/predicate), defaulting to the first.
function signalGroup(group: PlotGroup): (typeof GROUPS)[number] {
  return GROUPS.find((item) => item.id === group) ?? GROUPS[0];
}

// Keep non-joint signals; for joint signals, only those for an active joint.
function signalMatchesActiveJoints(signal: RunSignal, activeJointIndices: number[]): boolean {
  if (!signal.id.startsWith("joint.")) {
    return true;
  }
  const index = jointIndexFromSignalId(signal.id);
  return index !== null && activeJointIndices.includes(index);
}

// All signals in a plot group that belong to the currently active joints, sorted.
function signalsForGroup(runDetail: RunDetail | null, group: PlotGroup, activeJointIndices: number[]): RunSignal[] {
  if (!runDetail) {
    return [];
  }
  const config = signalGroup(group);
  return runDetail.signals
    .filter(config.predicate)
    .filter((signal) => signalMatchesActiveJoints(signal, activeJointIndices))
    .sort((a, b) => a.label.localeCompare(b.label));
}

// Pick the default selected signals for a group: the configured defaults that
// exist, else the first few available, capped at 8.
function defaultSignalIds(runDetail: RunDetail | null, group: PlotGroup, activeJointIndices: number[]): string[] {
  const available = signalsForGroup(runDetail, group, activeJointIndices);
  const ids = new Set(available.map((signal) => signal.id));
  const defaults = DEFAULT_BY_GROUP[group].filter((id) => ids.has(id));
  return (defaults.length > 0 ? defaults : available.slice(0, 4).map((signal) => signal.id)).slice(0, 8);
}

// Resolve the joint index a signal id refers to (named joints first, then the
// numeric "joint.N." form), or null when it is not a joint signal.
function jointIndexFromSignalId(signalId: string): number | null {
  if (signalId.includes("shoulder.left")) return 5;
  if (signalId.includes("shoulder.right")) return 6;
  if (signalId.includes("hip.left")) return 9;
  if (signalId.includes("hip.right")) return 10;
  if (signalId.includes("knee.left")) return 11;
  if (signalId.includes("knee.right")) return 12;
  if (signalId.includes("ankle.left")) return 13;
  if (signalId.includes("ankle.right")) return 14;
  if (signalId.includes("toe.left")) return 15;
  if (signalId.includes("toe.right")) return 18;
  if (signalId.includes("heel.left")) return 17;
  if (signalId.includes("heel.right")) return 20;
  const match = signalId.match(/^joint\.(\d+)\./);
  return match ? Number(match[1]) : null;
}

// Source-video time (seconds) for a processed frame, using its original
// videoFrame number so trimmed/cut runs stay aligned to the source clip.
function currentVideoTime(runDetail: RunDetail, frameIndex: number): number {
  const frame = runDetail.frames[frameIndex];
  return (frame?.videoFrame ?? frameIndex) / Math.max(1, runDetail.fps);
}

// Time on the processed (gap-free) preview timeline, where frame N maps to N/fps.
function processedTimelineVideoTime(runDetail: RunDetail, frameCursor: number): number {
  if (runDetail.frames.length === 0) {
    return 0;
  }
  const fps = Math.max(1, runDetail.fps || 30);
  return clamp(frameCursor, 0, runDetail.frames.length - 1) / fps;
}

// Video currentTime to show for a frame cursor, picking the source vs processed
// timebase depending on how the run's preview video was assembled.
function displayedVideoTime(runDetail: RunDetail, frameCursor: number): number {
  if (runDetail.previewVideoTimebase === "source") {
    return currentVideoTime(runDetail, clamp(Math.round(frameCursor), 0, Math.max(0, runDetail.frames.length - 1)));
  }
  return processedTimelineVideoTime(runDetail, frameCursor);
}

// Nearest processed-frame index for a source-video time, binary-searching the
// frames' original videoFrame numbers.
function frameIndexFromVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.frames.length === 0 || !Number.isFinite(timeSec)) {
    return 0;
  }
  const targetFrame = timeSec * Math.max(1, runDetail.fps);
  let lo = 0;
  let hi = runDetail.frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if ((runDetail.frames[mid]?.videoFrame ?? mid) < targetFrame) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const prev = Math.max(0, lo - 1);
  const prevDistance = Math.abs((runDetail.frames[prev]?.videoFrame ?? prev) - targetFrame);
  const currDistance = Math.abs((runDetail.frames[lo]?.videoFrame ?? lo) - targetFrame);
  return prevDistance <= currDistance ? prev : lo;
}

// Processed-frame index for a processed-timeline time (round of time * fps).
function frameIndexFromProcessedTimelineVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.frames.length === 0 || !Number.isFinite(timeSec)) {
    return 0;
  }
  const fps = Math.max(1, runDetail.fps || 30);
  return clamp(Math.round(timeSec * fps), 0, runDetail.frames.length - 1);
}

// Frame index for the displayed video time, choosing source vs processed timebase.
function frameIndexFromDisplayedVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.previewVideoTimebase === "source") {
    return frameIndexFromVideoTime(runDetail, timeSec);
  }
  return frameIndexFromProcessedTimelineVideoTime(runDetail, timeSec);
}

// Fractional frame cursor for a source-video time (interpolates between frames
// for smooth playhead motion), via binary search on videoFrame numbers.
function frameCursorFromVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.frames.length === 0 || !Number.isFinite(timeSec)) {
    return 0;
  }
  const targetFrame = timeSec * Math.max(1, runDetail.fps);
  let lo = 0;
  let hi = runDetail.frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if ((runDetail.frames[mid]?.videoFrame ?? mid) < targetFrame) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo === 0) {
    return 0;
  }
  const prev = lo - 1;
  const prevFrame = runDetail.frames[prev]?.videoFrame ?? prev;
  const currFrame = runDetail.frames[lo]?.videoFrame ?? lo;
  const span = Math.max(1e-6, currFrame - prevFrame);
  return prev + clamp((targetFrame - prevFrame) / span, 0, 1);
}

// Fractional frame cursor for a processed-timeline time (time * fps, clamped).
function frameCursorFromProcessedTimelineVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.frames.length === 0 || !Number.isFinite(timeSec)) {
    return 0;
  }
  const fps = Math.max(1, runDetail.fps || 30);
  return clamp(timeSec * fps, 0, runDetail.frames.length - 1);
}

// Fractional frame cursor for the displayed video time, choosing the timebase.
function frameCursorFromDisplayedVideoTime(runDetail: RunDetail, timeSec: number): number {
  if (runDetail.previewVideoTimebase === "source") {
    return frameCursorFromVideoTime(runDetail, timeSec);
  }
  return frameCursorFromProcessedTimelineVideoTime(runDetail, timeSec);
}

// Turn two drag corners (any order) into a normalized [0,1] {x,y,width,height} box.
function normalizeBox(startX: number, startY: number, endX: number, endY: number): SubjectBox {
  const x1 = clamp(Math.min(startX, endX), 0, 1);
  const y1 = clamp(Math.min(startY, endY), 0, 1);
  const x2 = clamp(Math.max(startX, endX), 0, 1);
  const y2 = clamp(Math.max(startY, endY), 0, 1);
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

// Convert a normalized subject box into a pixel "x1,y1,x2,y2" prompt string in
// the (optionally cropped) source frame's coordinate space, or null if degenerate.
function subjectBoxToPrompt(box: SubjectBox, video: HTMLVideoElement | null, cropBox: SubjectBox | null): string | null {
  const width = video?.videoWidth ?? 0;
  const height = video?.videoHeight ?? 0;
  if (!width || !height || box.width < 0.02 || box.height < 0.02) {
    return null;
  }
  const crop = validCropBox(cropBox) ? cropBox : { x: 0, y: 0, width: 1, height: 1 };
  const cropWidth = width * crop.width;
  const cropHeight = height * crop.height;
  const x1 = Math.round(clamp((box.x - crop.x) * width, 0, cropWidth));
  const y1 = Math.round(clamp((box.y - crop.y) * height, 0, cropHeight));
  const x2 = Math.round(clamp((box.x + box.width - crop.x) * width, 0, cropWidth));
  const y2 = Math.round(clamp((box.y + box.height - crop.y) * height, 0, cropHeight));
  if (x2 <= x1 || y2 <= y1) {
    return null;
  }
  return `${x1},${y1},${x2},${y2}`;
}

// A crop is valid only if it is reasonably sized AND actually crops something
// (i.e. it is not effectively the full frame).
function validCropBox(box: SubjectBox | null): box is SubjectBox {
  return Boolean(box && box.width > 0.05 && box.height > 0.05 && (box.x > 0.001 || box.y > 0.001 || box.width < 0.999 || box.height < 0.999));
}

// A subject box must be at least a couple percent of the frame to be usable.
function validSubjectBox(box: SubjectBox | null): box is SubjectBox {
  return Boolean(box && box.width > 0.02 && box.height > 0.02);
}

// Serialize a valid crop box as "x,y,w,h" normalized fields for the run request.
function cropBoxToRaw(box: SubjectBox | null): string | null {
  if (!validCropBox(box)) {
    return null;
  }
  return [box.x, box.y, box.width, box.height].map((value) => value.toFixed(6)).join(",");
}

// Treat a file as video by MIME type, falling back to its extension.
function isVideoFile(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  return file.type.startsWith("video/") || VIDEO_EXTENSIONS.has(extension);
}

// De-duplicate video files (by path+size+mtime) and return them path-sorted.
function uniqueVideoFiles(files: File[]): File[] {
  const seen = new Set<string>();
  const out: File[] = [];
  for (const file of files) {
    if (!isVideoFile(file)) {
      continue;
    }
    const key = `${file.webkitRelativePath || file.name}:${file.size}:${file.lastModified}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(file);
  }
  return out.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
}

// Recursively flatten a drag-and-drop FileSystemEntry tree (file or directory)
// into a flat list of File objects.
async function filesFromFileSystemEntry(entry: any): Promise<File[]> {
  if (!entry) {
    return [];
  }
  if (entry.isFile) {
    return new Promise((resolve) => {
      entry.file((file: File) => resolve([file]), () => resolve([]));
    });
  }
  if (!entry.isDirectory) {
    return [];
  }
  const reader = entry.createReader();
  const entries: any[] = [];
  while (true) {
    const chunk = await new Promise<any[]>((resolve) => {
      reader.readEntries((items: any[]) => resolve(items), () => resolve([]));
    });
    if (chunk.length === 0) {
      break;
    }
    entries.push(...chunk);
  }
  const nested = await Promise.all(entries.map((item) => filesFromFileSystemEntry(item)));
  return nested.flat();
}

// Extract dropped files, walking folder entries when the browser exposes them
// (so dropping a directory recurses), else falling back to the flat file list.
async function filesFromDrop(dataTransfer: DataTransfer): Promise<File[]> {
  const items = Array.from(dataTransfer.items ?? []);
  const entries = items
    .map((item) => ("webkitGetAsEntry" in item ? (item as any).webkitGetAsEntry() : null))
    .filter(Boolean);
  if (entries.length > 0) {
    const nested = await Promise.all(entries.map((entry) => filesFromFileSystemEntry(entry)));
    return nested.flat();
  }
  return Array.from(dataTransfer.files ?? []);
}


// Build a FoG annotation segment from a start/end frame range, clamping to the
// run and carrying along the original model-predicted range when provided.
function annotationSegmentFromFrames(
  runDetail: RunDetail,
  start: number,
  end: number,
  source: FogAnnotationSegment["source"],
  createdAt?: string,
  predictedRange?: { start: number; end: number } | null,
): FogAnnotationSegment | null {
  if (runDetail.frames.length === 0) {
    return null;
  }
  const startFrameIndex = clamp(Math.min(start, end), 0, runDetail.frames.length - 1);
  const endFrameIndex = clamp(Math.max(start, end), 0, runDetail.frames.length - 1);
  const startFrame = runDetail.frames[startFrameIndex];
  const endFrame = runDetail.frames[endFrameIndex];
  let predictedStartFrameIndex: number | undefined;
  let predictedEndFrameIndex: number | undefined;
  if (predictedRange) {
    const lastFrame = Math.max(0, runDetail.frames.length - 1);
    predictedStartFrameIndex = clamp(Math.min(predictedRange.start, predictedRange.end), 0, lastFrame);
    predictedEndFrameIndex = clamp(Math.max(predictedRange.start, predictedRange.end), 0, lastFrame);
  }
  return {
    startFrameIndex,
    endFrameIndex,
    startVideoFrame: startFrame?.videoFrame ?? startFrameIndex,
    endVideoFrame: endFrame?.videoFrame ?? endFrameIndex,
    startSec: currentVideoTime(runDetail, startFrameIndex),
    endSec: currentVideoTime(runDetail, endFrameIndex),
    label: "fog",
    source,
    createdAt: createdAt ?? new Date().toISOString(),
    predictedStartFrameIndex,
    predictedEndFrameIndex,
  };
}

// Per-frame FoG probability series, preferring the requested score signal and
// falling back to "fog.score" then each frame's own fogScore field.
function fogScoreValues(runDetail: RunDetail, scoreSignalId = "fog.score"): Array<number | null> {
  const scoreSignal =
    runDetail.signals.find((signal) => signal.id === scoreSignalId) ??
    runDetail.signals.find((signal) => signal.id === "fog.score");
  return runDetail.frames.map((frame, index) => {
    const value = scoreSignal?.values[index] ?? frame.fogScore;
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  });
}

// Derive the model's predicted FoG segments: prefer the run's summary segments
// when no override threshold is set, else threshold the per-frame score series
// (falling back to the boolean fogDetected flags).
function predictedFogSegments(runDetail: RunDetail, threshold: number | null, scoreSignalId = "fog.score"): FogAnnotationSegment[] {
  if (typeof threshold !== "number" || !Number.isFinite(threshold)) {
    const fromSummary = runDetail.fog?.segments
      .map((segment) =>
        annotationSegmentFromFrames(
          runDetail,
          segment.startFrameIndex,
          segment.endFrameIndex,
          "auto_predicted",
          "model",
          { start: segment.startFrameIndex, end: segment.endFrameIndex },
        ),
      )
      .filter((segment): segment is FogAnnotationSegment => Boolean(segment)) ?? [];
    if (fromSummary.length > 0) {
      return fromSummary;
    }
  }

  const scores = fogScoreValues(runDetail, scoreSignalId);
  const decisionThreshold = typeof threshold === "number" && Number.isFinite(threshold)
    ? threshold
    : runDetail.fog?.threshold ?? DEFAULT_FOG_THRESHOLD;
  if (scores.some((value) => value !== null)) {
    return segmentsFromBooleanSeries(
      runDetail,
      scores.map((value) => value !== null && value >= decisionThreshold),
    );
  }

  const fromFrames = segmentsFromBooleanSeries(runDetail, runDetail.frames.map((frame) => Boolean(frame.fogDetected)));
  if (fromFrames.length > 0) {
    return fromFrames;
  }
  return [];
}

// Clean up raw predicted segments by merging segments closer than mergeGapSec
// and dropping any shorter than minDurationSec; results are tagged auto_corrected.
function postProcessFogSegments(
  runDetail: RunDetail,
  segments: FogAnnotationSegment[],
  minDurationSec: number,
  mergeGapSec: number,
): FogAnnotationSegment[] {
  if (segments.length === 0 || runDetail.frames.length === 0) {
    return [];
  }
  const fps = Math.max(1, runDetail.fps || 30);
  const minFrames = Math.max(1, Math.round(Math.max(0, minDurationSec) * fps));
  const mergeGapFrames = Math.max(0, Math.round(Math.max(0, mergeGapSec) * fps));
  const sorted = [...segments]
    .map((segment) => ({
      start: Math.max(0, Math.min(segment.startFrameIndex, segment.endFrameIndex)),
      end: Math.min(runDetail.frames.length - 1, Math.max(segment.startFrameIndex, segment.endFrameIndex)),
    }))
    .sort((a, b) => a.start - b.start);
  const merged: Array<{ start: number; end: number }> = [];
  for (const segment of sorted) {
    const previous = merged.at(-1);
    if (previous && segment.start - previous.end - 1 <= mergeGapFrames) {
      previous.end = Math.max(previous.end, segment.end);
    } else {
      merged.push({ ...segment });
    }
  }
  return merged
    .filter((segment) => segment.end - segment.start + 1 >= minFrames)
    .map((segment) =>
      annotationSegmentFromFrames(
        runDetail,
        segment.start,
        segment.end,
        "auto_corrected",
        "postprocess",
        { start: segment.start, end: segment.end },
      ),
    )
    .filter((segment): segment is FogAnnotationSegment => Boolean(segment));
}

// True for segments produced by the model/post-processing (never human-edited),
// so they can be regenerated rather than preserved when predictions change.
function isAutoGeneratedAnnotation(segment: FogAnnotationSegment): boolean {
  return segment.source === "auto_predicted" || (segment.source === "auto_corrected" && segment.createdAt === "postprocess");
}

// Shallow equality of two annotation lists by frame span and source (used to
// skip redundant state updates).
function annotationListsEqual(a: FogAnnotationSegment[], b: FogAnnotationSegment[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((left, index) => {
    const right = b[index];
    return (
      left.startFrameIndex === right.startFrameIndex &&
      left.endFrameIndex === right.endFrameIndex &&
      left.source === right.source
    );
  });
}

// Frame-level precision/recall/F1 of predicted vs gold FoG segments, comparing
// the two per-frame masks.
function frameBenchmarkMetrics(
  predicted: Array<{ startFrameIndex: number; endFrameIndex: number }>,
  gold: Array<{ startFrameIndex: number; endFrameIndex: number }>,
  frameCount: number,
): BenchmarkMetrics {
  const predictedMask = segmentFrameMask(predicted, frameCount);
  const goldMask = segmentFrameMask(gold, frameCount);
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  for (let index = 0; index < frameCount; index += 1) {
    if (predictedMask[index] && goldMask[index]) {
      truePositive += 1;
    } else if (predictedMask[index]) {
      falsePositive += 1;
    } else if (goldMask[index]) {
      falseNegative += 1;
    }
  }
  const precision = truePositive + falsePositive > 0 ? truePositive / (truePositive + falsePositive) : null;
  const recall = truePositive + falseNegative > 0 ? truePositive / (truePositive + falseNegative) : null;
  const f1 = truePositive + falsePositive + falseNegative > 0
    ? (2 * truePositive) / (2 * truePositive + falsePositive + falseNegative)
    : null;
  return { precision, recall, f1 };
}

// Rasterize segments into a per-frame boolean mask of "is FoG" frames.
function segmentFrameMask(
  segments: Array<{ startFrameIndex: number; endFrameIndex: number }>,
  frameCount: number,
): boolean[] {
  if (frameCount <= 0) {
    return [];
  }
  const mask = new Array(frameCount).fill(false);
  for (const segment of segments) {
    const start = clamp(Math.min(segment.startFrameIndex, segment.endFrameIndex), 0, frameCount - 1);
    const end = clamp(Math.max(segment.startFrameIndex, segment.endFrameIndex), 0, frameCount - 1);
    for (let index = start; index <= end; index += 1) {
      mask[index] = true;
    }
  }
  return mask;
}

// Render a 0..1 metric as a whole-percent string, or "n/a" when undefined.
function fmtMetric(value: number | null): string {
  return value === null ? "n/a" : `${Math.round(value * 100)}%`;
}

// Human-readable duration: 125s -> "2min5s", 45s -> "45s", 3661s -> "1h1min".
function formatHumanDuration(totalSec: number): string {
  if (!Number.isFinite(totalSec) || totalSec <= 0) return "";
  const s = Math.round(totalSec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h${m > 0 ? `${m}min` : ""}`;
  if (m > 0) return `${m}min${sec > 0 ? `${sec}s` : ""}`;
  return `${sec}s`;
}

// Run duration from the frame count and fps (empty when fps is unknown).
function runDurationLabel(processedFrames: number, fps: number | null | undefined): string {
  if (!fps || fps <= 0) return "";
  return formatHumanDuration(processedFrames / fps);
}

// "<processed>/<total> frames (<pct>%)" for a job (total/percent omitted when unknown).
function jobFrameLabel(job: GenerationJob): string {
  const base = `${job.processedFrames}${job.totalFrames ? `/${job.totalFrames}` : ""} frames`;
  return job.progressPercent !== null ? `${base} · ${Math.round(job.progressPercent)}%` : base;
}

// Progress bar fill (0..100) for a job, inferring 100% for active jobs that have
// started but report no explicit percent.
function jobProgressPercent(job: GenerationJob): number {
  if (typeof job.progressPercent === "number" && Number.isFinite(job.progressPercent)) {
    return clamp(job.progressPercent, 0, 100);
  }
  return isActiveJob(job) && job.processedFrames > 0 ? 100 : 0;
}

// Asset-preload progress (0..100) for a run currently loading.
function loadProgressPercent(state: RunLoadState): number {
  return state.total > 0 ? clamp((100 * state.loaded) / state.total, 0, 100) : 0;
}

// Asset-preload label, appending "loaded/total" when there is more than one item.
function loadProgressLabel(state: RunLoadState): string {
  if (state.total <= 1) {
    return state.label;
  }
  return `${state.label} ${state.loaded}/${state.total}`;
}

// Full-pane placeholder shown while a run's assets are preloading.
function RunLoadingState({ state }: { state: RunLoadState }) {
  return (
    <div className="processing-state" role="status" aria-live="polite">
      <div className="processing-state-card">
        <div className="processing-state-head">
          <span>Loading</span>
          <strong>{state.runId}</strong>
        </div>
        <div className="job-progress">
          <span style={{ width: `${loadProgressPercent(state)}%` }} />
        </div>
        <div className="processing-state-meta">
          <span>{loadProgressLabel(state)}</span>
          <span>Preparing case</span>
        </div>
      </div>
    </div>
  );
}

// Full-pane placeholder showing a generation job's live status and progress.
function ProcessingState({ job }: { job: GenerationJob }) {
  const isActive = isActiveJob(job);
  return (
    <div className="processing-state">
      <div className="processing-state-card">
        <div className="processing-state-head">
          <span>{statusLabel(job.status)}</span>
          <strong>{job.runId}</strong>
        </div>
        <div className={`job-progress ${isActive && job.progressPercent === null ? "indeterminate" : ""}`}>
          <span style={{ width: `${jobProgressPercent(job)}%` }} />
        </div>
        <div className="processing-state-meta">
          <span>{jobFrameLabel(job)}</span>
          <span>{isActive ? "Building kinematics" : job.error ?? "Waiting for run data"}</span>
        </div>
      </div>
    </div>
  );
}

// Convert a per-frame boolean FoG series into contiguous auto_predicted segments.
function segmentsFromBooleanSeries(runDetail: RunDetail, values: boolean[]): FogAnnotationSegment[] {
  const segments: FogAnnotationSegment[] = [];
  let start: number | null = null;
  for (let index = 0; index <= values.length; index += 1) {
    const active = Boolean(values[index]);
    if (active && start === null) {
      start = index;
    } else if (!active && start !== null) {
      const end = index - 1;
      const segment = annotationSegmentFromFrames(
        runDetail,
        start,
        end,
        "auto_predicted",
        "model",
        { start, end },
      );
      if (segment) {
        segments.push(segment);
      }
      start = null;
    }
  }
  return segments;
}

type AnnotationBandKind = "predicted" | "manual";
type AnnotationBand = { start: number; end: number; kind: AnnotationBandKind };

// Build an SVG path for a rectangle with independently rounded left and
// right ends (top + bottom rounded together on each side). When two bands
// sit next to each other we use rl=0 on the inner edges so they meet flush.
function bandRectPath(x: number, y: number, w: number, h: number, rl: number, rr: number): string {
  const left = x;
  const right = x + w;
  const top = y;
  const bottom = y + h;
  const radiusLeft = Math.max(0, Math.min(rl, h / 2, w / 2));
  const radiusRight = Math.max(0, Math.min(rr, h / 2, w / 2));
  const parts: string[] = [];
  parts.push(`M ${left + radiusLeft} ${top}`);
  parts.push(`L ${right - radiusRight} ${top}`);
  if (radiusRight > 0) parts.push(`Q ${right} ${top} ${right} ${top + radiusRight}`);
  parts.push(`L ${right} ${bottom - radiusRight}`);
  if (radiusRight > 0) parts.push(`Q ${right} ${bottom} ${right - radiusRight} ${bottom}`);
  parts.push(`L ${left + radiusLeft} ${bottom}`);
  if (radiusLeft > 0) parts.push(`Q ${left} ${bottom} ${left} ${bottom - radiusLeft}`);
  parts.push(`L ${left} ${top + radiusLeft}`);
  if (radiusLeft > 0) parts.push(`Q ${left} ${top} ${left + radiusLeft} ${top}`);
  parts.push("Z");
  return parts.join(" ");
}

// Split a segment into colored bands so the UI can show how much of a
// segment is "what the model predicted" (blue) vs "what a human added or
// extended" (green). The order of returned bands is always left → right
// along the timeline.
function segmentBands(segment: FogAnnotationSegment): AnnotationBand[] {
  const curStart = segment.startFrameIndex;
  const curEnd = segment.endFrameIndex;
  if (
    segment.predictedStartFrameIndex === undefined ||
    segment.predictedEndFrameIndex === undefined
  ) {
    return [{ start: curStart, end: curEnd, kind: "manual" }];
  }
  const predStart = segment.predictedStartFrameIndex;
  const predEnd = segment.predictedEndFrameIndex;
  const overlapStart = Math.max(curStart, predStart);
  const overlapEnd = Math.min(curEnd, predEnd);
  if (overlapEnd < overlapStart) {
    // The segment has been moved entirely outside its original predicted
    // range — treat it as a fully manual band.
    return [{ start: curStart, end: curEnd, kind: "manual" }];
  }
  const bands: AnnotationBand[] = [];
  if (curStart < overlapStart) {
    bands.push({ start: curStart, end: overlapStart - 1, kind: "manual" });
  }
  bands.push({ start: overlapStart, end: overlapEnd, kind: "predicted" });
  if (curEnd > overlapEnd) {
    bands.push({ start: overlapEnd + 1, end: curEnd, kind: "manual" });
  }
  return bands;
}

// Type guard: segment originated from the model (predicted or post-corrected).
function isAutomaticAnnotation(
  segment: FogAnnotationSegment,
): segment is FogAnnotationSegment & { source: Extract<FogAnnotationSource, "auto_predicted" | "auto_corrected"> } {
  return segment.source === "auto_predicted" || segment.source === "auto_corrected";
}

// Two segments cover the exact same frame range.
function sameFrameSpan(a: FogAnnotationSegment, b: FogAnnotationSegment): boolean {
  return a.startFrameIndex === b.startFrameIndex && a.endFrameIndex === b.endFrameIndex;
}

// Provenance of a merged segment: any human touch wins, then auto_corrected,
// else auto_predicted.
function mergedAnnotationSource(a: FogAnnotationSource, b: FogAnnotationSource): FogAnnotationSource {
  if (a === "manual" || b === "manual") {
    return "manual";
  }
  if (a === "auto_corrected" || b === "auto_corrected") {
    return "auto_corrected";
  }
  return "auto_predicted";
}

// React hook: track an element's pixel size via ResizeObserver, returning a ref
// callback plus the current width/height (used to size the SVG timelines).
function useResponsiveSize(
  initialWidth = 1120,
  initialHeight = 192,
): { ref: (node: HTMLDivElement | null) => void; width: number; height: number } {
  const [size, setSize] = useState({ width: initialWidth, height: initialHeight });
  const observerRef = useRef<ResizeObserver | null>(null);

  const setRef = useCallback((node: HTMLDivElement | null) => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }
    if (!node || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setSize((prev) =>
            Math.abs(prev.width - width) < 0.5 && Math.abs(prev.height - height) < 0.5
              ? prev
              : { width: Math.round(width), height: Math.round(height) },
          );
        }
      }
    });
    observer.observe(node);
    observerRef.current = observer;
    const rect = node.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      setSize({ width: Math.round(rect.width), height: Math.round(rect.height) });
    }
  }, []);

  useEffect(() => () => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }
  }, []);

  return { ref: setRef, width: size.width, height: size.height };
}

// Coerce a possibly missing FoG score to a finite value clamped to [0, 1].
function evidenceScore(score: number | null | undefined): number {
  if (typeof score !== "number" || !Number.isFinite(score)) {
    return 0;
  }
  return clamp(score, 0, 1);
}

// The thin FoG-probability strip under the 3D view: draws the per-frame score
// curve with the decision threshold, lets the user scrub the playhead, and hosts
// the detection-settings popover (threshold / cleanup gaps / score source).
function FogProbabilityPanel({
  runDetail,
  frameIndex,
  threshold,
  cleanupMinDurationSec,
  cleanupMergeGapSec,
  settingsOpen,
  onCleanupMinDurationChange,
  onCleanupMergeGapChange,
  onSettingsOpenChange,
  onThresholdChange,
  onFrameSelect,
  scoreSignalId,
  onScoreSignalChange,
}: {
  runDetail: RunDetail;
  frameIndex: number;
  threshold: number | null;
  cleanupMinDurationSec: number;
  cleanupMergeGapSec: number;
  settingsOpen: boolean;
  onCleanupMinDurationChange: (value: number) => void;
  onCleanupMergeGapChange: (value: number) => void;
  onSettingsOpenChange: (open: boolean) => void;
  onThresholdChange: (threshold: number) => void;
  onFrameSelect: (index: number) => void;
  scoreSignalId: "fog.score" | "fog.score_smooth";
  onScoreSignalChange: (id: "fog.score" | "fog.score_smooth") => void;
}) {
  const { ref: containerRef, width, height } = useResponsiveSize(1120, 46);
  // Close the settings popover when clicking anywhere outside it (or pressing
  // Escape). The gear toggle is excluded so its own onClick keeps working.
  const popoverRef = useRef<HTMLDivElement>(null);
  const gearRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!settingsOpen) return;
    function handlePointerDown(event: globalThis.PointerEvent) {
      const target = event.target as Node;
      if (popoverRef.current?.contains(target)) return;
      if (gearRef.current?.contains(target)) return;
      onSettingsOpenChange(false);
    }
    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") onSettingsOpenChange(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [settingsOpen, onSettingsOpenChange]);
  // The FoG-probability card (.fog-probability) is itself inset ~13px from the
  // panel content box, so internal margins of 44 / 6 place its data area at
  // [panel + 56, panel - 18] — pixel-matching the annotation / ground-truth
  // tracks (CSS margin 56/18) and the uPlot kinematics plot (left axis 56,
  // right pad 18). All four share one time axis.
  const margin = { top: 6, right: 6, bottom: 4, left: 44 };
  const innerW = Math.max(20, width - margin.left - margin.right);
  const innerH = Math.max(20, height - margin.top - margin.bottom);
  const frameCount = runDetail.frames.length;
  const maskedRanges = useMemo<Array<[number, number]>>(() => {
    const out: Array<[number, number]> = [];
    let start = -1;
    for (let i = 0; i < runDetail.frames.length; i += 1) {
      const isMasked = runDetail.frames[i]?.inferenceStatus === "masked";
      if (isMasked && start < 0) {
        start = i;
      } else if (!isMasked && start >= 0) {
        out.push([start, i - 1]);
        start = -1;
      }
    }
    if (start >= 0) out.push([start, runDetail.frames.length - 1]);
    return out;
  }, [runDetail.frames]);
  const bins = Math.min(720, Math.max(1, Math.min(frameCount, Math.round(innerW * 1.5))));
  const signalMap = useMemo(() => new Map(runDetail.signals.map((signal) => [signal.id, signal])), [runDetail.signals]);
  const hasSmoothScore = signalMap.has("fog.score_smooth");
  const scoreSignal = signalMap.get(scoreSignalId) ?? signalMap.get("fog.score");
  const samples = useMemo(() => {
    return Array.from({ length: bins }, (_, bin) => {
      const start = Math.floor((bin * frameCount) / bins);
      const end = Math.max(start + 1, Math.floor(((bin + 1) * frameCount) / bins));
      let maxScore = 0;
      for (let index = start; index < Math.min(end, frameCount); index += 1) {
        maxScore = Math.max(
          maxScore,
          evidenceScore(scoreSignal?.values[index] ?? runDetail.frames[index]?.fogScore),
        );
      }
      return maxScore;
    });
  }, [bins, frameCount, runDetail.frames, scoreSignal?.values]);
  const xForSample = useCallback(
    (idx: number) => margin.left + (innerW * idx) / Math.max(1, samples.length - 1),
    [innerW, margin.left, samples.length],
  );
  const yForScore = useCallback(
    (score: number) => margin.top + innerH - clamp(score, 0, 1) * innerH,
    [innerH, margin.top],
  );
  const curvePath = useMemo(() => {
    if (samples.length === 0) {
      return "";
    }
    return samples
      .map((score, index) => `${index === 0 ? "M" : "L"} ${xForSample(index).toFixed(2)} ${yForScore(score).toFixed(2)}`)
      .join(" ");
  }, [samples, xForSample, yForScore]);
  const areaPath = curvePath
    ? `${curvePath} L ${(margin.left + innerW).toFixed(2)} ${(margin.top + innerH).toFixed(2)} L ${margin.left.toFixed(2)} ${(margin.top + innerH).toFixed(2)} Z`
    : "";
  const cursorX = frameCount > 1
    ? margin.left + (clamp(frameIndex, 0, frameCount - 1) / (frameCount - 1)) * innerW
    : margin.left;
  const currentScore = evidenceScore(scoreSignal?.values[frameIndex] ?? runDetail.frames[frameIndex]?.fogScore);
  const activeThreshold = threshold ?? runDetail.fog?.threshold ?? DEFAULT_FOG_THRESHOLD;
  const thresholdPercent = Math.round(clamp(activeThreshold, 0.01, 0.99) * 100);
  const thresholdY = yForScore(clamp(activeThreshold, 0, 1));
  const yPercentTicks = [0, 50, 100];
  // Translate a pointer x within the SVG into a frame index and select it.
  function selectFromClientX(clientX: number, currentTarget: SVGSVGElement): void {
    const rect = currentTarget.getBoundingClientRect();
    if (rect.width <= 0) return;
    const scale = width / rect.width;
    const localX = (clientX - rect.left) * scale - margin.left;
    const ratio = clamp(localX / Math.max(1, innerW), 0, 1);
    onFrameSelect(Math.round(ratio * Math.max(0, frameCount - 1)));
  }

  return (
    <div className="fog-probability">
      <div className="fog-probability-header">
        <div>
          <span>FoG probability</span>
        </div>
        <div className="fog-readout">
          <strong>{Math.round(currentScore * 100)}%</strong>
          <button
            ref={gearRef}
            type="button"
            className={`fog-gear ${settingsOpen ? "active" : ""}`}
            onClick={() => onSettingsOpenChange(!settingsOpen)}
            title="Detection settings"
            aria-label="Detection settings"
            aria-expanded={settingsOpen}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <path
                fill="currentColor"
                d="M12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Zm8.94-2.5a7.9 7.9 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.9 7.9 0 0 0-1.7-1l-.4-2.5H10l-.4 2.5a7.9 7.9 0 0 0-1.7 1l-2.4-1-2 3.4 2 1.6a7.9 7.9 0 0 0 0 2l-2 1.6 2 3.4 2.4-1c.5.4 1.1.8 1.7 1l.4 2.5h4l.4-2.5c.6-.2 1.2-.6 1.7-1l2.4 1 2-3.4-2-1.6Z"
              />
            </svg>
          </button>
        </div>
      </div>
      <div ref={containerRef} className="fog-probability-bar-wrap">
        <svg
          className="fog-probability-bar"
          viewBox={`0 0 ${width} ${height}`}
          role="slider"
          aria-label="FoG probability timeline"
          aria-valuemin={0}
          aria-valuemax={Math.max(0, frameCount - 1)}
          aria-valuenow={frameIndex}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            selectFromClientX(event.clientX, event.currentTarget);
          }}
          onPointerMove={(event) => {
            if (event.buttons === 1) {
              selectFromClientX(event.clientX, event.currentTarget);
            }
          }}
        >
          <rect x={0} y={0} width={width} height={height} className="fog-density-bg" />
          {maskedRanges.map(([s, e], i) => {
            const x1 = margin.left + (clamp(s, 0, Math.max(0, frameCount - 1)) / Math.max(1, frameCount - 1)) * innerW;
            const x2 = margin.left + (clamp(e, 0, Math.max(0, frameCount - 1)) / Math.max(1, frameCount - 1)) * innerW;
            const w = Math.max(1.5, x2 - x1);
            return (
              <g key={`masked-${i}`} pointerEvents="none">
                <rect x={x1} y={margin.top} width={w} height={innerH} fill="rgba(245,158,11,0.1)" stroke="rgba(245,158,11,0.4)" strokeDasharray="3 3" />
                {w > 40 ? (
                  <text x={x1 + w / 2} y={margin.top + 10} fill="rgba(245,158,11,0.95)" fontSize="9" textAnchor="middle">masked</text>
                ) : null}
              </g>
            );
          })}
          {yPercentTicks.map((tick) => {
            const y = yForScore(tick / 100);
            return (
              <line
                key={`fp-grid-${tick}`}
                x1={margin.left}
                x2={margin.left + innerW}
                y1={y}
                y2={y}
                className="fog-score-grid"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          <line
            x1={margin.left}
            x2={margin.left + innerW}
            y1={thresholdY}
            y2={thresholdY}
            className="fog-threshold-line"
            vectorEffect="non-scaling-stroke"
          />
          {areaPath ? <path d={areaPath} className="fog-score-area" /> : null}
          {curvePath ? (
            <path d={curvePath} className="fog-score-curve" vectorEffect="non-scaling-stroke" />
          ) : null}

          <line
            x1={margin.left}
            x2={margin.left}
            y1={margin.top}
            y2={margin.top + innerH}
            className="plot-axis"
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={margin.left}
            x2={margin.left + innerW}
            y1={margin.top + innerH}
            y2={margin.top + innerH}
            className="plot-axis"
            vectorEffect="non-scaling-stroke"
          />

          {yPercentTicks.map((tick) => {
            const y = yForScore(tick / 100);
            return (
              <g key={`fp-ytick-${tick}`}>
                <line
                  x1={margin.left - 3}
                  x2={margin.left}
                  y1={y}
                  y2={y}
                  className="plot-tick"
                  vectorEffect="non-scaling-stroke"
                />
                <text x={margin.left - 6} y={y + 3} textAnchor="end" className="plot-tick-label">
                  {tick}%
                </text>
              </g>
            );
          })}

          <text x={margin.left + innerW - 4} y={Math.max(10, thresholdY - 4)} textAnchor="end" className="fog-threshold-label">
            threshold {thresholdPercent}%
          </text>

          <line
            x1={cursorX}
            x2={cursorX}
            y1={margin.top}
            y2={margin.top + innerH}
            className="fog-density-cursor"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
      {settingsOpen ? (
        <div ref={popoverRef} className="fog-settings-popover" role="dialog" aria-label="FoG detection">
          <div className="fog-settings-head">
            <span>Detection settings</span>
            <button type="button" className="fog-settings-close" onClick={() => onSettingsOpenChange(false)} aria-label="Close">
              ✕
            </button>
          </div>
          <div className="fog-settings-row">
            <span>Source</span>
            <div className="fog-score-toggle" role="group" aria-label="Probability source">
              <button
                type="button"
                className={scoreSignalId === "fog.score" ? "active" : ""}
                onClick={() => onScoreSignalChange("fog.score")}
                title="Raw per-frame probability"
              >
                Raw
              </button>
              <button
                type="button"
                className={scoreSignalId === "fog.score_smooth" ? "active" : ""}
                onClick={() => onScoreSignalChange("fog.score_smooth")}
                disabled={!hasSmoothScore}
                title={hasSmoothScore ? "Temporally smoothed probability" : "No smoothed probability for this run"}
              >
                Smoothed
              </button>
            </div>
          </div>
          <label>
            <span>Threshold</span>
            <input
              type="range"
              min={35}
              max={95}
              step={1}
              value={thresholdPercent}
              onChange={(event) => onThresholdChange(Number(event.target.value) / 100)}
            />
            <strong>{thresholdPercent}%</strong>
          </label>
          <label>
            <span>Ignore &lt;</span>
            <input
              type="range"
              min={0}
              max={30}
              step={1}
              value={Math.round(cleanupMinDurationSec * 10)}
              onChange={(event) => onCleanupMinDurationChange(Number(event.target.value) / 10)}
            />
            <strong>{cleanupMinDurationSec.toFixed(1)}s</strong>
          </label>
          <label>
            <span>Merge gaps &lt;</span>
            <input
              type="range"
              min={0}
              max={50}
              step={1}
              value={Math.round(cleanupMergeGapSec * 10)}
              onChange={(event) => onCleanupMergeGapChange(Number(event.target.value) / 10)}
            />
            <strong>{cleanupMergeGapSec.toFixed(1)}s</strong>
          </label>
        </div>
      ) : null}
    </div>
  );
}

// Sort, order endpoints, and merge overlapping/adjacent annotation segments,
// unioning their predicted ranges and provenance so band rendering stays correct.
function normalizeAnnotationSegments(segments: FogAnnotationSegment[]): FogAnnotationSegment[] {
  const sorted = [...segments].sort((a, b) => a.startFrameIndex - b.startFrameIndex);
  const merged: FogAnnotationSegment[] = [];
  for (const segment of sorted) {
    const start = Math.min(segment.startFrameIndex, segment.endFrameIndex);
    const end = Math.max(segment.startFrameIndex, segment.endFrameIndex);
    const normalized = { ...segment, startFrameIndex: start, endFrameIndex: end };
    const previous = merged.at(-1);
    if (previous && normalized.startFrameIndex <= previous.endFrameIndex + 1) {
      previous.endFrameIndex = Math.max(previous.endFrameIndex, normalized.endFrameIndex);
      previous.endVideoFrame = Math.max(previous.endVideoFrame, normalized.endVideoFrame);
      previous.endSec = Math.max(previous.endSec, normalized.endSec);
      previous.source = mergedAnnotationSource(previous.source, normalized.source);
      // Union the predicted ranges so band rendering covers the model's
      // contribution from either side of the merge.
      const prevPredStart = previous.predictedStartFrameIndex;
      const prevPredEnd = previous.predictedEndFrameIndex;
      const nextPredStart = normalized.predictedStartFrameIndex;
      const nextPredEnd = normalized.predictedEndFrameIndex;
      if (prevPredStart !== undefined && prevPredEnd !== undefined) {
        if (nextPredStart !== undefined && nextPredEnd !== undefined) {
          previous.predictedStartFrameIndex = Math.min(prevPredStart, nextPredStart);
          previous.predictedEndFrameIndex = Math.max(prevPredEnd, nextPredEnd);
        }
        // else: keep previous predicted range (next is fully manual contributor)
      } else if (nextPredStart !== undefined && nextPredEnd !== undefined) {
        previous.predictedStartFrameIndex = nextPredStart;
        previous.predictedEndFrameIndex = nextPredEnd;
      }
    } else {
      merged.push(normalized);
    }
  }
  return merged;
}

// Flatten the run's dataset-provided labeled FoG episodes into gold-truth
// segments (preferring processed datasets, de-duped by label+frame span).
function goldTruthSegments(runDetail: RunDetail): GoldTruthSegment[] {
  const frameCount = runDetail.frames.length;
  const fps = Math.max(1, runDetail.fps || 30);
  const segments: GoldTruthSegment[] = [];
  const datasets = [...(runDetail.datasets ?? [])].sort((a, b) => {
    const aProcessed = a.datasetId.includes("processed") ? 0 : 1;
    const bProcessed = b.datasetId.includes("processed") ? 0 : 1;
    return aProcessed - bProcessed || a.datasetId.localeCompare(b.datasetId);
  });
  const seen = new Set<string>();
  for (const dataset of datasets) {
    for (const episode of dataset.labeledEpisodes ?? []) {
      const label = episode.label || "fog";
      const startFrame = episode.startFrameIndex ?? Math.round((episode.startMs / 1000) * fps);
      const endFrame = episode.endFrameIndex ?? Math.round((episode.endMs / 1000) * fps);
      const start = clamp(Math.min(startFrame, endFrame), 0, Math.max(0, frameCount - 1));
      const end = clamp(Math.max(startFrame, endFrame), 0, Math.max(0, frameCount - 1));
      if (end < start || frameCount === 0) {
        continue;
      }
      const key = `${label}:${start}:${end}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      segments.push({
        startFrameIndex: start,
        endFrameIndex: end,
        startSec: episode.startMs / 1000,
        endSec: episode.endMs / 1000,
        label,
        datasetId: dataset.datasetId,
        split: dataset.split,
        labelEventsPath: dataset.labelEventsPath,
      });
    }
  }
  return segments.sort((a, b) => a.startFrameIndex - b.startFrameIndex);
}

// Horizontal navigator (overview scrollbar) for the labelling tracks. Shows the
// whole timeline with a draggable/resizable window; dragging the window body
// pans, dragging an edge zooms. Drives the SHARED view window so the annotation,
// ground-truth and kinematics tracks all scroll in lockstep — essential on long
// videos where the full clip is too compressed to label precisely.
function TimelineNavigator({
  frameCount,
  frameIndex,
  windowStart,
  windowSpan,
  onWindowChange,
}: {
  frameCount: number;
  frameIndex: number;
  windowStart: number;
  windowSpan: number;
  onWindowChange: (start: number, span: number) => void;
}) {
  const { ref, width, height } = useResponsiveSize(1120, 30);
  const ML = 56;
  const MR = 18;
  const innerW = Math.max(20, width - ML - MR);
  const total = Math.max(1, frameCount);
  const fToX = (f: number) => ML + (clamp(f, 0, total) / total) * innerW;
  const xToF = (x: number) => clamp(Math.round(((x - ML) / innerW) * total), 0, total);
  const thumbX = fToX(windowStart);
  const thumbW = Math.max(8, fToX(windowStart + windowSpan) - thumbX);
  const cursorX = fToX(frameIndex);
  const dragRef = useRef<{ mode: "pan" | "left" | "right"; startX: number; start0: number; span0: number } | null>(null);

  const svgX = (clientX: number, target: SVGSVGElement): number => {
    const rect = target.getBoundingClientRect();
    return rect.width > 0 ? (clientX - rect.left) * (width / rect.width) : 0;
  };

  function onPointerDown(event: PointerEvent<SVGSVGElement>): void {
    const px = svgX(event.clientX, event.currentTarget);
    const left = thumbX;
    const right = thumbX + thumbW;
    let mode: "pan" | "left" | "right" | null = null;
    if (Math.abs(px - left) <= 8) mode = "left";
    else if (Math.abs(px - right) <= 8) mode = "right";
    else if (px >= left && px <= right) mode = "pan";
    if (mode === null) {
      // Click on the empty track: recentre the window there.
      const f = xToF(px);
      onWindowChange(clamp(f - Math.round(windowSpan / 2), 0, Math.max(0, total - windowSpan)), windowSpan);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { mode, startX: px, start0: windowStart, span0: windowSpan };
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>): void {
    const drag = dragRef.current;
    if (!drag) return;
    const px = svgX(event.clientX, event.currentTarget);
    const df = Math.round(((px - drag.startX) / innerW) * total);
    if (drag.mode === "pan") {
      onWindowChange(clamp(drag.start0 + df, 0, Math.max(0, total - drag.span0)), drag.span0);
    } else if (drag.mode === "left") {
      const newStart = clamp(drag.start0 + df, 0, drag.start0 + drag.span0 - 2);
      onWindowChange(newStart, Math.max(2, drag.start0 + drag.span0 - newStart));
    } else {
      onWindowChange(drag.start0, clamp(drag.span0 + df, 2, total - drag.start0));
    }
  }

  function onPointerUp(event: PointerEvent<SVGSVGElement>): void {
    dragRef.current = null;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // ignore
    }
  }

  return (
    <div ref={ref} className="timeline-navigator" title="Drag the window to scrub · drag an edge to zoom · double-click to fit all">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => onWindowChange(0, total)}
      >
        <rect x={ML} y={9} width={innerW} height={height - 18} className="tnav-track" rx={3} />
        <rect x={thumbX} y={4} width={thumbW} height={height - 8} className="tnav-window" rx={4} />
        <rect x={thumbX} y={4} width={4} height={height - 8} className="tnav-edge" rx={2} />
        <rect x={thumbX + thumbW - 4} y={4} width={4} height={height - 8} className="tnav-edge" rx={2} />
        <line x1={cursorX} x2={cursorX} y1={2} y2={height - 2} className="tnav-cursor" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

// The editable FoG annotation track: renders annotation segments (with model vs
// human bands), tombstones for deleted predictions, and the draft box, and
// handles pointer create/move/resize plus the zoom-window handle. All editing is
// delegated to the callbacks supplied by ViewerShell.
function ManualFogAnnotationBar({
  runDetail,
  frameIndex,
  segments,
  deletedSegments,
  draft,
  viewWindow,
  onDraftChange,
  onCommit,
  selectedSegmentIndex,
  onSelectSegment,
  onUpdateSegment,
  onDeleteSelected,
  onSplitSelected,
  onFrameSelect,
  onClear,
  onExportBundle,
  hasGroundTruth,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  onResetToPrediction,
  predictionCount,
}: {
  runDetail: RunDetail;
  frameIndex: number;
  segments: FogAnnotationSegment[];
  deletedSegments: DeletedFogPrediction[];
  draft: DraftAnnotation;
  viewWindow: { start: number; end: number } | null;
  onDraftChange: (draft: DraftAnnotation) => void;
  onCommit: (start: number, end: number) => void;
  selectedSegmentIndex: number | null;
  onSelectSegment: (index: number | null) => void;
  onUpdateSegment: (index: number, start: number, end: number) => void;
  onDeleteSelected: () => void;
  onSplitSelected: () => void;
  onFrameSelect: (index: number) => void;
  onClear: () => void;
  onExportBundle: (formats: ChannelFormats, includeTrackingVideo: boolean) => void | Promise<void>;
  hasGroundTruth: boolean;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onResetToPrediction: () => void;
  predictionCount: number;
}) {
  const width = 1120;
  const height = 46;
  const frameCount = runDetail.frames.length;
  const dragStateRef = useRef<AnnotationDragState>(null);
  // Single Export control: one button opens a panel where each channel's file
  // format is chosen independently, then everything downloads as one folder
  // (.zip). Closes on outside-click / Escape.
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [bundleFormats, setBundleFormats] = useState<ChannelFormats>(DEFAULT_CHANNEL_FORMATS);
  const [includeTrackingVideo, setIncludeTrackingVideo] = useState(true);
  const [bundleBusy, setBundleBusy] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const exportBtnRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!exportMenuOpen) return;
    function handlePointerDown(event: globalThis.PointerEvent) {
      const target = event.target as Node;
      if (exportMenuRef.current?.contains(target)) return;
      if (exportBtnRef.current?.contains(target)) return;
      setExportMenuOpen(false);
    }
    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setExportMenuOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [exportMenuOpen]);
  const viewStart = viewWindow?.start ?? 0;
  const viewEnd = viewWindow?.end ?? Math.max(0, frameCount - 1);
  const viewSpan = Math.max(1, viewEnd - viewStart);
  const cursorX = ((clamp(frameIndex, viewStart, viewEnd) - viewStart) / viewSpan) * width;
  const visibleSegments = segments.filter(
    (segment) => segment.endFrameIndex >= viewStart && segment.startFrameIndex <= viewEnd,
  );
  const visibleDeleted = deletedSegments.filter(
    (segment) => segment.endFrameIndex >= viewStart && segment.startFrameIndex <= viewEnd,
  );

  // Frame index under a pointer x, mapped through the current zoom window.
  function frameFromClientX(clientX: number, currentTarget: SVGSVGElement): number {
    const rect = currentTarget.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return Math.round(viewStart + ratio * viewSpan);
  }

  // Pixel x/width of a frame range within the current zoom window (min 5px wide).
  function segmentRect(startFrame: number, endFrame: number): { x: number; width: number } {
    const start = clamp(Math.min(startFrame, endFrame), viewStart, viewEnd);
    const end = clamp(Math.max(startFrame, endFrame), viewStart, viewEnd);
    const x = ((start - viewStart) / viewSpan) * width;
    const x2 = ((end - viewStart) / viewSpan) * width;
    return { x, width: Math.max(5, x2 - x) };
  }

  return (
    <div className="manual-fog">
      <div className="fog-density-meta">
        <div className="annotation-track-title">
          <span>Annotation</span>
        </div>
        <div className="manual-fog-actions">
          <button className="icon-btn" type="button" onClick={onUndo} disabled={!canUndo} title="Undo (Cmd/Ctrl+Z)" aria-label="Undo">
            <ActionIcon name="undo" />
          </button>
          <button className="icon-btn" type="button" onClick={onRedo} disabled={!canRedo} title="Redo (Cmd/Ctrl+Shift+Z)" aria-label="Redo">
            <ActionIcon name="redo" />
          </button>
          <button
            className="icon-btn"
            type="button"
            onClick={onResetToPrediction}
            disabled={predictionCount === 0}
            title="Restore model predictions"
            aria-label="Restore model predictions"
          >
            <ActionIcon name="restore" />
          </button>
          <button className="icon-btn" type="button" onClick={onSplitSelected} disabled={selectedSegmentIndex === null} title="Split segment" aria-label="Split">
            <ActionIcon name="split" />
          </button>
          <button className="icon-btn" type="button" onClick={onDeleteSelected} disabled={selectedSegmentIndex === null} title="Delete segment" aria-label="Delete">
            <ActionIcon name="delete" />
          </button>
          <button className="icon-btn" type="button" onClick={onClear} disabled={segments.length === 0} title="Clear all" aria-label="Clear all">
            <ActionIcon name="clear" />
          </button>
          <span className="manual-fog-sep" aria-hidden="true" />
          <div className="export-menu-wrap">
            <button
              ref={exportBtnRef}
              className={`export-btn ${exportMenuOpen ? "active" : ""}`}
              type="button"
              onClick={() => setExportMenuOpen((open) => !open)}
              aria-haspopup="dialog"
              aria-expanded={exportMenuOpen}
              title="Export all channels as a folder (.zip)"
            >
              <ActionIcon name="download" />
              Export
            </button>
            {exportMenuOpen ? (
              <div ref={exportMenuRef} className="export-panel" role="dialog" aria-label="Export options">
                <div className="export-panel-head">
                  <span className="export-panel-title">Export folder</span>
                  <span className="export-panel-sub">One .zip, one file per channel</span>
                </div>
                <label className="export-row">
                  <span className="export-row-label">Kinematics</span>
                  <select
                    className="export-row-select"
                    value={bundleFormats.kinematics}
                    onChange={(e) => setBundleFormats((f) => ({ ...f, kinematics: e.target.value as ChannelFormats["kinematics"] }))}
                  >
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                  </select>
                </label>
                <label className="export-row">
                  <span className="export-row-label">FoG probability</span>
                  <select
                    className="export-row-select"
                    value={bundleFormats.fogProbability}
                    onChange={(e) => setBundleFormats((f) => ({ ...f, fogProbability: e.target.value as ChannelFormats["fogProbability"] }))}
                  >
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                  </select>
                </label>
                <label className="export-row">
                  <span className="export-row-label">Annotation track</span>
                  <select
                    className="export-row-select"
                    value={bundleFormats.annotation}
                    onChange={(e) => setBundleFormats((f) => ({ ...f, annotation: e.target.value as ChannelFormats["annotation"] }))}
                  >
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                    <option value="eaf">ELAN (.eaf)</option>
                  </select>
                </label>
                <label className={`export-row ${hasGroundTruth ? "" : "is-disabled"}`}>
                  <span className="export-row-label">
                    Ground truth
                    {hasGroundTruth ? null : <span className="export-row-note">none</span>}
                  </span>
                  <select
                    className="export-row-select"
                    value={bundleFormats.groundTruth}
                    disabled={!hasGroundTruth}
                    onChange={(e) => setBundleFormats((f) => ({ ...f, groundTruth: e.target.value as ChannelFormats["groundTruth"] }))}
                  >
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                    <option value="eaf">ELAN (.eaf)</option>
                  </select>
                </label>
                <label className="export-row">
                  <span className="export-row-label">
                    Tracking-box video
                    <span className="export-row-note">.mp4</span>
                  </span>
                  <input
                    type="checkbox"
                    className="export-row-check"
                    checked={includeTrackingVideo}
                    onChange={(e) => setIncludeTrackingVideo(e.target.checked)}
                  />
                </label>
                <button
                  type="button"
                  className="export-panel-go"
                  disabled={bundleBusy}
                  onClick={async () => {
                    setBundleBusy(true);
                    try {
                      await onExportBundle(bundleFormats, includeTrackingVideo);
                      setExportMenuOpen(false);
                    } finally {
                      setBundleBusy(false);
                    }
                  }}
                >
                  <ActionIcon name="download" />
                  {bundleBusy ? "Rendering video…" : "Download folder (.zip)"}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>
      <svg
        className="manual-fog-bar"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="slider"
        aria-label="Manual FoG annotation timeline"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, frameCount - 1)}
        aria-valuenow={frameIndex}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const selected = frameFromClientX(event.clientX, event.currentTarget);
          dragStateRef.current = {
            mode: "create",
            segmentIndex: null,
            startFrame: selected,
            originalStart: selected,
            originalEnd: selected,
          };
          onSelectSegment(null);
          onFrameSelect(selected);
          onDraftChange({ start: selected, end: selected });
        }}
        onPointerMove={(event) => {
          if (event.buttons !== 1 || dragStateRef.current === null) {
            return;
          }
          const selected = frameFromClientX(event.clientX, event.currentTarget);
          onFrameSelect(selected);
          const drag = dragStateRef.current;
          if (drag.mode === "create") {
            onDraftChange({ start: drag.startFrame, end: selected });
            return;
          }
          if (drag.segmentIndex === null) {
            return;
          }
          if (drag.mode === "resize-start") {
            onUpdateSegment(drag.segmentIndex, selected, drag.originalEnd);
          } else if (drag.mode === "resize-end") {
            onUpdateSegment(drag.segmentIndex, drag.originalStart, selected);
          } else {
            const delta = selected - drag.startFrame;
            const duration = drag.originalEnd - drag.originalStart;
            const nextStart = clamp(drag.originalStart + delta, 0, Math.max(0, frameCount - 1 - duration));
            onUpdateSegment(drag.segmentIndex, nextStart, nextStart + duration);
          }
        }}
        onPointerUp={(event) => {
          const drag = dragStateRef.current;
          const end = frameFromClientX(event.clientX, event.currentTarget);
          dragStateRef.current = null;
          onDraftChange(null);
          if (drag?.mode === "create") {
            onCommit(drag.startFrame, end);
          }
        }}
        onPointerCancel={() => {
          dragStateRef.current = null;
          onDraftChange(null);
        }}
      >
        <rect x={0} y={0} width={width} height={height} rx={12} className="manual-fog-bg" />
        {visibleDeleted.map((deleted, idx) => {
          const rect = segmentRect(deleted.startFrameIndex, deleted.endFrameIndex);
          return (
            <line
              key={`deleted-${deleted.startFrameIndex}-${deleted.endFrameIndex}-${idx}`}
              x1={rect.x}
              x2={rect.x + rect.width}
              y1={height / 2}
              y2={height / 2}
              className="manual-fog-tombstone"
            >
              <title>
                Deleted prediction · frames {deleted.startFrameIndex}–{deleted.endFrameIndex}
              </title>
            </line>
          );
        })}
        {visibleSegments.map((segment) => {
          const index = segments.indexOf(segment);
          const rect = segmentRect(segment.startFrameIndex, segment.endFrameIndex);
          const isSelected = selectedSegmentIndex === index;
          const bands = segmentBands(segment);
          const startBandKind = bands[0]?.kind ?? "manual";
          const endBandKind = bands[bands.length - 1]?.kind ?? startBandKind;
          return (
            <g key={`${segment.startFrameIndex}-${segment.endFrameIndex}-${index}`}>
              {bands.map((band, bandIndex) => {
                const bandRect = segmentRect(band.start, band.end);
                if (bandRect.width <= 0) return null;
                // Use rounded corners only on the outer edges of the segment
                // so adjacent bands butt up cleanly against each other.
                const isFirst = bandIndex === 0;
                const isLast = bandIndex === bands.length - 1;
                const radius = 6;
                const path = bandRectPath(
                  bandRect.x,
                  7,
                  bandRect.width,
                  height - 14,
                  isFirst ? radius : 0,
                  isLast ? radius : 0,
                );
                return (
                  <path
                    key={`band-${bandIndex}-${band.start}-${band.end}`}
                    d={path}
                    className={`manual-fog-band band-${band.kind}`}
                  />
                );
              })}
              {isSelected ? (
                <rect
                  x={rect.x}
                  y={7}
                  width={rect.width}
                  height={height - 14}
                  rx={6}
                  className="manual-fog-selection"
                />
              ) : null}
              <rect
                x={rect.x}
                y={7}
                width={rect.width}
                height={height - 14}
                rx={6}
                className="manual-fog-hit"
                onPointerDown={(event) => {
                  event.stopPropagation();
                  event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
                  const selected = frameFromClientX(event.clientX, event.currentTarget.ownerSVGElement as SVGSVGElement);
                  dragStateRef.current = {
                    mode: "move",
                    segmentIndex: index,
                    startFrame: selected,
                    originalStart: segment.startFrameIndex,
                    originalEnd: segment.endFrameIndex,
                  };
                  onSelectSegment(index);
                  onFrameSelect(selected);
                }}
              />
              <rect
                x={Math.max(0, rect.x - 2)}
                y={9}
                width={5}
                height={height - 18}
                rx={3}
                className={`manual-fog-handle band-${startBandKind} ${isSelected ? "selected" : ""}`}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
                  const selected = frameFromClientX(event.clientX, event.currentTarget.ownerSVGElement as SVGSVGElement);
                  dragStateRef.current = {
                    mode: "resize-start",
                    segmentIndex: index,
                    startFrame: selected,
                    originalStart: segment.startFrameIndex,
                    originalEnd: segment.endFrameIndex,
                  };
                  onSelectSegment(index);
                  onFrameSelect(selected);
                }}
              />
              <rect
                x={Math.min(width - 5, rect.x + rect.width - 3)}
                y={9}
                width={5}
                height={height - 18}
                rx={3}
                className={`manual-fog-handle band-${endBandKind} ${isSelected ? "selected" : ""}`}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
                  const selected = frameFromClientX(event.clientX, event.currentTarget.ownerSVGElement as SVGSVGElement);
                  dragStateRef.current = {
                    mode: "resize-end",
                    segmentIndex: index,
                    startFrame: selected,
                    originalStart: segment.startFrameIndex,
                    originalEnd: segment.endFrameIndex,
                  };
                  onSelectSegment(index);
                  onFrameSelect(selected);
                }}
              />
            </g>
          );
        })}
        {draft ? (
          <rect
            x={segmentRect(draft.start, draft.end).x}
            y={4}
            width={segmentRect(draft.start, draft.end).width}
            height={height - 8}
            rx={8}
            className="manual-fog-draft"
          />
        ) : null}
        <line x1={cursorX} x2={cursorX} y1={0} y2={height} className="fog-density-cursor" />
      </svg>
    </div>
  );
}

// Read-only ground-truth track: shows dataset + imported FoG episodes, scores the
// human annotations against them (frame precision/recall/F1), and offers import.
function GoldTruthBar({
  runDetail,
  frameIndex,
  viewWindow,
  annotationSegments,
  importedSegments,
  onImport,
  onFrameSelect,
}: {
  runDetail: RunDetail;
  frameIndex: number;
  viewWindow: { start: number; end: number } | null;
  annotationSegments: FogAnnotationSegment[];
  importedSegments?: GoldTruthSegment[];
  onImport?: (files: FileList) => void;
  onFrameSelect: (index: number) => void;
}) {
  const width = 1120;
  const height = 30;
  const frameCount = runDetail.frames.length;
  const gtFileInputRef = useRef<HTMLInputElement | null>(null);
  const importedCount = importedSegments?.length ?? 0;
  const segments = useMemo(
    () => [...goldTruthSegments(runDetail), ...(importedSegments ?? [])],
    [runDetail, importedSegments],
  );
  const viewStart = viewWindow?.start ?? 0;
  const viewEnd = viewWindow?.end ?? Math.max(0, frameCount - 1);
  const viewSpan = Math.max(1, viewEnd - viewStart);
  const cursorX = ((clamp(frameIndex, viewStart, viewEnd) - viewStart) / viewSpan) * width;
  const visibleSegments = segments.filter(
    (segment) => segment.endFrameIndex >= viewStart && segment.startFrameIndex <= viewEnd,
  );
  const metrics = useMemo(
    () => frameBenchmarkMetrics(annotationSegments, segments, frameCount),
    [annotationSegments, frameCount, segments],
  );

  // Frame index under a pointer x, mapped through the current zoom window.
  function frameFromClientX(clientX: number, currentTarget: SVGSVGElement): number {
    const rect = currentTarget.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return Math.round(viewStart + ratio * viewSpan);
  }

  // Pixel x/width of a frame range within the current zoom window (min 5px wide).
  function segmentRect(startFrame: number, endFrame: number): { x: number; width: number } {
    const start = clamp(Math.min(startFrame, endFrame), viewStart, viewEnd);
    const end = clamp(Math.max(startFrame, endFrame), viewStart, viewEnd);
    const x = ((start - viewStart) / viewSpan) * width;
    const x2 = ((end - viewStart) / viewSpan) * width;
    return { x, width: Math.max(5, x2 - x) };
  }

  return (
    <div className="gold-truth">
      <div className="gold-truth-meta">
        <div className="gold-truth-title">
          <span>Ground truth</span>
          <small>
            {segments.length} FoG episode{segments.length === 1 ? "" : "s"}
            {importedCount > 0 ? ` · ${importedCount} imported` : ""}
          </small>
        </div>
        <div className="gold-truth-metrics">
          {segments.length === 0 ? (
            <span className="gold-truth-empty">No ground truth loaded — import to score the model</span>
          ) : (
            <>
              <span>Frame precision <strong>{fmtMetric(metrics.precision)}</strong></span>
              <span>Frame recall <strong>{fmtMetric(metrics.recall)}</strong></span>
              <span>Frame F1 <strong>{fmtMetric(metrics.f1)}</strong></span>
            </>
          )}
          {onImport ? (
            <>
              <input
                ref={gtFileInputRef}
                type="file"
                accept=".json,.csv,.tsv,.txt,.eaf,.xml,application/json,text/csv,text/tab-separated-values,text/plain,text/xml"
                multiple
                style={{ display: "none" }}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) onImport(e.target.files);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                className="gold-truth-import"
                onClick={() => gtFileInputRef.current?.click()}
                title="Import ground-truth FoG annotations — ELAN .eaf, kinematics .csv, annotations .json, kinesia label files, generic JSON [{startSec,endSec}] or CSV start,end[,label]"
              >
                + Import ground truth
              </button>
            </>
          ) : null}
        </div>
      </div>
      {segments.length > 0 ? (
      <svg
        className="gold-truth-bar"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="slider"
        aria-label="Ground truth FoG timeline"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, frameCount - 1)}
        aria-valuenow={frameIndex}
        onPointerDown={(event) => {
          onFrameSelect(frameFromClientX(event.clientX, event.currentTarget));
        }}
      >
        <rect x={0} y={0} width={width} height={height} rx={10} className="gold-truth-bg" />
        {visibleSegments.map((segment, index) => {
          const rect = segmentRect(segment.startFrameIndex, segment.endFrameIndex);
          return (
            <g key={`${segment.datasetId}-${segment.startFrameIndex}-${segment.endFrameIndex}-${index}`}>
              <rect
                x={rect.x}
                y={6}
                width={rect.width}
                height={height - 12}
                rx={5}
                className="gold-truth-segment"
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onFrameSelect(segment.startFrameIndex);
                }}
              />
              {rect.width > 34 ? (
                <title>
                  {segment.label}: {segment.startSec.toFixed(2)}s to {segment.endSec.toFixed(2)}s
                  {segment.source ? ` (${segment.source})` : ""}
                </title>
              ) : null}
            </g>
          );
        })}
        <line x1={cursorX} x2={cursorX} y1={0} y2={height} className="gold-truth-cursor" />
      </svg>
      ) : null}
    </div>
  );
}

// Top-level viewer: the runs/jobs sidebar, the upload + video-editing workflow,
// the synced video / 3D / kinematics stage, and the FoG annotation + ground-truth
// tracks. Owns essentially all of the page's interactive state. When
// `embeddedRunId` is set it runs chromeless, locked to a single run (wizard mode).
export function ViewerShell({ embeddedRunId }: { embeddedRunId?: string } = {}) {
  // Embedded mode (e.g. inside the wizard run step): hide the sidebar/chrome
  // and lock the viewer to a single run passed in by the parent.
  const embedded = embeddedRunId != null;
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  // Read ?run=X from the URL so the guided wizard can deep-link back here
  // with a freshly-processed run pre-selected.
  const initialRunFromUrl =
    embeddedRunId ??
    (typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("run")
      : null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(initialRunFromUrl);
  // Follow the embedded run prop when it changes (e.g. "Run another job").
  useEffect(() => {
    if (embeddedRunId) setSelectedRunId(embeddedRunId);
  }, [embeddedRunId]);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runLoadState, setRunLoadState] = useState<RunLoadState | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameCursor, setFrameCursor] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeedPercent, setPlaybackSpeedPercent] = useState(100);
  const [showMesh, setShowMesh] = useState(true);
  const [showJoints, setShowJoints] = useState(true);
  const [showBones, setShowBones] = useState(true);
  const [meshOpacityPercent, setMeshOpacityPercent] = useState(100);
  const [plotGroup, setPlotGroup] = useState<PlotGroup>("angles");
  const [activePlotJointIndices, setActivePlotJointIndices] = useState<number[]>(DEFAULT_ACTIVE_PLOT_JOINTS);
  const [selectedSignalIds, setSelectedSignalIds] = useState<string[]>([]);
  const [manualFogByRun, setManualFogByRun] = useState<Record<string, FogAnnotationSegment[]>>({});
  const [importedGtByRun, setImportedGtByRun] = useState<Record<string, GoldTruthSegment[]>>({});
  const [deletedAutoFogByRun, setDeletedAutoFogByRun] = useState<Record<string, DeletedFogPrediction[]>>({});
  const [fogThresholdByRun, setFogThresholdByRun] = useState<Record<string, number>>({});
  const [fogCleanupMinDurationSec, setFogCleanupMinDurationSec] = useState(DEFAULT_FOG_MIN_DURATION_SEC);
  const [fogCleanupMergeGapSec, setFogCleanupMergeGapSec] = useState(DEFAULT_FOG_MERGE_GAP_SEC);
  const [signalPickerOpen, setSignalPickerOpen] = useState(false);
  const [plotLayoutMode, setPlotLayoutMode] = useState<"stacked" | "overlay">("overlay");
  // Which view fills the left media pane (3D stays on the right).
  const [leftView, setLeftView] = useState<"video" | "box">("box");
  // Left/right media split, % width of the left pane within the media row.
  const [mediaColPct, setMediaColPct] = useState(50);
  // Joint + signal selectors are hidden behind a settings toggle to keep the
  // panel focused on the plots/tracks.
  const [plotSettingsOpen, setPlotSettingsOpen] = useState(false);
  const [fogSettingsOpen, setFogSettingsOpen] = useState(false);
  const [fogScoreSignalId, setFogScoreSignalId] = useState<"fog.score" | "fog.score_smooth">("fog.score");
  const [annotationWindowFrames, setAnnotationWindowFrames] = useState<number | null>(null);
  // Window POSITION (start frame) set by the timeline navigator. null = follow the
  // playhead; a number = the user scrolled there and the window stays put.
  const [windowStartFrame, setWindowStartFrame] = useState<number | null>(null);
  const [annotationCacheLoaded, setAnnotationCacheLoaded] = useState(false);
  const [draftFogAnnotation, setDraftFogAnnotation] = useState<DraftAnnotation>(null);
  const [selectedFogSegmentIndex, setSelectedFogSegmentIndex] = useState<number | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchUploadProgress, setBatchUploadProgress] = useState<{ done: number; total: number } | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingFileUrl, setPendingFileUrl] = useState<string | null>(null);
  const [pendingStagedUploadId, setPendingStagedUploadId] = useState<string | null>(null);
  const [patientDetectionMode, setPatientDetectionMode] = useState<PatientDetectionMode>("auto");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  // Vertical split between the media row (video + 3D) and the analysis panel,
  // as a percentage of the split region's height. Adjusted by the divider drag.
  const [mediaSplitPct, setMediaSplitPct] = useState(33);
  const [videoEditTool, setVideoEditTool] = useState<VideoEditTool>("subject");
  const [videoDurationSec, setVideoDurationSec] = useState(0);
  const [pendingVideoTimeSec, setPendingVideoTimeSec] = useState(0);
  const [isPendingVideoPlaying, setIsPendingVideoPlaying] = useState(false);
  const [trimStartSec, setTrimStartSec] = useState(0);
  const [trimEndSec, setTrimEndSec] = useState(0);
  const [timelineCutPointsSec, setTimelineCutPointsSec] = useState<number[]>([]);
  const [removedTimelineCutSegments, setRemovedTimelineCutSegments] = useState<TimelineRemovedSegment[]>([]);
  const [selectedTimelineSegmentId, setSelectedTimelineSegmentId] = useState<string | null>(null);
  const [cropBox, setCropBox] = useState<SubjectBox | null>(null);
  const [cropDragStart, setCropDragStart] = useState<{ x: number; y: number } | null>(null);
  const [subjectFrame, setSubjectFrame] = useState(0);
  const [subjectBox, setSubjectBox] = useState<SubjectBox | null>(null);
  const [subjectBoxFrameSec, setSubjectBoxFrameSec] = useState<number | null>(null);
  const [subjectDragStart, setSubjectDragStart] = useState<{ x: number; y: number } | null>(null);
  const [autoSubjectPreview, setAutoSubjectPreview] = useState<SubjectPreview | null>(null);
  const [autoSubjectPreviewSamples, setAutoSubjectPreviewSamples] = useState<SubjectPreview[]>([]);
  const [isPreviewingAutoSubject, setIsPreviewingAutoSubject] = useState(false);
  const [subjectPreviewError, setSubjectPreviewError] = useState<string | null>(null);
  const [subjectPromptText, setSubjectPromptText] = useState<string>(DEFAULT_SUBJECT_PROMPT);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [runName, setRunName] = useState("");
  const [inferenceTarget, setInferenceTarget] = useState<InferenceTarget>("body");
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Parse one or more dropped/selected annotation files into ground-truth
  // segments for the current run, de-duping against existing imports and
  // surfacing per-file parse failures.
  const importGroundTruth = useCallback(
    async (files: FileList) => {
      if (!selectedRunId || !runDetail) return;
      const fps = runDetail.fps || 30;
      const frameCount = runDetail.frames.length;
      const parsed: GoldTruthSegment[] = [];
      const failed: string[] = []; // files that threw or yielded nothing
      for (const file of Array.from(files)) {
        try {
          const segs = parseAnnotationsFile(await file.text(), file.name, { fps, frameCount });
          if (segs.length === 0) {
            failed.push(file.name);
            continue;
          }
          for (const seg of segs) {
            parsed.push({
              startFrameIndex: seg.startFrameIndex,
              endFrameIndex: seg.endFrameIndex,
              startSec: seg.startSec,
              endSec: seg.endSec,
              label: seg.label,
              datasetId: file.name,
              split: "holdout",
              labelEventsPath: null,
              source: seg.source,
            });
          }
        } catch {
          failed.push(file.name);
        }
      }
      if (parsed.length > 0) {
        // Dedupe inside the updater so it checks against the freshest state —
        // re-importing the same file (even before a prior import re-rendered)
        // never stacks duplicate episodes.
        setImportedGtByRun((current) => {
          const prior = current[selectedRunId] ?? [];
          const seen = new Set(prior.map((s) => `${s.label}:${s.startFrameIndex}:${s.endFrameIndex}`));
          const added: GoldTruthSegment[] = [];
          for (const seg of parsed) {
            const key = `${seg.label}:${seg.startFrameIndex}:${seg.endFrameIndex}`;
            if (seen.has(key)) continue;
            seen.add(key);
            added.push(seg);
          }
          return added.length ? { ...current, [selectedRunId]: [...prior, ...added] } : current;
        });
      }
      // Surface per-file failures even when other files imported successfully.
      if (failed.length > 0) {
        const supported =
          "Supported: ELAN .eaf, the kinematics .csv, the annotations .json, " +
          "kinesia label files ({episodes:[{start_ms,end_ms}]}), JSON [{startSec,endSec}] or CSV start,end.";
        setError(
          parsed.length === 0
            ? `No FoG intervals found in ${failed.join(", ")}. ${supported}`
            : `Imported some files, but found no FoG intervals in: ${failed.join(", ")}. ${supported}`,
        );
      }
    },
    [runDetail, selectedRunId],
  );
  const [isFullScreen, setIsFullScreen] = useState(false);
  // When true, the main area shows the guided 4-step wizard instead of the
  // 3D viewer. The sidebar (runs list, brand, etc.) remains visible so the
  // user can still navigate between processed runs while the wizard is open.
  const [wizardOpen, setWizardOpen] = useState(false);
  const { dispatch: wizardDispatch } = useWizard();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const rafPlaybackRef = useRef<number | null>(null);
  const latestFrameIndexRef = useRef(0);
  const latestFrameCursorRef = useRef(0);
  const lastPlaybackSeekRef = useRef(0);
  const pendingVideoRef = useRef<HTMLVideoElement | null>(null);
  const pendingFileRef = useRef<File | null>(null);
  const batchFilesRef = useRef<File[]>([]);
  const cameraPreviewRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const mediaSplitRef = useRef<HTMLDivElement | null>(null);
  const splitDragRef = useRef(false);

  // Horizontal divider between the media row (video + 3D) and the analysis panel.
  const onSplitDividerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!mediaSplitRef.current) return;
    event.preventDefault();
    splitDragRef.current = true;
    const el = event.currentTarget;
    el.classList.add("is-dragging");
    try {
      el.setPointerCapture(event.pointerId);
    } catch {
      // ignore: pointer capture is best-effort
    }
  }, []);
  const onSplitDividerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!splitDragRef.current || !mediaSplitRef.current) return;
    const rect = mediaSplitRef.current.getBoundingClientRect();
    if (rect.height <= 0) return;
    const pct = ((event.clientY - rect.top) / rect.height) * 100;
    setMediaSplitPct(clamp(pct, 20, 84));
  }, []);
  const onSplitDividerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
    splitDragRef.current = false;
    const el = event.currentTarget;
    el.classList.remove("is-dragging");
    try {
      el.releasePointerCapture(event.pointerId);
    } catch {
      // ignore
    }
  }, []);

  // Vertical divider between the left media pane and the 3D pane.
  const mediaRowRef = useRef<HTMLDivElement | null>(null);
  const colDragRef = useRef(false);
  const onColDividerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!mediaRowRef.current) return;
    event.preventDefault();
    colDragRef.current = true;
    const el = event.currentTarget;
    el.classList.add("is-dragging");
    try {
      el.setPointerCapture(event.pointerId);
    } catch {
      // ignore
    }
  }, []);
  const onColDividerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (!colDragRef.current || !mediaRowRef.current) return;
    const rect = mediaRowRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = ((event.clientX - rect.left) / rect.width) * 100;
    setMediaColPct(clamp(pct, 18, 82));
  }, []);
  const onColDividerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
    colDragRef.current = false;
    const el = event.currentTarget;
    el.classList.remove("is-dragging");
    try {
      el.releasePointerCapture(event.pointerId);
    } catch {
      // ignore
    }
  }, []);

  const frameCount = runDetail?.frames.length ?? 0;
  const safeFrameIndex = clamp(frameIndex, 0, Math.max(0, frameCount - 1));
  const safeFrameCursor = clamp(frameCursor, 0, Math.max(0, frameCount - 1));
  const availableSignals = useMemo(
    () => signalsForGroup(runDetail, plotGroup, activePlotJointIndices),
    [activePlotJointIndices, plotGroup, runDetail],
  );
  const selectedSignals = useMemo(() => {
    const map = new Map(availableSignals.map((signal) => [signal.id, signal]));
    return selectedSignalIds.map((id) => map.get(id)).filter((signal): signal is RunSignal => Boolean(signal));
  }, [availableSignals, selectedSignalIds]);
  // Contiguous frame ranges the user masked (kept in the video at original
  // timing, but skipped by inference → no kinematics). Shaded + labeled on the
  // plots so the data gap reads as intentional.
  const maskedFrameRanges = useMemo<Array<[number, number]>>(() => {
    const out: Array<[number, number]> = [];
    const frames = runDetail?.frames ?? [];
    let start = -1;
    for (let i = 0; i < frames.length; i += 1) {
      const isMasked = frames[i]?.inferenceStatus === "masked";
      if (isMasked && start < 0) {
        start = i;
      } else if (!isMasked && start >= 0) {
        out.push([start, i - 1]);
        start = -1;
      }
    }
    if (start >= 0) out.push([start, frames.length - 1]);
    return out;
  }, [runDetail]);
  const hasManualSubject = validSubjectBox(subjectBox);
  const effectiveTrimEndSec = trimEndSec || videoDurationSec;
  const timelineDurationSec = Math.max(videoDurationSec, 0.1);
  const pendingVideoCursorSec = clamp(pendingVideoTimeSec, 0, Math.max(videoDurationSec, 0));
  const normalizedTimelineCutPoints = useMemo(
    () => normalizeTimelineCutPoints(timelineCutPointsSec, timelineDurationSec),
    [timelineCutPointsSec, timelineDurationSec],
  );
  const removedTimelineSegments = useMemo(
    () =>
      mergeRemovedSegments(
        removedTimelineCutSegments.flatMap((segment) => {
          const normalized = normalizeCutSegment(segment.startSec, segment.endSec, timelineDurationSec);
          return normalized ? [{ ...segment, ...normalized }] : [];
        }),
      ),
    [removedTimelineCutSegments, timelineDurationSec],
  );
  const visibleTimelineSegments = useMemo(
    () => buildTimelineVisibleSegments(trimStartSec, effectiveTrimEndSec, normalizedTimelineCutPoints, removedTimelineSegments),
    [effectiveTrimEndSec, normalizedTimelineCutPoints, removedTimelineSegments, trimStartSec],
  );
  const timelineVisibleDurationSec = visibleTimelineSegments.at(-1)?.displayEndSec ?? 0;
  const timelineCursorOffsetSec = timelineOffsetFromOriginalTime(visibleTimelineSegments, pendingVideoCursorSec);
  const trimCursorPercent = timelineVisibleDurationSec > 0 ? (timelineCursorOffsetSec / timelineVisibleDurationSec) * 100 : 0;
  const selectedTimelineSegment = visibleTimelineSegments.find((segment) => segment.id === selectedTimelineSegmentId) ?? null;
  const removedDurationSec = removedTimelineSegments.reduce((total, segment) => {
    const start = Math.max(segment.startSec, trimStartSec);
    const end = Math.min(segment.endSec, effectiveTrimEndSec);
    return total + Math.max(0, end - start);
  }, 0);
  const editedVideoDurationSec = Math.max(0, timelineVisibleDurationSec || effectiveTrimEndSec - trimStartSec - removedDurationSec);
  const canDeleteSelectedTimelineSegment = Boolean(
    selectedTimelineSegment && editedVideoDurationSec - (selectedTimelineSegment.endSec - selectedTimelineSegment.startSec) >= MIN_TRIM_GAP_SEC,
  );
  const currentAutoSubjectPreview =
    autoSubjectPreview && Math.abs(autoSubjectPreview.frameSec - pendingVideoCursorSec) <= 0.2
      ? autoSubjectPreview
      : null;
  const canRunPendingFile = Boolean(
    pendingFile && !isUploading && editedVideoDurationSec >= MIN_TRIM_GAP_SEC && (patientDetectionMode === "auto" || hasManualSubject),
  );
  const isUploadEditorOpen = Boolean(pendingFileUrl);
  const showViewerControls = !isUploadEditorOpen && (!PUBLIC_BASIC_UI || Boolean(runDetail));
  const effectiveInferenceTarget: InferenceTarget = PUBLIC_BASIC_UI ? "body" : inferenceTarget;
  const selectedJob = useMemo(
    () => jobs.find((job) => selectedRunId && job.runId === selectedRunId) ?? null,
    [jobs, selectedRunId],
  );
  const activeJobForRun = selectedJob && isActiveJob(selectedJob) ? selectedJob : null;
  const activeJobProcessedFrames = activeJobForRun?.processedFrames ?? null;
  const activeJobStatus = activeJobForRun?.status ?? null;
  const activeRunLoadState = runLoadState?.runId === selectedRunId ? runLoadState : null;
  // Run dirs are created when a job starts, so an in-progress run would otherwise
  // show up under "Processed videos". Hide those — they belong to the Jobs list
  // until the job finishes.
  const activeJobRunIds = useMemo(
    () => new Set(jobs.filter((job) => isActiveJob(job)).map((job) => job.runId)),
    [jobs],
  );
  const processedRuns = useMemo(
    () => runs.filter((run) => !activeJobRunIds.has(run.id)),
    [runs, activeJobRunIds],
  );

  // Toggle 3D/video playback, rewinding to the start first if already at the end.
  function toggleRunPlayback(): void {
    if (!runDetail || frameCount <= 1) {
      return;
    }
    setIsPlaying((current) => {
      const next = !current;
      if (next && safeFrameIndex >= frameCount - 1) {
        setFrameIndex(0);
        setFrameCursor(0);
        const video = videoRef.current;
        if (video) {
          try {
            video.currentTime = displayedVideoTime(runDetail, 0);
          } catch {
            // Media metadata may not be ready yet.
          }
        }
      }
      return next;
    });
  }

  const manualFogSegments = selectedRunId ? manualFogByRun[selectedRunId] ?? [] : [];
  const deletedAutoFogSegments = selectedRunId ? deletedAutoFogByRun[selectedRunId] ?? [] : [];

  // ── Annotation undo/redo ──────────────────────────────────────────────────
  // Per-run history of {manual, deleted} snapshots. recordFogHistory() captures
  // the pre-mutation state and is called at the top of every annotation edit.
  const manualFogByRunRef = useRef(manualFogByRun);
  manualFogByRunRef.current = manualFogByRun;
  const deletedAutoFogByRunRef = useRef(deletedAutoFogByRun);
  deletedAutoFogByRunRef.current = deletedAutoFogByRun;
  const selectedRunIdRef = useRef(selectedRunId);
  selectedRunIdRef.current = selectedRunId;
  const fogHistoryRef = useRef<Record<string, { past: FogHistorySnapshot[]; future: FogHistorySnapshot[] }>>({});
  const [fogHistoryTick, setFogHistoryTick] = useState(0);

  const recordFogHistory = useCallback(() => {
    const runId = selectedRunIdRef.current;
    if (!runId) return;
    const h = fogHistoryRef.current[runId] ?? { past: [], future: [] };
    h.past.push({
      manual: manualFogByRunRef.current[runId] ?? [],
      deleted: deletedAutoFogByRunRef.current[runId] ?? [],
    });
    if (h.past.length > 100) h.past.shift();
    h.future = [];
    fogHistoryRef.current[runId] = h;
    setFogHistoryTick((t) => t + 1);
  }, []);

  // Restore the previous annotation snapshot, pushing the current one onto redo.
  const undoFogAnnotation = useCallback(() => {
    const runId = selectedRunIdRef.current;
    if (!runId) return;
    const h = fogHistoryRef.current[runId];
    if (!h || h.past.length === 0) return;
    const prev = h.past.pop()!;
    h.future.push({
      manual: manualFogByRunRef.current[runId] ?? [],
      deleted: deletedAutoFogByRunRef.current[runId] ?? [],
    });
    setManualFogByRun((c) => ({ ...c, [runId]: prev.manual }));
    setDeletedAutoFogByRun((c) => ({ ...c, [runId]: prev.deleted }));
    setSelectedFogSegmentIndex(null);
    setFogHistoryTick((t) => t + 1);
  }, []);

  // Re-apply the most recently undone snapshot, pushing the current one onto past.
  const redoFogAnnotation = useCallback(() => {
    const runId = selectedRunIdRef.current;
    if (!runId) return;
    const h = fogHistoryRef.current[runId];
    if (!h || h.future.length === 0) return;
    const next = h.future.pop()!;
    h.past.push({
      manual: manualFogByRunRef.current[runId] ?? [],
      deleted: deletedAutoFogByRunRef.current[runId] ?? [],
    });
    setManualFogByRun((c) => ({ ...c, [runId]: next.manual }));
    setDeletedAutoFogByRun((c) => ({ ...c, [runId]: next.deleted }));
    setSelectedFogSegmentIndex(null);
    setFogHistoryTick((t) => t + 1);
  }, []);

  const fogHistory = selectedRunId ? fogHistoryRef.current[selectedRunId] : undefined;
  const canUndoFog = (fogHistory?.past.length ?? 0) > 0;
  const canRedoFog = (fogHistory?.future.length ?? 0) > 0;
  void fogHistoryTick; // canUndo/canRedo recompute when the tick bumps

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const key = event.key.toLowerCase();
      if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        undoFogAnnotation();
      } else if ((key === "z" && event.shiftKey) || key === "y") {
        event.preventDefault();
        redoFogAnnotation();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undoFogAnnotation, redoFogAnnotation]);
  const activeFogThreshold = selectedRunId
    ? fogThresholdByRun[selectedRunId] ?? runDetail?.fog?.threshold ?? DEFAULT_FOG_THRESHOLD
    : DEFAULT_FOG_THRESHOLD;
  const autoFogSegments = useMemo(
    () => (runDetail ? predictedFogSegments(runDetail, activeFogThreshold, fogScoreSignalId) : []),
    [activeFogThreshold, fogScoreSignalId, runDetail],
  );
  const autoFogSegmentsForAnnotation = useMemo(
    () => autoFogSegments.filter((segment) => !deletedAutoFogSegments.some((deleted) => sameFrameSpan(segment, deleted))),
    [autoFogSegments, deletedAutoFogSegments],
  );
  const cleanedAutoFogSegments = useMemo(() => {
    if (!runDetail) {
      return [];
    }
    return postProcessFogSegments(
      runDetail,
      autoFogSegmentsForAnnotation,
      fogCleanupMinDurationSec,
      fogCleanupMergeGapSec,
    ).filter((segment) => !deletedAutoFogSegments.some((deleted) => sameFrameSpan(segment, deleted)));
  }, [
    autoFogSegmentsForAnnotation,
    deletedAutoFogSegments,
    fogCleanupMergeGapSec,
    fogCleanupMinDurationSec,
    runDetail,
  ]);
  const annotationWindow = useMemo(() => {
    if (!runDetail || annotationWindowFrames === null || frameCount <= 0) {
      return null;
    }
    const span = clamp(annotationWindowFrames, 1, frameCount);
    const maxStart = Math.max(0, frameCount - span);
    // Use the navigator's scrolled position if set; otherwise trail the playhead.
    const start =
      windowStartFrame !== null
        ? clamp(windowStartFrame, 0, maxStart)
        : clamp(safeFrameIndex - span + 1, 0, maxStart);
    return { start, end: Math.min(frameCount - 1, start + span - 1) };
  }, [annotationWindowFrames, windowStartFrame, frameCount, runDetail, safeFrameIndex]);

  useEffect(() => {
    try {
      // Manual annotations and deletions are intentionally session-only:
      // reload returns the viewer to the model's predictions. Purge any
      // legacy data so users who had localStorage from older builds get a
      // clean slate too. Export still lets annotators save their work.
      try {
        window.localStorage.removeItem(MANUAL_FOG_STORAGE_KEY);
        window.localStorage.removeItem(DELETED_AUTO_FOG_STORAGE_KEY);
      } catch {
        // ignore — localStorage may be unavailable
      }
      const rawThresholds = window.localStorage.getItem(FOG_THRESHOLD_STORAGE_KEY);
      if (rawThresholds) {
        const parsed = JSON.parse(rawThresholds) as Record<string, number>;
        if (parsed && typeof parsed === "object") {
          const thresholds: Record<string, number> = {};
          for (const [runId, value] of Object.entries(parsed)) {
            if (typeof value === "number" && Number.isFinite(value)) {
              thresholds[runId] = clamp(value, 0.01, 0.99);
            }
          }
          setFogThresholdByRun(thresholds);
        }
      }
      const rawPostProcess = window.localStorage.getItem(FOG_POSTPROCESS_STORAGE_KEY);
      if (rawPostProcess) {
        const parsed = JSON.parse(rawPostProcess) as { minDurationSec?: unknown; mergeGapSec?: unknown };
        if (typeof parsed.minDurationSec === "number" && Number.isFinite(parsed.minDurationSec)) {
          setFogCleanupMinDurationSec(clamp(parsed.minDurationSec, 0, 10));
        }
        if (typeof parsed.mergeGapSec === "number" && Number.isFinite(parsed.mergeGapSec)) {
          setFogCleanupMergeGapSec(clamp(parsed.mergeGapSec, 0, 10));
        }
      }
    } catch {
      // Ignore corrupt local annotation cache.
    } finally {
      setAnnotationCacheLoaded(true);
    }
  }, []);

  // manualFogByRun and deletedAutoFogByRun are deliberately NOT persisted —
  // a page reload should always show the model's predictions, not any human
  // edits. Annotators who want to keep their work use the Export JSON button.

  useEffect(() => {
    if (!annotationCacheLoaded) {
      return;
    }
    try {
      window.localStorage.setItem(FOG_THRESHOLD_STORAGE_KEY, JSON.stringify(fogThresholdByRun));
    } catch {
      // Local storage may be unavailable in private/browser-restricted contexts.
    }
  }, [annotationCacheLoaded, fogThresholdByRun]);

  useEffect(() => {
    if (!annotationCacheLoaded) {
      return;
    }
    try {
      window.localStorage.setItem(
        FOG_POSTPROCESS_STORAGE_KEY,
        JSON.stringify({
          minDurationSec: fogCleanupMinDurationSec,
          mergeGapSec: fogCleanupMergeGapSec,
        }),
      );
    } catch {
      // Local storage may be unavailable in private/browser-restricted contexts.
    }
  }, [annotationCacheLoaded, fogCleanupMergeGapSec, fogCleanupMinDurationSec]);

  useEffect(() => {
    if (!annotationCacheLoaded || !runDetail || !selectedRunId) {
      return;
    }
    setManualFogByRun((current) => {
      const existing = current[selectedRunId] ?? [];
      const userEditedSegments = existing.filter((segment) => !isAutoGeneratedAnnotation(segment));
      const synced = normalizeAnnotationSegments([...cleanedAutoFogSegments, ...userEditedSegments]);
      if (annotationListsEqual(existing, synced)) {
        return current;
      }
      return { ...current, [selectedRunId]: synced };
    });
    setSelectedFogSegmentIndex(null);
  }, [annotationCacheLoaded, cleanedAutoFogSegments, runDetail, selectedRunId]);

  // annotationSegmentFromFrames bound to the currently loaded run.
  function annotationSegmentFromCurrentRun(
    start: number,
    end: number,
    source: FogAnnotationSource,
    createdAt?: string,
    predictedRange?: { start: number; end: number } | null,
  ): FogAnnotationSegment | null {
    if (!runDetail || runDetail.frames.length === 0) {
      return null;
    }
    return annotationSegmentFromFrames(runDetail, start, end, source, createdAt, predictedRange);
  }

  // Add a new fully-manual FoG segment for the dragged frame range.
  const commitManualFogAnnotation = useCallback(
    (start: number, end: number): void => {
      if (!runDetail || !selectedRunId || runDetail.frames.length === 0) {
        return;
      }
      recordFogHistory();
      // Fully manual segments have no model lineage — predictedRange stays null.
      const segment = annotationSegmentFromCurrentRun(start, end, "manual", undefined, null);
      if (!segment) {
        return;
      }
      setManualFogByRun((current) => ({
        ...current,
        [selectedRunId]: normalizeAnnotationSegments([...(current[selectedRunId] ?? []), segment]),
      }));
      setSelectedFogSegmentIndex(null);
    },
    [runDetail, selectedRunId],
  );

  // Move/resize an existing segment; an edited auto segment becomes auto_corrected
  // while keeping its original predicted range for band rendering.
  const updateManualFogAnnotation = useCallback(
    (segmentIndex: number, start: number, end: number): void => {
      if (!runDetail || !selectedRunId) {
        return;
      }
      recordFogHistory();
      setManualFogByRun((current) => {
        const existing = current[selectedRunId] ?? [];
        const previous = existing[segmentIndex];
        if (!previous) {
          return current;
        }
        const updatedSource = previous.source === "manual" ? "manual" : "auto_corrected";
        const updatedCreatedAt = previous.source === "manual" ? previous.createdAt : new Date().toISOString();
        // Carry the original prediction range untouched so extending an
        // auto segment shows the new ends as manual additions (green).
        const previousPredictedRange =
          previous.predictedStartFrameIndex !== undefined && previous.predictedEndFrameIndex !== undefined
            ? { start: previous.predictedStartFrameIndex, end: previous.predictedEndFrameIndex }
            : null;
        const updated = annotationSegmentFromCurrentRun(
          start,
          end,
          updatedSource,
          updatedCreatedAt,
          previousPredictedRange,
        );
        if (!updated) {
          return current;
        }
        const next = existing.map((segment, index) => (index === segmentIndex ? updated : segment));
        return { ...current, [selectedRunId]: next };
      });
      setSelectedFogSegmentIndex(segmentIndex);
    },
    [runDetail, selectedRunId],
  );

  // Delete the selected segment; auto segments are tombstoned so they don't
  // simply re-appear from the prediction sync.
  const deleteSelectedManualFogAnnotation = useCallback((): void => {
    if (!selectedRunId || selectedFogSegmentIndex === null) {
      return;
    }
    recordFogHistory();
    const selectedSegment = manualFogSegments[selectedFogSegmentIndex];
    if (selectedSegment && isAutomaticAnnotation(selectedSegment)) {
      setDeletedAutoFogByRun((current) => ({
        ...current,
        [selectedRunId]: [
          ...(current[selectedRunId] ?? []),
          {
            ...selectedSegment,
            deletedAt: new Date().toISOString(),
            deletedAtFrameIndex: safeFrameIndex,
            originalSource: selectedSegment.source,
          },
        ],
      }));
    }
    setManualFogByRun((current) => {
      const nextSegments = (current[selectedRunId] ?? []).filter((_, index) => index !== selectedFogSegmentIndex);
      return { ...current, [selectedRunId]: nextSegments };
    });
    setSelectedFogSegmentIndex(null);
  }, [manualFogSegments, safeFrameIndex, selectedFogSegmentIndex, selectedRunId]);

  // Split the selected segment at the current playhead into two segments.
  const splitSelectedManualFogAnnotation = useCallback((): void => {
    if (!runDetail || !selectedRunId || selectedFogSegmentIndex === null) {
      return;
    }
    recordFogHistory();
    const segment = manualFogSegments[selectedFogSegmentIndex];
    if (!segment || safeFrameIndex <= segment.startFrameIndex || safeFrameIndex >= segment.endFrameIndex) {
      return;
    }
    const nextSource = segment.source === "manual" ? "manual" : "auto_corrected";
    const editedAt = segment.source === "manual" ? segment.createdAt : new Date().toISOString();
    // Each half inherits the parent's predicted range intersected with its
    // own frame span — so band rendering stays correct after a split.
    const parentPredStart = segment.predictedStartFrameIndex;
    const parentPredEnd = segment.predictedEndFrameIndex;
    function intersectPred(rangeStart: number, rangeEnd: number): { start: number; end: number } | null {
      if (parentPredStart === undefined || parentPredEnd === undefined) return null;
      const start = Math.max(parentPredStart, rangeStart);
      const end = Math.min(parentPredEnd, rangeEnd);
      if (end < start) return null;
      return { start, end };
    }
    const left = annotationSegmentFromCurrentRun(
      segment.startFrameIndex,
      safeFrameIndex - 1,
      nextSource,
      editedAt,
      intersectPred(segment.startFrameIndex, safeFrameIndex - 1),
    );
    const right = annotationSegmentFromCurrentRun(
      safeFrameIndex,
      segment.endFrameIndex,
      nextSource,
      new Date().toISOString(),
      intersectPred(safeFrameIndex, segment.endFrameIndex),
    );
    if (!left || !right) {
      return;
    }
    setManualFogByRun((current) => {
      const existing = current[selectedRunId] ?? [];
      const next = [
        ...existing.slice(0, selectedFogSegmentIndex),
        left,
        right,
        ...existing.slice(selectedFogSegmentIndex + 1),
      ];
      return { ...current, [selectedRunId]: next };
    });
    setSelectedFogSegmentIndex(selectedFogSegmentIndex + 1);
  }, [manualFogSegments, runDetail, safeFrameIndex, selectedFogSegmentIndex, selectedRunId]);

  // Remove all annotations for the run, tombstoning the auto ones so they stay
  // gone until reset-to-prediction.
  const clearManualFogAnnotations = useCallback((): void => {
    if (!selectedRunId) {
      return;
    }
    recordFogHistory();
    const autoSegments = manualFogSegments.filter(isAutomaticAnnotation);
    if (autoSegments.length > 0) {
      setDeletedAutoFogByRun((current) => ({
        ...current,
        [selectedRunId]: [
          ...(current[selectedRunId] ?? []),
          ...autoSegments.map((segment) => ({
            ...segment,
            deletedAt: new Date().toISOString(),
            deletedAtFrameIndex: safeFrameIndex,
            originalSource: segment.source,
          })),
        ],
      }));
    }
    setManualFogByRun((current) => {
      return { ...current, [selectedRunId]: [] };
    });
    setSelectedFogSegmentIndex(null);
  }, [manualFogSegments, safeFrameIndex, selectedRunId]);

  // Discard all human edits and restore the model's full post-processed prediction.
  const resetManualFogAnnotationsToPrediction = useCallback((): void => {
    if (!selectedRunId || !runDetail) {
      return;
    }
    recordFogHistory();
    // Restore the model's full post-processed predictions for this run:
    // - forget manual segments and edits
    // - forget tombstones so previously deleted predictions reappear
    const fullPredictions = postProcessFogSegments(
      runDetail,
      autoFogSegments,
      fogCleanupMinDurationSec,
      fogCleanupMergeGapSec,
    );
    setDeletedAutoFogByRun((current) => ({ ...current, [selectedRunId]: [] }));
    setManualFogByRun((current) => ({ ...current, [selectedRunId]: fullPredictions }));
    setSelectedFogSegmentIndex(null);
  }, [autoFogSegments, fogCleanupMergeGapSec, fogCleanupMinDurationSec, runDetail, selectedRunId]);

  // Set the per-run FoG decision threshold (clamped away from 0/1).
  const updateFogThreshold = useCallback(
    (threshold: number): void => {
      if (!selectedRunId) {
        return;
      }
      setFogThresholdByRun((current) => ({
        ...current,
        [selectedRunId]: clamp(threshold, 0.01, 0.99),
      }));
    },
    [selectedRunId],
  );

  // Export every channel at once as one ZIP that unpacks to a folder named
  // <video>_<timestamp>/, with a separate per-channel file inside (kinematics,
  // FoG probability, the annotation track, and — when present — ground truth).
  // Each channel's format is chosen independently in the export panel.
  const exportBundle = useCallback(async (formats: ChannelFormats, includeTrackingVideo: boolean): Promise<void> => {
    if (!runDetail || !selectedRunId) return;
    const now = new Date();
    const videoName = runs.find((run) => run.id === selectedRunId)?.id ?? selectedRunId;
    const baseName = bundleBaseName(videoName, now);
    const frameCount = runDetail.frames.length;
    const toInterval = (s: {
      startSec: number;
      endSec: number;
      startFrameIndex?: number;
      endFrameIndex?: number;
      label?: string;
      source?: string;
    }): ExportInterval => ({
      startSec: s.startSec,
      endSec: s.endSec,
      startFrameIndex: s.startFrameIndex,
      endFrameIndex: s.endFrameIndex,
      label: s.label,
      source: s.source,
    });
    const groundTruth = [
      ...goldTruthSegments(runDetail),
      ...(importedGtByRun[selectedRunId] ?? []),
    ].map(toInterval);
    const files = buildBundleFiles(
      {
        baseName,
        runId: selectedRunId,
        videoFileName: `${selectedRunId}.mp4`,
        dateIso: now.toISOString(),
        runDetail,
        fogScores: fogScoreValues(runDetail, "fog.score"),
        fogScoresSmooth: fogScoreValues(runDetail, "fog.score_smooth"),
        threshold: activeFogThreshold,
        annotatedMask: segmentFrameMask(manualFogSegments, frameCount),
        annotationSegments: manualFogSegments.map(toInterval),
        groundTruthSegments: groundTruth,
      },
      formats,
    );
    // Optionally fetch the server-rendered tracking-box MP4 (also written to the
    // run's output folder) and add it to the same ZIP as a binary file.
    const binaryFiles: Array<{ name: string; data: Uint8Array }> = [];
    if (includeTrackingVideo) {
      try {
        const res = await fetch(apiUrl(`/api/runs/${encodeURIComponent(selectedRunId)}/tracking-video`) ?? "");
        if (res.ok) {
          const buf = new Uint8Array(await res.arrayBuffer());
          if (buf.byteLength > 0) binaryFiles.push({ name: `${baseName}_trackingbox.mp4`, data: buf });
        }
      } catch {
        // Non-fatal: still export the data files without the video.
      }
    }
    downloadBundle(baseName, files, now, binaryFiles);
  }, [activeFogThreshold, importedGtByRun, manualFogSegments, runDetail, runs, selectedRunId]);

  // Fetch the processed-runs list, auto-selecting the first run only when nothing
  // is selected and no upload is in progress.
  const loadRuns = useCallback(async () => {
    const res = await apiFetch("/api/runs", { cache: "no-store" });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const body = (await res.json()) as RunsResponse;
    setRuns(body.runs);
    setSelectedRunId((current) => {
      if (current) {
        return current;
      }
      if (pendingFileRef.current || batchFilesRef.current.length > 0) {
        return null;
      }
      return body.runs[0]?.id ?? null;
    });
  }, []);

  // Fetch the generation-jobs list (silently ignores transient failures).
  const loadJobs = useCallback(async () => {
    const res = await apiFetch("/api/jobs", { cache: "no-store" });
    if (!res.ok) {
      return;
    }
    const body = (await res.json()) as JobsResponse;
    setJobs(body.jobs);
  }, []);

  useEffect(() => {
    void loadRuns().catch((err) => setError(String(err)));
    void loadJobs();
    const timer = setInterval(() => {
      void loadRuns().catch(() => undefined);
      void loadJobs();
    }, 2200);
    return () => clearInterval(timer);
  }, [loadJobs, loadRuns]);

  useEffect(() => {
    if (!selectedRunId) {
      setRunDetail(null);
      setRunLoadState(null);
      return;
    }
    let cancelled = false;
    const abortController = new AbortController();
    setIsPlaying(false);
    setFrameIndex(0);
    setFrameCursor(0);
    setRunDetail(null);
    setError(null);
    setRunLoadState({ runId: selectedRunId, loaded: 0, total: 1, label: "Run manifest" });
    void (async () => {
      const res = await apiFetch(`/api/runs/${encodeURIComponent(selectedRunId)}`, {
        cache: "no-store",
        signal: abortController.signal,
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const body = (await res.json()) as RunDetailResponse;
      if (cancelled) {
        return;
      }
      setRunLoadState({ runId: selectedRunId, loaded: 1, total: 1, label: "Run manifest" });
      await preloadRunAssets(
        body.run,
        (progress) => {
          if (!cancelled) {
            setRunLoadState({ runId: selectedRunId, ...progress });
          }
        },
        { includeMeshes: showMesh },
      );
      if (cancelled) {
        return;
      }
      setRunDetail(body.run);
      setRunLoadState(null);
    })().catch((err) => {
      if (cancelled || abortController.signal.aborted) {
        return;
      }
      setRunLoadState(null);
      setError(String(err));
    });
    return () => {
      cancelled = true;
      abortController.abort();
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || !activeJobForRun) {
      return;
    }
    let cancelled = false;
    apiFetch(`/api/runs/${encodeURIComponent(selectedRunId)}`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return (await res.json()) as RunDetailResponse;
      })
      .then((body) => {
        if (cancelled) {
          return;
        }
        setRunDetail(body.run);
        setRunLoadState(null);
        const lastFrame = Math.max(0, body.run.frames.length - 1);
        setFrameIndex(lastFrame);
        setFrameCursor(lastFrame);
        setSelectedSignalIds((current) => {
          const available = new Set(signalsForGroup(body.run, plotGroup, activePlotJointIndices).map((signal) => signal.id));
          const kept = current.filter((id) => available.has(id));
          return kept.length > 0 ? kept : defaultSignalIds(body.run, plotGroup, activePlotJointIndices);
        });
      })
      .catch(() => {
        // The job can be visible a few milliseconds before its first manifest.
        // Polling will pick it up on the next tick.
      });
    return () => {
      cancelled = true;
    };
  }, [activeJobForRun, activeJobProcessedFrames, activeJobStatus, activePlotJointIndices, plotGroup, selectedRunId]);

  useEffect(() => {
    if (!runDetail) {
      setSelectedSignalIds([]);
      return;
    }
    setSelectedSignalIds((current) => {
      const available = new Set(signalsForGroup(runDetail, plotGroup, activePlotJointIndices).map((signal) => signal.id));
      const kept = current.filter((id) => available.has(id));
      return kept.length > 0 ? kept : defaultSignalIds(runDetail, plotGroup, activePlotJointIndices);
    });
  }, [activePlotJointIndices, plotGroup, runDetail]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !runDetail || frameCount === 0) {
      return;
    }
    video.playbackRate = playbackSpeedPercent / 100;
    const target = displayedVideoTime(runDetail, safeFrameCursor);
    if (!isPlaying && Number.isFinite(target) && Math.abs(video.currentTime - target) > 0.04) {
      try {
        video.currentTime = target;
      } catch {
        // Media metadata may not be ready yet.
      }
    }
    if (isPlaying) {
      void video.play().catch(() => undefined);
    } else {
      video.pause();
    }
  }, [frameCount, isPlaying, playbackSpeedPercent, runDetail, safeFrameCursor]);

  useEffect(() => {
    latestFrameIndexRef.current = safeFrameIndex;
  }, [safeFrameIndex]);

  useEffect(() => {
    latestFrameCursorRef.current = safeFrameCursor;
  }, [safeFrameCursor]);

  useEffect(() => {
    if (!isPlaying) {
      setFrameCursor(safeFrameIndex);
    }
  }, [isPlaying, safeFrameIndex]);

  useEffect(() => {
    if (!runDetail || !isPlaying || frameCount <= 1) {
      return;
    }

    const startMs = performance.now();
    const startCursor = latestFrameCursorRef.current;
    const fps = Math.max(1, runDetail.fps || 30);
    const speed = playbackSpeedPercent / 100;
    let lastVideoTime = videoRef.current?.currentTime ?? displayedVideoTime(runDetail, startCursor);
    let lastVideoProgressMs = startMs;
    let lastRenderedIndex = latestFrameIndexRef.current;

    const tick = (nowMs: number): void => {
      if (!runDetail) {
        return;
      }
      const video = videoRef.current;
      let nextCursor: number;
      if (video && !video.paused && video.readyState >= 2) {
        if (Math.abs(video.currentTime - lastVideoTime) > 1e-4) {
          lastVideoTime = video.currentTime;
          lastVideoProgressMs = nowMs;
        }
        const videoIsProgressing = nowMs - lastVideoProgressMs < 350;
        if (videoIsProgressing) {
          nextCursor = frameCursorFromDisplayedVideoTime(runDetail, video.currentTime);
        } else {
          const elapsedFrames = ((nowMs - startMs) / 1000) * fps * speed;
          nextCursor = startCursor + elapsedFrames;
        }
      } else {
        const elapsedFrames = ((nowMs - startMs) / 1000) * fps * speed;
        nextCursor = startCursor + elapsedFrames;
      }
      const reachedEnd = nextCursor >= frameCount - 1;
      nextCursor = clamp(nextCursor, 0, frameCount - 1);
      const nextIndex = clamp(Math.round(nextCursor), 0, frameCount - 1);
      latestFrameCursorRef.current = nextCursor;
      setFrameCursor((current) => (Math.abs(current - nextCursor) < 0.001 ? current : nextCursor));
      if (nextIndex !== lastRenderedIndex) {
        lastRenderedIndex = nextIndex;
        latestFrameIndexRef.current = nextIndex;
        setFrameIndex((current) => (current === nextIndex ? current : nextIndex));
      }

      if (video) {
        const videoFrame = frameIndexFromDisplayedVideoTime(runDetail, video.currentTime);
        const needsSeek = Math.abs(videoFrame - nextIndex) > 10;
        if (needsSeek && nowMs - lastPlaybackSeekRef.current > 700) {
          try {
            video.currentTime = displayedVideoTime(runDetail, nextCursor);
            lastPlaybackSeekRef.current = nowMs;
          } catch {
            // Metadata can still be loading; the frame clock remains authoritative.
          }
        }
        if (reachedEnd) {
          video.pause();
          try {
            video.currentTime = displayedVideoTime(runDetail, frameCount - 1);
          } catch {
            // Metadata can still be loading; the frame clock remains authoritative.
          }
          setIsPlaying(false);
          return;
        }
        if (video.paused) {
          void video.play().catch(() => undefined);
        }
      } else if (reachedEnd) {
        setIsPlaying(false);
        return;
      }

      rafPlaybackRef.current = window.requestAnimationFrame(tick);
    };

    rafPlaybackRef.current = window.requestAnimationFrame(tick);
    return () => {
      if (rafPlaybackRef.current !== null) {
        window.cancelAnimationFrame(rafPlaybackRef.current);
      }
      rafPlaybackRef.current = null;
    };
  }, [frameCount, isPlaying, playbackSpeedPercent, runDetail]);

  useEffect(() => {
    const onFullScreenChange = (): void => {
      setIsFullScreen(document.fullscreenElement === stageRef.current);
    };
    document.addEventListener("fullscreenchange", onFullScreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullScreenChange);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.repeat) {
        return;
      }
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName.toLowerCase();
      const inputType = target instanceof HTMLInputElement ? target.type : "";
      const isTyping =
        (tagName === "input" && inputType !== "range" && inputType !== "button") ||
        tagName === "textarea" ||
        tagName === "select" ||
        target?.isContentEditable;
      if (event.code === "Space") {
        if (isTyping) {
          return;
        }
        event.preventDefault();
        const pendingVideo = pendingVideoRef.current;
        if (pendingFileUrl && pendingVideo) {
          togglePendingVideoPlayback();
          return;
        }
        if (!runDetail || frameCount <= 1) {
          return;
        }
        toggleRunPlayback();
      } else if (event.code === "Delete" || event.code === "Backspace") {
        if (isTyping) {
          return;
        }
        if (selectedFogSegmentIndex !== null) {
          event.preventDefault();
          deleteSelectedManualFogAnnotation();
        }
      } else if (event.code === "Escape") {
        setSelectedFogSegmentIndex(null);
        setDraftFogAnnotation(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    deleteSelectedManualFogAnnotation,
    frameCount,
    pendingFileUrl,
    pendingVideoCursorSec,
    runDetail,
    safeFrameIndex,
    selectedFogSegmentIndex,
    visibleTimelineSegments,
  ]);

  useEffect(() => {
    if (!pendingFile) {
      setPendingFileUrl(null);
      return;
    }
    const url = URL.createObjectURL(pendingFile);
    setPendingFileUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [pendingFile]);

  useEffect(() => {
    if (PUBLIC_BASIC_UI) {
      return;
    }
    void refreshCameraDevices(false);
    return () => {
      stopCameraStream();
    };
  }, []);

  useEffect(() => {
    if (PUBLIC_BASIC_UI) {
      return;
    }
    if (patientDetectionMode === "auto" && videoEditTool === "subject") {
      setVideoEditTool("crop");
    }
  }, [patientDetectionMode, videoEditTool]);

  // Select a single video for the upload/editing workflow, resetting all
  // per-video editing state (trim, cuts, crop, subject, detection mode).
  function chooseFile(file: File | null): void {
    pendingFileRef.current = file;
    batchFilesRef.current = [];
    setPendingFile(file);
    setBatchFiles([]);
    setBatchUploadProgress(null);
    setPendingStagedUploadId(null);
    setRunLoadState(null);
    setError(null);
    setVideoDurationSec(0);
    setPendingVideoTimeSec(0);
    setIsPendingVideoPlaying(false);
    setTrimStartSec(0);
    setTrimEndSec(0);
    setTimelineCutPointsSec([]);
    setRemovedTimelineCutSegments([]);
    setSelectedTimelineSegmentId(null);
    setCropBox(null);
    setCropDragStart(null);
    setVideoEditTool("subject");
    setSubjectBoxFrameSec(null);
    setAutoSubjectPreview(null);
    setAutoSubjectPreviewSamples([]);
    setSubjectPreviewError(null);
    if (file) {
      setPatientDetectionMode("auto");
      setRunDetail(null);
      setSelectedRunId(null);
      setSubjectFrame(0);
      setSubjectBox(null);
      setSubjectBoxFrameSec(null);
      const stem = file.name.replace(/\.[^.]+$/, "") || "input";
      setRunName(`${stem}_processed`);
    }
  }

  // Queue a folder of videos for batch processing (auto-detection only).
  function chooseBatchFiles(files: File[]): void {
    const videos = uniqueVideoFiles(files);
    pendingFileRef.current = null;
    batchFilesRef.current = videos;
    setBatchFiles(videos);
    setBatchUploadProgress(null);
    setRunLoadState(null);
    setPendingFile(null);
    setPendingFileUrl(null);
    setSubjectFrame(0);
    setSubjectBox(null);
    setSubjectBoxFrameSec(null);
    setAutoSubjectPreview(null);
    setAutoSubjectPreviewSamples([]);
    setSubjectPreviewError(null);
    setPatientDetectionMode("auto");
    setVideoDurationSec(0);
    setPendingVideoTimeSec(0);
    setIsPendingVideoPlaying(false);
    setTrimStartSec(0);
    setTrimEndSec(0);
    setCropBox(null);
    setCropDragStart(null);
    setError(videos.length === 0 ? "No supported video file found in this folder." : null);
    if (videos.length > 0) {
      setRunDetail(null);
      setSelectedRunId(null);
      setRunName("");
    }
  }

  // Discard the queued batch of videos.
  function clearBatchFiles(): void {
    batchFilesRef.current = [];
    setBatchFiles([]);
    setBatchUploadProgress(null);
    setError(null);
  }

  // Enumerate webcams (optionally prompting for permission first to get labels).
  async function refreshCameraDevices(requestPermission: boolean): Promise<void> {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraDevices([]);
      return;
    }
    let permissionStream: MediaStream | null = null;
    try {
      if (requestPermission && navigator.mediaDevices.getUserMedia) {
        permissionStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((device) => device.kind === "videoinput");
      setCameraDevices(videoDevices);
      setSelectedCameraId((current) => current || videoDevices[0]?.deviceId || "");
    } catch (err) {
      setError(`Camera device listing failed: ${String(err)}`);
    } finally {
      permissionStream?.getTracks().forEach((track) => track.stop());
    }
  }

  // Point the camera preview <video> at a stream (or detach with null).
  function attachCameraStream(stream: MediaStream | null): void {
    const video = cameraPreviewRef.current;
    if (!video) {
      return;
    }
    video.srcObject = stream;
    if (stream) {
      void video.play().catch(() => undefined);
    }
  }

  // Stop and detach the active camera stream (keeps status if still recording).
  function stopCameraStream(): void {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    attachCameraStream(null);
    setIsCameraActive(false);
    if (!isRecording) {
      setCameraStatus("idle");
    }
  }

  // Open the selected (or default) webcam at ~720p and show its live preview.
  async function startCamera(selectedDeviceId = selectedCameraId): Promise<MediaStream | null> {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera recording is not supported by this browser.");
      setCameraStatus("error");
      return null;
    }
    setError(null);
    setCameraStatus("starting");
    stopCameraStream();
    try {
      const constraints: MediaStreamConstraints = {
        audio: false,
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraStreamRef.current = stream;
      attachCameraStream(stream);
      setIsCameraActive(true);
      setCameraStatus("ready");
      const activeDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId;
      if (activeDeviceId) {
        setSelectedCameraId(activeDeviceId);
      }
      await refreshCameraDevices(false);
      return stream;
    } catch (err) {
      setCameraStatus("error");
      setError(`Camera start failed: ${String(err)}`);
      return null;
    }
  }

  // First MediaRecorder MIME type this browser supports, preferring VP9 WebM.
  function preferredRecordingMimeType(): string {
    const candidates = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ];
    return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
  }

  // Record the camera stream; on stop, wrap the chunks into a File and load it
  // into the upload workflow.
  async function startRecording(): Promise<void> {
    if (typeof MediaRecorder === "undefined") {
      setError("MediaRecorder is not supported by this browser.");
      return;
    }
    const stream = cameraStreamRef.current ?? (await startCamera());
    if (!stream) {
      return;
    }
    const mimeType = preferredRecordingMimeType();
    recordedChunksRef.current = [];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(recordedChunksRef.current, { type });
      const file = new File([blob], `recording_${Date.now()}.${ext}`, { type });
      chooseFile(file);
      setIsRecording(false);
      setCameraStatus("ready");
      mediaRecorderRef.current = null;
    };
    recorder.start(1000);
    setIsRecording(true);
    setCameraStatus("recording");
  }

  // Stop the active recorder (its onstop handler finalizes the file).
  function stopRecording(): void {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  // Snap a source time into the nearest kept (non-removed) timeline segment, so
  // the editor preview never lands inside a deleted span.
  function pendingPlayableTime(seconds: number): number {
    if (visibleTimelineSegments.length === 0) {
      return 0;
    }
    const first = visibleTimelineSegments[0];
    const last = visibleTimelineSegments[visibleTimelineSegments.length - 1];
    const safe = clamp(seconds, first.startSec, last.endSec);
    for (const segment of visibleTimelineSegments) {
      if (safe < segment.startSec) {
        return segment.startSec;
      }
      if (safe <= segment.endSec) {
        return safe;
      }
    }
    return last.endSec;
  }

  // Pause the upload-editor preview video.
  function pausePendingVideo(): void {
    const video = pendingVideoRef.current;
    setIsPendingVideoPlaying(false);
    if (!video) {
      return;
    }
    video.pause();
  }

  // Play the upload-editor preview, restarting from the first kept segment if the
  // cursor is at the end.
  function playPendingVideo(): void {
    const video = pendingVideoRef.current;
    if (!video || visibleTimelineSegments.length === 0) {
      return;
    }
    const first = visibleTimelineSegments[0];
    const last = visibleTimelineSegments[visibleTimelineSegments.length - 1];
    const start = pendingVideoCursorSec >= last.endSec - 0.02 ? first.startSec : pendingPlayableTime(pendingVideoCursorSec);
    setPendingVideoTimeSec(start);
    setSubjectFrame(Math.max(0, Math.round(start * 30)));
    try {
      if (Math.abs(video.currentTime - start) > 0.02) {
        video.currentTime = start;
      }
    } catch {
      // Metadata may not be ready yet.
    }
    void video.play().then(() => setIsPendingVideoPlaying(true)).catch(() => setIsPendingVideoPlaying(false));
  }

  // Play/pause toggle for the upload-editor preview.
  function togglePendingVideoPlayback(): void {
    const video = pendingVideoRef.current;
    if (video && !video.paused) {
      pausePendingVideo();
      return;
    }
    playPendingVideo();
  }

  // Seek the preview to a frame number (assumes 30fps for the subject picker).
  function seekPendingVideo(frame: number): void {
    const video = pendingVideoRef.current;
    pausePendingVideo();
    if (!video) {
      return;
    }
    const safeFrame = Math.max(0, Math.trunc(frame));
    setSubjectFrame(safeFrame);
    try {
      video.currentTime = safeFrame / 30;
    } catch {
      // Metadata may not be ready yet.
    }
  }

  // Seek the preview to a time in seconds (clamped to the video duration).
  function seekPendingVideoSec(seconds: number): void {
    const video = pendingVideoRef.current;
    pausePendingVideo();
    const videoDuration = video && Number.isFinite(video.duration) ? video.duration : 0;
    const safeSeconds = clamp(seconds, 0, Math.max(videoDurationSec, videoDuration, 0));
    setPendingVideoTimeSec(safeSeconds);
    if (video) {
      try {
        video.currentTime = safeSeconds;
      } catch {
        // Metadata may not be ready yet.
      }
    }
    setSubjectFrame(Math.round(safeSeconds * 30));
  }

  // On metadata load, capture the duration and re-clamp all trim/cut state to it.
  function onPendingVideoMetadata(): void {
    const video = pendingVideoRef.current;
    const duration = Number.isFinite(video?.duration) ? Math.max(0, video?.duration ?? 0) : 0;
    setIsPendingVideoPlaying(false);
    setVideoDurationSec(duration);
    setPendingVideoTimeSec((current) => clamp(current, 0, duration));
    setTrimStartSec((current) => clamp(current, 0, duration));
    setTrimEndSec((current) => (current > 0 ? clamp(current, 0, duration) : duration));
    setTimelineCutPointsSec((current) => normalizeTimelineCutPoints(current, duration));
    setRemovedTimelineCutSegments((current) =>
      current.flatMap((segment) => {
        const normalized = normalizeCutSegment(segment.startSec, segment.endSec, duration);
        return normalized ? [{ ...segment, ...normalized }] : [];
      }),
    );
  }

  // Keep preview playback inside the kept timeline: clamp to trim bounds, stop at
  // the end, and skip over deleted segments as they are reached.
  function onPendingVideoTimeUpdate(): void {
    const video = pendingVideoRef.current;
    if (!video || !Number.isFinite(video.currentTime)) {
      return;
    }
    if (visibleTimelineSegments.length === 0) {
      pausePendingVideo();
      return;
    }
    const firstSegment = visibleTimelineSegments[0];
    const lastSegment = visibleTimelineSegments[visibleTimelineSegments.length - 1];
    const safeSeconds = clamp(video.currentTime, 0, Math.max(videoDurationSec, video.duration || 0, 0));
    if (safeSeconds < firstSegment.startSec - 0.02) {
      video.currentTime = firstSegment.startSec;
      return;
    }
    if (safeSeconds >= lastSegment.endSec - 0.01) {
      video.pause();
      setIsPendingVideoPlaying(false);
      setPendingVideoTimeSec(lastSegment.endSec);
      setSubjectFrame(Math.max(0, Math.round(lastSegment.endSec * 30)));
      return;
    }
    const skippedSegment = removedTimelineSegments.find(
      (segment) => safeSeconds >= segment.startSec && safeSeconds < segment.endSec - 0.02,
    );
    if (skippedSegment && !video.paused) {
      video.currentTime = pendingPlayableTime(skippedSegment.endSec);
      return;
    }
    const playableSeconds = pendingPlayableTime(safeSeconds);
    if (Math.abs(playableSeconds - safeSeconds) > 0.02 && !video.paused) {
      video.currentTime = playableSeconds;
      return;
    }
    setPendingVideoTimeSec(safeSeconds);
    setSubjectFrame(Math.max(0, Math.round(safeSeconds * 30)));
  }

  // Source-video time under a pointer on the trim scrubber.
  function timelineSecondsFromPointer(event: PointerEvent<HTMLDivElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return originalTimeFromTimelineOffset(visibleTimelineSegments, ratio * Math.max(timelineVisibleDurationSec, 0));
  }

  // Seek the preview to the scrubber position under the pointer.
  function scrubPendingVideo(event: PointerEvent<HTMLDivElement>): void {
    if (videoDurationSec <= 0) {
      return;
    }
    event.preventDefault();
    seekPendingVideoSec(timelineSecondsFromPointer(event));
  }

  // Begin a scrubber drag (captures the pointer for off-element tracking).
  function startTimelineScrub(event: PointerEvent<HTMLDivElement>): void {
    event.currentTarget.setPointerCapture(event.pointerId);
    scrubPendingVideo(event);
  }

  // Continue a scrubber drag while the primary button is held.
  function updateTimelineScrub(event: PointerEvent<HTMLDivElement>): void {
    if ((event.buttons & 1) === 0) {
      return;
    }
    scrubPendingVideo(event);
  }

  // Keyboard scrubbing on the trim timeline (arrows step, Home/End jump to ends).
  function onTrimTimelineKeyDown(event: ReactKeyboardEvent<HTMLDivElement>): void {
    if (videoDurationSec <= 0) {
      return;
    }
    const step = event.shiftKey ? 1 : 0.05;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      seekPendingVideoSec(originalTimeFromTimelineOffset(visibleTimelineSegments, timelineCursorOffsetSec - step));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      seekPendingVideoSec(originalTimeFromTimelineOffset(visibleTimelineSegments, timelineCursorOffsetSec + step));
    } else if (event.key === "Home") {
      event.preventDefault();
      seekPendingVideoSec(visibleTimelineSegments[0]?.startSec ?? 0);
    } else if (event.key === "End") {
      event.preventDefault();
      seekPendingVideoSec(visibleTimelineSegments.at(-1)?.endSec ?? videoDurationSec);
    }
  }

  // Generate a unique id for a removed timeline segment.
  function newTimelineCutId(): string {
    return typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `cut-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  // Add a cut boundary at the current cursor (ignored if inside a deleted span).
  function addCutAtCursor(): void {
    if (videoDurationSec <= 0) {
      return;
    }
    const cursor = clamp(pendingVideoCursorSec, 0, videoDurationSec);
    const isInsideRemovedSegment = removedTimelineSegments.some(
      (segment) => cursor > segment.startSec + 0.01 && cursor < segment.endSec - 0.01,
    );
    if (isInsideRemovedSegment) {
      return;
    }
    setTimelineCutPointsSec((current) => normalizeTimelineCutPoints([...current, cursor], videoDurationSec));
    setSelectedTimelineSegmentId(null);
  }

  // Remove the selected timeline segment and advance the cursor to the next kept one.
  function deleteSelectedTimelineSegment(): void {
    if (!selectedTimelineSegment || !canDeleteSelectedTimelineSegment) {
      return;
    }
    const removedSegment = {
      id: newTimelineCutId(),
      startSec: selectedTimelineSegment.startSec,
      endSec: selectedTimelineSegment.endSec,
    };
    const nextRemovedSegments = mergeRemovedSegments([...removedTimelineSegments, removedSegment]);
    const nextVisibleSegments = buildTimelineVisibleSegments(
      trimStartSec,
      effectiveTrimEndSec,
      normalizedTimelineCutPoints,
      nextRemovedSegments,
    );
    setRemovedTimelineCutSegments(nextRemovedSegments);
    setSelectedTimelineSegmentId(null);
    const nextCursor =
      nextVisibleSegments.find((segment) => segment.startSec >= selectedTimelineSegment.endSec - 0.001)?.startSec ??
      nextVisibleSegments.at(-1)?.endSec ??
      trimStartSec;
    seekPendingVideoSec(nextCursor);
  }

  // Forget all cut points and removed segments (keeps trim bounds).
  function clearTimelineCuts(): void {
    setTimelineCutPointsSec([]);
    setRemovedTimelineCutSegments([]);
    setSelectedTimelineSegmentId(null);
  }

  // Reset trim bounds and cuts to the full clip.
  function resetTimelineTrim(): void {
    setTrimStartSec(0);
    setTrimEndSec(videoDurationSec);
    clearTimelineCuts();
    seekPendingVideoSec(0);
  }

  // Reset the entire video edit (trim, cuts, crop, tool) back to defaults.
  function resetVideoEdit(): void {
    setTrimStartSec(0);
    setTrimEndSec(videoDurationSec);
    clearTimelineCuts();
    setCropBox(null);
    setCropDragStart(null);
    setVideoEditTool("subject");
    seekPendingVideoSec(0);
  }

  // Normalized [0,1] point of a pointer within the video overlay.
  function overlayPoint(event: PointerEvent<HTMLDivElement>): { x: number; y: number } {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1),
      y: clamp((event.clientY - rect.top) / Math.max(1, rect.height), 0, 1),
    };
  }

  // Begin dragging either the patient subject box or the crop box, depending on
  // detection mode and the active tool.
  function startSubjectSelection(event: PointerEvent<HTMLDivElement>): void {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = overlayPoint(event);
    if (PUBLIC_BASIC_UI) {
      if (patientDetectionMode !== "manual") {
        return;
      }
      setSubjectDragStart(point);
      setSubjectBox({ x: point.x, y: point.y, width: 0, height: 0 });
      setSubjectBoxFrameSec(pendingVideoCursorSec);
      setSubjectPreviewError(null);
      return;
    }
    if (patientDetectionMode === "auto" || videoEditTool === "crop") {
      setCropDragStart(point);
      setCropBox({ x: point.x, y: point.y, width: 0, height: 0 });
      return;
    }
    setSubjectDragStart(point);
    setSubjectBox({ x: point.x, y: point.y, width: 0, height: 0 });
    setSubjectBoxFrameSec(pendingVideoCursorSec);
    setSubjectPreviewError(null);
  }

  // Resize the in-progress subject/crop box as the pointer moves.
  function updateSubjectSelection(event: PointerEvent<HTMLDivElement>): void {
    if (PUBLIC_BASIC_UI) {
      if (patientDetectionMode !== "manual" || !subjectDragStart) {
        return;
      }
      const point = overlayPoint(event);
      setSubjectBox(normalizeBox(subjectDragStart.x, subjectDragStart.y, point.x, point.y));
      return;
    }
    if (patientDetectionMode === "auto" || videoEditTool === "crop") {
      if (!cropDragStart) {
        return;
      }
      const point = overlayPoint(event);
      setCropBox(normalizeBox(cropDragStart.x, cropDragStart.y, point.x, point.y));
      return;
    }
    if (!subjectDragStart) {
      return;
    }
    const point = overlayPoint(event);
    setSubjectBox(normalizeBox(subjectDragStart.x, subjectDragStart.y, point.x, point.y));
  }

  // End a subject/crop drag.
  function endSubjectSelection(): void {
    setCropDragStart(null);
    setSubjectDragStart(null);
  }

  // Convert a source-video time to the frame index within the EDITED clip (trim +
  // removed segments accounted for) that the backend will see for the prompt box.
  function subjectPromptFrame(frameSec: number, fps = 30): number {
    const removedBeforeFrameSec = removedTimelineSegments.reduce((total, segment) => {
      if (segment.endSec <= trimStartSec || segment.startSec >= frameSec) {
        return total;
      }
      const start = Math.max(segment.startSec, trimStartSec);
      const end = Math.min(segment.endSec, frameSec);
      return total + Math.max(0, end - start);
    }, 0);
    const relativeSec = Math.max(0, frameSec - trimStartSec - removedBeforeFrameSec);
    return Math.max(0, Math.round(relativeSec * Math.max(1, fps)));
  }

  // True when a source time falls within the kept timeline (inside trim bounds
  // and not inside a removed segment).
  function isTimeInsideTrim(frameSec: number): boolean {
    const insideOuterTrim = frameSec >= trimStartSec - 0.05 && frameSec <= effectiveTrimEndSec + 0.05;
    const insideRemovedSegment = removedTimelineSegments.some(
      (segment) => frameSec >= segment.startSec - 0.05 && frameSec <= segment.endSec + 0.05,
    );
    return insideOuterTrim && !insideRemovedSegment;
  }

  // The subject prompt box to send with the run: the manual box in manual mode,
  // else the active auto-preview detection — but only if inside the kept timeline.
  function selectedSubjectPrompt(): { box: SubjectBox; frameSec: number; fps: number } | null {
    if (patientDetectionMode === "manual" && validSubjectBox(subjectBox)) {
      const frameSec = subjectBoxFrameSec ?? pendingVideoCursorSec;
      return isTimeInsideTrim(frameSec) ? { box: subjectBox, frameSec, fps: 30 } : null;
    }
    if (patientDetectionMode === "auto" && currentAutoSubjectPreview) {
      if (!isTimeInsideTrim(currentAutoSubjectPreview.frameSec)) {
        return null;
      }
      return {
        box: currentAutoSubjectPreview.box,
        frameSec: currentAutoSubjectPreview.frameSec,
        fps: currentAutoSubjectPreview.fps,
      };
    }
    return null;
  }

  // Upload a large file in fixed-size chunks, then finalize it server-side,
  // returning the staged-upload id to reference it in subsequent requests.
  async function uploadFileInChunks(file: File): Promise<string> {
    const uploadId = typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `upload-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const totalChunks = Math.max(1, Math.ceil(file.size / UPLOAD_CHUNK_BYTES));
    for (let index = 0; index < totalChunks; index += 1) {
      const start = index * UPLOAD_CHUNK_BYTES;
      const end = Math.min(file.size, start + UPLOAD_CHUNK_BYTES);
      const formData = new FormData();
      formData.append("uploadId", uploadId);
      formData.append("fileName", file.name);
      formData.append("chunkIndex", String(index));
      formData.append("totalChunks", String(totalChunks));
      formData.append("chunk", file.slice(start, end), file.name);
      const res = await apiFetch("/api/uploads/chunk", { method: "POST", body: formData });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      setBatchUploadProgress({ done: index + 1, total: totalChunks });
    }
    const completeData = new FormData();
    completeData.append("uploadId", uploadId);
    const completeRes = await apiFetch("/api/uploads/complete", { method: "POST", body: completeData });
    const completeBody = await completeRes.json().catch(() => null) as { stagedUploadId?: string; error?: string } | null;
    if (!completeRes.ok || !completeBody?.stagedUploadId) {
      throw new Error(completeBody?.error || "Failed to complete chunked upload.");
    }
    return completeBody.stagedUploadId;
  }

  // Staged-upload id for a file, chunk-uploading (and caching the id) only when it
  // exceeds the direct-upload size limit; small files return null (sent inline).
  async function stagedUploadIdForFile(file: File): Promise<string | null> {
    if (file.size <= DIRECT_UPLOAD_LIMIT_BYTES) {
      return null;
    }
    if (file === pendingFile && pendingStagedUploadId) {
      return pendingStagedUploadId;
    }
    const stagedUploadId = await uploadFileInChunks(file);
    if (file === pendingFile) {
      setPendingStagedUploadId(stagedUploadId);
    }
    return stagedUploadId;
  }

  // Ask the backend to auto-detect the patient subject at a given time, returning
  // the detected box plus selection metadata (or null if nothing valid found).
  async function fetchSubjectPreviewAt(frameSec: number): Promise<SubjectPreview | null> {
    if (!pendingFile) {
      return null;
    }
    const formData = new FormData();
    const stagedUploadId = await stagedUploadIdForFile(pendingFile);
    if (stagedUploadId) {
      formData.append("stagedUploadId", stagedUploadId);
    } else {
      formData.append("video", pendingFile);
    }
    formData.append("frameSec", frameSec.toFixed(3));
    formData.append("autoInitMode", "smart");
    formData.append("autoSelectStrategy", "patient");
    const promptText = subjectPromptText.trim();
    if (promptText) {
      formData.append("sam3TextPrompts", promptText);
    }
    const res = await apiFetch("/api/subject-preview", { method: "POST", body: formData });
    const body = (await res.json().catch(() => null)) as {
      preview?: {
        fps?: number;
        frame_index?: number;
        frame_sec?: number;
        detection?: { box?: SubjectBox } | null;
        info?: { selected_source?: unknown; num_candidates?: unknown };
      };
      error?: string;
    } | null;
    if (!res.ok || !body?.preview?.detection?.box || !validSubjectBox(body.preview.detection.box)) {
      return null;
    }
    const info = body.preview.info ?? {};
    return {
      box: body.preview.detection.box,
      frameSec: Number(body.preview.frame_sec ?? frameSec),
      frameIndex: Math.max(0, Math.trunc(Number(body.preview.frame_index ?? 0))),
      fps: Math.max(1, Number(body.preview.fps ?? 30)),
      source: typeof info.selected_source === "string" ? info.selected_source : null,
      candidateCount: Math.max(0, Math.trunc(Number(info.num_candidates ?? 0))),
    };
  }

  // Times to probe for the auto-subject preview.
  function pickPreviewSampleTimes(): number[] {
    // Sample evenly across the kept timeline so identity holds on intro,
    // middle, and end of the recording — not just the moment the user
    // happens to be paused on.
    const duration = Math.max(videoDurationSec, 0);
    const lower = Math.max(0, trimStartSec);
    const upper = Math.min(duration > 0 ? duration : Math.max(lower + 0.1, pendingVideoCursorSec + 0.1), effectiveTrimEndSec);
    if (upper <= lower + 0.05) {
      return [pendingVideoCursorSec];
    }
    const n = Math.max(2, AUTO_PREVIEW_SAMPLE_COUNT);
    const inset = (upper - lower) * 0.05;
    const start = lower + inset;
    const end = upper - inset;
    const step = (end - start) / (n - 1);
    const times: number[] = [];
    for (let i = 0; i < n; i += 1) {
      times.push(start + step * i);
    }
    if (!times.some((t) => Math.abs(t - pendingVideoCursorSec) < 0.05)) {
      times.push(pendingVideoCursorSec);
    }
    return times.map((t) => Math.max(0, Math.min(duration > 0 ? duration : t, t)));
  }

  // Run auto-subject detection across several sampled frames and lock the overlay
  // onto the sample nearest the current cursor.
  async function previewAutoSubject(): Promise<void> {
    if (!pendingFile) {
      setSubjectPreviewError("Choose a video first.");
      return;
    }
    setIsPreviewingAutoSubject(true);
    setSubjectPreviewError(null);
    try {
      const sampleTimes = pickPreviewSampleTimes();
      const results: SubjectPreview[] = [];
      for (const time of sampleTimes) {
        try {
          const preview = await fetchSubjectPreviewAt(time);
          if (preview) {
            results.push(preview);
          }
        } catch {
          // Continue sampling — one bad frame shouldn't poison the whole preview run.
        }
      }
      if (results.length === 0) {
        throw new Error("No patient subject detected across the sampled frames.");
      }
      // Sort chronologically and pick the sample closest to the current cursor as
      // the active one (so the UI overlay matches the playhead the user is on).
      results.sort((a, b) => a.frameSec - b.frameSec);
      const active =
        results.reduce<SubjectPreview | null>((best, sample) => {
          if (!best) return sample;
          const bestDist = Math.abs(best.frameSec - pendingVideoCursorSec);
          const sampleDist = Math.abs(sample.frameSec - pendingVideoCursorSec);
          return sampleDist < bestDist ? sample : best;
        }, null) ?? results[0];
      setAutoSubjectPreviewSamples(results);
      setAutoSubjectPreview(active);
    } catch (err) {
      setAutoSubjectPreviewSamples([]);
      setAutoSubjectPreview(null);
      setSubjectPreviewError(String(err));
    } finally {
      setIsPreviewingAutoSubject(false);
    }
  }

  // Build and POST the generation-job request for one file, attaching trim/cut,
  // crop and (optionally) subject-prompt parameters.
  async function createOneJob(file: File, runNameOverride: string, allowSubjectPrompt = true): Promise<CreateJobResponse> {
    const formData = new FormData();
    const stagedUploadId = await stagedUploadIdForFile(file);
    if (stagedUploadId) {
      formData.append("stagedUploadId", stagedUploadId);
      formData.append("videoFileName", file.name);
    } else {
      formData.append("video", file);
    }
    formData.append("runName", runNameOverride);
    formData.append("inferenceTarget", effectiveInferenceTarget);
    formData.append("precision", "float32");
    formData.append("autoInitMode", "smart");
    formData.append("autoSelectStrategy", "patient");
    formData.append("cameraMotionCompensation", "false");
    formData.append("renderPreview", "false");
    const promptText = subjectPromptText.trim();
    if (promptText) {
      formData.append("sam3TextPrompts", promptText);
    }
    if (trimStartSec > 0.01) {
      formData.append("trimStartSec", trimStartSec.toFixed(3));
    }
    if (videoDurationSec > 0 && trimEndSec > 0 && trimEndSec < videoDurationSec - 0.01) {
      formData.append("trimEndSec", trimEndSec.toFixed(3));
    } else if (removedTimelineSegments.length > 0) {
      formData.append("trimEndSec", effectiveTrimEndSec.toFixed(3));
    }
    if (removedTimelineSegments.length > 0) {
      formData.append(
        "removedSegments",
        JSON.stringify(
          removedTimelineSegments.map((segment) => ({
            startSec: segment.startSec,
            endSec: segment.endSec,
          })),
        ),
      );
    }
    const effectiveCropBox = PUBLIC_BASIC_UI ? null : cropBox;
    const cropRaw = cropBoxToRaw(effectiveCropBox);
    if (cropRaw) {
      formData.append("cropBox", cropRaw);
    }
    const subjectPrompt = allowSubjectPrompt ? selectedSubjectPrompt() : null;
    if (subjectPrompt) {
      const promptBBox = subjectBoxToPrompt(subjectPrompt.box, pendingVideoRef.current, effectiveCropBox);
      if (!promptBBox) {
        throw new Error("Draw or preview a patient subject box before running guided detection.");
      }
      formData.append("promptBBox", promptBBox);
      formData.append("promptBBoxFrame", String(subjectPromptFrame(subjectPrompt.frameSec, subjectPrompt.fps)));
    }
    const res = await apiFetch("/api/jobs", { method: "POST", body: formData });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    return (await res.json()) as CreateJobResponse;
  }

  // Submit the single pending video for processing and select its new run.
  async function createJob(): Promise<void> {
    if (!pendingFile) {
      setError("Choose a video first.");
      return;
    }
    const useManualSubject = patientDetectionMode === "manual";
    if (useManualSubject && !hasManualSubject) {
      setError("Draw a patient subject box or switch to auto detection.");
      return;
    }
    if (useManualSubject && !selectedSubjectPrompt()) {
      setError("The patient subject box is outside the kept timeline. Move the cursor outside deleted cuts and draw it again.");
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const body = await createOneJob(pendingFile, runName);
      setSelectedRunId(body.job.runId);
      pendingFileRef.current = null;
      setPendingFile(null);
      setIsPendingVideoPlaying(false);
      setSubjectBoxFrameSec(null);
      setAutoSubjectPreview(null);
      setAutoSubjectPreviewSamples([]);
      setSubjectPreviewError(null);
      setPendingStagedUploadId(null);
      setBatchUploadProgress(null);
      await loadJobs();
      await loadRuns();
    } catch (err) {
      setError(String(err));
    } finally {
      setIsUploading(false);
    }
  }

  // Submit every queued batch video sequentially and select the first run.
  async function createBatchJobs(): Promise<void> {
    if (batchFiles.length === 0) {
      setError("Choose a folder with videos first.");
      return;
    }
    setIsUploading(true);
    setError(null);
    setBatchUploadProgress({ done: 0, total: batchFiles.length });
    try {
      let firstRunId: string | null = null;
      for (let index = 0; index < batchFiles.length; index += 1) {
        const file = batchFiles[index];
        const stem = file.name.replace(/\.[^.]+$/, "") || `video_${index + 1}`;
        const body = await createOneJob(file, `${stem}_processed`, false);
        firstRunId = firstRunId ?? body.job.runId;
        setBatchUploadProgress({ done: index + 1, total: batchFiles.length });
      }
      setSelectedRunId(firstRunId);
      batchFilesRef.current = [];
      setBatchFiles([]);
      await loadJobs();
      await loadRuns();
    } catch (err) {
      setError(String(err));
    } finally {
      setIsUploading(false);
      setBatchUploadProgress(null);
    }
  }

  // Cancel a running/queued job, then refresh the job list.
  async function stopJob(jobId: string): Promise<void> {
    await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" }).catch(() => undefined);
    await loadJobs();
  }

  async function patchJob(jobId: string, action: "pause" | "resume" | "restart"): Promise<GenerationJob | null> {
    const res = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action }),
    }).catch(() => null);
    const json = (await res?.json().catch(() => null)) as { job?: GenerationJob } | null;
    await loadJobs();
    return json?.job ?? null;
  }

  async function pauseOrResumeJob(job: GenerationJob): Promise<void> {
    await patchJob(job.id, job.paused ? "resume" : "pause");
  }

  async function restartJob(job: GenerationJob): Promise<void> {
    const fresh = await patchJob(job.id, "restart");
    if (fresh?.runId) setSelectedRunId(fresh.runId);
  }

  // Delete a processed run after confirmation, clearing it if it was selected.
  async function deleteRun(runId: string): Promise<void> {
    if (!window.confirm(`Delete run "${runId}"?`)) {
      return;
    }
    const res = await apiFetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    if (selectedRunId === runId) {
      setSelectedRunId(null);
      setRunDetail(null);
      setRunLoadState(null);
    }
    await loadRuns();
  }

  // Toggle a kinematics signal in the plot selection (capped at 8 signals).
  function toggleSignal(signalId: string): void {
    setSelectedSignalIds((current) =>
      current.includes(signalId)
        ? current.filter((id) => id !== signalId)
        : [...current, signalId].slice(-8),
    );
  }

  // Toggle a joint in the active-joints set (kept sorted) for plot/3D filtering.
  function toggleActiveJoint(jointIndex: number): void {
    if (!PLOT_JOINT_INDEX_SET.has(jointIndex)) {
      return;
    }
    setActivePlotJointIndices((current) => {
      if (current.includes(jointIndex)) {
        return current.filter((index) => index !== jointIndex);
      }
      return [...current, jointIndex].sort((a, b) => a - b);
    });
  }

  // Toggle browser full-screen on the analysis stage.
  function toggleFullScreen(): void {
    const element = stageRef.current;
    if (!element) {
      return;
    }
    if (document.fullscreenElement === element) {
      void document.exitFullscreen();
    } else {
      void element.requestFullscreen();
    }
  }

  // Drag-and-drop handlers for the viewer root: accept dropped video file(s) and
  // route them to the single- or batch-upload workflow.
  const dragHandlers = {
    onDragOver: (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
    },
    onDrop: (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      // The guided wizard handles its own drops on the upload step, and the
      // embedded viewer is locked to a single run — don't hijack either.
      if (embedded || wizardOpen) return;
      void filesFromDrop(event.dataTransfer).then((files) => {
        const videos = uniqueVideoFiles(files);
        if (PUBLIC_BASIC_UI && videos.length > 1) {
          setError("Drop one video at a time.");
          return;
        }
        if (videos.length > 1) {
          chooseBatchFiles(videos);
        } else {
          chooseFile(videos[0] ?? null);
        }
      });
    },
  };

  return (
    <div className={`viewer-root${embedded ? " is-embedded" : ""}${wizardOpen && !embedded ? " wizard-active" : ""}${sidebarCollapsed && !embedded && !wizardOpen ? " sidebar-collapsed" : ""}`} {...dragHandlers}>
      {!embedded && !wizardOpen && sidebarCollapsed ? (
        <button
          type="button"
          className="sidebar-expand"
          onClick={() => setSidebarCollapsed(false)}
          title="Show sidebar"
          aria-label="Show sidebar"
        >
          »
        </button>
      ) : null}
      {!embedded && !wizardOpen ? (
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">F</div>
          <div className="brand-name">Kinesia</div>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(true)}
            title="Hide sidebar"
            aria-label="Hide sidebar"
          >
            «
          </button>
        </div>

        <button
          type="button"
          onClick={() => {
            wizardDispatch({ type: "reset_file" });
            setWizardOpen(true);
          }}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            background: "linear-gradient(135deg, #0e7490, #38bdf8)",
            color: "#021018",
            border: "none",
            borderRadius: 6,
            fontWeight: 700,
            fontSize: 13,
            padding: "10px 14px",
            cursor: "pointer",
            marginBottom: 14,
            width: "100%",
          }}
        >
          + New video
        </button>

        <div className="panel-section">
          <div className="section-title">Jobs</div>
          <div className="job-list">
            {jobs.length === 0 ? <div className="muted-card">No active job.</div> : null}
            {jobs.slice(0, 8).map((job) => (
              <div
                className={`job-card ${job.runId === selectedRunId ? "active" : ""}`}
                key={job.id}
                role="button"
                tabIndex={0}
                onClick={() => {
                  pendingFileRef.current = null;
                  setPendingFile(null);
                  if (job.runId !== selectedRunId) {
                    setRunLoadState({ runId: job.runId, loaded: 0, total: 1, label: "Run manifest" });
                  }
                  setSelectedRunId(job.runId);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    pendingFileRef.current = null;
                    setPendingFile(null);
                    if (job.runId !== selectedRunId) {
                      setRunLoadState({ runId: job.runId, loaded: 0, total: 1, label: "Run manifest" });
                    }
                    setSelectedRunId(job.runId);
                  }
                }}
              >
                <div className="job-card-top">
                  <strong>{isActiveJob(job) && job.paused ? "Paused" : statusLabel(job.status)}</strong>
                  {isActiveJob(job) ? (
                    <div className="job-actions">
                      <button
                        type="button"
                        className="job-action"
                        title="Restart"
                        aria-label="Restart"
                        onClick={(event) => {
                          event.stopPropagation();
                          void restartJob(job);
                        }}
                      >
                        <ActionIcon name="restart" />
                      </button>
                      <button
                        type="button"
                        className="job-action"
                        title={job.paused ? "Resume" : "Pause"}
                        aria-label={job.paused ? "Resume" : "Pause"}
                        onClick={(event) => {
                          event.stopPropagation();
                          void pauseOrResumeJob(job);
                        }}
                      >
                        <ActionIcon name={job.paused ? "play" : "pause"} />
                      </button>
                      <button
                        type="button"
                        className="job-action is-stop"
                        title="Stop"
                        aria-label="Stop"
                        onClick={(event) => {
                          event.stopPropagation();
                          void stopJob(job.id);
                        }}
                      >
                        <ActionIcon name="stop" />
                      </button>
                    </div>
                  ) : null}
                </div>
                <span>{job.runId}</span>
                <small>{jobFrameLabel(job)}</small>
                <div className={`job-progress ${isActiveJob(job) && job.progressPercent === null ? "indeterminate" : ""}`}>
                  <span style={{ width: `${jobProgressPercent(job)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel-section run-list-section">
          <div className="section-title">Processed videos</div>
          <div className="run-list">
            {processedRuns.length === 0 ? <div className="muted-card">No recovered runs found.</div> : null}
            {processedRuns.map((run) => (
              <div key={run.id} className={`run-item ${run.id === selectedRunId ? "active" : ""} ${run.id === selectedRunId && activeRunLoadState ? "loading" : ""}`}>
                <button
                  className="run-select"
                  type="button"
                  onClick={() => {
                    if (run.id !== selectedRunId) {
                      setRunLoadState({ runId: run.id, loaded: 0, total: 1, label: "Run manifest" });
                    }
                    setSelectedRunId(run.id);
                    pendingFileRef.current = null;
                    setPendingFile(null);
                  }}
                >
                  <span>{run.id}</span>
                  <small>
                    {run.id === selectedRunId && activeRunLoadState
                      ? loadProgressLabel(activeRunLoadState)
                      : `${run.processedFrames} frames${
                          runDurationLabel(run.processedFrames, run.fps) ? ` · ${runDurationLabel(run.processedFrames, run.fps)}` : ""
                        }`}
                  </small>
                </button>
                {!PUBLIC_BASIC_UI ? (
                  <button
                    className="run-delete"
                    type="button"
                    title="Delete run"
                    aria-label="Delete run"
                    onClick={() => void deleteRun(run.id).catch((err) => setError(String(err)))}
                  >
                    <ActionIcon name="minus" />
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      </aside>
      ) : null}

      {wizardOpen ? (
        <div className="kinesia-process is-in-viewer">
          <WizardPanel
            onClose={() => setWizardOpen(false)}
            actions={{
              onViewResults: (runId: string) => {
                setSelectedRunId(runId);
                setWizardOpen(false);
              },
            }}
          />
        </div>
      ) : (
      <main className="main">
        <div className="topbar">
          <div className="topbar-title">
            {activeRunLoadState ? <span>{loadProgressLabel(activeRunLoadState)}</span> : null}
            {activeJobForRun ? (
              <span className="live-pill">
                Processing {activeJobForRun.processedFrames}
                {activeJobForRun.totalFrames ? `/${activeJobForRun.totalFrames}` : ""}
                {activeJobForRun.progressPercent !== null ? ` · ${Math.round(activeJobForRun.progressPercent)}%` : ""}
              </span>
            ) : null}
          </div>
        </div>

        <section ref={stageRef} className={`analysis-stage ${isUploadEditorOpen ? "upload-mode" : ""}`}>
          {showViewerControls ? (
              <div className="timeline-row">
                <input
                  className="slider"
                  type="range"
                  min={0}
                  max={Math.max(0, frameCount - 1)}
                  value={safeFrameIndex}
                  disabled={!runDetail || frameCount === 0}
                  onPointerDown={() => setIsPlaying(false)}
                  onKeyDown={() => setIsPlaying(false)}
                  onChange={(event) => {
                    const nextFrame = Number(event.target.value);
                    setIsPlaying(false);
                    setFrameIndex(nextFrame);
                    setFrameCursor(nextFrame);
                  }}
                />
                <span>
                  f {frameCount ? safeFrameIndex + 1 : 0}/{frameCount}
                  {runDetail ? ` · ${runDetail.fps.toFixed(1)} fps` : ""}
                </span>
              </div>
          ) : null}

          <div ref={mediaSplitRef} className={`split-region${showViewerControls ? " has-panel" : ""}`}>
          <div className="stage-viewer" style={showViewerControls ? { flexBasis: `${mediaSplitPct}%` } : undefined}>
            {runDetail && frameCount > 0 ? (
              <div
                className="media-row"
                ref={mediaRowRef}
                style={{ gridTemplateColumns: `${mediaColPct}% 10px minmax(0, 1fr)` }}
              >
                <div className="media-pane media-pane-video">
                  <div className="media-view-picker" role="group" aria-label="Left view">
                    <button type="button" className={leftView === "video" ? "active" : ""} onClick={() => setLeftView("video")}>
                      Video
                    </button>
                    <button type="button" className={leftView === "box" ? "active" : ""} onClick={() => setLeftView("box")}>
                      Tracking box
                    </button>
                  </div>
                  {runDetail.previewVideoUrl ? (
                    <video
                      key={runDetail.id}
                      ref={videoRef}
                      src={apiUrl(runDetail.previewVideoUrl) ?? undefined}
                      muted
                      playsInline
                      preload="auto"
                      controls={false}
                      disablePictureInPicture
                    />
                  ) : (
                    <div className="media-pane-empty">No preview video for this run.</div>
                  )}
                  {leftView === "box" ? (
                    <VideoTrackingOverlay
                      frame={runDetail.frames[safeFrameIndex] ?? null}
                      videoWidth={runDetail.videoWidth}
                      videoHeight={runDetail.videoHeight}
                    />
                  ) : null}
                </div>
                <div
                  className="media-col-divider"
                  role="separator"
                  aria-orientation="vertical"
                  aria-label="Resize video and 3D panes"
                  onPointerDown={onColDividerDown}
                  onPointerMove={onColDividerMove}
                  onPointerUp={onColDividerUp}
                  onPointerCancel={onColDividerUp}
                />
                <div className="media-pane media-pane-three">
                  <div className="three-host">
                    <span className="media-pane-label">3D reconstruction</span>
                    <ThreeSpaceViewer
                      key={runDetail.id}
                      runDetail={runDetail}
                      frameIndex={safeFrameIndex}
                      frameCursor={safeFrameCursor}
                      uprightMode
                      showMesh={showMesh && runDetail.hasMeshes}
                      showJoints={showJoints}
                      showBones={showBones}
                      meshOpacity={meshOpacityPercent / 100}
                      selectedJointIndices={activePlotJointIndices}
                      onJointPick={toggleActiveJoint}
                    />
                  </div>
                  <div className="fog-under-three">
                    <FogProbabilityPanel
                      runDetail={runDetail}
                      frameIndex={safeFrameIndex}
                      threshold={activeFogThreshold}
                      cleanupMinDurationSec={fogCleanupMinDurationSec}
                      cleanupMergeGapSec={fogCleanupMergeGapSec}
                      settingsOpen={fogSettingsOpen}
                      onCleanupMinDurationChange={setFogCleanupMinDurationSec}
                      onCleanupMergeGapChange={setFogCleanupMergeGapSec}
                      onSettingsOpenChange={setFogSettingsOpen}
                      onThresholdChange={updateFogThreshold}
                      scoreSignalId={fogScoreSignalId}
                      onScoreSignalChange={setFogScoreSignalId}
                      onFrameSelect={(index) => {
                        setIsPlaying(false);
                        setFrameIndex(index);
                        setFrameCursor(index);
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : activeRunLoadState && !activeJobForRun ? (
              <RunLoadingState state={activeRunLoadState} />
            ) : activeJobForRun ? (
              <ProcessingState job={activeJobForRun} />
            ) : selectedJob?.status === "failed" ? (
              <ProcessingState job={selectedJob} />
            ) : pendingFileUrl ? (
              <div className="pending-preview">
                <div className="patient-workflow">
                  {!PUBLIC_BASIC_UI ? (
                    <div className="workflow-steps">
                      <span className="done">1 Video</span>
                      <span className={patientDetectionMode === "auto" || hasManualSubject ? "done" : "active"}>2 Patient</span>
                      <span className={patientDetectionMode === "auto" || hasManualSubject ? "active" : ""}>3 Run</span>
                    </div>
                  ) : null}
                  <div className="patient-mode-row">
                    <strong>
                      {patientDetectionMode === "auto"
                        ? currentAutoSubjectPreview
                          ? "Auto subject locked"
                          : "Auto detection"
                        : hasManualSubject
                          ? "Patient locked"
                          : "Draw patient box"}
                    </strong>
                    {patientDetectionMode === "auto" ? (
                      <button className="button secondary" type="button" disabled={isPreviewingAutoSubject} onClick={() => void previewAutoSubject()}>
                        {isPreviewingAutoSubject ? "Detecting..." : currentAutoSubjectPreview ? "Refresh auto" : "Preview auto"}
                      </button>
                    ) : null}
                  </div>
                  <label className="subject-prompt-row">
                    <span className="subject-prompt-label">Subject prompt</span>
                    <input
                      type="text"
                      className="subject-prompt-input"
                      value={subjectPromptText}
                      placeholder={DEFAULT_SUBJECT_PROMPT}
                      maxLength={120}
                      onChange={(event) => setSubjectPromptText(event.target.value)}
                      spellCheck={false}
                    />
                    <span className="subject-prompt-hint">
                      Natural language. SAM3 will lock onto the person matching this description (e.g. &quot;the patient&quot;, &quot;the guy with the blue shirt&quot;).
                    </span>
                  </label>
                  {patientDetectionMode === "auto" && autoSubjectPreviewSamples.length > 0 ? (
                    <div className="subject-prompt-samples">
                      <span className="subject-prompt-samples-label">Checked at</span>
                      {autoSubjectPreviewSamples.map((sample) => {
                        const isActive = autoSubjectPreview?.frameSec === sample.frameSec;
                        return (
                          <button
                            key={`${sample.frameSec}-${sample.frameIndex}`}
                            type="button"
                            className={`subject-prompt-sample ${isActive ? "active" : ""}`}
                            onClick={() => {
                              setAutoSubjectPreview(sample);
                              if (pendingVideoRef.current) {
                                try {
                                  pendingVideoRef.current.currentTime = sample.frameSec;
                                } catch {
                                  // ignore: video may not be ready yet
                                }
                              }
                            }}
                          >
                            {sample.frameSec.toFixed(1)}s
                          </button>
                        );
                      })}
                    </div>
                  ) : null}
                </div>
                {!PUBLIC_BASIC_UI ? (
                  <div className="video-editor-toolbar">
                    <div className="segmented-control">
                      {patientDetectionMode === "manual" ? (
                        <button
                          className={videoEditTool === "subject" ? "active" : ""}
                          type="button"
                          onClick={() => setVideoEditTool("subject")}
                        >
                          Subject
                        </button>
                      ) : null}
                      <button
                        className={videoEditTool === "crop" ? "active" : ""}
                        type="button"
                        onClick={() => setVideoEditTool("crop")}
                      >
                        Crop
                      </button>
                    </div>
                    <button className="button secondary" type="button" onClick={resetVideoEdit}>
                      Reset edit
                    </button>
                  </div>
                ) : null}
                <div className="subject-picker">
                  <video
                    ref={pendingVideoRef}
                    src={pendingFileUrl}
                    controls={false}
                    preload="metadata"
                    onLoadedMetadata={onPendingVideoMetadata}
                    onTimeUpdate={onPendingVideoTimeUpdate}
                    onPlay={() => setIsPendingVideoPlaying(true)}
                    onPause={() => setIsPendingVideoPlaying(false)}
                    onEnded={() => setIsPendingVideoPlaying(false)}
                  />
                  <div
                    className={`subject-overlay ${patientDetectionMode === "manual" ? "is-active" : ""}`}
                    onPointerDown={startSubjectSelection}
                    onPointerMove={updateSubjectSelection}
                    onPointerUp={endSubjectSelection}
                    onPointerCancel={endSubjectSelection}
                  >
                    {patientDetectionMode === "manual" && subjectBox ? (
                      <div
                        className="subject-box"
                        style={{
                          left: `${subjectBox.x * 100}%`,
                          top: `${subjectBox.y * 100}%`,
                          width: `${subjectBox.width * 100}%`,
                          height: `${subjectBox.height * 100}%`,
                        }}
                      />
                    ) : null}
                    {patientDetectionMode === "auto" && currentAutoSubjectPreview ? (
                      <div
                        className="subject-box auto-subject-box"
                        style={{
                          left: `${currentAutoSubjectPreview.box.x * 100}%`,
                          top: `${currentAutoSubjectPreview.box.y * 100}%`,
                          width: `${currentAutoSubjectPreview.box.width * 100}%`,
                          height: `${currentAutoSubjectPreview.box.height * 100}%`,
                        }}
                      />
                    ) : null}
                    {!PUBLIC_BASIC_UI && cropBox ? (
                      <div
                        className="crop-box"
                        style={{
                          left: `${cropBox.x * 100}%`,
                          top: `${cropBox.y * 100}%`,
                          width: `${cropBox.width * 100}%`,
                          height: `${cropBox.height * 100}%`,
                        }}
                      />
                    ) : null}
                  </div>
                </div>
                <div className="video-trim-panel">
                  <div className="trim-header">
                    <strong>{fmtTrimTime(timelineCursorOffsetSec)}</strong>
                    <span>/ {fmtTrimTime(editedVideoDurationSec)}</span>
                  </div>
                  <div
                    className="trim-scrubber"
                    role="slider"
                    tabIndex={0}
                    aria-label="Video timeline"
                    aria-valuemin={0}
                    aria-valuemax={Number(timelineVisibleDurationSec.toFixed(2))}
                    aria-valuenow={Number(timelineCursorOffsetSec.toFixed(2))}
                    onKeyDown={onTrimTimelineKeyDown}
                    onPointerDown={startTimelineScrub}
                    onPointerMove={updateTimelineScrub}
                  >
                    <div className="trim-track">
                      <div
                        className="trim-selection"
                        style={{
                          left: 0,
                          width: timelineVisibleDurationSec > 0 ? "100%" : 0,
                        }}
                      />
                      {visibleTimelineSegments.map((segment) => {
                        const left = (segment.displayStartSec / Math.max(timelineVisibleDurationSec, 0.001)) * 100;
                        const width = ((segment.displayEndSec - segment.displayStartSec) / Math.max(timelineVisibleDurationSec, 0.001)) * 100;
                        return (
                          <button
                            key={segment.id}
                            className={`timeline-segment ${segment.id === selectedTimelineSegmentId ? "selected" : ""}`}
                            type="button"
                            style={{ left: `${left}%`, width: `${Math.max(0.8, width)}%` }}
                            onPointerDown={(event) => {
                              event.stopPropagation();
                              setSelectedTimelineSegmentId(segment.id);
                              seekPendingVideoSec(segment.startSec);
                            }}
                            onClick={(event) => {
                              event.stopPropagation();
                              setSelectedTimelineSegmentId(segment.id);
                            }}
                            aria-label={`Segment ${segment.index}: ${fmtTrimTime(segment.startSec)} to ${fmtTrimTime(segment.endSec)}`}
                          >
                            <span>{segment.index}</span>
                          </button>
                        );
                      })}
                      {normalizedTimelineCutPoints.map((point) => {
                        const offset = timelineOffsetFromOriginalTime(visibleTimelineSegments, point);
                        return (
                          <div
                            key={point.toFixed(3)}
                            className="cut-boundary"
                            style={{ left: `${(offset / Math.max(timelineVisibleDurationSec, 0.001)) * 100}%` }}
                          />
                        );
                      })}
                      <div className="trim-boundary start" style={{ left: "0%" }} />
                      <div className="trim-boundary end" style={{ left: "100%" }} />
                      <div className="trim-playhead" style={{ left: `${trimCursorPercent}%` }} />
                    </div>
                  </div>
                  <div className="trim-actions">
                    <button className="button" type="button" disabled={videoDurationSec <= 0} onClick={togglePendingVideoPlayback}>
                      {isPendingVideoPlaying ? "Pause" : "Play"}
                    </button>
                    <button className="button secondary" type="button" disabled={videoDurationSec <= 0} onClick={addCutAtCursor}>
                      Cut
                    </button>
                    <button
                      className="button danger"
                      type="button"
                      disabled={!canDeleteSelectedTimelineSegment}
                      onClick={deleteSelectedTimelineSegment}
                    >
                      Delete segment
                    </button>
                    <button className="button secondary" type="button" disabled={videoDurationSec <= 0} onClick={resetTimelineTrim}>
                      Reset
                    </button>
                  </div>
                  <div className="trim-summary">
                    <span>
                      {fmtTrimTime(editedVideoDurationSec)}
                      {visibleTimelineSegments.length > 0 ? ` · ${visibleTimelineSegments.length} segment${visibleTimelineSegments.length === 1 ? "" : "s"}` : ""}
                    </span>
                  </div>
                </div>
                {!PUBLIC_BASIC_UI || patientDetectionMode === "manual" ? (
                  <div className="subject-picker-controls">
                    {!PUBLIC_BASIC_UI ? (
                      <label className="inline-control">
                        Target frame
                        <input
                          type="number"
                          min={0}
                          step={1}
                          value={subjectFrame}
                          onChange={(event) => seekPendingVideo(Number(event.target.value) || 0)}
                        />
                      </label>
                    ) : null}
                    {patientDetectionMode === "manual" ? (
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => {
                          setSubjectBox(null);
                          setSubjectBoxFrameSec(null);
                        }}
                      >
                        Clear subject
                      </button>
                    ) : null}
                    {!PUBLIC_BASIC_UI ? (
                      <>
                        <button className="button secondary" type="button" onClick={() => setCropBox(null)}>
                          Clear crop
                        </button>
                        <span>{patientDetectionMode === "manual" ? "Manual patient lock" : "Auto patient selection"}</span>
                      </>
                    ) : null}
                  </div>
                ) : null}
                {subjectPreviewError ? <div className="run-gate">{subjectPreviewError}</div> : null}
                {patientDetectionMode === "auto" && currentAutoSubjectPreview ? (
                  <div className="run-gate">
                    {currentAutoSubjectPreview.source ?? "detector"} · {currentAutoSubjectPreview.candidateCount} candidate
                    {currentAutoSubjectPreview.candidateCount === 1 ? "" : "s"}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="empty-state">Choose a run or drop a video.</div>
            )}
          </div>

          {showViewerControls ? (
            <div className="stage-toolbar below">
              <button
                className="icon-btn"
                type="button"
                onClick={toggleRunPlayback}
                disabled={!runDetail || frameCount <= 1}
                title={isPlaying ? "Pause" : "Play"}
                aria-label={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? (
                  <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M7 5h3.2v14H7zM13.8 5H17v14h-3.2z" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M8 5v14l11-7z" /></svg>
                )}
              </button>
              {!PUBLIC_BASIC_UI ? (
                <label className="inline-control icon-control" title="Playback speed">
                  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M4 6l7 6-7 6zM13 6l7 6-7 6z" /></svg>
                  <input
                    type="number"
                    min={5}
                    max={400}
                    step={5}
                    value={playbackSpeedPercent}
                    onChange={(event) => setPlaybackSpeedPercent(clamp(Number(event.target.value) || 100, 5, 400))}
                  />
                  <span className="unit">%</span>
                </label>
              ) : null}
              <button
                className={`icon-btn ${showMesh && runDetail?.hasMeshes ? "active" : ""}`}
                type="button"
                onClick={() => setShowMesh((value) => !value)}
                disabled={Boolean(runDetail && !runDetail.hasMeshes)}
                title={runDetail && !runDetail.hasMeshes ? "No mesh for this run" : "Mesh"}
                aria-label="Mesh"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" d="M12 3l8 4.6v8.8L12 21l-8-4.6V7.6zM12 3v18M4 7.6l8 4.6 8-4.6" /></svg>
              </button>
              <button
                className={`icon-btn ${showJoints ? "active" : ""}`}
                type="button"
                onClick={() => setShowJoints((value) => !value)}
                title="Joints"
                aria-label="Joints"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><g fill="currentColor"><circle cx="6" cy="7" r="2" /><circle cx="17.5" cy="6.5" r="2" /><circle cx="13" cy="13" r="2" /><circle cx="8" cy="18" r="2" /></g></svg>
              </button>
              <button
                className={`icon-btn ${showBones ? "active" : ""}`}
                type="button"
                onClick={() => setShowBones((value) => !value)}
                title="Bones"
                aria-label="Bones"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><g stroke="currentColor" strokeWidth="1.7" fill="none" strokeLinecap="round"><circle cx="6.5" cy="7" r="1.7" /><circle cx="17.5" cy="17" r="1.7" /><line x1="7.7" y1="8.3" x2="16.3" y2="15.7" /></g></svg>
              </button>
              <label className="inline-control icon-control" title="Mesh opacity">
                <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="1.7" /><path fill="currentColor" d="M12 4a8 8 0 0 1 0 16z" /></svg>
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={2}
                  value={meshOpacityPercent}
                  onChange={(event) => setMeshOpacityPercent(clamp(Number(event.target.value) || 0, 0, 100))}
                  disabled={Boolean(runDetail && (!runDetail.hasMeshes || !showMesh))}
                />
                <span className="unit">%</span>
              </label>
              {!PUBLIC_BASIC_UI ? (
                <button
                  className="icon-btn"
                  type="button"
                  onClick={toggleFullScreen}
                  title={isFullScreen ? "Exit fullscreen" : "Fullscreen"}
                  aria-label={isFullScreen ? "Exit fullscreen" : "Fullscreen"}
                >
                  {isFullScreen ? (
                    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" /></svg>
                  ) : (
                    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>
                  )}
                </button>
              ) : null}
            </div>
          ) : null}

          {showViewerControls ? (
            <div
              className="stage-divider"
              role="separator"
              aria-orientation="horizontal"
              aria-label="Resize media and analysis panels"
              onPointerDown={onSplitDividerDown}
              onPointerMove={onSplitDividerMove}
              onPointerUp={onSplitDividerUp}
              onPointerCancel={onSplitDividerUp}
            />
          ) : null}

          {showViewerControls ? (
          <div className="kinematics-panel">
            <div className="plot-settings-bar">
              <button
                type="button"
                className={`plot-settings-toggle ${plotSettingsOpen ? "active" : ""}`}
                onClick={() => setPlotSettingsOpen((value) => !value)}
                title="Choose joints and signals to plot"
              >
                ⚙ Signals & joints
              </button>
              <span className="plot-settings-summary">
                {selectedSignals.length} signal{selectedSignals.length > 1 ? "s" : ""} · {signalGroup(plotGroup).label}
                {" · "}
                {activePlotJointIndices.length}/{PLOT_JOINT_OPTIONS.length} joints
              </span>
            </div>
            {plotSettingsOpen ? (
            <>
            <div className="active-joints-panel">
              <div className="active-joints-head">
                <div>
                  <strong>Active joints</strong>
                  <span>{activePlotJointIndices.length}/{PLOT_JOINT_OPTIONS.length}</span>
                </div>
                <div className="active-joints-actions">
                  <button type="button" onClick={() => setActivePlotJointIndices(PLOT_JOINT_OPTIONS.map((joint) => joint.index))}>
                    All
                  </button>
                  <button type="button" onClick={() => setActivePlotJointIndices(DEFAULT_ACTIVE_PLOT_JOINTS)}>
                    Default
                  </button>
                  <button type="button" onClick={() => setActivePlotJointIndices([])}>
                    Clear
                  </button>
                </div>
              </div>
              <div className="active-joint-grid">
                {PLOT_JOINT_OPTIONS.map((joint) => {
                  const isActive = activePlotJointIndices.includes(joint.index);
                  return (
                    <button
                      key={joint.index}
                      className={isActive ? "active" : ""}
                      type="button"
                      onClick={() => toggleActiveJoint(joint.index)}
                    >
                      <span className="joint-dot" />
                      {joint.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="signal-picker-shell">
              <div className="signal-picker-head">
                <div>
                  <strong>{selectedSignals.length} signal{selectedSignals.length > 1 ? "s" : ""}</strong>
                  <span>{signalGroup(plotGroup).label}</span>
                </div>
                <div className="signal-picker-actions">
                  <div className="plot-layout-toggle" role="group" aria-label="Plot layout">
                    <button
                      type="button"
                      className={plotLayoutMode === "stacked" ? "active" : ""}
                      onClick={() => setPlotLayoutMode("stacked")}
                      title="One auto-scaled strip per signal"
                    >
                      Stacked
                    </button>
                    <button
                      type="button"
                      className={plotLayoutMode === "overlay" ? "active" : ""}
                      onClick={() => setPlotLayoutMode("overlay")}
                      title="All signals overlaid, one axis per unit"
                    >
                      Overlay
                    </button>
                  </div>
                  <button type="button" onClick={() => setSignalPickerOpen((open) => !open)}>
                    {signalPickerOpen ? "Hide signals" : "Choose signals"}
                  </button>
                </div>
              </div>
              <div className="selected-signal-strip">
                {selectedSignals.length === 0 ? <span>No signal selected</span> : null}
                {selectedSignals.map((signal, index) => (
                  <button key={signal.id} type="button" onClick={() => toggleSignal(signal.id)}>
                    <span className="chip-swatch" style={{ background: PLOT_COLORS[index % PLOT_COLORS.length] }} />
                    {signal.label}
                  </button>
                ))}
              </div>
              {signalPickerOpen ? (
                <>
                  <div className="kinematic-tabs">
                    {GROUPS.map((group) => (
                      <button
                        key={group.id}
                        className={plotGroup === group.id ? "active" : ""}
                        type="button"
                        onClick={() => setPlotGroup(group.id)}
                      >
                        {group.label}
                      </button>
                    ))}
                  </div>
                  <div className="signal-grid">
                    {availableSignals.length === 0 ? <div className="plot-empty">No same-unit signal available.</div> : null}
                    {availableSignals.map((signal) => {
                      const selectedIndex = selectedSignals.findIndex((item) => item.id === signal.id);
                      const isSelected = selectedIndex >= 0;
                      const color = PLOT_COLORS[(selectedIndex >= 0 ? selectedIndex : 0) % PLOT_COLORS.length];
                      const current = signal.values[safeFrameIndex] ?? null;
                      return (
                        <button
                          key={signal.id}
                          className={`signal-chip ${isSelected ? "selected" : ""}`}
                          type="button"
                          onClick={() => toggleSignal(signal.id)}
                        >
                          <span className="chip-swatch" style={{ background: isSelected ? color : "transparent" }} />
                          <span>{signal.label}</span>
                          {isSelected ? <strong>{fmtValue(current, signal.unit)}</strong> : null}
                        </button>
                      );
                    })}
                  </div>
                </>
              ) : null}
            </div>
            </>
            ) : null}
            {runDetail && frameCount > 1 ? (
              <TimelineNavigator
                frameCount={frameCount}
                frameIndex={safeFrameIndex}
                windowStart={annotationWindow?.start ?? 0}
                windowSpan={annotationWindow ? annotationWindow.end - annotationWindow.start + 1 : frameCount}
                onWindowChange={(start, span) => {
                  if (span >= frameCount) {
                    setAnnotationWindowFrames(null);
                    setWindowStartFrame(null);
                  } else {
                    setAnnotationWindowFrames(span);
                    setWindowStartFrame(start);
                  }
                }}
              />
            ) : null}
            {runDetail ? (
              <ManualFogAnnotationBar
                runDetail={runDetail}
                frameIndex={safeFrameIndex}
                segments={manualFogSegments}
                deletedSegments={deletedAutoFogSegments}
                draft={draftFogAnnotation}
                viewWindow={annotationWindow}
                onDraftChange={setDraftFogAnnotation}
                onCommit={commitManualFogAnnotation}
                selectedSegmentIndex={selectedFogSegmentIndex}
                onSelectSegment={setSelectedFogSegmentIndex}
                onUpdateSegment={updateManualFogAnnotation}
                onDeleteSelected={deleteSelectedManualFogAnnotation}
                onSplitSelected={splitSelectedManualFogAnnotation}
                onFrameSelect={(index) => {
                  setIsPlaying(false);
                  setFrameIndex(index);
                  setFrameCursor(index);
                }}
                onClear={clearManualFogAnnotations}
                onExportBundle={exportBundle}
                hasGroundTruth={
                  goldTruthSegments(runDetail).length +
                    (selectedRunId ? importedGtByRun[selectedRunId]?.length ?? 0 : 0) >
                  0
                }
                onUndo={undoFogAnnotation}
                onRedo={redoFogAnnotation}
                canUndo={canUndoFog}
                canRedo={canRedoFog}
                onResetToPrediction={resetManualFogAnnotationsToPrediction}
                predictionCount={cleanedAutoFogSegments.length}
              />
            ) : null}
            {runDetail ? (
              <GoldTruthBar
                runDetail={runDetail}
                frameIndex={safeFrameIndex}
                viewWindow={annotationWindow}
                annotationSegments={manualFogSegments}
                importedSegments={selectedRunId ? importedGtByRun[selectedRunId] : undefined}
                onImport={importGroundTruth}
                onFrameSelect={(index) => {
                  setIsPlaying(false);
                  setFrameIndex(index);
                  setFrameCursor(index);
                }}
              />
            ) : null}
            <KinematicsPlot
              signals={selectedSignals}
              frameIndex={safeFrameIndex}
              fps={runDetail?.fps ?? 30}
              frameCount={frameCount}
              mode={plotLayoutMode}
              viewWindow={annotationWindow}
              annotationSegments={manualFogSegments}
              maskedRanges={maskedFrameRanges}
              colorForId={(_id, index) => PLOT_COLORS[index % PLOT_COLORS.length]}
              onFrameSelect={(index) => {
                setIsPlaying(false);
                setFrameIndex(index);
                setFrameCursor(index);
              }}
            />
          </div>
          ) : null}
          </div>
        </section>

        {error ? <div className="error-toast">{error}</div> : null}
      </main>
      )}
    </div>
  );
}
