import { promises as fs } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

import { discoverRunDatasets } from "./datasets";
import { stabilizeXY } from "./stabilization_xy";
import {
  RUN_MANIFEST_FILE,
  RUN_METADATA_FILE,
  ensureSafeId,
  fileExists,
  legacyOutputRoot,
  projectRoot,
  readJsonIfExists,
  runDir,
  runsRoot,
  usesConfiguredRunsRoot,
} from "./store";
import type { FogSummary, RunDetail, RunFrame, RunSignal, RunSummary } from "./types";

type JsonRecord = Record<string, unknown>;

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"];
const MESH_DIR = "meshes";

// COCO-WholeBody joint indices used to derive gait signals from the pose output.
const JOINT = {
  leftShoulder: 5,
  rightShoulder: 6,
  leftHip: 9,
  rightHip: 10,
  leftKnee: 11,
  rightKnee: 12,
  leftAnkle: 13,
  rightAnkle: 14,
  leftBigToe: 15,
  leftSmallToe: 16,
  leftHeel: 17,
  rightBigToe: 18,
  rightSmallToe: 19,
  rightHeel: 20,
} as const;

// Coerce a JSON value to a finite number, or null when it is missing/NaN/Infinity.
function numericOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// Like numericOrNull but substitutes a caller-provided fallback for invalid values.
function numericOr(value: unknown, fallback: number): number {
  const parsed = numericOrNull(value);
  return parsed ?? fallback;
}

// Parse a JSON array into an (x, y, z) tuple, or null if any axis is non-finite.
function parseTriplet(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) || value.length < 3) {
    return null;
  }
  const out: [number, number, number] = [Number(value[0]), Number(value[1]), Number(value[2])];
  return out.every((axis) => Number.isFinite(axis)) ? out : null;
}

// Parse an array of joint positions; returns null if any joint is malformed (all-or-nothing).
function parseJoints(value: unknown): Array<[number, number, number]> | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const joints: Array<[number, number, number]> = [];
  for (const item of value) {
    const triplet = parseTriplet(item);
    if (!triplet) {
      return null;
    }
    joints.push(triplet);
  }
  return joints.length > 0 ? joints : null;
}

// Map a camera-frame point (x right, y down, z forward) into the viewer world axes.
function camToWorld(point: [number, number, number]): [number, number, number] {
  return [-point[2], point[0], -point[1]];
}

// Coerce a raw values array into a per-frame series of finite numbers (or null gaps).
function signalValue(values: unknown): Array<number | null> {
  if (!Array.isArray(values)) {
    return [];
  }
  return values.map((value) =>
    typeof value === "number" && Number.isFinite(value) ? value : null,
  );
}

// Validate a stored signal and pad/truncate its values to exactly frameCount samples.
function normalizeSignal(raw: unknown, frameCount: number): RunSignal | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const rec = raw as JsonRecord;
  const id = String(rec.id ?? "").trim();
  if (!id) {
    return null;
  }
  const values = signalValue(rec.values);
  while (values.length < frameCount) {
    values.push(null);
  }
  return {
    id,
    label: String(rec.label ?? id),
    unit: String(rec.unit ?? ""),
    description: String(rec.description ?? ""),
    values: values.slice(0, frameCount),
  };
}

// Whether root/name resolves to a directory (false on any stat error, e.g. missing path).
async function isRunDirectory(root: string, name: string): Promise<boolean> {
  return fs.stat(path.join(root, name)).then((stat) => stat.isDirectory()).catch(() => false);
}

// List run-id folders under root, also following symlinks that point at directories.
async function listRunIds(root: string): Promise<string[]> {
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const ids: string[] = [];
  for (const entry of entries) {
    if (entry.isDirectory() || (entry.isSymbolicLink() && await isRunDirectory(root, entry.name))) {
      ids.push(entry.name);
    }
  }
  return ids;
}

// Run ids under the configured/versioned runs root.
async function listVersionedRunIds(): Promise<string[]> {
  return listRunIds(runsRoot());
}

// Run ids under the legacy flat output root (pre-versioning layout).
async function listLegacyRunIds(): Promise<string[]> {
  return listRunIds(legacyOutputRoot());
}

// Whether a directory contains at least one mesh (.ply) file.
async function hasMeshFiles(directory: string): Promise<boolean> {
  const entries = await fs.readdir(directory).catch(() => []);
  return entries.some((name) => name.toLowerCase().endsWith(".ply"));
}

// Resolve the run's frame rate, preferring the manifest then metadata fields.
function getManifestFps(manifest: JsonRecord | null, metadata: JsonRecord | null): number | null {
  return numericOrNull(manifest?.fps) ?? numericOrNull(metadata?.fps_output) ?? numericOrNull(metadata?.fps);
}

// Best-effort processed-frame count across the several keys older/newer runs used.
function getProcessedFrames(manifest: JsonRecord | null, metadata: JsonRecord | null): number {
  return Math.max(
    0,
    Math.trunc(
      numericOrNull(manifest?.processed_frames) ??
        numericOrNull(manifest?.processedFrames) ??
        numericOrNull(manifest?.frame_count) ??
        numericOrNull(metadata?.total_frames_processed) ??
        (Array.isArray(metadata?.records) ? metadata.records.length : 0),
    ),
  );
}

// The most recent analysis id, from the explicit pointer or the last analyses entry.
function latestAnalysisId(manifest: JsonRecord | null): string | null {
  const direct = typeof manifest?.latest_analysis_id === "string" ? manifest.latest_analysis_id : null;
  if (direct) {
    return direct;
  }
  const analyses = Array.isArray(manifest?.analyses) ? manifest.analyses : [];
  const last = analyses.at(-1);
  return last && typeof last === "object" && typeof (last as JsonRecord).analysis_id === "string"
    ? String((last as JsonRecord).analysis_id)
    : null;
}

// Load a run's manifest JSON, or null if it does not exist.
async function readRunManifest(runId: string): Promise<JsonRecord | null> {
  return readJsonIfExists<JsonRecord>(path.join(runDir(runId), RUN_MANIFEST_FILE));
}

// Auto-compute FoG/kinematics analysis for a run that has pose joints but no
// analysis yet (e.g. a run created outside the web job flow, or one whose analyze
// step never ran) so the viewer never shows an all-zero FoG track. Idempotent and
// lock-guarded; a no-op once an analysis exists. Returns true if analysis is present
// afterwards.
export async function ensureRunAnalysis(runIdRaw: string): Promise<boolean> {
  const runId = ensureSafeId(runIdRaw);
  if (latestAnalysisId(await readRunManifest(runId))) return true;

  const meta = await readRunMetadata(runId);
  const records = Array.isArray(meta?.records) ? meta!.records : [];
  const hasJoints = records.some(
    (r) => r && typeof r === "object" && Array.isArray((r as JsonRecord).joints_cam_xyz),
  );
  if (!hasJoints) return false; // nothing to analyze (no pose output)

  const dir = runDir(runId);
  const lockPath = path.join(dir, ".analyze.lock");
  let lock: Awaited<ReturnType<typeof fs.open>> | null = null;
  try {
    lock = await fs.open(lockPath, "wx");
  } catch {
    // Another request is analysing — wait for the manifest to point at a result.
    for (let i = 0; i < 600; i += 1) {
      await new Promise((r) => setTimeout(r, 200));
      if (latestAnalysisId(await readRunManifest(runId))) return true;
    }
    return false;
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        "uv",
        ["run", "sam3d", "analyze", "--run-id", runId, "--preset", "clinical_fog_v1"],
        {
          cwd: projectRoot(),
          env: {
            ...process.env,
            PYTHONPATH: path.join(projectRoot(), "src"),
            HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "1",
            TRANSFORMERS_OFFLINE: process.env.TRANSFORMERS_OFFLINE || "1",
          },
          stdio: ["ignore", "ignore", "pipe"],
        },
      );
      let stderr = "";
      child.stderr?.on("data", (c: Buffer | string) => {
        stderr += String(c);
        if (stderr.length > 8000) stderr = stderr.slice(-8000);
      });
      child.on("error", reject);
      child.on("close", (code) =>
        code === 0 ? resolve() : reject(new Error(`analyze exited ${code}: ${stderr.trim()}`)),
      );
    });
    return Boolean(latestAnalysisId(await readRunManifest(runId)));
  } finally {
    await lock.close().catch(() => undefined);
    await fs.unlink(lockPath).catch(() => undefined);
  }
}

// Load a run's metadata JSON, falling back to the legacy location when not using a configured root.
async function readRunMetadata(runId: string): Promise<JsonRecord | null> {
  const versioned = await readJsonIfExists<JsonRecord>(path.join(runDir(runId), RUN_METADATA_FILE));
  if (versioned) {
    return versioned;
  }
  if (usesConfiguredRunsRoot()) {
    return null;
  }
  return readJsonIfExists<JsonRecord>(path.join(legacyOutputRoot(), runId, RUN_METADATA_FILE));
}

// Resolve the on-disk directory for a run, preferring the versioned layout over legacy.
async function runBaseDir(runId: string): Promise<string> {
  const versioned = runDir(runId);
  const legacy = path.join(legacyOutputRoot(), runId);
  const versionedExists = await isRunDirectory(runsRoot(), runId);
  if (usesConfiguredRunsRoot()) {
    return versioned;
  }
  return versionedExists ? versioned : legacy;
}

// Enumerate all runs (versioned + legacy, de-duplicated) as lightweight summaries for the run list.
export async function listRuns(): Promise<RunSummary[]> {
  const seen = new Set<string>();
  const legacyRunIds = usesConfiguredRunsRoot() ? [] : await listLegacyRunIds();
  const runIds = [...(await listVersionedRunIds()), ...legacyRunIds]
    .filter((id) => {
      if (seen.has(id)) {
        return false;
      }
      seen.add(id);
      return true;
    })
    .sort((a, b) => a.localeCompare(b));

  const summaries: RunSummary[] = [];
  for (const id of runIds) {
    try {
      const manifest = await readRunManifest(id);
      // `run_metadata.json` can be tens of MB because it contains per-frame records.
      // The run list only needs summary fields, which are already in the manifest
      // for current runs. Avoid loading every metadata file on initial page load.
      const metadata = manifest ? null : await readRunMetadata(id);
      if (!manifest && !metadata) {
        continue;
      }
      const baseDir = await runBaseDir(id);
      const stat = await fs.stat(path.join(baseDir, RUN_MANIFEST_FILE)).catch(() =>
        fs.stat(path.join(baseDir, RUN_METADATA_FILE)).catch(() => null),
      );
      const quality = manifest?.quality_summary;
      summaries.push({
        id,
        processedFrames: getProcessedFrames(manifest, metadata),
        hasMeshes: await hasMeshFiles(path.join(baseDir, MESH_DIR)),
        fps: getManifestFps(manifest, metadata),
        updatedAt:
          (typeof manifest?.updated_at === "string" ? manifest.updated_at : null) ??
          (stat ? stat.mtime.toISOString() : null),
        createdAt: typeof manifest?.created_at === "string" ? manifest.created_at : null,
        inferenceTarget:
          manifest?.inference_target === "hand" || metadata?.inference_target === "hand"
            ? "hand"
            : "body",
        latestAnalysisId: latestAnalysisId(manifest),
        qaStatus:
          quality && typeof quality === "object" && typeof (quality as JsonRecord).status === "string"
            ? String((quality as JsonRecord).status)
            : null,
      });
    } catch {
      // A recovered legacy folder can contain partial JSON; skip it instead of breaking the UI.
    }
  }
  return summaries.sort((a, b) => String(b.updatedAt ?? "").localeCompare(String(a.updatedAt ?? "")));
}

// Load a single analysis (frames/signals/qa); defaults to the latest if no id is given.
async function readAnalysis(runId: string, analysisIdRaw: string | null | undefined) {
  const manifest = await readRunManifest(runId);
  const selected = analysisIdRaw ?? latestAnalysisId(manifest);
  if (!selected) {
    return { id: null, frames: null, signals: null, qa: null };
  }
  const analysisId = ensureSafeId(selected);
  const analysisDir = path.join(runDir(runId), "analysis", analysisId);
  return {
    id: analysisId,
    frames: await readJsonIfExists<JsonRecord>(path.join(analysisDir, "frames.json")),
    signals: await readJsonIfExists<JsonRecord>(path.join(analysisDir, "signals.json")),
    qa: await readJsonIfExists<JsonRecord>(path.join(analysisDir, "qa.json")),
  };
}

// Merge an analysis frame with its per-frame metadata record into the viewer's RunFrame shape.
// Tolerates both snake_case and camelCase keys, and treats absent/lost/masked subjects as not present.
function normalizeFrame(
  runId: string,
  rawFrame: JsonRecord,
  index: number,
  metadataRecords: JsonRecord[],
): RunFrame {
  const record = metadataRecords[index] ?? {};
  const inferenceStatus =
    typeof rawFrame.inference_status === "string"
      ? rawFrame.inference_status
      : typeof rawFrame.inferenceStatus === "string"
        ? rawFrame.inferenceStatus
        : typeof record.inference_status === "string"
          ? record.inference_status
          : null;
  const subjectTrackingStatus =
    typeof rawFrame.subject_tracking_status === "string"
      ? rawFrame.subject_tracking_status
      : typeof rawFrame.subjectTrackingStatus === "string"
        ? rawFrame.subjectTrackingStatus
        : typeof record.subject_tracking_status === "string"
          ? record.subject_tracking_status
          : typeof record.identity_lock_status === "string"
            ? record.identity_lock_status
            : null;
  const explicitSubjectPresent =
    typeof rawFrame.subject_present === "boolean"
      ? rawFrame.subject_present
      : typeof rawFrame.subjectPresent === "boolean"
        ? rawFrame.subjectPresent
        : typeof record.subject_present === "boolean"
          ? record.subject_present
          : null;
  const absentStatus =
    inferenceStatus === "subject_not_initialized" ||
    inferenceStatus === "subject_lost" ||
    inferenceStatus === "subject_absent" ||
    inferenceStatus === "no_output" ||
    inferenceStatus === "masked";
  const subjectPresent =
    explicitSubjectPresent ?? (!Boolean(record.identity_is_lost) && !absentStatus);
  const meshPath = subjectPresent ? String(rawFrame.mesh_file ?? rawFrame.meshFile ?? record.mesh_path ?? "") : "";
  const meshFile = meshPath.trim() ? path.basename(meshPath) : null;
  const joints = subjectPresent ? (
    parseJoints(rawFrame.joints_cam) ??
    parseJoints(rawFrame.jointsCam) ??
    parseJoints(record.joints_cam_xyz) ??
    parseJoints(record.joints_space_cam_xyz)
  ) : null;
  const cameraComp =
    parseTriplet(rawFrame.camera_comp) ??
    parseTriplet(rawFrame.cameraComp) ??
    parseTriplet(record.camera_comp_cam_xyz) ??
    [0, 0, 0];
  const rootRaw = parseTriplet(rawFrame.root_world_raw) ?? parseTriplet(rawFrame.rootWorldRaw);
  const rootStab = parseTriplet(rawFrame.root_world_stabilized) ?? parseTriplet(rawFrame.rootWorldStabilized);
  const bboxRaw = record.bbox_xyxy ?? rawFrame.bbox_xyxy ?? rawFrame.bbox;
  // Gate the tracking box on subject presence, like meshFile/joints above —
  // otherwise an absent/lost/masked frame keeps the last subject's box and the
  // "Tracking box" overlay draws it over a frame with no patient.
  const bbox =
    subjectPresent &&
    Array.isArray(bboxRaw) && bboxRaw.length >= 4 && bboxRaw.slice(0, 4).every((v) => typeof v === "number" && Number.isFinite(v))
      ? ([Number(bboxRaw[0]), Number(bboxRaw[1]), Number(bboxRaw[2]), Number(bboxRaw[3])] as [number, number, number, number])
      : null;
  const focalRaw = numericOrNull(record.focal_length) ?? numericOrNull(rawFrame.focal_length);
  const focalLength = focalRaw != null && focalRaw > 0 ? focalRaw : null;
  const contactRaw = rawFrame.foot_contact ?? rawFrame.footContact;
  const footContact =
    contactRaw && typeof contactRaw === "object"
      ? {
          left: Boolean((contactRaw as JsonRecord).left),
          right: Boolean((contactRaw as JsonRecord).right),
          support: normalizeSupport((contactRaw as JsonRecord).support),
        }
      : undefined;

  return {
    index,
    videoFrame: Math.trunc(numericOrNull(rawFrame.video_frame) ?? numericOrNull(record.video_frame) ?? index),
    meshFile,
    meshUrl: meshFile ? `/api/runs/${encodeURIComponent(runId)}/mesh/${encodeURIComponent(meshFile)}` : null,
    subjectPresent,
    inferenceStatus,
    subjectTrackingStatus,
    cameraComp,
    jointsCam: joints,
    bbox,
    focalLength: focalLength ?? undefined,
    rootWorldRaw: rootRaw ?? undefined,
    rootWorldStabilized: rootStab ?? undefined,
    footContact,
    fogDetected: Boolean(rawFrame.fog_detected ?? rawFrame.fogDetected ?? false),
    fogScore: numericOrNull(rawFrame.fog_score) ?? numericOrNull(rawFrame.fogScore),
    fogScoreSmooth: numericOrNull(rawFrame.fog_score_smooth) ?? numericOrNull(rawFrame.fogScoreSmooth),
    fogComponents: normalizeNumberRecord(rawFrame.fog_components ?? rawFrame.fogComponents),
    // Per-frame subject-tracking confidence (gallery appearance match, else stability).
    trackingScore: subjectPresent
      ? numericOrNull(rawFrame.identity_gallery_similarity) ??
        numericOrNull(record.identity_gallery_similarity) ??
        numericOrNull(rawFrame.identity_stability_score) ??
        numericOrNull(record.identity_stability_score)
      : null,
  };
}

// Coerce a flat object of numbers (e.g. FoG components) into finite-or-null values.
function normalizeNumberRecord(value: unknown): Record<string, number | null> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const out: Record<string, number | null> = {};
  for (const [key, raw] of Object.entries(value as JsonRecord)) {
    out[key] = numericOrNull(raw);
  }
  return out;
}

// Constrain a raw foot-support value to the known set, defaulting to "none".
function normalizeSupport(value: unknown): "left" | "right" | "both" | "none" {
  return value === "left" || value === "right" || value === "both" ? value : "none";
}

// Extract the per-frame metadata records array, dropping any non-object entries.
function recordsFromMetadata(metadata: JsonRecord | null): JsonRecord[] {
  return Array.isArray(metadata?.records)
    ? metadata.records.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object")
    : [];
}

// Build viewer frames directly from metadata records when no analysis frames exist.
function framesFromMetadata(runId: string, metadata: JsonRecord | null): RunFrame[] {
  return recordsFromMetadata(metadata).map((record, index) => normalizeFrame(runId, record, index, recordsFromMetadata(metadata)));
}

// Build viewer frames from an analysis payload, falling back to raw metadata frames when empty.
function framesFromAnalysis(runId: string, framesPayload: JsonRecord | null, metadata: JsonRecord | null): RunFrame[] {
  const rawFrames = Array.isArray(framesPayload?.frames)
    ? framesPayload.frames.filter((item): item is JsonRecord => Boolean(item) && typeof item === "object")
    : [];
  if (rawFrames.length === 0) {
    return framesFromMetadata(runId, metadata);
  }
  const records = recordsFromMetadata(metadata);
  return rawFrames.map((frame, index) => normalizeFrame(runId, frame, index, records));
}

// Summarize the manifest's analysis history (id/preset/createdAt/qaStatus) for the run detail.
function analysesFromManifest(manifest: JsonRecord | null): RunDetail["analyses"] {
  const analyses = Array.isArray(manifest?.analyses) ? manifest.analyses : [];
  return analyses
    .filter((item): item is JsonRecord => Boolean(item) && typeof item === "object")
    .map((item) => {
      const qa = item.qa_summary;
      return {
        analysisId: String(item.analysis_id ?? ""),
        preset: String(item.preset ?? ""),
        createdAt: typeof item.created_at === "string" ? item.created_at : null,
        qaStatus: qa && typeof qa === "object" && typeof (qa as JsonRecord).status === "string"
          ? String((qa as JsonRecord).status)
          : null,
      };
    })
    .filter((item) => item.analysisId);
}

// Normalize the stored signals from an analysis payload to the given frame count.
function signalsFromAnalysis(signalsPayload: JsonRecord | null, frameCount: number): RunSignal[] {
  const rawSignals = Array.isArray(signalsPayload?.signals) ? signalsPayload.signals : [];
  return rawSignals
    .map((signal) => normalizeSignal(signal, frameCount))
    .filter((signal): signal is RunSignal => Boolean(signal));
}

// Construct a RunSignal record (thin helper to keep deriveSignals readable).
function makeSignal(
  id: string,
  label: string,
  unit: string,
  description: string,
  values: Array<number | null>,
): RunSignal {
  return { id, label, unit, description, values };
}

// Per-frame world-space position of one joint (null where the joint is missing).
function point(frames: RunFrame[], index: number): Array<[number, number, number] | null> {
  return frames.map((frame) => {
    const joint = frame.jointsCam?.[index] ?? null;
    return joint ? camToWorld(joint) : null;
  });
}

// Per-frame midpoint of two point series (null where either input is missing).
function midpoint(
  a: Array<[number, number, number] | null>,
  b: Array<[number, number, number] | null>,
): Array<[number, number, number] | null> {
  return a.map((value, index) => {
    const other = b[index];
    if (!value || !other) {
      return null;
    }
    return [(value[0] + other[0]) * 0.5, (value[1] + other[1]) * 0.5, (value[2] + other[2]) * 0.5];
  });
}

// Displacement vector from a to b.
function vector(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
}

// Per-frame interior angle (degrees) at `center`, between the rays to a and b.
function angleAt(
  a: Array<[number, number, number] | null>,
  center: Array<[number, number, number] | null>,
  b: Array<[number, number, number] | null>,
): Array<number | null> {
  return center.map((c, index) => {
    const p0 = a[index];
    const p1 = b[index];
    if (!p0 || !c || !p1) {
      return null;
    }
    const v0 = vector(c, p0);
    const v1 = vector(c, p1);
    const n0 = Math.hypot(...v0);
    const n1 = Math.hypot(...v1);
    if (n0 < 1e-9 || n1 < 1e-9) {
      return null;
    }
    const dot = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    const cos = Math.max(-1, Math.min(1, dot / (n0 * n1)));
    return (Math.acos(cos) * 180) / Math.PI;
  });
}

// Per-frame sagittal pitch (degrees) of the start->end segment, from horizontal vs. forward axes.
function pitchDeg(
  start: Array<[number, number, number] | null>,
  end: Array<[number, number, number] | null>,
): Array<number | null> {
  return start.map((value, index) => {
    const other = end[index];
    if (!value || !other) {
      return null;
    }
    const dx = other[0] - value[0];
    const dz = other[2] - value[2];
    return (Math.atan2(dx, dz) * 180) / Math.PI;
  });
}

// Extract one axis (x/y/z) from a point series as a scalar signal.
function component(
  points: Array<[number, number, number] | null>,
  axis: 0 | 1 | 2,
): Array<number | null> {
  return points.map((value) => value?.[axis] ?? null);
}

// Backward finite-difference rate of change per second; first sample and gaps are null.
function derivative(values: Array<number | null>, fps: number): Array<number | null> {
  const out: Array<number | null> = [null];
  const dt = 1 / Math.max(1, fps);
  for (let index = 1; index < values.length; index += 1) {
    const a = values[index - 1];
    const b = values[index];
    out.push(a === null || b === null ? null : (b - a) / dt);
  }
  return out;
}

// Per-frame 3D speed (m/s) from consecutive positions; first sample and gaps are null.
function speed(points: Array<[number, number, number] | null>, fps: number): Array<number | null> {
  const out: Array<number | null> = [null];
  const dt = 1 / Math.max(1, fps);
  for (let index = 1; index < points.length; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    out.push(!a || !b ? null : Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / dt);
  }
  return out;
}

// Per-frame 3D displacement from the first valid position (a drift/excursion measure).
function baselineDistance(points: Array<[number, number, number] | null>): Array<number | null> {
  const first = points.find((value) => value !== null) ?? null;
  return points.map((value) => (!value || !first ? null : Math.hypot(value[0] - first[0], value[1] - first[1], value[2] - first[2])));
}

// Per-frame a - b (null where either is missing); used for left/right asymmetry signals.
function difference(a: Array<number | null>, b: Array<number | null>): Array<number | null> {
  return a.map((value, index) => (value === null || b[index] === null ? null : value - (b[index] as number)));
}

// Compute the full set of kinematic signals (angles, speeds, heights, asymmetries) from pose frames.
function deriveSignals(frames: RunFrame[], fps: number): RunSignal[] {
  if (frames.length === 0) {
    return [];
  }
  const leftShoulder = point(frames, JOINT.leftShoulder);
  const rightShoulder = point(frames, JOINT.rightShoulder);
  const leftHip = point(frames, JOINT.leftHip);
  const rightHip = point(frames, JOINT.rightHip);
  const leftKnee = point(frames, JOINT.leftKnee);
  const rightKnee = point(frames, JOINT.rightKnee);
  const leftAnkle = point(frames, JOINT.leftAnkle);
  const rightAnkle = point(frames, JOINT.rightAnkle);
  const leftToe = midpoint(point(frames, JOINT.leftBigToe), point(frames, JOINT.leftSmallToe));
  const rightToe = midpoint(point(frames, JOINT.rightBigToe), point(frames, JOINT.rightSmallToe));
  const leftHeel = point(frames, JOINT.leftHeel);
  const rightHeel = point(frames, JOINT.rightHeel);
  const pelvis = midpoint(leftHip, rightHip);
  // Rough center of mass: the unweighted mean of all available joints per frame.
  const com = frames.map((frame) => {
    const points = frame.jointsCam?.map(camToWorld) ?? [];
    if (points.length === 0) {
      return null;
    }
    return [
      points.reduce((sum, value) => sum + value[0], 0) / points.length,
      points.reduce((sum, value) => sum + value[1], 0) / points.length,
      points.reduce((sum, value) => sum + value[2], 0) / points.length,
    ] as [number, number, number];
  });

  const leftKneeSpeed = speed(leftKnee, fps);
  const rightKneeSpeed = speed(rightKnee, fps);
  const leftAnkleSpeed = speed(leftAnkle, fps);
  const rightAnkleSpeed = speed(rightAnkle, fps);
  const comSpeed = speed(com, fps);
  const pelvisSpeed = speed(pelvis, fps);

  // Joint angles, computed once so we can also derive their angular rates.
  const hipLAngle = angleAt(leftShoulder, leftHip, leftKnee);
  const hipRAngle = angleAt(rightShoulder, rightHip, rightKnee);
  const kneeLAngle = angleAt(leftHip, leftKnee, leftAnkle);
  const kneeRAngle = angleAt(rightHip, rightKnee, rightAnkle);
  const ankleLAngle = angleAt(leftKnee, leftAnkle, leftToe);
  const ankleRAngle = angleAt(rightKnee, rightAnkle, rightToe);

  return [
    makeSignal("joint.hip.left.angle_deg", "Left Hip Angle", "deg", "Angle shoulder-hip-knee.", hipLAngle),
    makeSignal("joint.hip.right.angle_deg", "Right Hip Angle", "deg", "Angle shoulder-hip-knee.", hipRAngle),
    makeSignal("joint.knee.left.angle_deg", "Left Knee Flexion Angle", "deg", "Angle hip-knee-ankle.", kneeLAngle),
    makeSignal("joint.knee.right.angle_deg", "Right Knee Flexion Angle", "deg", "Angle hip-knee-ankle.", kneeRAngle),
    makeSignal("joint.ankle.left.angle_deg", "Left Ankle Angle", "deg", "Angle knee-ankle-toe.", ankleLAngle),
    makeSignal("joint.ankle.right.angle_deg", "Right Ankle Angle", "deg", "Angle knee-ankle-toe.", ankleRAngle),
    makeSignal("joint.hip.left.angular_velocity_deg", "Left Hip Angular Velocity", "deg/s", "Rate of change of the left hip angle.", derivative(hipLAngle, fps)),
    makeSignal("joint.hip.right.angular_velocity_deg", "Right Hip Angular Velocity", "deg/s", "Rate of change of the right hip angle.", derivative(hipRAngle, fps)),
    makeSignal("joint.knee.left.angular_velocity_deg", "Left Knee Angular Velocity", "deg/s", "Rate of change of the left knee angle.", derivative(kneeLAngle, fps)),
    makeSignal("joint.knee.right.angular_velocity_deg", "Right Knee Angular Velocity", "deg/s", "Rate of change of the right knee angle.", derivative(kneeRAngle, fps)),
    makeSignal("joint.ankle.left.angular_velocity_deg", "Left Ankle Angular Velocity", "deg/s", "Rate of change of the left ankle angle.", derivative(ankleLAngle, fps)),
    makeSignal("joint.ankle.right.angular_velocity_deg", "Right Ankle Angular Velocity", "deg/s", "Rate of change of the right ankle angle.", derivative(ankleRAngle, fps)),
    makeSignal("segment.thigh.left.pitch_deg", "Left Thigh Pitch", "deg", "Sagittal pitch of thigh segment.", pitchDeg(leftHip, leftKnee)),
    makeSignal("segment.thigh.right.pitch_deg", "Right Thigh Pitch", "deg", "Sagittal pitch of thigh segment.", pitchDeg(rightHip, rightKnee)),
    makeSignal("segment.shank.left.pitch_deg", "Left Shank Pitch", "deg", "Sagittal pitch of shank segment.", pitchDeg(leftKnee, leftAnkle)),
    makeSignal("segment.shank.right.pitch_deg", "Right Shank Pitch", "deg", "Sagittal pitch of shank segment.", pitchDeg(rightKnee, rightAnkle)),
    makeSignal("joint.knee.left.vz", "Left Knee Vertical Speed", "m/s", "Vertical speed of left knee.", derivative(component(leftKnee, 2), fps)),
    makeSignal("joint.knee.right.vz", "Right Knee Vertical Speed", "m/s", "Vertical speed of right knee.", derivative(component(rightKnee, 2), fps)),
    makeSignal("joint.ankle.left.vz", "Left Ankle Vertical Speed", "m/s", "Vertical speed of left ankle.", derivative(component(leftAnkle, 2), fps)),
    makeSignal("joint.ankle.right.vz", "Right Ankle Vertical Speed", "m/s", "Vertical speed of right ankle.", derivative(component(rightAnkle, 2), fps)),
    makeSignal("joint.knee.left.speed", "Left Knee Speed", "m/s", "3D speed of left knee.", leftKneeSpeed),
    makeSignal("joint.knee.right.speed", "Right Knee Speed", "m/s", "3D speed of right knee.", rightKneeSpeed),
    makeSignal("joint.ankle.left.speed", "Left Ankle Speed", "m/s", "3D speed of left ankle.", leftAnkleSpeed),
    makeSignal("joint.ankle.right.speed", "Right Ankle Speed", "m/s", "3D speed of right ankle.", rightAnkleSpeed),
    makeSignal("joint.com.speed", "Center of Mass Speed", "m/s", "Approximate center-of-mass speed.", comSpeed),
    makeSignal("joint.knee.left.acceleration", "Left Knee Acceleration", "m/s^2", "Derivative of left knee speed.", derivative(leftKneeSpeed, fps)),
    makeSignal("joint.knee.right.acceleration", "Right Knee Acceleration", "m/s^2", "Derivative of right knee speed.", derivative(rightKneeSpeed, fps)),
    makeSignal("joint.ankle.left.acceleration", "Left Ankle Acceleration", "m/s^2", "Derivative of left ankle speed.", derivative(leftAnkleSpeed, fps)),
    makeSignal("joint.ankle.right.acceleration", "Right Ankle Acceleration", "m/s^2", "Derivative of right ankle speed.", derivative(rightAnkleSpeed, fps)),
    makeSignal("joint.pelvis.speed", "Pelvis Speed", "m/s", "3D speed of the pelvis.", pelvisSpeed),
    makeSignal("joint.pelvis.acceleration", "Pelvis Acceleration", "m/s^2", "Derivative of pelvis speed.", derivative(pelvisSpeed, fps)),
    makeSignal("joint.com.acceleration", "Center of Mass Acceleration", "m/s^2", "Derivative of center-of-mass speed.", derivative(comSpeed, fps)),
    makeSignal("joint.pelvis.x", "Pelvis X", "m", "Pelvis position X.", component(pelvis, 0)),
    makeSignal("joint.pelvis.y", "Pelvis Y", "m", "Pelvis position Y.", component(pelvis, 1)),
    makeSignal("joint.pelvis.z", "Pelvis Z", "m", "Pelvis height.", component(pelvis, 2)),
    makeSignal("joint.knee.left.z", "Left Knee Height", "m", "Left knee height.", component(leftKnee, 2)),
    makeSignal("joint.knee.right.z", "Right Knee Height", "m", "Right knee height.", component(rightKnee, 2)),
    makeSignal("joint.ankle.left.z", "Left Ankle Height", "m", "Left ankle height.", component(leftAnkle, 2)),
    makeSignal("joint.ankle.right.z", "Right Ankle Height", "m", "Right ankle height.", component(rightAnkle, 2)),
    makeSignal("joint.knee.left.distance", "Left Knee Distance", "m", "3D displacement from first valid frame.", baselineDistance(leftKnee)),
    makeSignal("joint.knee.right.distance", "Right Knee Distance", "m", "3D displacement from first valid frame.", baselineDistance(rightKnee)),
    makeSignal("joint.ankle.left.distance", "Left Ankle Distance", "m", "3D displacement from first valid frame.", baselineDistance(leftAnkle)),
    makeSignal("joint.ankle.right.distance", "Right Ankle Distance", "m", "3D displacement from first valid frame.", baselineDistance(rightAnkle)),
    makeSignal("asymmetry.knee.angle_deg", "Knee Angle Asymmetry", "deg", "Left minus right knee angle.", difference(kneeLAngle, kneeRAngle)),
    makeSignal("asymmetry.ankle.speed", "Ankle Speed Asymmetry", "m/s", "Left minus right ankle speed.", difference(leftAnkleSpeed, rightAnkleSpeed)),
    // Upper-body + foot joints (height + 3D speed) so any of them can be plotted,
    // not just the lower-limb gait joints.
    makeSignal("joint.shoulder.left.z", "Left Shoulder Height", "m", "Left shoulder height.", component(leftShoulder, 2)),
    makeSignal("joint.shoulder.right.z", "Right Shoulder Height", "m", "Right shoulder height.", component(rightShoulder, 2)),
    makeSignal("joint.shoulder.left.speed", "Left Shoulder Speed", "m/s", "3D speed of left shoulder.", speed(leftShoulder, fps)),
    makeSignal("joint.shoulder.right.speed", "Right Shoulder Speed", "m/s", "3D speed of right shoulder.", speed(rightShoulder, fps)),
    makeSignal("joint.toe.left.z", "Left Toe Height", "m", "Left toe height.", component(leftToe, 2)),
    makeSignal("joint.toe.right.z", "Right Toe Height", "m", "Right toe height.", component(rightToe, 2)),
    makeSignal("joint.toe.left.speed", "Left Toe Speed", "m/s", "3D speed of left toe.", speed(leftToe, fps)),
    makeSignal("joint.toe.right.speed", "Right Toe Speed", "m/s", "3D speed of right toe.", speed(rightToe, fps)),
    makeSignal("joint.heel.left.z", "Left Heel Height", "m", "Left heel height.", component(leftHeel, 2)),
    makeSignal("joint.heel.right.z", "Right Heel Height", "m", "Right heel height.", component(rightHeel, 2)),
    makeSignal("joint.heel.left.speed", "Left Heel Speed", "m/s", "3D speed of left heel.", speed(leftHeel, fps)),
    makeSignal("joint.heel.right.speed", "Right Heel Speed", "m/s", "3D speed of right heel.", speed(rightHeel, fps)),
  ];
}

// Union of two signal lists keyed by id; primary wins on conflicts (stored over derived).
function mergeSignals(primary: RunSignal[], derived: RunSignal[]): RunSignal[] {
  const map = new Map<string, RunSignal>();
  for (const signal of [...primary, ...derived]) {
    if (!map.has(signal.id)) {
      map.set(signal.id, signal);
    }
  }
  return Array.from(map.values());
}

// Override/append the stabilized-root signals so charts use the same root the viewer displays.
function withDisplayRootSignals(signals: RunSignal[], frames: RunFrame[]): RunSignal[] {
  const replacements = [
    makeSignal("root.stab.x", "Root Stabilized X", "m", "Stabilized root translation X.", frames.map((frame) => frame.rootWorldStabilized?.[0] ?? null)),
    makeSignal("root.stab.y", "Root Stabilized Y", "m", "Stabilized root translation Y.", frames.map((frame) => frame.rootWorldStabilized?.[1] ?? null)),
    makeSignal("root.stab.z", "Root Stabilized Z", "m", "Stabilized root translation Z.", frames.map((frame) => frame.rootWorldStabilized?.[2] ?? null)),
  ];
  const replacementById = new Map(replacements.map((signal) => [signal.id, signal]));
  const seen = new Set<string>();
  const out = signals.map((signal) => {
    const replacement = replacementById.get(signal.id);
    seen.add(signal.id);
    return replacement ?? signal;
  });
  for (const replacement of replacements) {
    if (!seen.has(replacement.id)) {
      out.push(replacement);
    }
  }
  return out;
}

// Aggregate per-frame FoG detections into contiguous episode segments plus overall counts/ratio.
function fogSummary(frames: RunFrame[], signals: RunSignal[]): FogSummary | null {
  const score = signals.find((signal) => signal.id === "fog.score")?.values ?? [];
  const threshold =
    signals.find((signal) => signal.id === "fog.threshold")?.values.find((value) => typeof value === "number") ??
    null;
  const stateSignal = signals.find((signal) => signal.id === "fog.state")?.values ?? [];
  const detected = frames.map((frame, index) =>
    Boolean(frame.fogDetected || (typeof stateSignal[index] === "number" && (stateSignal[index] as number) > 0.5)),
  );
  const detectedFrameCount = detected.filter(Boolean).length;
  const segments: FogSummary["segments"] = [];
  let start: number | null = null;
  // Iterate one past the end so a segment still open at the final frame gets flushed
  // (detected[length] is undefined, taking the closing branch).
  for (let index = 0; index <= detected.length; index += 1) {
    if (detected[index] && start === null) {
      start = index;
    } else if (!detected[index] && start !== null) {
      const end = index - 1;
      const startFrame = frames[start];
      const endFrame = frames[end];
      segments.push({
        startFrameIndex: start,
        endFrameIndex: end,
        startVideoFrame: startFrame?.videoFrame ?? start,
        endVideoFrame: endFrame?.videoFrame ?? end,
        durationSec: Math.max(0, end - start + 1) / Math.max(1, frames.length),
      });
      start = null;
    }
  }
  if (score.length === 0 && threshold === null && detectedFrameCount === 0) {
    return null;
  }
  return {
    threshold: threshold ?? 0,
    detectedFrameCount,
    detectedRatio: frames.length > 0 ? detectedFrameCount / frames.length : 0,
    segments,
  };
}

// Locate the input or preview video by trying manifest/metadata hints, then known filenames,
// then any video in the run dir; preview falls back to the input video when none is found.
async function findVideoPath(baseDir: string, manifest: JsonRecord | null, metadata: JsonRecord | null, kind: "input" | "preview"): Promise<string | null> {
  const candidates: string[] = [];
  if (kind === "input") {
    if (typeof manifest?.source_video === "string") candidates.push(manifest.source_video);
    if (typeof metadata?.video_input === "string") candidates.push(metadata.video_input);
    candidates.push(path.join(baseDir, "input.mp4"));
  } else {
    const artifacts = manifest?.artifacts;
    if (artifacts && typeof artifacts === "object") {
      const preview = (artifacts as JsonRecord).preview_video;
      if (typeof preview === "string") candidates.push(path.isAbsolute(preview) ? preview : path.join(baseDir, preview));
    }
    if (typeof metadata?.output_video === "string") candidates.push(metadata.output_video);
    candidates.push(path.join(baseDir, "preview.mp4"));
    candidates.push(path.join(baseDir, "processed.mp4"));
    candidates.push(path.join(baseDir, `${runIdFromBaseDir(baseDir)}.mp4`));
    candidates.push(path.join(baseDir, `${runIdFromBaseDir(baseDir).replace(/_processed$/, "")}_processed.mp4`));
    candidates.push(path.join(baseDir, "patient_mesh_preview.mp4"));
  }
  for (const candidate of candidates) {
    if (!candidate) continue;
    if (await fileExists(candidate)) {
      return candidate;
    }
  }
  if (kind === "preview") {
    return findVideoPath(baseDir, manifest, metadata, "input");
  }
  const entries = await fs.readdir(baseDir).catch(() => []);
  const found = entries.find((name) => VIDEO_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext)));
  return found ? path.join(baseDir, found) : null;
}

// The run id is the final path segment of its base directory.
function runIdFromBaseDir(baseDir: string): string {
  return path.basename(baseDir);
}

// Assemble the full RunDetail for the viewer: frames, stabilization, derived signals, FoG summary, videos, datasets.
export async function getRunDetail(runIdRaw: string, analysisId?: string | null): Promise<RunDetail> {
  const runId = ensureSafeId(runIdRaw);
  const manifest = await readRunManifest(runId);
  const metadata = await readRunMetadata(runId);
  if (!manifest && !metadata) {
    throw new Error(`Run not found: ${runId}`);
  }
  const baseDir = await runBaseDir(runId);
  const analysis = await readAnalysis(runId, analysisId);
  const fps = numericOr(analysis.frames?.fps, getManifestFps(manifest, metadata) ?? 30);
  const frames = framesFromAnalysis(runId, analysis.frames, metadata);
  const stabilization = stabilizeXY(frames, fps);
  const displayFrames = frames.map((frame, index) => ({
    ...frame,
    rootWorldStabilized: stabilization.rootWorldStabilized[index] ?? frame.rootWorldStabilized,
    footContact: frame.footContact ?? stabilization.footContact[index],
  }));
  const hasMeshes = await hasMeshFiles(path.join(baseDir, MESH_DIR));
  const analysisSignals = signalsFromAnalysis(analysis.signals, frames.length);
  const signals = withDisplayRootSignals(mergeSignals(analysisSignals, deriveSignals(displayFrames, fps)), displayFrames);
  const qa = analysis.qa as RunDetail["qa"] | null;
  const inputVideo = await findVideoPath(baseDir, manifest, metadata, "input");
  const previewVideo = await findVideoPath(baseDir, manifest, metadata, "preview");
  const previewVideoTimebase = previewVideo && inputVideo && previewVideo === inputVideo ? "source" : "processed";
  const detail: RunDetail = {
    id: runId,
    analysisId: analysis.id,
    inferenceTarget:
      manifest?.inference_target === "hand" || metadata?.inference_target === "hand"
        ? "hand"
        : "body",
    processedFrames: getProcessedFrames(manifest, metadata) || displayFrames.length,
    hasMeshes,
    fps,
    spaceView: (metadata?.space_view as RunDetail["spaceView"]) ?? null,
    videoWidth: numericOrNull(metadata?.video_width),
    videoHeight: numericOrNull(metadata?.video_height),
    inputVideoUrl: inputVideo ? `/api/runs/${encodeURIComponent(runId)}/input-video` : null,
    previewVideoUrl: previewVideo ? `/api/runs/${encodeURIComponent(runId)}/preview-video` : null,
    previewVideoTimebase,
    fog: fogSummary(displayFrames, signals),
    signals,
    frames: displayFrames,
    analyses: analysesFromManifest(manifest),
    qa,
    datasets: [],
  };
  detail.datasets = await discoverRunDatasets(runId, displayFrames, fps).catch(() => []);
  return detail;
}

// Pick a video MIME type from the file extension (defaults to mp4).
function contentTypeFor(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".mov") return "video/quicktime";
  if (ext === ".webm") return "video/webm";
  if (ext === ".mkv") return "video/x-matroska";
  return "video/mp4";
}

// Resolve the on-disk path and MIME type for a run's input/preview video (used by the streaming route).
export async function resolveRunVideoFile(runIdRaw: string, kind: "input" | "preview"): Promise<{ filePath: string; contentType: string }> {
  const runId = ensureSafeId(runIdRaw);
  const manifest = await readRunManifest(runId);
  const metadata = await readRunMetadata(runId);
  const baseDir = await runBaseDir(runId);
  const filePath = await findVideoPath(baseDir, manifest, metadata, kind);
  if (!filePath) {
    throw new Error(`No ${kind} video found for run ${runId}`);
  }
  return { filePath, contentType: contentTypeFor(filePath) };
}

// Read a whole mesh .ply file as raw bytes.
export async function readMeshFile(runIdRaw: string, meshFileRaw: string): Promise<Buffer> {
  const runId = ensureSafeId(runIdRaw);
  const meshFile = ensureSafeId(meshFileRaw);
  const meshPath = await resolveMeshPath(runId, meshFile);
  return fs.readFile(meshPath);
}

// Find a mesh file under the run's mesh dir, falling back to absolute paths recorded in metadata.
async function resolveMeshPath(runId: string, meshFile: string): Promise<string> {
  const baseDir = await runBaseDir(runId);
  const direct = path.join(baseDir, MESH_DIR, meshFile);
  if (await fileExists(direct)) {
    return direct;
  }
  const metadata = await readRunMetadata(runId);
  for (const record of recordsFromMetadata(metadata)) {
    const meshPath = typeof record.mesh_path === "string" ? record.mesh_path : "";
    if (path.basename(meshPath) === meshFile && (await fileExists(meshPath))) {
      return meshPath;
    }
  }
  throw new Error(`Mesh not found: ${meshFile}`);
}

// Byte offsets and counts needed to slice a binary PLY into its vertex and face sections.
type PlyLayout = {
  headerBytes: number;
  vertexCount: number;
  faceCount: number;
  vertexBytes: number;
  faceBytesOffset: number;
};

// Parse a binary little-endian PLY header (only xyz-float vertices) into its on-disk layout.
function parseBinaryPlyLayout(data: Buffer): PlyLayout {
  const marker = Buffer.from("end_header\n");
  let headerEnd = data.indexOf(marker);
  let headerBytes = headerEnd >= 0 ? headerEnd + marker.length : -1;
  if (headerBytes < 0) {
    const markerCrLf = Buffer.from("end_header\r\n");
    headerEnd = data.indexOf(markerCrLf);
    headerBytes = headerEnd >= 0 ? headerEnd + markerCrLf.length : -1;
  }
  if (headerBytes < 0) {
    throw new Error("Invalid PLY header: missing end_header");
  }
  const header = data.subarray(0, headerBytes).toString("ascii");
  if (!header.includes("format binary_little_endian 1.0")) {
    throw new Error("Only binary_little_endian PLY meshes are supported by the web mesh stream.");
  }
  const vertexMatch = header.match(/element vertex\s+(\d+)/);
  const faceMatch = header.match(/element face\s+(\d+)/);
  if (!vertexMatch || !faceMatch) {
    throw new Error("Invalid PLY header: missing vertex/face counts");
  }
  const vertexCount = Number(vertexMatch[1]);
  const faceCount = Number(faceMatch[1]);
  if (!Number.isInteger(vertexCount) || vertexCount <= 0 || !Number.isInteger(faceCount) || faceCount <= 0) {
    throw new Error("Invalid PLY header counts");
  }
  const vertexBytes = vertexCount * 3 * 4;
  const faceBytesOffset = headerBytes + vertexBytes;
  if (data.length < faceBytesOffset) {
    throw new Error("Invalid PLY payload: truncated vertices");
  }
  return { headerBytes, vertexCount, faceCount, vertexBytes, faceBytesOffset };
}

// Stream just the raw vertex buffer for one frame's mesh (vertices change every frame).
export async function readMeshVertices(runIdRaw: string, meshFileRaw: string): Promise<{ data: Buffer; vertexCount: number }> {
  const runId = ensureSafeId(runIdRaw);
  const meshFile = ensureSafeId(meshFileRaw);
  const meshPath = await resolveMeshPath(runId, meshFile);
  const data = await fs.readFile(meshPath);
  const layout = parseBinaryPlyLayout(data);
  return {
    data: data.subarray(layout.headerBytes, layout.headerBytes + layout.vertexBytes),
    vertexCount: layout.vertexCount,
  };
}

// Read the shared face topology once (taken from the first frame's mesh) as a flat uint32 index buffer.
// All frames share the same SMPL-X faces, so the viewer fetches them a single time.
export async function readMeshFaces(runIdRaw: string): Promise<{ data: Buffer; vertexCount: number; faceCount: number }> {
  const runId = ensureSafeId(runIdRaw);
  const baseDir = await runBaseDir(runId);
  const meshDir = path.join(baseDir, MESH_DIR);
  const entries = await fs.readdir(meshDir);
  const firstMesh = entries.find((name) => /^frame_\d+\.ply$/.test(name));
  if (!firstMesh) {
    throw new Error(`No mesh files found for run ${runId}`);
  }
  const data = await fs.readFile(path.join(meshDir, firstMesh));
  const layout = parseBinaryPlyLayout(data);
  const out = Buffer.alloc(layout.faceCount * 3 * 4);
  let sourceOffset = layout.faceBytesOffset;
  let targetOffset = 0;
  for (let face = 0; face < layout.faceCount; face += 1) {
    const count = data.readUInt8(sourceOffset);
    sourceOffset += 1;
    if (count !== 3) {
      throw new Error(`Only triangular PLY faces are supported; found face with ${count} vertices`);
    }
    for (let cursor = 0; cursor < 3; cursor += 1) {
      const index = data.readInt32LE(sourceOffset);
      sourceOffset += 4;
      out.writeUInt32LE(index, targetOffset);
      targetOffset += 4;
    }
  }
  return { data: out, vertexCount: layout.vertexCount, faceCount: layout.faceCount };
}

// Run the Python `sam3d analyze` CLI for a run with the given FoG preset/parameters, then return the refreshed detail.
export async function createRunAnalysis(runIdRaw: string, body: {
  preset?: string;
  sensitivityPercent?: number;
  minDurationMs?: number;
  gapFillMs?: number;
}): Promise<RunDetail> {
  const runId = ensureSafeId(runIdRaw);
  const args = [
    "run",
    "sam3d",
    "analyze",
    "--run-id",
    runId,
    "--preset",
    String(body.preset || "clinical_fog_v1"),
    "--sensitivity-percent",
    String(Math.trunc(body.sensitivityPercent ?? 0)),
    "--min-duration-ms",
    String(Math.trunc(body.minDurationMs ?? 400)),
    "--gap-fill-ms",
    String(Math.trunc(body.gapFillMs ?? 220)),
  ];
  await new Promise<void>((resolve, reject) => {
    const proc = spawn("uv", args, { cwd: projectRoot(), stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr.trim() || `sam3d analyze exited with code ${code}`));
    });
  });
  return getRunDetail(runId);
}

// Delete a run's directory, checking the versioned location first then the legacy one.
export async function deleteRun(runIdRaw: string): Promise<void> {
  const runId = ensureSafeId(runIdRaw);
  const dir = runDir(runId);
  const legacy = path.join(legacyOutputRoot(), runId);
  const exists = await isRunDirectory(runsRoot(), runId);
  if (exists) {
    await fs.rm(dir, { recursive: true, force: true });
    return;
  }
  if (usesConfiguredRunsRoot()) {
    throw new Error(`Run not found: ${runId}`);
  }
  const legacyExists = await fs.stat(legacy).then((s) => s.isDirectory()).catch(() => false);
  if (legacyExists) {
    await fs.rm(legacy, { recursive: true, force: true });
    return;
  }
  throw new Error(`Run not found: ${runId}`);
}
