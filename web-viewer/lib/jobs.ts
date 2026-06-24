import { spawn, type ChildProcess } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

import { signalProcessTree } from "./process-tree";
import { projectRoot, runDir, uploadsRoot } from "./store";

export type GenerationJobStatus = "queued" | "running" | "completed" | "failed" | "canceled";
export type InferenceTarget = "body" | "hand";

export type GenerationJob = {
  id: string;
  runId: string;
  videoFileName: string;
  inferenceTarget: InferenceTarget;
  sam3TextPrompts: string[];
  status: GenerationJobStatus;
  // Running-but-suspended (SIGSTOP). Status stays "running"; the UI shows resume.
  paused?: boolean;
  createdAt: string;
  updatedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  processedFrames: number;
  totalFrames: number | null;
  progressPercent: number | null;
  error: string | null;
  inputVideoPath: string;
  runDir: string;
  recentLogs: string[];
};

type CreateOptions = {
  precisionRaw?: string | null;
  autoInitModeRaw?: string | null;
  autoSelectStrategyRaw?: string | null;
  cameraMotionCompensationRaw?: string | null;
  renderPreviewRaw?: string | null;
  promptBBoxRaw?: string | null;
  promptBBoxFrameRaw?: string | null;
  promptAnchorsJsonRaw?: string | null;
  startFrameRaw?: string | null;
  maxFramesRaw?: string | null;
  frameStepRaw?: string | null;
  trimStartSecRaw?: string | null;
  trimEndSecRaw?: string | null;
  removedSegmentsRaw?: string | null;
  maskedSegmentsRaw?: string | null;
  cropBoxRaw?: string | null;
  // Absolute path to the chosen subject's dense per-frame box track from the
  // detect step (validated server-side); drives the run via --subject-track-file.
  subjectTrackFile?: string | null;
};

type JobStore = {
  jobs: Map<string, GenerationJob>;
  processes: Map<string, ChildProcess>;
  jobOptions: Map<string, CreateOptions>;
};

const globalStore = globalThis as typeof globalThis & {
  __sam3dJobStore?: JobStore;
  __sam3dShutdownHook?: boolean;
};

const LOG_LIMIT = 80;

// Lazily create (and migrate) the process-global job store. State lives on
// globalThis so it survives Next.js dev hot-reloads of this module.
function store(): JobStore {
  if (!globalStore.__sam3dJobStore) {
    globalStore.__sam3dJobStore = { jobs: new Map(), processes: new Map(), jobOptions: new Map() };
  } else if (!globalStore.__sam3dJobStore.jobOptions) {
    globalStore.__sam3dJobStore.jobOptions = new Map();
  }
  installShutdownCleanup();
  return globalStore.__sam3dJobStore;
}

// On normal server exit, resume (SIGCONT) then terminate (SIGTERM) every tracked
// job group — otherwise a PAUSED job (SIGSTOP'd) is left a frozen orphan that
// never exits. Uses only the synchronous "exit" hook so it never changes the
// server's own shutdown behaviour. Registered once (flag survives hot-reloads).
function installShutdownCleanup(): void {
  if (globalStore.__sam3dShutdownHook) return;
  globalStore.__sam3dShutdownHook = true;
  process.once("exit", () => {
    const current = globalStore.__sam3dJobStore;
    if (!current) return;
    for (const child of current.processes.values()) {
      if (child.pid == null) continue;
      signalProcessTree(child, "SIGCONT"); // resume if paused (POSIX), then…
      signalProcessTree(child, "SIGTERM"); // …terminate the whole tree
    }
  });
}

function nowIso(): string {
  return new Date().toISOString();
}

// Sanitize an arbitrary string into a filesystem-safe slug (used for run ids).
function safeToken(raw: string): string {
  const token = raw.trim().replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  return token || "session";
}

// Coerce a raw inference-target string to a known value, defaulting to "body".
function normalizeTarget(raw: string | null | undefined): InferenceTarget {
  return raw === "hand" ? "hand" : "body";
}

// Parse the comma-separated SAM3 text prompts, capped at 8, with a per-target default.
function parsePrompts(raw: string | null | undefined, target: InferenceTarget): string[] {
  const fallback = target === "hand" ? ["hand"] : ["person"];
  if (!raw) {
    return fallback;
  }
  const prompts = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 8);
  return prompts.length > 0 ? prompts : fallback;
}

// Return `raw` if it is one of the allowed values, otherwise the fallback.
function normalizeEnum(raw: string | null | undefined, allowed: string[], fallback: string): string {
  return raw && allowed.includes(raw) ? raw : fallback;
}

// Parse an integer (truncating, clamped to >= min), falling back on empty/invalid input.
function parseInteger(raw: string | null | undefined, fallback: number | null, min = 0): number | null {
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, Math.trunc(value));
}

// Parse a strict "true"/"false" string, falling back on anything else.
function parseBoolean(raw: string | null | undefined, fallback: boolean): boolean {
  if (raw === "true") return true;
  if (raw === "false") return false;
  return fallback;
}

// Parse a float (clamped to >= min), falling back on empty/invalid input.
function parseNumber(raw: string | null | undefined, fallback: number | null, min = 0): number | null {
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(min, value);
}

// Validate and normalize an "x1,y1,x2,y2" pixel bbox; returns null if degenerate.
function normalizeBBox(raw: string | null | undefined): string | null {
  if (!raw) {
    return null;
  }
  const values = raw.split(",").map((item) => Number(item.trim()));
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    return null;
  }
  const [x1, y1, x2, y2] = values.map((value) => Math.max(0, Math.round(value)));
  if (x2 <= x1 || y2 <= y1) {
    return null;
  }
  return `${x1},${y1},${x2},${y2}`;
}

// Validate and clamp a normalized "x,y,w,h" crop box (0..1); returns null when
// it covers the whole frame (no-op crop) or is unparseable.
function normalizeCropBox(raw: string | null | undefined): [number, number, number, number] | null {
  if (!raw) {
    return null;
  }
  const values = raw.split(",").map((item) => Number(item.trim()));
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    return null;
  }
  const [xRaw, yRaw, wRaw, hRaw] = values;
  const x = Math.max(0, Math.min(0.98, xRaw));
  const y = Math.max(0, Math.min(0.98, yRaw));
  const width = Math.max(0.05, Math.min(1 - x, wRaw));
  const height = Math.max(0.05, Math.min(1 - y, hRaw));
  if (x < 0.001 && y < 0.001 && width > 0.999 && height > 0.999) {
    return null;
  }
  return [x, y, width, height];
}

type RemovedSegment = { startSec: number; endSec: number };
type KeepRange = { startSec: number; endSec: number };

// Parse the JSON segment list into clamped, sorted, non-overlapping spans
// (in original-video seconds). Used for both deleted and masked segments.
function normalizeRemovedSegments(
  raw: string | null | undefined,
  trimStartSec: number,
  trimEndSec: number | null,
): RemovedSegment[] {
  if (!raw) {
    return [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) {
    return [];
  }
  const segments: RemovedSegment[] = [];
  for (const item of parsed) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    const startRaw = Number(record.startSec);
    const endRaw = Number(record.endSec);
    if (!Number.isFinite(startRaw) || !Number.isFinite(endRaw)) {
      continue;
    }
    const lower = Math.max(0, trimStartSec);
    const upper = trimEndSec ?? Number.POSITIVE_INFINITY;
    const startSec = Math.max(lower, Math.min(startRaw, endRaw));
    const endSec = Math.min(upper, Math.max(startRaw, endRaw));
    if (endSec - startSec >= 0.05) {
      segments.push({ startSec, endSec });
    }
  }
  segments.sort((a, b) => a.startSec - b.startSec);
  const merged: RemovedSegment[] = [];
  for (const segment of segments) {
    const previous = merged.at(-1);
    if (previous && segment.startSec <= previous.endSec + 0.01) {
      previous.endSec = Math.max(previous.endSec, segment.endSec);
    } else {
      merged.push({ ...segment });
    }
  }
  return merged;
}

// Invert the deleted segments within [trimStart, trimEnd] into the kept spans
// that should survive into the edited video.
function buildKeepRanges(trimStartSec: number, trimEndSec: number | null, removedSegments: RemovedSegment[]): KeepRange[] {
  const endSec = trimEndSec ?? Number.POSITIVE_INFINITY;
  let cursor = Math.max(0, trimStartSec);
  const ranges: KeepRange[] = [];
  for (const segment of removedSegments) {
    const segmentStart = Math.max(cursor, segment.startSec);
    if (segmentStart - cursor >= 0.05) {
      ranges.push({ startSec: cursor, endSec: segmentStart });
    }
    cursor = Math.max(cursor, segment.endSec);
  }
  if (Number.isFinite(endSec)) {
    if (endSec - cursor >= 0.05) {
      ranges.push({ startSec: cursor, endSec });
    }
  } else {
    ranges.push({ startSec: cursor, endSec });
  }
  return ranges;
}

// Map an original-video timestamp to the corresponding time in the edited
// video (after trimStart and deleted segments are removed).
function originalToEditedSec(
  originalSec: number,
  trimStartSec: number,
  removedSegments: RemovedSegment[],
): number {
  const trim = Math.max(0, trimStartSec);
  let edited = originalSec - trim;
  for (const seg of removedSegments) {
    const segStart = Math.max(seg.startSec, trim);
    const segEnd = Math.max(seg.endSec, trim);
    if (segEnd <= originalSec) {
      edited -= Math.max(0, segEnd - segStart);
    } else if (segStart < originalSec) {
      edited -= Math.max(0, originalSec - segStart);
    }
  }
  return Math.max(0, edited);
}

// Build the --mask-time-ranges CLI value (edited-video seconds) from the user's
// masked spans (original-video seconds). Masked spans stay in the output video
// at their original timing but skip inference.
function buildMaskTimeRangesArg(options: CreateOptions): string {
  const trimStart = parseNumber(options.trimStartSecRaw, 0, 0) ?? 0;
  const trimEnd = parseNumber(options.trimEndSecRaw, null, 0);
  const masked = normalizeRemovedSegments(options.maskedSegmentsRaw, trimStart, trimEnd);
  if (masked.length === 0) {
    return "";
  }
  const deleted = normalizeRemovedSegments(options.removedSegmentsRaw, trimStart, trimEnd);
  const parts: string[] = [];
  for (const span of masked) {
    const start = originalToEditedSec(span.startSec, trimStart, deleted);
    const end = originalToEditedSec(span.endSec, trimStart, deleted);
    if (end > start + 0.02) {
      parts.push(`${start.toFixed(3)}-${end.toFixed(3)}`);
    }
  }
  return parts.join(",");
}

// Build an ffmpeg `crop` filter from a normalized crop box; dimensions are
// snapped to even pixels (yuv420p requires even width/height).
function cropFilter(crop: [number, number, number, number]): string {
  const [x, y, width, height] = crop;
  return [
    `crop=w='max(2,trunc(iw*${width.toFixed(6)}/2)*2)'`,
    `h='max(2,trunc(ih*${height.toFixed(6)}/2)*2)'`,
    `x='trunc(iw*${x.toFixed(6)}/2)*2'`,
    `y='trunc(ih*${y.toFixed(6)}/2)*2'`,
  ].join(":");
}

// Count exported .ply meshes in a run dir. A .ply is only written when a subject
// is present on the frame, so this UNDERCOUNTS during subject-absent stretches.
async function countMeshFrames(directory: string): Promise<number> {
  const meshes = path.join(directory, "meshes");
  const entries = await fs.readdir(meshes, { withFileTypes: true }).catch(() => []);
  return entries.filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".ply")).length;
}

// Read the live progress the pipeline writes into run_metadata.json. Unlike the
// .ply count, total_frames_processed counts EVERY processed frame (incl.
// subject-absent ones), and total_frames_target gives the denominator for a %.
async function readMetadataProgress(
  directory: string,
): Promise<{ processed: number; total: number | null } | null> {
  try {
    const raw = await fs.readFile(path.join(directory, "run_metadata.json"), "utf-8");
    const data = JSON.parse(raw) as {
      total_frames_processed?: unknown;
      total_frames_target?: unknown;
      total_frames_input?: unknown;
    };
    const processed = Number(data.total_frames_processed);
    const total = Number(data.total_frames_target ?? data.total_frames_input);
    return {
      processed: Number.isFinite(processed) ? processed : 0,
      total: Number.isFinite(total) && total > 0 ? total : null,
    };
  } catch {
    return null;
  }
}

// Find a non-colliding file path by appending _1, _2, ... if needed.
async function uniquePath(directory: string, baseName: string, extension: string): Promise<string> {
  let candidate = path.join(directory, `${baseName}${extension}`);
  for (let index = 1; await fs.stat(candidate).then(() => true).catch(() => false); index += 1) {
    candidate = path.join(directory, `${baseName}_${index}${extension}`);
  }
  return candidate;
}

// Find a non-colliding run id by appending _1, _2, ... if the run dir exists.
async function uniqueRunId(baseRun: string): Promise<string> {
  let candidate = baseRun;
  for (let index = 1; await fs.stat(runDir(candidate)).then((stat) => stat.isDirectory()).catch(() => false); index += 1) {
    candidate = `${baseRun}_${index}`;
  }
  return candidate;
}

// Run ffmpeg with the given args, rejecting (with captured stderr) on nonzero exit.
function runFfmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`ffmpeg failed with code ${code}: ${stderr.trim() || "unknown error"}`));
      }
    });
  });
}

// Apply the user's timeline edits (trim, deleted segments, crop) into a new mp4
// and return its path; returns the original path unchanged when there are no edits.
async function prepareEditedVideo(inputVideoPath: string, options: CreateOptions): Promise<string> {
  const trimStart = parseNumber(options.trimStartSecRaw, 0, 0) ?? 0;
  const trimEnd = parseNumber(options.trimEndSecRaw, null, 0);
  const crop = normalizeCropBox(options.cropBoxRaw);
  const removedSegments = normalizeRemovedSegments(options.removedSegmentsRaw, trimStart, trimEnd);
  const hasTrim = trimStart > 0.01 || (trimEnd !== null && trimEnd > trimStart + 0.1);
  if (!hasTrim && !crop && removedSegments.length === 0) {
    return inputVideoPath;
  }

  const parsed = path.parse(inputVideoPath);
  const outputPath = await uniquePath(parsed.dir, `${parsed.name}_edited`, ".mp4");
  if (removedSegments.length > 0) {
    const keepRanges = buildKeepRanges(trimStart, trimEnd, removedSegments);
    if (keepRanges.length === 0) {
      throw new Error("The timeline cuts remove the entire selected video segment.");
    }
    const filters: string[] = [];
    // Filtergraph stages:
    //   trim -> [v0..vN] (one per kept segment)
    //   concat (only if N > 1) -> [vconcat]
    //   crop (only if crop) -> [vout]
    // The final ffmpeg `-map` always reads [vout], so the LAST emitted filter
    // must label its output [vout]. Earlier wiring confused these labels when
    // there were multiple kept segments without a crop, which made ffmpeg
    // error out and (in some setups) silently fall back to the original
    // unedited video being processed.
    const needsConcat = keepRanges.length > 1;
    const trimOutputLabel = (index: number): string => {
      if (needsConcat) return `v${index}`;
      // Single trimmed range. If we still need to crop afterwards, hand off
      // through [v0]; otherwise jump straight to [vout].
      return crop ? "v0" : "vout";
    };
    const concatOutputLabel = crop ? "vconcat" : "vout";
    const cropInputLabel = needsConcat ? "vconcat" : "v0";
    for (const [index, range] of keepRanges.entries()) {
      const trimParts = [`start=${range.startSec.toFixed(3)}`];
      if (Number.isFinite(range.endSec)) {
        trimParts.push(`end=${range.endSec.toFixed(3)}`);
      }
      filters.push(`[0:v]trim=${trimParts.join(":")},setpts=PTS-STARTPTS[${trimOutputLabel(index)}]`);
    }
    if (needsConcat) {
      const concatInput = keepRanges.map((_, index) => `[v${index}]`).join("");
      filters.push(`${concatInput}concat=n=${keepRanges.length}:v=1:a=0[${concatOutputLabel}]`);
    }
    if (crop) {
      filters.push(`[${cropInputLabel}]${cropFilter(crop)}[vout]`);
    }
    await runFfmpeg([
      "-y",
      "-i",
      inputVideoPath,
      "-filter_complex",
      filters.join(";"),
      "-map",
      "[vout]",
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      outputPath,
    ]);
    return outputPath;
  }

  const args = ["-y"];
  if (trimStart > 0.01) {
    args.push("-ss", trimStart.toFixed(3));
  }
  args.push("-i", inputVideoPath);
  if (trimEnd !== null && trimEnd > trimStart + 0.1) {
    args.push("-t", (trimEnd - trimStart).toFixed(3));
  }
  if (crop) {
    args.push("-vf", cropFilter(crop));
  }
  args.push("-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", outputPath);
  await runFfmpeg(args);
  return outputPath;
}

// Write the initial run_metadata.json and run_manifest.json placeholders so the
// run dir is readable before the Python pipeline starts producing real output.
async function writeInitialRunFiles(job: GenerationJob, processingStatus: "queued" | "running" = "queued"): Promise<void> {
  const createdAt = nowIso();
  await fs.mkdir(job.runDir, { recursive: true });
  await fs.writeFile(
    path.join(job.runDir, "run_metadata.json"),
    JSON.stringify({
      video_input: job.inputVideoPath,
      output_video: null,
      mesh_dir: path.join(job.runDir, "meshes"),
      mesh_export_enabled: true,
      joint_timeseries_export_enabled: true,
      inference_target: job.inferenceTarget,
      fps_output: null,
      video_width: null,
      video_height: null,
      total_frames_processed: 0,
      processing_status: processingStatus,
      records: [],
    }),
    "utf-8",
  );
  await fs.writeFile(
    path.join(job.runDir, "run_manifest.json"),
    JSON.stringify({
      schema_version: "sam3d.run.v1",
      run_id: job.runId,
      created_at: createdAt,
      updated_at: createdAt,
      source_video: job.inputVideoPath,
      config_profile: "clinical_fog_workstation_v1",
      inference_target: job.inferenceTarget,
      subject_count: 1,
      frame_count: 0,
      processed_frames: 0,
      fps: null,
      video_width: null,
      video_height: null,
      quality_summary: null,
      artifacts: {
        run_metadata: "run_metadata.json",
        meshes: "meshes",
        preview_video: null,
        logs: "logs",
      },
      analysis_profiles: { default: "clinical_fog_v1" },
      analyses: [],
      latest_analysis_id: null,
      processing_status: processingStatus,
    }),
    "utf-8",
  );
}

// Recompute progressPercent from processed/total frames and bump updatedAt.
function updateProgress(job: GenerationJob): void {
  if (job.totalFrames && job.totalFrames > 0) {
    job.progressPercent = Math.max(0, Math.min(100, (100 * job.processedFrames) / job.totalFrames));
  } else {
    job.progressPercent = null;
  }
  // "paused" only makes sense while running — a finished/failed/canceled job must
  // never linger as paused (the UI would show a stuck Resume button).
  if (job.status !== "running") {
    job.paused = false;
  }
  job.updatedAt = nowIso();
}

// Whether the job was canceled (used to suppress post-cancel status updates).
function isCanceled(job: GenerationJob): boolean {
  return job.status === "canceled";
}

// Append a log line to the in-memory ring buffer (capped at LOG_LIMIT lines).
function pushLog(job: GenerationJob, line: string): void {
  const trimmed = line.trim();
  if (!trimmed) {
    return;
  }
  job.recentLogs.push(trimmed);
  while (job.recentLogs.length > LOG_LIMIT) {
    job.recentLogs.shift();
  }
}

// Append a log line to the run's persistent logs/job.log file.
async function appendRunLog(job: GenerationJob, line: string): Promise<void> {
  const trimmed = line.trim();
  if (!trimmed) {
    return;
  }
  const logPath = path.join(job.runDir, "logs", "job.log");
  await fs.mkdir(path.dirname(logPath), { recursive: true });
  await fs.appendFile(logPath, `${trimmed}\n`, "utf-8");
}

// Spawn the `uv` Python entrypoint for a job, streaming stdout/stderr into the
// job logs, and resolve with its exit code. Configures the offline/MPS env.
function runCommand(job: GenerationJob, args: string[]): Promise<number> {
  const child = spawn("uv", args, {
    // Own process group so stop/pause signals reach `uv` AND its python child
    // (otherwise SIGTERM only hit uv and left python running; SIGSTOP couldn't
    // pause the actual inference).
    detached: true,
    cwd: projectRoot(),
    env: {
      ...process.env,
      PYTHONPATH: path.join(projectRoot(), "src"),
      SAM3D_MHR_MODE: process.env.SAM3D_MHR_MODE || "native",
      // The SAM3 detector uses an op (aten::_assert_async) with no MPS kernel;
      // without the per-op CPU fallback it raises and detection silently finds
      // NOBODY on Apple Silicon (the whole run then reports subject-absent).
      // Force it on — harmless on CUDA/CPU. A parent "0" must not win here:
      // `"0" || "1"` === "0", which is exactly the footgun this replaces.
      PYTORCH_ENABLE_MPS_FALLBACK: "1",
      // Fully offline: models load from the local HF cache, never the network.
      // Set HF_HUB_OFFLINE=0 once if you need to download weights the first time.
      HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "1",
      TRANSFORMERS_OFFLINE: process.env.TRANSFORMERS_OFFLINE || "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  store().processes.set(job.id, child);
  return new Promise((resolve, reject) => {
    const onData = (chunk: Buffer): void => {
      const text = chunk.toString();
      for (const line of text.split(/\r?\n/g)) {
        pushLog(job, line);
        void appendRunLog(job, line);
      }
      updateProgress(job);
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("error", reject);
    child.on("close", (code) => {
      store().processes.delete(job.id);
      resolve(code ?? 1);
    });
  });
}

// Drive a single job end to end: prepare the video, run the `sam3d run` and
// `sam3d analyze` stages, track progress, and record the final status.
async function startJob(jobId: string, options: CreateOptions): Promise<void> {
  const job = store().jobs.get(jobId);
  if (!job || job.status === "canceled") {
    return;
  }
  job.status = "running";
  job.startedAt = nowIso();
  updateProgress(job);
  await writeInitialRunFiles(job, "running");

  const precision = normalizeEnum(options.precisionRaw, ["float32", "float16"], "float32");
  const autoInitMode = normalizeEnum(options.autoInitModeRaw, ["off", "smart", "sam3"], "sam3");
  const autoSelectStrategy = normalizeEnum(
    options.autoSelectStrategyRaw,
    ["patient", "largest", "leftmost", "rightmost", "center", "tightest"],
    "patient",
  );
  const startFrame = parseInteger(options.startFrameRaw, 0, 0) ?? 0;
  const maxFrames = parseInteger(options.maxFramesRaw, null, 1);
  const frameStep = parseInteger(options.frameStepRaw, 1, 1) ?? 1;
  const renderPreview = parseBoolean(options.renderPreviewRaw, false);
  // NOTE: camera-motion-compensation is accepted in the request for backward
  // compatibility but the CLI/pipeline does not implement it, so it is not
  // forwarded as a flag (doing so makes argparse exit with code 2).
  const promptBBox = normalizeBBox(options.promptBBoxRaw);
  const promptBBoxFrame = parseInteger(options.promptBBoxFrameRaw, null, 0);
  const runVideoInputPath = await prepareEditedVideo(job.inputVideoPath, options);
  const maskTimeRangesArg = buildMaskTimeRangesArg(options);
  if (runVideoInputPath !== job.inputVideoPath) {
    pushLog(job, `Prepared edited input video: ${runVideoInputPath}`);
    void appendRunLog(job, `Prepared edited input video: ${runVideoInputPath}`);
    job.inputVideoPath = runVideoInputPath;
    await writeInitialRunFiles(job, "running");
  }

  const progressTimer = setInterval(async () => {
    const current = store().jobs.get(job.id);
    if (!current || current.status !== "running") {
      clearInterval(progressTimer);
      return;
    }
    const meta = await readMetadataProgress(current.runDir);
    const meshes = await countMeshFrames(current.runDir);
    // Prefer the pipeline's processed count; never regress below the mesh count.
    current.processedFrames = Math.max(meta?.processed ?? 0, meshes);
    if (meta?.total && !current.totalFrames) current.totalFrames = meta.total;
    updateProgress(current);
  }, 1000);

  try {
    const runArgs = [
      "run",
      "sam3d",
      "run",
      "--video-input",
      runVideoInputPath,
      "--output-dir",
      job.runDir,
      "--run-id",
      job.runId,
      "--inference-target",
      job.inferenceTarget,
      "--precision",
      precision,
      "--auto-init-mode",
      autoInitMode,
      "--auto-select-strategy",
      autoSelectStrategy,
      "--start-frame",
      String(startFrame),
      "--frame-step",
      String(frameStep),
      ...(maxFrames ? ["--max-frames", String(maxFrames)] : []),
      ...(promptBBox ? ["--prompt-bbox", promptBBox] : []),
      ...(promptBBox && promptBBoxFrame !== null
        ? ["--prompt-bbox-frame", String(promptBBoxFrame)]
        : []),
      ...(options.promptAnchorsJsonRaw
        ? ["--prompt-anchors-json", options.promptAnchorsJsonRaw]
        : []),
      ...(options.subjectTrackFile
        ? ["--subject-track-file", options.subjectTrackFile]
        : []),
      ...(maskTimeRangesArg ? ["--mask-time-ranges", maskTimeRangesArg] : []),
      ...(!renderPreview ? ["--no-preview"] : []),
      ...(job.sam3TextPrompts.length > 0 ? ["--sam3-text-prompts", job.sam3TextPrompts.join(",")] : []),
    ];
    const runCode = await runCommand(job, runArgs);
    if (isCanceled(job)) {
      return;
    }
    {
      const meta = await readMetadataProgress(job.runDir);
      job.processedFrames = Math.max(meta?.processed ?? 0, await countMeshFrames(job.runDir));
      if (meta?.total && !job.totalFrames) job.totalFrames = meta.total;
    }
    updateProgress(job);
    if (runCode !== 0) {
      job.status = "failed";
      job.error = `sam3d run exited with code ${runCode}`;
      job.finishedAt = nowIso();
      updateProgress(job);
      return;
    }

    const analyzeCode = await runCommand(job, [
      "run",
      "sam3d",
      "analyze",
      "--run-id",
      job.runId,
      "--preset",
      "clinical_fog_v1",
    ]);
    if (isCanceled(job)) {
      return;
    }
    if (analyzeCode !== 0) {
      job.status = "failed";
      job.error = `sam3d analyze exited with code ${analyzeCode}`;
      job.finishedAt = nowIso();
      updateProgress(job);
      return;
    }

    {
      const meta = await readMetadataProgress(job.runDir);
      job.processedFrames = Math.max(meta?.processed ?? 0, await countMeshFrames(job.runDir));
      if (meta?.total && !job.totalFrames) job.totalFrames = meta.total;
    }
    job.status = "completed";
    job.error = null;
    job.finishedAt = nowIso();
    updateProgress(job);
  } catch (error) {
    if (!isCanceled(job)) {
      job.status = "failed";
      job.error = String(error);
      job.finishedAt = nowIso();
      updateProgress(job);
    }
  } finally {
    clearInterval(progressTimer);
    store().jobOptions.delete(jobId);
    scheduleJobs();
  }
}

// Start the oldest queued job if nothing is currently running (one job at a time).
function scheduleJobs(): void {
  const jobStore = store();
  const hasRunningJob = Array.from(jobStore.jobs.values()).some((job) => job.status === "running");
  if (hasRunningJob) {
    return;
  }
  const nextJob = Array.from(jobStore.jobs.values())
    .filter((job) => job.status === "queued")
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt))[0];
  if (!nextJob) {
    return;
  }
  const options = jobStore.jobOptions.get(nextJob.id) ?? {};
  void startJob(nextJob.id, options);
}

// List all jobs, newest first.
export function listJobs(): GenerationJob[] {
  return Array.from(store().jobs.values()).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

// Persist an uploaded video buffer to the uploads dir, then queue a job for it.
export async function createJob(
  fileName: string,
  fileData: Buffer,
  runName: string | null,
  inferenceTargetRaw: string | null,
  sam3TextPromptsRaw: string | null,
  options: CreateOptions = {},
): Promise<GenerationJob> {
  const uploadDir = uploadsRoot();
  await fs.mkdir(uploadDir, { recursive: true });
  const inputExt = path.extname(fileName).toLowerCase() || ".mp4";
  const inputBase = safeToken(path.basename(fileName, path.extname(fileName)));
  const inputVideoPath = await uniquePath(uploadDir, inputBase, inputExt);
  await fs.writeFile(inputVideoPath, fileData);
  return createJobFromExistingUpload(fileName, inputVideoPath, runName, inferenceTargetRaw, sam3TextPromptsRaw, options);
}

// Queue a job for a video already on disk. The path is validated to live inside
// the uploads dir (guards against path traversal) before the job is created.
export async function createJobFromExistingUpload(
  fileName: string,
  inputVideoPath: string,
  runName: string | null,
  inferenceTargetRaw: string | null,
  sam3TextPromptsRaw: string | null,
  options: CreateOptions = {},
): Promise<GenerationJob> {
  const resolvedInputPath = path.resolve(inputVideoPath);
  const resolvedUploadRoot = path.resolve(uploadsRoot());
  if (!resolvedInputPath.startsWith(`${resolvedUploadRoot}${path.sep}`)) {
    throw new Error("Input video must be inside the uploads directory");
  }
  const inputStat = await fs.stat(resolvedInputPath);
  if (!inputStat.isFile() || inputStat.size <= 0) {
    throw new Error("Input video file is missing");
  }
  const now = nowIso();
  const target = normalizeTarget(inferenceTargetRaw);
  const sourceBase = safeToken(runName || path.basename(fileName, path.extname(fileName)));
  const baseRun = sourceBase.endsWith("_processed") ? sourceBase : `${sourceBase}_processed`;
  const runId = await uniqueRunId(baseRun);
  const id = crypto.randomUUID();
  const directory = runDir(runId);
  await fs.mkdir(path.join(directory, "logs"), { recursive: true });

  const job: GenerationJob = {
    id,
    runId,
    videoFileName: fileName,
    inferenceTarget: target,
    sam3TextPrompts: parsePrompts(sam3TextPromptsRaw, target),
    status: "queued",
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
    processedFrames: 0,
    totalFrames: parseInteger(options.maxFramesRaw, null, 1),
    progressPercent: null,
    error: null,
    inputVideoPath: resolvedInputPath,
    runDir: directory,
    recentLogs: [],
  };
  await writeInitialRunFiles(job);
  store().jobs.set(id, job);
  store().jobOptions.set(id, options);
  scheduleJobs();
  return job;
}

// Signal a job's whole process tree (uv + python + children), cross-platform.
// Returns true if something was signalled.
function signalJobGroup(jobId: string, signal: NodeJS.Signals): boolean {
  const child = store().processes.get(jobId);
  if (!child) return false;
  return signalProcessTree(child, signal);
}

// Cancel a queued or running job: terminate its process tree (resuming first if
// paused) and mark it canceled. Returns the updated job, or null if unknown.
export function stopJob(jobId: string): GenerationJob | null {
  const job = store().jobs.get(jobId) ?? null;
  if (!job) {
    return null;
  }
  if (job.status === "queued" || job.status === "running") {
    // Resume first so a paused (SIGSTOP'd) process can actually receive SIGTERM.
    if (job.paused) signalJobGroup(job.id, "SIGCONT");
    signalJobGroup(job.id, "SIGTERM");
    job.status = "canceled";
    job.paused = false;
    job.error = "Canceled by user.";
    job.finishedAt = nowIso();
    updateProgress(job);
    store().jobOptions.delete(job.id);
    scheduleJobs();
  }
  return job;
}

// Suspend a running job's process group (SIGSTOP). Status stays "running".
export function pauseJob(jobId: string): GenerationJob | null {
  const job = store().jobs.get(jobId) ?? null;
  if (!job || job.status !== "running" || job.paused) return job;
  if (signalJobGroup(job.id, "SIGSTOP")) {
    job.paused = true;
    updateProgress(job);
  }
  return job;
}

// Resume a paused job's process group (SIGCONT).
export function resumeJob(jobId: string): GenerationJob | null {
  const job = store().jobs.get(jobId) ?? null;
  if (!job || job.status !== "running" || !job.paused) return job;
  // Always clear the flag, even if the process is already gone — otherwise a
  // failed SIGCONT would leave a stuck "Resume" button. If the process really
  // vanished, its close handler will finalize the job shortly after.
  signalJobGroup(job.id, "SIGCONT");
  job.paused = false;
  updateProgress(job);
  return job;
}

// Stop the current job and launch a fresh one with the same input + parameters.
export async function restartJob(jobId: string): Promise<GenerationJob | null> {
  const job = store().jobs.get(jobId) ?? null;
  if (!job) return null;
  const options = store().jobOptions.get(job.id) ?? {};
  const { videoFileName, inputVideoPath, inferenceTarget } = job;
  const prompts = job.sam3TextPrompts.join(",");
  removeJob(jobId);
  return createJobFromExistingUpload(
    videoFileName,
    inputVideoPath,
    videoFileName,
    inferenceTarget,
    prompts,
    options,
  );
}

/** Remove a job from the list entirely (used by the UI delete button). Stops it
 *  first if it is still running, then discards it from the store. */
export function removeJob(jobId: string): boolean {
  const job = store().jobs.get(jobId) ?? null;
  if (!job) {
    return false;
  }
  if (job.status === "queued" || job.status === "running") {
    if (job.paused) signalJobGroup(job.id, "SIGCONT");
    signalJobGroup(job.id, "SIGTERM");
  }
  store().processes.delete(job.id);
  store().jobOptions.delete(job.id);
  store().jobs.delete(jobId);
  scheduleJobs();
  return true;
}
