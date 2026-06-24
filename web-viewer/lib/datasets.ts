import { promises as fs } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";

import { datasetsRoot, ensureSafeId, projectRoot, readJsonIfExists } from "./store";
import type {
  DatasetAggregateMetrics,
  DatasetEvaluationSummary,
  DatasetRunMetrics,
  LabeledEpisode,
  RunDatasetLink,
  RunFrame,
} from "./types";

const DATASET_MANIFEST_FILE = "manifest.jsonl";

type DatasetManifestRow = Record<string, unknown>;

// Coerce a value to a finite number, or null for anything else (NaN, strings, missing).
function numericOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// Map an arbitrary split value to the two recognized splits; anything unknown is treated as holdout.
function normalizeSplit(value: unknown): "tuning" | "holdout" {
  return String(value ?? "holdout").trim().toLowerCase() === "tuning" ? "tuning" : "holdout";
}

// Parse a JSONL manifest into rows, skipping blank and malformed lines.
function parseJsonLines(text: string): DatasetManifestRow[] {
  const rows: DatasetManifestRow[] = [];
  for (const line of text.split(/\r?\n/g)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object") {
        rows.push(parsed as DatasetManifestRow);
      }
    } catch {
      // Ignore malformed lines so a partial manifest does not break the viewer.
    }
  }
  return rows;
}

// Resolve a manifest field path against the manifest's directory; absolute paths pass through unchanged.
function resolveManifestRelativePath(manifestPath: string, rawValue: string): string | null {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return null;
  }
  return path.isAbsolute(trimmed)
    ? trimmed
    : path.resolve(path.dirname(manifestPath), trimmed);
}

// Find the index of the frame whose videoFrame is closest to the target, or null if none.
function nearestFrameIndex(targetVideoFrame: number, frames: RunFrame[]): number | null {
  if (!Number.isFinite(targetVideoFrame) || frames.length === 0) {
    return null;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < frames.length; index += 1) {
    const distance = Math.abs(frames[index].videoFrame - targetVideoFrame);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

// Convert an episode timestamp (ms) to the nearest frame index; endExclusive nudges back 1ms so an
// end boundary lands on the last frame inside the episode rather than the first frame after it.
function episodeFrameIndex(ms: number, frames: RunFrame[], fps: number, endExclusive = false): number | null {
  const safeFps = Math.max(1, fps);
  const adjustedMs = endExclusive ? Math.max(0, ms - 1) : Math.max(0, ms);
  return nearestFrameIndex((adjustedMs / 1000) * safeFps, frames);
}

// Turn a label-events payload into validated episodes with frame indices, dropping empty/invalid spans.
function parseLabeledEpisodes(
  payload: Record<string, unknown> | null,
  frames: RunFrame[],
  fps: number,
): LabeledEpisode[] {
  if (!payload || !Array.isArray(payload.episodes)) {
    return [];
  }
  return payload.episodes
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
    .map((item) => {
      const startMs = Math.max(0, Math.round(Number(item.start_ms ?? 0)));
      const endMs = Math.max(startMs, Math.round(Number(item.end_ms ?? 0)));
      const label = String(item.label ?? "fog");
      if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
        return null;
      }
      const startFrameIndex = episodeFrameIndex(startMs, frames, fps, false);
      const endFrameIndex = episodeFrameIndex(endMs, frames, fps, true);
      return {
        label,
        startMs,
        endMs,
        durationMs: endMs - startMs,
        startFrameIndex,
        endFrameIndex,
      } satisfies LabeledEpisode;
    })
    .filter((item): item is LabeledEpisode => item !== null);
}

// Map snake_case per-run metrics from disk into the camelCase shape used by the viewer.
function normalizeRunMetrics(value: unknown): DatasetRunMetrics {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    precision: numericOrNull(raw.precision),
    recall: numericOrNull(raw.recall),
    f1Event: numericOrNull(raw.f1_event),
    matchedEvents: numericOrNull(raw.matched_events),
    falsePositiveEvents: numericOrNull(raw.false_positive_events),
    falseNegativeEvents: numericOrNull(raw.false_negative_events),
    meanOnsetLatencyMs: numericOrNull(raw.mean_onset_latency_ms),
    falsePositiveDurationPerMinS: numericOrNull(raw.false_positive_duration_per_min_s),
    nonInterpretableRate: numericOrNull(raw.non_interpretable_rate),
  };
}

// Map snake_case per-split aggregate metrics from disk into the camelCase shape used by the viewer.
function normalizeAggregateMetrics(value: unknown): DatasetAggregateMetrics {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    count: Math.max(0, Math.round(Number(raw.count ?? 0))),
    precision: numericOrNull(raw.precision),
    recall: numericOrNull(raw.recall),
    f1Event: numericOrNull(raw.f1_event),
    meanOnsetLatencyMs: numericOrNull(raw.mean_onset_latency_ms),
    falsePositiveDurationPerMinS: numericOrNull(raw.false_positive_duration_per_min_s),
    nonInterpretableRate: numericOrNull(raw.non_interpretable_rate),
  };
}

// Normalize a full evaluation summary (splits + per-run rows) from the on-disk JSON into viewer types.
function normalizeEvaluationPayload(payload: Record<string, unknown>): DatasetEvaluationSummary {
  const splits = payload.splits && typeof payload.splits === "object"
    ? (payload.splits as Record<string, unknown>)
    : {};
  const perRunRaw = Array.isArray(payload.per_run) ? payload.per_run : [];
  return {
    datasetId: String(payload.dataset_id ?? ""),
    preset: String(payload.preset ?? ""),
    generatedAt: typeof payload.generated_at === "string" ? payload.generated_at : null,
    runsEvaluated: Math.max(0, Math.round(Number(payload.runs_evaluated ?? 0))),
    splits: {
      tuning: normalizeAggregateMetrics(splits.tuning),
      holdout: normalizeAggregateMetrics(splits.holdout),
    },
    perRun: perRunRaw
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
      .map((item) => ({
        runId: String(item.run_id ?? ""),
        split: normalizeSplit(item.split),
        metrics: normalizeRunMetrics(item.metrics),
        qaStatus:
          item.qa && typeof item.qa === "object" && typeof (item.qa as Record<string, unknown>).status === "string"
            ? String((item.qa as Record<string, unknown>).status)
            : null,
        needsReview:
          Boolean(
            item.qa &&
            typeof item.qa === "object" &&
            (item.qa as Record<string, unknown>).needs_review,
          ),
      })),
  };
}

// Scan a dataset's evaluations/<preset>/latest.json files and return the most recently modified one.
async function readLatestDatasetEvaluationFile(datasetId: string): Promise<DatasetEvaluationSummary | null> {
  const filePath = path.join(datasetsRoot(), datasetId, "evaluations");
  const entries = await fs.readdir(filePath, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const latestPath = path.join(filePath, entry.name, "latest.json");
        const [payload, stat] = await Promise.all([
          readJsonIfExists<Record<string, unknown>>(latestPath),
          fs.stat(latestPath).catch(() => null),
        ]);
        if (!payload || !stat) {
          return null;
        }
        return {
          mtimeMs: stat.mtimeMs,
          payload: normalizeEvaluationPayload(payload),
        };
      }),
  );
  const latest = candidates
    .filter((item): item is { mtimeMs: number; payload: DatasetEvaluationSummary } => item !== null)
    .sort((a, b) => b.mtimeMs - a.mtimeMs)[0];
  return latest?.payload ?? null;
}

// Return the latest stored evaluation for a dataset, throwing if none exists.
export async function getLatestDatasetEvaluation(datasetIdRaw: string): Promise<DatasetEvaluationSummary> {
  const datasetId = ensureSafeId(datasetIdRaw);
  const payload = await readLatestDatasetEvaluationFile(datasetId);
  if (!payload) {
    throw new Error("No dataset evaluation found");
  }
  return payload;
}

// Run the `sam3d evaluate` CLI for a dataset and return the resulting summary, falling back to the
// freshly written file if the CLI's stdout JSON cannot be parsed.
export async function createDatasetEvaluation(
  datasetIdRaw: string,
  options?: { preset?: string },
): Promise<DatasetEvaluationSummary> {
  const datasetId = ensureSafeId(datasetIdRaw);
  const manifestPath = path.join(datasetsRoot(), datasetId, DATASET_MANIFEST_FILE);
  const preset = options?.preset?.trim() || "clinical_fog_v1";
  const output = await new Promise<string>((resolve, reject) => {
    const child = spawn("uv", [
      "run",
      "sam3d",
      "evaluate",
      "--dataset-manifest",
      manifestPath,
      "--preset",
      preset,
      "--json",
    ], {
      cwd: projectRoot(),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout.trim());
        return;
      }
      reject(new Error(stderr.trim() || `sam3d evaluate exited with code ${code}`));
    });
  });

  try {
    const parsed = JSON.parse(output);
    if (parsed && typeof parsed === "object") {
      return normalizeEvaluationPayload(parsed as Record<string, unknown>);
    }
  } catch {
    // Fall back to reading the written summary file below.
  }

  return getLatestDatasetEvaluation(datasetId);
}

// Find every dataset whose manifest references the given run, returning one link per manifest match
// enriched with resolved labels and the run's latest evaluation metrics, sorted by dataset id.
export async function discoverRunDatasets(
  runIdRaw: string,
  frames: RunFrame[],
  fps: number,
): Promise<RunDatasetLink[]> {
  const runId = ensureSafeId(runIdRaw);
  const root = datasetsRoot();
  const datasetDirs = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const links: RunDatasetLink[] = [];

  for (const entry of datasetDirs) {
    if (!entry.isDirectory()) {
      continue;
    }
    const datasetId = entry.name;
    const manifestPath = path.join(root, datasetId, DATASET_MANIFEST_FILE);
    const manifestText = await fs.readFile(manifestPath, "utf-8").catch(() => null);
    if (!manifestText) {
      continue;
    }
    const latestEvaluation = await readLatestDatasetEvaluationFile(datasetId);
    for (const row of parseJsonLines(manifestText)) {
      if (String(row.run_id ?? "").trim() !== runId) {
        continue;
      }
      const labelPathRaw = typeof row.label_events_path === "string" ? row.label_events_path : "";
      const resolvedLabelPath = resolveManifestRelativePath(manifestPath, labelPathRaw);
      const labelPayload = resolvedLabelPath
        ? await readJsonIfExists<Record<string, unknown>>(resolvedLabelPath)
        : null;
      const runEvaluation = latestEvaluation?.perRun.find((item) => item.runId === runId) ?? null;
      links.push({
        datasetId,
        split: normalizeSplit(row.split),
        patientId: typeof row.patient_id === "string" ? row.patient_id : null,
        sessionId: typeof row.session_id === "string" ? row.session_id : null,
        videoPath: typeof row.video_path === "string" ? row.video_path : null,
        notes: typeof row.notes === "string" ? row.notes : null,
        labelEventsPath: resolvedLabelPath,
        labeledEpisodes: parseLabeledEpisodes(labelPayload, frames, fps),
        latestEvaluation: latestEvaluation
          ? {
              preset: latestEvaluation.preset,
              generatedAt: latestEvaluation.generatedAt,
              metrics: runEvaluation?.metrics ?? null,
            }
          : null,
      });
    }
  }

  return links.sort((a, b) => a.datasetId.localeCompare(b.datasetId));
}
