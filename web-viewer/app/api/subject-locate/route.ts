import { spawn } from "node:child_process";
import { Buffer } from "node:buffer";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { NextResponse } from "next/server";

import { isAllowedVideoFileName, resolveStagedUpload } from "../../../lib/chunked-uploads";
import { signalProcessTree } from "../../../lib/process-tree";
import { projectRoot } from "../../../lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Scanning the whole video runs the detector dozens of times — give it room.
export const maxDuration = 300;

type LocateResult = {
  frame_sec: number;
  frame_index: number;
  detection: {
    xyxy: number[];
    box: { x: number; y: number; width: number; height: number };
  } | null;
  info?: Record<string, unknown>;
};

type SubjectLocateOutput = {
  ok: boolean;
  error?: string;
  mode?: string;
  fps?: number;
  total_frames?: number;
  duration_sec?: number;
  video_width?: number;
  video_height?: number;
  scanned?: number;
  hit_count?: number;
  results?: LocateResult[];
};

/** Comma-joined, trimmed prompt list capped at 8 entries; falls back when empty/invalid. */
function normalizePromptList(raw: FormDataEntryValue | null, fallback: string): string {
  if (typeof raw !== "string") {
    return fallback;
  }
  const cleaned = raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 8)
    .join(",");
  return cleaned || fallback;
}

/** Returns raw only if it is one of the allowed values, otherwise the fallback. */
function normalizeEnum(raw: FormDataEntryValue | null, allowed: string[], fallback: string): string {
  return typeof raw === "string" && allowed.includes(raw) ? raw : fallback;
}

/** Comma-joined, sanitized list of non-negative seconds. */
function normalizeSecondsList(raw: FormDataEntryValue | null): string {
  if (typeof raw !== "string") {
    return "";
  }
  return raw
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((n) => Number.isFinite(n) && n >= 0)
    .slice(0, 32)
    .map((n) => n.toFixed(3))
    .join(",");
}

/** Validate a JSON list of [start, end] second pairs. Returns "" if invalid. */
function normalizeRangesJson(raw: FormDataEntryValue | null): string {
  if (typeof raw !== "string" || !raw.trim()) {
    return "";
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return "";
    const ranges = parsed
      .map((entry) => (Array.isArray(entry) ? [Number(entry[0]), Number(entry[1])] : null))
      .filter(
        (pair): pair is [number, number] =>
          pair !== null &&
          Number.isFinite(pair[0]) &&
          Number.isFinite(pair[1]) &&
          pair[1] > pair[0],
      )
      .slice(0, 64);
    return ranges.length > 0 ? JSON.stringify(ranges) : "";
  } catch {
    return "";
  }
}

// Hard ceiling for a single scan. A runaway/hung detector is killed and
// reported rather than left to run — kept under the route's maxDuration.
const SCAN_TIMEOUT_MS = 280_000;

// Spawn the Python subject-preview module in "locate" mode and parse its
// final stdout line as JSON. Resolves with the parsed output even on a
// non-zero exit (surfacing stderr via `.error`); rejects only on spawn
// failure or unparseable output. The child runs in its own process group and
// is killed if the client aborts the request or the scan times out, so an
// abandoned scan never leaves a heavy SAM3 process (uv + python) alive —
// those used to pile up across retries and thrash the machine.
function runSubjectLocate(args: string[], signal?: AbortSignal): Promise<SubjectLocateOutput> {
  return new Promise((resolve, reject) => {
    const child = spawn("uv", args, {
      cwd: projectRoot(),
      env: {
        ...process.env,
        PYTHONPATH: path.join(projectRoot(), "src"),
        // MPS lacks a few ops the SAM3 detector uses (e.g. aten::_assert_async);
        // without the per-op CPU fallback it raises and detection silently finds
        // nobody on Apple Silicon. Force it on — a parent "0" must not win
        // (`"0" || "1"` === "0"). Harmless on CUDA/CPU. Mirror the job-runner env.
        PYTORCH_ENABLE_MPS_FALLBACK: "1",
        SAM3D_MHR_MODE: process.env.SAM3D_MHR_MODE || "native",
        HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "1",
        TRANSFORMERS_OFFLINE: process.env.TRANSFORMERS_OFFLINE || "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      // Own process group so we can signal `uv` AND its python grandchild.
      detached: true,
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    let timedOut = false;
    let aborted = false;

    // Terminate the whole process tree (uv + python + children), cross-platform.
    const killTree = (sig: NodeJS.Signals) => {
      signalProcessTree(child, sig);
    };
    const killChild = () => {
      killTree("SIGTERM");
      setTimeout(() => killTree("SIGKILL"), 2_000).unref();
    };

    const timer = setTimeout(() => {
      timedOut = true;
      killChild();
    }, SCAN_TIMEOUT_MS);
    const onAbort = () => {
      aborted = true;
      killChild();
    };
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (timedOut) {
        resolve({ ok: false, error: `Subject scan timed out after ${Math.round(SCAN_TIMEOUT_MS / 1000)}s.` });
        return;
      }
      if (aborted) {
        resolve({ ok: false, error: "Subject scan cancelled." });
        return;
      }
      const lines = stdout.trim().split(/\r?\n/g).filter(Boolean);
      const payload = lines.at(-1) ?? "";
      try {
        const parsed = JSON.parse(payload) as SubjectLocateOutput;
        if (code !== 0 && !parsed.error) {
          parsed.error = stderr.trim() || `subject locate exited with code ${code}`;
        }
        resolve(parsed);
      } catch (error) {
        reject(new Error(`Failed to parse subject locate output: ${String(error)} ${stderr.trim()}`));
      }
    });
  });
}

// Accept an uploaded (or previously staged) video, validate/sanitize the
// scan parameters, run the subject locator, and return its findings as JSON.
export async function POST(request: Request) {
  let tempDir: string | null = null;
  try {
    const formData = await request.formData();
    const rawFile = formData.get("video");
    const stagedUploadIdRaw = formData.get("stagedUploadId");
    const autoSelectStrategy = normalizeEnum(
      formData.get("autoSelectStrategy"),
      ["patient", "largest", "leftmost", "rightmost", "center", "tightest"],
      "patient",
    );
    const sam3TextPrompts = normalizePromptList(formData.get("sam3TextPrompts"), "person");
    const preferredSecs = normalizeSecondsList(formData.get("frameSecs"));
    const keptRanges = normalizeRangesJson(formData.get("keptRanges"));

    const stagedUpload =
      typeof stagedUploadIdRaw === "string" && stagedUploadIdRaw.trim()
        ? await resolveStagedUpload(stagedUploadIdRaw)
        : null;

    if (!(rawFile instanceof File) && !stagedUpload) {
      return NextResponse.json({ error: "Missing uploaded video file" }, { status: 400 });
    }
    const inputFileName = stagedUpload?.fileName ?? (rawFile instanceof File ? rawFile.name : "");
    if (!isAllowedVideoFileName(inputFileName)) {
      return NextResponse.json(
        { error: "Unsupported video format. Use .mp4/.mov/.m4v/.avi/.mkv/.webm" },
        { status: 400 },
      );
    }

    let inputPath = stagedUpload?.filePath ?? null;
    if (!inputPath && rawFile instanceof File) {
      tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "kinesia-subject-locate-"));
      inputPath = path.join(tempDir, `input${path.extname(rawFile.name).toLowerCase() || ".mp4"}`);
      await fs.writeFile(inputPath, Buffer.from(await rawFile.arrayBuffer()));
    }
    if (!inputPath) {
      return NextResponse.json({ error: "Missing uploaded video file" }, { status: 400 });
    }

    const args = [
      "run",
      "python",
      "-m",
      "sam_3d_pose_estimation.subject_preview",
      "--video-input",
      inputPath,
      "--mode",
      "locate",
      "--auto-init-mode",
      "sam3",
      "--auto-select-strategy",
      autoSelectStrategy,
      "--sam3-text-prompts",
      sam3TextPrompts,
    ];
    if (preferredSecs) {
      args.push("--frame-secs", preferredSecs);
    }
    if (keptRanges) {
      args.push("--kept-ranges", keptRanges);
    }

    const locate = await runSubjectLocate(args, request.signal);

    if (locate.error) {
      return NextResponse.json({ error: locate.error, locate }, { status: 422 });
    }
    return NextResponse.json({ locate });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to locate subject: ${String(error)}` },
      { status: 500 },
    );
  } finally {
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    }
  }
}
