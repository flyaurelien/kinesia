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

// Shape of the JSON the Python subject_preview module prints on its last stdout line.
type SubjectPreviewOutput = {
  ok: boolean;
  error?: string;
  fps?: number;
  frame_index?: number;
  frame_sec?: number;
  total_frames?: number;
  video_width?: number;
  video_height?: number;
  detection?: {
    xyxy: number[];
    box: { x: number; y: number; width: number; height: number };
  } | null;
  info?: Record<string, unknown>;
};

// Parse a finite form value to a number, clamped to >= min, falling back when absent/invalid.
function parseNumber(raw: FormDataEntryValue | null, fallback: number, min = 0): number {
  if (typeof raw !== "string") {
    return fallback;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? Math.max(min, value) : fallback;
}

// Accept a form value only if it is one of the allowed options, else use the fallback.
function normalizeEnum(raw: FormDataEntryValue | null, allowed: string[], fallback: string): string {
  return typeof raw === "string" && allowed.includes(raw) ? raw : fallback;
}

// Normalize a comma-separated prompt string: trim, drop blanks, cap at 8 entries, rejoin.
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

// Hard ceiling for a single-frame detection; a hung detector is killed and
// reported rather than left to run. Kept under the route's implicit budget.
const PREVIEW_TIMEOUT_MS = 120_000;

// Spawn the Python subject_preview module via `uv` and resolve with its parsed
// JSON output. The child runs in its own process group and is killed if the
// client aborts the request or the detection times out, so an abandoned preview
// never leaves a heavy SAM3 process (uv + python) alive to pile up on retries.
function runSubjectPreview(args: string[], signal?: AbortSignal): Promise<SubjectPreviewOutput> {
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
    }, PREVIEW_TIMEOUT_MS);
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
        resolve({ ok: false, error: `Subject detection timed out after ${Math.round(PREVIEW_TIMEOUT_MS / 1000)}s.` });
        return;
      }
      if (aborted) {
        resolve({ ok: false, error: "Subject detection cancelled." });
        return;
      }
      const lines = stdout.trim().split(/\r?\n/g).filter(Boolean);
      const payload = lines.at(-1) ?? "";
      try {
        const parsed = JSON.parse(payload) as SubjectPreviewOutput;
        if (code !== 0 && !parsed.error) {
          parsed.error = stderr.trim() || `subject preview exited with code ${code}`;
        }
        resolve(parsed);
      } catch (error) {
        reject(new Error(`Failed to parse subject preview output: ${String(error)} ${stderr.trim()}`));
      }
    });
  });
}

// Run subject detection on a single frame of an uploaded/staged video and return the box (or 422).
export async function POST(request: Request) {
  let tempDir: string | null = null;
  try {
    const formData = await request.formData();
    const rawFile = formData.get("video");
    const stagedUploadIdRaw = formData.get("stagedUploadId");
    const frameSec = parseNumber(formData.get("frameSec"), 0, 0);
    const autoInitMode = normalizeEnum(formData.get("autoInitMode"), ["smart", "sam3"], "sam3");
    const autoSelectStrategy = normalizeEnum(
      formData.get("autoSelectStrategy"),
      ["patient", "largest", "leftmost", "rightmost", "center", "tightest"],
      "patient",
    );
    // SAM3 is a promptable text-conditioned detector, so a user-provided
    // description ("the patient", "the guy with the blue shirt", ...) is the
    // most reliable way to disambiguate. We default to "the patient" but
    // forward whatever the UI sends through.
    const sam3TextPrompts = normalizePromptList(formData.get("sam3TextPrompts"), "the patient");

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
      tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "kinesia-subject-preview-"));
      inputPath = path.join(tempDir, `input${path.extname(rawFile.name).toLowerCase() || ".mp4"}`);
      await fs.writeFile(inputPath, Buffer.from(await rawFile.arrayBuffer()));
    }
    if (!inputPath) {
      return NextResponse.json({ error: "Missing uploaded video file" }, { status: 400 });
    }

    const preview = await runSubjectPreview([
      "run",
      "python",
      "-m",
      "sam_3d_pose_estimation.subject_preview",
      "--video-input",
      inputPath,
      "--frame-sec",
      frameSec.toFixed(3),
      "--auto-init-mode",
      autoInitMode,
      "--auto-select-strategy",
      autoSelectStrategy,
      "--sam3-text-prompts",
      sam3TextPrompts,
    ], request.signal);

    if (!preview.ok || !preview.detection) {
      return NextResponse.json(
        { error: preview.error ?? "No patient subject detected on this frame.", preview },
        { status: 422 },
      );
    }
    return NextResponse.json({ preview });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to preview subject detection: ${String(error)}` },
      { status: 500 },
    );
  } finally {
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    }
  }
}
