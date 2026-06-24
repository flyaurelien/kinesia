// Streaming detection-job runner: spawns `detect_stream.py` (MLX SAM3 person
// detection + re-ID) as a long-lived background process that writes incremental
// results to a scratch dir, and exposes start / stop / scratch-dir lookup. The
// HTTP routes poll the scratch files directly; this module only owns process
// lifecycle. Mirrors lib/jobs.ts (detached process group, globalThis store that
// survives dev hot-reloads, shutdown cleanup) but much lighter.

import { execFile, spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import { signalProcessTree } from "./process-tree";
import { projectRoot, uploadsRoot } from "./store";

const execFileAsync = promisify(execFile);

// Confirm a pid is actually a live detect_stream process before signalling it —
// a stale progress.json can hold a pid that the OS has since recycled to an
// unrelated process (e.g. the dev server itself), so killing by pid blindly is
// dangerous.
async function isDetectStreamProcess(pid: number): Promise<boolean> {
  if (!Number.isInteger(pid) || pid <= 1) return false;
  try {
    const { stdout } = await execFileAsync("ps", ["-p", String(pid), "-o", "command="]);
    return stdout.includes("sam_3d_pose_estimation.detect_stream");
  } catch {
    return false; // no such process
  }
}

export type DetectJobStatus = "running" | "completed" | "stopped" | "error";

export type DetectJob = {
  id: string;
  status: DetectJobStatus;
  outDir: string;
  prompt: string;
  stride: number;
  createdAt: string;
};

type DetectStore = {
  jobs: Map<string, DetectJob>;
  processes: Map<string, ChildProcess>;
};

const globalStore = globalThis as typeof globalThis & {
  __kinesiaDetectStore?: DetectStore;
  __kinesiaDetectShutdown?: boolean;
};

function store(): DetectStore {
  if (!globalStore.__kinesiaDetectStore) {
    globalStore.__kinesiaDetectStore = { jobs: new Map(), processes: new Map() };
  }
  installShutdownCleanup();
  return globalStore.__kinesiaDetectStore;
}

// On server shutdown, terminate every live detection process group so an
// abandoned SAM3 scan never lingers.
function installShutdownCleanup(): void {
  if (globalStore.__kinesiaDetectShutdown) return;
  globalStore.__kinesiaDetectShutdown = true;
  process.once("exit", () => {
    const current = globalStore.__kinesiaDetectStore;
    if (!current) return;
    for (const child of current.processes.values()) {
      if (child.pid == null) continue;
      signalProcessTree(child, "SIGTERM");
    }
  });
}

// Root for per-job scratch output, alongside the staged uploads.
export function detectScratchRoot(): string {
  return path.join(uploadsRoot(), ".detect");
}

// Resolve (and validate) a job's scratch dir from an id. Returns null for a
// malformed id so a poll/stop route can 404 instead of touching arbitrary paths.
export function getDetectScratchDir(id: string): string | null {
  if (!/^[a-f0-9]{16,40}$/.test(id)) return null;
  return path.join(detectScratchRoot(), id);
}

export function getDetectJob(id: string): DetectJob | null {
  return store().jobs.get(id) ?? null;
}

// Signal a job's whole process tree (uv + python grandchild) to terminate.
function signalGroup(child: ChildProcess, sig: NodeJS.Signals): void {
  signalProcessTree(child, sig);
}

// Terminate every other detection process before starting a new one so heavy
// SAM3 jobs never pile up and thrash the GPU/RAM. Kills both store-tracked
// children AND orphans (e.g. a job spawned before a dev-server restart) by
// reading the pid each job writes into its progress.json.
async function killOtherDetectProcesses(exceptId?: string): Promise<void> {
  for (const [id, child] of store().processes) {
    if (id === exceptId) continue;
    signalGroup(child, "SIGTERM");
    setTimeout(() => signalGroup(child, "SIGKILL"), 1500).unref();
    store().processes.delete(id);
    const job = store().jobs.get(id);
    if (job && job.status === "running") job.status = "stopped";
  }
  // Orphans not in this process's store (e.g. spawned before a dev-server
  // restart). Kill ONLY after ps-confirming the pid is really a detect_stream
  // process — a stale "running" progress.json can hold a recycled pid.
  const root = detectScratchRoot();
  const dirs = await fs.readdir(root).catch(() => [] as string[]);
  await Promise.all(
    dirs.map(async (d) => {
      if (d === exceptId) return;
      try {
        const raw = await fs.readFile(path.join(root, d, "progress.json"), "utf-8");
        const p = JSON.parse(raw) as { pid?: number; status?: string };
        const pid = Number(p.pid);
        if (!pid || !(p.status === "running" || p.status === "loading" || p.status === "starting")) {
          return;
        }
        if (await isDetectStreamProcess(pid)) {
          try {
            process.kill(pid, "SIGTERM");
          } catch {
            // already gone
          }
        }
      } catch {
        // no/invalid progress.json — skip
      }
    }),
  );
}

export async function startDetectJob(opts: {
  inputPath: string;
  prompt: string;
  stride: number;
  minDurationSec?: number;
}): Promise<DetectJob> {
  // Single detection at a time — kill any other running job first.
  await killOtherDetectProcesses();
  const id = randomUUID().replace(/-/g, "");
  const outDir = path.join(detectScratchRoot(), id);
  await fs.mkdir(outDir, { recursive: true });

  const job: DetectJob = {
    id,
    status: "running",
    outDir,
    prompt: opts.prompt,
    stride: opts.stride,
    createdAt: new Date().toISOString(),
  };

  const args = [
    "run",
    "python",
    "-m",
    "sam_3d_pose_estimation.detect_stream",
    "--video-input",
    opts.inputPath,
    "--out-dir",
    outDir,
    "--prompt",
    opts.prompt,
    "--stride",
    String(opts.stride),
    "--min-duration-sec",
    String(opts.minDurationSec ?? 1.0),
  ];

  const child = spawn("uv", args, {
    cwd: projectRoot(),
    env: {
      ...process.env,
      PYTHONPATH: path.join(projectRoot(), "src"),
      // MLX runs natively, but keep the MPS fallback flag set to match the rest
      // of the stack (harmless) and the offline flags so weights load from cache.
      PYTORCH_ENABLE_MPS_FALLBACK: "1",
      SAM3D_MHR_MODE: process.env.SAM3D_MHR_MODE || "native",
      HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "1",
      TRANSFORMERS_OFFLINE: process.env.TRANSFORMERS_OFFLINE || "1",
    },
    stdio: ["ignore", "ignore", "pipe"],
    detached: true,
  });

  const recentErr: string[] = [];
  child.stderr?.on("data", (chunk: Buffer) => {
    const text = chunk.toString();
    recentErr.push(text);
    if (recentErr.length > 20) recentErr.shift();
  });
  child.on("close", (code) => {
    store().processes.delete(id);
    const current = store().jobs.get(id);
    if (!current) return;
    // The python process writes the authoritative status into progress.json;
    // only flip to "error" here when it died non-zero without a clean stop.
    if (current.status === "running") {
      current.status = code === 0 ? "completed" : "error";
    }
  });

  store().jobs.set(id, job);
  store().processes.set(id, child);
  return job;
}

export function stopDetectJob(id: string): boolean {
  const child = store().processes.get(id);
  const job = store().jobs.get(id);
  if (job && job.status === "running") job.status = "stopped";
  if (!child) return false;
  signalGroup(child, "SIGTERM");
  setTimeout(() => signalGroup(child, "SIGKILL"), 2_000).unref();
  return true;
}
