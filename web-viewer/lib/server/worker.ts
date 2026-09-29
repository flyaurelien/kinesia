import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { projectRoot, pythonExecutable, runDir } from "./paths";

function environment(): NodeJS.ProcessEnv {
  const root = projectRoot();
  return {
    ...process.env,
    KINESIA_ROOT: root,
    PYTHONPATH: path.join(root, "src"),
    PYTHONUNBUFFERED: "1",
  };
}

/** Start `kinesia process <id>` detached: it outlives this server and resumes on re-run. */
export function startProcessing(id: string): number {
  const dir = runDir(id);
  const log = fs.openSync(path.join(dir, "orchestrator.log"), "a");
  const child = spawn(pythonExecutable(), ["-m", "kinesia.cli", "process", id], {
    cwd: projectRoot(),
    env: environment(),
    detached: true,
    stdio: ["ignore", log, log],
  });
  child.unref();
  fs.closeSync(log);
  return child.pid ?? 0;
}

/** `kinesia cancel` waits for the local worker to stop before deleting the job. */
export const CANCEL_TIMEOUT_MS = 150_000;

/** Run a short `kinesia` command and return its output. */
export function runCli(args: string[], timeoutMs = 60_000): Promise<{ code: number; output: string }> {
  return new Promise((resolve) => {
    const child = spawn(pythonExecutable(), ["-m", "kinesia.cli", ...args], {
      cwd: projectRoot(),
      env: environment(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.on("data", (chunk) => (output += chunk));
    child.stderr.on("data", (chunk) => (output += chunk));
    const timer = setTimeout(() => child.kill("SIGTERM"), timeoutMs);
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? 1, output });
    });
  });
}
