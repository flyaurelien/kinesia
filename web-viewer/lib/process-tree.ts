import { spawnSync, type ChildProcess } from "node:child_process";

// Cross-platform signalling of a DETACHED child process AND its descendants
// (the `uv` launcher → Python → its children). The pipeline spawns those jobs
// with `detached: true`; on POSIX that puts them in their own process group, so a
// signal to the negative pid reaches the whole tree (falling back to the direct
// child). Windows has no POSIX process groups or job-control signals: a
// terminating signal maps to `taskkill /T /F` (kills the whole tree), and the
// pause/resume signals (SIGSTOP/SIGCONT) have no equivalent and are skipped.
// Returns true if a signal was delivered. The POSIX path is intentionally
// identical to the previous inline code.
export function signalProcessTree(child: ChildProcess, signal: NodeJS.Signals): boolean {
  const pid = child.pid;
  if (pid == null) return false;

  if (process.platform === "win32") {
    if (signal === "SIGSTOP" || signal === "SIGCONT") return false; // no job control on Windows
    try {
      spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
      return true;
    } catch {
      try {
        child.kill();
        return true;
      } catch {
        return false;
      }
    }
  }

  try {
    process.kill(-pid, signal);
    return true;
  } catch {
    try {
      child.kill(signal);
      return true;
    } catch {
      return false;
    }
  }
}
