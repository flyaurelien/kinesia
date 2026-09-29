import path from "node:path";
import fs from "node:fs";

/** The checkout root: the folder holding `pyproject.toml` above the web app. */
export function projectRoot(): string {
  const configured = process.env.KINESIA_ROOT?.trim();
  if (configured) return path.resolve(configured);
  let dir = process.cwd();
  for (let i = 0; i < 6; i++) {
    if (fs.existsSync(path.join(dir, "pyproject.toml")) && fs.existsSync(path.join(dir, "src", "kinesia"))) return dir;
    dir = path.dirname(dir);
  }
  return path.resolve(process.cwd(), "..");
}

export function runsRoot(): string {
  const configured = process.env.KINESIA_RUNS_ROOT?.trim();
  if (configured) return path.isAbsolute(configured) ? configured : path.join(projectRoot(), configured);
  return path.join(projectRoot(), "output");
}

const RUN_ID = /^[a-z0-9][a-z0-9-]{0,120}$/;

/** The folder of one analysis; throws on identifiers that could escape the runs root. */
export function runDir(id: string): string {
  if (!RUN_ID.test(id)) throw new Error(`invalid run id: ${id}`);
  return path.join(runsRoot(), id);
}

/** Resolve a file inside a run folder, refusing anything outside it. */
export function runFile(id: string, relative: string): string {
  const base = runDir(id);
  const target = path.resolve(base, relative);
  if (!target.startsWith(base + path.sep)) throw new Error("path outside the run");
  return target;
}

export function pythonExecutable(): string {
  const root = projectRoot();
  const venv = path.join(root, ".venv", "bin", "python");
  return fs.existsSync(venv) ? venv : "python3";
}
