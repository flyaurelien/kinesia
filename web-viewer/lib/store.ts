import { promises as fs } from "node:fs";
import path from "node:path";

export const RUN_MANIFEST_FILE = "run_manifest.json";
export const RUN_METADATA_FILE = "run_metadata.json";

// Repo root, resolved one level up from the Next.js app's working directory.
export function projectRoot(): string {
  return path.resolve(process.cwd(), "..");
}

// Default workspace directory where pipeline outputs (runs, datasets) live.
export function workspaceRoot(): string {
  return path.join(projectRoot(), "output");
}

// Resolve an env-supplied path: blank -> fallback; relative -> anchored at the repo root.
function configuredProjectPath(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim();
  if (!trimmed) {
    return fallback;
  }
  return path.isAbsolute(trimmed) ? trimmed : path.join(projectRoot(), trimmed);
}

// Directory holding run outputs; overridable via KINESIA_RUNS_ROOT.
export function runsRoot(): string {
  return configuredProjectPath(process.env.KINESIA_RUNS_ROOT, workspaceRoot());
}

// True when the runs root comes from KINESIA_RUNS_ROOT rather than the default.
export function usesConfiguredRunsRoot(): boolean {
  return Boolean(process.env.KINESIA_RUNS_ROOT?.trim());
}

// Directory holding uploaded input videos; overridable via KINESIA_UPLOADS_ROOT.
export function uploadsRoot(): string {
  return configuredProjectPath(process.env.KINESIA_UPLOADS_ROOT, path.join(projectRoot(), "input"));
}

// Pre-configurable output location kept for backward compatibility.
export function legacyOutputRoot(): string {
  return path.join(projectRoot(), "output");
}

// Validate an id used as a path segment, rejecting separators/null to prevent traversal.
export function ensureSafeId(raw: string): string {
  const value = raw.trim();
  if (!value || value.includes("/") || value.includes("\\") || value.includes("\0")) {
    throw new Error("Invalid identifier");
  }
  return value;
}

// Absolute directory for a single run, after validating the run id.
export function runDir(runIdRaw: string): string {
  return path.join(runsRoot(), ensureSafeId(runIdRaw));
}

// Read and parse JSON, returning null on any read/parse failure (missing file, bad JSON).
export async function readJsonIfExists<T>(filePath: string): Promise<T | null> {
  try {
    const text = await fs.readFile(filePath, "utf-8");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

// True only if the path exists and is a regular file (not a directory).
export async function fileExists(filePath: string): Promise<boolean> {
  try {
    const stat = await fs.stat(filePath);
    return stat.isFile();
  } catch {
    return false;
  }
}
