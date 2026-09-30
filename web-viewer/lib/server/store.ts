import fs from "node:fs";
import path from "node:path";

import type { RunStatus, RunSummary } from "../runs";
import { isActive } from "../runs";
import { runDir, runsRoot } from "./paths";

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as T;
  } catch {
    return null;
  }
}

function alive(pid: number | undefined): boolean {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function readRun(id: string): RunSummary | null {
  const dir = runDir(id);
  const meta = readJson<{ id: string; name: string; created_at: string; original_name?: string; video?: RunSummary["video"] }>(
    path.join(dir, "run.json"),
  );
  if (!meta) return null;
  const status = readJson<RunStatus>(path.join(dir, "status.json")) ?? { state: "new" };
  const scene = readJson<{ people: unknown[] }>(path.join(dir, "scene", "scene.json"));
  return {
    id,
    name: meta.name,
    created_at: meta.created_at,
    original_name: meta.original_name,
    video: meta.video ?? null,
    status,
    worker_alive: isActive(status.state) && alive(status.pid),
    people: scene ? scene.people.length : null,
    has_poster: fs.existsSync(path.join(dir, "poster.jpg")),
    has_scene: scene !== null,
  };
}

export function listRuns(): RunSummary[] {
  const root = runsRoot();
  if (!fs.existsSync(root)) return [];
  const runs: RunSummary[] = [];
  for (const name of fs.readdirSync(root)) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(name)) continue;
    try {
      const run = readRun(name);
      if (run) runs.push(run);
    } catch {
      /* not an analysis folder */
    }
  }
  return runs.sort((a, b) => b.created_at.localeCompare(a.created_at));
}

function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "video"
  );
}

function stamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

export type NewRunOptions = { name: string; originalName: string; maxPeople?: number };

/** Create the folder of a new analysis; the caller streams the video into `sourcePath`. */
export function createRun(options: NewRunOptions): { id: string; sourcePath: string } {
  const now = new Date();
  const id = `${slug(options.name)}-${stamp(now)}`;
  const dir = runDir(id);
  fs.mkdirSync(dir, { recursive: true });
  const extension = (path.extname(options.originalName) || ".mp4").toLowerCase();
  const source = `source${extension}`;
  const request: Record<string, unknown> = {};
  if (options.maxPeople) request.max_people = options.maxPeople;
  fs.writeFileSync(
    path.join(dir, "run.json"),
    JSON.stringify({ id, name: options.name, source, original_name: options.originalName, created_at: now.toISOString(), request }, null, 1),
  );
  fs.writeFileSync(path.join(dir, "status.json"), JSON.stringify({ state: "new", history: [] }));
  return { id, sourcePath: path.join(dir, source) };
}

export function renameRun(id: string, name: string): void {
  const file = path.join(runDir(id), "run.json");
  const meta = readJson<Record<string, unknown>>(file);
  if (!meta) throw new Error("unknown run");
  meta.name = name;
  fs.writeFileSync(file, JSON.stringify(meta, null, 1));
}

export function deleteRun(id: string): void {
  fs.rmSync(runDir(id), { recursive: true, force: true });
}
