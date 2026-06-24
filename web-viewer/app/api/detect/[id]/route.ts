import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import { getDetectScratchDir, stopDetectJob } from "../../../../lib/detect-jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type FrameLine = { f: number; t: number; dets: Array<{ id: number; b: number[]; s: number }> };

async function readJsonSafe<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, "utf-8")) as T;
  } catch {
    return null;
  }
}

// Poll a detection job: latest progress + tracks + the frame lines with index
// >= sinceFrame (the client passes the highest frame it already has so each poll
// only ships new detections).
export async function GET(request: Request, { params }: { params: { id: string } }) {
  const id = decodeURIComponent(params.id);
  const dir = getDetectScratchDir(id);
  if (!dir) {
    return NextResponse.json({ error: "Bad detection id" }, { status: 400 });
  }
  const sinceFrame = Number(new URL(request.url).searchParams.get("sinceFrame") ?? "0");
  const since = Number.isFinite(sinceFrame) ? sinceFrame : 0;

  const progress = await readJsonSafe<Record<string, unknown>>(path.join(dir, "progress.json"));
  if (!progress) {
    // Job just started; files not written yet.
    return NextResponse.json({ status: "starting", processed: 0, frames: [], tracks: [] });
  }
  const tracksDoc = await readJsonSafe<{ tracks: unknown[] }>(path.join(dir, "tracks.json"));

  let frames: FrameLine[] = [];
  try {
    const raw = await fs.readFile(path.join(dir, "frames.jsonl"), "utf-8");
    frames = raw
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line) as FrameLine;
        } catch {
          return null;
        }
      })
      .filter((f): f is FrameLine => f !== null && f.f >= since);
  } catch {
    frames = [];
  }

  return NextResponse.json({
    status: progress.status ?? "running",
    processed: progress.processed ?? 0,
    totalToProcess: progress.total_to_process ?? 0,
    totalFrames: progress.total_frames ?? 0,
    lastFrame: progress.last_frame ?? -1,
    videoWidth: progress.video_width ?? 0,
    videoHeight: progress.video_height ?? 0,
    fps: progress.fps ?? 30,
    stride: progress.stride ?? 1,
    tracks: tracksDoc?.tracks ?? [],
    frames,
  });
}

// Stop a running detection job (used by the "Stop" button / re-prompt).
export async function DELETE(_request: Request, { params }: { params: { id: string } }) {
  const id = decodeURIComponent(params.id);
  const dir = getDetectScratchDir(id);
  if (!dir) {
    return NextResponse.json({ error: "Bad detection id" }, { status: 400 });
  }
  stopDetectJob(id);
  return NextResponse.json({ ok: true });
}
