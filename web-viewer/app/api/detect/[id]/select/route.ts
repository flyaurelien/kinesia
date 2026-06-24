import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import { getDetectScratchDir } from "../../../../../lib/detect-jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type FrameLine = { f: number; dets: Array<{ id: number; b: number[]; s?: number }> };
type SubjectReq = { subjectId: number; trackIds: number[] };

// Materialize the chosen subject(s) as dense per-frame box tracks the run
// consumes (`sam3d run --subject-track-file`). Each subject is the union of its
// (possibly merged) track ids; on a frame where several of its tracks have a
// box, the highest-scoring one wins. Boxes are de-normalized [x,y,w,h] → pixel
// [x1,y1,x2,y2]. Accepts {subjects:[{subjectId,trackIds}]} or legacy {trackId}.
export async function POST(request: Request, { params }: { params: { id: string } }) {
  const id = decodeURIComponent(params.id);
  const dir = getDetectScratchDir(id);
  if (!dir) {
    return NextResponse.json({ error: "Bad detection id" }, { status: 400 });
  }
  const body = (await request.json().catch(() => ({}))) as {
    subjects?: unknown;
    trackId?: unknown;
  };

  let subjectReqs: SubjectReq[] = [];
  if (Array.isArray(body.subjects)) {
    subjectReqs = body.subjects
      .map((s) => s as { subjectId?: unknown; trackIds?: unknown })
      .filter((s) => Number.isFinite(Number(s.subjectId)) && Array.isArray(s.trackIds))
      .map((s) => ({
        subjectId: Number(s.subjectId),
        trackIds: (s.trackIds as unknown[]).map(Number).filter(Number.isFinite),
      }))
      .filter((s) => s.trackIds.length > 0);
  } else if (Number.isFinite(Number(body.trackId))) {
    subjectReqs = [{ subjectId: Number(body.trackId), trackIds: [Number(body.trackId)] }];
  }
  if (subjectReqs.length === 0) {
    return NextResponse.json({ error: "No subjects selected" }, { status: 400 });
  }

  let progress: { video_width?: number; video_height?: number } | null = null;
  try {
    progress = JSON.parse(await fs.readFile(path.join(dir, "progress.json"), "utf-8"));
  } catch {
    progress = null;
  }
  const width = Number(progress?.video_width) || 0;
  const height = Number(progress?.video_height) || 0;
  if (!width || !height) {
    return NextResponse.json({ error: "Detection has no frame dimensions yet" }, { status: 409 });
  }

  let raw = "";
  try {
    raw = await fs.readFile(path.join(dir, "frames.jsonl"), "utf-8");
  } catch {
    return NextResponse.json({ error: "No detections to select from" }, { status: 409 });
  }
  const lines: FrameLine[] = raw
    .split("\n")
    .filter(Boolean)
    .map((l) => {
      try {
        return JSON.parse(l) as FrameLine;
      } catch {
        return null;
      }
    })
    .filter((l): l is FrameLine => l !== null);

  const toPx = (b: number[]): [number, number, number, number] => {
    const [x, y, w, h] = b;
    return [
      Math.round(x * width),
      Math.round(y * height),
      Math.round((x + w) * width),
      Math.round((y + h) * height),
    ];
  };

  const subjects = subjectReqs.map((req, i) => {
    const wanted = new Set(req.trackIds);
    const frames: Record<string, [number, number, number, number]> = {};
    for (const line of lines) {
      let best: { b: number[]; s: number } | null = null;
      for (const d of line.dets) {
        if (!wanted.has(d.id) || d.b.length < 4) continue;
        const s = typeof d.s === "number" ? d.s : 1;
        if (!best || s > best.s) best = { b: d.b, s };
      }
      if (best) frames[String(line.f)] = toPx(best.b);
    }
    return { subjectId: req.subjectId, label: i + 1, frameCount: Object.keys(frames).length, frames };
  });

  const totalFrames = subjects.reduce((n, s) => n + s.frameCount, 0);
  if (totalFrames === 0) {
    return NextResponse.json({ error: "Chosen subject(s) have no detected frames" }, { status: 409 });
  }

  const trackFilePath = path.join(dir, "chosen_subject_track.json");
  await fs.writeFile(
    trackFilePath,
    JSON.stringify({ videoWidth: width, videoHeight: height, subjects }),
  );

  return NextResponse.json({
    trackFilePath,
    subjectCount: subjects.length,
    frameCounts: subjects.map((s) => s.frameCount),
  });
}
