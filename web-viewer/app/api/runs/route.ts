import fs from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import { createRun, deleteRun, listRuns } from "@/lib/server/store";
import { startProcessing } from "@/lib/server/worker";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const VIDEO_TYPES = /\.(mp4|mov|m4v|avi|mkv|webm)$/i;

export async function GET() {
  return Response.json({ runs: listRuns() });
}

/**
 * Upload a video and start its analysis. The body is the raw file; options
 * travel in the query string (`name`, `filename`, `max_people`, `pools`).
 */
export async function POST(request: Request) {
  const url = new URL(request.url);
  const filename = url.searchParams.get("filename") ?? "video.mp4";
  if (!VIDEO_TYPES.test(filename)) {
    return Response.json({ error: "Unsupported file type: use MP4, MOV, M4V, AVI, MKV or WebM." }, { status: 400 });
  }
  if (!request.body) return Response.json({ error: "Empty upload." }, { status: 400 });
  const name = (url.searchParams.get("name") ?? "").trim() || filename.replace(/\.[^.]+$/, "");
  const maxPeople = Number(url.searchParams.get("max_people") ?? "") || undefined;
  const pools = (url.searchParams.get("pools") ?? "").split(",").map((p) => p.trim()).filter(Boolean);

  const { id, sourcePath } = createRun({ name, originalName: filename, maxPeople, gpuPools: pools });
  try {
    await pipeline(Readable.fromWeb(request.body as never), fs.createWriteStream(sourcePath));
  } catch (error) {
    deleteRun(id);
    return Response.json({ error: `Upload interrupted: ${String(error)}` }, { status: 500 });
  }
  if (fs.statSync(sourcePath).size === 0) {
    deleteRun(id);
    return Response.json({ error: "The uploaded file is empty." }, { status: 400 });
  }
  startProcessing(id);
  return Response.json({ id }, { status: 201 });
}
