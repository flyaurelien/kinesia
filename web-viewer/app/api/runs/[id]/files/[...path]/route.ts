import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";

import { runFile } from "@/lib/server/paths";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const ALLOWED = /^(video\.mp4|poster\.jpg|scene\/(scene\.json|mesh\.bin|people\.bin|overlay\.json|metrics\.json))$/;

const TYPES: Record<string, string> = {
  ".mp4": "video/mp4",
  ".jpg": "image/jpeg",
  ".json": "application/json",
  ".bin": "application/octet-stream",
};

/** Serve an analysis file, with byte ranges so the video can seek. */
export async function GET(request: Request, { params }: { params: { id: string; path: string[] } }) {
  const relative = params.path.join("/");
  if (!ALLOWED.test(relative)) return new Response("Not found", { status: 404 });
  let file: string;
  try {
    file = runFile(params.id, relative);
  } catch {
    return new Response("Not found", { status: 404 });
  }
  let stat: fs.Stats;
  try {
    stat = fs.statSync(file);
  } catch {
    return new Response("Not found", { status: 404 });
  }
  const headers: Record<string, string> = {
    "content-type": TYPES[path.extname(file)] ?? "application/octet-stream",
    "accept-ranges": "bytes",
    "cache-control": relative.startsWith("scene/") ? "no-cache" : "private, max-age=3600",
    "last-modified": stat.mtime.toUTCString(),
  };
  const range = request.headers.get("range");
  const match = range?.match(/^bytes=(\d*)-(\d*)$/);
  if (match && (match[1] || match[2])) {
    let start = match[1] ? Number(match[1]) : Math.max(0, stat.size - Number(match[2]));
    let end = match[1] && match[2] ? Number(match[2]) : stat.size - 1;
    end = Math.min(end, stat.size - 1);
    if (start > end || start >= stat.size) {
      return new Response(null, { status: 416, headers: { "content-range": `bytes */${stat.size}` } });
    }
    const stream = fs.createReadStream(file, { start, end });
    return new Response(Readable.toWeb(stream) as ReadableStream, {
      status: 206,
      headers: { ...headers, "content-range": `bytes ${start}-${end}/${stat.size}`, "content-length": String(end - start + 1) },
    });
  }
  const stream = fs.createReadStream(file);
  return new Response(Readable.toWeb(stream) as ReadableStream, {
    status: 200,
    headers: { ...headers, "content-length": String(stat.size) },
  });
}
