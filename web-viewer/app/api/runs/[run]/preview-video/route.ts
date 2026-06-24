import { NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";
import { mkdir, open, readFile, rename, stat, unlink } from "node:fs/promises";

import { resolveRunVideoFile } from "../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Per-run subfolder holding derived web-only assets (kept out of the run's primary outputs).
const WEB_CACHE_DIR = ".web_cache";
const WEB_PANELS_FILE = "preview_web_panels_v1.mp4";
const VIDEO_CACHE_CONTROL = "public, max-age=3600";

// Run ffmpeg with the given args, rejecting (with the tail of stderr) on any non-zero exit.
async function runFfmpeg(args: string[]): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    ffmpeg.stderr.on("data", (chunk: Buffer | string) => {
      stderr += String(chunk);
      if (stderr.length > 8000) {
        stderr = stderr.slice(-8000);
      }
    });
    ffmpeg.on("error", reject);
    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`ffmpeg failed (code=${code}): ${stderr.trim() || "unknown error"}`));
    });
  });
}

// Poll for another process to finish writing the cache file (newer than minMtimeMs and non-trivial size).
async function waitForFreshFile(filePath: string, minMtimeMs: number): Promise<boolean> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const st = await stat(filePath).catch(() => null);
    if (st && st.isFile() && st.size > 1024 && st.mtimeMs >= minMtimeMs) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  return false;
}

// Build (and cache) a portrait side-by-side panels preview from the source video, transcoding
// only when the cache is missing/stale. A lock file serializes concurrent requests; on any
// failure (or losing the lock race) we fall back to streaming the original source.
async function ensureWebPanelsPreview(sourcePath: string): Promise<string> {
  const sourceStat = await stat(sourcePath);
  const runDir = path.dirname(sourcePath);
  const cacheDir = path.join(runDir, WEB_CACHE_DIR);
  const outPath = path.join(cacheDir, WEB_PANELS_FILE);
  const lockPath = `${outPath}.lock`;

  const cached = await stat(outPath).catch(() => null);
  if (cached && cached.isFile() && cached.size > 1024 && cached.mtimeMs >= sourceStat.mtimeMs) {
    return outPath;
  }

  await mkdir(cacheDir, { recursive: true });

  let lockFile: Awaited<ReturnType<typeof open>> | null = null;
  try {
    lockFile = await open(lockPath, "wx");
  } catch {
    const ready = await waitForFreshFile(outPath, sourceStat.mtimeMs);
    return ready ? outPath : sourcePath;
  }

  const tmpPath = `${outPath}.tmp-${process.pid}-${Date.now()}.mp4`;
  try {
    await runFfmpeg([
      "-y",
      "-i",
      sourcePath,
      "-filter_complex",
      [
        "[0:v]crop=w=iw/2:h=ih/2:x=iw/2:y=0,scale=540:960:flags=lanczos[left]",
        "[0:v]crop=w=iw/2:h=ih/2:x=0:y=0,scale=540:960:flags=lanczos[right]",
        "[left][right]hstack=inputs=2[v]",
      ].join(";"),
      "-map",
      "[v]",
      "-an",
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "30",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      tmpPath,
    ]);
    await rename(tmpPath, outPath);
    return outPath;
  } catch {
    await unlink(tmpPath).catch(() => undefined);
    return sourcePath;
  } finally {
    if (lockFile) {
      await lockFile.close().catch(() => undefined);
    }
    await unlink(lockPath).catch(() => undefined);
  }
}

// Stream a run's preview video, honoring HTTP range requests so the player can seek.
// The optional `variant=web-panels` query selects the cached side-by-side panels render.
export async function GET(
  request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const { filePath: sourcePath, contentType } = await resolveRunVideoFile(run, "preview");
    const url = new URL(request.url);
    const variant = url.searchParams.get("variant");
    const filePath =
      variant === "web-panels"
        ? await ensureWebPanelsPreview(sourcePath)
        : sourcePath;
    const effectiveContentType = filePath.endsWith(".mp4") ? "video/mp4" : contentType;
    const fileStat = await stat(filePath);
    const fileSize = fileStat.size;
    const range = request.headers.get("range");

    if (range) {
      const match = /bytes=(\d*)-(\d*)/.exec(range);
      if (!match) {
        return new NextResponse(null, { status: 416 });
      }
      let start = match[1] ? Number(match[1]) : 0;
      let end = match[2] ? Number(match[2]) : fileSize - 1;
      if (!Number.isFinite(start) || !Number.isFinite(end)) {
        return new NextResponse(null, { status: 416 });
      }
      start = Math.max(0, Math.min(start, fileSize - 1));
      end = Math.max(start, Math.min(end, fileSize - 1));

      const chunkSize = end - start + 1;
      const file = await open(filePath, "r");
      const chunk = Buffer.allocUnsafe(chunkSize);
      try {
        await file.read(chunk, 0, chunkSize, start);
      } finally {
        await file.close();
      }
      return new NextResponse(chunk, {
        status: 206,
        headers: {
          "content-type": effectiveContentType,
          "accept-ranges": "bytes",
          "content-range": `bytes ${start}-${end}/${fileSize}`,
          "content-length": String(chunkSize),
          "cache-control": VIDEO_CACHE_CONTROL,
        },
      });
    }

    const wholeFile = await readFile(filePath);
    return new NextResponse(wholeFile, {
      status: 200,
      headers: {
        "content-type": effectiveContentType,
        "accept-ranges": "bytes",
        "content-length": String(fileSize),
        "cache-control": VIDEO_CACHE_CONTROL,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to stream preview video: ${String(error)}` },
      { status: 404 },
    );
  }
}
