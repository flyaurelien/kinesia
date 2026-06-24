import { NextResponse } from "next/server";
import { spawn } from "node:child_process";
import path from "node:path";
import { mkdir, open, readFile, stat, unlink } from "node:fs/promises";

import { projectRoot, runDir } from "../../../../../lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The "Tracking box" MP4 lives in the run's OWN output folder (the user's "dossier
// de sortie"), generated on demand from run_metadata.json + the source video.
const OUTPUT_FILE = "tracking_box.mp4";
const VIDEO_CACHE_CONTROL = "public, max-age=3600";

// Spawn the Python renderer (source video + per-frame box → H.264 MP4), resolving
// when it finishes or rejecting with the tail of stderr.
async function renderTrackingVideo(dir: string, outPath: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      "uv",
      ["run", "python", "-m", "sam_3d_pose_estimation.render_tracking_video",
        "--run-dir", dir, "--out", outPath],
      {
        cwd: projectRoot(),
        env: { ...process.env, PYTHONPATH: path.join(projectRoot(), "src") },
        stdio: ["ignore", "ignore", "pipe"],
      },
    );
    let stderr = "";
    child.stderr.on("data", (c: Buffer | string) => {
      stderr += String(c);
      if (stderr.length > 8000) stderr = stderr.slice(-8000);
    });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`renderer failed (code=${code}): ${stderr.trim()}`)),
    );
  });
}

// Wait (poll) for a concurrent render to finish, up to ~5 min.
async function waitForFile(filePath: string, minMtimeMs: number): Promise<boolean> {
  for (let i = 0; i < 2000; i += 1) {
    const st = await stat(filePath).catch(() => null);
    if (st?.isFile() && st.size > 1024 && st.mtimeMs >= minMtimeMs) return true;
    await new Promise((r) => setTimeout(r, 150));
  }
  return false;
}

// Generate the tracking MP4 if missing or older than run_metadata.json; a lock
// file serializes concurrent requests. Returns the absolute path.
async function ensureTrackingVideo(dir: string): Promise<string> {
  const outPath = path.join(dir, OUTPUT_FILE);
  const metaStat = await stat(path.join(dir, "run_metadata.json"));
  const cached = await stat(outPath).catch(() => null);
  if (cached?.isFile() && cached.size > 1024 && cached.mtimeMs >= metaStat.mtimeMs) {
    return outPath;
  }

  const lockPath = `${outPath}.lock`;
  let lock: Awaited<ReturnType<typeof open>> | null = null;
  try {
    lock = await open(lockPath, "wx");
  } catch {
    // Another request is rendering — wait for it.
    if (await waitForFile(outPath, metaStat.mtimeMs)) return outPath;
    throw new Error("Timed out waiting for tracking video render");
  }
  try {
    await renderTrackingVideo(dir, outPath);
    return outPath;
  } finally {
    await lock.close().catch(() => undefined);
    await unlink(lockPath).catch(() => undefined);
  }
}

export async function GET(request: Request, { params }: { params: { run: string } }) {
  try {
    const run = decodeURIComponent(params.run);
    const dir = runDir(run);
    await mkdir(dir, { recursive: true }).catch(() => undefined);
    const filePath = await ensureTrackingVideo(dir);

    const fileStat = await stat(filePath);
    const fileSize = fileStat.size;
    const range = request.headers.get("range");

    if (range) {
      const match = /bytes=(\d*)-(\d*)/.exec(range);
      if (!match) return new NextResponse(null, { status: 416 });
      let start = match[1] ? Number(match[1]) : 0;
      let end = match[2] ? Number(match[2]) : fileSize - 1;
      if (!Number.isFinite(start) || !Number.isFinite(end)) return new NextResponse(null, { status: 416 });
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
          "content-type": "video/mp4",
          "accept-ranges": "bytes",
          "content-range": `bytes ${start}-${end}/${fileSize}`,
          "content-length": String(chunkSize),
          "cache-control": VIDEO_CACHE_CONTROL,
        },
      });
    }

    const whole = await readFile(filePath);
    return new NextResponse(whole, {
      status: 200,
      headers: {
        "content-type": "video/mp4",
        "accept-ranges": "bytes",
        "content-length": String(fileSize),
        "content-disposition": `attachment; filename="${OUTPUT_FILE}"`,
        "cache-control": VIDEO_CACHE_CONTROL,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to render tracking video: ${String(error)}` },
      { status: 500 },
    );
  }
}
