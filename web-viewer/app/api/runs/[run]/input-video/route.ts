import { NextResponse } from "next/server";
import { open, readFile, stat } from "node:fs/promises";

import { resolveRunVideoFile } from "../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const { filePath, contentType } = await resolveRunVideoFile(run, "input");
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
          "content-type": contentType,
          "accept-ranges": "bytes",
          "content-range": `bytes ${start}-${end}/${fileSize}`,
          "content-length": String(chunkSize),
          "cache-control": "public, max-age=60",
        },
      });
    }

    const wholeFile = await readFile(filePath);
    return new NextResponse(wholeFile, {
      status: 200,
      headers: {
        "content-type": contentType,
        "accept-ranges": "bytes",
        "content-length": String(fileSize),
        "cache-control": "public, max-age=60",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to stream input video: ${String(error)}` },
      { status: 404 },
    );
  }
}
