// Serve a reconstructed scene object (a .glb mesh, or the .json that says where
// it sits) from a run's `scene/` directory.
//
// These come from SAM 3D Objects, which reconstructs a static object ONCE for
// the whole clip rather than per frame — the object does not move, so there is
// nothing to stream. One file, cached hard.
import { NextResponse } from "next/server";
import path from "node:path";
import { readFile } from "node:fs/promises";

import { runSceneFilePath } from "../../../../../../lib/runs";

const CONTENT_TYPES: Record<string, string> = {
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".json": "application/json",
  ".stl": "model/stl",
};

export async function GET(
  _request: Request,
  { params }: { params: { run: string; file: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const file = decodeURIComponent(params.file);
    const filePath = await runSceneFilePath(run, file);
    const data = await readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    return new NextResponse(data, {
      status: 200,
      headers: {
        "Content-Type": CONTENT_TYPES[ext] ?? "application/octet-stream",
        "Content-Length": String(data.byteLength),
        // Static for the life of the run: safe to cache aggressively.
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown error";
    const missing = message.includes("ENOENT") || message.includes("not found");
    return NextResponse.json({ error: message }, { status: missing ? 404 : 500 });
  }
}
