import { NextResponse } from "next/server";

import { readMeshFaces } from "../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const payload = await readMeshFaces(run);
    return new NextResponse(new Uint8Array(payload.data), {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "public, max-age=86400, immutable",
        "x-kinesia-vertex-count": String(payload.vertexCount),
        "x-kinesia-face-count": String(payload.faceCount),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to stream mesh faces: ${String(error)}` },
      { status: 404 },
    );
  }
}
