import { NextResponse } from "next/server";

import { readMeshVertices } from "../../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { run: string; frame: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const frame = decodeURIComponent(params.frame);
    const payload = await readMeshVertices(run, frame);
    return new NextResponse(new Uint8Array(payload.data), {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "public, max-age=86400, immutable",
        "x-kinesia-vertex-count": String(payload.vertexCount),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to stream mesh vertices: ${String(error)}` },
      { status: 404 },
    );
  }
}
