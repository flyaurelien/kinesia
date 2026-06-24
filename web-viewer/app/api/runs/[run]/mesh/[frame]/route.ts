import { NextResponse } from "next/server";

import { readMeshFile } from "../../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { run: string; frame: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const frame = decodeURIComponent(params.frame);
    const data = await readMeshFile(run, frame);
    return new NextResponse(new Uint8Array(data), {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "public, max-age=86400, immutable",
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to stream mesh: ${String(error)}` },
      { status: 404 },
    );
  }
}
