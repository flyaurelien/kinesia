import { NextResponse } from "next/server";

import { getRunDetail } from "../../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { run: string; analysis: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const analysis = decodeURIComponent(params.analysis);
    const detail = await getRunDetail(run, analysis);
    return NextResponse.json({ run: detail });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read analysis detail: ${String(error)}` },
      { status: 404 },
    );
  }
}