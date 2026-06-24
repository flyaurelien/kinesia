import { NextResponse } from "next/server";

import { createRunAnalysis } from "../../../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(
  request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const body = (await request.json().catch(() => ({}))) as {
      preset?: string;
      sensitivityPercent?: number;
      minDurationMs?: number;
      gapFillMs?: number;
    };
    const analysis = await createRunAnalysis(run, body);
    return NextResponse.json({ analysis }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to create analysis: ${String(error)}` },
      { status: 400 },
    );
  }
}
