import { NextResponse } from "next/server";

import { listRuns } from "../../../lib/runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const runs = await listRuns();
    return NextResponse.json({ runs });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to list runs: ${String(error)}` },
      { status: 500 },
    );
  }
}
