import { NextResponse } from "next/server";

import { getLatestDatasetEvaluation } from "../../../../../../lib/datasets";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: { dataset: string } },
) {
  try {
    const dataset = decodeURIComponent(params.dataset);
    const evaluation = await getLatestDatasetEvaluation(dataset);
    return NextResponse.json({ evaluation });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read dataset evaluation: ${String(error)}` },
      { status: 404 },
    );
  }
}
