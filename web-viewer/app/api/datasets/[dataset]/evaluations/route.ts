import { NextResponse } from "next/server";

import { createDatasetEvaluation } from "../../../../../lib/datasets";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(
  request: Request,
  { params }: { params: { dataset: string } },
) {
  try {
    const dataset = decodeURIComponent(params.dataset);
    const body = (await request.json().catch(() => ({}))) as {
      preset?: string;
    };
    const evaluation = await createDatasetEvaluation(dataset, body);
    return NextResponse.json({ evaluation }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to create dataset evaluation: ${String(error)}` },
      { status: 400 },
    );
  }
}
