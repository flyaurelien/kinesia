import { NextResponse } from "next/server";

import { deleteRun, ensureRunAnalysis, getRunDetail } from "../../../../lib/runs";
import { listJobs } from "../../../../lib/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const url = new URL(request.url);
    const analysisId = url.searchParams.get("analysisId");
    // Auto-compute FoG/kinematics if a finished run has none (old/CLI runs, or a run
    // whose analyze step never completed). Skip while a job is still generating it.
    if (!analysisId) {
      const busy = listJobs().some(
        (job) => job.runId === run && (job.status === "queued" || job.status === "running"),
      );
      if (!busy) await ensureRunAnalysis(run).catch(() => undefined);
    }
    const detail = await getRunDetail(run, analysisId);
    return NextResponse.json({ run: detail });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read run detail: ${String(error)}` },
      { status: 404 },
    );
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: { run: string } },
) {
  try {
    const run = decodeURIComponent(params.run);
    const locked = listJobs().some(
      (job) =>
        job.runId === run &&
        (job.status === "queued" || job.status === "running"),
    );
    if (locked) {
      return NextResponse.json(
        { error: "Run is currently being generated and cannot be deleted." },
        { status: 409 },
      );
    }
    await deleteRun(run);
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    const message = String(error);
    const status = message.includes("not found") ? 404 : 400;
    return NextResponse.json(
      { error: `Failed to delete run: ${message}` },
      { status },
    );
  }
}
