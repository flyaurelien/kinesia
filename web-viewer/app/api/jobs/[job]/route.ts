import { NextResponse } from "next/server";

import { removeJob, pauseJob, resumeJob, restartJob } from "../../../../lib/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Control a job mid-flight: pause (SIGSTOP), resume (SIGCONT), or restart (stop
// + relaunch with the same input/params). Stop/remove stays on DELETE.
export async function PATCH(
  request: Request,
  { params }: { params: { job: string } },
) {
  try {
    const jobId = decodeURIComponent(params.job);
    const body = (await request.json().catch(() => ({}))) as { action?: string };
    const job =
      body.action === "pause"
        ? pauseJob(jobId)
        : body.action === "resume"
          ? resumeJob(jobId)
          : body.action === "restart"
            ? await restartJob(jobId)
            : undefined;
    if (job === undefined) {
      return NextResponse.json({ error: "Unknown action." }, { status: 400 });
    }
    if (job === null) {
      return NextResponse.json({ error: "Job not found." }, { status: 404 });
    }
    return NextResponse.json({ job });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to update job: ${String(error)}` },
      { status: 500 },
    );
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: { job: string } },
) {
  try {
    const jobId = decodeURIComponent(params.job);
    const removed = removeJob(jobId);
    if (!removed) {
      return NextResponse.json(
        { error: "Job not found." },
        { status: 404 },
      );
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to delete job: ${String(error)}` },
      { status: 500 },
    );
  }
}
