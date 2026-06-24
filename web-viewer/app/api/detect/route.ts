import { NextResponse } from "next/server";

import { resolveStagedUpload } from "../../../lib/chunked-uploads";
import { startDetectJob } from "../../../lib/detect-jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Start a streaming detection job over a previously-staged upload. The video is
// staged once (shared with the eventual run), so this only needs its id.
export async function POST(request: Request) {
  try {
    const body = (await request.json().catch(() => ({}))) as {
      stagedUploadId?: unknown;
      prompt?: unknown;
      stride?: unknown;
      minDurationSec?: unknown;
    };

    const stagedUploadId = typeof body.stagedUploadId === "string" ? body.stagedUploadId.trim() : "";
    if (!stagedUploadId) {
      return NextResponse.json({ error: "Missing stagedUploadId" }, { status: 400 });
    }
    const staged = await resolveStagedUpload(stagedUploadId).catch(() => null);
    if (!staged) {
      return NextResponse.json({ error: "Staged upload not found" }, { status: 404 });
    }

    const prompt =
      typeof body.prompt === "string" && body.prompt.trim() ? body.prompt.trim().slice(0, 120) : "person";
    const strideRaw = Number(body.stride);
    const stride = Number.isFinite(strideRaw) ? Math.min(30, Math.max(1, Math.round(strideRaw))) : 5;
    // Hallucination filter in seconds (subjects present for less are ignored).
    const durRaw = Number(body.minDurationSec);
    const minDurationSec = Number.isFinite(durRaw) ? Math.min(30, Math.max(0, durRaw)) : 1.0;

    const job = await startDetectJob({ inputPath: staged.filePath, prompt, stride, minDurationSec });
    return NextResponse.json({ detectId: job.id, stride, prompt, minDurationSec }, { status: 201 });
  } catch (error) {
    return NextResponse.json({ error: `Failed to start detection: ${String(error)}` }, { status: 500 });
  }
}
