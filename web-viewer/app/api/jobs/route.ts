import { Buffer } from "node:buffer";
import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import { isAllowedVideoFileName, resolveStagedUpload } from "../../../lib/chunked-uploads";
import { detectScratchRoot } from "../../../lib/detect-jobs";
import { createJob, createJobFromExistingUpload, listJobs } from "../../../lib/jobs";

// Accept a chosen-subject track file only when it lives inside the detect-job
// scratch root and exists — never an arbitrary client-supplied path.
async function validateSubjectTrackFile(raw: FormDataEntryValue | null): Promise<string | null> {
  if (typeof raw !== "string" || !raw.trim()) return null;
  const resolved = path.resolve(raw.trim());
  const root = `${path.resolve(detectScratchRoot())}${path.sep}`;
  if (!resolved.startsWith(root) || path.basename(resolved) !== "chosen_subject_track.json") return null;
  try {
    await fs.access(resolved);
    return resolved;
  } catch {
    return null;
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET /api/jobs — return all known generation jobs for the dashboard/wizard.
export async function GET() {
  try {
    return NextResponse.json({ jobs: listJobs() });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to list jobs: ${String(error)}` },
      { status: 500 },
    );
  }
}

// POST /api/jobs — create a generation job from a multipart upload or a previously
// staged (chunked) upload, validating the file and forwarding the wizard's run options.
export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const rawFile = formData.get("video");
    const stagedUploadIdRaw = formData.get("stagedUploadId");
    const videoFileNameRaw = formData.get("videoFileName");
    const runNameRaw = formData.get("runName");
    const inferenceTargetRaw = formData.get("inferenceTarget");
    const precisionRaw = formData.get("precision");
    const autoInitModeRaw = formData.get("autoInitMode");
    const autoSelectStrategyRaw = formData.get("autoSelectStrategy");
    const cameraMotionCompensationRaw = formData.get("cameraMotionCompensation");
    const renderPreviewRaw = formData.get("renderPreview");
    const promptBBoxRaw = formData.get("promptBBox");
    const promptBBoxFrameRaw = formData.get("promptBBoxFrame");
    const startFrameRaw = formData.get("startFrame");
    const maxFramesRaw = formData.get("maxFrames");
    const frameStepRaw = formData.get("frameStep");
    const trimStartSecRaw = formData.get("trimStartSec");
    const trimEndSecRaw = formData.get("trimEndSec");
    const removedSegmentsRaw = formData.get("removedSegments");
    const maskedSegmentsRaw = formData.get("maskedSegments");
    const cropBoxRaw = formData.get("cropBox");
    const sam3TextPromptsRaw = formData.get("sam3TextPrompts");
    const promptAnchorsJsonRaw = formData.get("promptAnchorsJson");
    const subjectTrackFile = await validateSubjectTrackFile(formData.get("subjectTrackFile"));
    const runName =
      typeof runNameRaw === "string" && runNameRaw.trim().length > 0
        ? runNameRaw.trim()
        : null;
    const inferenceTarget =
      typeof inferenceTargetRaw === "string" && inferenceTargetRaw.trim().length > 0
        ? inferenceTargetRaw
        : null;

    const stagedUpload =
      typeof stagedUploadIdRaw === "string" && stagedUploadIdRaw.trim()
        ? await resolveStagedUpload(stagedUploadIdRaw)
        : null;

    if (!(rawFile instanceof File) && !stagedUpload) {
      return NextResponse.json(
        { error: "Missing uploaded video file" },
        { status: 400 },
      );
    }
    const inputFileName = stagedUpload?.fileName ?? (rawFile instanceof File ? rawFile.name : String(videoFileNameRaw ?? ""));
    if (!isAllowedVideoFileName(inputFileName)) {
      return NextResponse.json(
        { error: "Unsupported video format. Use .mp4/.mov/.m4v/.avi/.mkv/.webm" },
        { status: 400 },
      );
    }
    if (rawFile instanceof File && (!Number.isFinite(rawFile.size) || rawFile.size <= 0)) {
      return NextResponse.json(
        { error: "Uploaded file is empty" },
        { status: 400 },
      );
    }

    const createOptions = {
      precisionRaw: typeof precisionRaw === "string" ? precisionRaw : null,
      autoInitModeRaw: typeof autoInitModeRaw === "string" ? autoInitModeRaw : null,
      autoSelectStrategyRaw: typeof autoSelectStrategyRaw === "string" ? autoSelectStrategyRaw : null,
      cameraMotionCompensationRaw:
        typeof cameraMotionCompensationRaw === "string" ? cameraMotionCompensationRaw : null,
      renderPreviewRaw: typeof renderPreviewRaw === "string" ? renderPreviewRaw : null,
      promptBBoxRaw: typeof promptBBoxRaw === "string" ? promptBBoxRaw : null,
      promptBBoxFrameRaw: typeof promptBBoxFrameRaw === "string" ? promptBBoxFrameRaw : null,
      startFrameRaw: typeof startFrameRaw === "string" ? startFrameRaw : null,
      maxFramesRaw: typeof maxFramesRaw === "string" ? maxFramesRaw : null,
      frameStepRaw: typeof frameStepRaw === "string" ? frameStepRaw : null,
      trimStartSecRaw: typeof trimStartSecRaw === "string" ? trimStartSecRaw : null,
      trimEndSecRaw: typeof trimEndSecRaw === "string" ? trimEndSecRaw : null,
      removedSegmentsRaw: typeof removedSegmentsRaw === "string" ? removedSegmentsRaw : null,
      maskedSegmentsRaw: typeof maskedSegmentsRaw === "string" ? maskedSegmentsRaw : null,
      cropBoxRaw: typeof cropBoxRaw === "string" ? cropBoxRaw : null,
      promptAnchorsJsonRaw:
        typeof promptAnchorsJsonRaw === "string" && promptAnchorsJsonRaw.trim().length > 0
          ? promptAnchorsJsonRaw.trim()
          : null,
      subjectTrackFile,
    };
    const sam3TextPrompts =
      typeof sam3TextPromptsRaw === "string" && sam3TextPromptsRaw.trim().length > 0
        ? sam3TextPromptsRaw.trim()
        : null;
    const job = stagedUpload
      ? await createJobFromExistingUpload(stagedUpload.fileName, stagedUpload.filePath, runName, inferenceTarget, sam3TextPrompts, createOptions)
      : await createJob(inputFileName, Buffer.from(await (rawFile as File).arrayBuffer()), runName, inferenceTarget, sam3TextPrompts, createOptions);
    return NextResponse.json({ job }, { status: 201 });
  } catch (error) {
    const message = String(error);
    const status =
      message.includes("multipart/form-data") ||
      message.includes("application/x-www-form-urlencoded") ||
      message.includes("Missing model files:")
        ? 400
        : 500;
    return NextResponse.json(
      { error: `Failed to create generation job: ${message}` },
      { status },
    );
  }
}
