import { NextResponse } from "next/server";

import { completeChunkedUpload } from "../../../../lib/chunked-uploads";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const uploadId = formData.get("uploadId");
    if (typeof uploadId !== "string") {
      return NextResponse.json({ error: "Missing upload id" }, { status: 400 });
    }
    const staged = await completeChunkedUpload(uploadId);
    return NextResponse.json({
      stagedUploadId: staged.uploadId,
      fileName: staged.fileName,
      size: staged.size,
    });
  } catch (error) {
    return NextResponse.json({ error: `Failed to complete upload: ${String(error)}` }, { status: 400 });
  }
}
