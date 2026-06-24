import { Buffer } from "node:buffer";

import { NextResponse } from "next/server";

import { saveUploadChunk } from "../../../../lib/chunked-uploads";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const rawChunk = formData.get("chunk");
    const uploadId = formData.get("uploadId");
    const fileName = formData.get("fileName");
    const chunkIndex = Number(formData.get("chunkIndex"));
    const totalChunks = Number(formData.get("totalChunks"));
    if (!(rawChunk instanceof File) || typeof uploadId !== "string" || typeof fileName !== "string") {
      return NextResponse.json({ error: "Missing upload chunk fields" }, { status: 400 });
    }
    const result = await saveUploadChunk({
      uploadId,
      fileName,
      chunkIndex,
      totalChunks,
      chunkData: Buffer.from(await rawChunk.arrayBuffer()),
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json({ error: `Failed to save upload chunk: ${String(error)}` }, { status: 400 });
  }
}
