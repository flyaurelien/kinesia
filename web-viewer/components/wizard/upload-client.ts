// Browser helper to stage large uploads via the chunked-upload API.

import { apiFetch } from "../../lib/api-client";

const CHUNK_BYTES = 8 * 1024 * 1024;

export type StagedUploadResult = {
  stagedUploadId: string;
  fileName: string;
};

function randomId(): string {
  const arr = new Uint8Array(16);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    crypto.getRandomValues(arr);
  } else {
    for (let i = 0; i < arr.length; i += 1) arr[i] = Math.floor(Math.random() * 256);
  }
  return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function stageUpload(
  file: File,
  onProgress?: (loaded: number, total: number) => void,
): Promise<StagedUploadResult> {
  // For small files, the backend accepts the file directly in a single POST,
  // but the job and subject-preview routes still expect a stagedUploadId for
  // multi-step flows. Always go chunked for consistency — backend handles 1 chunk.
  const uploadId = randomId();
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_BYTES));
  let loaded = 0;
  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
    const start = chunkIndex * CHUNK_BYTES;
    const end = Math.min(file.size, start + CHUNK_BYTES);
    const blob = file.slice(start, end);
    const formData = new FormData();
    formData.append("uploadId", uploadId);
    formData.append("fileName", file.name);
    formData.append("chunkIndex", String(chunkIndex));
    formData.append("totalChunks", String(totalChunks));
    // Pass a filename so the server-side `instanceof File` check passes.
    formData.append("chunk", blob, `${file.name}.part${chunkIndex}`);
    const resp = await apiFetch("/api/uploads/chunk", { method: "POST", body: formData });
    if (!resp.ok) {
      throw new Error(`Chunk ${chunkIndex + 1}/${totalChunks} upload failed (${resp.status})`);
    }
    loaded = end;
    onProgress?.(loaded, file.size);
  }
  const completeForm = new FormData();
  completeForm.append("uploadId", uploadId);
  const completeResp = await apiFetch("/api/uploads/complete", {
    method: "POST",
    body: completeForm,
  });
  if (!completeResp.ok) {
    throw new Error(`Upload finalize failed (${completeResp.status})`);
  }
  return { stagedUploadId: uploadId, fileName: file.name };
}
