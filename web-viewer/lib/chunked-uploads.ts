import { promises as fs } from "node:fs";
import path from "node:path";

import { uploadsRoot } from "./store";

// Video extensions accepted for upload; anything else is rejected up front.
const ALLOWED_VIDEO_EXT = new Set([".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"]);
// Per-chunk upper bound (32 MiB) guarding against oversized/malicious requests.
const MAX_CHUNK_BYTES = 32 * 1024 * 1024;

// On-disk state for an in-progress chunked upload session.
type ChunkSessionMeta = {
  uploadId: string;
  fileName: string;
  totalChunks: number;
  createdAt: string;
  updatedAt: string;
};

// A fully reassembled upload, ready for the pipeline to consume.
export type StagedUpload = {
  uploadId: string;
  fileName: string;
  filePath: string;
  size: number;
};

// Validate an upload id so it can be used as a path segment without traversal risk.
function safeUploadId(raw: string): string {
  const value = raw.trim();
  if (!/^[a-zA-Z0-9_-]{8,120}$/.test(value)) {
    throw new Error("Invalid upload id");
  }
  return value;
}

// Strip any path components and confirm the extension is an allowed video format.
function safeFileName(raw: string): string {
  const baseName = path.basename(raw.trim());
  const ext = path.extname(baseName).toLowerCase();
  if (!baseName || !ALLOWED_VIDEO_EXT.has(ext)) {
    throw new Error("Unsupported video format. Use .mp4/.mov/.m4v/.avi/.mkv/.webm");
  }
  return baseName;
}

// Directory holding in-progress chunk sessions, one subdirectory per upload.
function chunkRoot(): string {
  return path.join(uploadsRoot(), ".chunks");
}

// Directory holding fully reassembled uploads plus their sidecar metadata.
function stagedRoot(): string {
  return path.join(uploadsRoot(), ".staged");
}

// Per-upload scratch directory under the chunk root.
function sessionDir(uploadId: string): string {
  return path.join(chunkRoot(), safeUploadId(uploadId));
}

// Path to a session's metadata file tracking expected file name and chunk count.
function metaPath(uploadId: string): string {
  return path.join(sessionDir(uploadId), "meta.json");
}

// Path to a single uploaded chunk, named by its index.
function chunkPath(uploadId: string, chunkIndex: number): string {
  return path.join(sessionDir(uploadId), `${chunkIndex}.part`);
}

// Path to the JSON sidecar describing a completed staged upload.
function stagedMetaPath(uploadId: string): string {
  return path.join(stagedRoot(), `${safeUploadId(uploadId)}.json`);
}

// Path to the reassembled video; named by upload id so it stays inside stagedRoot.
function stagedVideoPath(uploadId: string, fileName: string): string {
  return path.join(stagedRoot(), `${safeUploadId(uploadId)}${path.extname(safeFileName(fileName)).toLowerCase()}`);
}

// Read and parse a JSON file as type T.
async function readJson<T>(filePath: string): Promise<T> {
  const text = await fs.readFile(filePath, "utf-8");
  return JSON.parse(text) as T;
}

// True if the file name carries an accepted video extension.
export function isAllowedVideoFileName(fileName: string): boolean {
  return ALLOWED_VIDEO_EXT.has(path.extname(fileName).toLowerCase());
}

// Persist one chunk of an upload, creating the session metadata on first write and
// rejecting chunks whose declared file name or chunk count diverge from the session.
// Returns how many chunks have arrived so the caller can detect completion.
export async function saveUploadChunk(input: {
  uploadId: string;
  fileName: string;
  chunkIndex: number;
  totalChunks: number;
  chunkData: Buffer;
}): Promise<{ received: number; totalChunks: number }> {
  const uploadId = safeUploadId(input.uploadId);
  const fileName = safeFileName(input.fileName);
  const chunkIndex = Math.trunc(input.chunkIndex);
  const totalChunks = Math.trunc(input.totalChunks);
  if (totalChunks < 1 || totalChunks > 10000 || chunkIndex < 0 || chunkIndex >= totalChunks) {
    throw new Error("Invalid chunk index");
  }
  if (input.chunkData.length <= 0 || input.chunkData.length > MAX_CHUNK_BYTES) {
    throw new Error("Invalid chunk size");
  }

  const directory = sessionDir(uploadId);
  await fs.mkdir(directory, { recursive: true });
  const now = new Date().toISOString();
  const meta: ChunkSessionMeta = await fs
    .readFile(metaPath(uploadId), "utf-8")
    .then((text) => JSON.parse(text) as ChunkSessionMeta)
    .catch(() => ({ uploadId, fileName, totalChunks, createdAt: now, updatedAt: now }));
  if (meta.fileName !== fileName || meta.totalChunks !== totalChunks) {
    throw new Error("Upload session metadata mismatch");
  }
  meta.updatedAt = now;
  await fs.writeFile(chunkPath(uploadId, chunkIndex), input.chunkData);
  await fs.writeFile(metaPath(uploadId), JSON.stringify(meta, null, 2));

  const entries = await fs.readdir(directory).catch(() => []);
  const received = entries.filter((entry) => entry.endsWith(".part")).length;
  return { received, totalChunks };
}

// Concatenate all chunks in order into the staged video file, write its metadata
// sidecar, and discard the chunk session. Returns the resulting staged upload.
export async function completeChunkedUpload(uploadIdRaw: string): Promise<StagedUpload> {
  const uploadId = safeUploadId(uploadIdRaw);
  const meta = await readJson<ChunkSessionMeta>(metaPath(uploadId));
  const fileName = safeFileName(meta.fileName);
  await fs.mkdir(stagedRoot(), { recursive: true });
  const outputPath = stagedVideoPath(uploadId, fileName);
  const handle = await fs.open(outputPath, "w");
  try {
    for (let index = 0; index < meta.totalChunks; index += 1) {
      const chunk = await fs.readFile(chunkPath(uploadId, index));
      await handle.write(chunk);
    }
  } finally {
    await handle.close();
  }
  const stat = await fs.stat(outputPath);
  const staged: StagedUpload = { uploadId, fileName, filePath: outputPath, size: stat.size };
  await fs.writeFile(stagedMetaPath(uploadId), JSON.stringify(staged, null, 2));
  await fs.rm(sessionDir(uploadId), { recursive: true, force: true }).catch(() => undefined);
  return staged;
}

// Load a previously staged upload's metadata, re-validating the id, file name, and
// that the resolved path stays under stagedRoot before confirming the file exists.
export async function resolveStagedUpload(uploadIdRaw: string): Promise<StagedUpload> {
  const uploadId = safeUploadId(uploadIdRaw);
  const staged = await readJson<StagedUpload>(stagedMetaPath(uploadId));
  if (staged.uploadId !== uploadId || !isAllowedVideoFileName(staged.fileName)) {
    throw new Error("Invalid staged upload metadata");
  }
  const root = path.resolve(stagedRoot());
  const filePath = path.resolve(staged.filePath);
  if (!filePath.startsWith(`${root}${path.sep}`)) {
    throw new Error("Invalid staged upload path");
  }
  const stat = await fs.stat(filePath);
  if (!stat.isFile() || stat.size <= 0) {
    throw new Error("Staged upload file is missing");
  }
  return { ...staged, filePath, size: stat.size };
}
