// Structured kinematics export ("Export folder").
//
// One user action produces a single ZIP that unpacks to a folder named
//   <video>_<YYYYMMDD-HHMMSS>/
// containing the per-joint kinematics file and, optionally, the rendered
// tracking-box MP4:
//   <base>_kinematics.{csv|json}
//   <base>_trackingbox.mp4        (added by downloadBundle when present)
//
// The ZIP is written here in pure TypeScript (STORE / no compression) so the
// feature has no runtime dependency and works fully offline. Everything in this
// module except downloadBundle() is DOM-free and unit-testable in Node.

import type { RunDetail } from "./types";
import { buildKinematicsCsv, jointLabel, maxJointCount, safeFileName } from "./export";

// Per-frame channels can be CSV or JSON.
export type FrameFormat = "csv" | "json";

export type ChannelFormats = {
  kinematics: FrameFormat;
};

export const DEFAULT_CHANNEL_FORMATS: ChannelFormats = {
  kinematics: "csv",
};

export type BundleInputs = {
  // Shared folder/file stem, already sanitized + timestamped.
  baseName: string;
  runId: string;
  videoFileName?: string | null;
  dateIso: string;
  // Kinematics source (frames + signals + fps).
  runDetail: RunDetail;
};

// One file ready to drop into the ZIP.
export type BundleFile = { name: string; text: string };

function timestampMs(videoFrame: number, index: number, fps: number): number {
  const safeFps = Math.max(1, fps || 30);
  return Math.round((Number.isFinite(videoFrame) ? videoFrame : index) / safeFps * 1000);
}

// ── Kinematics ──────────────────────────────────────────────────────────────-
export function buildKinematicsJson(inputs: BundleInputs): string {
  const { runDetail, runId } = inputs;
  const { frames, signals, fps } = runDetail;
  // Full per-joint 3D positions (camera space) so the export is a COMPLETE kinematics.
  const jc = maxJointCount(frames);
  const jointNames = Array.from({ length: jc }, (_, j) => jointLabel(j));
  const payload = {
    schema: "kinesia.kinematics.v1",
    runId,
    fps,
    frameCount: frames.length,
    signals: signals.map((s) => ({ id: s.id, label: s.label, unit: s.unit })),
    joints: { count: jc, names: jointNames, space: "camera", unit: "m", order: ["x", "y", "z"] },
    frames: frames.map((f, i) => {
      const videoFrame = f.videoFrame ?? i;
      const values: Record<string, number | null> = {};
      for (const s of signals) {
        const v = s.values[i];
        values[s.id] = typeof v === "number" && Number.isFinite(v) ? v : null;
      }
      const joints: Record<string, [number, number, number] | null> = {};
      for (let j = 0; j < jc; j += 1) {
        const p = f.jointsCam && f.jointsCam[j];
        joints[jointNames[j]] = p ? [p[0], p[1], p[2]] : null;
      }
      return {
        index: i,
        videoFrame,
        timestampMs: timestampMs(videoFrame, i, fps),
        values,
        joints,
      };
    }),
  };
  return JSON.stringify(payload, null, 2);
}

// ── Assemble the file list for the bundle ────────────────────────────────────-
const FRAME_EXT: Record<FrameFormat, string> = { csv: "csv", json: "json" };

export function buildBundleFiles(inputs: BundleInputs, formats: ChannelFormats): BundleFile[] {
  const base = inputs.baseName;
  const path = (channel: string, ext: string) => `${base}/${base}_${channel}.${ext}`;
  const files: BundleFile[] = [];

  // Per-joint kinematics (the tracking-box MP4 is added separately by downloadBundle).
  files.push({
    name: path("kinematics", FRAME_EXT[formats.kinematics]),
    text: formats.kinematics === "json"
      ? buildKinematicsJson(inputs)
      : buildKinematicsCsv(inputs.runDetail),
  });

  return files;
}

// ── Folder / file naming ─────────────────────────────────────────────────────-
function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

// "<video>_<YYYYMMDD-HHMMSS>", sanitized for the filesystem.
export function bundleBaseName(videoName: string, date: Date): string {
  const safe = safeFileName(videoName);
  const stamp =
    `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}` +
    `-${pad2(date.getHours())}${pad2(date.getMinutes())}${pad2(date.getSeconds())}`;
  return `${safe}_${stamp}`;
}

// ── Pure ZIP writer (STORE, no compression) ──────────────────────────────────-
let CRC_TABLE: Uint32Array | null = null;
function crcTable(): Uint32Array {
  if (CRC_TABLE) return CRC_TABLE;
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  CRC_TABLE = table;
  return table;
}

export function crc32(bytes: Uint8Array): number {
  const table = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) c = table[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function dosDateTime(date: Date): { time: number; date: number } {
  const year = Math.max(1980, date.getFullYear());
  const dosDate = (((year - 1980) & 0x7f) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  return { time: dosTime & 0xffff, date: dosDate & 0xffff };
}

// Build a valid ZIP archive (single-segment, STORE method). UTF-8 filenames are
// flagged via general-purpose bit 11 so unzippers decode the folder names right.
export function buildZip(files: Array<{ name: string; data: Uint8Array }>, date = new Date()): Uint8Array {
  const { time: dosTime, date: dosDate } = dosDateTime(date);
  const encoder = new TextEncoder();
  const entries = files.map((f) => {
    const nameBytes = encoder.encode(f.name);
    return { nameBytes, data: f.data, crc: crc32(f.data) };
  });

  let localSize = 0;
  let centralSize = 0;
  for (const e of entries) {
    localSize += 30 + e.nameBytes.length + e.data.length;
    centralSize += 46 + e.nameBytes.length;
  }
  const out = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(out.buffer);
  let offset = 0;
  const localOffsets: number[] = [];

  for (const e of entries) {
    localOffsets.push(offset);
    view.setUint32(offset, 0x04034b50, true); offset += 4; // local file header sig
    view.setUint16(offset, 20, true); offset += 2;          // version needed
    view.setUint16(offset, 0x0800, true); offset += 2;      // flags: UTF-8 names
    view.setUint16(offset, 0, true); offset += 2;           // method: store
    view.setUint16(offset, dosTime, true); offset += 2;
    view.setUint16(offset, dosDate, true); offset += 2;
    view.setUint32(offset, e.crc, true); offset += 4;
    view.setUint32(offset, e.data.length, true); offset += 4; // compressed size
    view.setUint32(offset, e.data.length, true); offset += 4; // uncompressed size
    view.setUint16(offset, e.nameBytes.length, true); offset += 2;
    view.setUint16(offset, 0, true); offset += 2;           // extra length
    out.set(e.nameBytes, offset); offset += e.nameBytes.length;
    out.set(e.data, offset); offset += e.data.length;
  }

  const centralStart = offset;
  entries.forEach((e, i) => {
    view.setUint32(offset, 0x02014b50, true); offset += 4; // central dir header sig
    view.setUint16(offset, 20, true); offset += 2;          // version made by
    view.setUint16(offset, 20, true); offset += 2;          // version needed
    view.setUint16(offset, 0x0800, true); offset += 2;      // flags: UTF-8 names
    view.setUint16(offset, 0, true); offset += 2;           // method: store
    view.setUint16(offset, dosTime, true); offset += 2;
    view.setUint16(offset, dosDate, true); offset += 2;
    view.setUint32(offset, e.crc, true); offset += 4;
    view.setUint32(offset, e.data.length, true); offset += 4;
    view.setUint32(offset, e.data.length, true); offset += 4;
    view.setUint16(offset, e.nameBytes.length, true); offset += 2;
    view.setUint16(offset, 0, true); offset += 2;           // extra length
    view.setUint16(offset, 0, true); offset += 2;           // comment length
    view.setUint16(offset, 0, true); offset += 2;           // disk number start
    view.setUint16(offset, 0, true); offset += 2;           // internal attrs
    view.setUint32(offset, 0, true); offset += 4;           // external attrs
    view.setUint32(offset, localOffsets[i], true); offset += 4; // local header offset
    out.set(e.nameBytes, offset); offset += e.nameBytes.length;
  });

  view.setUint32(offset, 0x06054b50, true); offset += 4; // end of central dir sig
  view.setUint16(offset, 0, true); offset += 2;           // disk number
  view.setUint16(offset, 0, true); offset += 2;           // central dir start disk
  view.setUint16(offset, entries.length, true); offset += 2;
  view.setUint16(offset, entries.length, true); offset += 2;
  view.setUint32(offset, centralSize, true); offset += 4;
  view.setUint32(offset, centralStart, true); offset += 4;
  view.setUint16(offset, 0, true); offset += 2;           // comment length

  return out;
}

// ── Browser download ─────────────────────────────────────────────────────────-
// Optional `binaryFiles` (e.g. the tracking-box MP4) are added to the same ZIP as
// raw bytes alongside the text channel files.
export function downloadBundle(
  baseName: string,
  files: BundleFile[],
  date = new Date(),
  binaryFiles: Array<{ name: string; data: Uint8Array }> = [],
): void {
  const encoder = new TextEncoder();
  const zip = buildZip(
    [
      ...files.map((f) => ({ name: f.name, data: encoder.encode(f.text) })),
      ...binaryFiles,
    ],
    date,
  );
  // Copy into a standalone ArrayBuffer so Blob gets exactly the archive bytes.
  const blob = new Blob([zip.slice()], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${baseName}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
