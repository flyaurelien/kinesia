// Kinematics export helpers.
//
// Kinematics .csv — every computed signal, one column per signal, one row per
// frame, time-aligned (frame_index, video_frame, timestamp_ms), plus each
// joint's 3D position in metres (camera space). ML-training ready.

import type { RunDetail } from "./types";

// Format one numeric CSV cell; nullish / non-finite values become an empty cell.
function csvCell(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "";
  // Trim to a sane precision without scientific notation for typical ranges.
  return Number.isInteger(v) ? String(v) : v.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

// RFC-4180 quoting for header cells (signal ids/units could in theory contain
// a comma, quote or newline).
function csvQuote(v: string): string {
  if (!/[",\n\r]/.test(v)) return v;
  return `"${v.replace(/"/g, '""')}"`;
}

// Friendly names for the model's named pose joints; the rest export as jointNN.
const JOINT_NAMES: Record<number, string> = {
  5: "left_shoulder", 6: "right_shoulder", 9: "left_hip", 10: "right_hip",
  11: "left_knee", 12: "right_knee", 13: "left_ankle", 14: "right_ankle",
  15: "left_big_toe", 16: "left_small_toe", 17: "left_heel",
  18: "right_big_toe", 19: "right_small_toe", 20: "right_heel",
};

export function jointLabel(index: number): string {
  return JOINT_NAMES[index] ?? `joint${String(index).padStart(2, "0")}`;
}

// Largest per-frame joint count across the run (so the full skeleton is exported).
export function maxJointCount(frames: RunDetail["frames"]): number {
  let n = 0;
  for (const f of frames) {
    if (Array.isArray(f.jointsCam)) n = Math.max(n, f.jointsCam.length);
  }
  return n;
}

// ── Kinematics CSV ──────────────────────────────────────────────────────────
// Per-frame signal table: frame_index, video_frame, timestamp_ms, one column
// per computed signal, then every joint's 3D position (x/y/z, metres).
export function buildKinematicsCsv(runDetail: RunDetail): string {
  const { frames, signals, fps } = runDetail;
  const safeFps = Math.max(1, fps || 30);
  // Every joint's 3D position (camera space, metres), so it's a COMPLETE kinematics.
  const jc = maxJointCount(frames);
  const jointCols: string[] = [];
  const jointUnits: string[] = [];
  for (let j = 0; j < jc; j += 1) {
    const name = jointLabel(j);
    jointCols.push(`${name}_x`, `${name}_y`, `${name}_z`);
    jointUnits.push("m", "m", "m");
  }
  const header = [
    "frame_index", "video_frame", "timestamp_ms",
    ...signals.map((s) => csvQuote(s.id)),
    ...jointCols.map((c) => csvQuote(c)),
  ];
  const unitRow = [
    "", "", "ms",
    ...signals.map((s) => csvQuote(s.unit || "")),
    ...jointUnits,
  ];
  const lines: string[] = [header.join(","), unitRow.join(",")];
  for (let i = 0; i < frames.length; i += 1) {
    const f = frames[i];
    const tsMs = Math.round((f.videoFrame ?? i) / safeFps * 1000);
    const row: string[] = [
      String(i),
      String(f.videoFrame ?? i),
      String(tsMs),
    ];
    for (const s of signals) row.push(csvCell(s.values[i]));
    const joints = f.jointsCam;
    for (let j = 0; j < jc; j += 1) {
      const p = joints && joints[j];
      row.push(csvCell(p ? p[0] : null), csvCell(p ? p[1] : null), csvCell(p ? p[2] : null));
    }
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

// Sanitize a string into a filesystem-safe download name (kept short, with a
// non-empty fallback).
export function safeFileName(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 120) || "kinesia";
}
