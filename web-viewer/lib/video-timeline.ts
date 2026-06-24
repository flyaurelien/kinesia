// Pure helpers for the in-browser video timeline.

// Constrain `value` to the inclusive [min, max] range.
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Format seconds as "M:SS.ss"; negative/non-finite inputs render as 0:00.00.
export function formatTimecode(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

/* ===================================================================== *
 * Segment model — the timeline is a contiguous list of segments that    *
 * tile [0, duration]. Each segment has a processing mode:               *
 *   keep   → processed by the algorithm and present in the output       *
 *   mask   → present in the output at its ORIGINAL timing, but skipped  *
 *            by the algorithm (saves compute, keeps sensor alignment)   *
 *   delete → removed from the output entirely (timeline compresses)     *
 * The whole clip starts as one "keep" segment; deletedRanges/maskedRanges *
 * are what the run step forwards to the backend.                        *
 * ===================================================================== */

export type SegmentMode = "keep" | "mask" | "delete";

export type Segment = {
  id: string;
  startSec: number;
  endSec: number;
  mode: SegmentMode;
};

// Monotonic counter backing nextSegmentId so every segment gets a stable,
// unique React key for the lifetime of the page.
let segmentIdCounter = 0;
function nextSegmentId(): string {
  segmentIdCounter += 1;
  return `seg-${segmentIdCounter}`;
}

// Start the timeline as a single "keep" segment spanning the whole clip.
export function makeInitialSegments(durationSec: number): Segment[] {
  const end = durationSec > 0 ? durationSec : 0;
  return [{ id: nextSegmentId(), startSec: 0, endSec: end, mode: "keep" }];
}

// Ranges removed from the output (mode = delete) → backend "removedSegments".
export function deletedRanges(segments: Segment[]): { startSec: number; endSec: number }[] {
  return segments
    .filter((s) => s.mode === "delete")
    .map((s) => ({ startSec: s.startSec, endSec: s.endSec }));
}

// Ranges kept in the output but skipped by inference (mode = mask).
export function maskedRanges(segments: Segment[]): { startSec: number; endSec: number }[] {
  return segments
    .filter((s) => s.mode === "mask")
    .map((s) => ({ startSec: s.startSec, endSec: s.endSec }));
}
