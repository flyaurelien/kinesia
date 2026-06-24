// Export / import helpers for the FoG annotation workstation.
//
// Three export targets:
//   - ELAN .eaf (XML)  — annotations as time-aligned tiers, openable in ELAN
//     (https://archive.mpi.nl/tla/elan), with a media descriptor pointing at the
//     run video. Round-trippable via parseEaf().
//   - Kinematics .csv  — every computed signal, one column per signal, one row
//     per frame, time-aligned (frame_index, video_frame, timestamp_ms). ELAN can
//     link this as a secondary time-series file; it's also ML-training ready.
//   - Annotations .json (v2) — the native format, now including detection
//     settings + the deleted-prediction audit trail.

import type { RunDetail } from "./types";

export type ExportSegment = {
  startSec: number;
  endSec: number;
  startFrameIndex?: number;
  endFrameIndex?: number;
  label?: string;
  source?: string;
};

export type EafTier = { id: string; segments: ExportSegment[] };

// Escape the five XML predefined entities so arbitrary label/url text is safe
// to drop into the generated .eaf markup.
function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

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

// ── Kinematics CSV ──────────────────────────────────────────────────────────
// Per-frame signal table. `fog_detected` is the model's per-frame flag;
// `fog_annotated` (added when an annotation mask is supplied) is the human's
// final annotation layer — the importer reconstructs FoG segments from the
// latter when present, so the CSV round-trips the annotator's own labels.
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

export function buildKinematicsCsv(
  runDetail: RunDetail,
  opts?: { annotatedMask?: boolean[] },
): string {
  const { frames, signals, fps } = runDetail;
  const safeFps = Math.max(1, fps || 30);
  const annotatedMask = opts?.annotatedMask;
  const hasAnnotated = Array.isArray(annotatedMask);
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
    "frame_index", "video_frame", "timestamp_ms", "fog_detected",
    ...(hasAnnotated ? ["fog_annotated"] : []),
    ...signals.map((s) => csvQuote(s.id)),
    ...jointCols.map((c) => csvQuote(c)),
  ];
  const unitRow = [
    "", "", "ms", "bool",
    ...(hasAnnotated ? ["bool"] : []),
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
      f.fogDetected ? "1" : "0",
    ];
    if (hasAnnotated) row.push(annotatedMask![i] ? "1" : "0");
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

// ── ELAN .eaf ─────────────────────────────────────────────────────────────--
// Serialize annotation tiers into an EAFv3.0 document: dedup all segment
// boundaries into shared TIME_SLOTs, then emit one aligned annotation per
// segment. Round-trippable via parseEaf().
export function buildEaf(opts: {
  runId: string;
  videoFileName?: string | null;
  mediaUrl?: string | null;
  fps: number;
  tiers: EafTier[];
  dateIso: string;
}): string {
  const { runId, videoFileName, mediaUrl, tiers, dateIso } = opts;

  // Collect unique time values (ms) across all tiers → time slots.
  const timeSet = new Set<number>();
  for (const t of tiers) {
    for (const seg of t.segments) {
      timeSet.add(Math.max(0, Math.round(seg.startSec * 1000)));
      timeSet.add(Math.max(0, Math.round(seg.endSec * 1000)));
    }
  }
  const times = Array.from(timeSet).sort((a, b) => a - b);
  const slotIdByTime = new Map<number, string>();
  times.forEach((ms, i) => slotIdByTime.set(ms, `ts${i + 1}`));

  const timeOrder = times
    .map((ms) => `    <TIME_SLOT TIME_SLOT_ID="${slotIdByTime.get(ms)}" TIME_VALUE="${ms}"/>`)
    .join("\n");

  let annId = 0;
  const tierXml = tiers
    .map((tier) => {
      const validSegs = tier.segments.filter((seg) => seg.endSec > seg.startSec);
      if (validSegs.length === 0) return null; // omit empty tiers (no stray blank lines)
      const anns = validSegs
        .map((seg) => {
          annId += 1;
          const t1 = slotIdByTime.get(Math.max(0, Math.round(seg.startSec * 1000)));
          const t2 = slotIdByTime.get(Math.max(0, Math.round(seg.endSec * 1000)));
          const value = escapeXml(seg.label || "fog");
          return (
            `      <ANNOTATION>\n` +
            `        <ALIGNABLE_ANNOTATION ANNOTATION_ID="a${annId}" TIME_SLOT_REF1="${t1}" TIME_SLOT_REF2="${t2}">\n` +
            `          <ANNOTATION_VALUE>${value}</ANNOTATION_VALUE>\n` +
            `        </ALIGNABLE_ANNOTATION>\n` +
            `      </ANNOTATION>`
          );
        })
        .join("\n");
      return (
        `  <TIER LINGUISTIC_TYPE_REF="fog_event" TIER_ID="${escapeXml(tier.id)}">\n` +
        `${anns}\n` +
        `  </TIER>`
      );
    })
    .filter(Boolean)
    .join("\n");

  const mediaDescriptor = mediaUrl
    ? `    <MEDIA_DESCRIPTOR MEDIA_URL="${escapeXml(mediaUrl)}" MIME_TYPE="video/mp4"${videoFileName ? ` RELATIVE_MEDIA_URL="./${escapeXml(videoFileName)}"` : ""}/>\n`
    : "";

  return (
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<ANNOTATION_DOCUMENT AUTHOR="Kinesia" DATE="${dateIso}" FORMAT="3.0" VERSION="3.0" ` +
    `xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ` +
    `xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">\n` +
    `  <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">\n` +
    mediaDescriptor +
    `    <PROPERTY NAME="FoGDetection.runId">${escapeXml(runId)}</PROPERTY>\n` +
    `    <PROPERTY NAME="lastUsedAnnotationId">${annId}</PROPERTY>\n` +
    `  </HEADER>\n` +
    `  <TIME_ORDER>\n${timeOrder}\n  </TIME_ORDER>\n` +
    `${tierXml}\n` +
    `  <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="fog_event" TIME_ALIGNABLE="true"/>\n` +
    `</ANNOTATION_DOCUMENT>\n`
  );
}

// ── ELAN .eaf import (round-trip) ───────────────────────────────────────────-
export type ParsedEafSegment = { startSec: number; endSec: number; label: string; tier: string };

// Parse an ELAN .eaf document into flat segments, resolving both aligned and
// (ELAN-authored) dependent annotations to their time slots. Returns [] when
// DOMParser is unavailable (SSR) or the XML is malformed.
export function parseEaf(xml: string): ParsedEafSegment[] {
  if (typeof DOMParser === "undefined") return [];
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (doc.querySelector("parsererror")) return [];
  // time slot id -> ms
  const slots = new Map<string, number>();
  doc.querySelectorAll("TIME_ORDER > TIME_SLOT").forEach((el) => {
    const id = el.getAttribute("TIME_SLOT_ID");
    const v = el.getAttribute("TIME_VALUE");
    if (id && v != null && Number.isFinite(Number(v))) slots.set(id, Number(v));
  });

  // First pass: index every annotation by id. Time-aligned annotations carry
  // their own slots; dependent annotations (REF_ANNOTATION, used by tiers
  // authored in ELAN) inherit them from a parent via ANNOTATION_REF.
  type Rec = { tier: string; value: string; t1?: number; t2?: number; ref?: string };
  const byId = new Map<string, Rec>();
  const order: string[] = [];
  let synthetic = 0;
  doc.querySelectorAll("TIER").forEach((tier) => {
    const tierId = tier.getAttribute("TIER_ID") || "FoG";
    tier.querySelectorAll("ALIGNABLE_ANNOTATION").forEach((ann) => {
      const id = ann.getAttribute("ANNOTATION_ID") || `__a${(synthetic += 1)}`;
      const r1 = ann.getAttribute("TIME_SLOT_REF1");
      const r2 = ann.getAttribute("TIME_SLOT_REF2");
      const value = ann.querySelector("ANNOTATION_VALUE")?.textContent?.trim() || "fog";
      byId.set(id, { tier: tierId, value, t1: r1 ? slots.get(r1) : undefined, t2: r2 ? slots.get(r2) : undefined });
      order.push(id);
    });
    tier.querySelectorAll("REF_ANNOTATION").forEach((ann) => {
      const id = ann.getAttribute("ANNOTATION_ID") || `__r${(synthetic += 1)}`;
      const ref = ann.getAttribute("ANNOTATION_REF") || undefined;
      const value = ann.querySelector("ANNOTATION_VALUE")?.textContent?.trim() || "fog";
      byId.set(id, { tier: tierId, value, ref });
      order.push(id);
    });
  });

  // Follow ANNOTATION_REF transitively to the aligned ancestor's time slots.
  const resolveSlots = (id: string, seen: Set<string>): { t1?: number; t2?: number } => {
    const rec = byId.get(id);
    if (!rec || seen.has(id)) return {};
    seen.add(id);
    if (rec.t1 != null && rec.t2 != null) return { t1: rec.t1, t2: rec.t2 };
    return rec.ref ? resolveSlots(rec.ref, seen) : {};
  };

  const out: ParsedEafSegment[] = [];
  for (const id of order) {
    const rec = byId.get(id);
    if (!rec) continue;
    const { t1, t2 } = resolveSlots(id, new Set());
    if (t1 == null || t2 == null) continue;
    out.push({ startSec: t1 / 1000, endSec: t2 / 1000, label: rec.value, tier: rec.tier });
  }
  return out;
}

// ── Unified annotation import ───────────────────────────────────────────────-
// One parser for every file a user can drop on the "Ground truth" track. It
// round-trips all three of the viewer's own exports (ELAN .eaf, the per-frame
// kinematics .csv, the native annotations .json) plus the project's canonical
// label files (kinesia.labels.v1: {episodes:[{label,start_ms,end_ms}]}) and
// loose JSON/CSV from other tools. Times are normalized to seconds and frame
// indices (into the run's frame list); labels are preserved.
export type ParsedAnnotation = {
  startSec: number;
  endSec: number;
  startFrameIndex: number;
  endFrameIndex: number;
  label: string;
  // Provenance when the file carries it (manual / auto_corrected / auto_predicted,
  // an ELAN tier id, or "csv"). Kept for callers that want to distinguish sources.
  source?: string;
};

// Coerce a JSON/CSV value to a number, returning NaN for anything that isn't a
// finite number or a non-blank numeric string.
function toNumber(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}

// Pull a start/end time (in seconds) out of one object, accepting seconds,
// milliseconds (*_ms / *Ms) or frame indices (*Frame / *_frame) under any of
// the common spellings. Seconds win, then ms, then frames.
function pickSeconds(item: Record<string, unknown>, which: "start" | "end", fps: number): number {
  const sec =
    which === "start"
      ? item.startSec ?? item.start_sec ?? item.start_time ?? item.onset ?? item.start ?? item.begin
      : item.endSec ?? item.end_sec ?? item.end_time ?? item.offset ?? item.end ?? item.stop;
  const s = toNumber(sec);
  if (Number.isFinite(s)) return s;
  const ms = which === "start"
    ? item.startMs ?? item.start_ms ?? item.onset_ms
    : item.endMs ?? item.end_ms ?? item.offset_ms;
  const m = toNumber(ms);
  if (Number.isFinite(m)) return m / 1000;
  const fr = which === "start"
    ? item.startFrameIndex ?? item.start_frame_index ?? item.startFrame ?? item.start_frame
    : item.endFrameIndex ?? item.end_frame_index ?? item.endFrame ?? item.end_frame;
  const f = toNumber(fr);
  if (Number.isFinite(f)) return f / Math.max(1, fps);
  return NaN;
}

// Pull a non-empty string label out of one object under any of the common keys.
function pickLabel(item: Record<string, unknown>): string | undefined {
  const v = item.label ?? item.value ?? item.type ?? item.event ?? item.class;
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

export function parseAnnotationsFile(
  text: string,
  fileName: string,
  opts: { fps: number; frameCount: number },
): ParsedAnnotation[] {
  const fps = Math.max(1, opts.fps || 30);
  const lastFrame = Math.max(0, (opts.frameCount | 0) - 1);
  const out: ParsedAnnotation[] = [];
  const seen = new Set<string>();
  // Dedupe identical (label + frame span) so a viewer-exported EAF whose
  // "annotated" and "model" tiers coincide collapses to one segment.
  const emit = (sf: number, ef: number, startSec: number, endSec: number, label?: string, source?: string): void => {
    const lab = label && label.trim() ? label.trim() : "fog";
    const key = `${lab}:${sf}:${ef}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ startSec, endSec, startFrameIndex: sf, endFrameIndex: ef, label: lab, source });
  };
  // From seconds (JSON / generic CSV). Inclusive frame rounding matches the
  // dataset loader, so the native JSON export round-trips frame-exact.
  const push = (startSec: number, endSec: number, label?: string, source?: string): void => {
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) return;
    const s = Math.max(0, Math.min(startSec, endSec));
    const e = Math.max(0, Math.max(startSec, endSec));
    const rawSf = Math.round(s * fps);
    const rawEf = Math.round(e * fps);
    // Drop segments whose whole span lies outside the run — clamping would
    // otherwise pin them to a degenerate blip at frame 0 or the last frame.
    if (rawEf < 0 || rawSf > lastFrame) return;
    const sf = clampInt(rawSf, 0, lastFrame);
    const ef = clampInt(rawEf, 0, lastFrame);
    if (ef < sf) return;
    emit(sf, ef, s, e, label, source);
  };
  // From exact inclusive frame indices (kinematics CSV). endSec spans through
  // the end of the last ON frame so a single-frame run has positive duration
  // and survives EAF / episodes re-export (which drop endSec <= startSec).
  const pushFrames = (sf0: number, ef0: number, label?: string, source?: string): void => {
    const sf = clampInt(Math.min(sf0, ef0), 0, lastFrame);
    const ef = clampInt(Math.max(sf0, ef0), 0, lastFrame);
    if (ef < sf) return;
    emit(sf, ef, sf / fps, (ef + 1) / fps, label, source);
  };

  const trimmed = text.trim();
  if (!trimmed) return out;

  // 1) ELAN .eaf / XML
  if (/\.(eaf|xml)$/i.test(fileName) || trimmed.includes("<ANNOTATION_DOCUMENT")) {
    const segs = parseEaf(text);
    if (segs.length > 0) {
      // When the document distinguishes tiers (a viewer export carries
      // annotated / model / ground-truth tiers) prefer the most authoritative
      // one so re-import lands cleanly instead of stacking every tier:
      // ground-truth > annotated/manual > everything. Whole-word matching keeps
      // unrelated external tier names ("groundwork", "Goldberg") from matching.
      const isGround = (t: string) => /\b(ground[ _-]?truth|gold|reference|truth)\b/i.test(t);
      const isAnnotated = (t: string) => /\b(annotat\w*|manual|human|corrected|reviewed)\b/i.test(t);
      const keep = segs.some((s) => isGround(s.tier))
        ? (t: string) => isGround(t)
        : segs.some((s) => isAnnotated(s.tier))
          ? (t: string) => isAnnotated(t)
          : () => true;
      for (const s of segs) if (keep(s.tier)) push(s.startSec, s.endSec, s.label, s.tier);
      return out;
    }
    // Not a real EAF — fall through and try JSON / CSV.
  }

  // 2) JSON
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    let data: unknown;
    try {
      data = JSON.parse(trimmed);
    } catch {
      return out;
    }
    const container = data as Record<string, unknown>;
    // Pick the first key holding a NON-EMPTY array (preserving precedence) — a
    // nullish-coalescing chain would stop at an empty `labels: []` and hide a
    // populated `episodes`, since [] is neither null nor undefined.
    let arr: unknown[] = Array.isArray(data) ? data : [];
    if (!Array.isArray(data)) {
      const keys = ["labels", "episodes", "events", "annotations", "fog", "segments", "intervals", "goldTruth", "ground_truth"];
      for (const k of keys) {
        const v = container[k];
        if (Array.isArray(v) && v.length > 0) {
          arr = v;
          break;
        }
      }
    }
    for (const item of arr) {
      if (Array.isArray(item) && item.length >= 2) {
        push(toNumber(item[0]), toNumber(item[1]), typeof item[2] === "string" ? item[2] : undefined);
        continue;
      }
      if (!item || typeof item !== "object") continue;
      const obj = item as Record<string, unknown>;
      const source = typeof obj.source === "string" ? obj.source : undefined;
      push(pickSeconds(obj, "start", fps), pickSeconds(obj, "end", fps), pickLabel(obj), source);
    }
    return out;
  }

  // 3) CSV / TSV
  const rows = trimmed.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (rows.length === 0) return out;
  const splitRow = (l: string) => l.split(/[,;\t]/).map((c) => c.trim());
  const isNumeric = (c: string | undefined) => c !== undefined && c !== "" && Number.isFinite(Number(c));
  const header = splitRow(rows[0]);
  const headerCells = header.map((c) => c.toLowerCase());
  // It's a header row only when the first two cells aren't a numeric start/end
  // pair — so a headerless "1.0,2.0,fog" row (text label in column 3) is still
  // treated as data, not mistaken for a header.
  const firstIsHeader = !(isNumeric(header[0]) && isNumeric(header[1]));

  // 3a) The per-frame kinematics CSV the viewer exports — reconstruct FoG
  //     segments from runs of the fog column (line 2 is a units row). Prefer
  //     the human annotation layer (fog_annotated) over the model flag.
  const annotatedColIdx = headerCells.indexOf("fog_annotated");
  const detectedColIdx = headerCells.indexOf("fog_detected");
  const fogCol = annotatedColIdx >= 0 ? annotatedColIdx : detectedColIdx;
  const frameCol = headerCells.indexOf("frame_index");
  const tsCol = headerCells.indexOf("timestamp_ms");
  if (firstIsHeader && fogCol >= 0 && (frameCol >= 0 || tsCol >= 0)) {
    let startFrame: number | null = null;
    let lastOnFrame = -1;
    for (let i = 1; i < rows.length; i += 1) {
      const cells = splitRow(rows[i]);
      const hasFrame = frameCol >= 0 && isNumeric(cells[frameCol]);
      const hasTs = tsCol >= 0 && isNumeric(cells[tsCol]);
      if (!hasFrame && !hasTs) continue; // units row or blank line
      const idx = hasFrame ? Math.round(Number(cells[frameCol])) : Math.round((Number(cells[tsCol]) / 1000) * fps);
      const cell = cells[fogCol];
      const on = isNumeric(cell) ? Number(cell) >= 0.5 : /^(true|yes|fog)$/i.test(cell ?? "");
      if (on) {
        if (startFrame === null) startFrame = idx;
        lastOnFrame = idx;
      } else if (startFrame !== null) {
        pushFrames(startFrame, lastOnFrame, "fog", "csv");
        startFrame = null;
      }
    }
    if (startFrame !== null) pushFrames(startFrame, lastOnFrame, "fog", "csv");
    return out;
  }

  // 3b) A CSV with named start/end columns (seconds, ms or frames).
  if (firstIsHeader) {
    const findCol = (names: string[]) => headerCells.findIndex((h) => names.includes(h));
    const sCol = findCol(["start", "start_sec", "startsec", "start_s", "start_time", "onset", "begin"]);
    const eCol = findCol(["end", "end_sec", "endsec", "end_s", "end_time", "offset", "stop"]);
    const sMsCol = findCol(["start_ms", "startms", "onset_ms"]);
    const eMsCol = findCol(["end_ms", "endms", "offset_ms"]);
    const sFrCol = findCol(["start_frame", "startframe", "start_frame_index", "startframeindex"]);
    const eFrCol = findCol(["end_frame", "endframe", "end_frame_index", "endframeindex"]);
    const lCol = findCol(["label", "value", "type", "event", "class"]);
    const haveStart = sCol >= 0 || sMsCol >= 0 || sFrCol >= 0;
    const haveEnd = eCol >= 0 || eMsCol >= 0 || eFrCol >= 0;
    if (haveStart && haveEnd) {
      const at = (cells: string[], col: number, div: number) =>
        col >= 0 && isNumeric(cells[col]) ? Number(cells[col]) / div : NaN;
      for (let i = 1; i < rows.length; i += 1) {
        const cells = splitRow(rows[i]);
        let startSec = at(cells, sCol, 1);
        if (!Number.isFinite(startSec)) startSec = at(cells, sMsCol, 1000);
        if (!Number.isFinite(startSec)) startSec = at(cells, sFrCol, 1) / fps;
        let endSec = at(cells, eCol, 1);
        if (!Number.isFinite(endSec)) endSec = at(cells, eMsCol, 1000);
        if (!Number.isFinite(endSec)) endSec = at(cells, eFrCol, 1) / fps;
        push(startSec, endSec, lCol >= 0 ? cells[lCol] : undefined);
      }
      return out;
    }
  }

  // 3c) Positional fallback: "start,end[,label]" rows in seconds.
  for (let i = firstIsHeader ? 1 : 0; i < rows.length; i += 1) {
    const cells = splitRow(rows[i]);
    if (cells.length < 2) continue;
    const a = Number(cells[0]);
    const b = Number(cells[1]);
    if (Number.isFinite(a) && Number.isFinite(b)) {
      push(a, b, typeof cells[2] === "string" ? cells[2] : undefined);
    }
  }
  return out;
}

// Round to an integer and clamp into [lo, hi]; non-finite input falls back to lo.
function clampInt(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, Math.round(value)));
}

// Build the canonical "episodes" array (kinesia.labels.v1 / Python evaluator
// shape: {label, start_ms, end_ms}) from second-based segments, so an exported
// annotation file doubles as a usable ground-truth label file.
export function segmentsToEpisodes(
  segments: Array<{ startSec: number; endSec: number; label?: string }>,
): Array<{ label: string; start_ms: number; end_ms: number }> {
  return segments
    .filter((s) => s.endSec > s.startSec)
    .map((s) => ({
      label: s.label || "fog",
      start_ms: Math.max(0, Math.round(s.startSec * 1000)),
      end_ms: Math.max(0, Math.round(s.endSec * 1000)),
    }));
}

// Sanitize a string into a filesystem-safe download name (kept short, with a
// non-empty fallback).
export function safeFileName(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 120) || "kinesia";
}
