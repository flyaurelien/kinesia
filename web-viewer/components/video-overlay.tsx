"use client";

// Tracking overlay drawn on top of the original video: one SAM3 subject box
// PER SUBJECT of the selection (the primary run + its sibling runs), each in
// its detect-step palette colour. Boxes are stored in original-video pixel
// coordinates (bbox_xyxy) and are therefore pixel-exact. (The skeleton overlay
// was removed — projecting the 3D joints back onto the video is only
// approximate because the model's exact 2D keypoints aren't stored; the
// accurate skeleton lives in the 3D view.)

import { useEffect, useMemo, useRef, useState } from "react";
import type { RunFrame } from "../lib/types";

export type OverlaySubject = {
  frame: RunFrame | null;
  color?: string | null;
  label?: string | null;
};

type Props = {
  subjects: OverlaySubject[];
  videoWidth: number | null;
  videoHeight: number | null;
};

const DEFAULT_BOX_COLOR = "#facc15";

function subjectBox(frame: RunFrame | null): [number, number, number, number] | null {
  if (!frame || !frame.bbox) return null;
  // No box when the subject isn't on this frame (absent/lost/masked) — a stale
  // box must never be drawn over a frame with no person.
  if (frame.subjectPresent === false) return null;
  const [x1, y1, x2, y2] = frame.bbox;
  if (!(x2 > x1) || !(y2 > y1)) return null; // degenerate / zero-area
  return frame.bbox;
}

// Overlays each subject's bounding box on the video, mapping the boxes from
// original-video pixels into the letterboxed (object-fit: contain) display rect.
export function VideoTrackingOverlay({ subjects, videoWidth, videoHeight }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const update = () => setSize({ w: host.clientWidth, h: host.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(host);
    return () => ro.disconnect();
  }, []);

  const boxes = useMemo(
    () =>
      subjects
        .map((subject) => ({
          box: subjectBox(subject.frame),
          color: subject.color || DEFAULT_BOX_COLOR,
          label: subject.label ?? null,
          score: subject.frame?.trackingScore ?? null,
        }))
        .filter(
          (entry): entry is typeof entry & { box: [number, number, number, number] } =>
            entry.box !== null,
        ),
    [subjects],
  );

  if (!videoWidth || !videoHeight) return <div ref={hostRef} className="video-overlay" />;

  // Content rect of the video within the pane (object-fit: contain).
  const { w: pw, h: ph } = size;
  let cw = pw;
  let ch = ph;
  let ox = 0;
  let oy = 0;
  if (pw > 0 && ph > 0) {
    const va = videoWidth / videoHeight;
    const pa = pw / ph;
    if (va > pa) {
      cw = pw;
      ch = pw / va;
      oy = (ph - ch) / 2;
    } else {
      ch = ph;
      cw = ph * va;
      ox = (pw - cw) / 2;
    }
  }

  return (
    <div ref={hostRef} className="video-overlay">
      {boxes.length > 0 && pw > 0 ? (
        <svg
          className="video-overlay-svg"
          style={{ left: ox, top: oy, width: cw, height: ch }}
          viewBox={`0 0 ${videoWidth} ${videoHeight}`}
          preserveAspectRatio="none"
        >
          {boxes.map((entry, index) => (
            <rect
              key={index}
              className="vo-box"
              style={{ stroke: entry.color }}
              x={entry.box[0]}
              y={entry.box[1]}
              width={Math.max(0, entry.box[2] - entry.box[0])}
              height={Math.max(0, entry.box[3] - entry.box[1])}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : null}
      {pw > 0
        ? boxes.map((entry, index) =>
            entry.label || entry.score != null ? (
              <div
                key={`chip-${index}`}
                className="vo-score"
                style={{
                  left: ox + (entry.box[2] / videoWidth) * cw,
                  top: oy + (entry.box[1] / videoHeight) * ch,
                  color: entry.color,
                }}
              >
                {[entry.label, entry.score != null ? `${Math.round(entry.score * 100)}%` : null]
                  .filter(Boolean)
                  .join(" · ")}
              </div>
            ) : null,
          )
        : null}
    </div>
  );
}

export default VideoTrackingOverlay;
