"use client";

// Tracking overlay drawn on top of the original video: the SAM3 subject box,
// which is stored in original-video pixel coordinates (bbox_xyxy) and is
// therefore pixel-exact. (The skeleton overlay was removed — projecting the 3D
// joints back onto the video is only approximate because the model's exact 2D
// keypoints aren't stored; the accurate skeleton lives in the 3D view.)

import { useEffect, useMemo, useRef, useState } from "react";
import type { RunFrame } from "../lib/types";

type Props = {
  frame: RunFrame | null;
  videoWidth: number | null;
  videoHeight: number | null;
};

// Overlays the subject bounding box on the video, mapping the box from
// original-video pixels into the letterboxed (object-fit: contain) display rect.
export function VideoTrackingOverlay({ frame, videoWidth, videoHeight }: Props) {
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

  const box = useMemo<[number, number, number, number] | null>(() => {
    if (!frame || !frame.bbox) return null;
    // No box when the subject isn't on this frame (absent/lost/masked) — a stale
    // box must never be drawn over a frame with no patient.
    if (frame.subjectPresent === false) return null;
    const [x1, y1, x2, y2] = frame.bbox;
    if (!(x2 > x1) || !(y2 > y1)) return null; // degenerate / zero-area
    return frame.bbox;
  }, [frame]);

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
      {box && pw > 0 ? (
        <svg
          className="video-overlay-svg"
          style={{ left: ox, top: oy, width: cw, height: ch }}
          viewBox={`0 0 ${videoWidth} ${videoHeight}`}
          preserveAspectRatio="none"
        >
          <rect
            className="vo-box"
            x={box[0]}
            y={box[1]}
            width={Math.max(0, box[2] - box[0])}
            height={Math.max(0, box[3] - box[1])}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      ) : null}
      {box && pw > 0 && frame?.trackingScore != null ? (
        <div
          className="vo-score"
          style={{
            left: ox + (box[2] / videoWidth) * cw,
            top: oy + (box[1] / videoHeight) * ch,
          }}
        >
          {Math.round(frame.trackingScore * 100)}%
        </div>
      ) : null}
    </div>
  );
}

export default VideoTrackingOverlay;
