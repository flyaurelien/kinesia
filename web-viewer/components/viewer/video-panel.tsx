"use client";

import { useEffect, useRef } from "react";

import type { OverlayFile } from "@/lib/scene/types";
import type { PlaybackClock } from "@/lib/viewer/clock";

type Props = {
  src: string;
  width: number;
  height: number;
  clock: PlaybackClock;
  overlay: OverlayFile | null;
  colors: string[];
  hidden: Set<number>;
  selected: number | null;
  showOutlines: boolean;
  onSelect: (id: number | null) => void;
  onVideo?: (video: HTMLVideoElement | null) => void;
};

function hexToRgba(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** The source video with each person's outline drawn in their colour. */
export function VideoPanel({ src, width, height, clock, overlay, colors, hidden, selected, showOutlines, onSelect, onVideo }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    clock.attach(video.current);
    onVideo?.(video.current);
    return () => {
      clock.attach(null);
      onVideo?.(null);
    };
  }, [clock, onVideo]);

  useEffect(() => {
    let raf = 0;
    let lastFrame = -1;
    let lastKey = "";
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const c = canvas.current;
      if (!c) return;
      const frame = Math.floor(clock.frame());
      const key = `${frame}|${selected}|${showOutlines}|${hidden.size}`;
      if (frame === lastFrame && key === lastKey) return;
      lastFrame = frame;
      lastKey = key;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, c.width, c.height);
      if (!overlay || !showOutlines) return;
      const entries = overlay.frames[frame] ?? [];
      for (const [person, x0, y0, x1, y1, polygons] of entries) {
        if (hidden.has(person)) continue;
        const colour = colors[person] ?? "#2d5bff";
        const active = selected === null || selected === person;
        ctx.lineWidth = selected === person ? 5 : 3;
        ctx.strokeStyle = hexToRgba(colour, active ? 0.95 : 0.35);
        ctx.fillStyle = hexToRgba(colour, selected === person ? 0.3 : active ? 0.16 : 0.06);
        for (const poly of polygons) {
          ctx.beginPath();
          for (let i = 0; i < poly.length; i += 2) {
            if (i === 0) ctx.moveTo(poly[i], poly[i + 1]);
            else ctx.lineTo(poly[i], poly[i + 1]);
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
        if (polygons.length === 0) {
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        }
        if (active) {
          const label = `${person + 1}`;
          ctx.font = "600 26px Inter, system-ui, sans-serif";
          const w = ctx.measureText(label).width + 16;
          ctx.fillStyle = hexToRgba(colour, 0.95);
          ctx.beginPath();
          ctx.roundRect(x0, Math.max(0, y0 - 34), w, 30, 8);
          ctx.fill();
          ctx.fillStyle = "white";
          ctx.fillText(label, x0 + 8, Math.max(0, y0 - 34) + 23);
        }
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [clock, overlay, colors, hidden, selected, showOutlines]);

  function click(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!overlay) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    const y = ((event.clientY - rect.top) / rect.height) * height;
    const entries = overlay.frames[Math.floor(clock.frame())] ?? [];
    let best: number | null = null;
    let bestArea = Infinity;
    for (const [person, x0, y0, x1, y1] of entries) {
      if (hidden.has(person)) continue;
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        const area = (x1 - x0) * (y1 - y0);
        if (area < bestArea) {
          best = person;
          bestArea = area;
        }
      }
    }
    onSelect(best === selected ? null : best);
  }

  return (
    <div className="video-panel" style={{ aspectRatio: `${width} / ${height}` }}>
      <video ref={video} src={src} muted playsInline preload="auto" />
      <canvas ref={canvas} width={width} height={height} onClick={click} />
    </div>
  );
}
