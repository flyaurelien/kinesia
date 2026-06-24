"use client";

// Professional multi-signal kinematics plot built on uPlot (canvas).
//
// Two layout modes:
//   - "stacked": one synced strip per signal, each independently auto-scaled,
//     sharing a single time x-axis. This is the default — it guarantees that
//     signals with very different ranges (a hip angle vs an ankle speed) never
//     crush each other onto one shared scale.
//   - "overlay": a single plot, one auto-scaled y-axis per unit (deg on the
//     left, m/s on the right, ...). Best for comparing a few same-family signals.
//
// All strips share the time x-axis and a synced cursor; a playhead line tracks
// the current frame; clicking/dragging seeks. Masked gaps and FoG annotation /
// ground-truth segments are shaded behind the curves. onPlotBox reports the
// canvas plotting area (left + width in CSS px, relative to the component) so
// sibling timeline tracks can pixel-align to the exact same x-axis.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { RunSignal } from "../lib/types";

// Canvas plotting area in CSS px, relative to the component (for track alignment).
export type PlotBox = { left: number; width: number };
// A contiguous, frame-indexed span shaded behind the curves (annotation/ground truth).
type Seg = { startFrameIndex: number; endFrameIndex: number; source?: string };

type Props = {
  signals: RunSignal[];
  frameIndex: number;
  fps: number;
  frameCount: number;
  mode?: "stacked" | "overlay";
  // Shared zoom/scroll window (in frames). When set, the x-axis is locked to it
  // so the plot scrolls in sync with the annotation/ground-truth tracks.
  viewWindow?: { start: number; end: number } | null;
  annotationSegments?: Seg[];
  goldSegments?: Seg[];
  maskedRanges?: Array<[number, number]>;
  colorForId?: (id: string, index: number) => string;
  onFrameSelect?: (frameIndex: number) => void;
  onPlotBox?: (box: PlotBox | null) => void;
};

const PALETTE = [
  "#38bdf8",
  "#fb923c",
  "#22c55e",
  "#f43f5e",
  "#a78bfa",
  "#facc15",
  "#2dd4bf",
  "#f472b6",
  "#60a5fa",
  "#a3e635",
];

const AXIS_STROKE = "#7e8aa3";
const GRID_STROKE = "rgba(148, 163, 184, 0.12)";
const PLOT_BG = "rgba(5, 9, 21, 0.55)";

// Format a live readout value: em-dash for missing/non-finite, adaptive decimals.
function fmtNum(v: number | null | undefined, unit: string): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  const dp = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return `${v.toFixed(dp)}${unit ? ` ${unit}` : ""}`;
}

// Format an x-axis time value as seconds (under a minute) or m:ss.
function fmtTime(sec: number): string {
  if (!Number.isFinite(sec)) return "0s";
  if (sec < 60) {
    // Drop trailing zeros: 5, 12.5, 0.25
    return `${Number(sec.toFixed(2))}s`;
  }
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Clean y-axis tick: no thousands separators, adaptive decimals.
function fmtAxisTick(v: number): string {
  if (!Number.isFinite(v)) return "";
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

// Pad a data range by a fraction so the curve never touches the strip edges.
function paddedRange(dmin: number | null, dmax: number | null): [number, number] {
  if (dmin == null || dmax == null || !Number.isFinite(dmin) || !Number.isFinite(dmax)) {
    return [0, 1];
  }
  if (dmin === dmax) {
    const e = Math.abs(dmin) > 1e-9 ? Math.abs(dmin) * 0.1 : 1;
    return [dmin - e, dmax + e];
  }
  const pad = (dmax - dmin) * 0.08;
  return [dmin - pad, dmax + pad];
}

// Synced multi-signal kinematics chart (see file header for layout modes).
export function KinematicsPlot({
  signals,
  frameIndex,
  fps,
  frameCount,
  mode = "stacked",
  viewWindow,
  annotationSegments,
  goldSegments,
  maskedRanges,
  colorForId,
  onFrameSelect,
  onPlotBox,
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const plotsRef = useRef<uPlot[]>([]);
  const sizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });

  // Latest props for the canvas plugins (they read these at draw time so we can
  // redraw without rebuilding the chart).
  const overlaysRef = useRef({ annotationSegments, goldSegments, maskedRanges, fps });
  overlaysRef.current = { annotationSegments, goldSegments, maskedRanges, fps };
  const seekRef = useRef(onFrameSelect);
  seekRef.current = onFrameSelect;
  const plotBoxRef = useRef(onPlotBox);
  plotBoxRef.current = onPlotBox;
  const viewWindowRef = useRef(viewWindow);
  viewWindowRef.current = viewWindow;
  // One AbortController per chart's seek listeners, so we can detach them when
  // charts are rebuilt or the component unmounts (no listener accumulation).
  const seekAbortRef = useRef<AbortController[]>([]);

  const safeFps = Math.max(1, fps || 30);
  const times = useMemo(() => {
    const xs = new Float64Array(frameCount);
    for (let i = 0; i < frameCount; i += 1) xs[i] = i / safeFps;
    return xs;
  }, [frameCount, safeFps]);

  const color = useCallback(
    (id: string, index: number) => colorForId?.(id, index) ?? PALETTE[index % PALETTE.length],
    [colorForId],
  );

  // Identity key: rebuild charts when the signal set, mode, frame count, OR fps
  // changes — fps feeds the x-scale (time = frame / fps) and the click-to-seek
  // conversion, so a stale fps would mis-map clicks and the playhead.
  const signalKey = signals.map((s) => s.id).join("|") + "::" + mode + "::" + frameCount + "::" + Math.round(safeFps * 100);

  // Shade masked gaps + annotation/ground-truth segments behind the series.
  const drawBands = useCallback((u: uPlot) => {
    const ctx = u.ctx;
    const { annotationSegments: ann, goldSegments: gold, maskedRanges: masked, fps: f } = overlaysRef.current;
    const sf = Math.max(1, f || 30);
    const top = u.bbox.top;
    const height = u.bbox.height;
    const xToPx = (frame: number) => Math.round(u.valToPos(frame / sf, "x", true));
    const paint = (segs: Seg[] | undefined, fill: string) => {
      if (!segs) return;
      ctx.save();
      ctx.fillStyle = fill;
      for (const s of segs) {
        const x1 = xToPx(s.startFrameIndex);
        const x2 = xToPx(s.endFrameIndex + 1);
        ctx.fillRect(x1, top, Math.max(1, x2 - x1), height);
      }
      ctx.restore();
    };
    if (masked) {
      ctx.save();
      ctx.fillStyle = "rgba(148, 163, 184, 0.10)";
      for (const [a, b] of masked) {
        const x1 = xToPx(a);
        const x2 = xToPx(b + 1);
        ctx.fillRect(x1, top, Math.max(1, x2 - x1), height);
      }
      ctx.restore();
    }
    paint(gold, "rgba(250, 204, 21, 0.10)");
    paint(ann, "rgba(56, 189, 248, 0.12)");
  }, []);

  // (Re)build the uPlot instances.
  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const w = Math.max(120, Math.floor(host.clientWidth));
    sizeRef.current.w = w;

    // Destroy any existing instances and detach their seek listeners.
    for (const c of seekAbortRef.current) c.abort();
    seekAbortRef.current = [];
    for (const p of plotsRef.current) p.destroy();
    plotsRef.current = [];
    host.innerHTML = "";

    if (signals.length === 0 || frameCount === 0) {
      plotBoxRef.current?.(null);
      return;
    }

    const totalH = Math.max(120, Math.floor(host.clientHeight));
    sizeRef.current.h = totalH;

    const bandsPlugin = { hooks: { drawClear: (u: uPlot) => drawBands(u) } };

    const xAxis = (showLabels: boolean) => ({
      scale: "x",
      stroke: AXIS_STROKE,
      grid: { stroke: GRID_STROKE, width: 1 },
      ticks: { stroke: GRID_STROKE, width: 1, size: 4 },
      font: "9px ui-sans-serif, system-ui, sans-serif",
      size: showLabels ? 26 : 6,
      gap: 4,
      values: (_u: uPlot, vals: number[]) => (showLabels ? vals.map((v) => fmtTime(v)) : vals.map(() => "")),
    });

    const makeCursor = () => ({
      sync: { key: "kinesia-kin", setSeries: false },
      drag: { x: true, y: false, uni: 12 },
      points: { size: 6 },
      focus: { prox: 24 },
    });

    const xFull: [number, number] = [times[0] ?? 0, times[times.length - 1] ?? 1];
    const onReadyBindSeek = (u: uPlot) => {
      const over = u.over;
      const ac = new AbortController();
      seekAbortRef.current.push(ac);
      const opt: AddEventListenerOptions = { signal: ac.signal };
      let dragging = false;
      let moved = false;
      const seekAt = (clientX: number) => {
        const rect = over.getBoundingClientRect();
        const left = clientX - rect.left;
        const t = u.posToVal(left, "x");
        const fr = Math.round(t * Math.max(1, overlaysRef.current.fps || 30));
        seekRef.current?.(Math.max(0, Math.min(frameCount - 1, fr)));
      };
      // Plain drag/click scrubs the playhead; shift-drag zooms (uPlot native).
      over.addEventListener("mousedown", () => {
        dragging = true;
        moved = false;
      }, opt);
      over.addEventListener("mousemove", (e) => {
        if (dragging && !(e as MouseEvent).shiftKey) {
          moved = true;
          seekAt((e as MouseEvent).clientX);
        }
      }, opt);
      const endDrag = () => {
        dragging = false;
      };
      over.addEventListener("mouseup", endDrag, opt);
      over.addEventListener("mouseleave", endDrag, opt);
      over.addEventListener("click", (e) => {
        if (!moved) seekAt((e as MouseEvent).clientX);
      }, opt);
      over.addEventListener("dblclick", () => {
        u.setScale("x", { min: xFull[0], max: xFull[1] });
      }, opt);
    };

    if (mode === "overlay") {
      // One scale + axis per unit.
      const units = Array.from(new Set(signals.map((s) => s.unit || "")));
      const scales: uPlot.Options["scales"] = { x: { time: false } };
      const axes: uPlot.Axis[] = [xAxis(true)];
      units.forEach((unit, ui) => {
        const key = unit || `u${ui}`;
        scales[key] = { range: (_u, dmin, dmax) => paddedRange(dmin, dmax) };
        axes.push({
          scale: key,
          side: ui % 2 === 0 ? 3 : 1,
          stroke: AXIS_STROKE,
          grid: { show: ui === 0, stroke: GRID_STROKE, width: 1 },
          ticks: { stroke: GRID_STROKE, width: 1, size: 4 },
          font: "9px ui-sans-serif, system-ui, sans-serif",
          size: 56,
          label: unit || undefined,
          labelSize: unit ? 14 : 0,
          labelFont: "11px ui-sans-serif, system-ui, sans-serif",
          values: (_u: uPlot, vals: number[]) => vals.map((v) => fmtAxisTick(v)),
        });
      });
      const series: uPlot.Series[] = [{}];
      signals.forEach((s, i) => {
        series.push({
          label: s.label,
          scale: s.unit || `u${units.indexOf(s.unit || "")}`,
          stroke: color(s.id, i),
          width: 1.6,
          points: { show: false },
          spanGaps: false,
        });
      });
      const data: uPlot.AlignedData = [times, ...signals.map((s) => Float64ArrayFromValues(s.values))];
      const u = new uPlot(
        {
          width: w,
          height: totalH,
          padding: [6, 18, null, null],
          scales,
          axes,
          series,
          legend: { show: true, live: true },
          cursor: makeCursor(),
          plugins: [bandsPlugin],
          hooks: { ready: [onReadyBindSeek] },
        },
        data,
        host,
      );
      plotsRef.current = [u];
    } else {
      // Stacked: one strip per signal, each its own auto-scaled y.
      const n = signals.length;
      const gap = 4;
      const stripH = Math.max(26, Math.floor((totalH - gap * (n - 1)) / n));
      signals.forEach((s, i) => {
        const isLast = i === n - 1;
        const yKey = "y";
        const u = new uPlot(
          {
            width: w,
            height: isLast ? stripH + 22 : stripH,
            padding: [4, 18, isLast ? 2 : 0, null],
            scales: {
              x: { time: false },
              [yKey]: { range: (_u, dmin, dmax) => paddedRange(dmin, dmax) },
            },
            axes: [
              xAxis(isLast),
              {
                scale: yKey,
                side: 3,
                stroke: AXIS_STROKE,
                grid: { stroke: GRID_STROKE, width: 1 },
                ticks: { stroke: GRID_STROKE, width: 1, size: 3 },
                font: "9px ui-sans-serif, system-ui, sans-serif",
                size: 56,
                splits: (u2) => {
                  const sc = u2.scales[yKey];
                  const lo = sc.min ?? 0;
                  const hi = sc.max ?? 1;
                  const mid = (lo + hi) / 2;
                  return [lo + (hi - lo) * 0.12, mid, hi - (hi - lo) * 0.12];
                },
                values: (_u, vals) => vals.map((v) => fmtAxisTick(v)),
              },
            ],
            series: [
              {},
              {
                label: s.label,
                scale: yKey,
                stroke: color(s.id, i),
                fill: hexToRgba(color(s.id, i), 0.08),
                width: 1.7,
                points: { show: false },
                spanGaps: false,
              },
            ],
            legend: { show: false },
            cursor: makeCursor(),
            plugins: [bandsPlugin],
            hooks: { ready: [onReadyBindSeek] },
          },
          [times, Float64ArrayFromValues(s.values)],
          host,
        );
        plotsRef.current.push(u);
      });
    }

    // Lock the x-axis to the shared zoom/scroll window right away (no full-range
    // flash) so the plot lines up with the annotation/ground-truth tracks.
    {
      const win = viewWindowRef.current;
      const xmin = win ? win.start / safeFps : xFull[0];
      const xmax = win ? (win.end + 1) / safeFps : xFull[1];
      for (const u of plotsRef.current) {
        try {
          u.setScale("x", { min: xmin, max: xmax });
        } catch {
          // ignore
        }
      }
    }

    // Report the plotting box of the first chart for track alignment.
    const first = plotsRef.current[0];
    if (first) {
      const left = first.bbox.left / devicePixelRatioSafe();
      const width = first.bbox.width / devicePixelRatioSafe();
      plotBoxRef.current?.({ left, width });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signalKey, drawBands]);

  // Re-apply the shared window when it changes (pan/zoom) without rebuilding.
  useEffect(() => {
    const xmin = viewWindow ? viewWindow.start / safeFps : times[0] ?? 0;
    const xmax = viewWindow ? (viewWindow.end + 1) / safeFps : times[times.length - 1] ?? 1;
    for (const u of plotsRef.current) {
      try {
        u.setScale("x", { min: xmin, max: xmax });
      } catch {
        // ignore
      }
    }
  }, [viewWindow, safeFps, times]);

  // Responsive resize.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const ro = new ResizeObserver(() => {
      const w = Math.max(120, Math.floor(host.clientWidth));
      const totalH = Math.max(120, Math.floor(host.clientHeight));
      if (Math.abs(w - sizeRef.current.w) < 1 && Math.abs(totalH - sizeRef.current.h) < 1) return;
      sizeRef.current = { w, h: totalH };
      const plots = plotsRef.current;
      if (plots.length === 0) return;
      if (plots.length === 1) {
        plots[0].setSize({ width: w, height: totalH });
      } else {
        const n = plots.length;
        const gap = 4;
        const stripH = Math.max(26, Math.floor((totalH - gap * (n - 1)) / n));
        plots.forEach((p, i) => p.setSize({ width: w, height: i === n - 1 ? stripH + 22 : stripH }));
      }
      const first = plots[0];
      if (first) {
        plotBoxRef.current?.({ left: first.bbox.left / devicePixelRatioSafe(), width: first.bbox.width / devicePixelRatioSafe() });
      }
    });
    ro.observe(host);
    return () => ro.disconnect();
  }, []);

  // Position the playhead within a chart's plotting area using the x-scale
  // fraction (robust against zoom and uPlot coordinate quirks).
  const positionPlayhead = useCallback(
    (u: uPlot, frame: number) => {
      const el = u.over.querySelector<HTMLDivElement>(".kin-playhead");
      if (!el) return;
      const sc = u.scales.x;
      const min = sc.min ?? 0;
      const max = sc.max ?? 1;
      const w = u.over.clientWidth;
      const t = frame / safeFps;
      if (max <= min || w <= 0) {
        el.style.display = "none";
        return;
      }
      if (t < min - (max - min) * 0.002 || t > max + (max - min) * 0.002) {
        el.style.display = "none";
        return;
      }
      el.style.display = "block";
      // Use uPlot's own value→pixel mapping (CSS px in the plotting area) so the
      // playhead sits exactly where the data does — and lines up with the tracks,
      // which inset their data area by the same axis width.
      el.style.transform = `translateX(${u.valToPos(t, "x")}px)`;
    },
    [safeFps],
  );

  // Update the per-strip label (name + live value) in stacked mode.
  const updateStripLabel = useCallback(
    (u: uPlot, i: number, frame: number) => {
      if (mode !== "stacked") return;
      const s = signals[i];
      if (!s) return;
      const nameEl = u.over.querySelector<HTMLElement>(".kin-strip-name");
      const valEl = u.over.querySelector<HTMLElement>(".kin-strip-val");
      if (nameEl) nameEl.textContent = s.label;
      if (valEl) valEl.textContent = fmtNum(s.values[frame], s.unit);
    },
    [signals, mode],
  );

  // Move the playhead + refresh strip values when the current frame changes.
  useEffect(() => {
    plotsRef.current.forEach((u, i) => {
      positionPlayhead(u, frameIndex);
      updateStripLabel(u, i, frameIndex);
    });
  }, [frameIndex, positionPlayhead, updateStripLabel, signalKey]);

  // Inject playhead + strip-label elements into each chart once built.
  useEffect(() => {
    plotsRef.current.forEach((u, i) => {
      if (!u.over.querySelector(".kin-playhead")) {
        const ph = document.createElement("div");
        ph.className = "kin-playhead";
        u.over.appendChild(ph);
      }
      if (mode === "stacked" && !u.over.querySelector(".kin-strip-label")) {
        const lbl = document.createElement("div");
        lbl.className = "kin-strip-label";
        const name = document.createElement("span");
        name.className = "kin-strip-name";
        const val = document.createElement("span");
        val.className = "kin-strip-val";
        lbl.appendChild(name);
        lbl.appendChild(val);
        // color dot
        const dot = document.createElement("span");
        dot.className = "kin-strip-dot";
        dot.style.background = color(signals[i]?.id ?? "", i);
        lbl.insertBefore(dot, name);
        u.over.appendChild(lbl);
      }
      positionPlayhead(u, frameIndex);
      updateStripLabel(u, i, frameIndex);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signalKey]);

  // Redraw bands when overlay data (or fps, which maps frames→x) changes.
  useEffect(() => {
    for (const u of plotsRef.current) u.redraw(false, false);
  }, [annotationSegments, goldSegments, maskedRanges, safeFps]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      for (const c of seekAbortRef.current) c.abort();
      seekAbortRef.current = [];
      for (const p of plotsRef.current) p.destroy();
      plotsRef.current = [];
    };
  }, []);

  return (
    <div className="kin-plot" style={{ background: PLOT_BG }}>
      <div ref={hostRef} className="kin-plot-host" />
      {signals.length === 0 ? <div className="kin-plot-empty">Select one or more signals to plot.</div> : null}
    </div>
  );
}

// Pass series values straight through to uPlot (which accepts (number|null)[];
// nulls render as gaps). Kept as a named helper to document that intent.
function Float64ArrayFromValues(values: Array<number | null>): (number | null)[] {
  return values;
}

// Device pixel ratio, defaulting to 1 when running without a window (SSR).
function devicePixelRatioSafe(): number {
  if (typeof window === "undefined") return 1;
  return window.devicePixelRatio || 1;
}

// Convert a #rrggbb hex color to an rgba() string; falls back to sky blue.
function hexToRgba(hex: string, alpha: number): string {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return `rgba(56,189,248,${alpha})`;
  const r = parseInt(m[1], 16);
  const g = parseInt(m[2], 16);
  const b = parseInt(m[3], 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default KinematicsPlot;
