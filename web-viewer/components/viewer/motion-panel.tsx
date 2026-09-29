"use client";

import "uplot/dist/uPlot.min.css";

import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";

import { number } from "@/lib/format";
import type { LoadedScene } from "@/lib/scene/load";
import type { PlaybackClock } from "@/lib/viewer/clock";
import { Close, Download } from "../icons";

type Props = {
  data: LoadedScene;
  clock: PlaybackClock;
  selected: number | null;
  labels: Record<number, string>;
  onSelect: (id: number | null) => void;
};

const SERIES_ORDER = ["speed", "height", "left_knee", "right_knee", "left_hip", "right_hip", "left_elbow", "right_elbow", "trunk_lean"];
const SHORT: Record<string, string> = {
  speed: "Speed",
  height: "Pelvis height",
  left_knee: "Knee L",
  right_knee: "Knee R",
  left_hip: "Hip L",
  right_hip: "Hip R",
  left_elbow: "Elbow L",
  right_elbow: "Elbow R",
  trunk_lean: "Trunk lean",
};

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** x (seconds) and y arrays for one person's series, with gaps between visible spans. */
function seriesData(data: LoadedScene, person: number, key: string): [number[], (number | null)[]] {
  const track = data.people[person];
  const values = data.metrics?.people[person]?.series[key] ?? [];
  const fps = data.scene.fps;
  const xs: number[] = [];
  const ys: (number | null)[] = [];
  for (let i = 0; i < track.frames.length; i++) {
    if (i > 0 && track.frames[i] !== track.frames[i - 1] + 1) {
      xs.push((track.frames[i - 1] + 0.5) / fps);
      ys.push(null);
    }
    xs.push(track.frames[i] / fps);
    ys.push(values[i] ?? null);
  }
  return [xs, ys];
}

function Chart({ data, clock, person, seriesKey }: { data: LoadedScene; clock: PlaybackClock; person: number; seriesKey: string }) {
  const host = useRef<HTMLDivElement>(null);
  const line = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const colour = data.scene.people[person].color;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const [xs, ys] = seriesData(data, person, seriesKey);
    const options: uPlot.Options = {
      width: el.clientWidth,
      height: 150,
      legend: { show: false },
      cursor: { drag: { x: false, y: false }, points: { show: false } },
      scales: { x: { time: false, min: 0, max: clock.duration } },
      axes: [
        { stroke: "#858c99", grid: { stroke: "#eceff3" }, ticks: { stroke: "#e2e5ea" }, size: 28, values: (_u, v) => v.map((s) => `${s}s`) },
        { stroke: "#858c99", grid: { stroke: "#eceff3" }, ticks: { stroke: "#e2e5ea" }, size: 44 },
      ],
      series: [{}, { stroke: colour, width: 2, spanGaps: false, points: { show: false } }],
    };
    plot.current?.destroy();
    plot.current = new uPlot(options, [xs, ys as number[]], el);
    const resize = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: 150 }));
    resize.observe(el);
    return () => {
      resize.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, [data, person, seriesKey, colour, clock.duration]);

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const u = plot.current;
      if (!u || !line.current) return;
      const x = u.valToPos(clock.now(), "x");
      line.current.style.transform = `translateX(${u.bbox.left / devicePixelRatio + x}px)`;
      line.current.style.top = `${u.bbox.top / devicePixelRatio}px`;
      line.current.style.height = `${u.bbox.height / devicePixelRatio}px`;
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [clock]);

  return (
    <div className="chart" onClick={(e) => {
      const u = plot.current;
      if (!u) return;
      const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
      clock.seek(u.posToVal(e.clientX - rect.left - u.bbox.left / devicePixelRatio, "x"));
    }}>
      <div ref={host} />
      <div ref={line} className="chart-playhead" />
    </div>
  );
}

/** Every person's measures as one JSON file. */
export function exportAllMotion(data: LoadedScene, labels: Record<number, string>) {
  const payload = data.scene.people.map((p) => ({
    id: p.id,
    label: labels[p.id] ?? p.label,
    frames: Array.from(data.people[p.id].frames),
    summary: p.summary,
    series: data.metrics?.people[p.id]?.series ?? {},
  }));
  download("kinesia_motion.json", JSON.stringify({ fps: data.scene.fps, people: payload }), "application/json");
}

export function MotionPanel({ data, clock, selected, labels, onSelect }: Props) {
  const [seriesKey, setSeriesKey] = useState("speed");
  const people = data.scene.people;

  function exportCsv(person: number) {
    const track = data.people[person];
    const series = data.metrics?.people[person]?.series ?? {};
    const keys = SERIES_ORDER.filter((k) => series[k]);
    const rows = [["frame", "time_s", ...keys].join(",")];
    for (let i = 0; i < track.frames.length; i++) {
      rows.push([track.frames[i], (track.frames[i] / data.scene.fps).toFixed(3), ...keys.map((k) => series[k][i])].join(","));
    }
    const name = (labels[person] ?? people[person].label).replace(/\s+/g, "_");
    download(`${name}_motion.csv`, rows.join("\n"), "text/csv");
  }

  if (selected === null) return null;

  const person = people[selected];
  const s = person.summary;
  const available = SERIES_ORDER.filter((k) => data.metrics?.people[selected]?.series[k]);
  return (
    <section className="panel motion-panel">
      <header className="panel-head">
        <h3>
          <i className="swatch" style={{ background: person.color }} /> {labels[selected] ?? person.label}
        </h3>
        <span className="panel-actions">
          <button className="btn btn-sm btn-ghost" onClick={() => exportCsv(selected)} title="This person's measures as CSV">
            <Download size={14} /> CSV
          </button>
          <button className="icon-btn" onClick={() => onSelect(null)} title="Close (Esc)">
            <Close size={15} />
          </button>
        </span>
      </header>
      <p className="faint panel-note">Estimated from one camera: compare people within this clip rather than across recordings.</p>
      <div className="stats">
        <div><span>Seen</span><strong className="tabular">{number(s.visible_seconds, 1)} s</strong></div>
        <div><span>Distance</span><strong className="tabular">{number(s.distance_m, 0)} m</strong></div>
        <div><span>Mean speed</span><strong className="tabular">{number(s.mean_speed, 1)} m/s</strong></div>
        <div><span>Top speed</span><strong className="tabular">{number(s.top_speed, 1)} m/s</strong></div>
        <div><span>Height</span><strong className="tabular">{number(s.height_m, 2)} m</strong></div>
        <div>
          <span>Jumps</span>
          <strong className="tabular">
            {s.jumps.length}
            {s.jumps.length ? <small> · {Math.round(s.highest_jump_m * 100)} cm</small> : null}
          </strong>
        </div>
      </div>
      <div className="series-picker">
        {available.map((k) => (
          <button key={k} className={`pill${k === seriesKey ? " on" : ""}`} onClick={() => setSeriesKey(k)}>
            {SHORT[k] ?? k}
          </button>
        ))}
      </div>
      <Chart data={data} clock={clock} person={selected} seriesKey={available.includes(seriesKey) ? seriesKey : available[0] ?? "speed"} />
      {s.jumps.length > 0 && (
        <div className="jumps">
          {s.jumps.slice(0, 8).map((j) => (
            <button key={j.frame} className="pill" onClick={() => clock.seek(j.frame / data.scene.fps)}>
              Jump {Math.round(j.height * 100)} cm · {(j.frame / data.scene.fps).toFixed(1)} s
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
