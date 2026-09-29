"use client";

import { useEffect, useRef, useState, type PointerEvent } from "react";

import { clock as formatClock } from "@/lib/format";
import type { PersonEntry } from "@/lib/scene/types";
import type { PlaybackClock } from "@/lib/viewer/clock";
import { Pause, Play, StepBack, StepForward } from "../icons";

const RATES = [0.25, 0.5, 1, 2];

type Props = {
  clock: PlaybackClock;
  frames: number;
  people: PersonEntry[];
  labels: Record<number, string>;
  hidden: Set<number>;
  selected: number | null;
  onSelect: (id: number | null) => void;
};

function usePlaying(clock: PlaybackClock) {
  const [state, setState] = useState({ playing: clock.playing, rate: clock.rate });
  useEffect(() => {
    const update = () => setState({ playing: clock.playing, rate: clock.rate });
    const unsubscribe = clock.subscribe(update);
    const timer = setInterval(update, 250); // catches the video ending on its own
    return () => {
      unsubscribe();
      clearInterval(timer);
    };
  }, [clock]);
  return state;
}

export function Timeline({ clock, frames, people, labels, hidden, selected, onSelect }: Props) {
  const { playing, rate } = usePlaying(clock);
  const track = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const time = useRef<HTMLSpanElement>(null);
  const [dragging, setDragging] = useState(false);
  const duration = clock.duration;

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const t = clock.now();
      if (head.current) head.current.style.left = `${(100 * t) / duration}%`;
      if (time.current) time.current.textContent = formatClock(t);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [clock, duration]);

  function seekTo(event: PointerEvent) {
    const el = track.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    clock.seek(f * duration);
  }

  const ordered = [...people].sort((a, b) => a.first - b.first);
  const ticks: number[] = [];
  const step = duration > 120 ? 30 : duration > 40 ? 10 : duration > 12 ? 5 : 1;
  for (let t = 0; t <= duration + 1e-6; t += step) ticks.push(t);

  return (
    <div className="timeline">
      <div className="transport">
        <button className="icon-btn" onClick={() => clock.step(-1)} title="Previous frame (←)">
          <StepBack />
        </button>
        <button className="play-btn" onClick={() => clock.toggle()} title="Play / pause (space)">
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button className="icon-btn" onClick={() => clock.step(1)} title="Next frame (→)">
          <StepForward />
        </button>
        <span className="time tabular">
          <span ref={time}>00:00.0</span>
          <span className="faint"> / {formatClock(duration)}</span>
        </span>
        <div className="segmented rates">
          {RATES.map((r) => (
            <button key={r} className={r === rate ? "on" : ""} onClick={() => clock.setRate(r)}>
              {r}×
            </button>
          ))}
        </div>
      </div>

      <div className="lanes-wrap">
        <div className="lanes-labels">
          <div className="ruler-spacer" />
          {ordered.map((p) => (
            <button
              key={p.id}
              className={`lane-label${selected === p.id ? " on" : ""}${hidden.has(p.id) ? " off" : ""}`}
              onClick={() => onSelect(selected === p.id ? null : p.id)}
            >
              <i style={{ background: p.color }} />
              {labels[p.id] ?? p.label}
            </button>
          ))}
        </div>
        <div
          className={`lanes${dragging ? " dragging" : ""}`}
          ref={track}
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
            setDragging(true);
            seekTo(e);
          }}
          onPointerMove={(e) => dragging && seekTo(e)}
          onPointerUp={() => setDragging(false)}
        >
          <div className="ruler">
            {ticks.map((t) => (
              <span key={t} className="tick" style={{ left: `${(100 * t) / duration}%` }}>
                {formatClock(t).replace(/\.\d$/, "")}
              </span>
            ))}
          </div>
          {ordered.map((p) => (
            <div key={p.id} className={`lane${selected === p.id ? " on" : ""}${hidden.has(p.id) ? " off" : ""}`}>
              {p.segments.map(([a, b]) => (
                <span
                  key={a}
                  className="segment"
                  style={{
                    left: `${(100 * a) / frames}%`,
                    width: `${Math.max(0.15, (100 * (b - a + 1)) / frames)}%`,
                    background: p.color,
                  }}
                />
              ))}
            </div>
          ))}
          <div className="playhead" ref={head} />
        </div>
      </div>
    </div>
  );
}
