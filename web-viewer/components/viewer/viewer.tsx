"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { bytes, duration } from "@/lib/format";
import type { RunSummary } from "@/lib/runs";
import { useScene } from "@/lib/scene/load";
import type { Edits } from "@/lib/scene/types";
import { PlaybackClock } from "@/lib/viewer/clock";
import { Back, Body, Bones, Camera, Grid, Orbit, Swap, Tag, Target, Trail, Trash, Video } from "../icons";
import { ClusterBadge } from "../topbar";
import { MotionPanel, exportAllMotion } from "./motion-panel";
import { PeoplePanel } from "./people-panel";
import { Stage, type CameraMode, type Display } from "./stage";
import { Timeline } from "./timeline";
import { VideoPanel } from "./video-panel";

type Props = { run: RunSummary; onRenamed: () => void; onDelete: () => void };

const CAMERA_MODES: { mode: CameraMode; label: string; icon: JSX.Element; key: string; hint: string }[] = [
  { mode: "orbit", label: "Orbit", icon: <Orbit size={14} />, key: "o", hint: "Free orbit around the scene" },
  { mode: "follow", label: "Follow", icon: <Target size={14} />, key: "f", hint: "Keep the selected person centred" },
  { mode: "top", label: "Top", icon: <Grid size={14} />, key: "t", hint: "Tactical view from above" },
  { mode: "camera", label: "Camera", icon: <Camera size={14} />, key: "c", hint: "Through the recording camera" },
];

export function Viewer({ run, onRenamed, onDelete }: Props) {
  const { data, error, bytes: loaded } = useScene(run.id);
  const clock = useMemo(() => (data ? new PlaybackClock(data.scene.fps, data.scene.frames) : null), [data]);
  const [selected, setSelected] = useState<number | null>(null);
  const [solo, setSolo] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [labels, setLabels] = useState<Record<number, string>>({});
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [display, setDisplay] = useState<Display>({ mesh: true, skeleton: false, trails: true, labels: true });
  const [videoLayout, setVideoLayout] = useState<"pip" | "split" | "hidden">("pip");
  const [outlines, setOutlines] = useState(true);
  const [visibleNow, setVisibleNow] = useState<Set<number>>(new Set());
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!data) return;
    setLabels(Object.fromEntries(Object.entries(data.edits.labels ?? {}).map(([k, v]) => [Number(k), v])));
    setHidden(new Set(data.edits.hidden ?? []));
  }, [data]);

  const persist = useCallback(
    (nextLabels: Record<number, string>, nextHidden: Set<number>) => {
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        const edits: Edits = { labels: Object.fromEntries(Object.entries(nextLabels)), hidden: [...nextHidden] };
        void fetch(`/api/runs/${run.id}/edits`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(edits) });
      }, 400);
    },
    [run.id],
  );

  const updateHidden = useCallback(
    (next: Set<number>) => {
      setHidden(next);
      persist(labels, next);
      if (selected !== null && next.has(selected)) setSelected(null);
    },
    [labels, persist, selected],
  );

  useEffect(() => {
    if (!data || !clock) return;
    const timer = setInterval(() => {
      const frame = clock.frame();
      const now = new Set<number>();
      for (const p of data.people) if (p.visibleAt(frame)) now.add(p.entry.id);
      setVisibleNow((prev) => (prev.size === now.size && [...now].every((id) => prev.has(id)) ? prev : now));
    }, 200);
    return () => clearInterval(timer);
  }, [data, clock]);

  useEffect(() => {
    if (!clock) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === " ") {
        event.preventDefault();
        clock.toggle();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const sign = event.key === "ArrowLeft" ? -1 : 1;
        if (event.shiftKey) clock.seek(clock.now() + sign);
        else clock.step(sign);
      } else if (event.key === "Escape") {
        setSelected(null);
      } else {
        const mode = CAMERA_MODES.find((m) => m.key === event.key.toLowerCase());
        if (mode && !event.metaKey && !event.ctrlKey) setCameraMode(mode.mode);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clock]);

  async function rename() {
    const name = window.prompt("Name of this analysis", run.name)?.trim();
    if (!name || name === run.name) return;
    await fetch(`/api/runs/${run.id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) });
    onRenamed();
  }

  const colors = data?.scene.people.map((p) => p.color) ?? [];
  const toggle = (key: keyof Display) => setDisplay((d) => ({ ...d, [key]: !d[key] }));

  return (
    <div className="viewer">
      <header className="viewer-head">
        <Link href="/" className="crumb">
          <Back size={14} /> Analyses
        </Link>
        <button className="viewer-title" onClick={rename} title="Rename">
          {run.name}
        </button>
        {data && (
          <span className="viewer-meta faint">
            {data.scene.people.length} people · {duration(run.video?.duration)} · {data.scene.fps.toFixed(0)} fps
            {run.status.receipts?.gpu ? ` · ${run.status.receipts.gpu.replace("NVIDIA ", "")}` : ""}
          </span>
        )}
        <div className="topbar-spacer" />
        <ClusterBadge />
        <button className="icon-btn" onClick={onDelete} title="Delete analysis">
          <Trash size={16} />
        </button>
      </header>

      {error ? (
        <div className="viewer-loading">
          <div className="error-box">{error}</div>
        </div>
      ) : !data || !clock ? (
        <div className="viewer-loading">Loading the scene… {loaded ? bytes(loaded) : ""}</div>
      ) : (
        <>
          <div className="viewer-body">
            <div className={`stage-area layout-${videoLayout}`}>
              <div className="stage-wrap">
                <Stage
                  data={data}
                  clock={clock}
                  hidden={hidden}
                  selected={selected}
                  solo={solo}
                  display={display}
                  cameraMode={cameraMode}
                  labels={labels}
                  video={videoElement}
                  onSelect={setSelected}
                />
                <div className="stage-toolbar left">
                  <div className="segmented">
                    {CAMERA_MODES.map((m) => (
                      <button
                        key={m.mode}
                        className={cameraMode === m.mode ? "on" : ""}
                        onClick={() => setCameraMode(m.mode)}
                        title={`${m.hint} (${m.key.toUpperCase()})`}
                        disabled={m.mode === "follow" && selected === null}
                      >
                        {m.icon} {m.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="stage-toolbar right">
                  <button className={`icon-btn${display.mesh ? " active" : ""}`} onClick={() => toggle("mesh")} title="Body meshes">
                    <Body />
                  </button>
                  <button className={`icon-btn${display.skeleton ? " active" : ""}`} onClick={() => toggle("skeleton")} title="Skeletons">
                    <Bones />
                  </button>
                  <button className={`icon-btn${display.trails ? " active" : ""}`} onClick={() => toggle("trails")} title="Trails">
                    <Trail />
                  </button>
                  <button className={`icon-btn${display.labels ? " active" : ""}`} onClick={() => toggle("labels")} title="Name tags">
                    <Tag />
                  </button>
                  <span className="toolbar-sep" />
                  <button
                    className={`icon-btn${videoLayout !== "hidden" ? " active" : ""}`}
                    onClick={() => setVideoLayout((l) => (l === "hidden" ? "pip" : "hidden"))}
                    title="Source video"
                  >
                    <Video />
                  </button>
                  <button
                    className={`icon-btn${videoLayout === "split" ? " active" : ""}`}
                    onClick={() => setVideoLayout((l) => (l === "split" ? "pip" : "split"))}
                    title="Side by side"
                  >
                    <Swap />
                  </button>
                </div>
                {cameraMode === "follow" && selected === null && <div className="stage-hint">Select a person to follow</div>}
              </div>
              <div className="video-dock">
                <VideoPanel
                  src={`/api/runs/${run.id}/files/video.mp4`}
                  width={data.scene.width}
                  height={data.scene.height}
                  clock={clock}
                  overlay={data.overlay}
                  colors={colors}
                  hidden={hidden}
                  selected={selected}
                  showOutlines={outlines}
                  onSelect={setSelected}
                  onVideo={setVideoElement}
                />
                <label className="video-option">
                  <input type="checkbox" checked={outlines} onChange={(e) => setOutlines(e.target.checked)} /> Outlines
                </label>
              </div>
            </div>
            <aside className="viewer-side">
              <PeoplePanel
                people={data.scene.people}
                labels={labels}
                hidden={hidden}
                selected={selected}
                solo={solo}
                visibleNow={visibleNow}
                onSelect={setSelected}
                onSolo={setSolo}
                onToggleHidden={(id) => {
                  const next = new Set(hidden);
                  if (next.has(id)) next.delete(id);
                  else next.add(id);
                  updateHidden(next);
                }}
                onShowOnly={(ids) =>
                  updateHidden(ids === null ? new Set() : new Set(data.scene.people.map((p) => p.id).filter((id) => !ids.includes(id))))
                }
                onExport={() => exportAllMotion(data, labels)}
                onRename={(id, label) => {
                  const next = { ...labels };
                  if (label.trim()) next[id] = label.trim();
                  else delete next[id];
                  setLabels(next);
                  persist(next, hidden);
                }}
              />
              <MotionPanel data={data} clock={clock} selected={selected} labels={labels} onSelect={setSelected} />
            </aside>
          </div>
          <Timeline
            clock={clock}
            frames={data.scene.frames}
            people={data.scene.people}
            labels={labels}
            hidden={hidden}
            selected={selected}
            onSelect={setSelected}
          />
        </>
      )}
    </div>
  );
}
