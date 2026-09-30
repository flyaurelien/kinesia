"use client";

import { useEffect, useState } from "react";

import { clock, duration, elapsed } from "@/lib/format";
import { isActive, type RunSummary } from "@/lib/runs";
import { describe, stepIndex, stepProgress, stepsOf } from "@/lib/status";
import { Check, Chip, Refresh, Stop, Trash } from "../icons";

type Props = {
  run: RunSummary;
  onAction: (action: "process" | "cancel" | "delete") => Promise<void>;
};

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

export function Processing({ run, onAction }: Props) {
  const s = run.status;
  const active = isActive(s.state);
  const now = useNow(active);
  const [busy, setBusy] = useState<string | null>(null);
  const steps = stepsOf(s);
  const current = stepIndex(s);
  const progress = stepProgress(s);
  const summary = describe(run);
  const onGpuSeconds = elapsed(s.gpu_started_at, s.gpu_finished_at, now);
  const totalSeconds = elapsed(s.started_at, s.finished_at, now);
  const stalled = active && !run.worker_alive;
  const sceneMissing = s.state === "done" && !run.has_scene;

  async function act(action: "process" | "cancel" | "delete") {
    if (action === "cancel" && !window.confirm("Stop the analysis? Its GPU work stops at once.")) return;
    setBusy(action);
    try {
      await onAction(action);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="processing">
      <section className="card processing-main">
        <header className="processing-head">
          <div>
            <h2>{s.state === "failed" ? "The analysis failed" : s.state === "cancelled" ? "Cancelled" : "Processing"}</h2>
            <p className="muted">{stalled ? "Paused: the worker stopped. Resume to carry on where it left off." : s.message}</p>
          </div>
          <span className={`chip ${summary.tone}`}>
            <span className={`dot${active && !stalled ? " pulse" : ""}`} />
            {summary.label}
          </span>
        </header>

        <ol className="steps">
          {steps.map((step, index) => {
            const state = index < current ? "done" : index === current && active ? "current" : "pending";
            return (
              <li key={step.key} className={`step ${state}`}>
                <span className="step-marker">{state === "done" ? <Check size={13} /> : index + 1}</span>
                <div className="step-body">
                  <div className="step-title">{step.label}</div>
                  <div className="step-detail faint">{step.detail}</div>
                  {state === "current" && (
                    <div className={`progress${progress == null ? " indeterminate" : ""}`} style={{ marginTop: 8 }}>
                      <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
                    </div>
                  )}
                  {state === "current" && s.progress && (
                    <div className="faint tabular" style={{ fontSize: 12, marginTop: 4 }}>
                      {s.progress.done} / {s.progress.total} frames
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>

        {s.state === "failed" && s.error && <div className="error-box">{s.error}</div>}
      </section>

      <aside className="processing-side">
        <section className="card side-card">
          <h3>
            <Chip size={15} /> GPU
          </h3>
          <dl className="facts">
            <dt>Where</dt>
            <dd>{s.location ?? "–"}</dd>
            <dt>GPU</dt>
            <dd>{s.receipts?.gpu ?? "–"}</dd>
            <dt>On GPU</dt>
            <dd className="tabular">{onGpuSeconds != null ? clock(onGpuSeconds) : "–"}</dd>
            <dt>Total</dt>
            <dd className="tabular">{totalSeconds != null ? duration(totalSeconds) : "–"}</dd>
          </dl>
        </section>

        <section className="card side-card">
          <h3>Video</h3>
          <dl className="facts">
            <dt>Length</dt>
            <dd>{duration(run.video?.duration)}</dd>
            <dt>Frames</dt>
            <dd className="tabular">{run.video?.frames ?? "–"}</dd>
            <dt>Rate</dt>
            <dd>{run.video ? `${run.video.fps.toFixed(2)} fps` : "–"}</dd>
            <dt>Size</dt>
            <dd>{run.video ? `${run.video.width}×${run.video.height}` : "–"}</dd>
          </dl>
        </section>

        <div className="side-actions">
          {(stalled || sceneMissing || s.state === "failed" || s.state === "cancelled" || s.state === "new") && (
            <button className="btn btn-primary" disabled={busy !== null} onClick={() => act("process")}>
              <Refresh />{" "}
              {sceneMissing ? "Rebuild the 3D scene" : s.state === "failed" ? "Retry" : stalled ? "Resume" : "Start"}
            </button>
          )}
          {active && (
            <button className="btn btn-danger" disabled={busy !== null} onClick={() => act("cancel")}>
              <Stop /> Cancel
            </button>
          )}
          {!active && (
            <button className="btn btn-danger" disabled={busy !== null} onClick={() => act("delete")}>
              <Trash /> Delete analysis
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}
