"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiFetch } from "../../../lib/api-client";
import { clamp, formatTimecode } from "../../../lib/video-timeline";
import { stageUpload } from "../upload-client";
import { useWizard, type DetectDet } from "../state";

const DEFAULT_STRIDE = 5;

/** Detect step: run a full-video, detection-only pass that streams per-frame
 *  boxes for every person (SAM3 via MLX), play the annotated result as it fills
 *  in, and pick which tracked subject to reconstruct. */
export function DetectStep() {
  const { state, dispatch, goNext, goBack } = useWizard();
  const { file, fileUrl, subjectPrompt, stagedUpload, detect } = state;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const cursorRef = useRef(0); // next frame index to request from the poller
  const isScrubbingRef = useRef(false); // suppress timeupdate while dragging the scrubber
  const [currentSec, setCurrentSec] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fps = detect?.fps || 30;
  const lastFrame = detect?.lastFrame ?? -1;
  // Playback frontier: only the detected range is playable; it grows as the scan
  // progresses, so the preview fills in and can be stopped early.
  const frontierSec = lastFrame >= 0 ? (lastFrame + 1) / fps : 0;
  // Keep polling through every in-progress state. CRITICAL: include "loading"
  // (written by the detector while it loads the model) — otherwise polling stops
  // during load and never resumes, freezing the UI at 0.
  const running =
    detect?.status === "starting" || detect?.status === "loading" || detect?.status === "running";
  const pct = detect && detect.totalToProcess > 0
    ? Math.min(100, Math.round((detect.processed / detect.totalToProcess) * 100))
    : 0;

  /* ===== Start / stop / re-run ===== */

  const startDetect = useCallback(async () => {
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      let staged = stagedUpload;
      if (!staged) {
        const s = await stageUpload(file, () => undefined);
        staged = { stagedUploadId: s.stagedUploadId, fileName: s.fileName };
        dispatch({ type: "set_staged_upload", staged });
      }
      const resp = await apiFetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stagedUploadId: staged.stagedUploadId,
          prompt: subjectPrompt.trim() || "person",
          stride: DEFAULT_STRIDE,
        }),
      });
      const json = await resp.json();
      if (!resp.ok || !json.detectId) {
        throw new Error(json.error || "Failed to start detection");
      }
      cursorRef.current = 0;
      dispatch({ type: "detect_start", id: json.detectId, prompt: json.prompt, stride: json.stride });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [file, stagedUpload, subjectPrompt, dispatch]);

  const stopDetect = useCallback(async () => {
    if (!detect) return;
    try {
      await apiFetch(`/api/detect/${detect.id}`, { method: "DELETE" });
    } catch {
      // ignore — the poll will reflect the terminal status
    }
  }, [detect]);

  const reRun = useCallback(async () => {
    await stopDetect();
    dispatch({ type: "detect_reset" });
    setIsPlaying(false);
  }, [stopDetect, dispatch]);

  /* ===== Polling ===== */

  useEffect(() => {
    if (!detect || !running) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await apiFetch(`/api/detect/${detect.id}?sinceFrame=${cursorRef.current}`, {
          cache: "no-store",
        });
        const j = await resp.json();
        if (cancelled || !resp.ok) return;
        const frames = (Array.isArray(j.frames) ? j.frames : []) as Array<{ f: number; dets: DetectDet[] }>;
        if (frames.length > 0) {
          cursorRef.current = Math.max(cursorRef.current, ...frames.map((f) => f.f + 1));
        }
        dispatch({
          type: "detect_progress",
          patch: {
            status: j.status,
            processed: j.processed ?? 0,
            totalToProcess: j.totalToProcess ?? 0,
            totalFrames: j.totalFrames ?? 0,
            lastFrame: j.lastFrame ?? -1,
            videoW: j.videoWidth ?? 0,
            videoH: j.videoHeight ?? 0,
            fps: j.fps ?? 30,
            stride: j.stride ?? DEFAULT_STRIDE,
            tracks: Array.isArray(j.tracks) ? j.tracks : [],
          },
          frames,
        });
      } catch {
        // transient — try again next tick
      }
    };
    void tick();
    const id = window.setInterval(tick, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detect?.id, running]);

  /* ===== Playback (clamped to the detected frontier) ===== */

  const onTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    // While the user is dragging the scrubber, don't let the (async, lagging)
    // video time write back over the drag position — that caused the jumping.
    if (isScrubbingRef.current) return;
    // Clamp to the detected frontier only during playback (not when seeking).
    if (!v.paused && frontierSec > 0 && v.currentTime > frontierSec) {
      v.currentTime = frontierSec;
      v.pause();
      setIsPlaying(false);
    }
    setCurrentSec(v.currentTime);
  }, [frontierSec]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      if (frontierSec > 0 && v.currentTime >= frontierSec - 0.02) {
        v.currentTime = 0;
      }
      void v.play();
      setIsPlaying(true);
    } else {
      v.pause();
      setIsPlaying(false);
    }
  }, [frontierSec]);

  const seek = useCallback((sec: number) => {
    const v = videoRef.current;
    if (!v) return;
    const t = clamp(sec, 0, frontierSec > 0 ? frontierSec : sec);
    v.currentTime = t;
    setCurrentSec(t);
  }, [frontierSec]);

  /* ===== Subjects (track groups) + colours ===== */

  // Group surfaced tracks into subjects: default each track is its own subject;
  // the user can merge re-ID splits (person 1/3/5) into one. Sorted by total hits.
  const subjects = useMemo(() => {
    if (!detect) return [] as Array<{
      subjectId: number; trackIds: number[]; color: string; label: number;
      frameCount: number; firstFrame: number; lastFrame: number; repFrame: number;
    }>;
    const subjectOf = detect.subjectOf;
    const groups = new Map<number, { trackIds: number[]; color: string; frameCount: number; firstFrame: number; lastFrame: number; repFrame: number; repHits: number }>();
    for (const t of detect.tracks) {
      const sid = subjectOf[t.id] ?? t.id;
      const g = groups.get(sid);
      if (!g) {
        groups.set(sid, { trackIds: [t.id], color: t.color, frameCount: t.frameCount, firstFrame: t.firstFrame, lastFrame: t.lastFrame, repFrame: t.repFrame, repHits: t.frameCount });
      } else {
        g.trackIds.push(t.id);
        g.frameCount += t.frameCount;
        g.firstFrame = Math.min(g.firstFrame, t.firstFrame);
        g.lastFrame = Math.max(g.lastFrame, t.lastFrame);
        if (t.frameCount > g.repHits) { g.repHits = t.frameCount; g.repFrame = t.repFrame; g.color = t.color; }
      }
    }
    return Array.from(groups.entries())
      .map(([subjectId, g]) => ({ subjectId, trackIds: g.trackIds, color: g.color, label: 0, frameCount: g.frameCount, firstFrame: g.firstFrame, lastFrame: g.lastFrame, repFrame: g.repFrame }))
      .sort((a, b) => b.frameCount - a.frameCount)
      .map((s, i) => ({ ...s, label: i + 1 }));
  }, [detect]);

  // Raw track id -> subject colour / subject id (for the overlay + clicks).
  const subjectColor = useMemo(() => {
    const m = new Map<number, string>();
    subjects.forEach((s) => s.trackIds.forEach((tid) => m.set(tid, s.color)));
    return m;
  }, [subjects]);
  const subjectIdOfTrack = useMemo(() => {
    const m = new Map<number, number>();
    subjects.forEach((s) => s.trackIds.forEach((tid) => m.set(tid, s.subjectId)));
    return m;
  }, [subjects]);

  const currentBoxes = useMemo<DetectDet[]>(() => {
    if (!detect || detect.stride <= 0) return [];
    const f = Math.round(currentSec * fps);
    const key = Math.floor(f / detect.stride) * detect.stride;
    return detect.framesById[key] ?? detect.framesById[key - detect.stride] ?? [];
  }, [detect, currentSec, fps]);

  /* ===== Selection + continue (materialize the chosen subjects) ===== */

  const selectedSet = useMemo(() => new Set(detect?.selectedSubjects ?? []), [detect?.selectedSubjects]);
  const canContinue = (detect?.selectedSubjects.length ?? 0) > 0;

  const onContinue = useCallback(async () => {
    if (!detect || detect.selectedSubjects.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      // Map each selected subject to its (possibly merged) track ids.
      const payload = detect.selectedSubjects.map((sid) => ({
        subjectId: sid,
        trackIds: Object.entries(detect.subjectOf)
          .filter(([, s]) => s === sid)
          .map(([tid]) => Number(tid)),
      }));
      const resp = await apiFetch(`/api/detect/${detect.id}/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subjects: payload }),
      });
      const j = await resp.json();
      if (!resp.ok || !j.trackFilePath) {
        throw new Error(j.error || "Could not prepare the chosen subjects");
      }
      dispatch({ type: "set_detect_track_file", path: j.trackFilePath });
      goNext();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [detect, dispatch, goNext]);

  if (!file || !fileUrl) {
    return <div className="of-banner">No video selected. Go back to the upload step.</div>;
  }

  const vw = detect?.videoW || 16;
  const vh = detect?.videoH || 9;
  const hasStarted = detect !== null;

  return (
    <div>
      <div className="of-step-header">
        <h2 className="of-step-title">Detect the subject</h2>
        <p className="of-step-subtitle">
          Detect everyone across the whole video, watch it fill in, then click the person you want to
          reconstruct. The tracker re-matches them if they leave and come back; a different person gets
          their own colour.
        </p>
      </div>

      {error ? <div className="of-banner">{error}</div> : null}

      <div className="of-detect-stage">
        {/* Prompt + run controls */}
        <div className="of-prompt">
          <span className="of-prompt-label">Subject prompt</span>
          <div className="of-prompt-row">
            <input
              className="of-input"
              type="text"
              value={subjectPrompt}
              placeholder="person"
              disabled={running}
              onChange={(e) => dispatch({ type: "set_subject_prompt", prompt: e.target.value })}
              maxLength={120}
            />
            {!hasStarted ? (
              <button type="button" className="of-btn is-primary" onClick={startDetect} disabled={busy}>
                {busy ? "Starting…" : "Detect people in video"}
              </button>
            ) : running ? (
              <button type="button" className="of-btn is-warn" onClick={stopDetect}>
                Stop
              </button>
            ) : (
              <button type="button" className="of-btn is-ghost" onClick={reRun} disabled={busy}>
                Re-run with new prompt
              </button>
            )}
          </div>
          {hasStarted ? (
            <div className="of-detect-progress" aria-label="Detection progress">
              <div className="of-detect-progress-bar">
                <div className="of-detect-progress-fill" style={{ width: `${pct}%` }} />
              </div>
              <span className="of-detect-progress-label">
                {detect.status === "starting" || detect.status === "loading"
                  ? "Loading SAM3 model on the GPU…"
                  : detect.status === "running"
                    ? `Scanning… ${pct}% · ${formatTimecode(detect.lastFrame >= 0 ? (detect.lastFrame + 1) / fps : 0)} / ${formatTimecode(detect.totalFrames > 0 ? detect.totalFrames / fps : 0)} · ${detect.tracks.length} ${detect.tracks.length === 1 ? "person" : "people"} so far`
                    : detect.status === "stopped"
                      ? `Stopped at ${detect.processed} frames — pick a subject below or re-run.`
                      : detect.status === "error"
                        ? "Detection failed. Try re-running."
                        : `Done — ${detect.processed} frames scanned.`}
              </span>
            </div>
          ) : (
            <span className="of-prompt-hint">
              Detection only (no kinematics) — fast and light. You can stop early and re-run if it looks wrong.
            </span>
          )}
        </div>

        {/* Annotated preview video */}
        <div className="of-detect-video" style={{ marginTop: 12 }}>
          <video
            ref={videoRef}
            src={fileUrl}
            preload="auto"
            onTimeUpdate={onTimeUpdate}
            onLoadedMetadata={() => setCurrentSec(0)}
            controls={false}
          />
          {detect && currentBoxes.length > 0 ? (
            <svg
              className="of-detect-overlay"
              viewBox={`0 0 ${vw} ${vh}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {currentBoxes.map((d) => {
                const sid = subjectIdOfTrack.get(d.id);
                const subj = subjects.find((s) => s.subjectId === sid);
                const color = subjectColor.get(d.id) ?? "#94a3b8";
                const sel = sid !== undefined && selectedSet.has(sid);
                const x = d.b[0] * vw;
                const y = d.b[1] * vh;
                const w = d.b[2] * vw;
                const h = d.b[3] * vh;
                return (
                  <g
                    key={d.id}
                    style={{ cursor: sid !== undefined ? "pointer" : "default" }}
                    onClick={() => {
                      if (sid !== undefined) dispatch({ type: "detect_toggle_subject", subjectId: sid });
                    }}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      fill={sel ? color : "none"}
                      fillOpacity={sel ? 0.12 : 0}
                      stroke={color}
                      strokeWidth={sel ? 4 : 2.5}
                      vectorEffect="non-scaling-stroke"
                      opacity={sel ? 1 : 0.9}
                      rx={3}
                    />
                    <rect x={x} y={Math.max(0, y - 22)} width={108} height={20} fill={color} rx={3} />
                    <text x={x + 6} y={Math.max(13, y - 7)} fontSize={14} fontWeight={700} fill="#04140c">
                      {sel ? "✓ " : ""}Person {subj?.label ?? "?"}
                    </text>
                  </g>
                );
              })}
            </svg>
          ) : null}
        </div>

        {/* Transport */}
        {hasStarted ? (
          <div className="of-detect-transport" style={{ marginTop: 8 }}>
            <button type="button" className="of-btn is-sm is-ghost" onClick={togglePlay} disabled={lastFrame < 0}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <input
              className="of-detect-scrub"
              type="range"
              min={0}
              max={Math.max(0.1, frontierSec)}
              step={0.05}
              value={Math.min(currentSec, frontierSec)}
              onPointerDown={() => {
                isScrubbingRef.current = true;
                videoRef.current?.pause();
                setIsPlaying(false);
              }}
              onPointerUp={() => { isScrubbingRef.current = false; }}
              onPointerCancel={() => { isScrubbingRef.current = false; }}
              onChange={(e) => {
                const t = Number(e.target.value);
                setCurrentSec(t); // immediate, so the thumb tracks the drag
                seek(t);
              }}
            />
            <span className="of-detect-time">
              <strong>{formatTimecode(currentSec)}</strong>
              {detect && detect.totalFrames > 0 ? ` / ${formatTimecode(detect.totalFrames / fps)}` : ""}
            </span>
          </div>
        ) : null}

        {/* Subject chips — select one or several to reconstruct; merge re-ID splits */}
        {detect && subjects.length > 0 ? (
          <div className="of-track-chips" style={{ marginTop: 14 }}>
            <div className="of-track-chips-head">
              <span className="of-track-chips-label">
                Select the subject(s) to reconstruct — pick several to put them in one 3D scene.
              </span>
              <div className="of-track-chips-actions">
                <button
                  type="button"
                  className="of-btn is-sm is-ghost"
                  disabled={(detect.selectedSubjects.length ?? 0) < 2}
                  title="Merge the selected people into one identity (fix a re-ID split)"
                  onClick={() => dispatch({ type: "detect_merge_selected" })}
                >
                  Merge selected
                </button>
              </div>
            </div>
            <div className="of-track-chips-row">
              {subjects.map((s) => {
                const sel = selectedSet.has(s.subjectId);
                const durSec = (s.lastFrame - s.firstFrame + 1) / fps;
                const merged = s.trackIds.length > 1;
                return (
                  <button
                    key={s.subjectId}
                    type="button"
                    className={`of-track-chip ${sel ? "is-selected" : ""}`}
                    style={{ borderColor: sel ? s.color : undefined }}
                    onClick={() => dispatch({ type: "detect_toggle_subject", subjectId: s.subjectId })}
                    onMouseEnter={() => seek(s.repFrame / fps)}
                  >
                    <span className="of-track-chip-dot" style={{ background: s.color }} />
                    <span className="of-track-chip-name">Person {s.label}</span>
                    <span className="of-track-chip-meta">
                      {Math.round(durSec)}s · {s.frameCount} hits{merged ? ` · ${s.trackIds.length} merged` : ""}
                    </span>
                    {merged ? (
                      <span
                        className="of-track-chip-unmerge"
                        title="Unmerge"
                        onClick={(e) => {
                          e.stopPropagation();
                          dispatch({ type: "detect_unmerge_subject", subjectId: s.subjectId });
                        }}
                      >
                        ⤬
                      </span>
                    ) : null}
                    {sel ? <span className="of-track-chip-check">✓</span> : null}
                  </button>
                );
              })}
            </div>
          </div>
        ) : hasStarted && !running ? (
          <div className="of-prompt-hint" style={{ marginTop: 12 }}>
            No people detected. Try a different prompt and re-run.
          </div>
        ) : null}
      </div>

      <div className="of-action-bar">
        <button className="of-btn is-ghost" type="button" onClick={goBack}>
          ← Back
        </button>
        <span className="of-action-hint">
          {canContinue
            ? `${detect?.selectedSubjects.length} subject${(detect?.selectedSubjects.length ?? 0) > 1 ? "s" : ""} selected — reconstruct ${(detect?.selectedSubjects.length ?? 0) > 1 ? "them together in one 3D scene" : "their kinematics"}.`
            : "Click a person (box or chip) to select them. Merge any duplicates first."}
        </span>
        <div className="of-action-bar-spacer" />
        <button
          className="of-btn is-primary"
          type="button"
          onClick={onContinue}
          disabled={!canContinue || busy}
          title={!canContinue ? "Pick at least one subject first" : undefined}
        >
          {busy
            ? "Preparing…"
            : `Continue to run${canContinue ? ` (${detect?.selectedSubjects.length})` : ""} →`}
        </button>
      </div>
    </div>
  );
}
