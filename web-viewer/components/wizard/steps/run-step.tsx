"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import { apiFetch } from "../../../lib/api-client";
import { deletedRanges, maskedRanges } from "../../../lib/video-timeline";
import { stageUpload } from "../upload-client";
import { useWizard, useWizardActions } from "../state";

// The full 3D viewer is heavy (three.js) and client-only. Load it lazily and
// without SSR so it doesn't bloat the wizard bundle or create an import cycle.
const EmbeddedViewer = dynamic(
  () => import("../../viewer-shell").then((m) => ({ default: m.ViewerShell })),
  {
    ssr: false,
    loading: () => <div className="of-empty-run">Loading 3D viewer…</div>,
  },
);

type Job = {
  id: string;
  runId: string;
  videoFileName: string;
  status: "queued" | "running" | "completed" | "failed" | "canceled";
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  processedFrames: number;
  totalFrames: number | null;
  progressPercent: number | null;
  error: string | null;
};

const POLL_INTERVAL_MS = 2200;
// Title-case a job status for display ("running" → "Running").
function statusLabel(status: Job["status"]): string {
  return status[0].toUpperCase() + status.slice(1);
}

// Final wizard step: launch a processing job that reconstructs the subject
// chosen in the detect step (driven by their dense per-frame track), then poll
// job status and embed the live viewer.
export function RunStep() {
  const { state, dispatch, goBack } = useWizard();
  const actions = useWizardActions();
  const {
    file,
    fileUrl,
    subjectPrompt,
    segments,
    stagedUpload,
    detect,
    detectTrackFile,
    activeJobRunId,
  } = state;

  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(activeJobRunId);
  const [submittedRunId, setSubmittedRunId] = useState<string | null>(activeJobRunId);
  const [uploadProgress, setUploadProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  /* ===== Job polling ===== */

  const loadJobs = useCallback(async () => {
    try {
      const resp = await apiFetch("/api/jobs");
      if (!resp.ok) return;
      const json = (await resp.json()) as { jobs?: Job[] };
      if (Array.isArray(json.jobs)) {
        setJobs(json.jobs);
      }
    } catch {
      // network blip — keep last view
    }
  }, []);

  useEffect(() => {
    void loadJobs();
    const handle = window.setInterval(() => void loadJobs(), POLL_INTERVAL_MS);
    return () => window.clearInterval(handle);
  }, [loadJobs]);

  const deleteJob = useCallback(
    async (jobId: string, runId: string) => {
      try {
        await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
      } catch {
        // ignore — the next poll reflects reality
      }
      if (selectedRunId === runId) setSelectedRunId(null);
      await loadJobs();
    },
    [loadJobs, selectedRunId],
  );

  /* ===== Submit the job ===== */

  const submitJob = useCallback(async () => {
    if (!file) {
      setSubmitError("Missing file");
      return;
    }
    setIsStarting(true);
    setSubmitError(null);
    try {
      // The video was already staged once in the detect step — reuse it instead
      // of re-uploading. Fall back to staging here if the user somehow skipped it.
      const staged =
        stagedUpload ??
        (await stageUpload(file, (loaded, total) => setUploadProgress({ loaded, total })));

      const formData = new FormData();
      formData.append("stagedUploadId", staged.stagedUploadId);
      formData.append("videoFileName", staged.fileName);
      formData.append("inferenceTarget", "body");
      formData.append("precision", "float32");
      formData.append("autoInitMode", "sam3");
      formData.append("autoSelectStrategy", "patient");
      formData.append("renderPreview", "true");
      formData.append("sam3TextPrompts", subjectPrompt.trim() || "person");

      // Delete segments are cut out of the video (timeline compresses);
      // mask segments stay in the video but are skipped by inference.
      const deleted = deletedRanges(segments);
      const masked = maskedRanges(segments);
      if (deleted.length > 0) {
        formData.append("removedSegments", JSON.stringify(deleted));
      }
      if (masked.length > 0) {
        formData.append("maskedSegments", JSON.stringify(masked));
      }

      // The detect step produced a dense per-frame box track for the chosen
      // subject(s). Multi-subject selections spawn ONE JOB PER SUBJECT (the
      // pipeline reconstructs one person per run); the jobs queue serialises
      // them and the viewer reunites the sibling runs in a single 3D scene.
      const subjectCount = detectTrackFile
        ? Math.max(1, detect?.selectedSubjects.length ?? 1)
        : 1;
      const firstJobs: Job[] = [];
      for (let subjectIndex = 0; subjectIndex < subjectCount; subjectIndex += 1) {
        const fd = new FormData();
        formData.forEach((value, key) => fd.append(key, value));
        if (detectTrackFile) {
          fd.append("subjectTrackFile", detectTrackFile);
          fd.append("subjectIndex", String(subjectIndex));
          fd.append("subjectCount", String(subjectCount));
        }
        const resp = await apiFetch("/api/jobs", { method: "POST", body: fd });
        if (!resp.ok) {
          const errText = await resp.text().catch(() => "");
          throw new Error(`Job creation failed (${resp.status}) ${errText}`);
        }
        const json = (await resp.json()) as { job?: Job };
        if (!json.job) {
          throw new Error("Job creation returned no job object");
        }
        firstJobs.push(json.job);
      }
      const primary = firstJobs[0];
      setSubmittedRunId(primary.runId);
      setSelectedRunId(primary.runId);
      dispatch({ type: "set_active_run_id", runId: primary.runId });
      await loadJobs();
      // Drop straight into the full viewer on the just-launched run, which shows
      // live progress + the 3D/kinematics building up — the embedded run-step
      // view is redundant with it.
      actions.onViewResults?.(primary.runId);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsStarting(false);
      setUploadProgress(null);
    }
  }, [
    actions,
    dispatch,
    file,
    loadJobs,
    segments,
    stagedUpload,
    detect,
    detectTrackFile,
    subjectPrompt,
  ]);

  /* ===== Derived ===== */

  const activeRunId = selectedRunId ?? submittedRunId;
  const selectedJob = useMemo(
    () => jobs.find((j) => j.runId === activeRunId) ?? null,
    [jobs, activeRunId],
  );
  const isComplete = selectedJob?.status === "completed";

  /* ===== Layout ===== */

  if (!file) {
    return <div className="of-banner">No video selected — go back to upload.</div>;
  }

  const hasSubmitted = submittedRunId !== null;

  return (
    <div>
      <div className="of-step-header">
        <h2 className="of-step-title">Run & monitor</h2>
        <p className="of-step-subtitle">
          Launch processing and watch the 3D pose and kinematics build up live, frame by frame. Multiple jobs can run in parallel.
        </p>
      </div>

      {submitError ? <div className="of-banner">{submitError}</div> : null}

      {!hasSubmitted ? (
        <div className="of-run-launch">
          {fileUrl ? (
            <video
              className="of-run-launch-video"
              src={fileUrl}
              muted
              playsInline
              preload="auto"
              controls={false}
              // Seek a hair past 0 so a real frame paints instead of a black box.
              onLoadedMetadata={(e) => {
                try {
                  e.currentTarget.currentTime = 0.1;
                } catch {
                  /* metadata not ready */
                }
              }}
            />
          ) : null}
          <div className="of-run-launch-info">
            <strong className="of-run-launch-name">{file.name}</strong>
            <span className="of-run-launch-sub">Subject: {subjectPrompt.trim() || "person"}</span>
          </div>
          <button
            className="of-btn is-primary of-run-launch-btn"
            type="button"
            onClick={submitJob}
            disabled={isStarting}
          >
            {isStarting
              ? uploadProgress
                ? `Uploading ${Math.round((uploadProgress.loaded / uploadProgress.total) * 100)}%`
                : "Starting…"
              : "Start processing"}
          </button>
        </div>
      ) : (
      <div className="of-run-grid">
        <aside className="of-job-list">
          <h3>Jobs</h3>
          {jobs.length === 0 ? (
            <div className="of-anchor-empty">No jobs yet. Start one.</div>
          ) : (
            jobs.map((job) => {
              const pct = job.progressPercent
                ? Math.round(job.progressPercent)
                : job.totalFrames
                  ? Math.round((job.processedFrames / Math.max(1, job.totalFrames)) * 100)
                  : 0;
              return (
                <div
                  key={job.id}
                  className={`of-job-card ${job.runId === activeRunId ? "is-active" : ""}`}
                  onClick={() => setSelectedRunId(job.runId)}
                >
                  <div className="of-job-card-top">
                    <span className="of-job-name">{job.videoFileName}</span>
                    <span className={`of-job-status is-${job.status}`}>
                      {statusLabel(job.status)}
                    </span>
                    <button
                      type="button"
                      className="of-job-del"
                      title="Delete job"
                      aria-label="Delete job"
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteJob(job.id, job.runId);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                  <div className="of-job-prog-bar">
                    <div className="of-job-prog-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="of-job-meta">
                    {job.processedFrames}{job.totalFrames ? ` / ${job.totalFrames}` : ""} frames · {pct}%
                  </div>
                </div>
              );
            })
          )}
          <button
            className="of-btn is-primary"
            type="button"
            onClick={submitJob}
            disabled={isStarting}
            style={{ marginTop: 10 }}
          >
            {isStarting
              ? uploadProgress
                ? `Uploading ${Math.round((uploadProgress.loaded / uploadProgress.total) * 100)}%`
                : "Starting…"
              : hasSubmitted
                ? "Run another job"
                : "Start processing"}
          </button>
        </aside>

        <div className="of-run-detail">
          {!activeRunId ? (
            <div className="of-empty-run">
              {hasSubmitted ? "Loading job details…" : "Click Start processing to launch the run."}
            </div>
          ) : (
            <>
              <div className="of-run-detail-head">
                <h3>{selectedJob?.videoFileName ?? activeRunId}</h3>
                {selectedJob ? (
                  <span className={`of-job-status is-${selectedJob.status}`}>
                    {statusLabel(selectedJob.status)}
                  </span>
                ) : null}
                {isComplete && actions.onViewResults ? (
                  <button
                    className="of-btn is-sm is-primary"
                    type="button"
                    onClick={() => actions.onViewResults?.(activeRunId)}
                    style={{ marginLeft: "auto" }}
                  >
                    Open in full browser →
                  </button>
                ) : null}
              </div>

              {selectedJob?.error ? <div className="of-banner">{selectedJob.error}</div> : null}

              {/* Full live 3D viewer + kinematics plots for this run. */}
              <div className="of-embedded-viewer">
                <EmbeddedViewer key={activeRunId} embeddedRunId={activeRunId} />
              </div>
            </>
          )}
        </div>
      </div>
      )}

      <div className="of-action-bar">
        <button className="of-btn is-ghost" type="button" onClick={goBack}>
          ← Back
        </button>
        <span className="of-action-hint">
          {isComplete
            ? "Processing complete. Review the 3D and kinematics plots above, or open the full browser."
            : selectedJob
              ? "Processing in background — the 3D and plots update live as frames arrive."
              : "Multi-job runs are supported. Each job appears in the left rail."}
        </span>
        <div className="of-action-bar-spacer" />
      </div>
    </div>
  );
}
