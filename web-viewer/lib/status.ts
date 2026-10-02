import type { RunStatus, RunSummary } from "./runs";

export type Tone = "ok" | "busy" | "warn" | "bad" | "idle";

export type Step = { key: string; label: string; detail: string };

/** The steps of an analysis on this machine; a runner can declare others in `status.steps`. */
export const STEPS: Step[] = [
  { key: "preparing", label: "Prepare", detail: "Normalize the video (frame rate, size, rotation)" },
  { key: "tracking", label: "Track people", detail: "SAM 3.1 follows everyone through the video" },
  { key: "camera", label: "Camera", detail: "Estimate the lens's focal length from the picture (MoGe-2)" },
  { key: "bodies", label: "Bodies", detail: "SAM 3D Body reconstructs each person in 3D" },
  { key: "appearance", label: "Appearance", detail: "DINOv3 describes each person, to tell people apart" },
  { key: "building", label: "3D scene", detail: "Who is who, smooth motion, ground the feet" },
];

export function stepsOf(status: RunStatus): Step[] {
  return status.steps?.length ? status.steps : STEPS;
}

/** Index of the step an analysis is currently in (the number of steps when done). */
export function stepIndex(status: RunStatus): number {
  const steps = stepsOf(status);
  if (status.state === "done") return steps.length;
  if (status.state === "new") return 0;
  // While the GPU stages run, the stage names the step; otherwise the state does.
  const stage = status.stage === "packing" ? "appearance" : status.stage; // packing closes the GPU work
  const key = status.state === "running" ? stage ?? "tracking" : status.state;
  const index = steps.findIndex((step) => step.key === key);
  if (index >= 0) return index;
  return status.state === "running" ? steps.findIndex((step) => step.key === "tracking") : -1;
}

export function stepProgress(status: RunStatus): number | null {
  const p = status.progress;
  if (!p || !p.total) return null;
  return Math.max(0, Math.min(1, p.done / p.total));
}

export function describe(run: Pick<RunSummary, "status" | "worker_alive">): { label: string; tone: Tone } {
  const s = run.status;
  switch (s.state) {
    case "done":
      return { label: "Ready", tone: "ok" };
    case "failed":
      return { label: "Failed", tone: "bad" };
    case "cancelled":
      return { label: "Cancelled", tone: "idle" };
    case "new":
      return { label: "Not started", tone: "idle" };
    default: {
      if (!run.worker_alive) return { label: "Paused", tone: "warn" };
      const step = stepsOf(s)[stepIndex(s)];
      const progress = stepProgress(s);
      const pct = progress != null && s.state === "running" ? ` ${Math.round(progress * 100)}%` : "";
      return { label: `${step?.label ?? "Working"}${pct}`, tone: s.state === "queued" ? "warn" : "busy" };
    }
  }
}
