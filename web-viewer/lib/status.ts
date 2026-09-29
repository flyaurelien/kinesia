import type { RunStatus, RunSummary } from "./runs";

export type Tone = "ok" | "busy" | "warn" | "bad" | "idle";

export type Step = { key: string; label: string; detail: string };

/** The steps of an analysis, in order, as shown on the processing screen. */
export const STEPS: Step[] = [
  { key: "preparing", label: "Prepare", detail: "Normalize the video (frame rate, size, rotation)" },
  { key: "uploading", label: "Upload", detail: "Send the video to the remote GPU server" },
  { key: "queued", label: "GPU", detail: "Wait for a free GPU" },
  { key: "tracking", label: "Track people", detail: "SAM 3.1 follows everyone through the video" },
  { key: "camera", label: "Camera", detail: "Estimate the lens's focal length from the picture (MoGe-2)" },
  { key: "bodies", label: "Bodies", detail: "SAM 3D Body reconstructs each person in 3D" },
  { key: "downloading", label: "Download", detail: "Bring the results back; the GPU is released" },
  { key: "building", label: "3D scene", detail: "Clean and link identities, smooth motion, ground the feet" },
];

/** Index of the step an analysis is currently in (STEPS.length when done). */
export function stepIndex(status: RunStatus): number {
  switch (status.state) {
    case "new":
    case "preparing":
      return 0;
    case "uploading":
      return 1;
    case "queued":
      return 2;
    case "running": {
      const stage = status.stage ?? "";
      if (stage === "bodies" || stage === "packing") return stage === "packing" ? 6 : 5;
      if (stage === "camera") return 4;
      return 3;
    }
    case "downloading":
      return 6;
    case "building":
      return 7;
    case "done":
      return STEPS.length;
    default:
      return -1;
  }
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
      const index = stepIndex(s);
      const step = STEPS[index];
      const progress = stepProgress(s);
      const pct = progress != null && s.state === "running" ? ` ${Math.round(progress * 100)}%` : "";
      return { label: `${step?.label ?? "Working"}${pct}`, tone: s.state === "queued" ? "warn" : "busy" };
    }
  }
}
