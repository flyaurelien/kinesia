import type { RunFrame } from "./types";

// Per-frame ground-contact flags for each foot, plus the resulting support phase.
export type FootContactState = {
  left: boolean;
  right: boolean;
  support: "left" | "right" | "both" | "none";
};

export type StabilizationXYOutput = {
  rootWorldStabilized: Array<[number, number, number]>;
  footContact: FootContactState[];
};

// Map a camera-space point into the viewer's world axes (camera Z->X, X->Y, Y->Z, with sign flips).
function camToWorldXYZ(x: number, y: number, z: number): [number, number, number] {
  return [-z, x, -y];
}

// Pick the best available world-space root for a frame, falling back to the camera-derived position.
function frameRoot(frame: RunFrame): [number, number, number] {
  return frame.rootWorldRaw ?? frame.rootWorldStabilized ?? [-frame.cameraComp[2], frame.cameraComp[0], -frame.cameraComp[1]];
}

// Moving-average the horizontal (X/Y) root path over a ~0.6 s window to damp jitter; height (Z) is left untouched.
function smoothRoots(roots: Array<[number, number, number]>, fps: number): Array<[number, number, number]> {
  const window = Math.max(3, Math.round(Math.max(1, fps) * 0.60));
  return roots.map((root, index) => {
    const lo = Math.max(0, index - window);
    const hi = Math.min(roots.length, index + window + 1);
    const slice = roots.slice(lo, hi);
    const meanX = slice.reduce((sum, value) => sum + value[0], 0) / Math.max(1, slice.length);
    const meanY = slice.reduce((sum, value) => sum + value[1], 0) / Math.max(1, slice.length);
    return [meanX, meanY, root[2]];
  });
}

// Lower-is-better drift score: bounding-box extent plus net start-to-end displacement in the X/Y plane.
function xyStabilityScore(roots: Array<[number, number, number]>): number {
  if (roots.length < 2) {
    return 0;
  }
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const root of roots) {
    minX = Math.min(minX, root[0]);
    maxX = Math.max(maxX, root[0]);
    minY = Math.min(minY, root[1]);
    maxY = Math.max(maxY, root[1]);
  }
  const extent = Math.hypot(maxX - minX, maxY - minY);
  const first = roots[0];
  const last = roots[roots.length - 1];
  return extent + Math.hypot(last[0] - first[0], last[1] - first[1]);
}

// Lowest world-space height (Z) among the given foot joints, or null if no joints are available.
function minFootZ(frame: RunFrame, indices: number[]): number | null {
  const joints = frame.jointsCam;
  if (!joints) {
    return null;
  }
  const values: number[] = [];
  for (const index of indices) {
    const joint = joints[index];
    if (!joint) {
      continue;
    }
    values.push(camToWorldXYZ(joint[0], joint[1], joint[2])[2]);
  }
  return values.length > 0 ? Math.min(...values) : null;
}

// Classify foot contact for a frame: a foot is "down" when its lowest joint sits within a small margin of the lower foot.
function contact(frame: RunFrame): FootContactState {
  const leftZ = minFootZ(frame, [13, 15, 16, 17]);
  const rightZ = minFootZ(frame, [14, 18, 19, 20]);
  if (leftZ === null && rightZ === null) {
    return { left: false, right: false, support: "none" };
  }
  const floor = Math.min(leftZ ?? Number.POSITIVE_INFINITY, rightZ ?? Number.POSITIVE_INFINITY);
  const margin = 0.055;
  const left = leftZ !== null && leftZ <= floor + margin;
  const right = rightZ !== null && rightZ <= floor + margin;
  return {
    left,
    right,
    support: left && right ? "both" : left ? "left" : right ? "right" : "none",
  };
}

// Produce a stabilized root path plus per-frame foot contact: prefer precomputed stored roots only when they drift less than the freshly smoothed ones.
export function stabilizeXY(frames: RunFrame[], fps: number): StabilizationXYOutput {
  const roots = frames.map(frameRoot);
  const contacts = frames.map(contact);
  const smoothedRoots = smoothRoots(roots, fps);
  const storedRoots = frames.map((frame) => frame.rootWorldStabilized ?? null);
  const hasStoredRoots = storedRoots.every((root): root is [number, number, number] => root !== null);
  const rootWorldStabilized =
    hasStoredRoots && xyStabilityScore(storedRoots) < xyStabilityScore(smoothedRoots)
      ? storedRoots
      : smoothedRoots;
  return {
    rootWorldStabilized,
    footContact: contacts,
  };
}
