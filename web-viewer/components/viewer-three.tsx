"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import { apiFetch } from "../lib/api-client";
import { filterSeries, smoothingAlpha } from "../lib/one_euro";
import type { DynamicObject, RunDetail, RunFrame, SceneObject, SceneObjectMatrix } from "../lib/types";

type DisplayAnchor = {
  centerX: number;
  centerZ: number;
  // Camera-space floor height. When present, world Z is translated by this
  // amount so the measured floor at world Z = -floorY becomes viewer Z = 0.
  floorY?: number;
};

// A sibling run of the same multi-subject selection, rendered in the SAME 3D
// scene as the primary run. Both runs share the source video's camera space,
// so placing the sibling with the primary's anchor/pivot preserves the two
// people's true relative positions.
export type SiblingSubject = {
  runDetail: RunDetail;
  color: string | null;
};

type ThreeSpaceViewerProps = {
  runDetail: RunDetail;
  frameIndex: number;
  frameCursor: number;
  uprightMode: boolean;
  showMesh: boolean;
  showJoints: boolean;
  showBones: boolean;
  meshOpacity: number;
  selectedJointIndices: number[];
  onJointPick?: (jointIndex: number) => void;
  subjectColor?: string | null;
  siblings?: SiblingSubject[];
  // Hide the primary subject's body while keeping it as the scene's reference.
  // The ground plane, anchor and camera framing all derive from the primary
  // run, so isolating another subject must not stop computing them — only the
  // primary's mesh/joints are withheld.
  showPrimary?: boolean;
  // Reconstructed static and dynamic objects placed in the
  // subject's space.
  showSceneObjects?: boolean;
  sceneObjectOpacity?: number;
  // Hold each subject at a fixed spot on the floor, showing pose only: the
  // joint rotations are what a clinician reads, while in-plane travel is the
  // noisiest thing a single camera infers. Vertical motion is kept, so a jump
  // or a crouch still reads.
  lockInPlace?: boolean;
};

const CAM_TO_WORLD = new THREE.Matrix4().set(
  0, 0, -1, 0,
  1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
);
const DISPLAY_YAW_CLOCKWISE_90 = new THREE.Matrix4().makeRotationZ(-Math.PI / 2);
const MAX_VERTEX_CACHE_ENTRIES = 1080; // per run — every subject of a multi-subject scene keeps its own budget
const INITIAL_RUN_PRELOAD_FRAMES = 360;
const PLAYBACK_PRELOAD_RADIUS = 360;
const RUN_PRELOAD_CONCURRENCY = 8;
// Pose fallback stays LOCAL: beyond this, showing a distant frame's pose is a
// visible "pop", so the mesh holds its last shown geometry instead (sticky).
const MESH_FALLBACK_RADIUS = 12;
const PRELOAD_WINDOW_STEP = 30;
// One-Euro parameters for the displayed root trajectory, per VIEWER axis.
// Viewer Y is the camera-depth axis after the world remap + display yaw —
// monocular depth is by far the noisiest signal (it derives from bbox scale),
// so it gets the lowest cutoff. Viewer X (camera-right) is optically stable.
const ROOT_FILTER_X = { minCutoff: 1.0, beta: 0.3, dCutoff: 1.0 };
const ROOT_FILTER_Y = { minCutoff: 0.5, beta: 0.15, dCutoff: 1.0 };
const ROOT_FILTER_Z = { minCutoff: 1.2, beta: 0.4, dCutoff: 1.0 };
// Contact/flight model: the run's ground height is the 10th percentile of the
// lowest foot joint's height (robust to jumps being a minority of frames).
// Clearance below SNAP is treated as ground contact (depth/pose noise), so the
// foot sits exactly on the grid; above it the body is airborne and rises by
// (clearance - SNAP) — continuous at the threshold, so takeoff cannot pop.
const GROUND_PERCENTILE = 0.1;
const CONTACT_SNAP_M = 0.05;
const LIFT_FILTER = { minCutoff: 1.5, beta: 0.6, dCutoff: 1.0 };
// Display-time One-Euro on the mesh vertex buffer (body-relative, i.e. after
// pivot subtraction, so trajectory motion is untouched): residual per-frame
// POSE noise — limbs faintly wiggling, hallucinated legs trembling — melts
// away at rest, while a fast real gesture (metres/second) raises the cutoff
// far above the floor and passes through essentially unfiltered.
const VERTEX_FILTER = { minCutoff: 1.5, beta: 3.0, dCutoff: 1.0 };
const VERTEX_FILTER_MAX_STEP_FRAMES = 6; // seek/jump => reset instead of blending

// One-Euro over a body-relative buffer that advances with the playhead.
//
// The mesh vertices and the skeleton joints describe the SAME body, so they
// have to be smoothed by the SAME filter: smoothing the mesh alone leaves the
// bones running ahead of the flesh on every movement. Both call this, which is
// what makes drifting apart impossible rather than merely unlikely.
//
// A seek, a rewind or a long gap resets instead of blending across it.
function useBodyRelativeFilter() {
  const stateRef = useRef<{ values: Float32Array; derivs: Float32Array; cursor: number } | null>(null);
  return useCallback((array: Float32Array, frameCursor: number, fps: number) => {
    const state = stateRef.current;
    const stepFrames = state ? frameCursor - state.cursor : 0;
    if (
      !state ||
      state.values.length !== array.length ||
      stepFrames <= 0 ||
      stepFrames > VERTEX_FILTER_MAX_STEP_FRAMES
    ) {
      stateRef.current = { values: array.slice(), derivs: new Float32Array(array.length), cursor: frameCursor };
      return;
    }
    const dt = stepFrames / Math.max(1, fps);
    const aD = smoothingAlpha(VERTEX_FILTER.dCutoff, dt);
    const { values: prev, derivs } = state;
    for (let index = 0; index < array.length; index += 1) {
      const raw = array[index];
      const dRaw = (raw - prev[index]) / dt;
      const deriv = derivs[index] + aD * (dRaw - derivs[index]);
      derivs[index] = deriv;
      const cutoff = VERTEX_FILTER.minCutoff + VERTEX_FILTER.beta * Math.abs(deriv);
      const a = smoothingAlpha(cutoff, dt);
      const filtered = prev[index] + a * (raw - prev[index]);
      prev[index] = filtered;
      array[index] = filtered;
    }
    state.cursor = frameCursor;
  }, []);
}
// Filter for the upright-correction TARGET (the body-up vector): posture
// changes are slow, shoulder-line wobble is not.
const UPRIGHT_FILTER = { minCutoff: 0.4, beta: 1.0, dCutoff: 1.0 };
const BODY_BONE_SEGMENTS: Array<[number, number]> = [
  [69, 0],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [69, 5],
  [69, 6],
  [5, 6],
  [5, 9],
  [6, 10],
  [9, 10],
  [5, 7],
  [7, 62],
  [6, 8],
  [8, 41],
  [9, 11],
  [11, 13],
  [13, 15],
  [13, 16],
  [13, 17],
  [10, 12],
  [12, 14],
  [14, 18],
  [14, 19],
  [14, 20],
];
const RIGHT_HAND_BONE_SEGMENTS: Array<[number, number]> = [
  [41, 24],
  [24, 23],
  [23, 22],
  [22, 21],
  [41, 28],
  [28, 27],
  [27, 26],
  [26, 25],
  [41, 32],
  [32, 31],
  [31, 30],
  [30, 29],
  [41, 36],
  [36, 35],
  [35, 34],
  [34, 33],
  [41, 40],
  [40, 39],
  [39, 38],
  [38, 37],
];
const LEFT_HAND_BONE_SEGMENTS: Array<[number, number]> = [
  [62, 45],
  [45, 44],
  [44, 43],
  [43, 42],
  [62, 49],
  [49, 48],
  [48, 47],
  [47, 46],
  [62, 53],
  [53, 52],
  [52, 51],
  [51, 50],
  [62, 57],
  [57, 56],
  [56, 55],
  [55, 54],
  [62, 61],
  [61, 60],
  [60, 59],
  [59, 58],
];
const SKELETON_BONE_SEGMENTS = [
  ...BODY_BONE_SEGMENTS,
  ...RIGHT_HAND_BONE_SEGMENTS,
  ...LEFT_HAND_BONE_SEGMENTS,
];
const meshFacesCache = new Map<string, Promise<Uint32Array>>();
type MeshVertexResource = {
  promise: Promise<Float32Array>;
  data: Float32Array | null;
};
// Vertex buffers are cached PER RUN with a per-run budget. A single shared
// budget flickers with multi-subject scenes: several runs' preloaders evict
// each other's entries, so the frame being displayed keeps losing its vertices
// and the mesh blinks in and out. Per-run FIFO keeps every subject's playback
// window resident no matter how many subjects share the scene.
const meshVerticesCacheByRun = new Map<string, Map<string, MeshVertexResource>>();

function runVertexCache(runId: string): Map<string, MeshVertexResource> {
  let cache = meshVerticesCacheByRun.get(runId);
  if (!cache) {
    cache = new Map<string, MeshVertexResource>();
    meshVerticesCacheByRun.set(runId, cache);
  }
  return cache;
}

// Insert a vertex resource into the run's cache, evicting its oldest entry once the cap is hit.
function rememberVertexResource(
  runId: string,
  meshFile: string,
  resource: MeshVertexResource,
): Promise<Float32Array> {
  const cache = runVertexCache(runId);
  if (!cache.has(meshFile) && cache.size >= MAX_VERTEX_CACHE_ENTRIES) {
    const oldest = cache.keys().next().value as string | undefined;
    if (oldest) {
      cache.delete(oldest);
    }
  }
  cache.set(meshFile, resource);
  return resource.promise;
}

// Fetch (and memoize per run) the shared mesh face/index buffer; topology is constant across frames.
function loadMeshFaces(runId: string): Promise<Uint32Array> {
  const existing = meshFacesCache.get(runId);
  if (existing) {
    return existing;
  }
  const promise = apiFetch(`/api/runs/${encodeURIComponent(runId)}/mesh-faces`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Mesh faces request failed: ${response.status}`);
      }
      return response.arrayBuffer();
    })
    .then((buffer) => new Uint32Array(buffer));
  meshFacesCache.set(runId, promise);
  return promise;
}

// Fetch a frame's vertex buffer, caching the in-flight promise and its result; on failure the entry is dropped so it can be retried.
function loadMeshVertices(runId: string, meshFile: string): Promise<Float32Array> {
  const cache = runVertexCache(runId);
  const existing = cache.get(meshFile);
  if (existing) {
    return existing.promise;
  }
  let resource: MeshVertexResource;
  const promise = apiFetch(`/api/runs/${encodeURIComponent(runId)}/mesh-vertices/${encodeURIComponent(meshFile)}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Mesh vertices request failed: ${response.status}`);
      }
      return response.arrayBuffer();
    })
    .then((buffer) => {
      const vertices = new Float32Array(buffer);
      resource.data = vertices;
      return vertices;
    })
    .catch((error) => {
      if (cache.get(meshFile) === resource) {
        cache.delete(meshFile);
      }
      throw error;
    });
  resource = { data: null, promise };
  return rememberVertexResource(runId, meshFile, resource);
}

// Return already-resolved vertices for a frame, or null if not yet loaded (no fetch is triggered).
function cachedMeshVertices(runId: string, meshFile: string): Float32Array | null {
  return meshVerticesCacheByRun.get(runId)?.get(meshFile)?.data ?? null;
}

// Whether a fetch for this frame's vertices has been started (resolved or in-flight).
function hasMeshVertexResource(runId: string, meshFile: string): boolean {
  return meshVerticesCacheByRun.get(runId)?.has(meshFile) ?? false;
}

// Find the closest frame (within radius) whose mesh is already cached, so playback can show a
// nearby pose instead of popping to empty while the exact frame is still loading.
function nearestCachedMeshFrameIndex(
  runId: string,
  frames: RunFrame[],
  frameIndex: number,
  radius = MESH_FALLBACK_RADIUS,
): number | null {
  if (frames[frameIndex]?.subjectPresent === false) {
    return null;
  }
  for (let offset = 0; offset <= radius; offset += 1) {
    const candidates = offset === 0 ? [frameIndex] : [frameIndex - offset, frameIndex + offset];
    for (const index of candidates) {
      const frame = frames[index];
      const meshFile = frame?.subjectPresent !== false ? frame?.meshFile : null;
      if (meshFile && cachedMeshVertices(runId, meshFile)) {
        return index;
      }
    }
  }
  return null;
}

export type RunAssetPreloadProgress = {
  loaded: number;
  total: number;
  label: string;
};

// Preload faces then the given vertex frames with a bounded worker pool, reporting progress as each loads.
function preloadMeshFiles(
  runId: string,
  meshFiles: string[],
  onProgress: (progress: RunAssetPreloadProgress) => void,
): Promise<void> {
  const total = meshFiles.length + 1;
  let loaded = 0;
  onProgress({ loaded, total, label: "Mesh topology" });
  return loadMeshFaces(runId)
    .then(async () => {
      loaded += 1;
      onProgress({ loaded, total, label: "3D frames" });
      let nextIndex = 0;
      async function worker(): Promise<void> {
        while (nextIndex < meshFiles.length) {
          const meshFile = meshFiles[nextIndex];
          nextIndex += 1;
          if (!cachedMeshVertices(runId, meshFile)) {
            await loadMeshVertices(runId, meshFile);
          }
          loaded += 1;
          onProgress({ loaded, total, label: "3D frames" });
        }
      }
      await Promise.all(
        Array.from({ length: Math.min(RUN_PRELOAD_CONCURRENCY, meshFiles.length) }, () => worker()),
      );
    });
}

// Warm the mesh cache for the start of a run before playback, collecting the first batch of unique
// mesh files (skipping absent-subject frames) and preloading them; a no-op when meshes are disabled.
export async function preloadRunAssets(
  runDetail: RunDetail,
  onProgress: (progress: RunAssetPreloadProgress) => void,
  options: { includeMeshes?: boolean } = {},
): Promise<void> {
  if (!runDetail.hasMeshes || options.includeMeshes === false) {
    onProgress({ loaded: 1, total: 1, label: "Kinematics" });
    return;
  }
  const meshFiles: string[] = [];
  const seen = new Set<string>();
  for (const frame of runDetail.frames) {
    if (frame.subjectPresent === false || !frame.meshFile || seen.has(frame.meshFile)) {
      continue;
    }
    seen.add(frame.meshFile);
    meshFiles.push(frame.meshFile);
    if (meshFiles.length >= INITIAL_RUN_PRELOAD_FRAMES) {
      break;
    }
  }
  if (meshFiles.length === 0) {
    onProgress({ loaded: 1, total: 1, label: "Kinematics" });
    return;
  }
  await preloadMeshFiles(runDetail.id, meshFiles, onProgress);
}

// Neutral body color for the reconstructed mesh.
const MESH_COLOR = "#9aa7b8";

// Extract the world-space recentering anchor (in camera coords) from the run,
// preserving runs written before floor_y was stored.
function displayAnchor(runDetail: RunDetail): DisplayAnchor | null {
  const anchor = runDetail.spaceView?.world_anchor ?? null;
  if (!anchor || !Number.isFinite(anchor.center_x) || !Number.isFinite(anchor.center_z)) {
    return null;
  }
  return {
    centerX: anchor.center_x,
    centerZ: anchor.center_z,
    ...(Number.isFinite(anchor.floor_y) ? { floorY: anchor.floor_y } : {}),
  };
}

// Shift a world-space point so the capture's anchor sits at the origin (mutates and returns point).
function centerWorld(point: THREE.Vector3, anchor: DisplayAnchor | null): THREE.Vector3 {
  if (!anchor) {
    return point;
  }
  // Anchor is stored in camera coordinates. After CAM_TO_WORLD, horizontal
  // centre is [-centerZ, centerX] and the floor is world Z = -floorY.
  point.x += anchor.centerZ;
  point.y -= anchor.centerX;
  point.z += anchor.floorY ?? 0;
  return point;
}

// Convert a camera-space point into viewer space: axis remap, recenter on the anchor, then apply display yaw.
function camToWorld(point: [number, number, number], anchor: DisplayAnchor | null): THREE.Vector3 {
  return centerWorld(new THREE.Vector3(-point[2], point[0], -point[1]), anchor).applyMatrix4(DISPLAY_YAW_CLOCKWISE_90);
}

// Convert an already-world-space point into viewer space (recenter on the anchor, then apply display yaw).
function worldToViewer(point: THREE.Vector3, anchor: DisplayAnchor | null): THREE.Vector3 {
  return centerWorld(point.clone(), anchor).applyMatrix4(DISPLAY_YAW_CLOCKWISE_90);
}

// Matrix4.set takes row-major arguments while Three stores the result internally
// in column-major order, so this is the direct boundary for the artifact API.
function matrixFromRowMajor(values: SceneObjectMatrix): THREE.Matrix4 {
  return new THREE.Matrix4().set(
    values[0], values[1], values[2], values[3],
    values[4], values[5], values[6], values[7],
    values[8], values[9], values[10], values[11],
    values[12], values[13], values[14], values[15],
  );
}

function matrixToRowMajor(matrix: THREE.Matrix4): SceneObjectMatrix {
  const values = matrix.elements;
  return [
    values[0], values[4], values[8], values[12],
    values[1], values[5], values[9], values[13],
    values[2], values[6], values[10], values[14],
    values[3], values[7], values[11], values[15],
  ];
}

// Convert a raw-mesh -> world artifact transform into the viewer's fixed
// coordinate system. This mirrors worldToViewer for points: recenter first,
// then apply the display yaw, giving viewerFromWorld * objectToWorld.
function sceneObjectToViewerMatrix(
  objectToWorld: SceneObjectMatrix,
  anchor: DisplayAnchor | null,
): THREE.Matrix4 {
  const viewerFromWorld = DISPLAY_YAW_CLOCKWISE_90.clone();
  if (anchor) {
    viewerFromWorld.multiply(
      new THREE.Matrix4().makeTranslation(anchor.centerZ, -anchor.centerX, anchor.floorY ?? 0),
    );
  }
  return viewerFromWorld.multiply(matrixFromRowMajor(objectToWorld));
}

// Viewer-space root position for a frame, preferring stabilized/raw world root and falling back to the camera-comp.
function framePosition(frame: RunFrame, sourceType: "stabilized" | "raw", anchor: DisplayAnchor | null): THREE.Vector3 {
  const source =
    (sourceType === "stabilized" ? frame.rootWorldStabilized : frame.rootWorldRaw) ??
    frame.rootWorldStabilized ??
    frame.rootWorldRaw ??
    [-frame.cameraComp[2], frame.cameraComp[0], -frame.cameraComp[1]];
  return worldToViewer(new THREE.Vector3(source[0], source[1], source[2]), anchor);
}

// Linear interpolation between two vectors with t clamped to [0, 1] (does not mutate inputs).
function lerpVector(a: THREE.Vector3, b: THREE.Vector3, t: number): THREE.Vector3 {
  return a.clone().lerp(b, Math.max(0, Math.min(1, t)));
}

// Scalar linear interpolation with t clamped to [0, 1].
function lerpNumber(a: number, b: number, t: number): number {
  const k = Math.max(0, Math.min(1, t));
  return a * (1 - k) + b * k;
}

// Component-wise linear interpolation between two [x, y, z] triplets with t clamped to [0, 1].
function interpolateTriplet(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  const amount = Math.max(0, Math.min(1, t));
  return [
    a[0] + (b[0] - a[0]) * amount,
    a[1] + (b[1] - a[1]) * amount,
    a[2] + (b[2] - a[2]) * amount,
  ];
}

// Interpolate two joint arrays; falls back to a when shapes differ so we never blend mismatched skeletons.
function interpolateJoints(
  a: Array<[number, number, number]> | null | undefined,
  b: Array<[number, number, number]> | null | undefined,
  t: number,
): Array<[number, number, number]> | null | undefined {
  if (!a || !b || a.length !== b.length) {
    return a;
  }
  return a.map((joint, index) => interpolateTriplet(joint, b[index], t));
}

// Tween a frame's joints toward the next frame; returns a unchanged when either frame lacks a subject.
function interpolateFrame(a: RunFrame, b: RunFrame, t: number): RunFrame {
  if (a === b || t <= 0 || a.subjectPresent === false || b.subjectPresent === false) {
    return a;
  }
  return {
    ...a,
    jointsCam: interpolateJoints(a.jointsCam, b.jointsCam, t),
  };
}

// Viewer-space centroid of the torso/hip/knee joints (falling back to all joints) used as the camera focus.
function bodyFocusPosition(frame: RunFrame, anchor: DisplayAnchor | null): THREE.Vector3 | null {
  if (frame.subjectPresent === false) {
    return null;
  }
  const joints = frame.jointsCam ?? null;
  if (!joints || joints.length === 0) {
    return null;
  }
  const preferred = [5, 6, 9, 10, 11, 12, 13, 14]
    .map((index) => joints[index])
    .filter((joint): joint is [number, number, number] => Boolean(joint));
  const source = preferred.length > 0 ? preferred : joints;
  const center = new THREE.Vector3();
  let count = 0;
  for (const joint of source) {
    const p = camToWorld(joint, anchor);
    if (Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)) {
      center.add(p);
      count += 1;
    }
  }
  return count > 0 ? center.multiplyScalar(1 / count) : null;
}

// Whether the frame carries a usable world root (subject present and root displaced from the origin).
function frameHasMeaningfulRoot(frame: RunFrame): boolean {
  if (frame.subjectPresent === false) {
    return false;
  }
  const root = frame.rootWorldRaw ?? frame.rootWorldStabilized;
  if (!root) {
    return false;
  }
  return Math.hypot(root[0], root[1], root[2]) > 1e-4;
}

// Rotation that stands the body upright by aligning its hip->shoulder (or foot->shoulder) axis with world up;
// returns identity for small tilts or when the required joints are missing.
// Raw body-up direction (shoulder centre relative to feet/hips) in viewer
// space; null when the joints are unavailable.
function bodyUpVector(frame: RunFrame | null, anchor: DisplayAnchor | null): THREE.Vector3 | null {
  if (frame?.subjectPresent === false) {
    return null;
  }
  const joints = frame?.jointsCam ?? null;
  if (!joints || joints.length < 15) {
    return null;
  }
  const leftHip = joints[9] ? camToWorld(joints[9], anchor) : null;
  const rightHip = joints[10] ? camToWorld(joints[10], anchor) : null;
  const leftShoulder = joints[5] ? camToWorld(joints[5], anchor) : null;
  const rightShoulder = joints[6] ? camToWorld(joints[6], anchor) : null;
  if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) {
    return null;
  }
  const hipCenter = leftHip.clone().add(rightHip).multiplyScalar(0.5);
  const shoulderCenter = leftShoulder.clone().add(rightShoulder).multiplyScalar(0.5);
  const footPoints = [13, 14, 15, 16, 17, 18, 19, 20]
    .map((index) => joints[index])
    .filter((joint): joint is [number, number, number] => Boolean(joint))
    .map((joint) => camToWorld(joint, anchor));
  const footCenter =
    footPoints.length > 0
      ? footPoints.reduce((sum, point) => sum.add(point), new THREE.Vector3()).multiplyScalar(1 / footPoints.length)
      : null;
  const trunkUp = shoulderCenter.clone().sub(hipCenter);
  const fullBodyUp = footCenter ? shoulderCenter.clone().sub(footCenter) : trunkUp;
  const bodyUp = fullBodyUp.length() > 1e-5 ? fullBodyUp : trunkUp;
  return bodyUp.length() > 1e-5 ? bodyUp.normalize() : null;
}

// Upright correction from an up vector, with a graduated fade: a SMALL tilt is
// reconstruction lean error and gets fully straightened, but a LARGE tilt is a
// real posture — bending down, a floor roll, a fall — and must be shown as-is.
function uprightFromUp(bodyUp: THREE.Vector3 | null): THREE.Quaternion {
  if (!bodyUp) {
    return new THREE.Quaternion();
  }
  const worldUp = new THREE.Vector3(0, 0, 1);
  const tilt = bodyUp.angleTo(worldUp);
  if (!Number.isFinite(tilt) || tilt < THREE.MathUtils.degToRad(2)) {
    return new THREE.Quaternion();
  }
  const fadeStart = THREE.MathUtils.degToRad(25);
  const fadeEnd = THREE.MathUtils.degToRad(50);
  const full = new THREE.Quaternion().setFromUnitVectors(bodyUp, worldUp);
  if (tilt <= fadeStart) {
    return full;
  }
  const weight = tilt >= fadeEnd ? 0 : 1 - (tilt - fadeStart) / (fadeEnd - fadeStart);
  return new THREE.Quaternion().slerp(full, weight);
}

// Angle (radians) of the shortest rotation between two quaternions.
function quaternionAngle(a: THREE.Quaternion, b: THREE.Quaternion): number {
  return 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
}

// Per-frame upright rotations, temporally smoothed: reject implausibly large raw tilts and slerp toward
// each candidate (more slowly across big jumps) so the standing-up correction doesn't jitter during playback.
function stableUprightQuaternions(
  frames: RunFrame[],
  anchor: DisplayAnchor | null,
  fps: number,
): THREE.Quaternion[] {
  const identity = new THREE.Quaternion();
  const maxRawTilt = THREE.MathUtils.degToRad(70);
  const maxJump = THREE.MathUtils.degToRad(28);
  // One-Euro the UP-VECTOR TARGET itself: it is derived from the shoulder
  // line, which wobbles with every gesture, and the correction rotates the
  // whole body about the pelvis — target noise reads as the head swaying
  // forward/back. Real posture changes are slow and pass through.
  const ups = frames.map((frame) => bodyUpVector(frame, anchor));
  const xs = filterSeries(ups.map((u) => (u ? u.x : null)), fps, UPRIGHT_FILTER);
  const ys = filterSeries(ups.map((u) => (u ? u.y : null)), fps, UPRIGHT_FILTER);
  const zs = filterSeries(ups.map((u) => (u ? u.z : null)), fps, UPRIGHT_FILTER);
  const out: THREE.Quaternion[] = [];
  for (let index = 0; index < frames.length; index += 1) {
    const x = xs[index];
    const y = ys[index];
    const z = zs[index];
    const filteredUp =
      x !== null && y !== null && z !== null
        ? new THREE.Vector3(x, y, z)
        : null;
    const up = filteredUp && filteredUp.length() > 1e-5 ? filteredUp.normalize() : null;
    const raw = uprightFromUp(up);
    const rawTilt = quaternionAngle(identity, raw);
    const candidate =
      up && Number.isFinite(rawTilt) && rawTilt <= maxRawTilt ? raw : (out.at(-1) ?? identity);
    if (out.length === 0) {
      out.push(candidate.clone());
      continue;
    }
    const previous = out[out.length - 1];
    const jump = quaternionAngle(previous, candidate);
    const alpha = jump > maxJump ? 0.1 : 0.25;
    out.push(previous.clone().slerp(candidate, alpha).normalize());
  }
  return out;
}

// ── Sole-pitch calibration ───────────────────────────────────────────────────
// Even after the trunk is straightened, the reconstructed bodies often stand
// "on their heels": monocular reconstruction (with its guessed focal length)
// biases the whole body backward, and a rigid trunk correction cannot flatten
// the FEET. Physics again provides the target: a planted foot's sole
// (heel -> toe) must be horizontal. Measure the median signed sole pitch over
// all contact frames (a per-run constant — it is a systematic bias, not
// noise) and counter-rotate the body about the HIP AXIS, which makes the fix
// independent of which way the person is facing.

const SOLE_PITCH_MIN_RAD = THREE.MathUtils.degToRad(1.5); // below: leave as-is
const SOLE_PITCH_MAX_RAD = THREE.MathUtils.degToRad(20); // safety clamp (measured
// bias on real footage is ~14-15 deg, remarkably consistent across subjects)

// Horizontalized hip axis of a frame after the upright rotation (unit), or null.
function hipAxisAfter(
  frame: RunFrame,
  anchor: DisplayAnchor | null,
  upright: THREE.Quaternion,
): THREE.Vector3 | null {
  const joints = frame.jointsCam ?? null;
  if (!joints || !joints[9] || !joints[10] || frame.subjectPresent === false) {
    return null;
  }
  const axis = camToWorld(joints[10], anchor)
    .sub(camToWorld(joints[9], anchor))
    .applyQuaternion(upright);
  axis.z = 0;
  return axis.length() > 1e-4 ? axis.normalize() : null;
}

// Sole (heel -> big toe) pitch of one planted foot after upright, in radians;
// positive = toes above heel. Null when unusable.
function solePitch(
  frame: RunFrame,
  anchor: DisplayAnchor | null,
  upright: THREE.Quaternion,
  toeIndex: number,
  heelIndex: number,
): number | null {
  const joints = frame.jointsCam ?? null;
  if (!joints || !joints[toeIndex] || !joints[heelIndex]) {
    return null;
  }
  const v = camToWorld(joints[toeIndex], anchor)
    .sub(camToWorld(joints[heelIndex], anchor))
    .applyQuaternion(upright);
  const horizontal = Math.hypot(v.x, v.y);
  if (horizontal < 0.05) {
    return null; // degenerate (foot seen end-on)
  }
  return Math.atan2(v.z, horizontal);
}

// Median signed sole-pitch of the run's planted feet (radians), with the
// rotation SIGN resolved empirically: of the two candidate corrections about
// the hip axis, keep the one that actually flattens a representative sole.
function computeSolePitchFix(
  frames: RunFrame[],
  anchor: DisplayAnchor | null,
  uprights: THREE.Quaternion[],
  videoHeight: number | null,
): number {
  const samples: number[] = [];
  let representative: { frame: RunFrame; upright: THREE.Quaternion } | null = null;
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    if (!feetVisible(frame, videoHeight)) {
      continue;
    }
    const support = frame.footContact?.support;
    if (support !== "left" && support !== "right" && support !== "both") {
      continue;
    }
    const upright = uprights[index] ?? new THREE.Quaternion();
    const feet: Array<[number, number]> = [];
    if (support === "left" || support === "both") {
      feet.push([15, 17]);
    }
    if (support === "right" || support === "both") {
      feet.push([18, 20]);
    }
    for (const [toe, heel] of feet) {
      const pitch = solePitch(frame, anchor, upright, toe, heel);
      if (pitch !== null && Number.isFinite(pitch)) {
        samples.push(pitch);
        representative = representative ?? { frame, upright };
      }
    }
  }
  if (samples.length < 10 || !representative) {
    return 0;
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length / 2)];
  if (Math.abs(median) < SOLE_PITCH_MIN_RAD) {
    return 0;
  }
  const magnitude = Math.min(Math.abs(median), SOLE_PITCH_MAX_RAD);
  // Resolve the rotation sign on real data instead of reasoning about axis
  // handedness: pick the candidate that reduces the representative sole pitch.
  const axis = hipAxisAfter(representative.frame, anchor, representative.upright);
  if (!axis) {
    return 0;
  }
  const measure = (angle: number): number => {
    const fixed = new THREE.Quaternion()
      .setFromAxisAngle(axis, angle)
      .multiply(representative!.upright);
    const p = solePitch(representative!.frame, anchor, fixed, 15, 17) ??
      solePitch(representative!.frame, anchor, fixed, 18, 20);
    return p === null ? Number.POSITIVE_INFINITY : Math.abs(p);
  };
  return measure(magnitude) <= measure(-magnitude) ? magnitude : -magnitude;
}

// Lowest BODY-joint height (after pivot subtraction and the upright rotation)
// so the body's lowest contact point can be dropped onto the grid. Whole body,
// not just the feet: during a floor roll the torso is the contact point — a
// foot-only rule would push the body below the grid. Standing/walking the
// lowest joint IS a foot, so gait is unchanged. Null when joints are missing.
function bodyGroundOffset(
  frame: RunFrame | null,
  quaternion: THREE.Quaternion,
  anchor: DisplayAnchor | null,
  pivot: THREE.Vector3,
): number | null {
  const joints = frame?.jointsCam ?? null;
  if (!joints || joints.length === 0) {
    return null;
  }
  let min: number | null = null;
  for (const joint of joints) {
    if (!joint) {
      continue;
    }
    const p = camToWorld(joint, anchor).sub(pivot).applyQuaternion(quaternion);
    if (Number.isFinite(p.z) && (min === null || p.z < min)) {
      min = p.z;
    }
  }
  return min;
}

// Absolute (viewer-space, pre-pivot) height of the lowest foot joint of a frame.
function lowestFootViewerZ(frame: RunFrame, anchor: DisplayAnchor | null): number | null {
  const joints = frame.jointsCam ?? null;
  if (!joints || joints.length < 15 || frame.subjectPresent === false) {
    return null;
  }
  let min: number | null = null;
  for (const index of [13, 14, 15, 16, 17, 18, 19, 20]) {
    const joint = joints[index];
    if (!joint) {
      continue;
    }
    const z = camToWorld(joint, anchor).z;
    if (Number.isFinite(z) && (min === null || z < min)) {
      min = z;
    }
  }
  return min;
}

// Absolute height of the lowest joint of the WHOLE body. This — not the feet —
// is the right reference for ground contact in general: during a floor roll
// the torso is the lowest point (feet are in the air; a foot-based rule would
// wrongly hoist the body), while standing/walking the lowest body joint IS a
// foot, so ordinary gait behaves identically.
function lowestBodyViewerZ(frame: RunFrame, anchor: DisplayAnchor | null): number | null {
  const joints = frame.jointsCam ?? null;
  if (!joints || joints.length === 0 || frame.subjectPresent === false) {
    return null;
  }
  let min: number | null = null;
  for (const joint of joints) {
    if (!joint) {
      continue;
    }
    const z = camToWorld(joint, anchor).z;
    if (Number.isFinite(z) && (min === null || z < min)) {
      min = z;
    }
  }
  return min;
}

// Whether the subject's feet are actually IN the picture. SAM 3D Body predicts
// a full SMPL body even when the video is framed at the waist: the legs below
// the crop are HALLUCINATED and jitter freely. Any logic that trusts foot
// joints (ground contact, support anchoring, foot-based grounding) must be
// gated on this — a detection box clipped by the bottom edge of the frame
// means the lower body is cut off.
function feetVisible(frame: RunFrame, videoHeight: number | null): boolean {
  if (!frame.bbox || !videoHeight) {
    return false;
  }
  return frame.bbox[3] < videoHeight * 0.985;
}

// Support-foot horizontal anchoring (a zero-velocity update, as used in
// inertial navigation and marker-based mocap cleanup).
//
// Monocular depth comes from bbox scale, so it wobbles with POSE: raising an
// arm enlarges the crop and the whole body appears to lunge forward/backward.
// A temporal filter cannot remove this — it has real-motion velocity. But
// physics gives a hard invariant: a foot in ground contact is STATIONARY in
// the world. Any apparent translation of the planted foot between frames is
// reconstruction noise, so the negated foot drift is applied to the WHOLE
// body. Standing bodies become rock solid regardless of gesturing; walking
// keeps true step lengths (the anchor advances at every support switch);
// flight phases (jump — no contact) integrate the raw trajectory untouched.

const LEFT_FOOT_JOINTS = [13, 15, 16, 17]; // ankle, bigToe, smallToe, heel
const RIGHT_FOOT_JOINTS = [14, 18, 19, 20];
// The correction cancels the planted foot's apparent drift, which for a still
// body is pure reconstruction noise — measured on real footage: per-frame XY
// jitter p95 7.2mm -> 1.1mm, and a standing subject's spurious 19cm depth
// wander drops to real body sway.
//
// Left unbounded, though, it also cancels WALKING: the contact labels can read
// "both" for a whole clip so the anchor never releases, and the support foot of
// someone crossing the room genuinely translates. The correction then grows to
// metres and drags the body with it — measured on a real walking clip, 2.86 m
// of travel reached the screen as 1.00 m.
//
// The bound is what separates the two cases, because they differ by an order of
// magnitude: reconstruction wobble is CENTIMETRIC, locomotion is METRIC. On a
// standing subject the correction never exceeds 11 cm on its own, so a 15 cm
// ceiling never binds and the full benefit is kept; on a walking one it
// saturates, capping how far the body can be displaced while the travel itself
// passes through. Measured at 15 cm: standing depth wander still 0.19 m -> 0.06
// m, walking travel 2.89 m -> 2.74 m.
const ANCHOR_MAX_CORRECTION_M = 0.15;

function footXY(
  frame: RunFrame,
  indices: number[],
  anchor: DisplayAnchor | null,
): THREE.Vector2 | null {
  const joints = frame.jointsCam ?? null;
  if (!joints || joints.length < 21 || frame.subjectPresent === false) {
    return null;
  }
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const index of indices) {
    const joint = joints[index];
    if (!joint) {
      continue;
    }
    const p = camToWorld(joint, anchor);
    if (Number.isFinite(p.x) && Number.isFinite(p.y)) {
      sx += p.x;
      sy += p.y;
      n += 1;
    }
  }
  return n > 0 ? new THREE.Vector2(sx / n, sy / n) : null;
}

// Which foot (if any) is in ground contact on this frame. Prefers the
// analysis-provided contact labels; falls back to the lowest-foot clearance.
// Contact labels are themselves derived from foot joints, so a frame whose
// feet are OUT OF THE PICTURE (hallucinated legs) can never claim support.
function frameSupport(
  frame: RunFrame,
  anchor: DisplayAnchor | null,
  ground: number | null,
  videoHeight: number | null,
): "left" | "right" | "both" | "none" {
  if (!feetVisible(frame, videoHeight)) {
    return "none";
  }
  const stored = frame.footContact?.support;
  if (stored === "left" || stored === "right" || stored === "both") {
    return stored;
  }
  if (stored === "none") {
    return "none";
  }
  if (ground === null) {
    return "none";
  }
  const z = lowestFootViewerZ(frame, anchor);
  return z !== null && z - ground <= CONTACT_SNAP_M * 2 ? "both" : "none";
}

function anchoredTrajectory(
  frames: RunFrame[],
  rawRoots: THREE.Vector3[],
  anchor: DisplayAnchor | null,
  videoHeight: number | null,
): THREE.Vector3[] {
  // Ground estimate for the contact fallback (same as computeLiftSeries).
  const heights: number[] = [];
  for (const frame of frames) {
    const z = lowestBodyViewerZ(frame, anchor);
    if (z !== null) {
      heights.push(z);
    }
  }
  heights.sort((a, b) => a - b);
  const ground =
    heights.length > 0
      ? heights[Math.min(heights.length - 1, Math.floor(heights.length * GROUND_PERCENTILE))]
      : null;

  const correction = new THREE.Vector2(0, 0);
  let anchorXY: THREE.Vector2 | null = null;
  let lastSupport: string = "none";
  const out: THREE.Vector3[] = [];
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    const support = frameSupport(frame, anchor, ground, videoHeight);
    const indices =
      support === "left"
        ? LEFT_FOOT_JOINTS
        : support === "right"
          ? RIGHT_FOOT_JOINTS
          : support === "both"
            ? [...LEFT_FOOT_JOINTS, ...RIGHT_FOOT_JOINTS]
            : null;
    const supportXY = indices ? footXY(frame, indices, anchor) : null;
    if (supportXY) {
      if (support !== lastSupport || anchorXY === null) {
        // Re-capture at the foot's current CORRECTED position: the correction
        // target equals the current correction at the switch instant, so the
        // displayed trajectory is value-continuous across support changes.
        anchorXY = supportXY.clone().add(correction);
      }
      correction.copy(anchorXY).sub(supportXY);
      correction.clampLength(0, ANCHOR_MAX_CORRECTION_M);
    }
    // No contact (flight, missing joints): the correction is frozen — the raw
    // motion passes through untouched.
    lastSupport = supportXY ? support : "none";
    const raw = rawRoots[index] ?? new THREE.Vector3();
    out.push(new THREE.Vector3(raw.x + correction.x, raw.y + correction.y, raw.z));
  }
  return out;
}

// Pin a trajectory to the origin, keeping its vertical motion.
//
// Sitting on (0, 0) puts the subject on the axis cross at the middle of the
// grid, which is where you want it when the point is to read pose rather than
// travel. Several subjects pinned at once therefore overlap — use the subject
// picker to show one at a time.
function pinnedInPlace(positions: THREE.Vector3[]): THREE.Vector3[] {
  return positions.map((point) => new THREE.Vector3(0, 0, point.z));
}

// The full displayed root trajectory of one subject: support-foot anchored
// (kills pose-correlated depth wobble while any foot is planted), then
// One-Euro filtered per axis (kills residual high-frequency jitter with
// near-zero lag during real motion).
function filteredDisplayTrajectory(
  frames: RunFrame[],
  rawRoots: THREE.Vector3[],
  anchor: DisplayAnchor | null,
  fps: number,
  videoHeight: number | null,
): THREE.Vector3[] {
  const anchored = anchoredTrajectory(frames, rawRoots, anchor, videoHeight);
  const xs = filterSeries(anchored.map((p) => p.x), fps, ROOT_FILTER_X);
  const ys = filterSeries(anchored.map((p) => p.y), fps, ROOT_FILTER_Y);
  const zs = filterSeries(anchored.map((p) => p.z), fps, ROOT_FILTER_Z);
  return anchored.map(
    (p, index) =>
      new THREE.Vector3(
        (xs[index] as number) ?? p.x,
        (ys[index] as number) ?? p.y,
        (zs[index] as number) ?? p.z,
      ),
  );
}

// Per-frame vertical LIFT of a subject above the run's ground plane.
//
// The previous design pinned the lowest foot to z=0 on EVERY frame, which made
// vertical motion physically impossible to display: a jump stayed glued to the
// floor (and mid-air tucking pushed the pelvis DOWN). Instead, estimate the
// ground once per run (robust low percentile of the lowest-foot height across
// the whole clip), then per frame let the body rise by its foot clearance above
// that plane. Within CONTACT_SNAP_M of the ground the clearance is treated as
// noise and snapped to zero — walking and standing stay exactly on the grid —
// and the mapping `lift = max(0, clearance - SNAP)` is continuous, so takeoff
// and landing cannot pop. The clearance itself runs through a snappy One-Euro
// filter (jitter gone at rest, jumps tracked with minimal lag).
function computeLiftSeries(
  frames: RunFrame[],
  anchor: DisplayAnchor | null,
  fps: number,
  videoHeight: number | null,
): number[] {
  const rawClearance: Array<number | null> = new Array(frames.length).fill(null);
  const heights: number[] = [];
  for (let index = 0; index < frames.length; index += 1) {
    // Clearance of the lowest BODY joint (not just the feet): a floor roll's
    // reference is the torso; a jump raises every joint. Frames whose lower
    // body is cut off by the picture edge contribute nothing — hallucinated
    // legs would inject pure noise.
    const frame = frames[index];
    const z = feetVisible(frame, videoHeight) ? lowestBodyViewerZ(frame, anchor) : null;
    rawClearance[index] = z;
    if (z !== null) {
      heights.push(z);
    }
  }
  if (heights.length === 0) {
    return new Array(frames.length).fill(0);
  }
  heights.sort((a, b) => a - b);
  const ground = heights[Math.min(heights.length - 1, Math.floor(heights.length * GROUND_PERCENTILE))];
  const clearance = rawClearance.map((z) => (z === null ? null : z - ground));
  const filtered = filterSeries(clearance, fps, LIFT_FILTER);
  return filtered.map((c) => (c === null ? 0 : Math.max(0, c - CONTACT_SNAP_M)));
}

// Builds and renders the SMPL mesh for one frame: loads faces + vertices, optionally tweens toward the
// next frame's vertices, and bakes the camera->viewer transform and pivot into the geometry.
function MeshGeometry({
  runId,
  meshFile,
  nextMeshFile,
  meshInterpolation,
  pivot,
  anchor,
  meshOpacity,
  color,
  frameCursor,
  fps,
}: {
  runId: string;
  meshFile: string;
  nextMeshFile: string | null;
  meshInterpolation: number;
  pivot: THREE.Vector3;
  anchor: DisplayAnchor | null;
  meshOpacity: number;
  color?: string | null;
  frameCursor: number;
  fps: number;
}) {
  const [faces, setFaces] = useState<Uint32Array | null>(null);
  const [verticesState, setVerticesState] = useState<{ meshFile: string; vertices: Float32Array } | null>(null);

  // Load the shared face/index buffer once per run.
  useEffect(() => {
    let alive = true;
    loadMeshFaces(runId)
      .then((payload) => {
        if (alive) {
          setFaces(payload);
        }
      })
      .catch(() => {
        if (alive) {
          setFaces(null);
        }
      });
    return () => {
      alive = false;
    };
  }, [runId]);

  // Load the current frame's vertices, using the cache synchronously when available.
  useEffect(() => {
    let alive = true;
    const cached = cachedMeshVertices(runId, meshFile);
    if (cached) {
      setVerticesState({ meshFile, vertices: cached });
      return () => {
        alive = false;
      };
    }
    setVerticesState(null);
    loadMeshVertices(runId, meshFile)
      .then((payload) => {
        if (alive) {
          setVerticesState({ meshFile, vertices: payload });
        }
      })
      .catch(() => {
        if (alive) {
          setVerticesState(null);
        }
      });
    return () => {
      alive = false;
    };
  }, [meshFile, runId]);

  // Prefetch the next frame's vertices so interpolation has both endpoints ready.
  useEffect(() => {
    if (nextMeshFile && nextMeshFile !== meshFile) {
      void loadMeshVertices(runId, nextMeshFile).catch(() => undefined);
    }
  }, [meshFile, nextMeshFile, runId]);

  const vertices = cachedMeshVertices(runId, meshFile) ?? (verticesState?.meshFile === meshFile ? verticesState.vertices : null);
  const nextVertices = nextMeshFile ? cachedMeshVertices(runId, nextMeshFile) : null;
  // Temporal vertex filter state: previous body-relative positions +
  // derivative estimates, advanced as the playhead moves forward.
  const filterBuffer = useBodyRelativeFilter();
  const geometry = useMemo(() => {
    if (!faces || !vertices) {
      return null;
    }
    const next = new THREE.BufferGeometry();
    const positions = vertices.slice();
    const amount = Math.max(0, Math.min(1, meshInterpolation));
    if (nextVertices && nextVertices.length === vertices.length && amount > 1e-4) {
      for (let index = 0; index < positions.length; index += 1) {
        positions[index] = positions[index] * (1 - amount) + nextVertices[index] * amount;
      }
    }
    next.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    next.setIndex(new THREE.BufferAttribute(faces, 1));
    next.applyMatrix4(CAM_TO_WORLD);
    if (anchor) {
      next.translate(anchor.centerZ, -anchor.centerX, anchor.floorY ?? 0);
    }
    next.applyMatrix4(DISPLAY_YAW_CLOCKWISE_90);
    next.translate(-pivot.x, -pivot.y, -pivot.z);

    // Smooth the BODY-RELATIVE vertex buffer (root already subtracted, so this
    // touches pose only, never the trajectory) — with the same filter the
    // skeleton uses, so the bones stay inside the body.
    filterBuffer(next.getAttribute("position").array as Float32Array, frameCursor, fps);

    next.computeBoundingSphere();
    return next;
  }, [anchor, faces, filterBuffer, fps, frameCursor, meshInterpolation, nextVertices, pivot, vertices]);
  // STICKY geometry: while the exact frame's vertices are still loading (or
  // were evicted), keep showing the last built pose instead of returning null —
  // a blinking mesh is far more jarring than a briefly frozen one.
  // Disposal is deferred to AFTER commit: the R3F render loop can draw the
  // previous tree between render and commit, so disposing mid-render throws
  // inside the frame loop and the error boundary tears the scene down.
  const lastGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  useEffect(() => {
    if (geometry && geometry !== lastGeometryRef.current) {
      const previous = lastGeometryRef.current;
      lastGeometryRef.current = geometry;
      previous?.dispose();
    }
  }, [geometry]);
  useEffect(
    () => () => {
      lastGeometryRef.current?.dispose();
      lastGeometryRef.current = null;
    },
    [],
  );

  const shown = geometry ?? lastGeometryRef.current;
  if (!shown) {
    return null;
  }

  return (
    <mesh geometry={shown} castShadow receiveShadow>
      <meshBasicMaterial
        color={color || MESH_COLOR}
        transparent={meshOpacity < 0.995}
        opacity={meshOpacity}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Draws the skeleton as line segments connecting the configured bone joint pairs.
function SkeletonBones({ joints }: { joints: THREE.Vector3[] }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    for (const [startIndex, endIndex] of SKELETON_BONE_SEGMENTS) {
      const start = joints[startIndex];
      const end = joints[endIndex];
      if (!start || !end) {
        continue;
      }
      positions.push(start.x, start.y, start.z, end.x, end.y, end.z);
    }
    const nextGeometry = new THREE.BufferGeometry();
    nextGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return nextGeometry;
  }, [joints]);

  useEffect(() => {
    return () => geometry.dispose();
  }, [geometry]);

  if ((geometry.getAttribute("position")?.count ?? 0) === 0) {
    return null;
  }

  return (
    <lineSegments geometry={geometry} renderOrder={5}>
      <lineBasicMaterial color="#0f172a" transparent opacity={0.78} depthTest={false} />
    </lineSegments>
  );
}

// Assembles one subject for a frame: drops it onto the grid, applies the upright rotation, and renders
// the mesh, bones, and clickable joint markers according to the visibility toggles.

// Resolve a generic model transform at a fractional run-frame cursor. Poses
// are decomposed to preserve rotation and scale while interpolating; directly
// blending matrix coefficients would introduce shear.
function dynamicObjectMatrixAtCursor(
  object: DynamicObject,
  frameCursor: number,
): SceneObjectMatrix | null {
  const { poses } = object;
  let lower = 0;
  let upper = poses.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (poses[middle].frameIndex < frameCursor) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }

  const next = poses[lower];
  if (next && Math.abs(next.frameIndex - frameCursor) < 1e-6) {
    return next.objectToWorld;
  }
  const previous = poses[lower - 1];
  if (!previous || !next) {
    return null;
  }

  const frameGap = next.frameIndex - previous.frameIndex;
  if (frameGap <= 0 || frameGap > object.maxInterpolationGapFrames) {
    return null;
  }
  const amount = (frameCursor - previous.frameIndex) / frameGap;
  const start = matrixFromRowMajor(previous.objectToWorld);
  const end = matrixFromRowMajor(next.objectToWorld);
  const startPosition = new THREE.Vector3();
  const startRotation = new THREE.Quaternion();
  const startScale = new THREE.Vector3();
  const endPosition = new THREE.Vector3();
  const endRotation = new THREE.Quaternion();
  const endScale = new THREE.Vector3();
  start.decompose(startPosition, startRotation, startScale);
  end.decompose(endPosition, endRotation, endScale);
  const interpolated = new THREE.Matrix4().compose(
    startPosition.lerp(endPosition, amount),
    startRotation.slerp(endRotation, amount),
    startScale.lerp(endScale, amount),
  );
  return matrixToRowMajor(interpolated);
}

function DynamicObjectMesh({
  object,
  anchor,
  frameCursor,
  opacity,
}: {
  object: DynamicObject;
  anchor: DisplayAnchor | null;
  frameCursor: number;
  opacity: number;
}) {
  const [scene, setScene] = useState<THREE.Group | null>(null);
  useEffect(() => {
    let cancelled = false;
    let loaded: THREE.Group | null = null;
    setScene(null);
    import("three/examples/jsm/loaders/GLTFLoader.js")
      .then(({ GLTFLoader }) => {
        if (cancelled) return;
        new GLTFLoader().load(object.meshUrl, (gltf) => {
          if (cancelled) {
            disposeSceneObject(gltf.scene);
            return;
          }
          loaded = gltf.scene;
          setScene(loaded);
        }, undefined, () => {
          if (!cancelled) setScene(null);
        });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      if (loaded) setTimeout(() => disposeSceneObject(loaded!), 0);
    };
  }, [object.meshUrl]);
  useEffect(() => {
    if (scene) applySceneObjectMaterial(scene, opacity);
  }, [opacity, scene]);
  const matrix = useMemo(() => {
    const objectToWorld = dynamicObjectMatrixAtCursor(object, frameCursor);
    return objectToWorld ? sceneObjectToViewerMatrix(objectToWorld, anchor) : null;
  }, [anchor, frameCursor, object]);
  if (!scene || !matrix) return null;
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <primitive object={scene} />
    </group>
  );
}

// Set the shared scene-object display state without changing any geometry or
// transforms. Keeping the loaded GLTF hierarchy intact is important: exports
// often use several mesh nodes, and the old geometry path displayed only one.
function applySceneObjectMaterial(root: THREE.Object3D, opacity: number): void {
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      material.transparent = opacity < 1;
      material.opacity = opacity;
      material.side = THREE.DoubleSide;
      material.needsUpdate = true;
    }
  });
}

// GLTF primitives are not automatically disposed by React Three Fiber. The
// loader creates a fresh scene for each object, so it is safe to release every
// mesh resource when that object is replaced or unmounted.
function disposeSceneObject(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    geometries.add(mesh.geometry);
    const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of meshMaterials) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
}

// A static object reconstructed once for the clip (a chair, a step) and placed
// in the subject's coordinate space, so the two can be read against each other.
function SceneObjectMesh({
  object,
  anchor,
  opacity,
}: {
  object: SceneObject;
  anchor: DisplayAnchor | null;
  opacity: number;
}) {
  const [scene, setScene] = useState<THREE.Group | null>(null);
  const [legacyOffset, setLegacyOffset] = useState(() => new THREE.Vector3());
  const hasCanonicalTransform = object.objectToWorld !== null;
  const legacyUpAxis = hasCanonicalTransform ? null : object.upAxis;

  useEffect(() => {
    let cancelled = false;
    let loaded: THREE.Group | null = null;
    setScene(null);
    setLegacyOffset(new THREE.Vector3());
    import("three/examples/jsm/loaders/GLTFLoader.js")
      .then(({ GLTFLoader }) => {
        if (cancelled) return;
        new GLTFLoader().load(object.meshUrl, (gltf) => {
          const root = gltf.scene;
          if (cancelled) {
            disposeSceneObject(root);
            return;
          }

          const offset = new THREE.Vector3();
          if (legacyUpAxis) {
            // The legacy placement branch needs the same footprint normalization
            // as before, now calculated across every mesh in the GLTF hierarchy.
            root.updateMatrixWorld(true);
            const box = new THREE.Box3().setFromObject(root);
            if (!box.isEmpty()) {
              const mid = box.getCenter(new THREE.Vector3());
              const axis = legacyUpAxis === "X" ? "x" : legacyUpAxis === "Z" ? "z" : "y";
              const base = box.min[axis];
              offset.set(
                axis === "x" ? -base : -mid.x,
                axis === "y" ? -base : -mid.y,
                axis === "z" ? -base : -mid.z,
              );
            }
          }

          loaded = root;
          setLegacyOffset(offset);
          setScene(root);
        }, undefined, () => {
          if (!cancelled) setScene(null);
        });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      // Disposal is deferred like the body meshes: the frame loop may still be
      // drawing the previous tree when this runs.
      if (loaded) setTimeout(() => disposeSceneObject(loaded!), 0);
    };
  }, [legacyUpAxis, object.meshUrl]);

  useEffect(() => {
    if (scene) applySceneObjectMaterial(scene, opacity);
  }, [opacity, scene]);

  const objectToViewer = useMemo(
    () => object.objectToWorld ? sceneObjectToViewerMatrix(object.objectToWorld, anchor) : null,
    [anchor, object.objectToWorld],
  );

  const legacyPosition = useMemo(() => {
    const source = object.positionWorld ?? object.centreWorld;
    const point = worldToViewer(new THREE.Vector3(...source), anchor);
    // Both the fitted point and the mesh origin now sit on the floor, so the
    // object rests on the grid the bodies stand on.
    return new THREE.Vector3(point.x, point.y, 0);
  }, [anchor, object.centreWorld, object.positionWorld]);

  if (!scene) return null;
  // New artifacts provide the complete raw-mesh -> world transform. No
  // up-axis, heading, footprint, or floor heuristic is allowed on this path:
  // the parent matrix is the single placement transform applied to the GLTF.
  if (objectToViewer) {
    return (
      <group matrix={objectToViewer} matrixAutoUpdate={false}>
        <primitive object={scene} />
      </group>
    );
  }

  // Legacy artifacts lack a canonical matrix. Retain their historic placement
  // behavior so already-generated runs remain viewable.
  // Ground it the way bodies are grounded: the viewer draws its grid at z = 0
  // and places each body so its lowest point rests there, so an object placed
  // by raw world height instead lands wherever that height happens to fall —
  // below the floor, as it turned out. Only X and Y come from the solved world
  // position; Z puts the object's underside on the same grid.
  // Prefer the silhouette-fitted floor position: it is where the object must
  // stand for its outline to be the one observed, which the mask's bbox centre
  // only approximates. Fall back to that centre when no fit was written.
  // The mesh's own up axis is not the viewer's: its solved height runs along
  // upAxis, while the viewer's up is Z.
  const toViewerUp =
    object.upAxis === "Y"
      ? new THREE.Euler(Math.PI / 2, 0, 0)
      : object.upAxis === "X"
        ? new THREE.Euler(0, 0, Math.PI / 2)
        : new THREE.Euler(0, 0, 0);
  // The fitted heading turns the object about the viewer's vertical, outside
  // the axis fix so it is a heading rather than a spin about the mesh's own up.
  //
  // The extra half turn is derived, not tuned. The fit expresses the mesh in a
  // frame that sends its axes to (-x, z, y); `toViewerUp` here sends them to
  // (x, -z, y). Both are proper rotations, but they differ by exactly a half
  // turn about the shared up axis — so a heading solved in one frame arrives
  // backwards in the other. That is the 180 degrees, and it is why the object
  // could be in precisely the right place, at the right size, facing away.
  const heading = (object.yawRad ?? 0) - Math.PI / 2 + Math.PI;

  return (
    <group position={legacyPosition} rotation={[0, 0, heading]}>
      <group rotation={toViewerUp} scale={object.scale}>
        <group position={legacyOffset}>
          <primitive object={scene} />
        </group>
      </group>
    </group>
  );
}

function MeshBody({
  runId,
  frame,
  meshFrame,
  nextMeshFrame,
  meshInterpolation,
  position,
  pivot,
  meshPivot,
  quaternion,
  anchor,
  showMesh,
  showJoints,
  showBones,
  meshOpacity,
  selectedJointIndices,
  onJointPick,
  color,
  lift = 0,
  feetTrusted,
  frameCursor,
  fps,
}: {
  runId: string;
  frame: RunFrame;
  meshFrame: RunFrame;
  nextMeshFrame: RunFrame;
  meshInterpolation: number;
  position: THREE.Vector3;
  pivot: THREE.Vector3;
  meshPivot: THREE.Vector3;
  quaternion: THREE.Quaternion;
  anchor: DisplayAnchor | null;
  showMesh: boolean;
  showJoints: boolean;
  showBones: boolean;
  meshOpacity: number;
  selectedJointIndices: number[];
  onJointPick?: (jointIndex: number) => void;
  color?: string | null;
  lift?: number;
  feetTrusted?: boolean;
  frameCursor: number;
  fps: number;
}) {
  const subjectVisible = frame.subjectPresent !== false;
  const jointOffset = useMemo(() => bodyGroundOffset(frame, quaternion, anchor, pivot), [anchor, frame, pivot, quaternion]);
  // When joints briefly drop out, HOLD the last known offset instead of
  // coalescing to 0 — 0 put the PELVIS on the floor and the body visibly sank
  // for the gap, then popped back. While the lower body is cut off by the
  // picture edge (feetTrusted=false), the model's PREDICTED legs still carry
  // real information (MHR extrapolates leg pose from the torso) but tremble
  // frame to frame — so the height TRACKS the prediction slowly instead of
  // following it raw: sustained posture changes show, per-frame jitter dies.
  const lastOffsetRef = useRef<number | null>(null);
  if (jointOffset !== null) {
    if (feetTrusted !== false || lastOffsetRef.current === null) {
      lastOffsetRef.current = jointOffset;
    } else {
      lastOffsetRef.current += 0.03 * (jointOffset - lastOffsetRef.current);
    }
  }
  // Z places the lowest foot on the grid, PLUS the subject's lift above the
  // run's ground plane — zero in contact, positive during flight, so jumps
  // actually leave the floor. XY comes from ViewerScene's filtered root.
  const groundedPosition = useMemo(() => {
    const p = position.clone();
    p.z = -(lastOffsetRef.current ?? jointOffset ?? 0) + lift;
    return p;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jointOffset, lift, position, feetTrusted]);

  // The skeleton must describe the SAME instant and the SAME smoothing as the
  // mesh it is drawn inside. When the current frame's vertices are missing the
  // mesh shows a nearby cached pose, so the joints follow it there (with that
  // frame's pivot) instead of staying on the live frame and tearing away from
  // the body; and they go through the mesh's filter for the same reason.
  const filterJoints = useBodyRelativeFilter();
  const followsMesh = showMesh && Boolean(meshFrame.meshFile);
  const jointSource = followsMesh ? meshFrame : frame;
  const jointNext = followsMesh ? nextMeshFrame : frame;
  const jointPivot = followsMesh ? meshPivot : pivot;
  // Blend the SAME pair of frames by the SAME amount the mesh does. The mesh
  // interpolates towards the next frame between playhead steps; joints that
  // only ever showed the current frame therefore trailed it by up to a frame,
  // which reads as the skeleton lagging inside the body.
  const jointBlend = followsMesh ? Math.max(0, Math.min(1, meshInterpolation)) : 0;
  const joints = useMemo(() => {
    if (!subjectVisible) return [];
    const source = jointSource.jointsCam ?? [];
    if (source.length === 0) return [];
    const next = jointNext.jointsCam ?? null;
    const blendable = jointBlend > 1e-4 && next != null && next.length === source.length;
    const flat = new Float32Array(source.length * 3);
    source.forEach((joint, index) => {
      const point = camToWorld(joint, anchor).sub(jointPivot);
      const other = blendable ? next![index] : null;
      if (other) {
        point.lerp(camToWorld(other, anchor).sub(jointPivot), jointBlend);
      }
      flat[index * 3] = point.x;
      flat[index * 3 + 1] = point.y;
      flat[index * 3 + 2] = point.z;
    });
    filterJoints(flat, frameCursor, fps);
    return Array.from({ length: source.length }, (_, index) =>
      new THREE.Vector3(flat[index * 3], flat[index * 3 + 1], flat[index * 3 + 2]),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor, filterJoints, fps, frameCursor, jointBlend, jointNext, jointPivot, jointSource, subjectVisible]);
  const selected = useMemo(() => new Set(selectedJointIndices), [selectedJointIndices]);

  return (
    <group position={groundedPosition} quaternion={quaternion}>
      {subjectVisible && showMesh && meshFrame.subjectPresent !== false && meshFrame.meshFile ? (
        <MeshGeometry
          runId={runId}
          meshFile={meshFrame.meshFile}
          nextMeshFile={nextMeshFrame.subjectPresent === false ? null : nextMeshFrame.meshFile}
          meshInterpolation={meshInterpolation}
          pivot={meshPivot}
          anchor={anchor}
          meshOpacity={meshOpacity}
          color={color}
          frameCursor={frameCursor}
          fps={fps}
        />
      ) : null}
      {showBones ? <SkeletonBones joints={joints} /> : null}
      {showJoints
        ? joints.map((joint, index) => {
            const isSelected = selected.has(index);
            return (
              <mesh
                key={`${index}-${joint.x.toFixed(3)}-${joint.y.toFixed(3)}-${joint.z.toFixed(3)}`}
                position={joint}
                onClick={(event) => {
                  event.stopPropagation();
                  onJointPick?.(index);
                }}
              >
                <sphereGeometry args={[isSelected ? 0.025 : 0.015, 12, 12]} />
                <meshBasicMaterial color={isSelected ? "#22c7f2" : "#f97316"} />
              </mesh>
            );
          })
        : null}
    </group>
  );
}

// Headless component that keeps the cache warm ahead of the playhead by preloading meshes in a forward
// window (snapped to PRELOAD_WINDOW_STEP) using a bounded worker pool; renders nothing.
function MeshPreloader({
  runId,
  frames,
  frameIndex,
  radius = PLAYBACK_PRELOAD_RADIUS,
}: {
  runId: string;
  frames: RunFrame[];
  frameIndex: number;
  radius?: number;
}) {
  const meshFiles = useMemo(() => {
    const windowStart = Math.floor(Math.max(0, frameIndex) / PRELOAD_WINDOW_STEP) * PRELOAD_WINDOW_STEP;
    const start = Math.max(0, windowStart - 2);
    const end = Math.min(frames.length - 1, windowStart + radius);
    const next: string[] = [];
    for (let index = start; index <= end; index += 1) {
      const frame = frames[index];
      const meshFile = frame?.subjectPresent !== false ? frame?.meshFile : null;
      if (meshFile) {
        next.push(meshFile);
      }
    }
    return next;
  }, [frameIndex, frames, radius]);

  useEffect(() => {
    let cancelled = false;
    void loadMeshFaces(runId).catch(() => undefined);
    void (async () => {
      const missing = meshFiles.filter((meshFile) => !hasMeshVertexResource(runId, meshFile));
      let nextIndex = 0;
      async function worker(): Promise<void> {
        while (!cancelled && nextIndex < missing.length) {
          const meshFile = missing[nextIndex];
          nextIndex += 1;
          await loadMeshVertices(runId, meshFile).catch(() => undefined);
        }
      }
      await Promise.all(
        Array.from({ length: Math.min(RUN_PRELOAD_CONCURRENCY, missing.length) }, () => worker()),
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [meshFiles, runId]);

  return null;
}

// Orbit controls that frame the subject on first mount (and whenever resetKey changes), then leave the
// camera under user control; the Z-up convention and polar-angle clamps keep the view above the ground.
function SceneControls({ focus, resetKey }: { focus: THREE.Vector3; resetKey: string }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const initializedRef = useRef(false);

  // Re-arm the one-shot framing whenever the run changes.
  useEffect(() => {
    initializedRef.current = false;
  }, [resetKey]);

  // Frame the subject exactly once per run, then hand control to the user.
  useEffect(() => {
    if (!controlsRef.current || initializedRef.current) {
      return;
    }
    const target = focus.clone();
    target.z = Math.max(target.z, 0.88);
    camera.up.set(0, 0, 1);
    camera.position.set(target.x + 3.2, target.y - 6.2, target.z + 0.55);
    camera.lookAt(target);
    controlsRef.current.target.copy(target);
    controlsRef.current.minPolarAngle = THREE.MathUtils.degToRad(54);
    controlsRef.current.maxPolarAngle = THREE.MathUtils.degToRad(89);
    controlsRef.current.update();
    initializedRef.current = true;
  }, [camera, focus, resetKey]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.12}
      rotateSpeed={0.42}
      zoomSpeed={0.82}
      panSpeed={1.05}
      screenSpacePanning={false}
      minDistance={1.15}
      maxDistance={12.5}
      minPolarAngle={THREE.MathUtils.degToRad(54)}
      maxPolarAngle={THREE.MathUtils.degToRad(89)}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      }}
      touches={{
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN,
      }}
    />
  );
}

// Floor grid plus a small axes helper, laid flat in the Z-up viewer space.
function Ground() {
  const grid = useMemo(() => new THREE.GridHelper(18, 90, "#b6c4d4", "#dbe5f0"), []);
  useEffect(() => () => grid.dispose(), [grid]);
  return (
    <group>
      <primitive object={grid} rotation={[Math.PI / 2, 0, 0]} />
      <axesHelper args={[0.55]} />
    </group>
  );
}

// Core scene graph: resolves the (fractional) playhead into interpolated pose/position/upright state,
// chooses a fallback mesh frame when the exact one isn't cached, and wires up lights, ground and the body.
function ViewerScene(props: ThreeSpaceViewerProps) {
  const maxFrameIndex = Math.max(0, props.runDetail.frames.length - 1);
  const safeCursor = Math.max(0, Math.min(props.frameCursor, maxFrameIndex));
  const safeIndex = Math.max(0, Math.min(Math.floor(safeCursor), maxFrameIndex));
  const displayCursor = safeCursor;
  const baseIndex = Math.max(0, Math.min(Math.floor(displayCursor), maxFrameIndex));
  const nextIndex = Math.max(0, Math.min(baseIndex + 1, maxFrameIndex));
  const interpolation = Math.max(0, Math.min(1, displayCursor - baseIndex));
  const frameBase = props.runDetail.frames[baseIndex];
  const frameNext = props.runDetail.frames[nextIndex] ?? frameBase;
  const cachedMeshIndex = props.showMesh
    ? nearestCachedMeshFrameIndex(props.runDetail.id, props.runDetail.frames, baseIndex)
    : null;
  const meshBaseIndex = cachedMeshIndex ?? baseIndex;
  const meshFrameBase = props.runDetail.frames[meshBaseIndex] ?? frameBase;
  const meshFrameNext = meshBaseIndex === baseIndex ? frameNext : meshFrameBase;
  const meshInterpolation = meshBaseIndex === baseIndex ? interpolation : 0;
  // Joints are kept in raw camera space here. Any smoothing applied at this
  // layer would diverge from the mesh PLY vertices (which stay raw) and from
  // the foot-anchor offset computed in stabilization_xy.ts, producing both
  // mesh-skeleton misalignment and visible foot drift equal to the smoothing
  // error. If smoothing is ever needed, it must happen one level up — in
  // lib/runs.ts/stabilization_xy.ts — so the manifest, the anchor math and
  // the displayed joints all observe the same smoothed values.
  const frame = useMemo(
    () => interpolateFrame(frameBase, frameNext, interpolation),
    [frameBase, frameNext, interpolation],
  );
  const anchor = useMemo(() => displayAnchor(props.runDetail), [props.runDetail]);
  const rawRootPositions = useMemo(
    () => props.runDetail.frames.map((item) => framePosition(item, "raw", anchor)),
    [anchor, props.runDetail.frames],
  );
  // Displayed root = One-Euro-filtered RAW root. Unlike the previous fixed-alpha
  // EMA (4-frame group delay at any speed, causing foot slide during
  // acceleration because the pivot is unsmoothed), the One-Euro filter adapts
  // its cutoff to speed: heavy smoothing at rest (monocular depth jitter
  // disappears), near-zero lag during fast motion (steps and jumps track
  // truthfully).
  const smoothPositions = useMemo(() => {
    const trajectory = filteredDisplayTrajectory(
      props.runDetail.frames,
      rawRootPositions,
      anchor,
      Math.max(1, props.runDetail.fps || 30),
      props.runDetail.videoHeight,
    );
    return props.lockInPlace ? pinnedInPlace(trajectory) : trajectory;
  }, [
    anchor,
    props.lockInPlace,
    props.runDetail.fps,
    props.runDetail.frames,
    props.runDetail.videoHeight,
    rawRootPositions,
  ]);
  // Vertical placement: per-subject lift above the run's fixed ground plane —
  // zero in contact (feet snap to the grid), free during flight (jumps rise).
  const liftSeries = useMemo(
    () =>
      computeLiftSeries(
        props.runDetail.frames,
        anchor,
        Math.max(1, props.runDetail.fps || 30),
        props.runDetail.videoHeight,
      ),
    [anchor, props.runDetail.frames, props.runDetail.fps, props.runDetail.videoHeight],
  );
  // Each sibling subject gets its own anchored+filtered trajectory and its own
  // raw-root pivot: the runs' depth noise is independent (separate per-frame
  // reconstructions), so borrowing the primary's pivot would leave the
  // sibling's own wobble uncorrected. All trajectories live in the shared
  // viewer space, so relative placement stays true.
  const siblingMotion = useMemo(() => {
    const fps = Math.max(1, props.runDetail.fps || 30);
    const map = new Map<
      string,
      { lifts: number[]; positions: THREE.Vector3[]; rawRoots: THREE.Vector3[] }
    >();
    for (const sibling of props.siblings ?? []) {
      const sFrames = sibling.runDetail.frames;
      const sHeight = sibling.runDetail.videoHeight;
      const rawRoots = sFrames.map((item) => framePosition(item, "raw", anchor));
      const positions = filteredDisplayTrajectory(sFrames, rawRoots, anchor, fps, sHeight);
      map.set(sibling.runDetail.id, {
        lifts: computeLiftSeries(sFrames, anchor, fps, sHeight),
        positions: props.lockInPlace ? pinnedInPlace(positions) : positions,
        rawRoots,
      });
    }
    return map;
  }, [anchor, props.lockInPlace, props.runDetail.fps, props.siblings]);
  const displayPosition = smoothPositions[nextIndex]
    ? lerpVector(smoothPositions[baseIndex], smoothPositions[nextIndex], interpolation)
    : smoothPositions[baseIndex] ?? new THREE.Vector3();
  const rootPivot = rawRootPositions[nextIndex]
    ? lerpVector(rawRootPositions[baseIndex], rawRootPositions[nextIndex], interpolation)
    : rawRootPositions[baseIndex] ?? displayPosition;
  const bodyFocus = bodyFocusPosition(frame, anchor) ?? rootPivot;
  const hasMeaningfulRoot = frameHasMeaningfulRoot(frameBase);
  // Mesh vertices and joints are both exported in camera space. Use the raw pelvis
  // root as the local pivot, then apply the stabilized/smoothed root as the group
  // position. This keeps the mesh locked to the skeleton and avoids offsets from
  // using a broad body-average pivot.
  const pivot = hasMeaningfulRoot ? rootPivot : bodyFocus;
  const position = hasMeaningfulRoot ? displayPosition : bodyFocus;
  const focus = hasMeaningfulRoot ? position.clone().add(bodyFocus.clone().sub(rootPivot)) : bodyFocus;
  // When the current frame's PLY isn't cached, the mesh falls back to a nearby
  // cached frame (meshBaseIndex). That frame's vertices bake in their OWN root
  // translation, so the mesh must subtract THAT frame's raw root — not the
  // current frame's pivot. Otherwise the mesh lands at
  // (rawRoot[meshFrame] - rawRoot[frame]) away from the skeleton. The joints
  // always use the current frame, so without this they stay glued to the video
  // while the mesh drifts — worst after repeated turn-arounds thrash the cache.
  const meshPivot = hasMeaningfulRoot && meshBaseIndex !== baseIndex
    ? (rawRootPositions[meshBaseIndex] ?? pivot)
    : pivot;
  const uprightQuaternions = useMemo(
    () =>
      props.uprightMode
        ? stableUprightQuaternions(
            props.runDetail.frames,
            anchor,
            Math.max(1, props.runDetail.fps || 30),
          )
        : [],
    [anchor, props.runDetail.fps, props.runDetail.frames, props.uprightMode],
  );
  // Per-run sole-pitch calibration: a constant counter-rotation about the hip
  // axis so planted feet sit FLAT instead of "on their heels" (systematic
  // monocular backward-lean bias that the trunk correction cannot fix).
  const soleFixAngle = useMemo(
    () =>
      props.uprightMode
        ? computeSolePitchFix(
            props.runDetail.frames,
            anchor,
            uprightQuaternions,
            props.runDetail.videoHeight,
          )
        : 0,
    [anchor, props.runDetail.frames, props.runDetail.videoHeight, props.uprightMode, uprightQuaternions],
  );
  const uprightBase = props.uprightMode
    ? (uprightQuaternions[baseIndex] ?? new THREE.Quaternion())
        .clone()
        .slerp(uprightQuaternions[nextIndex] ?? uprightQuaternions[baseIndex] ?? new THREE.Quaternion(), interpolation)
    : new THREE.Quaternion();
  const soleAxis = soleFixAngle !== 0 ? hipAxisAfter(frameBase, anchor, uprightBase) : null;
  const upright = soleAxis
    ? new THREE.Quaternion().setFromAxisAngle(soleAxis, soleFixAngle).multiply(uprightBase)
    : uprightBase;

  return (
    <>
      <color attach="background" args={["#eaf0f8"]} />
      <ambientLight intensity={0.72} />
      <directionalLight position={[3, -4, 5]} intensity={1.8} castShadow />
      <Ground />
      {props.showSceneObjects !== false
        ? (props.runDetail.sceneObjects ?? []).map((object) => (
            <SceneObjectMesh
              key={object.name}
              object={object}
              anchor={anchor}
              opacity={props.sceneObjectOpacity ?? 1}
            />
          ))
        : null}
      {props.showSceneObjects !== false
          ? (props.runDetail.dynamicObjects ?? []).map((object) => (
            <DynamicObjectMesh
              key={`dynamic-${object.name}`}
              object={object}
              anchor={anchor}
              frameCursor={displayCursor}
              opacity={props.sceneObjectOpacity ?? 1}
            />
          ))
        : null}
      {props.showMesh && props.showPrimary !== false ? (
        <MeshPreloader runId={props.runDetail.id} frames={props.runDetail.frames} frameIndex={safeIndex} />
      ) : null}
      {props.showPrimary === false ? null : (
      <MeshBody
        runId={props.runDetail.id}
        frame={frame}
        meshFrame={meshFrameBase}
        nextMeshFrame={meshFrameNext}
        meshInterpolation={meshInterpolation}
        position={position}
        pivot={pivot}
        meshPivot={meshPivot}
        quaternion={upright}
        anchor={anchor}
        showMesh={props.showMesh}
        showJoints={props.showJoints}
        showBones={props.showBones}
        meshOpacity={props.meshOpacity}
        selectedJointIndices={props.selectedJointIndices}
        onJointPick={props.onJointPick}
        color={props.subjectColor}
        lift={lerpNumber(liftSeries[baseIndex] ?? 0, liftSeries[nextIndex] ?? 0, interpolation)}
        feetTrusted={feetVisible(frameBase, props.runDetail.videoHeight)}
        frameCursor={displayCursor}
        fps={Math.max(1, props.runDetail.fps || 30)}
      />
      )}
      {(props.siblings ?? []).map((sibling) => {
        // Sibling subjects of the same selection share the source video's
        // camera space, so every subject's anchored trajectory lives in the
        // same viewer frame and relative placement stays true. Each sibling
        // is placed by its OWN anchored+filtered trajectory and its OWN
        // raw-root pivot (the runs' reconstruction noise is independent), and
        // grounds its own feet.
        const siblingFrames = sibling.runDetail.frames;
        const motion = siblingMotion.get(sibling.runDetail.id);
        if (siblingFrames.length === 0 || !motion) {
          return null;
        }
        const sBase = Math.min(baseIndex, siblingFrames.length - 1);
        const sNext = Math.min(nextIndex, siblingFrames.length - 1);
        const sFrameBase = siblingFrames[sBase];
        const sFrameNext = siblingFrames[sNext] ?? sFrameBase;
        const sCachedMeshIndex = props.showMesh
          ? nearestCachedMeshFrameIndex(sibling.runDetail.id, siblingFrames, sBase)
          : null;
        const sMeshIndex = sCachedMeshIndex ?? sBase;
        const sMeshFrame = siblingFrames[sMeshIndex] ?? sFrameBase;
        const sMeshNext = sMeshIndex === sBase ? sFrameNext : sMeshFrame;
        const sMeshInterpolation = sMeshIndex === sBase ? interpolation : 0;
        const sPosition = motion.positions[sNext]
          ? lerpVector(motion.positions[sBase], motion.positions[sNext], interpolation)
          : motion.positions[sBase] ?? position;
        const sPivot = motion.rawRoots[sNext]
          ? lerpVector(motion.rawRoots[sBase], motion.rawRoots[sNext], interpolation)
          : motion.rawRoots[sBase] ?? sPosition;
        return (
          <group key={sibling.runDetail.id}>
            {props.showMesh ? (
              <MeshPreloader
                runId={sibling.runDetail.id}
                frames={siblingFrames}
                frameIndex={sBase}
              />
            ) : null}
            <MeshBody
              runId={sibling.runDetail.id}
              frame={interpolateFrame(sFrameBase, sFrameNext, interpolation)}
              meshFrame={sMeshFrame}
              nextMeshFrame={sMeshNext}
              meshInterpolation={sMeshInterpolation}
              position={sPosition}
              pivot={sPivot}
              // A fallback mesh frame bakes ITS OWN root translation; subtract
              // that frame's raw root so the sibling's mesh stays on its
              // skeleton instead of drifting by rawRoot[fallback] - rawRoot[current].
              meshPivot={
                sMeshIndex === sBase ? sPivot : motion.rawRoots[sMeshIndex] ?? sPivot
              }
              quaternion={upright}
              anchor={anchor}
              showMesh={props.showMesh}
              showJoints={false}
              showBones={props.showBones}
              meshOpacity={props.meshOpacity}
              selectedJointIndices={[]}
              color={sibling.color}
              lift={lerpNumber(
                motion.lifts[sBase] ?? 0,
                motion.lifts[sNext] ?? 0,
                interpolation,
              )}
              feetTrusted={feetVisible(sFrameBase, sibling.runDetail.videoHeight)}
              frameCursor={displayCursor}
              fps={Math.max(1, props.runDetail.fps || 30)}
            />
          </group>
        );
      })}
      <SceneControls focus={focus} resetKey={props.runDetail.id} />
    </>
  );
}

// Public entry point: hosts the R3F Canvas (Z-up camera) and mounts the scene, remounting per run via key.
export function ThreeSpaceViewer(props: ThreeSpaceViewerProps) {
  return (
    <div className="space-viewer-shell">
      <Canvas
        key={props.runDetail.id}
        shadows
        camera={{ position: [3.2, -6.2, 1.45], fov: 43, near: 0.01, far: 200, up: [0, 0, 1] }}
      >
        <ViewerScene {...props} />
      </Canvas>
    </div>
  );
}
