"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import { apiFetch } from "../lib/api-client";
import type { RunDetail, RunFrame } from "../lib/types";

type DisplayAnchor = {
  centerX: number;
  centerZ: number;
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
const MESH_FALLBACK_RADIUS = 360;
const PRELOAD_WINDOW_STEP = 30;
const DISPLAY_ROOT_SMOOTHING_ALPHA = 0.8;
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

// Extract the world-space recentering anchor (in camera coords) from the run, or null if not finite.
function displayAnchor(runDetail: RunDetail): DisplayAnchor | null {
  const anchor = runDetail.spaceView?.world_anchor ?? null;
  if (!anchor || !Number.isFinite(anchor.center_x) || !Number.isFinite(anchor.center_z)) {
    return null;
  }
  return { centerX: anchor.center_x, centerZ: anchor.center_z };
}

// Shift a world-space point so the capture's anchor sits at the origin (mutates and returns point).
function centerWorld(point: THREE.Vector3, anchor: DisplayAnchor | null): THREE.Vector3 {
  if (!anchor) {
    return point;
  }
  // Anchor is stored in camera coordinates. After CAM_TO_WORLD, horizontal center is [-centerZ, centerX].
  point.x += anchor.centerZ;
  point.y -= anchor.centerX;
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
function bodyUprightQuaternion(frame: RunFrame | null, anchor: DisplayAnchor | null): THREE.Quaternion {
  if (frame?.subjectPresent === false) {
    return new THREE.Quaternion();
  }
  const joints = frame?.jointsCam ?? null;
  if (!joints || joints.length < 15) {
    return new THREE.Quaternion();
  }
  const leftHip = joints[9] ? camToWorld(joints[9], anchor) : null;
  const rightHip = joints[10] ? camToWorld(joints[10], anchor) : null;
  const leftShoulder = joints[5] ? camToWorld(joints[5], anchor) : null;
  const rightShoulder = joints[6] ? camToWorld(joints[6], anchor) : null;
  if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) {
    return new THREE.Quaternion();
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
  if (bodyUp.length() < 1e-5) {
    return new THREE.Quaternion();
  }
  bodyUp.normalize();
  const worldUp = new THREE.Vector3(0, 0, 1);
  const tilt = bodyUp.angleTo(worldUp);
  if (!Number.isFinite(tilt) || tilt < THREE.MathUtils.degToRad(2)) {
    return new THREE.Quaternion();
  }
  return new THREE.Quaternion().setFromUnitVectors(bodyUp, worldUp);
}

// Angle (radians) of the shortest rotation between two quaternions.
function quaternionAngle(a: THREE.Quaternion, b: THREE.Quaternion): number {
  return 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
}

// Per-frame upright rotations, temporally smoothed: reject implausibly large raw tilts and slerp toward
// each candidate (more slowly across big jumps) so the standing-up correction doesn't jitter during playback.
function stableUprightQuaternions(frames: RunFrame[], anchor: DisplayAnchor | null): THREE.Quaternion[] {
  const identity = new THREE.Quaternion();
  const maxRawTilt = THREE.MathUtils.degToRad(70);
  const maxJump = THREE.MathUtils.degToRad(28);
  const out: THREE.Quaternion[] = [];
  for (const frame of frames) {
    const raw = bodyUprightQuaternion(frame, anchor);
    const rawTilt = quaternionAngle(identity, raw);
    const candidate = Number.isFinite(rawTilt) && rawTilt <= maxRawTilt ? raw : (out.at(-1) ?? identity);
    if (out.length === 0) {
      out.push(candidate.clone());
      continue;
    }
    const previous = out[out.length - 1];
    const jump = quaternionAngle(previous, candidate);
    const alpha = jump > maxJump ? 0.22 : 0.48;
    out.push(previous.clone().slerp(candidate, alpha).normalize());
  }
  return out;
}

// Lowest foot-joint height (after pivot subtraction and the upright rotation) so the body can be dropped
// onto the grid; null when no foot joints are available.
function footJointGroundOffset(
  frame: RunFrame | null,
  quaternion: THREE.Quaternion,
  anchor: DisplayAnchor | null,
  pivot: THREE.Vector3,
): number | null {
  const joints = frame?.jointsCam ?? null;
  if (!joints || joints.length < 15) {
    return null;
  }
  const footIndices = [13, 14, 15, 16, 17, 18, 19, 20];
  const zValues: number[] = [];
  for (const index of footIndices) {
    const joint = joints[index];
    if (!joint) {
      continue;
    }
    const p = camToWorld(joint, anchor).sub(pivot).applyQuaternion(quaternion);
    if (Number.isFinite(p.z)) {
      zValues.push(p.z);
    }
  }
  return zValues.length > 0 ? Math.min(...zValues) : null;
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
}: {
  runId: string;
  meshFile: string;
  nextMeshFile: string | null;
  meshInterpolation: number;
  pivot: THREE.Vector3;
  anchor: DisplayAnchor | null;
  meshOpacity: number;
  color?: string | null;
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
      next.translate(anchor.centerZ, -anchor.centerX, 0);
    }
    next.applyMatrix4(DISPLAY_YAW_CLOCKWISE_90);
    next.translate(-pivot.x, -pivot.y, -pivot.z);
    next.computeBoundingSphere();
    return next;
  }, [anchor, faces, meshInterpolation, nextVertices, pivot, vertices]);
  useEffect(() => () => geometry?.dispose(), [geometry]);

  if (!geometry) {
    return null;
  }

  return (
    <mesh geometry={geometry} castShadow receiveShadow>
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
}) {
  const subjectVisible = frame.subjectPresent !== false;
  const jointOffset = useMemo(() => footJointGroundOffset(frame, quaternion, anchor, pivot), [anchor, frame, pivot, quaternion]);
  // Z is driven by the lowest foot joint so the lowest contact point always
  // sits exactly on the grid (no foot traversal). XY comes from ViewerScene,
  // which already locks the body to the grid origin to remove drift.
  const groundedPosition = useMemo(() => {
    const p = position.clone();
    p.z = -(jointOffset ?? 0);
    return p;
  }, [jointOffset, position]);

  const joints = useMemo(
    () => subjectVisible ? (frame.jointsCam ?? []).map((joint) => camToWorld(joint, anchor).sub(pivot)) : [],
    [anchor, frame.jointsCam, pivot, subjectVisible],
  );
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
  const displayRootPositions = useMemo(
    () => props.runDetail.frames.map((item) => framePosition(item, "stabilized", anchor)),
    [anchor, props.runDetail.frames],
  );
  const rawRootPositions = useMemo(
    () => props.runDetail.frames.map((item) => framePosition(item, "raw", anchor)),
    [anchor, props.runDetail.frames],
  );
  const smoothPositions = useMemo(() => {
    const alpha = DISPLAY_ROOT_SMOOTHING_ALPHA;
    if (displayRootPositions.length === 0 || alpha <= 0) {
      return displayRootPositions;
    }
    const out: THREE.Vector3[] = [];
    for (let index = 0; index < displayRootPositions.length; index += 1) {
      if (index === 0) {
        out.push(displayRootPositions[index].clone());
      } else {
        out.push(out[index - 1].clone().lerp(displayRootPositions[index], 1 - alpha));
      }
    }
    return out;
  }, [displayRootPositions]);
  // Drift suppression is now handled physically at load time by the
  // foot-anchoring pass in stabilization_xy.ts (the support foot is locked
  // in world space, the pelvis is derived from it, anchors switch at
  // heel-strike). So we can trust smoothPositions here and use the EMA
  // output directly: turning-in-place stays planted because the anchor
  // doesn't drift, walking actually translates because every step shifts
  // the anchor by the natural step length.
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
    () => (props.uprightMode ? stableUprightQuaternions(props.runDetail.frames, anchor) : []),
    [anchor, props.runDetail.frames, props.uprightMode],
  );
  const upright = props.uprightMode
    ? (uprightQuaternions[baseIndex] ?? new THREE.Quaternion())
        .clone()
        .slerp(uprightQuaternions[nextIndex] ?? uprightQuaternions[baseIndex] ?? new THREE.Quaternion(), interpolation)
    : new THREE.Quaternion();

  return (
    <>
      <color attach="background" args={["#eaf0f8"]} />
      <ambientLight intensity={0.72} />
      <directionalLight position={[3, -4, 5]} intensity={1.8} castShadow />
      <Ground />
      {props.showMesh ? <MeshPreloader runId={props.runDetail.id} frames={props.runDetail.frames} frameIndex={safeIndex} /> : null}
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
      />
      {(props.siblings ?? []).map((sibling) => {
        // Sibling subjects of the same selection share the source video's
        // camera space, so rendering them with the PRIMARY run's anchor,
        // pivot, position and upright rotation preserves the true relative
        // placement of the people: relative offset = upright * (siblingCam -
        // primaryCam). Each sibling grounds its own feet (MeshBody re-derives
        // Z from its own frame's foot joints).
        const siblingFrames = sibling.runDetail.frames;
        if (siblingFrames.length === 0) {
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
              position={position}
              pivot={pivot}
              meshPivot={pivot}
              quaternion={upright}
              anchor={anchor}
              showMesh={props.showMesh}
              showJoints={false}
              showBones={props.showBones}
              meshOpacity={props.meshOpacity}
              selectedJointIndices={[]}
              color={sibling.color}
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
