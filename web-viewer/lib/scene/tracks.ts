/** Typed views over the scene binaries, and per-person sampling in time. */

import { STATE, forwardKinematics, slerp } from "./skeleton";
import type { PersonEntry, SceneFile, Section } from "./types";

export const JOINTS = 127;
export const KEYPOINTS = 70;

type Typed = Uint8Array | Uint16Array | Int16Array | Int32Array | Float32Array;

export function view(buffer: ArrayBuffer, section: Section): Typed {
  const count = section.shape.reduce((a, b) => a * b, 1);
  switch (section.type) {
    case "uint8":
      return new Uint8Array(buffer, section.offset, count);
    case "uint16":
      return new Uint16Array(buffer, section.offset, count);
    case "int16":
      return new Int16Array(buffer, section.offset, count);
    case "int32":
      return new Int32Array(buffer, section.offset, count);
    case "float32":
      return new Float32Array(buffer, section.offset, count);
  }
}

export type MeshData = {
  faces: Uint16Array;
  skinIndex: Uint8Array;
  skinWeight: Float32Array;
  inverseBind: Float32Array;
  parents: Int16Array;
  vertices: number;
};

export function meshData(scene: SceneFile, buffer: ArrayBuffer): MeshData {
  const layout = scene.mesh.layout;
  return {
    faces: view(buffer, layout.faces) as Uint16Array,
    skinIndex: view(buffer, layout.skin_index) as Uint8Array,
    skinWeight: view(buffer, layout.skin_weight) as Float32Array,
    inverseBind: view(buffer, layout.inverse_bind) as Float32Array,
    parents: view(buffer, layout.parents) as Int16Array,
    vertices: scene.mesh.vertices,
  };
}

export type Pose = {
  /** World joint states, `JOINTS * 8`. */
  joints: Float64Array;
  /** World keypoints, `KEYPOINTS * 3` (metres). */
  keypoints: Float32Array;
  measured: boolean;
  contacts: [boolean, boolean];
};

/** One person's animation: which frames they appear in and their pose at any time. */
export class PersonTrack {
  readonly entry: PersonEntry;
  readonly frames: Int32Array;
  readonly restVertices: Float32Array;
  private readonly pelvis: Float32Array;
  private readonly localQ: Int16Array;
  private readonly localT: Int16Array;
  private readonly keypoints: Int16Array;
  private readonly flags: Uint8Array;
  private readonly jointScales: Float32Array;
  private readonly local = new Float64Array(JOINTS * STATE);
  private readonly scratch = new Float64Array(STATE * 2);

  constructor(
    entry: PersonEntry,
    buffer: ArrayBuffer,
    private readonly scene: SceneFile,
    private readonly parents: Int16Array,
  ) {
    const l = entry.layout;
    this.entry = entry;
    this.restVertices = view(buffer, l.rest_vertices) as Float32Array;
    this.jointScales = view(buffer, l.joint_scales) as Float32Array;
    this.frames = view(buffer, l.frames) as Int32Array;
    this.pelvis = view(buffer, l.pelvis) as Float32Array;
    this.localQ = view(buffer, l.local_q) as Int16Array;
    this.localT = view(buffer, l.local_t) as Int16Array;
    this.keypoints = view(buffer, l.keypoints) as Int16Array;
    this.flags = view(buffer, l.flags) as Uint8Array;
  }

  /** Index into this person's arrays for a video frame, or -1 when absent. */
  indexOf(frame: number): number {
    const f = this.frames;
    let lo = 0, hi = f.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (f[mid] === frame) return mid;
      if (f[mid] < frame) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  /** Whether the person is on screen at a (fractional) frame. */
  visibleAt(frame: number): boolean {
    return this.indexOf(Math.floor(frame)) >= 0;
  }

  /** Pose at a fractional frame, interpolated between the two nearest frames. */
  sample(frame: number, out: Pose): boolean {
    const i = this.indexOf(Math.floor(frame));
    if (i < 0) return false;
    const next = i + 1 < this.frames.length && this.frames[i + 1] === this.frames[i] + 1 ? i + 1 : i;
    const t = next === i ? 0 : frame - Math.floor(frame);
    const { quat_scale: qs, local_t_unit: tu, keypoint_unit: ku, pelvis_joint: pj } = this.scene.encoding;

    const local = this.local;
    for (let j = 0; j < JOINTS; j++) {
      const o = j * STATE, q4 = j * 4, t3 = j * 3;
      const qa = (i * JOINTS) * 4 + q4, qb = (next * JOINTS) * 4 + q4;
      const s = this.scratch;
      s[0] = this.localQ[qa] / qs; s[1] = this.localQ[qa + 1] / qs; s[2] = this.localQ[qa + 2] / qs; s[3] = this.localQ[qa + 3] / qs;
      s[4] = this.localQ[qb] / qs; s[5] = this.localQ[qb + 1] / qs; s[6] = this.localQ[qb + 2] / qs; s[7] = this.localQ[qb + 3] / qs;
      slerp(s, 0, s, 4, t, local, o + 3);
      const ta = (i * JOINTS) * 3 + t3, tb = (next * JOINTS) * 3 + t3;
      for (let k = 0; k < 3; k++) local[o + k] = ((1 - t) * this.localT[ta + k] + t * this.localT[tb + k]) * tu;
      local[o + 7] = this.jointScales[j];
    }

    const pa = i * STATE, pb = next * STATE;
    const pelvis = this.scratch;
    for (let k = 0; k < 3; k++) pelvis[k] = (1 - t) * this.pelvis[pa + k] + t * this.pelvis[pb + k];
    slerp(this.pelvis, pa + 3, this.pelvis, pb + 3, t, pelvis, 3);
    pelvis[7] = (1 - t) * this.pelvis[pa + 7] + t * this.pelvis[pb + 7];
    forwardKinematics(this.parents, pj, pelvis, local, out.joints);

    const ka = i * KEYPOINTS * 3, kb = next * KEYPOINTS * 3;
    for (let k = 0; k < KEYPOINTS * 3; k++) {
      out.keypoints[k] = ((1 - t) * this.keypoints[ka + k] + t * this.keypoints[kb + k]) * ku;
    }
    const flags = this.flags[t < 0.5 ? i : next];
    const f = this.scene.encoding.flags;
    out.measured = (flags & f.measured) !== 0;
    out.contacts = [(flags & f.left_contact) !== 0, (flags & f.right_contact) !== 0];
    return true;
  }

  /** Pelvis position (world, metres) at an index into this person's frames. */
  pelvisAt(index: number): [number, number, number] {
    const o = index * STATE;
    return [this.pelvis[o], this.pelvis[o + 1], this.pelvis[o + 2]];
  }
}

export function newPose(): Pose {
  return { joints: new Float64Array(JOINTS * STATE), keypoints: new Float32Array(KEYPOINTS * 3), measured: false, contacts: [false, false] };
}
