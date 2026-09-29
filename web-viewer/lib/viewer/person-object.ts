/**
 * The three.js objects of one person: a skinned body mesh driven directly by
 * the decoded skeleton, a skeleton overlay, a floor ring and a motion trail.
 *
 * Bones are not part of the scene graph: their world matrices are written
 * from the forward kinematics each frame, and the mesh uses a detached bind
 * (identity bind matrix), so skinning maps the MHR rest mesh (centimetres,
 * model space) straight to world metres.
 */

import * as THREE from "three";

import { JOINTS, KEYPOINTS, newPose, type MeshData, type PersonTrack, type Pose } from "../scene/tracks";

const FADE_FRAMES = 6;
const TRAIL_SECONDS = 2.5;

export type DisplayOptions = {
  mesh: boolean;
  skeleton: boolean;
  trails: boolean;
  markers: boolean;
  dimmed: boolean;
  selected: boolean;
  opacity: number;
};

const scratchPosition = new THREE.Vector3();
const scratchQuaternion = new THREE.Quaternion();
const scratchScale = new THREE.Vector3();

function stateMatrix(s: ArrayLike<number>, o: number, out: THREE.Matrix4): THREE.Matrix4 {
  scratchPosition.set(s[o], s[o + 1], s[o + 2]);
  scratchQuaternion.set(s[o + 3], s[o + 4], s[o + 5], s[o + 6]);
  scratchScale.setScalar(s[o + 7]);
  return out.compose(scratchPosition, scratchQuaternion, scratchScale);
}

/** Shared buffers of the body topology (one copy for every person). */
export class SharedBody {
  readonly index: THREE.BufferAttribute;
  readonly skinIndex: THREE.BufferAttribute;
  readonly skinWeight: THREE.BufferAttribute;
  readonly boneInverses: THREE.Matrix4[];

  constructor(mesh: MeshData) {
    this.index = new THREE.BufferAttribute(mesh.faces, 1);
    this.skinIndex = new THREE.BufferAttribute(mesh.skinIndex, 4);
    this.skinWeight = new THREE.BufferAttribute(mesh.skinWeight, 4);
    this.boneInverses = Array.from({ length: JOINTS }, (_, j) => stateMatrix(mesh.inverseBind, j * 8, new THREE.Matrix4()));
  }
}

export class PersonObject {
  readonly track: PersonTrack;
  readonly root = new THREE.Group();
  readonly body: THREE.SkinnedMesh;
  readonly labelAnchor = new THREE.Object3D();
  readonly pose: Pose = newPose();
  private readonly bones: THREE.Bone[];
  private readonly material: THREE.MeshStandardMaterial;
  private readonly skeletonLines: THREE.LineSegments;
  private readonly ring: THREE.Mesh;
  private readonly marker: THREE.Mesh;
  private readonly trail: THREE.Line;
  private readonly trailPositions: Float32Array;
  private readonly bonePairs: [number, number][];
  private readonly colour: THREE.Color;
  private visible = false;
  private lastTrailFrame = -1;
  private readonly lastFrame: number;

  constructor(track: PersonTrack, shared: SharedBody, bones: [number, number][], fps: number, frames: number) {
    this.lastFrame = frames - 1;
    this.track = track;
    this.colour = new THREE.Color(track.entry.color);
    this.bonePairs = bones;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(track.restVertices, 3));
    geometry.setIndex(shared.index);
    geometry.computeVertexNormals();
    geometry.setAttribute("skinIndex", shared.skinIndex);
    geometry.setAttribute("skinWeight", shared.skinWeight);

    this.material = new THREE.MeshStandardMaterial({
      color: this.colour,
      roughness: 0.62,
      metalness: 0.04,
      alphaHash: true,
    });
    this.body = new THREE.SkinnedMesh(geometry, this.material);
    this.bones = Array.from({ length: JOINTS }, () => {
      const bone = new THREE.Bone();
      bone.matrixAutoUpdate = false;
      bone.matrixWorldAutoUpdate = false;
      return bone;
    });
    this.body.bindMode = THREE.DetachedBindMode;
    this.body.bind(new THREE.Skeleton(this.bones, shared.boneInverses), new THREE.Matrix4());
    this.body.frustumCulled = false;
    this.body.castShadow = true;
    this.body.receiveShadow = false;
    this.body.userData.personId = track.entry.id;

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bones.length * 6), 3));
    this.skeletonLines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: this.colour.clone().multiplyScalar(0.55), depthTest: false, transparent: true }),
    );
    this.skeletonLines.frustumCulled = false;
    this.skeletonLines.renderOrder = 10;

    this.ring = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.46, 48),
      new THREE.MeshBasicMaterial({ color: this.colour, transparent: true, opacity: 0.9, depthWrite: false }),
    );
    this.ring.position.z = 0.005;

    this.marker = new THREE.Mesh(
      new THREE.CircleGeometry(0.42, 40),
      new THREE.MeshBasicMaterial({ color: this.colour, transparent: true, opacity: 0.85, depthWrite: false }),
    );
    this.marker.position.z = 0.004;

    const trailCount = Math.max(2, Math.round(TRAIL_SECONDS * fps));
    this.trailPositions = new Float32Array(trailCount * 3);
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute("position", new THREE.BufferAttribute(this.trailPositions, 3));
    this.trail = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ color: this.colour, transparent: true, opacity: 0.75 }));
    this.trail.frustumCulled = false;

    this.root.add(this.body, this.skeletonLines, this.ring, this.marker, this.trail, this.labelAnchor);
  }

  get isVisible(): boolean {
    return this.visible;
  }

  /** Pelvis position in world coordinates (valid when visible). */
  pelvis(out: THREE.Vector3): THREE.Vector3 {
    const j = this.pose.joints;
    return out.set(j[8], j[9], j[10]);
  }

  /** Opacity near an appearance or disappearance; the clip's own ends do not fade. */
  private fade(frame: number, lastFrame: number): number {
    for (const [start, end] of this.track.entry.segments) {
      if (frame >= start && frame <= end + 0.999) {
        const fromStart = start === 0 ? Infinity : frame - start;
        const toEnd = end >= lastFrame ? Infinity : end + 1 - frame;
        return Math.max(0.15, Math.min(1, Math.min(fromStart, toEnd) / FADE_FRAMES + 0.15));
      }
    }
    return 0;
  }

  update(frame: number, options: DisplayOptions): void {
    const present = this.track.sample(frame, this.pose);
    this.visible = present;
    this.root.visible = present;
    if (!present) return;

    const joints = this.pose.joints;
    for (let j = 0; j < JOINTS; j++) stateMatrix(joints, j * 8, this.bones[j].matrixWorld);

    const alpha = this.fade(frame, this.lastFrame) * (options.dimmed ? 0.25 : 1) * options.opacity;
    this.body.visible = options.mesh;
    this.material.opacity = alpha;
    this.material.emissive.copy(this.colour).multiplyScalar(options.selected ? 0.18 : 0);

    this.skeletonLines.visible = options.skeleton;
    if (options.skeleton) {
      const kp = this.pose.keypoints;
      const position = this.skeletonLines.geometry.getAttribute("position") as THREE.BufferAttribute;
      const array = position.array as Float32Array;
      this.bonePairs.forEach(([a, b], i) => {
        array.set([kp[a * 3], kp[a * 3 + 1], kp[a * 3 + 2], kp[b * 3], kp[b * 3 + 1], kp[b * 3 + 2]], i * 6);
      });
      position.needsUpdate = true;
      (this.skeletonLines.material as THREE.LineBasicMaterial).opacity = options.dimmed ? 0.3 : 1;
    }

    this.ring.visible = options.selected;
    this.ring.position.x = joints[8];
    this.ring.position.y = joints[9];
    this.marker.visible = options.markers;
    this.marker.position.x = joints[8];
    this.marker.position.y = joints[9];
    (this.marker.material as THREE.MeshBasicMaterial).opacity = options.dimmed ? 0.2 : 0.85;

    const kp = this.pose.keypoints;
    const head = Math.max(kp[2], kp[3 * 3 + 2], kp[4 * 3 + 2]);
    this.labelAnchor.position.set(kp[0], kp[1], head + 0.28);

    this.trail.visible = options.trails;
    if (options.trails) this.updateTrail(Math.floor(frame));
  }

  private updateTrail(frame: number): void {
    if (frame === this.lastTrailFrame) return;
    this.lastTrailFrame = frame;
    const count = this.trailPositions.length / 3;
    const now = this.track.indexOf(frame);
    let written = 0;
    for (let k = 0; k < count && now - k >= 0; k++) {
      const index = now - k;
      if (this.track.frames[index] !== frame - k) break; // stop at a gap
      const [x, y] = this.track.pelvisAt(index);
      this.trailPositions.set([x, y, 0.02], written * 3);
      written += 1;
    }
    const geometry = this.trail.geometry;
    geometry.setDrawRange(0, written);
    (geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.body.geometry.dispose();
    this.material.dispose();
    this.skeletonLines.geometry.dispose();
    (this.skeletonLines.material as THREE.Material).dispose();
    this.ring.geometry.dispose();
    (this.ring.material as THREE.Material).dispose();
    this.marker.geometry.dispose();
    (this.marker.material as THREE.Material).dispose();
    this.trail.geometry.dispose();
    (this.trail.material as THREE.Material).dispose();
  }
}

export { KEYPOINTS };
