/**
 * Joint states and forward kinematics, mirroring `kinesia/scene/quat.py`.
 *
 * A state is 8 numbers `[tx, ty, tz, qx, qy, qz, qw, s]`: a translation, a unit
 * quaternion and a uniform scale. `compose(a, b)` applies `b` first, then `a`.
 */

export const STATE = 8;

export function rotate(q: ArrayLike<number>, qo: number, v: [number, number, number]): [number, number, number] {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  // t = 2 * cross(u, v); v' = v + w * t + cross(u, t)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

/** out[o..o+8] = a ∘ b, where a and b are states at the given offsets. */
export function compose(
  a: ArrayLike<number>, ao: number,
  b: ArrayLike<number>, bo: number,
  out: Float32Array | Float64Array, o: number,
): void {
  const sa = a[ao + 7];
  const [rx, ry, rz] = rotate(a, ao + 3, [sa * b[bo], sa * b[bo + 1], sa * b[bo + 2]]);
  const ax = a[ao + 3], ay = a[ao + 4], az = a[ao + 5], aw = a[ao + 6];
  const bx = b[bo + 3], by = b[bo + 4], bz = b[bo + 5], bw = b[bo + 6];
  out[o] = a[ao] + rx;
  out[o + 1] = a[ao + 1] + ry;
  out[o + 2] = a[ao + 2] + rz;
  out[o + 3] = aw * bx + ax * bw + ay * bz - az * by;
  out[o + 4] = aw * by - ax * bz + ay * bw + az * bx;
  out[o + 5] = aw * bz + ax * by - ay * bx + az * bw;
  out[o + 6] = aw * bw - ax * bx - ay * by - az * bz;
  out[o + 7] = sa * b[bo + 7];
}

/** Spherical interpolation of quaternions `a` and `b` into `out`. */
export function slerp(
  a: ArrayLike<number>, ao: number,
  b: ArrayLike<number>, bo: number,
  t: number,
  out: Float32Array | Float64Array, o: number,
): void {
  let bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  let dot = a[ao] * bx + a[ao + 1] * by + a[ao + 2] * bz + a[ao + 3] * bw;
  if (dot < 0) {
    bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot;
  }
  let wa = 1 - t, wb = t;
  if (dot < 0.9995) {
    const theta = Math.acos(Math.min(1, dot));
    const sin = Math.sin(theta);
    wa = Math.sin((1 - t) * theta) / sin;
    wb = Math.sin(t * theta) / sin;
  }
  const x = wa * a[ao] + wb * bx, y = wa * a[ao + 1] + wb * by;
  const z = wa * a[ao + 2] + wb * bz, w = wa * a[ao + 3] + wb * bw;
  const n = Math.hypot(x, y, z, w) || 1;
  out[o] = x / n; out[o + 1] = y / n; out[o + 2] = z / n; out[o + 3] = w / n;
}

/**
 * World joint states from the pelvis world state and every joint's local state.
 *
 * `local` holds `[t, q, s]` per joint relative to its parent; joints are in
 * parent-before-child order and the pelvis is the root of the animated tree
 * (joint 0 is a fixed world root no vertex is skinned to).
 */
export function forwardKinematics(
  parents: ArrayLike<number>,
  pelvisJoint: number,
  pelvis: ArrayLike<number>,
  local: Float32Array | Float64Array,
  out: Float32Array | Float64Array,
): void {
  for (let k = 0; k < STATE; k++) out[pelvisJoint * STATE + k] = pelvis[k];
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0; out[4] = 0; out[5] = 0; out[6] = 1; out[7] = 1;
  const joints = parents.length;
  for (let j = pelvisJoint + 1; j < joints; j++) {
    compose(out, parents[j] * STATE, local, j * STATE, out, j * STATE);
  }
}
