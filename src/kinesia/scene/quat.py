"""Quaternion and similarity-transform helpers (numpy, ``[x, y, z, w]`` order).

The body model (Momentum Human Rig) stores a joint as ``[tx, ty, tz, qx, qy,
qz, qw, s]``: a translation, a unit quaternion and a uniform scale. The same
layout is used everywhere in Kinesia, including the files the viewer reads.
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by unit quaternions ``q`` (broadcasting)."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def from_matrix(m: np.ndarray) -> np.ndarray:
    """Unit quaternions from rotation matrices (Shepperd's method, batched)."""
    m = np.asarray(m, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = np.stack(
        [
            np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + trace], -1),
            np.stack([1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
            np.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
            np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], m[..., 1, 0] - m[..., 0, 1]], -1),
        ],
        -2,
    )
    diag = np.stack([trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    best = np.argmax(diag, axis=-1)
    q = np.take_along_axis(cands, best[..., None, None], axis=-2)[..., 0, :]
    return normalize(q)


def align_hemisphere(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs along ``axis`` so consecutive quaternions stay on one hemisphere."""
    q = np.array(q, dtype=np.float64, copy=True)
    moved = np.moveaxis(q, axis, 0)
    for i in range(1, moved.shape[0]):
        flip = np.sum(moved[i] * moved[i - 1], axis=-1) < 0
        moved[i][flip] *= -1
    return q


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)[..., None] if np.ndim(t) else float(t)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot).clip(-1, 1)
    theta = np.arccos(dot)
    small = theta < 1e-6
    sin = np.sin(theta)
    wa = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, sin))
    wb = np.where(small, t, np.sin(t * theta) / np.where(small, 1, sin))
    return normalize(wa * a + wb * b)


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between unit quaternions, radians."""
    return 2 * np.arccos(np.abs(np.sum(a * b, axis=-1)).clip(0, 1))


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarity transform ``a ∘ b`` on ``[t, q, s]`` states (apply ``b`` first)."""
    ta, qa, sa = a[..., :3], a[..., 3:7], a[..., 7:8]
    tb, qb, sb = b[..., :3], b[..., 3:7], b[..., 7:8]
    t = ta + rotate(qa, sa * tb)
    return np.concatenate([t, multiply(qa, qb), sa * sb], axis=-1)


def apply(state: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform points by ``[t, q, s]`` states (broadcasting)."""
    return state[..., :3] + rotate(state[..., 3:7], state[..., 7:8] * points)


def axis_angle(axis: np.ndarray, radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(radians / 2), [np.cos(radians / 2)]])
