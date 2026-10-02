"""Planted feet that stay put: two-bone inverse kinematics of the legs.

The trajectory solve moves the whole body. What is left of a planted foot's
creep comes from the leg pose itself (SAM 3D Body's hip and knee angles vary
a little from frame to frame), and a pose that keeps the legs straight while
the pelvis drops pushes the feet into the floor. As in the post-processing of
GVHMR (Shen et al., SIGGRAPH Asia 2024), the legs are then bent by inverse
kinematics so that each planted foot holds one spot on the floor and no foot
goes below it. The foot keeps its orientation; hip and knee turn as little
as they can, in the leg's own bending plane, and each correction fades in
and out over a few frames around its contact.
"""

from __future__ import annotations

import numpy as np

from . import quat

RAMP_SECONDS = 0.15
MAX_SHIFT = 0.15  # metres: a bigger correction means a mistaken contact, not a creeping foot
LEGS = (("l_upleg", "l_lowleg", "l_foot"), ("r_upleg", "r_lowleg", "r_foot"))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def shifts(feet: np.ndarray, planted: np.ndarray, rate: float) -> np.ndarray:
    """How far each ankle should move (2, N, 3), from the foot keypoints (2, N, K, 3).

    During a contact the foot is brought to the mean spot of that contact,
    sole on the floor; no foot may be below the floor at any time.
    """
    count = feet.shape[1]
    ramp = max(1, int(round(RAMP_SECONDS * rate)))
    out = np.zeros((2, count, 3))
    for side in range(2):
        centre = feet[side].mean(axis=1)
        sole = feet[side][..., 2].min(axis=1)
        target, weight = np.zeros((count, 3)), np.zeros(count)
        for a, b in _runs(planted[:, side]):
            target[a:b, :2] = centre[a:b, :2].mean(axis=0) - centre[a:b, :2]
            target[a:b, 2] = -sole[a:b]
            weight[a:b] = 1.0
            for k in range(1, ramp + 1):  # fade the correction in before and out after
                fade = 1.0 - k / (ramp + 1)
                for t, source in ((a - k, a), (b - 1 + k, b - 1)):
                    if 0 <= t < count and weight[t] < fade:
                        weight[t], target[t] = fade, target[source]
        shift = target * weight[:, None]
        shift[:, 2] += np.clip(-(sole + shift[:, 2]), 0.0, None)  # nothing below the floor
        size = np.linalg.norm(shift, axis=1, keepdims=True)
        out[side] = shift * np.minimum(1.0, MAX_SHIFT / size.clip(1e-9))
    return out


def bend_legs(joint_names: list[str], parents: np.ndarray, states: np.ndarray, local_q: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Local joint rotations (N, J, 4) with each leg bent so that its ankle moves by ``moves`` (2, N, 3).

    ``states`` (N, J, 8) are the world joint states the moves are measured in.
    """
    out = local_q.copy()
    for side, names in enumerate(LEGS):
        hip, knee, ankle = (joint_names.index(name) for name in names)
        rows = np.flatnonzero(np.linalg.norm(moves[side], axis=1) > 1e-4)
        if not len(rows):
            continue
        H, K, A = states[rows, hip, :3], states[rows, knee, :3], states[rows, ankle, :3]
        target = A + moves[side, rows]
        thigh, shin = np.linalg.norm(K - H, axis=1), np.linalg.norm(A - K, axis=1)
        reach = target - H
        distance = np.linalg.norm(reach, axis=1).clip(1e-9)
        direction = reach / distance[:, None]
        distance = np.clip(distance, np.abs(thigh - shin) + 1e-4, (thigh + shin) * (1 - 1e-4))
        bend = (K - H) - np.sum((K - H) * direction, axis=1, keepdims=True) * direction  # the knee's side
        size = np.linalg.norm(bend, axis=1)
        straight = size < 1e-6
        bend /= size.clip(1e-9)[:, None]
        along = (thigh**2 - shin**2 + distance**2) / (2 * distance)
        knee_at = H + along[:, None] * direction + np.sqrt((thigh**2 - along**2).clip(0))[:, None] * bend
        ankle_at = H + distance[:, None] * direction
        turn_thigh = quat.between(K - H, knee_at - H)
        turn_shin = quat.between(quat.rotate(turn_thigh, A - K), ankle_at - knee_at)
        hip_q = quat.multiply(turn_thigh, states[rows, hip, 3:7])
        knee_q = quat.multiply(turn_shin, quat.multiply(turn_thigh, states[rows, knee, 3:7]))
        keep = rows[~straight]
        parent_q = states[rows, parents[hip], 3:7]
        out[keep, hip] = quat.multiply(quat.conjugate(parent_q), hip_q)[~straight]
        out[keep, knee] = quat.multiply(quat.conjugate(hip_q), knee_q)[~straight]
        out[keep, ankle] = quat.multiply(quat.conjugate(knee_q), states[rows, ankle, 3:7])[~straight]  # the foot keeps its turn
    return out
