"""Per-person motion measures shown in the viewer's motion panel.

Everything is derived from the final (smoothed, grounded) world keypoints, so
the numbers agree with what the 3D scene shows. Monocular depth is uncertain:
distances and speeds are estimates, best compared between people of the same
clip rather than read as laboratory measurements.
"""

from __future__ import annotations

import numpy as np

from . import filters
from .keypoints import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NECK,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)  # fmt: skip
from .motion import PersonMotion

SERIES = {
    "speed": "Speed (m/s)",
    "height": "Pelvis height (m)",
    "left_knee": "Left knee flexion (deg)",
    "right_knee": "Right knee flexion (deg)",
    "left_hip": "Left hip flexion (deg)",
    "right_hip": "Right hip flexion (deg)",
    "left_elbow": "Left elbow flexion (deg)",
    "right_elbow": "Right elbow flexion (deg)",
    "trunk_lean": "Trunk lean (deg)",
}
JUMP_MIN_HEIGHT = 0.15  # both feet this far off the floor counts as a jump


def _flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """180 minus the angle at ``b`` between ``a`` and ``c`` (0 = straight limb)."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)).clip(1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _hip_flexion(shoulder_mid: np.ndarray, hip: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Angle between the thigh and the trunk line, 0 when standing straight."""
    trunk = shoulder_mid - hip
    thigh = knee - hip
    cos = np.sum(trunk * thigh, axis=-1) / (np.linalg.norm(trunk, axis=-1) * np.linalg.norm(thigh, axis=-1)).clip(1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def person_metrics(motion: PersonMotion, rate: float) -> dict:
    kp = motion.keypoints
    pelvis = 0.5 * (kp[:, LEFT_HIP] + kp[:, RIGHT_HIP])
    shoulders = 0.5 * (kp[:, LEFT_SHOULDER] + kp[:, RIGHT_SHOULDER])
    series: dict[str, np.ndarray] = {}
    speed = np.zeros(len(kp))
    distance = 0.0
    for lo, hi in _index_spans(motion):
        xy = pelvis[lo:hi, :2]
        if hi - lo > 1:
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            distance += float(step.sum())
            v = np.concatenate([[step[0]], step]) * rate
            speed[lo:hi] = filters.smooth(v, rate, min_cutoff=1.0, beta=0.2)
    series["speed"] = speed
    series["height"] = pelvis[:, 2]
    series["left_knee"] = _flexion(kp[:, LEFT_HIP], kp[:, LEFT_KNEE], kp[:, LEFT_ANKLE])
    series["right_knee"] = _flexion(kp[:, RIGHT_HIP], kp[:, RIGHT_KNEE], kp[:, RIGHT_ANKLE])
    series["left_hip"] = _hip_flexion(shoulders, kp[:, LEFT_HIP], kp[:, LEFT_KNEE])
    series["right_hip"] = _hip_flexion(shoulders, kp[:, RIGHT_HIP], kp[:, RIGHT_KNEE])
    series["left_elbow"] = _flexion(kp[:, LEFT_SHOULDER], kp[:, LEFT_ELBOW], kp[:, LEFT_WRIST])
    series["right_elbow"] = _flexion(kp[:, RIGHT_SHOULDER], kp[:, RIGHT_ELBOW], kp[:, RIGHT_WRIST])
    up = kp[:, NECK] - pelvis
    series["trunk_lean"] = np.degrees(np.arccos(np.clip(up[:, 2] / np.linalg.norm(up, axis=1).clip(1e-9), -1, 1)))

    feet_low = np.minimum(kp[:, [15, 16, 17], 2].min(axis=1), kp[:, [18, 19, 20], 2].min(axis=1))
    jumps = []
    for lo, hi in _index_spans(motion):  # a jump never spans a gap in the footage
        airborne = feet_low[lo:hi] > JUMP_MIN_HEIGHT
        for start, end in _runs(airborne):
            if end - start >= 4 and start > 0 and end < hi - lo:  # take-off and landing both seen
                peak = lo + start + int(np.argmax(feet_low[lo + start : lo + end]))
                jumps.append({"frame": int(motion.frames[peak]), "height": round(float(feet_low[peak]), 3), "duration": round((end - start) / rate, 3)})

    measured = motion.measured
    # Speeds from glimpses shorter than a second are dominated by noise.
    steady = np.zeros(len(kp), dtype=bool)
    for lo, hi in _index_spans(motion):
        if hi - lo >= rate:
            steady[lo:hi] = True
    trusted = measured & steady
    summary = {
        "visible_seconds": round(float(measured.sum()) / rate, 2),
        "distance_m": round(distance, 1),
        "top_speed": round(float(np.percentile(speed[trusted], 98)) if trusted.any() else 0.0, 2),
        "mean_speed": round(float(speed[trusted].mean()) if trusted.any() else 0.0, 2),
        "height_m": round(float(motion.stats.get("height_m", 0.0)), 2),
        "jumps": jumps[:50],
        "highest_jump_m": max((j["height"] for j in jumps), default=0.0),
    }
    return {"summary": summary, "series": {k: np.round(v, 3).tolist() for k, v in series.items()}}


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _index_spans(motion: PersonMotion) -> list[tuple[int, int]]:
    spans, index = [], 0
    for lo, hi in motion.segments:
        n = hi - lo + 1
        spans.append((index, index + n))
        index += n
    return spans

