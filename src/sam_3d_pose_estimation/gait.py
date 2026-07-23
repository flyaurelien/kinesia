"""Clinical gait-analysis layer.

Turns the per-frame reconstruction into the deliverables of a professional
gait lab:

- ZERO-PHASE FILTERING: joint trajectories are low-passed with a 4th-order
  Butterworth applied forward and backward (filtfilt) — the biomechanics
  standard — before any angle is computed. Zero phase means the curves carry
  no temporal lag, so event timings stay honest.
- CLINICAL SAGITTAL ANGLES: hip flexion, knee flexion and ankle dorsiflexion,
  measured in the subject's own sagittal plane (defined per frame from the
  pelvis axis and the feet's pointing direction), not in camera axes.
- GAIT EVENTS: heel-strikes and toe-offs from the ground-contact labels,
  refined to the local minimum of the (filtered) heel height.
- SPATIOTEMPORAL PARAMETERS: cadence, step/stride time and length, walking
  speed, stance/swing and double-support percentages — with per-event values
  and mean +/- SD aggregates.
- STATIC NEUTRAL REFERENCE: the monocular reconstruction carries a systematic
  standing-posture bias (shank tilted forward, toes up), so quiet-stance spans
  are detected and used as a calibration pose, exactly as a gait lab uses a
  static trial. The measured offsets are reported and are reversible.
- GAIT-CYCLE NORMALIZATION: every angle resampled to 0-100% of the gait cycle
  per stride, aggregated as mean +/- SD per side.

Everything operates on the analysis frame dicts produced by analytics.py
(joints in camera space; world = (-z, x, -y), Z up, metres).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

# Pose joint indices (COCO-WholeBody subset exported by the reconstruction).
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 9, 10
L_KNEE, R_KNEE = 11, 12
L_ANKLE, R_ANKLE = 13, 14
L_BIG_TOE, L_HEEL = 15, 17
R_BIG_TOE, R_HEEL = 18, 20

DEFAULT_CUTOFF_HZ = 6.0  # standard gait low-pass cutoff
FILTER_ORDER = 4
MIN_STRIDE_S = 0.4
MAX_STRIDE_S = 3.0
MIN_EVENT_GAP_S = 0.25
CYCLE_POINTS = 101  # 0..100% inclusive
IN_PLACE_SPEED_M_S = 0.10  # below: treat as stepping in place (no step length)

# Static (quiet-stance) calibration pose detection.
STATIC_MAX_SPEED_M_S = 0.08
STATIC_MIN_RUN_S = 0.5
STATIC_MIN_TOTAL_S = 1.0
STATIC_MAX_TRUNK_LEAN_DEG = 25.0
STATIC_MAX_OFFSET_DEG = 45.0  # beyond this the "static" pose is not a stance


def cam_to_world(point: list[float] | tuple[float, float, float]) -> np.ndarray:
    """Camera space -> world space: world X back-projects depth, Z is up."""
    return np.array([-point[2], point[0], -point[1]], dtype=np.float64)


def joint_world_series(frames: list[dict[str, Any]], index: int) -> np.ndarray:
    """(n, 3) world positions of one joint; NaN rows where unavailable."""
    out = np.full((len(frames), 3), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        joints = frame.get("joints_cam")
        if not frame.get("subject_present", True) or not joints or len(joints) <= index:
            continue
        joint = joints[index]
        if joint is None:
            continue
        world = cam_to_world(joint)
        if np.all(np.isfinite(world)):
            out[i] = world
    return out


def zero_phase_lowpass(values: np.ndarray, fps: float, cutoff_hz: float = DEFAULT_CUTOFF_HZ) -> np.ndarray:
    """4th-order Butterworth low-pass applied forward+backward (zero phase).

    NaN gaps are bridged by linear interpolation for filtering and restored
    afterwards, so absent-subject spans stay absent. Series too short to
    filter are returned untouched.
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        return x
    nyquist = max(fps, 1.0) / 2.0
    wn = min(0.99, cutoff_hz / nyquist)
    b, a = butter(FILTER_ORDER, wn)
    pad = 3 * max(len(a), len(b))
    if x.shape[0] <= pad:
        return x
    idx = np.arange(x.shape[0], dtype=np.float64)
    bridged = x.copy()
    bridged[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    filtered = filtfilt(b, a, bridged)
    filtered[~finite] = np.nan
    return filtered


def filter_positions(series: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase filter each axis of an (n, 3) position series."""
    out = series.copy()
    for axis in range(series.shape[1]):
        out[:, axis] = zero_phase_lowpass(series[:, axis], fps, cutoff_hz)
    return out


# ── Clinical sagittal angles ─────────────────────────────────────────────────

def _sagittal_angle(vector: np.ndarray, forward: np.ndarray, up: np.ndarray) -> float:
    """Angle (deg, CCW) of a vector inside the sagittal plane, from `forward`."""
    return math.degrees(math.atan2(float(np.dot(vector, up)), float(np.dot(vector, forward))))


def _wrap_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def compute_clinical_angles(
    frames: list[dict[str, Any]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> dict[str, list[float | None]]:
    """Sagittal-plane hip/knee flexion and ankle dorsiflexion, both sides (deg).

    The sagittal plane is the subject's own: its normal is the (horizontal)
    pelvis axis; `forward` is the feet's mean pointing direction, so the
    convention holds whichever way the person walks relative to the camera.
    Joint positions are zero-phase filtered BEFORE the angles are formed.
    """
    joints = {
        index: filter_positions(joint_world_series(frames, index), fps, cutoff_hz)
        for index in (
            L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
            L_BIG_TOE, L_HEEL, R_BIG_TOE, R_HEEL,
        )
    }
    up = np.array([0.0, 0.0, 1.0])
    n = len(frames)
    out: dict[str, list[float | None]] = {
        "hip.left": [None] * n, "hip.right": [None] * n,
        "knee.left": [None] * n, "knee.right": [None] * n,
        "ankle.left": [None] * n, "ankle.right": [None] * n,
    }
    for i in range(n):
        l_hip, r_hip = joints[L_HIP][i], joints[R_HIP][i]
        if not (np.all(np.isfinite(l_hip)) and np.all(np.isfinite(r_hip))):
            continue
        lateral = r_hip - l_hip
        lateral[2] = 0.0
        if np.linalg.norm(lateral) < 1e-6:
            continue
        lateral /= np.linalg.norm(lateral)
        # Forward = the feet's mean heel->toe direction (horizontalized). It is
        # body-derived, so the sign convention survives any walking direction.
        forwards = []
        for toe_i, heel_i in ((L_BIG_TOE, L_HEEL), (R_BIG_TOE, R_HEEL)):
            toe, heel = joints[toe_i][i], joints[heel_i][i]
            if np.all(np.isfinite(toe)) and np.all(np.isfinite(heel)):
                v = toe - heel
                v[2] = 0.0
                if np.linalg.norm(v) > 1e-6:
                    forwards.append(v / np.linalg.norm(v))
        if not forwards:
            continue
        forward = np.mean(forwards, axis=0)
        # Remove any lateral component so (forward, up) spans the sagittal plane.
        forward = forward - np.dot(forward, lateral) * lateral
        if np.linalg.norm(forward) < 1e-6:
            continue
        forward /= np.linalg.norm(forward)

        for side, hip_i, knee_i, ankle_i, toe_i, heel_i in (
            ("left", L_HIP, L_KNEE, L_ANKLE, L_BIG_TOE, L_HEEL),
            ("right", R_HIP, R_KNEE, R_ANKLE, R_BIG_TOE, R_HEEL),
        ):
            hip, knee = joints[hip_i][i], joints[knee_i][i]
            ankle = joints[ankle_i][i]
            toe, heel = joints[toe_i][i], joints[heel_i][i]
            if not (
                np.all(np.isfinite(hip)) and np.all(np.isfinite(knee)) and np.all(np.isfinite(ankle))
            ):
                continue
            thigh_angle = _sagittal_angle(knee - hip, forward, up)  # standing ~ -90
            shank_angle = _sagittal_angle(ankle - knee, forward, up)
            out[f"hip.{side}"][i] = _wrap_deg(thigh_angle + 90.0)
            out[f"knee.{side}"][i] = _wrap_deg(thigh_angle - shank_angle)
            if np.all(np.isfinite(toe)) and np.all(np.isfinite(heel)):
                foot_angle = _sagittal_angle(toe - heel, forward, up)  # flat ~ 0
                out[f"ankle.{side}"][i] = _wrap_deg(foot_angle - (shank_angle + 90.0))
    return out


# ── Static neutral reference ─────────────────────────────────────────────────

def find_static_frames(
    frames: list[dict[str, Any]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> np.ndarray:
    """Boolean mask of quiet-stance frames usable as a calibration pose.

    A frame qualifies when both feet are on the ground, the pelvis is barely
    moving and the trunk is upright; qualifying frames must also belong to a
    run of at least STATIC_MIN_RUN_S so that the brief double-support phases of
    normal walking never pass for a stance.
    """
    n = len(frames)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    hips = 0.5 * (
        filter_positions(joint_world_series(frames, L_HIP), fps, cutoff_hz)
        + filter_positions(joint_world_series(frames, R_HIP), fps, cutoff_hz)
    )
    shoulders = 0.5 * (
        filter_positions(joint_world_series(frames, L_SHOULDER), fps, cutoff_hz)
        + filter_positions(joint_world_series(frames, R_SHOULDER), fps, cutoff_hz)
    )
    speed = np.full(n, np.inf)
    if n > 1:
        step = np.linalg.norm(np.diff(hips[:, :2], axis=0), axis=1) * fps
        speed[1:] = step
        speed[0] = speed[1]
    both = _contact_series(frames, "left") & _contact_series(frames, "right")
    trunk = shoulders - hips
    trunk_norm = np.linalg.norm(trunk, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lean = np.degrees(np.arccos(np.clip(trunk[:, 2] / trunk_norm, -1.0, 1.0)))
    candidate = both & np.isfinite(speed) & (speed <= STATIC_MAX_SPEED_M_S)
    candidate &= np.isfinite(lean) & (lean <= STATIC_MAX_TRUNK_LEAN_DEG)

    min_run = max(1, int(round(STATIC_MIN_RUN_S * fps)))
    start = None
    for i in range(n + 1):
        active = bool(candidate[i]) if i < n else False
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= min_run:
                mask[start:i] = True
            start = None
    if mask.sum() < int(round(STATIC_MIN_TOTAL_S * fps)):
        mask[:] = False
    return mask


def compute_neutral_reference(
    frames: list[dict[str, Any]],
    angles: dict[str, list[float | None]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> dict[str, Any]:
    """Measure the calibration-pose offset of every clinical angle.

    Quiet standing is anatomically ~0 deg of hip flexion, knee flexion and
    ankle dorsiflexion, so whatever the angles read over a detected stance is
    the systematic offset of the reconstruction (monocular depth places the
    shank slightly forward-tilted and the toes slightly up). Subtracting it is
    the same operation a gait lab performs with a static trial.
    """
    static = find_static_frames(frames, fps, cutoff_hz)
    result: dict[str, Any] = {
        "applied": False,
        "method": "static quiet-stance median (subject's own calibration pose)",
        "static_frames": int(static.sum()),
        "static_duration_s": round(float(static.sum()) / max(fps, 1e-6), 3),
        "offsets_deg": {},
        "note": (
            "Angles are reported relative to the subject's own quiet stance. "
            "Add the offsets back to recover raw reconstruction angles. A "
            "genuinely non-neutral standing posture is absorbed into the offset."
        ),
    }
    if not static.any():
        result["note"] = (
            "No quiet stance found in this clip, so angles are raw "
            "reconstruction values and may carry a systematic posture offset."
        )
        return result
    offsets: dict[str, float] = {}
    for key, values in angles.items():
        selected = np.array(
            [values[i] for i in range(len(values)) if static[i] and values[i] is not None],
            dtype=np.float64,
        )
        if selected.size < max(4, int(round(0.2 * fps))):
            return result
        offsets[key] = float(np.median(selected))
    if any(abs(value) > STATIC_MAX_OFFSET_DEG for value in offsets.values()):
        result["note"] = (
            "The detected stance produced implausible offsets, so no neutral "
            "reference was applied and angles are raw reconstruction values."
        )
        return result
    result["applied"] = True
    result["offsets_deg"] = {key: round(value, 3) for key, value in offsets.items()}
    return result


def apply_neutral_reference(
    angles: dict[str, list[float | None]],
    neutral: dict[str, Any],
) -> dict[str, list[float | None]]:
    """Subtract the calibration-pose offsets so quiet stance reads ~0 deg."""
    if not neutral.get("applied"):
        return angles
    offsets = neutral["offsets_deg"]
    return {
        key: [None if v is None else _wrap_deg(v - offsets.get(key, 0.0)) for v in values]
        for key, values in angles.items()
    }


# ── Gait events ──────────────────────────────────────────────────────────────

def _contact_series(frames: list[dict[str, Any]], side: str) -> np.ndarray:
    out = np.zeros(len(frames), dtype=bool)
    for i, frame in enumerate(frames):
        contact = frame.get("foot_contact") or {}
        out[i] = bool(contact.get(side)) and bool(frame.get("subject_present", True))
    return out


def detect_gait_events(
    frames: list[dict[str, Any]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> list[dict[str, Any]]:
    """Heel-strikes / toe-offs per side from the contact labels, with each
    heel-strike refined to the local minimum of the filtered heel height."""
    events: list[dict[str, Any]] = []
    min_gap = max(1, int(round(MIN_EVENT_GAP_S * fps)))
    refine = max(1, int(round(0.12 * fps)))
    for side, heel_index in (("left", L_HEEL), ("right", R_HEEL)):
        contact = _contact_series(frames, side)
        heel_z = zero_phase_lowpass(joint_world_series(frames, heel_index)[:, 2], fps, cutoff_hz)
        last_at = {"heel_strike": -(10 ** 9), "toe_off": -(10 ** 9)}
        for i in range(1, len(frames)):
            kind: str | None = None
            if contact[i] and not contact[i - 1]:
                kind = "heel_strike"
            elif not contact[i] and contact[i - 1]:
                kind = "toe_off"
            if kind is None or i - last_at[kind] < min_gap:
                continue
            frame_index = i
            if kind == "heel_strike":
                lo, hi = max(0, i - refine), min(len(frames), i + refine + 1)
                window = heel_z[lo:hi]
                if np.any(np.isfinite(window)):
                    frame_index = lo + int(np.nanargmin(window))
            last_at[kind] = i
            events.append(
                {
                    "frame": int(frame_index),
                    "time_s": round(frame_index / max(fps, 1e-6), 4),
                    "side": side,
                    "type": kind,
                }
            )
    events.sort(key=lambda e: (e["frame"], e["side"], e["type"]))
    return events


# ── Spatiotemporal parameters ────────────────────────────────────────────────

def _agg(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def compute_spatiotemporal(
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> dict[str, Any]:
    """Cadence, step/stride time & length, speed, stance/swing/double-support."""
    heel = {
        "left": filter_positions(joint_world_series(frames, L_HEEL), fps, cutoff_hz),
        "right": filter_positions(joint_world_series(frames, R_HEEL), fps, cutoff_hz),
    }
    both_contact = _contact_series(frames, "left") & _contact_series(frames, "right")
    hs = {
        side: [e for e in events if e["side"] == side and e["type"] == "heel_strike"]
        for side in ("left", "right")
    }
    to = {
        side: [e for e in events if e["side"] == side and e["type"] == "toe_off"]
        for side in ("left", "right")
    }

    stride_times: list[float] = []
    stride_lengths: list[float] = []
    speeds: list[float] = []
    stance_pcts: list[float] = []
    double_support_pcts: list[float] = []
    for side in ("left", "right"):
        strikes = hs[side]
        for a, b in zip(strikes, strikes[1:]):
            duration = (b["frame"] - a["frame"]) / max(fps, 1e-6)
            if not (MIN_STRIDE_S <= duration <= MAX_STRIDE_S):
                continue
            stride_times.append(duration)
            pa, pb = heel[side][a["frame"], :2], heel[side][b["frame"], :2]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                length = float(np.linalg.norm(pb - pa))
                stride_lengths.append(length)
                speeds.append(length / duration)
            offs = [e for e in to[side] if a["frame"] < e["frame"] < b["frame"]]
            if offs:
                stance_pcts.append(100.0 * (offs[0]["frame"] - a["frame"]) / (b["frame"] - a["frame"]))
            span = both_contact[a["frame"]:b["frame"]]
            if span.size > 0:
                double_support_pcts.append(100.0 * float(span.mean()))

    step_times: list[float] = []
    step_lengths: list[float] = []
    all_hs = sorted(hs["left"] + hs["right"], key=lambda e: e["frame"])
    for a, b in zip(all_hs, all_hs[1:]):
        if a["side"] == b["side"]:
            continue
        duration = (b["frame"] - a["frame"]) / max(fps, 1e-6)
        if not (MIN_STRIDE_S / 2 <= duration <= MAX_STRIDE_S):
            continue
        step_times.append(duration)
        pa = heel[a["side"]][a["frame"], :2]
        pb = heel[b["side"]][b["frame"], :2]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            step_lengths.append(float(np.linalg.norm(pb - pa)))

    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    walking = mean_speed >= IN_PLACE_SPEED_M_S
    stance = _agg(stance_pcts)
    return {
        "walking_detected": bool(walking and len(stride_times) >= 2),
        "cadence_steps_per_min": round(60.0 / float(np.mean(step_times)), 2) if step_times else None,
        "step_time_s": _agg(step_times),
        "step_length_m": _agg(step_lengths) if walking else _agg([]),
        "stride_time_s": _agg(stride_times),
        "stride_length_m": _agg(stride_lengths) if walking else _agg([]),
        "walking_speed_m_s": _agg(speeds) if walking else _agg([]),
        "stance_pct": stance,
        "swing_pct": (
            {"mean": round(100.0 - stance["mean"], 4), "sd": stance["sd"], "n": stance["n"]}
            if stance["mean"] is not None
            else stance
        ),
        "double_support_pct": _agg(double_support_pcts),
    }


# ── Gait-cycle normalization ─────────────────────────────────────────────────

def normalize_cycles(
    series: list[float | None],
    events: list[dict[str, Any]],
    side: str,
    fps: float,
) -> dict[str, Any]:
    """Resample one signal to 0-100% of each same-side gait cycle; mean +/- SD."""
    values = np.array([np.nan if v is None else float(v) for v in series], dtype=np.float64)
    strikes = [e for e in events if e["side"] == side and e["type"] == "heel_strike"]
    cycles: list[np.ndarray] = []
    for a, b in zip(strikes, strikes[1:]):
        start, end = a["frame"], b["frame"]
        duration = (end - start) / max(fps, 1e-6)
        if not (MIN_STRIDE_S <= duration <= MAX_STRIDE_S) or end - start < 4:
            continue
        segment = values[start:end + 1]
        finite = np.isfinite(segment)
        if finite.mean() < 0.8:
            continue  # too much of the cycle is missing to be trustworthy
        idx = np.arange(segment.size, dtype=np.float64)
        bridged = np.interp(idx, idx[finite], segment[finite])
        cycles.append(np.interp(np.linspace(0, segment.size - 1, CYCLE_POINTS), idx, bridged))
    if not cycles:
        return {"n_cycles": 0, "mean": None, "sd": None}
    stack = np.vstack(cycles)
    return {
        "n_cycles": int(stack.shape[0]),
        "mean": [round(float(v), 3) for v in stack.mean(axis=0)],
        "sd": [round(float(v), 3) for v in stack.std(axis=0, ddof=1 if stack.shape[0] > 1 else 0)],
    }


# ── Entry point ──────────────────────────────────────────────────────────────

ANGLE_SIGNALS: list[tuple[str, str, str]] = [
    ("hip.left", "gait.hip.left.flexion_deg", "Hip Flexion L (clinical)"),
    ("hip.right", "gait.hip.right.flexion_deg", "Hip Flexion R (clinical)"),
    ("knee.left", "gait.knee.left.flexion_deg", "Knee Flexion L (clinical)"),
    ("knee.right", "gait.knee.right.flexion_deg", "Knee Flexion R (clinical)"),
    ("ankle.left", "gait.ankle.left.dorsiflexion_deg", "Ankle Dorsiflexion L (clinical)"),
    ("ankle.right", "gait.ankle.right.dorsiflexion_deg", "Ankle Dorsiflexion R (clinical)"),
]


def build_gait_analysis(
    frames: list[dict[str, Any]],
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> dict[str, Any]:
    """Full gait layer: filtered clinical angles, events, spatiotemporal
    parameters and cycle-normalized curves. Degrades gracefully (empty events,
    zero cycles) on non-gait content such as standing subjects."""
    angles = compute_clinical_angles(frames, fps, cutoff_hz)
    neutral = compute_neutral_reference(frames, angles, fps, cutoff_hz)
    angles = apply_neutral_reference(angles, neutral)
    events = detect_gait_events(frames, fps, cutoff_hz)
    spatiotemporal = compute_spatiotemporal(frames, events, fps, cutoff_hz)
    cycles: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for angle_key, signal_id, _label in ANGLE_SIGNALS:
            if angle_key.endswith(f".{side}"):
                cycles[side][signal_id] = normalize_cycles(angles[angle_key], events, side, fps)
    return {
        "params": {"filter": "butterworth", "order": FILTER_ORDER, "cutoff_hz": cutoff_hz, "zero_phase": True},
        "neutral_reference": neutral,
        "angles": angles,
        "events": events,
        "spatiotemporal": spatiotemporal,
        "cycles": cycles,
    }
