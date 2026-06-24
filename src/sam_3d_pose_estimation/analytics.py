from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    append_analysis_to_run_manifest,
    build_analysis_manifest,
    ensure_run_layout,
    read_json,
    run_manifest_path,
    run_metadata_path,
    write_json,
)
from .workspace import DEFAULT_ANALYSIS_PRESET, analysis_dir, project_root_from, sanitize_run_id

LEFT_ANKLE = 13
RIGHT_ANKLE = 14
LEFT_BIG_TOE = 15
LEFT_SMALL_TOE = 16
LEFT_HEEL = 17
RIGHT_BIG_TOE = 18
RIGHT_SMALL_TOE = 19
RIGHT_HEEL = 20


@dataclass(frozen=True)
class AnalysisParams:
    """Tunable parameters for a single kinematics analysis pass."""

    preset: str = DEFAULT_ANALYSIS_PRESET

    def as_dict(self) -> dict[str, Any]:
        """Serialize the params to a plain, JSON-friendly dict."""
        return {"preset": self.preset}


def analyze_run(
    *,
    run_id: str,
    params: AnalysisParams,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run the full analysis for a run, write all artifacts to disk, and update the run manifest."""
    project_root_resolved = project_root_from(project_root)
    safe_run_id = sanitize_run_id(run_id)
    run_directory = ensure_run_layout(safe_run_id, project_root_resolved)
    manifest = read_json(run_manifest_path(safe_run_id, project_root_resolved))
    metadata = read_json(run_metadata_path(safe_run_id, project_root_resolved))
    analysis_id = build_analysis_id(safe_run_id, params)
    analysis_directory = analysis_dir(safe_run_id, analysis_id, project_root_resolved)
    analysis_directory.mkdir(parents=True, exist_ok=True)

    payload = build_analysis_payload(run_id=safe_run_id, metadata=metadata, params=params)
    write_json(
        analysis_directory / "signals.json",
        {"analysis_id": analysis_id, "preset": params.preset, "signals": payload["signals"]},
    )
    write_json(
        analysis_directory / "frames.json",
        {
            "analysis_id": analysis_id,
            "preset": params.preset,
            "fps": payload["fps"],
            "frames": payload["frames"],
        },
    )
    write_json(analysis_directory / "qa.json", payload["qa"])
    write_parquet(analysis_directory / "kinematics.parquet", payload["frames"], payload["signals"])
    analysis_manifest = build_analysis_manifest(
        run_id=safe_run_id,
        analysis_id=analysis_id,
        preset=params.preset,
        parameters=params.as_dict(),
        qa_summary=payload["qa"],
    )
    write_json(analysis_directory / "analysis_manifest.json", analysis_manifest)
    next_manifest = append_analysis_to_run_manifest(
        manifest,
        analysis_id=analysis_id,
        preset=params.preset,
        parameters=params.as_dict(),
        qa_summary=payload["qa"],
    )
    write_json(run_directory / "run_manifest.json", next_manifest)
    return {
        "analysis_id": analysis_id,
        "manifest": analysis_manifest,
        "signals": payload["signals"],
        "qa": payload["qa"],
        "duration_ms": payload["duration_ms"],
    }


def build_analysis_id(run_id: str, params: AnalysisParams) -> str:
    """Derive a stable analysis id from the run id, preset, and a hash of the params."""
    payload = json.dumps(params.as_dict(), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"{sanitize_run_id(run_id)}_{params.preset}_{digest}"


def build_analysis_payload(
    *,
    run_id: str,
    metadata: dict[str, Any],
    params: AnalysisParams,
) -> dict[str, Any]:
    """Build the complete analysis payload (frames, signals, QA) from per-frame metadata.

    Stabilizes the root trajectory and derives the per-frame kinematic signal series.
    """
    records_raw = metadata.get("records")
    records = [item for item in records_raw if isinstance(item, dict)] if isinstance(records_raw, list) else []
    fps = numeric_or_default(metadata.get("fps_output"), 30.0)
    frames: list[dict[str, Any]] = []
    root_world_raw: list[tuple[float, float, float]] = []

    for index, record in enumerate(records):
        camera_comp = parse_triplet(record.get("camera_comp_cam_xyz"), default=(0.0, 0.0, 0.0))
        joints = parse_joint_array(record.get("joints_cam_xyz")) or parse_joint_array(record.get("joints_space_cam_xyz"))
        root = estimate_root_world(joints, camera_comp)
        root_world_raw.append(root)
        frames.append(
            {
                "index": index,
                "video_frame": int(record.get("video_frame") or index),
                "mesh_file": Path(str(record.get("mesh_path") or "")).name,
                "subject_present": bool(record.get("subject_present", True)),
                "inference_status": str(record.get("inference_status") or ""),
                "camera_comp": list(camera_comp),
                "joints_cam": joints,
                "root_world_raw": list(root),
            }
        )

    stabilization = stabilize_xy(frames, root_world_raw, fps)
    for index, frame in enumerate(frames):
        frame["root_world_stabilized"] = list(stabilization["root_world_stabilized"][index])
        frame["foot_contact"] = stabilization["foot_contact"][index]

    metrics = build_kinematic_metrics(frames, fps)
    signals = build_signals(frames, stabilization, metrics)
    qa = build_qa_summary(metadata, frames)
    duration_ms = int(round((len(frames) / max(fps, 1e-6)) * 1000.0))
    return {
        "fps": fps,
        "frames": frames,
        "signals": signals,
        "qa": qa,
        "duration_ms": duration_ms,
    }


def stabilize_xy(
    frames: list[dict[str, Any]],
    root_world_raw: list[tuple[float, float, float]],
    fps: float,
) -> dict[str, Any]:
    """Stabilize the horizontal root trajectory using support-foot anchoring and smoothing.

    Picks whichever of the support-anchored or plainly-smoothed trajectory is steadier,
    and also returns per-frame foot contact states and left/right foot slip speeds.
    """
    n = len(root_world_raw)
    contacts = [detect_foot_contact(frame) for frame in frames]
    if n == 0:
        return {"root_world_stabilized": [], "foot_contact": [], "slip_speed_left": [], "slip_speed_right": []}

    raw_radius = max(1, int(round(max(fps, 1.0) * 0.60)))
    smoothed_raw = smooth_xy_roots(root_world_raw, raw_radius)

    support_corrected: list[tuple[float, float, float]] = []
    correction_xy = np.zeros(2, dtype=np.float64)
    support_anchor_xy: np.ndarray | None = None
    previous_support = "none"
    for frame, root, contact_state in zip(frames, root_world_raw, contacts, strict=False):
        support = str(contact_state.get("support") or "none")
        support_xy = support_foot_xy(frame, support)
        if support_xy is not None:
            if support_anchor_xy is None or support != previous_support:
                support_anchor_xy = support_xy + correction_xy
            target_correction = support_anchor_xy - support_xy
            correction_xy = correction_xy * 0.78 + target_correction * 0.22
        support_corrected.append((root[0] + float(correction_xy[0]), root[1] + float(correction_xy[1]), root[2]))
        previous_support = support

    support_radius = max(1, int(round(max(fps, 1.0) * 0.22)))
    support_stabilized = smooth_xy_roots(support_corrected, support_radius)
    root_world_stabilized = (
        support_stabilized
        if xy_stability_score(support_stabilized) < xy_stability_score(smoothed_raw)
        else smoothed_raw
    )

    slip_speed_left: list[float | None] = []
    slip_speed_right: list[float | None] = []
    previous_left_xy: np.ndarray | None = None
    previous_right_xy: np.ndarray | None = None

    for frame in frames:
        left_xy = support_foot_xy(frame, "left")
        right_xy = support_foot_xy(frame, "right")
        slip_speed_left.append(xy_speed(previous_left_xy, left_xy, fps))
        slip_speed_right.append(xy_speed(previous_right_xy, right_xy, fps))
        previous_left_xy = left_xy if left_xy is not None else previous_left_xy
        previous_right_xy = right_xy if right_xy is not None else previous_right_xy

    return {
        "root_world_stabilized": root_world_stabilized,
        "foot_contact": contacts,
        "slip_speed_left": slip_speed_left,
        "slip_speed_right": slip_speed_right,
    }


def support_foot_xy(frame: dict[str, Any], support: str) -> np.ndarray | None:
    """Return the mean world XY of the named support foot's joints, or None if unavailable."""
    joints = frame.get("joints_cam")
    if not isinstance(joints, list):
        return None
    if support == "left":
        indices = [LEFT_ANKLE, LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL]
    elif support == "right":
        indices = [RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL]
    elif support == "both":
        indices = [LEFT_ANKLE, LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL, RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL]
    else:
        return None
    points: list[tuple[float, float, float]] = []
    for index in indices:
        point = joint_world(joints, index)
        if point is not None:
            points.append(point)
    if not points:
        return None
    return np.asarray([mean(point[0] for point in points), mean(point[1] for point in points)], dtype=np.float64)


def xy_stability_score(roots: list[tuple[float, float, float]]) -> float:
    """Lower-is-steadier score: bounding-box extent plus net XY drift of the trajectory."""
    if len(roots) < 2:
        return 0.0
    xs = [root[0] for root in roots]
    ys = [root[1] for root in roots]
    extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    net = math.hypot(roots[-1][0] - roots[0][0], roots[-1][1] - roots[0][1])
    return extent + net


def smooth_xy_roots(roots: list[tuple[float, float, float]], radius: int) -> list[tuple[float, float, float]]:
    """Centered moving-average smooth of the X/Y components of a root trajectory; Z untouched."""
    window = max(1, int(radius) * 2 + 1)
    xs = centered_rolling_mean([root[0] for root in roots], window)
    ys = centered_rolling_mean([root[1] for root in roots], window)
    return [
        (
            float(xs[index]) if xs[index] is not None else root[0],
            float(ys[index]) if ys[index] is not None else root[1],
            root[2],
        )
        for index, root in enumerate(roots)
    ]


def xy_speed(previous: np.ndarray | None, current: np.ndarray | None, fps: float) -> float | None:
    """Horizontal speed (m/s) between two consecutive XY points, or None if either is missing."""
    if previous is None or current is None:
        return None
    delta = current - previous
    if not np.isfinite(delta).all():
        return None
    return float(np.linalg.norm(delta) * max(float(fps), 1.0))


def build_signals(
    frames: list[dict[str, Any]],
    stabilization: dict[str, Any],
    metrics: dict[str, list[float | None]],
) -> list[dict[str, Any]]:
    """Assemble the ordered list of labeled, unit-tagged kinematics signals from the computed metrics."""
    roots = [frame["root_world_stabilized"] for frame in frames]
    return [
        make_signal("root.stab.x", "Root Stabilized X", "m", "Stabilized root translation X.", [value[0] for value in roots]),
        make_signal("root.stab.y", "Root Stabilized Y", "m", "Stabilized root translation Y.", [value[1] for value in roots]),
        make_signal("root.stab.z", "Root Stabilized Z", "m", "Stabilized root translation Z.", [value[2] for value in roots]),
        make_signal("foot.left.slip_speed", "Left Foot Slip Speed", "m/s", "Estimated XY slip of the left support foot.", stabilization["slip_speed_left"]),
        make_signal("foot.right.slip_speed", "Right Foot Slip Speed", "m/s", "Estimated XY slip of the right support foot.", stabilization["slip_speed_right"]),
        make_signal("root.xy_speed", "Pelvis XY Speed", "m/s", "Horizontal pelvis progression speed.", metrics["root.xy_speed"]),
        make_signal("gait.ankle_relative_speed", "Relative Ankle Activity", "m/s", "Mean lower-limb activity relative to the pelvis.", metrics["gait.ankle_relative_speed"]),
        make_signal("turn.yaw_rate", "Body Turn Rate", "deg/s", "Absolute yaw-rate estimated from the hip/shoulder axis.", metrics["turn.yaw_rate"]),
        make_signal("step.effective_event", "Effective Step Event", "bool", "Detected alternating effective foot lift/placement event.", metrics["step.effective_event"]),
        make_signal("step.cadence_hz", "Step Cadence", "Hz", "Instantaneous cadence from alternating effective step events.", metrics["step.cadence_hz"]),
        make_signal("step.time_since_effective", "Time Since Effective Step", "s", "Seconds since the last alternating effective step event.", metrics["step.time_since_effective"]),
    ]


def build_kinematic_metrics(frames: list[dict[str, Any]], fps: float) -> dict[str, list[float | None]]:
    """Compute the per-frame kinematic signal series (speeds, turn rate, step events).

    Returns a series of all-None signals when there are too few frames to differentiate.
    """
    roots = [tuple(frame.get("root_world_stabilized") or frame.get("root_world_raw") or (0.0, 0.0, 0.0)) for frame in frames]
    n = len(roots)
    if n < 3:
        empty = [None] * n
        return {
            "root.xy_speed": empty,
            "gait.ankle_relative_speed": empty,
            "turn.yaw_rate": empty,
            "step.effective_event": empty,
            "step.cadence_hz": empty,
            "step.time_since_effective": empty,
        }

    root_xy = np.asarray([[float(root[0]), float(root[1])] for root in roots], dtype=np.float64)
    root_xy = smooth_array(root_xy, window=max(3, int(round(max(fps, 1.0) * 0.22))))
    root_speed = derivative_norm(root_xy, fps)

    ankle_rel = build_ankle_relative_series(frames, roots)
    ankle_activity = derivative_norm(
        smooth_array(ankle_rel, window=max(3, int(round(max(fps, 1.0) * 0.10)))),
        fps,
    )
    ankle_activity = rolling_mean(ankle_activity, window=max(3, int(round(max(fps, 1.0) * 0.35))))
    turn_rate = build_body_turn_rate(frames, fps)
    step_metrics = build_effective_step_metrics(frames, roots, fps)

    return {
        "root.xy_speed": root_speed,
        "gait.ankle_relative_speed": ankle_activity,
        "turn.yaw_rate": turn_rate,
        "step.effective_event": step_metrics["step.effective_event"],
        "step.cadence_hz": step_metrics["step.cadence_hz"],
        "step.time_since_effective": step_metrics["step.time_since_effective"],
    }


def build_qa_summary(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the quality-assurance summary (interpretability status, visibility ratios)."""
    total = len(frames)
    visible = sum(1 for frame in frames if isinstance(frame.get("joints_cam"), list) and frame["joints_cam"])
    critical = 0
    for frame in frames:
        joints = frame.get("joints_cam")
        if isinstance(joints, list) and len(joints) > RIGHT_HEEL:
            critical += 1
    joint_visibility_ratio = visible / max(total, 1)
    critical_joint_visibility_ratio = critical / max(total, 1)
    status = "interpretable"
    reasons: list[str] = []
    if critical_joint_visibility_ratio < 0.25:
        status = "non_interpretable"
        reasons.append("critical_joints_missing")
    elif joint_visibility_ratio < 0.60:
        status = "needs_review"
        reasons.append("low_joint_visibility")
    return {
        "status": status,
        "needs_review": status != "interpretable",
        "tracking_score": average_record_value(metadata, "identity_stability_score"),
        "joint_visibility_ratio": joint_visibility_ratio,
        "critical_joint_visibility_ratio": critical_joint_visibility_ratio,
        "camera_motion_severity": camera_motion_severity(metadata),
        "reasons": reasons,
    }


def write_parquet(file_path: Path, frames: list[dict[str, Any]], signals: list[dict[str, Any]]) -> None:
    """Write per-frame metadata plus every signal column to a Parquet kinematics table."""
    columns: dict[str, list[Any]] = {
        "frame_index": [frame["index"] for frame in frames],
        "video_frame": [frame["video_frame"] for frame in frames],
        "mesh_file": [frame["mesh_file"] for frame in frames],
    }
    for signal in signals:
        columns[str(signal["id"])] = list(signal["values"])
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, file_path)


def detect_foot_contact(frame: dict[str, Any]) -> dict[str, Any]:
    """Infer left/right foot ground contact by comparing each foot's lowest joint to the floor."""
    joints = frame.get("joints_cam")
    if not isinstance(joints, list) or len(joints) <= RIGHT_HEEL:
        return {"left": False, "right": False, "support": "none"}
    left_z = min_world_z(joints, [LEFT_ANKLE, LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL])
    right_z = min_world_z(joints, [RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL])
    floor = min(value for value in [left_z, right_z] if value is not None) if left_z is not None or right_z is not None else None
    if floor is None:
        return {"left": False, "right": False, "support": "none"}
    margin = 0.055
    left = left_z is not None and left_z <= floor + margin
    right = right_z is not None and right_z <= floor + margin
    support = "both" if left and right else "left" if left else "right" if right else "none"
    return {"left": left, "right": right, "support": support}


def min_world_z(joints: list[Any], indices: list[int]) -> float | None:
    """Lowest world-Z (height) among the given joint indices, or None if none are valid."""
    values: list[float] = []
    for index in indices:
        if index >= len(joints) or not isinstance(joints[index], list) or len(joints[index]) < 3:
            continue
        world = cam_to_world_xyz(float(joints[index][0]), float(joints[index][1]), float(joints[index][2]))
        values.append(world[2])
    return min(values) if values else None


def estimate_root_world(
    joints_cam: list[list[float]] | None,
    camera_comp: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Estimate the world-space root: midpoint of the hips, falling back to pelvis then camera comp."""
    if joints_cam:
        hip_points = [joint_world(joints_cam, LEFT_HIP_INDEX), joint_world(joints_cam, RIGHT_HIP_INDEX)]
        hip_points = [point for point in hip_points if point is not None]
        if hip_points:
            return tuple(mean(point[axis] for point in hip_points) for axis in range(3))  # type: ignore[return-value]
        pelvis = joint_world(joints_cam, 0)
        if pelvis is not None:
            return pelvis
    return cam_to_world_xyz(*camera_comp)


LEFT_HIP_INDEX = 9
RIGHT_HIP_INDEX = 10


def joint_world(joints_cam: list[list[float]], index: int) -> tuple[float, float, float] | None:
    """World-space XYZ of one camera-space joint, or None if the index/values are invalid."""
    if index >= len(joints_cam) or len(joints_cam[index]) < 3:
        return None
    try:
        return cam_to_world_xyz(float(joints_cam[index][0]), float(joints_cam[index][1]), float(joints_cam[index][2]))
    except (TypeError, ValueError):
        return None


def smooth_array(values: np.ndarray, window: int) -> np.ndarray:
    """Centered edge-padded moving-average along axis 0 of an (N, ...) array; odd window enforced."""
    if values.size == 0:
        return values
    win = max(1, int(window))
    if win <= 1:
        return values.astype(np.float64, copy=True)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    values_2d = values.reshape((values.shape[0], -1)).astype(np.float64)
    padded = np.pad(values_2d, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    out = np.vstack([
        np.convolve(padded[:, dim], kernel, mode="valid")
        for dim in range(values_2d.shape[1])
    ]).T
    return out.reshape(values.shape)


def derivative_norm(values: np.ndarray, fps: float) -> list[float | None]:
    """Per-frame speed: norm of the frame-to-frame difference scaled by fps (first frame is None)."""
    if len(values) == 0:
        return []
    out: list[float | None] = [None]
    scale = max(float(fps), 1.0)
    values_2d = values.reshape((values.shape[0], -1)).astype(np.float64)
    for index in range(1, len(values_2d)):
        delta = values_2d[index] - values_2d[index - 1]
        if not np.isfinite(delta).all():
            out.append(None)
        else:
            out.append(float(np.linalg.norm(delta) * scale))
    return out


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    """Centered rolling mean over finite values only; windows with no finite values yield None."""
    if not values:
        return []
    win = max(1, int(window))
    pad = win // 2
    out: list[float | None] = []
    for index in range(len(values)):
        lo = max(0, index - pad)
        hi = min(len(values), index + pad + 1)
        finite = [float(value) for value in values[lo:hi] if isinstance(value, (int, float)) and math.isfinite(float(value))]
        out.append(sum(finite) / len(finite) if finite else None)
    return out


def centered_rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    """Like rolling_mean but forces an odd window for a symmetric, truly centered average."""
    if not values:
        return []
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    out: list[float | None] = []
    for index in range(len(values)):
        lo = max(0, index - pad)
        hi = min(len(values), index + pad + 1)
        finite = [float(value) for value in values[lo:hi] if isinstance(value, (int, float)) and math.isfinite(float(value))]
        out.append(sum(finite) / len(finite) if finite else None)
    return out


def build_ankle_relative_series(
    frames: list[dict[str, Any]],
    roots: list[tuple[float, float, float]],
) -> np.ndarray:
    """Per-frame left/right ankle positions expressed relative to the root, as a (N, 6) array.

    Non-finite or missing frames carry forward the previous frame's values (or zeros).
    """
    rows: list[list[float]] = []
    last: list[float] | None = None
    for frame, root in zip(frames, roots, strict=False):
        joints = frame.get("joints_cam")
        left = joint_world(joints, LEFT_ANKLE) if isinstance(joints, list) else None
        right = joint_world(joints, RIGHT_ANKLE) if isinstance(joints, list) else None
        values: list[float] = []
        for point in (left, right):
            if point is None:
                values.extend([0.0, 0.0, 0.0])
            else:
                values.extend([point[0] - root[0], point[1] - root[1], point[2] - root[2]])
        if not all(math.isfinite(value) for value in values):
            values = last if last is not None else [0.0] * 6
        rows.append(values)
        last = values
    return np.asarray(rows, dtype=np.float64)


def build_effective_step_metrics(
    frames: list[dict[str, Any]],
    roots: list[tuple[float, float, float]],
    fps: float,
) -> dict[str, Any]:
    """Detect effective alternating foot-lift steps and derive cadence, recency, and alternation.

    Merges left/right lift events, suppresses ones closer than a refractory interval, and
    tracks the subject-specific expected step interval used downstream for step-arrest scoring.
    """
    n = len(frames)
    left = build_foot_relative_series(frames, roots, "left")
    right = build_foot_relative_series(frames, roots, "right")
    left_events = detect_foot_lift_events(left, fps)
    right_events = detect_foot_lift_events(right, fps)
    raw_events = sorted([(frame, "left") for frame in left_events] + [(frame, "right") for frame in right_events])

    min_interval = max(1, int(round(max(fps, 1.0) * 0.22)))
    effective_events: list[tuple[int, str, bool]] = []
    last_frame: int | None = None
    last_side: str | None = None
    for frame, side in raw_events:
        if last_frame is not None and frame - last_frame < min_interval:
            continue
        alternates = last_side is None or side != last_side
        effective_events.append((frame, side, alternates))
        last_frame = frame
        last_side = side

    effective_event_signal: list[float | None] = [0.0] * n
    event_side_signal: list[str | None] = [None] * n
    alternation_ok_signal: list[float | None] = [None] * n
    cadence: list[float | None] = [None] * n
    time_since: list[float | None] = [None] * n

    alternating_frames = [frame for frame, _side, alternates in effective_events if alternates]
    intervals_s = [
        (current - previous) / max(float(fps), 1e-6)
        for previous, current in zip(alternating_frames, alternating_frames[1:], strict=False)
        if current > previous
    ]
    valid_intervals = [value for value in intervals_s if 0.20 <= value <= 2.50]
    expected_interval = quantile(valid_intervals, 0.65) if valid_intervals else 0.85
    expected_interval = clamp(expected_interval, 0.45, 1.60)

    last_effective_frame: int | None = None
    event_cursor = 0
    previous_effective_frame: int | None = None
    previous_cadence: float | None = None
    for index in range(n):
        while event_cursor < len(effective_events) and effective_events[event_cursor][0] == index:
            frame, side, alternates = effective_events[event_cursor]
            effective_event_signal[frame] = 1.0 if alternates else 0.0
            event_side_signal[frame] = side
            alternation_ok_signal[frame] = 1.0 if alternates else 0.0
            if alternates:
                if previous_effective_frame is not None and frame > previous_effective_frame:
                    interval = (frame - previous_effective_frame) / max(float(fps), 1e-6)
                    previous_cadence = 1.0 / interval if interval > 1e-6 else None
                previous_effective_frame = frame
                last_effective_frame = frame
            event_cursor += 1
        cadence[index] = previous_cadence
        if last_effective_frame is None:
            time_since[index] = 0.0
        else:
            time_since[index] = (index - last_effective_frame) / max(float(fps), 1e-6)

    return {
        "step.effective_event": effective_event_signal,
        "step.event_side": event_side_signal,
        "step.alternation_ok": alternation_ok_signal,
        "step.cadence_hz": cadence,
        "step.time_since_effective": time_since,
        "step.expected_interval_s": expected_interval,
    }


def build_foot_relative_series(
    frames: list[dict[str, Any]],
    roots: list[tuple[float, float, float]],
    side: str,
) -> np.ndarray:
    """Per-frame (N, 3) series of one foot's mean position: XY relative to the root, Z in world space."""
    indices = [LEFT_ANKLE, LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_HEEL] if side == "left" else [RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_HEEL]
    rows: list[list[float]] = []
    last: list[float] | None = None
    for frame, root in zip(frames, roots, strict=False):
        joints = frame.get("joints_cam")
        points: list[tuple[float, float, float]] = []
        if isinstance(joints, list):
            for index in indices:
                point = joint_world(joints, index)
                if point is not None:
                    points.append(point)
        if points:
            foot = [mean(point[axis] for point in points) for axis in range(3)]
            values = [foot[0] - root[0], foot[1] - root[1], foot[2]]
        else:
            values = last if last is not None else [0.0, 0.0, 0.0]
        if not all(math.isfinite(value) for value in values):
            values = last if last is not None else [0.0, 0.0, 0.0]
        rows.append(values)
        last = values
    return np.asarray(rows, dtype=np.float64)


def detect_foot_lift_events(foot_rel: np.ndarray, fps: float) -> list[int]:
    """Find foot-lift event frames as local height peaks that also clear lift and motion thresholds."""
    n = len(foot_rel)
    if n < 5:
        return []
    smooth_window = max(3, int(round(max(fps, 1.0) * 0.08)))
    values = smooth_array(foot_rel, smooth_window)
    height = values[:, 2]
    xy = values[:, :2]
    finite_height = [float(value) for value in height if math.isfinite(float(value))]
    if not finite_height:
        return []
    floor = quantile(finite_height, 0.12)
    lift = np.maximum(height - floor, 0.0)
    lift_finite = [float(value) for value in lift if math.isfinite(float(value))]
    lift_threshold = max(0.012, (quantile(lift_finite, 0.88) - quantile(lift_finite, 0.25)) * 0.35)

    half_motion_window = max(1, int(round(max(fps, 1.0) * 0.18)))
    displacement = np.zeros(n, dtype=np.float64)
    for index in range(n):
        lo = max(0, index - half_motion_window)
        hi = min(n - 1, index + half_motion_window)
        delta = xy[hi] - xy[lo]
        displacement[index] = float(np.linalg.norm(delta)) if np.isfinite(delta).all() else 0.0
    disp_values = [float(value) for value in displacement if math.isfinite(float(value))]
    movement_threshold = max(0.010, quantile(disp_values, 0.65) * 0.35)

    candidates: list[int] = []
    for index in range(1, n - 1):
        if lift[index] < lift_threshold or displacement[index] < movement_threshold:
            continue
        if lift[index] >= lift[index - 1] and lift[index] >= lift[index + 1]:
            candidates.append(index)

    min_gap = max(1, int(round(max(fps, 1.0) * 0.25)))
    events: list[int] = []
    for candidate in candidates:
        if not events or candidate - events[-1] >= min_gap:
            events.append(candidate)
            continue
        if lift[candidate] > lift[events[-1]]:
            events[-1] = candidate
    return events


def build_body_turn_rate(frames: list[dict[str, Any]], fps: float) -> list[float | None]:
    """Per-frame absolute body yaw rate (deg/s) from the hip axis, falling back to the shoulder axis."""
    angles: list[float | None] = []
    for frame in frames:
        joints = frame.get("joints_cam")
        angle = None
        if isinstance(joints, list):
            angle = body_axis_angle(joints, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
            if angle is None:
                angle = body_axis_angle(joints, 5, 6)
        angles.append(angle)
    unwrapped = unwrap_axial_angles(angles)
    if not unwrapped:
        return []
    out: list[float | None] = [None]
    scale = max(float(fps), 1.0) * 180.0 / math.pi
    for index in range(1, len(unwrapped)):
        current = unwrapped[index]
        previous = unwrapped[index - 1]
        if current is None or previous is None:
            out.append(None)
        else:
            out.append(abs(current - previous) * scale)
    return rolling_mean(out, window=max(3, int(round(max(fps, 1.0) * 0.20))))


def body_axis_angle(joints_cam: list[list[float]], left_index: int, right_index: int) -> float | None:
    """Heading angle (radians) of the world-XY axis from the left to the right body landmark."""
    left = joint_world(joints_cam, left_index)
    right = joint_world(joints_cam, right_index)
    if left is None or right is None:
        return None
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def unwrap_axial_angles(angles: list[float | None]) -> list[float | None]:
    """Fill gaps and phase-unwrap an axial angle series, preserving None where input was missing."""
    valid = [value for value in angles if value is not None and math.isfinite(value)]
    if not valid:
        return [None] * len(angles)
    filled: list[float] = []
    last = valid[0]
    for value in angles:
        if value is not None and math.isfinite(value):
            last = value
        filled.append(last)
    # A hip or shoulder left-right axis is axial: theta and theta + pi are equivalent.
    unwrapped = np.unwrap(np.asarray(filled, dtype=np.float64) * 2.0) / 2.0
    out: list[float | None] = []
    for source, value in zip(angles, unwrapped, strict=False):
        out.append(None if source is None else float(value))
    return out


def parse_joint_array(value: Any) -> list[list[float]] | None:
    """Validate and coerce a raw value into a list of finite XYZ joints, or None if malformed."""
    if not isinstance(value, list):
        return None
    out: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return None
        x, y, z = float(item[0]), float(item[1]), float(item[2])
        if not all(math.isfinite(axis) for axis in (x, y, z)):
            return None
        out.append([x, y, z])
    return out


def parse_triplet(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Coerce a raw value into a finite XYZ triplet, returning default if missing or malformed."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return default
    try:
        triplet = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return default
    if not all(math.isfinite(axis) for axis in triplet):
        return default
    return triplet


def cam_to_world_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert camera-space XYZ to the world frame (Z forward, X right, Y up -> world X/Y/Z)."""
    return (-z, x, -y)


def make_signal(
    signal_id: str,
    label: str,
    unit: str,
    description: str,
    values: list[Any],
) -> dict[str, Any]:
    """Pack a signal's id, label, unit, description, and value series into the signal dict shape."""
    return {"id": signal_id, "label": label, "unit": unit, "description": description, "values": values}


def average_record_value(metadata: dict[str, Any], key: str) -> float | None:
    """Mean of a numeric per-record field across metadata records, or None if none are present."""
    records = metadata.get("records")
    if not isinstance(records, list):
        return None
    values = [float(item[key]) for item in records if isinstance(item, dict) and isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def camera_motion_severity(metadata: dict[str, Any]) -> float | None:
    """Normalize the mean per-step camera shift into a [0,1] motion-severity score, or None."""
    camera_motion = metadata.get("camera_motion_compensation")
    if not isinstance(camera_motion, dict):
        return None
    value = camera_motion.get("shift_px_step_mean_small")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return clamp(float(value) / 15.0, 0.0, 1.0)


def numeric_or_default(value: Any, default: float) -> float:
    """Return value as a float if it is finite numeric, otherwise the supplied default."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of the values, or 0.0 for an empty iterable."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile q over the finite values, or 0.0 if there are none."""
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    pos = (len(finite) - 1) * clamp(q, 0.0, 1.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def clamp(value: float, lo: float, hi: float) -> float:
    """Constrain value to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))
