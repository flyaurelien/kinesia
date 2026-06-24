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
DEFAULT_FOG_PROBABILITY_THRESHOLD = 0.37


@dataclass(frozen=True)
class AnalysisParams:
    """Tunable parameters for a single FoG analysis pass (preset + event-shaping knobs)."""

    preset: str = DEFAULT_ANALYSIS_PRESET
    sensitivity_percent: int = 0
    min_duration_ms: int = 400
    gap_fill_ms: int = 220

    def as_dict(self) -> dict[str, Any]:
        """Serialize the params to a plain, JSON-friendly dict with normalized integer fields."""
        return {
            "preset": self.preset,
            "sensitivity_percent": int(self.sensitivity_percent),
            "min_duration_ms": int(self.min_duration_ms),
            "gap_fill_ms": int(self.gap_fill_ms),
        }


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
    write_json(analysis_directory / "events.json", payload["events"])
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
        "events": payload["events"],
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
    """Build the complete analysis payload (frames, signals, events, QA) from per-frame metadata.

    Stabilizes the root trajectory, derives FoG kinematic metrics, applies the decision
    threshold with gap filling, and annotates each frame with its per-frame outputs.
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

    fog_metrics = build_fog_metrics(frames, fps)
    fog_score = fog_metrics.get("fog.score", [None] * len(frames))
    fog_score_smooth = centered_rolling_mean(fog_score, window=max(3, int(round(max(fps, 1.0) * 0.55))))
    fog_metrics["fog.score_smooth"] = fog_score_smooth
    threshold = infer_threshold([value for value in fog_score if value is not None], params)
    fog_state = fill_short_gaps(
        [value is not None and value >= threshold for value in fog_score],
        max_gap=max(1, round((params.gap_fill_ms / 1000.0) * fps)),
    )

    for index, frame in enumerate(frames):
        frame["fog_score"] = fog_score[index]
        frame["fog_score_smooth"] = fog_score_smooth[index]
        frame["fog_components"] = {
            signal_id: values[index]
            for signal_id, values in fog_metrics.items()
            if signal_id.startswith("fog.") and signal_id not in {"fog.score", "fog.score_smooth"} and index < len(values)
        }
        frame["fog_detected"] = bool(fog_state[index])

    signals = build_signals(frames, stabilization, fog_metrics, threshold, fog_state)
    events = build_events(fog_state, fps, params)
    qa = build_qa_summary(metadata, frames, fog_score, threshold)
    duration_ms = int(round((len(frames) / max(fps, 1e-6)) * 1000.0))
    events["duration_ms"] = duration_ms
    events["run_id"] = run_id
    return {
        "fps": fps,
        "frames": frames,
        "signals": signals,
        "events": events,
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
    fog_metrics: dict[str, list[float | None]],
    threshold: float,
    fog_state: list[bool],
) -> list[dict[str, Any]]:
    """Assemble the ordered list of labeled, unit-tagged display signals from the computed metrics."""
    roots = [frame["root_world_stabilized"] for frame in frames]
    return [
        make_signal("root.stab.x", "Root Stabilized X", "m", "Stabilized root translation X.", [value[0] for value in roots]),
        make_signal("root.stab.y", "Root Stabilized Y", "m", "Stabilized root translation Y.", [value[1] for value in roots]),
        make_signal("root.stab.z", "Root Stabilized Z", "m", "Stabilized root translation Z.", [value[2] for value in roots]),
        make_signal("foot.left.slip_speed", "Left Foot Slip Speed", "m/s", "Estimated XY slip of the left support foot.", stabilization["slip_speed_left"]),
        make_signal("foot.right.slip_speed", "Right Foot Slip Speed", "m/s", "Estimated XY slip of the right support foot.", stabilization["slip_speed_right"]),
        make_signal("root.xy_speed", "Pelvis XY Speed", "m/s", "Horizontal pelvis speed used as a progression context signal.", fog_metrics["root.xy_speed"]),
        make_signal("gait.ankle_relative_speed", "Relative Ankle Activity", "m/s", "Mean lower-limb activity relative to pelvis.", fog_metrics["gait.ankle_relative_speed"]),
        make_signal("gait.context", "Gait-Cycle Context", "score", "Recent evidence that the subject is in a gait-cycle task.", fog_metrics["gait.context"]),
        make_signal("turn.yaw_rate", "Body Turn Rate", "deg/s", "Absolute yaw-rate estimated from the hip/shoulder axis; context only.", fog_metrics["turn.yaw_rate"]),
        make_signal("turn.context", "Turning Task Context", "score", "Recent evidence that the subject is attempting a turn or stepping-in-place task.", fog_metrics["turn.context"]),
        make_signal("fog.walking_context", "Walking Context", "score", "Automatic context weight from center-of-mass XY progression.", fog_metrics["fog.walking_context"]),
        make_signal("fog.turning_context", "Turning Context", "score", "Automatic context weight from body yaw rotation.", fog_metrics["fog.turning_context"]),
        make_signal("fog.locomotor_context", "Locomotor Context", "score", "Automatic gait/turning task context used by the FoG probability estimate.", fog_metrics["fog.locomotor_context"]),
        make_signal("step.effective_event", "Effective Step Event", "bool", "Detected alternating effective foot lift/placement event.", fog_metrics["step.effective_event"]),
        make_signal("step.cadence_hz", "Step Cadence", "Hz", "Instantaneous cadence from alternating effective step events.", fog_metrics["step.cadence_hz"]),
        make_signal("step.time_since_effective", "Time Since Effective Step", "s", "Seconds since the last alternating effective step event.", fog_metrics["step.time_since_effective"]),
        make_signal("fog.progression_arrest", "Progression Arrest Evidence", "score", "Low horizontal pelvis progression evidence; not valid alone during turning-in-place.", fog_metrics["fog.progression_arrest"]),
        make_signal("fog.leg_activity_arrest", "Foot Activity Arrest Evidence", "score", "Low relative ankle activity evidence for akinetic freezing.", fog_metrics["fog.leg_activity_arrest"]),
        make_signal("fog.leg_activity_abs_arrest", "Absolute Foot Activity Arrest", "score", "Conservative low lower-limb activity gate in m/s, used to suppress successful-motion false positives.", fog_metrics["fog.leg_activity_abs_arrest"]),
        make_signal("fog.freezing_index", "Freezing Index 3-8/0.5-3 Hz", "ratio", "Power ratio between freezing band and locomotor band.", fog_metrics["fog.freezing_index"]),
        make_signal("fog.freezing_index_score", "Freezing-Index Evidence", "score", "Normalized freezing-index evidence for trembling freezing.", fog_metrics["fog.freezing_index_score"]),
        make_signal("fog.step_arrest", "Step Arrest Evidence", "score", "No alternating effective step for longer than the subject-specific expected interval.", fog_metrics["fog.step_arrest"]),
        make_signal("fog.alternation_break", "Alternation Break Evidence", "score", "Breakdown of left-right step alternation.", fog_metrics["fog.alternation_break"]),
        make_signal("fog.yaw_arrest", "Turn Arrest Evidence", "score", "Low pelvis/body yaw-rate during a recent turning task context.", fog_metrics["fog.yaw_arrest"]),
        make_signal("fog.contextual_arrest", "Contextual Motion Arrest", "score", "Automatically blends center-of-mass progression arrest and yaw arrest according to current motion context.", fog_metrics["fog.contextual_arrest"]),
        make_signal("fog.ineffective_stepping", "Ineffective Stepping Evidence", "score", "Visible stepping attempt without an effective alternating step.", fog_metrics["fog.ineffective_stepping"]),
        make_signal("fog.akinetic", "Akinetic FoG Evidence", "score", "No clinically visible lower-limb movement during recent stepping/turning context.", fog_metrics["fog.akinetic"]),
        make_signal("fog.trembling", "Kinetic-Trembling FoG Evidence", "score", "High 3-8 Hz lower-limb oscillation without effective stepping.", fog_metrics["fog.trembling"]),
        make_signal("fog.kinetic_no_trembling", "Kinetic-No-Trembling FoG Evidence", "score", "Ineffective non-trembling foot movement, including shuffling/festinating-freezing.", fog_metrics["fog.kinetic_no_trembling"]),
        make_signal("fog.step_failure", "Effective-Step Failure", "probability", "Probability-like evidence that effective stepping has failed in the current task context.", fog_metrics["fog.step_failure"]),
        make_signal("fog.attempted_stepping", "Attempted Stepping Context", "probability", "Observable kinematic context that a gait-related stepping task is underway.", fog_metrics["fog.attempted_stepping"]),
        make_signal("fog.probability", "FoG Event Probability", "probability", "Uncalibrated kinematics-only probability-like estimate of a FoG event.", fog_metrics["fog.probability"]),
        make_signal("fog.score", "FoG Event Probability", "probability", "Compatibility alias for fog.probability.", fog_metrics["fog.score"]),
        make_signal("fog.score_smooth", "FoG Smoothed Probability", "probability", "Centered smoothed FoG probability for display color and live mini-plot.", fog_metrics["fog.score_smooth"]),
        make_signal("fog.threshold", "FoG Probability Threshold", "probability", "Decision threshold used for the kinematics-only FoG probability.", [threshold] * len(frames)),
        make_signal("fog.state", "FoG Detected", "bool", "FoG binary state (1=detected, 0=normal).", [1 if value else 0 for value in fog_state]),
    ]


def build_fog_metrics(frames: list[dict[str, Any]], fps: float) -> dict[str, list[float | None]]:
    """Compute every per-frame FoG kinematic signal and combine them into the FoG probability.

    Derives progression/leg-activity/turning context and arrest evidence, then fuzzy-combines
    them into akinetic/trembling/kinetic FoG sub-scores and the final fog.probability series.
    Returns a series of all-None signals when there are too few frames to differentiate.
    """
    roots = [tuple(frame.get("root_world_stabilized") or frame.get("root_world_raw") or (0.0, 0.0, 0.0)) for frame in frames]
    n = len(roots)
    if n < 3:
        empty = [None] * n
        return {
            "root.xy_speed": empty,
            "gait.ankle_relative_speed": empty,
            "gait.context": empty,
            "turn.yaw_rate": empty,
            "turn.context": empty,
            "fog.walking_context": empty,
            "fog.turning_context": empty,
            "fog.locomotor_context": empty,
            "step.effective_event": empty,
            "step.cadence_hz": empty,
            "step.time_since_effective": empty,
            "fog.progression_arrest": empty,
            "fog.leg_activity_arrest": empty,
            "fog.leg_activity_abs_arrest": empty,
            "fog.freezing_index": empty,
            "fog.freezing_index_score": empty,
            "fog.step_arrest": empty,
            "fog.alternation_break": empty,
            "fog.yaw_arrest": empty,
            "fog.contextual_arrest": empty,
            "fog.ineffective_stepping": empty,
            "fog.akinetic": empty,
            "fog.trembling": empty,
            "fog.kinetic_no_trembling": empty,
            "fog.step_failure": empty,
            "fog.attempted_stepping": empty,
            "fog.probability": empty,
            "fog.score": empty,
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
    freeze_ratio = build_freezing_ratio(ankle_rel, fps)
    turn_rate = build_body_turn_rate(frames, fps)
    step_metrics = build_effective_step_metrics(frames, roots, fps)

    progression_arrest = inverse_quantile_score(root_speed, lo_q=0.12, hi_q=0.82)
    leg_arrest = inverse_quantile_score(ankle_activity, lo_q=0.12, hi_q=0.82)
    leg_abs_arrest = inverse_range_score(ankle_activity, lo=0.10, hi=0.30)
    freeze_band = quantile_score(freeze_ratio, lo_q=0.55, hi_q=0.92)
    progression_activity = quantile_score(root_speed, lo_q=0.18, hi_q=0.82)
    leg_activity_score = quantile_score(ankle_activity, lo_q=0.18, hi_q=0.82)
    gait_context = build_gait_context(progression_activity, leg_activity_score, fps)
    turning_context = quantile_score(turn_rate, lo_q=0.55, hi_q=0.90)
    turning_task_context = rolling_max_score(turning_context, window=max(3, int(round(max(fps, 1.0) * 2.5))))
    walking_task_context = build_walking_context(root_xy, fps)
    locomotor_context = [
        max_optional(walk, turn, leg)
        for walk, turn, leg in zip(walking_task_context, turning_task_context, leg_activity_score, strict=False)
    ]
    yaw_arrest = [
        fuzzy_and([inverse, task])
        for inverse, task in zip(inverse_quantile_score(turn_rate, lo_q=0.12, hi_q=0.75), turning_task_context, strict=False)
    ]
    step_arrest = build_step_arrest_score(step_metrics["step.time_since_effective"], step_metrics["step.expected_interval_s"])
    alternation_break = build_alternation_break_score(step_metrics["step.alternation_ok"], step_metrics["step.event_side"])

    contextual_arrest_score: list[float | None] = []
    ineffective_stepping_score: list[float | None] = []
    akinetic_score: list[float | None] = []
    trembling_score: list[float | None] = []
    kinetic_no_trembling_score: list[float | None] = []
    step_failure_score: list[float | None] = []
    attempted_stepping_score: list[float | None] = []
    fog_probability: list[float | None] = []
    for index in range(n):
        task_context = locomotor_context[index]
        walk_context = walking_task_context[index]
        turn_context = turning_task_context[index]
        visible_attempt = max_optional(leg_activity_score[index], freeze_band[index], alternation_break[index])
        contextual_arrest = context_weighted_arrest(
            progression_arrest[index],
            yaw_arrest[index],
            walk_context,
            turn_context,
        )
        step_failure = max_optional(
            leg_abs_arrest[index],
            contextual_arrest,
            fuzzy_and([step_arrest[index], leg_arrest[index], task_context]),
        )
        attempted_stepping = max_optional(task_context, visible_attempt)
        trembling = fuzzy_and(
            [
                attempted_stepping,
                contextual_arrest,
                freeze_band[index],
            ]
        )
        kinetic_no_trembling = fuzzy_and(
            [
                attempted_stepping,
                contextual_arrest,
                alternation_break[index],
                inverse_score(freeze_band[index]),
            ]
        )
        ineffective = max_optional(trembling, kinetic_no_trembling)
        akinetic = fuzzy_and(
            [
                attempted_stepping,
                leg_abs_arrest[index],
            ]
        )
        probability = leg_abs_arrest[index]
        contextual_arrest_score.append(contextual_arrest)
        ineffective_stepping_score.append(ineffective)
        akinetic_score.append(akinetic)
        trembling_score.append(trembling)
        kinetic_no_trembling_score.append(kinetic_no_trembling)
        step_failure_score.append(step_failure)
        attempted_stepping_score.append(attempted_stepping)
        fog_probability.append(probability)
    return {
        "root.xy_speed": root_speed,
        "gait.ankle_relative_speed": ankle_activity,
        "gait.context": gait_context,
        "turn.yaw_rate": turn_rate,
        "turn.context": turning_task_context,
        "fog.walking_context": walking_task_context,
        "fog.turning_context": turning_task_context,
        "fog.locomotor_context": locomotor_context,
        "step.effective_event": step_metrics["step.effective_event"],
        "step.cadence_hz": step_metrics["step.cadence_hz"],
        "step.time_since_effective": step_metrics["step.time_since_effective"],
        "fog.progression_arrest": progression_arrest,
        "fog.leg_activity_arrest": leg_arrest,
        "fog.leg_activity_abs_arrest": leg_abs_arrest,
        "fog.freezing_index": freeze_ratio,
        "fog.freezing_index_score": freeze_band,
        "fog.step_arrest": step_arrest,
        "fog.alternation_break": alternation_break,
        "fog.yaw_arrest": yaw_arrest,
        "fog.contextual_arrest": contextual_arrest_score,
        "fog.ineffective_stepping": ineffective_stepping_score,
        "fog.akinetic": akinetic_score,
        "fog.trembling": trembling_score,
        "fog.kinetic_no_trembling": kinetic_no_trembling_score,
        "fog.step_failure": step_failure_score,
        "fog.attempted_stepping": attempted_stepping_score,
        "fog.probability": fog_probability,
        "fog.score": fog_probability,
    }


def infer_threshold(values: list[float], params: AnalysisParams) -> float:
    """Decision threshold for fog.probability, shifted by the sensitivity param and clamped."""
    finite = [value for value in values if math.isfinite(value)]
    sensitivity_shift = clamp(params.sensitivity_percent / 100.0, -0.5, 0.5) * 0.20
    if not finite:
        return clamp(DEFAULT_FOG_PROBABILITY_THRESHOLD - sensitivity_shift, 0.10, 0.90)
    return clamp(DEFAULT_FOG_PROBABILITY_THRESHOLD - sensitivity_shift, 0.10, 0.90)


def build_events(fog_state: list[bool], fps: float, params: AnalysisParams) -> dict[str, Any]:
    """Collapse the per-frame FoG state into discrete episodes, dropping ones below min duration."""
    min_frames = max(1, round((params.min_duration_ms / 1000.0) * fps))
    segments: list[dict[str, Any]] = []
    start: int | None = None
    for index, active in enumerate([*fog_state, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= min_frames:
                segments.append(
                    {
                        "label": "fog",
                        "start_frame": start,
                        "end_frame": index - 1,
                        "start_ms": int(round((start / max(fps, 1e-6)) * 1000)),
                        "end_ms": int(round((index / max(fps, 1e-6)) * 1000)),
                    }
                )
            start = None
    return {
        "episodes": [{"label": item["label"], "start_ms": item["start_ms"], "end_ms": item["end_ms"]} for item in segments],
        "summary": {"segments": segments, "count": len(segments)},
    }


def build_qa_summary(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    fog_score: list[float | None],
    threshold: float,
) -> dict[str, Any]:
    """Build the quality-assurance summary (interpretability status, visibility ratios, confidence)."""
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
    finite_scores = [value for value in fog_score if isinstance(value, (int, float)) and math.isfinite(value)]
    event_confidence = None
    if finite_scores:
        margins = [max(0.0, value - threshold) for value in finite_scores]
        event_confidence = sum(margins) / len(margins)
    return {
        "status": status,
        "needs_review": status != "interpretable",
        "tracking_score": average_record_value(metadata, "identity_stability_score"),
        "joint_visibility_ratio": joint_visibility_ratio,
        "critical_joint_visibility_ratio": critical_joint_visibility_ratio,
        "camera_motion_severity": camera_motion_severity(metadata),
        "event_confidence": event_confidence,
        "reasons": reasons,
    }


def write_parquet(file_path: Path, frames: list[dict[str, Any]], signals: list[dict[str, Any]]) -> None:
    """Write per-frame metadata plus every signal column to a Parquet kinematics table."""
    columns: dict[str, list[Any]] = {
        "frame_index": [frame["index"] for frame in frames],
        "video_frame": [frame["video_frame"] for frame in frames],
        "mesh_file": [frame["mesh_file"] for frame in frames],
        "fog_detected": [bool(frame.get("fog_detected")) for frame in frames],
        "fog.score": [frame.get("fog_score") for frame in frames],
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


def fill_short_gaps(values: list[bool], max_gap: int) -> list[bool]:
    """Bridge runs of False no longer than max_gap that are flanked by True on both sides."""
    out = list(values)
    index = 0
    while index < len(out):
        if out[index]:
            index += 1
            continue
        start = index
        while index < len(out) and not out[index]:
            index += 1
        end = index
        if start > 0 and end < len(out) and end - start <= max_gap:
            for fill in range(start, end):
                out[fill] = True
    return out


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


def finite_quantile(values: list[float | None], q: float) -> float | None:
    """Quantile q over the finite entries of values, or None if there are none."""
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return None
    return quantile(finite, q)


def inverse_quantile_score(values: list[float | None], lo_q: float, hi_q: float) -> list[float | None]:
    """Map values to [0,1] inverted between their lo_q/hi_q quantiles (low value -> high score)."""
    lo = finite_quantile(values, lo_q)
    hi = finite_quantile(values, hi_q)
    if lo is None or hi is None or hi <= lo + 1e-9:
        return [None if value is None else 0.0 for value in values]
    return [
        None if value is None or not math.isfinite(float(value))
        else clamp(1.0 - ((float(value) - lo) / (hi - lo)), 0.0, 1.0)
        for value in values
    ]


def quantile_score(values: list[float | None], lo_q: float, hi_q: float) -> list[float | None]:
    """Map values to [0,1] between their lo_q/hi_q quantiles (high value -> high score)."""
    lo = finite_quantile(values, lo_q)
    hi = finite_quantile(values, hi_q)
    if lo is None or hi is None or hi <= lo + 1e-9:
        return [None if value is None else 0.0 for value in values]
    return [
        None if value is None or not math.isfinite(float(value))
        else clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)
        for value in values
    ]


def inverse_range_score(values: list[float | None], lo: float, hi: float) -> list[float | None]:
    """Map values to [0,1] inverted between fixed absolute bounds lo/hi (low value -> high score)."""
    if hi <= lo + 1e-9:
        return [None if value is None else 0.0 for value in values]
    return [
        None if value is None or not math.isfinite(float(value))
        else clamp(1.0 - ((float(value) - lo) / (hi - lo)), 0.0, 1.0)
        for value in values
    ]


def weighted_score(components: list[tuple[float | None, float]]) -> float | None:
    """Weighted average of (value, weight) pairs over finite values, clamped to [0,1]."""
    total_weight = 0.0
    score = 0.0
    for value, weight in components:
        if value is None or not math.isfinite(value):
            continue
        score += float(value) * weight
        total_weight += weight
    return None if total_weight <= 0 else clamp(score / total_weight, 0.0, 1.0)


def max_optional(*values: float | None) -> float | None:
    """Maximum over the finite arguments, or None if none are finite."""
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def inverse_score(value: float | None) -> float | None:
    """Fuzzy NOT: 1 - value clamped to [0,1], passing None/non-finite through as None."""
    if value is None or not math.isfinite(float(value)):
        return None
    return clamp(1.0 - float(value), 0.0, 1.0)


def fuzzy_and(values: Iterable[float | None]) -> float | None:
    """Fuzzy AND: minimum of the finite values (each clamped to [0,1]), or None if none finite."""
    finite = [
        clamp(float(value), 0.0, 1.0)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return min(finite) if finite else None


def context_weighted_arrest(
    progression_arrest: float | None,
    yaw_arrest: float | None,
    walking_context: float | None,
    turning_context: float | None,
) -> float | None:
    """Blend progression and yaw arrest evidence by the current walking vs. turning context weights."""
    walk = 0.0 if walking_context is None or not math.isfinite(float(walking_context)) else clamp(float(walking_context), 0.0, 1.0)
    turn = 0.0 if turning_context is None or not math.isfinite(float(turning_context)) else clamp(float(turning_context), 0.0, 1.0)
    total = walk + turn
    if total <= 1e-6:
        return weighted_score([(progression_arrest, 0.50), (yaw_arrest, 0.50)])
    return weighted_score(
        [
            (progression_arrest, walk / total),
            (yaw_arrest, turn / total),
        ]
    )


def rolling_max_score(values: list[float | None], window: int) -> list[float | None]:
    """Centered rolling maximum over finite values; windows with no finite values yield None."""
    if not values:
        return []
    win = max(1, int(window))
    pad = win // 2
    out: list[float | None] = []
    for index in range(len(values)):
        lo = max(0, index - pad)
        hi = min(len(values), index + pad + 1)
        finite = [float(value) for value in values[lo:hi] if isinstance(value, (int, float)) and math.isfinite(float(value))]
        out.append(max(finite) if finite else None)
    return out


def build_gait_context(
    progression_activity: list[float | None],
    leg_activity: list[float | None],
    fps: float,
) -> list[float | None]:
    """Recent evidence of an active gait cycle: windowed max of blended progression/leg activity."""
    n = max(len(progression_activity), len(leg_activity))
    if n == 0:
        return []
    half_window = max(3, int(round(max(fps, 1.0) * 1.2)))
    out: list[float | None] = []
    for index in range(n):
        lo = max(0, index - half_window)
        hi = min(n, index + half_window + 1)
        values: list[float] = []
        for cursor in range(lo, hi):
            progression = progression_activity[cursor] if cursor < len(progression_activity) else None
            leg = leg_activity[cursor] if cursor < len(leg_activity) else None
            local = weighted_score([(progression, 0.45), (leg, 0.55)])
            if local is not None and math.isfinite(local):
                values.append(local)
        out.append(max(values) if values else None)
    return out


def build_walking_context(root_xy: np.ndarray, fps: float) -> list[float | None]:
    """Walking-task context score from windowed pelvis XY displacement against absolute thresholds."""
    n = len(root_xy)
    if n == 0:
        return []
    window = max(3, int(round(max(fps, 1.0) * 2.0)))
    out: list[float | None] = []
    for index in range(n):
        previous = max(0, index - window)
        delta = root_xy[index] - root_xy[previous]
        if not np.isfinite(delta).all():
            out.append(None)
            continue
        displacement = float(np.linalg.norm(delta))
        # Absolute thresholds keep turning-in-place from being re-labeled as walking
        # just because it has locally high but small pelvis drift.
        out.append(clamp((displacement - 0.12) / (0.45 - 0.12), 0.0, 1.0))
    return rolling_mean(out, window=max(3, int(round(max(fps, 1.0) * 0.45))))


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


def build_step_arrest_score(values: list[float | None], expected_interval_s: float) -> list[float | None]:
    """Score [0,1] for how overdue the next step is, ramping past the subject's expected interval."""
    start = clamp(expected_interval_s * 1.35, 0.65, 1.45)
    full = clamp(expected_interval_s * 2.50, 1.30, 2.20)
    if full <= start + 1e-6:
        full = start + 0.60
    out: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            out.append(None)
        else:
            out.append(clamp((float(value) - start) / (full - start), 0.0, 1.0))
    return out


def build_alternation_break_score(
    alternation_ok: list[float | None],
    event_side: list[str | None],
) -> list[float | None]:
    """Evidence that left-right step alternation has broken down, decaying after each bad step."""
    out: list[float | None] = []
    last_bad_frame: int | None = None
    decay_frames = 18
    for index, (ok, side) in enumerate(zip(alternation_ok, event_side, strict=False)):
        if side is not None and ok == 0.0:
            last_bad_frame = index
        if last_bad_frame is None:
            out.append(0.0)
        else:
            out.append(clamp(1.0 - ((index - last_bad_frame) / max(decay_frames, 1)), 0.0, 1.0))
    return out


def build_freezing_ratio(values: np.ndarray, fps: float) -> list[float | None]:
    """Sliding-window freezing index: 3-8 Hz vs. 0.5-3 Hz spectral power ratio of foot-relative motion."""
    n = len(values)
    if n == 0:
        return []
    window = max(16, int(round(max(fps, 1.0) * 2.0)))
    if window > n:
        return [None] * n
    half = window // 2
    sample_rate = max(float(fps), 1.0)
    freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
    loco_mask = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_mask = (freqs >= 3.0) & (freqs <= min(8.0, sample_rate * 0.45))
    if not np.any(loco_mask) or not np.any(freeze_mask):
        return [None] * n
    out: list[float | None] = [None] * n
    window_fn = np.hanning(window)
    for center in range(n):
        lo = center - half
        hi = lo + window
        if lo < 0 or hi > n:
            continue
        chunk = values[lo:hi].astype(np.float64)
        chunk = chunk - np.mean(chunk, axis=0, keepdims=True)
        chunk = chunk * window_fn[:, None]
        spectrum = np.fft.rfft(chunk, axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        loco_power = float(np.sum(power[loco_mask]))
        freeze_power = float(np.sum(power[freeze_mask]))
        if not math.isfinite(loco_power + freeze_power) or (loco_power + freeze_power) <= 1e-12:
            out[center] = None
        else:
            out[center] = freeze_power / max(loco_power, 1e-12)
    return out


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
