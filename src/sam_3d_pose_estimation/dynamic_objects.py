"""Track a known-size spherical object through a fixed-camera reconstruction.

This module intentionally does *not* pretend that a generic moving object can
be recovered metrically from one camera. A sphere with a supplied diameter is a
well-constrained first case: its apparent radius supplies depth, while its mask
centre supplies the viewing ray. The output is a time-indexed world trajectory
in the same camera-derived frame as the subject reconstruction.

The resulting position is only valid for a fixed camera and a visible, roughly
circular sphere. Ball spin is not emitted: it is unobservable for an unmarked
sphere from a monocular video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .object_masks import segment_object_instances
from .video_io import open_video

SCHEMA = "kinesia.dynamic_sphere.v1"
STAGE_MARKER = "[scene]"
MIN_RADIUS_PX = 3.0
MAX_GAP_FOR_INTERPOLATION_FRAMES = 12
CAMERA_MOTION_MIN_VALID_PAIRS = 4
CAMERA_MOTION_MAX_SAMPLES = 80
CAMERA_MOTION_MAX_TRANSLATION_DIAGONAL_RATIO_PER_FRAME = 0.001
CAMERA_MOTION_MAX_ROTATION_DEG_PER_FRAME = 0.12
CAMERA_MOTION_MAX_LOG_SCALE_PER_FRAME = 0.003


def cam_to_world(point: np.ndarray) -> np.ndarray:
    """Map camera coordinates (right, down, forward) to Kinesia world axes."""
    return np.array([-point[2], point[0], -point[1]], dtype=np.float64)


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    """Return an xyxy bounding box for a non-empty boolean mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def circularity_from_mask(mask: np.ndarray, radius_px: float) -> float:
    """Return occupied-area / enclosing-circle-area, clipped to a confidence scale."""
    if radius_px <= 0:
        return 0.0
    value = float(mask.sum()) / (np.pi * radius_px * radius_px)
    return float(np.clip(value, 0.0, 1.0))


def sphere_measurement_from_mask(
    mask: np.ndarray,
    *,
    focal_px: float,
    image_width: int,
    image_height: int,
    diameter_m: float,
    detection_score: float,
) -> dict[str, Any] | None:
    """Lift one spherical mask into a metric camera/world point.

    Args:
        mask: Boolean segmentation mask for the sphere.
        focal_px: Calibrated/effective focal length in pixels.
        image_width: Source frame width.
        image_height: Source frame height.
        diameter_m: Known physical sphere diameter in metres.
        detection_score: SAM 3 instance confidence.

    Returns:
        A serializable measurement, or ``None`` when the mask cannot support a
        stable radius/depth estimate.
    """
    if (
        mask.ndim != 2
        or not np.any(mask)
        or focal_px <= 0
        or image_width <= 0
        or image_height <= 0
        or diameter_m <= 0
    ):
        return None

    import cv2

    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if len(points) < 12:
        return None
    (u, v), radius_px = cv2.minEnclosingCircle(points.reshape(-1, 1, 2))
    radius_px = float(radius_px)
    if not np.isfinite(radius_px) or radius_px < MIN_RADIUS_PX:
        return None

    radius_m = diameter_m / 2.0
    depth_m = float(focal_px * radius_m / radius_px)
    if not np.isfinite(depth_m) or depth_m <= 0:
        return None
    cam = np.array(
        [
            (float(u) - image_width / 2.0) * depth_m / focal_px,
            (float(v) - image_height / 2.0) * depth_m / focal_px,
            depth_m,
        ],
        dtype=np.float64,
    )
    circularity = circularity_from_mask(mask, radius_px)
    confidence = float(np.clip(float(detection_score) * circularity, 0.0, 1.0))
    return {
        "center_px": [float(u), float(v)],
        "radius_px": radius_px,
        "circularity": circularity,
        "confidence": confidence,
        "bbox_xyxy": mask_bbox(mask),
        "position_cam": [float(value) for value in cam],
        "position_world": [float(value) for value in cam_to_world(cam)],
    }


@dataclass
class SphereCandidate:
    """A 2D candidate used only to preserve the object identity over time."""

    mask: np.ndarray
    detector_score: float
    center_px: np.ndarray
    radius_px: float
    circularity: float

    @property
    def quality(self) -> float:
        """Combine detector and circle evidence into a 0..1 selection score."""
        return float(np.clip(self.detector_score * self.circularity, 0.0, 1.0))


@dataclass
class SphereTrackState:
    """Last reliable image-space state of one selected sphere."""

    frame_index: int
    center_px: np.ndarray
    radius_px: float
    velocity_px_per_frame: np.ndarray


def _candidate_from_mask(mask: np.ndarray, detector_score: float) -> SphereCandidate | None:
    """Describe a mask without committing it to the active track."""
    if mask.ndim != 2 or not np.any(mask):
        return None
    import cv2

    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    (u, v), radius_px = cv2.minEnclosingCircle(points.reshape(-1, 1, 2))
    radius_px = float(radius_px)
    if not np.isfinite(radius_px) or radius_px < MIN_RADIUS_PX:
        return None
    return SphereCandidate(
        mask=mask,
        detector_score=float(detector_score),
        center_px=np.array([float(u), float(v)], dtype=np.float64),
        radius_px=radius_px,
        circularity=circularity_from_mask(mask, radius_px),
    )


class SphereIdentityTracker:
    """Choose one spherical instance while tolerating fast but finite motion.

    This deliberately uses geometry and temporal continuity, not the human
    appearance/re-identification heuristics. It returns no measurement on an
    uncertain frame rather than silently switching to another ball.
    """

    def __init__(self) -> None:
        self._state: SphereTrackState | None = None

    def choose(
        self,
        candidates: Iterable[SphereCandidate],
        *,
        frame_index: int,
        image_width: int,
        image_height: int,
    ) -> SphereCandidate | None:
        """Select the candidate consistent with the previous observation."""
        available = [candidate for candidate in candidates if candidate.circularity >= 0.35]
        if not available:
            return None
        if self._state is None:
            selected = max(available, key=lambda candidate: candidate.quality)
            self._update(selected, frame_index)
            return selected

        state = self._state
        gap = max(1, int(frame_index - state.frame_index))
        predicted_center = state.center_px + state.velocity_px_per_frame * gap
        diagonal = max(float(np.hypot(image_width, image_height)), 1.0)
        best: tuple[float, SphereCandidate] | None = None
        for candidate in available:
            motion = float(np.linalg.norm(candidate.center_px - predicted_center)) / (diagonal * 0.35 * gap)
            scale_change = abs(float(np.log(max(candidate.radius_px, 1e-6) / max(state.radius_px, 1e-6))))
            # A strong detector can recover after a short occlusion, but a weak
            # far-away object must not steal an established track.
            continuity = max(0.0, 1.0 - motion - 0.25 * scale_change)
            score = 0.60 * candidate.quality + 0.40 * continuity
            if motion > 1.4 and candidate.quality < 0.80:
                continue
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            return None
        selected = best[1]
        self._update(selected, frame_index)
        return selected

    def _update(self, candidate: SphereCandidate, frame_index: int) -> None:
        """Update velocity with a conservative exponential blend."""
        if self._state is None:
            velocity = np.zeros(2, dtype=np.float64)
        else:
            gap = max(1, frame_index - self._state.frame_index)
            observed = (candidate.center_px - self._state.center_px) / gap
            velocity = 0.55 * self._state.velocity_px_per_frame + 0.45 * observed
        self._state = SphereTrackState(
            frame_index=frame_index,
            center_px=candidate.center_px.copy(),
            radius_px=candidate.radius_px,
            velocity_px_per_frame=velocity,
        )


def derive_velocities(poses: list[dict[str, Any]]) -> None:
    """Add finite-difference world velocities in place for observed poses."""
    for index, pose in enumerate(poses):
        before = poses[index - 1] if index > 0 else None
        after = poses[index + 1] if index + 1 < len(poses) else None
        if before is None and after is None:
            pose["velocity_world_m_s"] = None
            continue
        left = before or pose
        right = after or pose
        dt = float(right["time_s"]) - float(left["time_s"])
        if dt <= 1e-6:
            pose["velocity_world_m_s"] = None
            continue
        velocity = (
            (np.asarray(right["position_world"], dtype=np.float64)
             - np.asarray(left["position_world"], dtype=np.float64))
            / dt
        )
        pose["velocity_world_m_s"] = [float(value) for value in velocity]


def _focal_for_records(records: list[dict[str, Any]]) -> float | None:
    """Median valid focal length, used only when an individual record lacks one."""
    values = [float(record["focal_length"]) for record in records if record.get("focal_length")]
    return float(np.median(values)) if values else None


def effective_focal_from_subject_records(
    records: list[dict[str, Any]],
    image_height: int,
    floor_z_world: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Choose the camera scale that is consistent with the reconstructed body.

    A ball's known radius makes a textbook pinhole depth estimate possible, but
    the body model's stored focal/scale can carry a small, systematic bias. A
    static object already calibrates that relation from the subject's feet;
    use the same calibration here so a tracked ball and the subject occupy the
    same metric world rather than two nearly-compatible ones.

    Args:
        records: Per-frame reconstruction records from ``run_metadata.json``.
        image_height: Processed video height in pixels.
        floor_z_world: Optional run-level floor reference. When supplied it
            must be the same world floor used by the viewer and static objects.

    Returns:
        The effective focal length and serializable evidence describing whether
        it was calibrated from feet or fell back to the recorded focal.
    """
    recorded_focal = _focal_for_records(records)
    try:
        # Import lazily: static mesh reconstruction is optional, while these
        # lightweight calibration helpers share the exact floor convention.
        from .scene_objects import calibrate_floor_from_feet, floor_height

        floor_z = floor_z_world if floor_z_world is not None else floor_height(records)
        calibration = calibrate_floor_from_feet(records, image_height)
    except (ImportError, ValueError, TypeError):
        floor_z = None
        calibration = None

    if (
        calibration is not None
        and floor_z is not None
        and np.isfinite(floor_z)
        and abs(floor_z) > 1e-6
    ):
        v_times_z, spread = calibration
        focal = float(v_times_z / abs(floor_z))
        if np.isfinite(focal) and focal > 0:
            return focal, {
                "method": "subject_feet",
                "effective_focal_px": focal,
                "recorded_focal_px": recorded_focal,
                "floor_z_m": float(floor_z),
                "floor_calibration_spread": float(spread),
            }

    return recorded_focal, {
        "method": "recorded_focal",
        "effective_focal_px": recorded_focal,
        "recorded_focal_px": recorded_focal,
        "floor_z_m": None,
        "floor_calibration_spread": None,
    }


def classify_camera_motion_samples(
    samples: Iterable[tuple[float, float, float]],
) -> dict[str, Any]:
    """Classify global image motion from normalized affine-motion samples.

    Args:
        samples: ``(translation_diagonal_ratio_per_frame,
        rotation_deg_per_frame, abs_log_scale_per_frame)`` triples estimated
        from background feature tracks.

    Returns:
        A serializable fixed/moving/unverified assessment. ``unverified`` is
        deliberately not treated as proof of a moving camera: a textureless
        fixed-camera clip may have too few background features.
    """
    values = np.asarray(list(samples), dtype=np.float64)
    thresholds = {
        "translation_diagonal_ratio_per_frame": CAMERA_MOTION_MAX_TRANSLATION_DIAGONAL_RATIO_PER_FRAME,
        "rotation_deg_per_frame": CAMERA_MOTION_MAX_ROTATION_DEG_PER_FRAME,
        "abs_log_scale_per_frame": CAMERA_MOTION_MAX_LOG_SCALE_PER_FRAME,
    }
    if values.ndim != 2 or values.shape[1] != 3:
        return {
            "status": "unverified",
            "valid_pairs": int(len(values)) if values.ndim == 2 else 0,
            "thresholds": thresholds,
        }
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < CAMERA_MOTION_MIN_VALID_PAIRS:
        return {
            "status": "unverified",
            "valid_pairs": int(len(values)),
            "thresholds": thresholds,
        }

    median = np.median(values, axis=0)
    p95 = np.quantile(values, 0.95, axis=0)
    moving = bool(
        median[0] > CAMERA_MOTION_MAX_TRANSLATION_DIAGONAL_RATIO_PER_FRAME
        or median[1] > CAMERA_MOTION_MAX_ROTATION_DEG_PER_FRAME
        or median[2] > CAMERA_MOTION_MAX_LOG_SCALE_PER_FRAME
        # A large, isolated pan/zoom is still enough to break a trajectory,
        # but use a generous multiple so one noisy affine fit does not reject
        # an otherwise stable tripod clip.
        or p95[0] > CAMERA_MOTION_MAX_TRANSLATION_DIAGONAL_RATIO_PER_FRAME * 4.0
        or p95[1] > CAMERA_MOTION_MAX_ROTATION_DEG_PER_FRAME * 4.0
        or p95[2] > CAMERA_MOTION_MAX_LOG_SCALE_PER_FRAME * 4.0
    )
    return {
        "status": "moving" if moving else "fixed",
        "valid_pairs": int(len(values)),
        "median_translation_diagonal_ratio_per_frame": float(median[0]),
        "p95_translation_diagonal_ratio_per_frame": float(p95[0]),
        "median_rotation_deg_per_frame": float(median[1]),
        "p95_rotation_deg_per_frame": float(p95[1]),
        "median_abs_log_scale_per_frame": float(median[2]),
        "p95_abs_log_scale_per_frame": float(p95[2]),
        "thresholds": thresholds,
    }


def assess_camera_stability(video_path: Path) -> dict[str, Any]:
    """Estimate whether a video contains enough global motion to reject a sphere track.

    The estimate follows sparse background features with RANSAC similarity
    transforms. It catches pan, rotation and zoom, while a textureless clip is
    reported as ``unverified`` rather than falsely classified as moving.
    """
    import cv2

    capture = open_video(video_path)
    samples: list[tuple[float, float, float]] = []
    previous_gray: np.ndarray | None = None
    previous_frame = 0
    original_diagonal: float | None = None
    resize_scale = 1.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_stride = max(1, int(round(fps / 4.0)))
        if total_frames > 0:
            sample_stride = max(sample_stride, int(np.ceil(total_frames / CAMERA_MOTION_MAX_SAMPLES)))
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride != 0:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if original_diagonal is None:
                height, width = gray.shape[:2]
                original_diagonal = max(float(np.hypot(width, height)), 1.0)
                resize_scale = min(1.0, 640.0 / max(width, height))
            if resize_scale < 1.0:
                gray = cv2.resize(gray, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
            if previous_gray is not None:
                features = cv2.goodFeaturesToTrack(
                    previous_gray,
                    maxCorners=600,
                    qualityLevel=0.012,
                    minDistance=6,
                    blockSize=7,
                )
                if features is not None and len(features) >= 12:
                    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                        previous_gray,
                        gray,
                        features,
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                    )
                    if tracked is not None and status is not None:
                        keep = status.reshape(-1).astype(bool)
                        source = features.reshape(-1, 2)[keep]
                        destination = tracked.reshape(-1, 2)[keep]
                        if len(source) >= 12:
                            transform, inliers = cv2.estimateAffinePartial2D(
                                source,
                                destination,
                                method=cv2.RANSAC,
                                ransacReprojThreshold=2.0,
                                maxIters=2000,
                                confidence=0.99,
                            )
                            if transform is not None and inliers is not None:
                                inlier_count = int(inliers.reshape(-1).sum())
                                if inlier_count >= 12 and inlier_count / len(source) >= 0.45:
                                    frame_gap = max(1, frame_index - previous_frame)
                                    a, b, tx = transform[0]
                                    c, _d, ty = transform[1]
                                    scale = float(np.hypot(a, c))
                                    if np.isfinite(scale) and scale > 1e-6:
                                        translation = float(np.hypot(tx, ty)) / max(resize_scale, 1e-6)
                                        samples.append((
                                            translation / max(original_diagonal * frame_gap, 1e-6),
                                            abs(float(np.degrees(np.arctan2(c, a)))) / frame_gap,
                                            abs(float(np.log(scale))) / frame_gap,
                                        ))
            previous_gray = gray
            previous_frame = frame_index
            frame_index += 1
    finally:
        capture.release()
    return classify_camera_motion_samples(samples)


def track_dynamic_spheres(
    run_dir: Path,
    prompts: tuple[str, ...],
    video_path: Path,
    *,
    diameter_m: float,
    frame_stride: int = 1,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Track prompted known-diameter spheres and write time-indexed scene records.

    Args:
        run_dir: Completed human reconstruction directory.
        prompts: One or more sphere prompts such as ``("ball",)``.
        video_path: The exact processed video used by the run.
        diameter_m: Known physical diameter in metres.
        frame_stride: Process every Nth reconstructed frame.
        project_root: Optional project root used to locate SAM 3.
        log: Progress sink.

    Returns:
        Written object records, failures, and an optional skip reason.
    """
    del project_root  # The installed SAM 3 runtime resolves its own configured roots.
    if diameter_m <= 0 or not np.isfinite(diameter_m):
        raise ValueError("dynamic sphere diameter must be a positive number of metres")
    if not prompts:
        return {"objects": [], "failures": [], "skipped": "no dynamic objects requested"}
    if not video_path.exists():
        return {"objects": [], "failures": [], "skipped": "source video missing"}

    camera_motion = assess_camera_stability(video_path)
    if camera_motion["status"] == "moving":
        return {
            "objects": [],
            "failures": [],
            "skipped": "camera motion detected; a fixed camera is required for metric sphere tracking",
            "camera_motion": camera_motion,
        }
    if camera_motion["status"] == "unverified":
        log(f"{STAGE_MARKER} camera stability could not be verified; assuming a fixed camera")

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [record for record in metadata.get("records") or [] if isinstance(record, dict)]
    recorded_focal = _focal_for_records(records)
    if recorded_focal is None:
        return {"objects": [], "failures": [], "skipped": "no focal length from subject reconstruction"}
    width = int(metadata.get("video_width") or 0)
    height = int(metadata.get("video_height") or 0)
    fps = float(metadata.get("fps_output") or metadata.get("fps_input") or 30.0)
    if width <= 0 or height <= 0 or fps <= 0:
        return {"objects": [], "failures": [], "skipped": "invalid video metadata"}
    from .scene_objects import floor_height_for_run, object_name

    floor_z, floor_reference = floor_height_for_run(metadata, records)
    effective_focal, calibration = effective_focal_from_subject_records(
        records,
        height,
        floor_z_world=floor_z,
    )
    calibration["floor_reference"] = floor_reference
    if effective_focal is None:
        return {"objects": [], "failures": [], "skipped": "no usable focal length from subject reconstruction"}

    from .sam3d_runtime import select_device, try_build_human_detector
    from .subject_preview import DEFAULT_SAM3_CODE_ROOT

    records_by_video_frame = {
        int(record.get("video_frame", index)): (index, record)
        for index, record in enumerate(records)
    }
    if not records_by_video_frame:
        return {"objects": [], "failures": [], "skipped": "no reconstructed frames"}

    detector = try_build_human_detector(
        detector_name="sam3",
        device=select_device(),
        sam3_code_root=DEFAULT_SAM3_CODE_ROOT,
    )
    if detector is None:
        return {"objects": [], "failures": [], "skipped": "detector unavailable"}

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(frame_stride))
    output: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        for prompt_index, prompt in enumerate(prompts, start=1):
            log(f"{STAGE_MARKER} tracking dynamic sphere {prompt} ({prompt_index}/{len(prompts)})")
            tracker = SphereIdentityTracker()
            poses: list[dict[str, Any]] = []
            capture = open_video(video_path)
            try:
                video_frame = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    record_info = records_by_video_frame.get(video_frame)
                    if record_info is None:
                        video_frame += 1
                        continue
                    record_index, record = record_info
                    if record_index % stride != 0:
                        video_frame += 1
                        continue
                    # When feet permit calibration, a single effective focal
                    # makes all ball poses agree with the body's world scale.
                    # Otherwise retain the record-specific focal when present.
                    focal = (
                        effective_focal
                        if calibration["method"] == "subject_feet"
                        else float(record.get("focal_length") or recorded_focal)
                    )
                    instances = segment_object_instances(frame, prompt, detector)
                    candidates = [
                        candidate
                        for mask, score in instances
                        if (candidate := _candidate_from_mask(mask, score)) is not None
                    ]
                    selected = tracker.choose(
                        candidates,
                        frame_index=record_index,
                        image_width=width,
                        image_height=height,
                    )
                    if selected is not None:
                        measurement = sphere_measurement_from_mask(
                            selected.mask,
                            focal_px=focal,
                            image_width=width,
                            image_height=height,
                            diameter_m=diameter_m,
                            detection_score=selected.detector_score,
                        )
                        if measurement is not None:
                            poses.append({
                                "frame_index": record_index,
                                "video_frame": video_frame,
                                "time_s": float(video_frame / fps),
                                "detector_score": float(selected.detector_score),
                                **measurement,
                            })
                    video_frame += 1
            finally:
                capture.release()

            derive_velocities(poses)
            name = f"{object_name(prompt)}_dynamic"
            if not poses:
                failures.append({"prompt": prompt, "error": "no reliable spherical observations"})
                continue
            circularities = [float(pose["circularity"]) for pose in poses]
            confidences = [float(pose["confidence"]) for pose in poses]
            record = {
                "schema": SCHEMA,
                "kind": "sphere",
                "name": name,
                "prompt": prompt,
                "diameter_m": float(diameter_m),
                "radius_m": float(diameter_m / 2.0),
                "frame_stride": stride,
                "poses": poses,
                "camera_calibration": calibration,
                "camera_motion": camera_motion,
                "quality": {
                    "observed_frames": len(poses),
                    "coverage": float(len(poses) / max(1, len(records))),
                    "median_circularity": float(np.median(circularities)),
                    "median_confidence": float(np.median(confidences)),
                    # At minimum interpolate between adjacent sampled poses;
                    # otherwise an intentional CLI stride above the default
                    # gap makes a valid sphere blink between observations.
                    "max_interpolation_gap_frames": max(MAX_GAP_FOR_INTERPOLATION_FRAMES, stride),
                },
                "limitations": {
                    "camera": "fixed only",
                    "metric_scale": "known sphere diameter required",
                    "orientation": "not observable for an unmarked sphere",
                },
            }
            (scene_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
            output.append(record)
            log(f"{STAGE_MARKER} tracked {prompt}: {len(poses)} measurements "
                f"({record['quality']['coverage'] * 100:.0f}% coverage)")
    finally:
        del detector

    return {"objects": output, "failures": failures, "skipped": None}
