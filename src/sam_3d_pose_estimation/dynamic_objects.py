"""Reconstruct generic moving scene objects from per-frame SAM model poses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .object_masks import (
    build_sam3_processor,
    segment_prompt_instances,
    select_mask_for_reference,
    write_mask,
)
from .object_shapes import objects_root, reconstruct_mesh, reconstruct_pose
from .video_io import open_video

SCHEMA = "kinesia.dynamic_object.v1"
STAGE_MARKER = "[scene]"
DEFAULT_FRAME_STRIDE = 1


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection-over-union for two binary masks."""
    union = int(np.logical_or(left, right).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def mask_center(mask: np.ndarray) -> np.ndarray | None:
    """Return the image-space centroid of a non-empty binary mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


@dataclass
class ObjectTrackState:
    """The latest mask and centre used to retain one generic object identity."""

    frame_index: int
    mask: np.ndarray
    center_px: np.ndarray
    velocity_px_per_frame: np.ndarray


class ObjectIdentityTracker:
    """Select an instance using generic mask overlap and image continuity.

    This is association only: it has no object-category rules and does not
    infer scale, orientation, or position. Those values come directly from
    SAM 3D Objects for every accepted frame.
    """

    def __init__(self) -> None:
        self._state: ObjectTrackState | None = None

    def choose(
        self,
        candidates: Iterable[tuple[np.ndarray, float]],
        *,
        frame_index: int,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, float] | None:
        """Choose the prompt instance most consistent with the active track."""
        available = [
            (mask, float(score), center)
            for mask, score in candidates
            if mask.ndim == 2 and (center := mask_center(mask)) is not None
        ]
        if not available:
            return None
        if self._state is None:
            selected = max(available, key=lambda item: item[1])
            self._update(selected[0], selected[2], frame_index)
            return selected[0], selected[1]

        state = self._state
        gap = max(1, frame_index - state.frame_index)
        predicted = state.center_px + state.velocity_px_per_frame * gap
        diagonal = max(float(np.hypot(image_width, image_height)), 1.0)
        selected = max(
            available,
            key=lambda item: (
                0.55 * mask_iou(state.mask, item[0])
                + 0.25 * max(0.0, 1.0 - float(np.linalg.norm(item[2] - predicted)) / (diagonal * 0.45 * gap))
                + 0.20 * item[1]
            ),
        )
        self._update(selected[0], selected[2], frame_index)
        return selected[0], selected[1]

    def _update(self, mask: np.ndarray, center_px: np.ndarray, frame_index: int) -> None:
        """Update image velocity without changing the model-derived pose."""
        if self._state is None:
            velocity = np.zeros(2, dtype=np.float64)
        else:
            gap = max(1, frame_index - self._state.frame_index)
            observed = (center_px - self._state.center_px) / gap
            velocity = 0.5 * self._state.velocity_px_per_frame + 0.5 * observed
        self._state = ObjectTrackState(
            frame_index=frame_index,
            mask=mask.copy(),
            center_px=center_px.copy(),
            velocity_px_per_frame=velocity,
        )


def _frame_paths(scene_dir: Path, name: str, frame_index: int) -> tuple[Path, Path, Path]:
    """Return isolated image, mask, and pose paths for one model invocation."""
    cache_dir = scene_dir / ".dynamic" / name
    stem = f"frame_{frame_index:06d}"
    return cache_dir / f"{stem}.png", cache_dir / f"{stem}_mask.png", cache_dir / f"{stem}_pose.json"


def _write_frame_image(frame: np.ndarray, path: Path) -> None:
    """Persist one BGR frame for the external model runtime."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write frame image: {path}")


def track_dynamic_objects(
    run_dir: Path,
    prompts: tuple[str, ...],
    video_path: Path,
    *,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build one mesh and sample generic SAM 3D poses for each moving prompt.

    Every accepted image/mask pair is sent to SAM 3D Objects. Its local-to-
    camera translation, rotation, and scale are converted with the same fixed
    camera/world convention as static objects; no category-specific geometry
    or physical-size rule is applied.
    """
    if not prompts:
        return {"objects": [], "failures": [], "skipped": "no dynamic objects requested"}
    if not video_path.exists():
        return {"objects": [], "failures": [], "skipped": "source video missing"}
    if frame_stride < 1:
        raise ValueError("dynamic frame stride must be at least 1")

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [record for record in metadata.get("records") or [] if isinstance(record, dict)]
    width = int(metadata.get("video_width") or 0)
    height = int(metadata.get("video_height") or 0)
    fps = float(metadata.get("fps_output") or metadata.get("fps_input") or 30.0)
    if width <= 0 or height <= 0 or fps <= 0:
        return {"objects": [], "failures": [], "skipped": "invalid video metadata"}
    records_by_video_frame = {
        int(record.get("video_frame", index)): (index, record)
        for index, record in enumerate(records)
    }
    if not records_by_video_frame:
        return {"objects": [], "failures": [], "skipped": "no reconstructed frames"}

    from .scene_objects import (
        _subject_on_frame,
        _subject_mesh_world_on_frame,
        _subject_prompts_for_run,
        align_body_to_scene_pointmap,
        apply_scene_alignment_to_model_pose,
        object_name,
    )

    processor = build_sam3_processor(confidence=0.5)
    subject_prompts = _subject_prompts_for_run(run_dir)

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    alignments_by_frame: dict[int, dict[str, Any]] = {}
    try:
        for prompt_index, prompt in enumerate(prompts, start=1):
            name = f"{object_name(prompt)}_dynamic"
            log(f"{STAGE_MARKER} reconstructing dynamic object {prompt} ({prompt_index}/{len(prompts)})")
            tracker = ObjectIdentityTracker()
            poses: list[dict[str, Any]] = []
            reference_mesh = scene_dir / f"{name}.glb"
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
                    frame_index, _record = record_info
                    if frame_index % frame_stride != 0:
                        video_frame += 1
                        continue
                    reference_subject = _subject_on_frame(run_dir, video_frame)
                    if reference_subject is None:
                        video_frame += 1
                        continue
                    segmented = segment_prompt_instances(
                        frame,
                        (prompt, *subject_prompts),
                        processor,
                    )
                    selected_subject = select_mask_for_reference(
                        (
                            instance
                            for subject_prompt in subject_prompts
                            for instance in segmented.get(subject_prompt, [])
                        ),
                        reference_subject,
                    )
                    if selected_subject is None:
                        video_frame += 1
                        continue
                    subject_mask, _subject_score = selected_subject
                    selected = tracker.choose(
                        segmented.get(prompt, []),
                        frame_index=frame_index,
                        image_width=width,
                        image_height=height,
                    )
                    if selected is None:
                        video_frame += 1
                        continue
                    mask, score = selected
                    image_path, mask_path, pose_path = _frame_paths(scene_dir, name, frame_index)
                    _write_frame_image(frame, image_path)
                    write_mask(mask, mask_path)
                    cache_dir = pose_path.parent / "cache"
                    shared_pointmap = (
                        scene_dir
                        / ".dynamic"
                        / "shared"
                        / f"frame_{video_frame:06d}_pointmap.npz"
                    )
                    if not reference_mesh.exists():
                        reconstruct_mesh(
                            image_path, mask_path, reference_mesh, pose_path, cache_dir,
                            pointmap_path=shared_pointmap,
                            root=objects_root(project_root), log=log,
                        )
                    else:
                        reconstruct_pose(
                            image_path, mask_path, pose_path, cache_dir,
                            pointmap_path=shared_pointmap,
                            root=objects_root(project_root), log=log,
                        )
                    model_pose = json.loads(pose_path.read_text())
                    pointmap_name = model_pose.get("scene_pointmap")
                    if not pointmap_name:
                        raise ValueError(
                            f"frame {video_frame} has no shared scene point map"
                        )
                    pointmap_path = Path(str(pointmap_name))
                    if not pointmap_path.is_absolute():
                        pointmap_path = pose_path.parent / pointmap_path
                    if not pointmap_path.is_file():
                        raise ValueError(
                            f"frame {video_frame} scene point map is missing"
                        )
                    subject_mesh_world, focal = _subject_mesh_world_on_frame(
                        run_dir, video_frame
                    )
                    if subject_mesh_world is None or focal is None:
                        raise ValueError(
                            f"frame {video_frame} has no metric subject mesh"
                        )
                    with np.load(pointmap_path) as pointmap_file:
                        pointmap = np.asarray(
                            pointmap_file["pointmap"], dtype=np.float64
                        )
                    alignment_evidence = alignments_by_frame.get(video_frame)
                    if alignment_evidence is None:
                        alignment_evidence = align_body_to_scene_pointmap(
                            pointmap,
                            subject_mesh_world,
                            focal,
                            subject_mask,
                            width,
                            height,
                        )
                        alignment_evidence["source_frame"] = video_frame
                        alignments_by_frame[video_frame] = alignment_evidence
                    scene_alignment = alignment_evidence["scene_to_body"]
                    model_pose.pop("scene_pointmap", None)
                    poses.append({
                        "frame_index": frame_index,
                        "video_frame": video_frame,
                        "time_s": float(video_frame / fps),
                        "detector_score": score,
                        "model_pose": model_pose,
                        "body_object_alignment": alignment_evidence,
                        "object_to_world": apply_scene_alignment_to_model_pose(
                            model_pose, scene_alignment
                        ).tolist(),
                    })
                    video_frame += 1
            finally:
                capture.release()

            if not poses:
                failures.append({"prompt": prompt, "error": "no pose could be reconstructed"})
                continue
            record = {
                "schema": SCHEMA,
                "kind": "mesh",
                "name": name,
                "prompt": prompt,
                "mesh": reference_mesh.name,
                "frame_stride": frame_stride,
                "poses": poses,
                "quality": {
                    "observed_frames": len(poses),
                    "coverage": float(len(poses) / max(1, len(records))),
                    # A missing model pose is hidden rather than extrapolated.
                    # With the default stride of one, the viewer never fills a
                    # multi-frame gap with an invented trajectory.
                    "max_interpolation_gap_frames": frame_stride,
                    "source": "sam3d_objects_model_pose_shared_subject_pointmap",
                    "metric_alignment": "official_body_to_moge_height_center",
                },
            }
            (scene_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
            output.append(record)
            log(f"{STAGE_MARKER} reconstructed {prompt}: {len(poses)} model poses")
    finally:
        del processor
        shared_pointmap_dir = scene_dir / ".dynamic" / "shared"
        if shared_pointmap_dir.is_dir():
            for pointmap_path in shared_pointmap_dir.glob("*_pointmap.npz"):
                pointmap_path.unlink()
    return {"objects": output, "failures": failures, "skipped": None}
