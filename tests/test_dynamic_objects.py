"""Regression tests for generic dynamic-object association."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sam_3d_pose_estimation.dynamic_objects import (
    ObjectIdentityTracker,
    mask_iou,
    track_dynamic_objects,
)


def rectangular_mask(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    width: int = 200,
    height: int = 160,
) -> np.ndarray:
    """Build an object-agnostic binary mask fixture."""
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class ObjectIdentityTests(unittest.TestCase):
    """Association stays with the spatially continuous prompted instance."""

    def test_mask_iou_is_category_independent(self) -> None:
        self.assertAlmostEqual(
            mask_iou(rectangular_mask(10, 10, 40, 40), rectangular_mask(25, 10, 55, 40)),
            0.3333333333333333,
        )

    def test_keeps_nearby_identity_over_higher_scoring_distractor(self) -> None:
        tracker = ObjectIdentityTracker()
        first = rectangular_mask(20, 50, 50, 80)
        self.assertIsNotNone(tracker.choose(
            [(first, 0.70)], frame_index=0, image_width=200, image_height=160,
        ))

        nearby = rectangular_mask(26, 50, 56, 80)
        distant = rectangular_mask(155, 105, 190, 145)
        selected = tracker.choose(
            [(distant, 0.98), (nearby, 0.68)],
            frame_index=1,
            image_width=200,
            image_height=160,
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertTrue(np.array_equal(selected[0], nearby))


class DynamicObjectArtifactTests(unittest.TestCase):
    """Artifacts retain SAM 3D pose fields without geometric substitutions."""

    def test_writes_model_derived_transforms_for_each_sampled_frame(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.index = 0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.index >= 2:
                    return False, None
                self.index += 1
                return True, np.zeros((80, 100, 3), dtype=np.uint8)

            def release(self) -> None:
                return None

        pose = {
            "translation_l2c": [1.0, 2.0, 3.0],
            "rotation_quaternion_wxyz_l2c": [1.0, 0.0, 0.0, 0.0],
            "scale_l2c": [2.0, 3.0, 4.0],
        }

        def write_first_mesh(
            _image: Path, _mask: Path, output_glb: Path, output_pose: Path, _cache: Path, **_kwargs: object,
        ) -> Path:
            output_glb.write_bytes(b"glb")
            output_pose.write_text(json.dumps(pose))
            return output_glb

        def write_pose(_image: Path, _mask: Path, output_pose: Path, _cache: Path, **_kwargs: object) -> Path:
            output_pose.write_text(json.dumps(pose))
            return output_pose

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            video = run_dir / "input.mp4"
            video.write_bytes(b"video")
            (run_dir / "run_metadata.json").write_text(json.dumps({
                "video_width": 100,
                "video_height": 80,
                "fps_output": 30,
                "records": [{"video_frame": 0}, {"video_frame": 1}],
            }))
            mask = rectangular_mask(20, 20, 40, 40, width=100, height=80)
            with (
                mock.patch("sam_3d_pose_estimation.dynamic_objects.open_video", return_value=Capture()),
                mock.patch("sam_3d_pose_estimation.dynamic_objects.segment_object_instances", return_value=[(mask, 0.8)]),
                mock.patch("sam_3d_pose_estimation.dynamic_objects.reconstruct_mesh", side_effect=write_first_mesh),
                mock.patch("sam_3d_pose_estimation.dynamic_objects.reconstruct_pose", side_effect=write_pose),
                mock.patch("sam_3d_pose_estimation.sam3d_runtime.try_build_human_detector", return_value=object()),
            ):
                result = track_dynamic_objects(run_dir, ("toy",), video, frame_stride=1, log=lambda _line: None)

        self.assertEqual(result["failures"], [])
        poses = result["objects"][0]["poses"]
        self.assertEqual(len(poses), 2)
        self.assertEqual(poses[0]["model_pose"], pose)
        self.assertEqual(poses[0]["object_to_world"][0], [0.0, 0.0, -4.0, -3.0])


if __name__ == "__main__":
    unittest.main()
