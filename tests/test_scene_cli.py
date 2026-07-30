"""Regression tests for scene-command failure signalling."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from sam_3d_pose_estimation.cli import cmd_scene


def scene_args(**overrides: object) -> Namespace:
    """Build the small command namespace consumed by ``cmd_scene``."""
    values: dict[str, object] = {
        "run_id": "fixture",
        "prompts": "chair",
        "stage": "shape",
        "video": None,
        "subject_track": None,
        "dynamic_sphere_diameter_m": 0.22,
        "dynamic_frame_stride": 1,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


class SceneCommandTests(unittest.TestCase):
    """Requested artifacts surface a failure without invalidating the human run."""

    def test_static_runtime_skip_returns_nonzero_for_the_job_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_metadata.json").write_text("{}")
            with (
                mock.patch("sam_3d_pose_estimation.workspace.run_dir", return_value=run_dir),
                mock.patch(
                    "sam_3d_pose_estimation.scene_objects.build_object_shapes",
                    return_value={"shapes": [], "failures": [], "skipped": "runtime unavailable"},
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = cmd_scene(scene_args())

        self.assertEqual(result, 1)

    def test_dynamic_mode_rejects_multiple_ambiguous_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_metadata.json").write_text("{}")
            with mock.patch("sam_3d_pose_estimation.workspace.run_dir", return_value=run_dir):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = cmd_scene(scene_args(
                        stage="dynamic",
                        prompts="ball, balloon",
                    ))

        self.assertEqual(result, 2)

    def test_dynamic_mode_rejects_a_non_positive_frame_stride(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_metadata.json").write_text("{}")
            with mock.patch("sam_3d_pose_estimation.workspace.run_dir", return_value=run_dir):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = cmd_scene(scene_args(
                        stage="dynamic",
                        prompts="ball",
                        dynamic_frame_stride=0,
                    ))

        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
