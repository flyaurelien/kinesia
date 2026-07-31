"""Tests for the tracked SAM 3D Objects wrapper."""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_sam3d_objects_pose


class ObjectPoseRunnerTests(unittest.TestCase):
    """The wrapper exports only the GLB artifact consumed by Kinesia."""

    def test_full_reconstruction_exports_model_glb_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            mask = root / "mask.png"
            image.touch()
            mask.touch()
            output = root / "object.glb"
            args = argparse.Namespace(
                image=image,
                mask=mask,
                output=output,
                cache_dir=root / "cache",
                simplify=0.9,
            )
            mesh = mock.Mock()
            pipeline = mock.Mock()
            pipeline.run.return_value = {"glb": mesh, "translation": [0, 0, 1]}
            upstream = types.ModuleType("main")
            upstream.load_image = mock.Mock(return_value="image")
            upstream.load_mask_from_file = mock.Mock(return_value="mask")

            with (
                mock.patch.object(run_sam3d_objects_pose, "build_pipeline", return_value=pipeline),
                mock.patch.dict(sys.modules, {"main": upstream}),
            ):
                result = run_sam3d_objects_pose.run_full_reconstruction(args)

        self.assertIs(result["glb"], mesh)
        mesh.export.assert_called_once_with(str(output.resolve()))
        pipeline.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
