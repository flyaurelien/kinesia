"""Regression tests for runtime paths outside an editable installation."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from sam_3d_pose_estimation.sam3d_runtime import default_sam3d_code_root


class RuntimePathTests(unittest.TestCase):
    """External runtimes resolve from the project, not site-packages."""

    def test_default_runtime_root_uses_the_active_project(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "sam_3d_pose_estimation.sam3d_runtime.project_root_from",
                return_value=Path("/workspace/kinesia"),
            ),
        ):
            self.assertEqual(
                default_sam3d_code_root(),
                Path("/workspace/kinesia/vendor/sam-3d-body-main"),
            )

    def test_environment_runtime_root_takes_precedence(self) -> None:
        with mock.patch.dict(os.environ, {"SAM3D_CODE_ROOT": "/models/body"}, clear=True):
            self.assertEqual(default_sam3d_code_root(), Path("/models/body"))


if __name__ == "__main__":
    unittest.main()
