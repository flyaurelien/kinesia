"""Tests for local body-model path resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sam_3d_pose_estimation.cli import absolute_model_path, resolve_body_model_root


def create_complete_model(root: Path) -> None:
    """Create the three files that define a usable local model snapshot."""
    (root / "assets").mkdir(parents=True)
    (root / "model_config.yaml").touch()
    (root / "model.ckpt").touch()
    (root / "assets" / "mhr_model.pt").touch()


class BodyModelPathTests(unittest.TestCase):
    """A cached complete snapshot can replace a missing project-local copy."""

    def test_prefers_complete_project_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preferred = root / "preferred"
            cached = (
                root
                / "hub"
                / "models--facebook--sam-3d-body-dinov3"
                / "snapshots"
                / "snapshot"
            )
            create_complete_model(preferred)
            create_complete_model(cached)

            resolved = resolve_body_model_root(preferred, root / "hub")

        self.assertEqual(resolved, preferred)

    def test_uses_complete_cached_snapshot_when_preferred_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preferred = root / "preferred"
            cached = (
                root
                / "hub"
                / "models--facebook--sam-3d-body-dinov3"
                / "snapshots"
                / "snapshot"
            )
            create_complete_model(cached)

            resolved = resolve_body_model_root(preferred, root / "hub")

        self.assertEqual(resolved, cached)

    def test_ignores_incomplete_cached_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preferred = root / "preferred"
            incomplete = (
                root
                / "hub"
                / "models--facebook--sam-3d-body-dinov3"
                / "snapshots"
                / "snapshot"
            )
            incomplete.mkdir(parents=True)
            (incomplete / "model.ckpt").touch()

            resolved = resolve_body_model_root(preferred, root / "hub")

        self.assertEqual(resolved, preferred)

    def test_absolute_model_path_preserves_snapshot_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            blob = root / "blobs" / "checkpoint"
            snapshot.mkdir()
            blob.parent.mkdir()
            blob.touch()
            checkpoint = snapshot / "model.ckpt"
            checkpoint.symlink_to(blob)

            absolute = absolute_model_path(checkpoint)

        self.assertEqual(absolute, checkpoint)
        self.assertNotEqual(absolute, blob)


if __name__ == "__main__":
    unittest.main()
