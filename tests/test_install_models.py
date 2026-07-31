"""Tests for project-local model materialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import install_models


class InstallModelsTests(unittest.TestCase):
    """Cached model snapshots become independent project-local files."""

    def test_copy_cached_snapshot_dereferences_blob_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / "blobs" / "weights"
            blob.parent.mkdir()
            blob.write_bytes(b"model-weights")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.ckpt").symlink_to(blob)
            destination = root / "local"

            install_models.copy_cached_snapshot(
                snapshot,
                destination,
                ("model.ckpt",),
            )

            local_file = destination / "model.ckpt"
            self.assertEqual(local_file.read_bytes(), b"model-weights")
            self.assertFalse(local_file.is_symlink())

    def test_cached_snapshot_ignores_incomplete_newer_revision(self) -> None:
        spec = install_models.ModelSnapshot(
            repo_id="publisher/model",
            directory="model",
            allow_patterns=("model.ckpt",),
            required_files=("model.ckpt",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            snapshots = cache_root / "models--publisher--model" / "snapshots"
            complete = snapshots / "complete"
            incomplete = snapshots / "incomplete"
            complete.mkdir(parents=True)
            (complete / "model.ckpt").touch()
            incomplete.mkdir()

            with mock.patch.object(install_models, "HF_HUB_CACHE", str(cache_root)):
                resolved = install_models.cached_snapshot(spec)

        self.assertEqual(resolved, complete)


if __name__ == "__main__":
    unittest.main()
