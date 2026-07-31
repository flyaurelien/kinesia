"""Tests for the external object-runtime process boundary."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from sam_3d_pose_estimation.object_shapes import (
    _display_runtime_line,
    _run_runtime,
    _runtime_environment,
)


class ObjectRuntimeTests(unittest.TestCase):
    """The slow object process must remain observable and bounded."""

    def test_runtime_streams_child_output(self) -> None:
        events: list[tuple[str, float]] = []
        started_at = time.monotonic()
        with tempfile.TemporaryDirectory() as temporary:
            completed = _run_runtime(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('first'); time.sleep(0.5); print('second')",
                ],
                environment=dict(os.environ),
                cwd=Path(temporary),
                timeout_s=2,
                activity="testing runtime",
                log=lambda line: events.append((line, time.monotonic() - started_at)),
            )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual([line for line, _ in events], ["  first", "  second"])
        # If output were still captured until process exit, both callbacks
        # would arrive together instead of spanning the child's sleep.
        self.assertGreater(events[1][1] - events[0][1], 0.3)
        self.assertIn("first\nsecond", completed.stdout)

    def test_runtime_environment_forces_unbuffered_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = _runtime_environment(Path(temporary))

        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")

    def test_model_tensor_debug_is_not_forwarded_to_the_ui(self) -> None:
        self.assertIsNone(_display_runtime_line(
            "[DEBUG SLAT] x.feats=tensor([[1, 2, 3]], device='mps:0')",
            "reconstructing chair",
        ))
        self.assertEqual(
            _display_runtime_line(
                "2026-01-01 | INFO | stage complete", "reconstructing chair"
            ),
            "2026-01-01 | INFO | stage complete",
        )

    def test_runtime_timeout_terminates_quiet_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                _run_runtime(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    environment=dict(os.environ),
                    cwd=Path(temporary),
                    timeout_s=0.1,
                    activity="testing timeout",
                    log=lambda _line: None,
                )


if __name__ == "__main__":
    unittest.main()
