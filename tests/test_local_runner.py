"""The local runner: progress from the GPU job, failures, and cancelling mid-job."""

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from kinesia import pipeline
from kinesia.pipeline import LocalRunner, Status

FAKE_JOB = textwrap.dedent('''
    """Stands in for kinesia.inference.job: prints the same events."""
    import json, os, sys, time
    from pathlib import Path

    def emit(event, **fields):
        print("KINESIA " + json.dumps({"event": event, **fields}), flush=True)

    run = Path(sys.argv[sys.argv.index("--run-dir") + 1])
    mode = os.environ["FAKE_MODE"]
    Path(os.environ["FAKE_PID_FILE"]).write_text(str(os.getpid()))
    print("loading models...", flush=True)
    emit("stage", stage="tracking", state="running", total=10)
    if mode == "slow":
        for i in range(600):
            emit("progress", stage="tracking", done=i, total=600)
            time.sleep(0.05)
    if mode == "fail":
        emit("error", message="RuntimeError: SAM 3.1 video tracking needs a CUDA GPU")
        sys.exit(1)
    emit("progress", stage="tracking", done=10, total=10)
    emit("stage", stage="tracking", state="done")
    (run / "raw").mkdir(exist_ok=True)
    (run / "raw" / "receipts.json").write_text(json.dumps({"gpu": "Fake GPU"}))
    (run / "raw" / "bodies.npz").write_bytes(b"")
    emit("done")
''')


class LocalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        (self.tmp / "fake_gpu_job.py").write_text(FAKE_JOB)
        self.pid_file = self.tmp / "job.pid"
        env = {
            "KINESIA_RUNS_ROOT": str(self.tmp / "runs"),
            "PYTHONPATH": os.pathsep.join([str(self.tmp), *sys.path]),
            "FAKE_PID_FILE": str(self.pid_file),
        }
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        for target, value in ((pipeline, "prepare"), (pipeline.Runner, "build_scene")):
            p = mock.patch.object(target, value)
            p.start()
            self.addCleanup(p.stop)
        self.folder = self.tmp / "runs" / "clip-20260101-120000"
        self.folder.mkdir(parents=True)
        (self.folder / "status.json").write_text(json.dumps({"state": "new", "history": []}))

    def runner(self, mode: str) -> LocalRunner:
        os.environ["FAKE_MODE"] = mode
        runner = LocalRunner(self.folder.name)
        runner.job_module = "fake_gpu_job"
        return runner

    def state(self) -> dict:
        return json.loads((self.folder / "status.json").read_text())

    def test_a_finished_job_leaves_a_done_run_with_its_receipts(self):
        self.runner("ok").process()
        status = self.state()
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["receipts"], {"gpu": "Fake GPU"})
        self.assertEqual([s["key"] for s in status["steps"]][0], "preparing")
        self.assertIn("running", [h["state"] for h in status["history"]])
        self.assertIn("loading models...", (self.folder / "gpu.log").read_text())

    def test_a_failing_job_reports_its_own_error(self):
        with self.assertRaises(RuntimeError):
            self.runner("fail").process()
        status = self.state()
        self.assertEqual(status["state"], "failed")
        self.assertIn("needs a CUDA GPU", status["error"])

    def test_a_cancel_stops_the_gpu_job(self):
        runner = self.runner("slow")
        worker = threading.Thread(target=runner.process)
        worker.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (self.state().get("progress") or {}).get("done", 0) < 3:
            time.sleep(0.05)
        Status(self.folder / "status.json").update("cancelled", "Cancelled")
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.state()["state"], "cancelled")
        pid = int(self.pid_file.read_text())
        time.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
