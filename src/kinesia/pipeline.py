"""The life of one analysis: prepare the video, run the GPU stages, build the scene.

Every step is idempotent, so ``process`` can be re-run after a crash or a
cancel: finished steps are skipped, and the GPU job itself resumes stage by
stage.

By default the GPU stages run on this machine's CUDA GPU, in a child process
(``python -m kinesia.inference.job``). To run them somewhere else, write a
:class:`Runner` subclass, and point ``KINESIA_RUNNER`` at it as
``module:Class`` (with ``KINESIA_PLUGIN_PATH`` naming the folders to import it
from).
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .paths import project_root, run_dir
from .settings import inference_environment, load_local_env, model_files
from .video import normalize, probe, write_poster

EVENT_PREFIX = "KINESIA "
# SAM 3.1 keeps every decoded frame in host memory (~6 MB each) and its GPU
# state grows by ~19 MB per frame with 20 people: see "Capture advice".
MAX_FRAMES = 6000

DEFAULT_REQUEST = {
    "prompt": "person",  # measured: finds more people than "human", "people", "player" or "athlete"
    "max_people": 64,
    "min_person_height": 48,
}

STAGE_MESSAGES = {
    "tracking": "Tracking people",
    "camera": "Measuring the camera",
    "bodies": "Reconstructing bodies",
    "appearance": "Describing people",
    "packing": "Packaging results",
}

# What the GPU stages write to raw/; an analysis missing one (made by an older
# version) gets just that stage when processed again.
GPU_RESULTS = ("tracks.jsonl.gz", "camera.json", "bodies.npz", "appearance.npz")

# The steps the processing screen shows, as (state or GPU stage, label, detail).
LOCAL_STEPS = [
    {"key": "preparing", "label": "Prepare", "detail": "Normalize the video (frame rate, size, rotation)"},
    {"key": "tracking", "label": "Track people", "detail": "SAM 3.1 follows everyone through the video"},
    {"key": "camera", "label": "Camera", "detail": "Estimate the lens's focal length from the picture (MoGe-2)"},
    {"key": "bodies", "label": "Bodies", "detail": "SAM 3D Body reconstructs each person in 3D"},
    {"key": "appearance", "label": "Appearance", "detail": "DINOv3 describes each person, to tell people apart"},
    {"key": "building", "label": "3D scene", "detail": "Who is who, smooth motion, ground the feet"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status:
    """``status.json``: what the web app shows while an analysis runs."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"history": []}
        self.opened = time.time()
        self.reload()

    def reload(self) -> None:
        """Pick up changes written by another process (a cancel from the web app)."""
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass

    @property
    def cancelled(self) -> bool:
        """Whether a cancel arrived after this process started working on the run."""
        self.reload()
        return self.data.get("state") == "cancelled" and float(self.data.get("cancelled_at") or 0) >= self.opened

    def update(self, state: str | None = None, message: str | None = None, **fields) -> None:
        self.reload()
        if state not in (None, "cancelled") and self.cancelled:
            return  # a concurrent cancel wins over this process's progress reports
        if state == "cancelled":
            fields["cancelled_at"] = time.time()
        if state is not None and state != self.data.get("state"):
            self.data.setdefault("history", []).append({"t": now(), "state": state, "message": message})
            self.data["state"] = state
        if message is not None:
            self.data["message"] = message
        self.data.update(fields)
        self.data["updated_at"] = now()
        partial = self.path.with_name(self.path.name + ".part")
        partial.write_text(json.dumps(self.data, indent=1))
        partial.replace(self.path)

    def get(self, key: str, default=None):
        return self.data.get(key, default)


def parse_event(line: str) -> dict | None:
    index = line.find(EVENT_PREFIX)
    if index < 0:
        return None
    try:
        return json.loads(line[index + len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None


def parse_events(log: str) -> list[dict]:
    return [event for event in map(parse_event, log.splitlines()) if event is not None]


def prepare(folder: Path) -> dict:
    """Normalize the upload and write ``request.json``; return the request."""
    meta = json.loads((folder / "run.json").read_text())
    video = folder / "video.mp4"
    if not video.is_file():
        info = normalize(folder / meta["source"], video)
    else:
        info = probe(video, count_frames=True)
    if info.frames > MAX_FRAMES:
        raise ValueError(
            f"the video has {info.frames} frames after normalization; trim it to at most "
            f"{MAX_FRAMES} ({MAX_FRAMES / max(info.fps, 1):.0f} s at {info.fps:.0f} fps)"
        )
    if not (folder / "poster.jpg").is_file():
        write_poster(video, folder / "poster.jpg")
    request = {**DEFAULT_REQUEST, **meta.get("request", {}), "video": info.to_json()}
    (folder / "request.json").write_text(json.dumps(request, indent=1))
    meta["video"] = info.to_json()
    (folder / "run.json").write_text(json.dumps(meta, indent=1))
    return request


def is_process(pid: int, *words: str) -> bool:
    """Whether ``pid`` is alive and its command line contains every one of ``words``.

    Guards against signalling a process that reused the pid of a finished one.
    """
    if not pid:
        return False
    result = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    command = result.stdout.split()
    return result.returncode == 0 and all(any(word in part for part in command) for word in words)


def stop_process(pid: int, *words: str, timeout: float = 45) -> None:
    """SIGTERM ``pid`` if it is still the expected process, and wait for it to exit."""
    if pid == os.getpid() or not is_process(pid, *words):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and is_process(pid, *words):
        time.sleep(0.5)


class Runner:
    """Carries one analysis through its steps, reporting progress in ``status.json``."""

    steps: list[dict] = LOCAL_STEPS

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.folder = run_dir(run_id)
        self.status = Status(self.folder / "status.json")

    @classmethod
    def check(cls) -> dict:
        """``{"ok", "label", "detail"}``: whether analyses can run, for the app's status badge."""
        raise NotImplementedError

    def process(self) -> None:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def build_scene(self) -> None:
        from .scene.build import build_scene

        self.status.update("building", "Building the 3D scene")
        build_scene(self.folder)

    def stop_worker(self) -> None:
        """Stop this run's ``kinesia process`` worker (not this process)."""
        stop_process(int(self.status.get("pid") or 0), "kinesia", "process", self.run_id)


class LocalRunner(Runner):
    """The GPU stages on this machine's CUDA GPU."""

    job_module = "kinesia.inference.job"

    @classmethod
    def check(cls) -> dict:
        missing = [f"{name} ({path})" for name, path in model_files().items() if not path.exists()]
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu else 0.0
        except Exception as error:  # noqa: BLE001 - reported, not raised
            gpu, memory, missing = None, 0.0, [*missing, f"torch ({error})"]
        if gpu is None:
            return {"ok": False, "label": "No CUDA GPU", "detail": "The GPU stages need an NVIDIA GPU with CUDA."}
        if missing:
            return {"ok": False, "label": "Models missing", "detail": "Not found: " + ", ".join(missing)}
        return {"ok": True, "label": gpu.replace("NVIDIA ", ""), "detail": f"{gpu}, {memory:.0f} GB"}

    def process(self) -> None:
        self.status.update(
            None, pid=os.getpid(), error=None, steps=self.steps, location="this computer",
            started_at=self.status.get("started_at") or now(),
        )  # fmt: skip
        try:
            computed = False
            if not all((self.folder / "raw" / name).is_file() for name in GPU_RESULTS):
                self.status.update("preparing", "Preparing the video")
                prepare(self.folder)
                self._gpu_stages()
                if self.status.cancelled:
                    return
                computed = True
            if computed or not (self.folder / "scene" / "scene.json").is_file():
                self.build_scene()
            self.status.update("done", "Ready", finished_at=now(), progress=None, stage=None)
        except Exception as error:
            if not self.status.cancelled:
                self.status.update("failed", str(error).splitlines()[0][:300], error=str(error)[-4000:])
            raise

    def _gpu_stages(self) -> None:
        self.status.update("running", "Starting on the GPU", stage=None, progress=None, gpu_started_at=now())
        command = [sys.executable, "-u", "-m", self.job_module, "--run-dir", str(self.folder)]
        log_path = self.folder / "gpu.log"
        error = None
        with log_path.open("a") as log, subprocess.Popen(
            command, cwd=project_root(), env=inference_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as job:  # fmt: skip
            self.status.update(None, job_pid=job.pid)
            try:
                assert job.stdout is not None
                for line in job.stdout:
                    log.write(line)
                    event = parse_event(line)
                    if event is None:
                        continue
                    kind, stage = event.get("event"), event.get("stage")
                    if kind == "error":
                        error = event.get("message")
                    elif kind == "stage":
                        done = event.get("state") == "done"
                        self.status.update("running", STAGE_MESSAGES.get(stage, "Working on the GPU"), stage=stage,
                                           progress={"done": 1, "total": 1} if done else None)  # fmt: skip
                    elif kind == "progress":
                        self.status.update(None, stage=stage, progress={"done": event["done"], "total": event["total"]})
                    if self.status.cancelled:
                        break
                code = job.wait() if not self.status.cancelled else None
            finally:
                if job.poll() is None:  # this worker is stopping: never leave the GPU job behind
                    job.terminate()
                    try:
                        job.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        job.kill()
        if self.status.cancelled:
            return
        if code != 0:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(error or f"the GPU stages failed (exit {code}):\n{tail}")
        receipts = self.folder / "raw" / "receipts.json"
        self.status.update(
            None, gpu_finished_at=now(), job_pid=None,
            receipts=json.loads(receipts.read_text()) if receipts.is_file() else None,
        )  # fmt: skip

    def cancel(self) -> None:
        self.status.update("cancelled", "Cancelled")
        self.stop_worker()  # it stops its GPU job on the way out
        stop_process(int(self.status.get("job_pid") or 0), self.job_module, self.run_id)


def runner_class() -> type[Runner]:
    """The configured runner: ``KINESIA_RUNNER`` (``module:Class``), or the local GPU."""
    load_local_env()
    spec = os.environ.get("KINESIA_RUNNER", "").strip()
    if not spec:
        return LocalRunner
    for entry in filter(None, os.environ.get("KINESIA_PLUGIN_PATH", "").split(os.pathsep)):
        path = Path(entry).expanduser()
        path = path if path.is_absolute() else project_root() / path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module, _, name = spec.partition(":")
    runner = getattr(importlib.import_module(module), name)
    if not (isinstance(runner, type) and issubclass(runner, Runner)):
        raise TypeError(f"KINESIA_RUNNER={spec} is not a kinesia.pipeline.Runner subclass")
    return runner
