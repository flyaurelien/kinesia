"""One analysis on the GPU, run as a batch job that ends (and frees the GPU) when done.

    python -m kinesia.remote.job --run-dir /scratch/.../kinesia/runs/<id>

Stages are resumable: a stage whose output already exists is skipped, so a job
preempted on a shared GPU and restarted by the scheduler carries on. Progress
is printed as ``KINESIA {json}`` lines, which the local app reads from the pod
log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

EVENT_PREFIX = "KINESIA "


def emit(event: str, **fields) -> None:
    print(EVENT_PREFIX + json.dumps({"event": event, "t": round(time.time(), 1), **fields}), flush=True)


class Progress:
    """Throttled progress events for one stage."""

    def __init__(self, stage: str, every: float = 5.0):
        self.stage, self.every, self._last = stage, every, 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if done >= total or now - self._last >= self.every:
            self._last = now
            emit("progress", stage=self.stage, done=done, total=total)


def run(run_dir: Path, *, sam31_checkpoint: str, sam3_source: str, body_dir: str) -> None:
    import numpy as np

    request = json.loads((run_dir / "request.json").read_text())
    video = run_dir / "video.mp4"
    info = request["video"]
    width, height, frames = int(info["width"]), int(info["height"]), int(info["frames"])
    raw = run_dir / "raw"
    raw.mkdir(exist_ok=True)
    receipts: dict = {}
    receipts_path = raw / "receipts.json"
    if receipts_path.is_file():
        receipts = json.loads(receipts_path.read_text())

    import torch

    receipts["gpu"] = torch.cuda.get_device_name(0)
    receipts["torch"] = torch.__version__
    receipts["node"] = os.environ.get("NODE_NAME")

    def save_receipts() -> None:
        receipts_path.write_text(json.dumps(receipts, indent=1))

    # 1. People: SAM 3.1 video tracking.
    tracks_path = raw / "tracks.jsonl.gz"
    if not tracks_path.is_file():
        from .tracking import build_predictor, track_video

        emit("stage", stage="tracking", state="loading")
        predictor = build_predictor(
            sam31_checkpoint,
            sam3_source,
            max_objects=int(request.get("max_people", 64)),
        )
        emit("stage", stage="tracking", state="running", total=frames)
        result = track_video(
            predictor,
            video,
            tracks_path,
            prompt=request.get("prompt", "person"),
            frame_count=frames,
            width=width,
            height=height,
            header={"fps": info["fps"], "frames": frames, "max_people": request.get("max_people", 64)},
            on_frame=Progress("tracking"),
        )
        receipts["tracking"] = {**result, "model": "facebook/sam3.1 sam3.1_multiplex.pt"}
        save_receipts()
        del predictor
        torch.cuda.empty_cache()
        # SAM 3's constructors enter bf16 autocast "for the entire process" and
        # never leave it; nothing after tracking should run under it.
        torch.set_autocast_enabled("cuda", False)
        emit("stage", stage="tracking", state="done", **result)

    from ..tracks import read_tracks

    tracks = read_tracks(tracks_path)

    # 2. The (fixed) camera's focal length, which the body model needs.
    camera_path = raw / "camera.json"
    if not camera_path.is_file():
        from .lens import estimate_focal

        emit("stage", stage="camera", state="running")
        weights = os.environ.get("MOGE_WEIGHTS")
        focal, source, samples = estimate_focal(video, frames, width, height, Path(weights) if weights else None)
        K = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
        camera_path.write_text(json.dumps({"K": K, "source": source, "samples": samples}))
        receipts["camera"] = {"focal": round(focal, 1), "source": source, "samples": samples}
        save_receipts()
        emit("stage", stage="camera", state="done", focal=round(focal, 1), source=source)
    cam_int = np.asarray(json.loads(camera_path.read_text())["K"], dtype=np.float32)

    # 3. Bodies: SAM 3D Body for every tracked person.
    bodies_path = raw / "bodies.npz"
    if not bodies_path.is_file():
        from .bodies import load_model, merge_chunks, reconstruct_bodies

        emit("stage", stage="bodies", state="loading")
        estimator = load_model(body_dir)
        emit("stage", stage="bodies", state="running", total=frames)
        result = reconstruct_bodies(
            estimator,
            video,
            tracks.frames,
            raw / "bodies",
            width=width,
            height=height,
            cam_int=cam_int,
            min_height=float(request.get("min_person_height", 48)),
            on_frame=Progress("bodies"),
        )
        rows = merge_chunks(raw / "bodies", bodies_path)
        receipts["bodies"] = {**result, "rows": rows, "model": "facebook/sam-3d-body-dinov3"}
        save_receipts()
        del estimator
        torch.cuda.empty_cache()
        emit("stage", stage="bodies", state="done", **result)

    # 4. Package everything the local app downloads.
    emit("stage", stage="packing", state="running")
    package = run_dir / "out.tar"
    partial = run_dir / "out.tar.part"
    with tarfile.open(partial, "w") as archive:
        for name in ("tracks.jsonl.gz", "camera.json", "bodies.npz", "receipts.json"):
            path = raw / name
            if path.is_file():
                archive.add(path, arcname=f"raw/{name}")
    partial.replace(package)
    emit("done", package=package.name, size=package.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sam31-checkpoint", default=os.environ.get("SAM31_CHECKPOINT"))
    parser.add_argument("--sam3-source", default=os.environ.get("SAM3_SOURCE"))
    parser.add_argument("--body-dir", default=os.environ.get("SAM3D_BODY_DIR"))
    args = parser.parse_args()
    try:
        run(
            args.run_dir,
            sam31_checkpoint=args.sam31_checkpoint,
            sam3_source=args.sam3_source,
            body_dir=args.body_dir,
        )
    except Exception as error:
        emit("error", message=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
