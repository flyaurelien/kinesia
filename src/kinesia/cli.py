"""``kinesia`` command line: create, process and inspect analyses.

    kinesia new input/match.mp4 --name "Sunday match"   # prints the run id
    kinesia process <run-id>                           # GPU stages, then the 3D scene
    kinesia cancel <run-id>                            # stop it (the GPU is freed)
    kinesia scene <run-id>                             # rebuild the 3D scene
    kinesia appearance <run-id> [--device mps]         # describe people (older analyses), rebuild
    kinesia doctor [--json]                            # can analyses run here?
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .paths import run_dir, runs_root
from .settings import load_local_env


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:48] or "video"


def new_run(source: Path, name: str | None = None, request: dict | None = None) -> str:
    """Create an analysis folder holding a copy of ``source``; return its id."""
    if not source.is_file():
        raise FileNotFoundError(source)
    label = name or source.stem.replace("_", " ").replace("-", " ").strip() or "Video"
    run_id = f"{slug(label)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder = run_dir(run_id)
    folder.mkdir(parents=True)
    target = folder / f"source{source.suffix.lower() or '.mp4'}"
    shutil.copy2(source, target)
    meta = {
        "id": run_id,
        "name": label,
        "source": target.name,
        "original_name": source.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": request or {},
    }
    (folder / "run.json").write_text(json.dumps(meta, indent=1))
    (folder / "status.json").write_text(json.dumps({"state": "new", "history": []}))
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinesia", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("new", help="create an analysis from a video file")
    create.add_argument("video", type=Path)
    create.add_argument("--name")
    create.add_argument("--max-people", type=int)

    for command in ("process", "cancel", "status", "scene"):
        sub.add_parser(command).add_argument("run_id")

    describe = sub.add_parser("appearance", help="describe the people of an older analysis here, then rebuild its scene")
    describe.add_argument("run_id")
    describe.add_argument("--device", default="auto", help="cuda, mps or cpu (default: the best one available)")

    doctor = sub.add_parser("doctor", help="check the GPU and the model files")
    doctor.add_argument("--json", action="store_true")

    sub.add_parser("list", help="list analyses")

    args = parser.parse_args(argv)
    load_local_env()

    if args.command == "new":
        request = {"max_people": args.max_people} if args.max_people else {}
        print(new_run(args.video, args.name, request))
        return 0
    if args.command == "list":
        for folder in sorted(runs_root().glob("*/run.json")):
            status = folder.parent / "status.json"
            state = json.loads(status.read_text()).get("state") if status.is_file() else "?"
            print(f"{folder.parent.name}\t{state}")
        return 0
    if args.command == "status":
        print((run_dir(args.run_id) / "status.json").read_text())
        return 0
    if args.command == "scene":
        from .scene.build import build_scene

        build_scene(run_dir(args.run_id))
        return 0
    if args.command == "appearance":
        import torch

        from .inference.appearance import HUB, describe_run, load_model
        from .paths import models_root
        from .scene.build import build_scene
        from .settings import model_files

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = load_model(model_files()["DINOV3_WEIGHTS"], device, models_root() / "torch" / "hub" / HUB)
        print(describe_run(model, run_dir(args.run_id), device=device))
        del model
        build_scene(run_dir(args.run_id))
        return 0
    if args.command in {"process", "cancel"}:
        import signal

        from .pipeline import runner_class

        # Stopping the worker must still run its cleanup (it stops the GPU job).
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

        runner = runner_class()(args.run_id)
        if args.command == "cancel":
            runner.cancel()
        else:
            runner.process()
        return 0
    if args.command == "doctor":
        from .pipeline import runner_class

        report = runner_class().check()
        print(json.dumps(report) if args.json else f"{'ok' if report['ok'] else 'not ready'}: {report['label']} ({report['detail']})")
        return 0 if report["ok"] else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
