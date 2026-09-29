"""``kinesia`` command line: create, process and inspect analyses.

    kinesia new input/match.mp4 --name "Sunday match"   # prints the run id
    kinesia process <run-id>                           # cluster GPU, then the 3D scene
    kinesia cancel <run-id>                            # deletes the job, frees the GPU
    kinesia scene <run-id>                             # rebuild the 3D scene locally
    kinesia cluster check | setup
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
    create.add_argument("--prompt")

    for command in ("process", "cancel", "status", "scene"):
        sub.add_parser(command).add_argument("run_id")

    cluster = sub.add_parser("cluster", help="cluster access and one-time setup")
    cluster.add_argument("action", choices=["check", "setup"])

    sub.add_parser("list", help="list analyses")

    args = parser.parse_args(argv)

    if args.command == "new":
        request = {k: v for k, v in {"max_people": args.max_people, "prompt": args.prompt}.items() if v}
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
    if args.command in {"process", "cancel"}:
        import signal

        from .cluster.orchestrate import Orchestrator

        # Stopping the worker must still run its cleanup (it deletes transfer pods).
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

        orchestrator = Orchestrator(args.run_id)
        if args.command == "cancel":
            orchestrator.cancel()
        else:
            orchestrator.process()
        return 0
    if args.command == "cluster":
        from .cluster.config import load_config
        from .cluster.kube import Cluster
        from .cluster.remote import setup
        from .cluster.transfer import io_pod

        cluster_ = Cluster(load_config())
        print(f"the job scheduler user: {cluster_.check_access()}")
        if args.action == "setup":
            with io_pod(cluster_) as pod:
                print(setup(cluster_, pod)[-1500:])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
