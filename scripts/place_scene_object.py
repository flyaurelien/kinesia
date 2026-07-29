"""Place a reconstructed object in the same metric world as the subject.

Thin command-line front end; the solving lives in
sam_3d_pose_estimation.scene_objects so the pipeline can call it directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam_3d_pose_estimation.scene_objects import place_object  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mesh", required=True, help="reconstructed object (.glb)")
    parser.add_argument("--mask", required=True, help="its mask (.png), same frame")
    parser.add_argument("--name", default="object")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    run_dir = args.project_root / "output" / args.run_id
    try:
        place_object(run_dir, Path(args.mesh), Path(args.mask), args.name)
    except (ValueError, OSError) as error:
        print(f"cannot place: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
