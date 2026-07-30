"""Run SAM 3D Objects and persist its model pose beside the exported GLB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    """Convert tensors emitted by the optional runtime into JSON values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value.tolist() if hasattr(value, "tolist") else value


def main() -> None:
    """Invoke the external image+mask model without discarding its pose."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--simplify", type=float, default=0.9)
    parser.add_argument("--pose-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    if args.pose_only:
        result = run_pose_only(args)
    else:
        from main import run_pipeline

        result = run_pipeline(
            image_path=str(args.image.resolve()), mask_path=str(args.mask.resolve()),
            output_path=str(args.output.resolve()), output_mesh=True,
            cache_dir=str(args.cache_dir.resolve()), simplify_ratio=args.simplify,
        )
    required = ("translation", "rotation", "scale")
    missing = [field for field in required if not result or result.get(field) is None]
    if missing:
        raise RuntimeError(f"missing SAM 3D Objects pose fields: {', '.join(missing)}")
    args.pose_output.parent.mkdir(parents=True, exist_ok=True)
    args.pose_output.write_text(json.dumps({
        "schema": "kinesia.sam3d_objects_pose.v1",
        "translation_l2c": json_value(result["translation"]),
        "rotation_quaternion_wxyz_l2c": json_value(result["rotation"]),
        "scale_l2c": json_value(result["scale"]),
    }, indent=1))


def run_pose_only(args: argparse.Namespace) -> dict[str, Any]:
    """Run only the object model stages needed to emit the model pose.

    The external ``run_pipeline(..., output_mesh=False)`` writes a dense STL of
    every voxel, which is unnecessary and disproportionately slow for a pose
    sample. Calling its low-memory pipeline directly preserves the model output
    while stopping before that export.
    """
    from main import load_image, load_mask_from_file
    from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory

    pipeline = InferencePipelineLowMemory(
        config_path="checkpoints/hf/pipeline.yaml",
        device="cpu",
        dtype="float16",
        cache_dir=str(args.cache_dir.resolve()),
    )
    return pipeline.run(
        load_image(str(args.image.resolve())),
        load_mask_from_file(str(args.mask.resolve())),
        seed=42,
        stage1_only=True,
        stage1_inference_steps=12,
        stage2_inference_steps=12,
    )


if __name__ == "__main__":
    main()
