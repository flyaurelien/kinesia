"""Run SAM 3D Objects and persist its model pose beside the exported GLB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument("--pointmap-output", type=Path)
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
    pose_payload = {
        "schema": "kinesia.sam3d_objects_pose.v1",
        "translation_l2c": json_value(result["translation"]),
        "rotation_quaternion_wxyz_l2c": json_value(result["rotation"]),
        "scale_l2c": json_value(result["scale"]),
    }
    if args.pointmap_output is not None:
        pointmap = result.get("pointmap")
        if pointmap is None:
            pointmap = compute_pointmap(args)
        if hasattr(pointmap, "detach"):
            pointmap = pointmap.detach().cpu().numpy()
        pointmap = np.asarray(pointmap, dtype=np.float32)
        if pointmap.ndim != 3 or pointmap.shape[2] != 3:
            raise RuntimeError("object runtime produced an invalid scene point map")
        args.pointmap_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.pointmap_output, pointmap=pointmap)
        try:
            pointmap_reference = args.pointmap_output.resolve().relative_to(
                args.pose_output.resolve().parent
            )
        except ValueError:
            pointmap_reference = args.pointmap_output.resolve()
        pose_payload["scene_pointmap"] = str(pointmap_reference)
    args.pose_output.parent.mkdir(parents=True, exist_ok=True)
    args.pose_output.write_text(json.dumps(pose_payload, indent=1))


def compute_pointmap(args: argparse.Namespace) -> Any:
    """Compute the scene point map when a cached pose omitted that output."""
    from main import load_image, load_mask_from_file
    from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory

    pipeline = InferencePipelineLowMemory(
        config_path="checkpoints/hf/pipeline.yaml",
        device="cpu",
        dtype="float16",
        cache_dir=str(args.cache_dir.resolve()),
    )
    image = load_image(str(args.image.resolve()))
    mask = load_mask_from_file(str(args.mask.resolve()))
    merged = pipeline.merge_image_and_mask(image, mask)
    result = pipeline.compute_pointmap(merged)
    return result["pointmap"].permute(1, 2, 0)


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
