"""Place a reconstructed object in the SAME metric world as the reconstructed
person, and write the placement the viewer consumes.

SAM 3D Objects returns a normalised shape: its mesh spans about one unit and
its predicted translation/scale live in the model's own frame, not metres. Used
directly, a chair comes out 35 cm tall and 1.5 m from the camera. Any
"hand-to-object distance in centimetres" built on that would be quietly wrong.

So the scale is not taken from the object model at all. It is solved from
geometry the human reconstruction already gives us:

  - the object stands on the floor, and the floor height is known from the
    subject's own feet across the clip;
  - the ray through the BOTTOM of the object's mask therefore meets the floor
    at exactly one depth;
  - at that depth, the mask's pixel height fixes the object's real height,
    through the same pinhole focal length the pipeline used for the person.

Both bodies then share one metric frame because they were solved against the
same floor and the same camera — not because two independent scale estimates
happened to agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

# COCO-WholeBody foot joints, as used everywhere else in the pipeline.
FOOT_JOINTS = [13, 15, 16, 17, 14, 18, 19, 20]
FLOOR_QUANTILE = 0.10


def cam_to_world(point: np.ndarray) -> np.ndarray:
    """Camera space -> world space, the pipeline's convention."""
    return np.array([-point[2], point[0], -point[1]], dtype=np.float64)


def floor_height(records: list[dict]) -> float | None:
    """World height of the floor, from the lowest foot seen in each frame."""
    lows: list[float] = []
    for record in records:
        joints = record.get("joints_cam_xyz")
        if not joints:
            continue
        heights = [
            cam_to_world(np.asarray(joints[i], dtype=np.float64))[2]
            for i in FOOT_JOINTS
            if i < len(joints) and joints[i]
        ]
        if heights:
            lows.append(min(heights))
    if not lows:
        return None
    lows.sort()
    return lows[min(len(lows) - 1, int(len(lows) * FLOOR_QUANTILE))]


def median_focal(records: list[dict]) -> float | None:
    values = [float(r["focal_length"]) for r in records if r.get("focal_length")]
    return float(np.median(values)) if values else None


def calibrate_floor_from_feet(records: list[dict], height: int) -> tuple[float, float] | None:
    """Fit the image->depth relation of the FLOOR from the subject's own feet.

    Every subject in a scene agrees with every other because they all reach
    world space the same way: through the depth this pipeline estimates. An
    object placed by textbook geometry instead does NOT inherit that estimate's
    biases, so it lands beside them rather than among them — which is exactly
    what a chair next to a person looks like when it is subtly wrong.

    So the mapping is measured rather than assumed. A point on the floor at
    depth z projects to v = f * h / z, so v * z is constant for every floor
    point — and the subject's feet ARE floor points, sampled across the clip.
    Fitting that constant captures the effective camera the reconstruction
    behaves as if it had, including whatever bias it carries. Returns
    (v_times_z, residual_fraction).
    """
    products: list[float] = []
    cy = height / 2.0
    for record in records:
        joints = record.get("joints_cam_xyz")
        focal = record.get("focal_length")
        if not joints or not focal:
            continue
        feet = [np.asarray(joints[i], dtype=np.float64) for i in FOOT_JOINTS
                if i < len(joints) and joints[i]]
        if not feet:
            continue
        lowest = max(feet, key=lambda p: p[1])  # camera y grows downward
        if lowest[2] <= 1e-6:
            continue
        v = float(focal) * lowest[1] / lowest[2]  # image row, relative to centre
        products.append(v * lowest[2])
    if len(products) < 30:
        return None
    values = np.asarray(products)
    median = float(np.median(values))
    if abs(median) < 1e-6:
        return None
    return median, float(values.std() / abs(median))


def solve_placement(
    mask: np.ndarray, focal: float, floor_z: float, width: int, height: int,
    v_times_z: float | None = None,
) -> dict:
    """Where the object sits, in world metres, from its mask and the floor."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask")
    u0, u1 = float(xs.min()), float(xs.max())
    v0, v1 = float(ys.min()), float(ys.max())

    cx, cy = width / 2.0, height / 2.0
    if v_times_z is not None:
        # Calibrated: the object's bottom is a floor point like the feet were,
        # so it obeys the same constant.
        v_bottom = v1 - cy
        if abs(v_bottom) < 1e-6:
            raise ValueError("object bottom is on the horizon; depth undetermined")
        depth = v_times_z / v_bottom
        # The focal implied by that same fit, used for the lateral direction so
        # both axes come from one calibration rather than two sources.
        focal = v_times_z / max(abs(floor_z), 1e-6)
    else:
        ray_y = (v1 - cy) / focal
        if abs(ray_y) < 1e-9:
            raise ValueError("object bottom is on the horizon; depth undetermined")
        depth = -floor_z / ray_y
    if depth <= 0:
        raise ValueError(f"non-physical depth {depth:.3f} m — check the floor estimate")

    real_height = (v1 - v0) / focal * depth
    real_width = (u1 - u0) / focal * depth
    centre_cam = np.array(
        [((u0 + u1) / 2 - cx) / focal * depth, ((v0 + v1) / 2 - cy) / focal * depth, depth]
    )
    centre_world = cam_to_world(centre_cam)
    return {
        "depth_m": float(depth),
        "height_m": float(real_height),
        "width_m": float(real_width),
        "centre_world": [float(v) for v in centre_world],
        "floor_z": float(floor_z),
        "focal_px": float(focal),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mesh", required=True, help="reconstructed object (.glb)")
    parser.add_argument("--mask", required=True, help="its mask (.png), same frame")
    parser.add_argument("--name", default="object")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    import cv2

    run_dir = args.project_root / "output" / args.run_id
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [r for r in (metadata.get("records") or []) if isinstance(r, dict)]

    focal = median_focal(records)
    floor_z = floor_height(records)
    if focal is None or floor_z is None:
        print("cannot place: the run has no focal length or no visible feet")
        return 2

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"cannot read mask: {args.mask}")
        return 2
    height_px = int(metadata["video_height"])
    calib = calibrate_floor_from_feet(records, height_px)
    if calib is not None:
        v_times_z, residual = calib
        print(f"floor calibrated on the subject's feet: v*z = {v_times_z:.1f} "
              f"(spread {residual * 100:.1f}%, effective focal {v_times_z / abs(floor_z):.0f} px "
              f"vs {focal:.0f} recorded)")
    else:
        v_times_z, residual = None, None
        print("not enough foot samples to calibrate; falling back to the recorded focal")
    placement = solve_placement(
        mask > 0, focal, floor_z, int(metadata["video_width"]), height_px, v_times_z
    )
    placement["calibrated"] = v_times_z is not None
    if residual is not None:
        placement["calibration_spread"] = residual

    mesh = trimesh.load(args.mesh, force="mesh")
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(0) - vertices.min(0)
    up_axis = int(np.argmax(extent))
    # One factor: the mesh is normalised, so matching its height to the solved
    # real height scales every dimension correctly.
    scale = placement["height_m"] / float(extent[up_axis])

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{args.name}.glb"
    if Path(args.mesh).resolve() != (scene_dir / mesh_name).resolve():
        (scene_dir / mesh_name).write_bytes(Path(args.mesh).read_bytes())

    out = {
        "schema": "kinesia.scene_object.v1",
        "name": args.name,
        "mesh": mesh_name,
        "scale": scale,
        "up_axis": "XYZ"[up_axis],
        "centre_world": placement["centre_world"],
        "solved": placement,
        "note": (
            "Scale and position are solved against the SUBJECT's floor and the "
            "pipeline's focal length, not taken from the object model, whose "
            "output is normalised rather than metric."
        ),
    }
    (scene_dir / f"{args.name}.json").write_text(json.dumps(out, indent=1))

    print(f"focal {focal:.0f} px | floor z {floor_z:.3f} m")
    print(f"depth {placement['depth_m']:.2f} m")
    print(f"solved size: {placement['width_m'] * 100:.0f} x {placement['height_m'] * 100:.0f} cm")
    print(f"mesh scale factor: {scale:.4f}")
    print(f"centre (world): {[round(v, 3) for v in placement['centre_world']]}")
    print(f"written: {scene_dir / f'{args.name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
