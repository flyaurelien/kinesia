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

import json
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

# COCO-WholeBody foot joints, as used everywhere else in the pipeline.
FOOT_JOINTS = [13, 15, 16, 17, 14, 18, 19, 20]
FLOOR_QUANTILE = 0.10

SCHEMA = "kinesia.scene_object.v1"


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


def fit_pose_to_silhouette(
    mesh_vertices: np.ndarray,
    up_axis: int,
    scale: float,
    mask: np.ndarray,
    focal: float,
    floor_z: float,
    width: int,
    height: int,
) -> dict:
    """Find the floor position and heading whose silhouette matches the mask.

    Centring the mesh on the mask's bounding box is wrong twice over: the
    projection of a 3D centre is not the centre of its silhouette under
    perspective, and it says nothing at all about which way the object faces.

    Both fall out of one question — where must the object stand, and facing
    where, for its outline to be the one we observed. Height is not searched:
    the object rests on the floor, which the feet already fixed.
    """
    import cv2

    cx, cy = width / 2.0, height / 2.0
    verts = mesh_vertices * scale
    # Work in a frame where Z is up and the object's underside is at zero.
    #
    # Reordering axes to put `up` last is a PERMUTATION, and an odd one mirrors
    # the object rather than rotating it. A mirrored chair is indistinguishable
    # from one turned 180 degrees, so the fit would land on the right spot at
    # the right size facing exactly backwards — which is what it did. Negating
    # one axis when the permutation is odd keeps the handedness.
    order = [i for i in range(3) if i != up_axis]
    parity = 1.0 if (order[0], order[1], up_axis) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} else -1.0
    local = np.stack(
        [parity * verts[:, order[0]], verts[:, order[1]], verts[:, up_axis]], axis=1
    )
    local[:, 2] -= local[:, 2].min()
    local[:, 0] -= local[:, 0].mean()
    local[:, 1] -= local[:, 1].mean()

    target = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(target)
    if ys.size == 0:
        raise ValueError("empty mask")

    def silhouette(x_world: float, y_world: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        rotated = np.stack(
            [local[:, 0] * c - local[:, 1] * s, local[:, 0] * s + local[:, 1] * c, local[:, 2]],
            axis=1,
        )
        # world -> camera is the inverse of cam_to_world = (-z, x, -y).
        world = rotated + np.array([x_world, y_world, floor_z])
        cam = np.stack([world[:, 1], -world[:, 2], -world[:, 0]], axis=1)
        keep = cam[:, 2] > 1e-3
        if keep.sum() < 10:
            return np.zeros_like(target)
        u = focal * cam[keep, 0] / cam[keep, 2] + cx
        v = focal * cam[keep, 1] / cam[keep, 2] + cy
        # Splat the projected vertices rather than filling their convex hull:
        # a chair is mostly holes — between the legs, under the seat — and a
        # hull fills every one of them, so it would be scored against a shape
        # the object does not have.
        #
        # Sharper silhouettes were tried and measured: sampling the surface
        # densely, and rasterising the triangles outright. Both raise the
        # overlap score and both move the object about 22 cm in depth, away
        # from where it visibly stands. The mask is truncated where the subject
        # occludes the object, and the floor contact only pins depth to within
        # 14 cm at this distance, so a higher score does not mean a better
        # placement. Left as is deliberately.
        canvas = np.zeros_like(target)
        ui = np.clip(np.round(u).astype(np.int32), 0, width - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, height - 1)
        canvas[vi, ui] = 1
        # Close the gaps between samples without swallowing the real openings.
        return cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=2)

    def iou(x_world: float, y_world: float, yaw: float) -> float:
        got = silhouette(x_world, y_world, yaw)
        union = np.logical_or(got, target).sum()
        return float(np.logical_and(got, target).sum() / union) if union else 0.0

    # Seed from the naive placement, then coarse-to-fine over floor position and
    # heading together — they trade off against each other, so searching one at
    # a time would settle in the wrong place.
    v_bottom = float(ys.max()) - cy
    depth0 = abs(floor_z) * focal / max(v_bottom, 1e-6)
    x0 = -depth0
    y0 = (float(xs.mean()) - cx) / focal * depth0
    best = (iou(x0, y0, 0.0), x0, y0, 0.0)
    span_xy, span_yaw, steps = 0.9, np.pi, 7
    for _ in range(5):
        _, bx, by, byaw = best
        for dx in np.linspace(-span_xy, span_xy, steps):
            for dy in np.linspace(-span_xy, span_xy, steps):
                for dyaw in np.linspace(-span_yaw, span_yaw, steps):
                    score = iou(bx + dx, by + dy, byaw + dyaw)
                    if score > best[0]:
                        best = (score, bx + dx, by + dy, byaw + dyaw)
        span_xy *= 0.45
        span_yaw *= 0.45
    score, x_world, y_world, yaw = best
    return {
        "iou": float(score),
        "position_world": [float(x_world), float(y_world), float(floor_z)],
        "yaw_rad": float(np.arctan2(np.sin(yaw), np.cos(yaw))),
        "seed_iou": float(iou(x0, y0, 0.0)),
    }


def place_object(
    run_dir: Path,
    mesh_path: Path,
    mask_path: Path,
    name: str = "object",
    log: Callable[[str], None] = print,
) -> dict:
    """Solve an object's pose against a finished run and write it under scene/.

    Returns the record written to scene/<name>.json.
    """
    import cv2

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [r for r in (metadata.get("records") or []) if isinstance(r, dict)]

    focal = median_focal(records)
    floor_z = floor_height(records)
    if focal is None or floor_z is None:
        raise ValueError("the run has no focal length or no visible feet")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {mask_path}")
    width_px = int(metadata["video_width"])
    height_px = int(metadata["video_height"])

    calib = calibrate_floor_from_feet(records, height_px)
    if calib is not None:
        v_times_z, residual = calib
        log(f"floor calibrated on the subject's feet: v*z = {v_times_z:.1f} "
            f"(spread {residual * 100:.1f}%, effective focal {v_times_z / abs(floor_z):.0f} px "
            f"vs {focal:.0f} recorded)")
    else:
        v_times_z, residual = None, None
        log("not enough foot samples to calibrate; falling back to the recorded focal")

    placement = solve_placement(mask > 0, focal, floor_z, width_px, height_px, v_times_z)
    placement["calibrated"] = v_times_z is not None
    if residual is not None:
        placement["calibration_spread"] = residual

    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(0) - vertices.min(0)
    up_axis = int(np.argmax(extent))
    # One factor: the mesh is normalised, so matching its height to the solved
    # real height scales every dimension correctly.
    scale = placement["height_m"] / float(extent[up_axis])

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{name}.glb"
    if Path(mesh_path).resolve() != (scene_dir / mesh_name).resolve():
        (scene_dir / mesh_name).write_bytes(Path(mesh_path).read_bytes())

    fit = fit_pose_to_silhouette(
        vertices, up_axis, scale, mask > 0, placement["focal_px"], floor_z,
        width_px, height_px,
    )
    log(f"silhouette: IoU {fit['seed_iou']:.3f} (naive centring) -> {fit['iou']:.3f} (fitted)")
    log(f"  floor position: {[round(v, 3) for v in fit['position_world']]}  "
        f"heading {np.degrees(fit['yaw_rad']):.0f} deg")

    out = {
        "schema": SCHEMA,
        "name": name,
        "mesh": mesh_name,
        "scale": scale,
        "up_axis": "XYZ"[up_axis],
        "centre_world": placement["centre_world"],
        "position_world": fit["position_world"],
        "yaw_rad": fit["yaw_rad"],
        "fit_iou": fit["iou"],
        "solved": placement,
        "note": (
            "Scale and position are solved against the SUBJECT's floor and the "
            "pipeline's focal length, not taken from the object model, whose "
            "output is normalised rather than metric."
        ),
    }
    (scene_dir / f"{name}.json").write_text(json.dumps(out, indent=1))

    log(f"focal {focal:.0f} px | floor z {floor_z:.3f} m")
    log(f"depth {placement['depth_m']:.2f} m")
    log(f"solved size: {placement['width_m'] * 100:.0f} x {placement['height_m'] * 100:.0f} cm")
    log(f"mesh scale factor: {scale:.4f}")
    log(f"centre (world): {[round(v, 3) for v in placement['centre_world']]}")
    log(f"written: {scene_dir / f'{name}.json'}")
    return out
