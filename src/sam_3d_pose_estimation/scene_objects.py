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
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

# COCO-WholeBody foot joints, as used everywhere else in the pipeline.
FOOT_JOINTS = [13, 15, 16, 17, 14, 18, 19, 20]
FLOOR_QUANTILE = 0.10

SCHEMA = "kinesia.scene_object.v1"
TRANSFORM_SCHEMA = "kinesia.scene_object_transform.v1"

# How many frames to try the prompt on before knowing where the object is. The
# object is static, so one good look is enough — but the first frame tried may
# be one where the subject stands right over it, so a few spread across the clip
# make the search robust without costing much model time.
SEED_FRAMES = 3

# How many of the best-ranked frames get segmented again for the mask actually
# used. The ranking works from an approximate outline; these passes produce the
# real one.
FINALIST_FRAMES = 3

# At most this many objects per run, whatever the prompt asks for: each one
# costs minutes and gigabytes, and a comma-separated list is easy to overfill.
MAX_OBJECTS = 4

# Shapes that are built but not yet standing anywhere.
PENDING_DIR = ".pending"

# Prefix the job runner recognises on a log line to show what is happening.
# The object steps have no frame count to drive a progress bar, so without
# this the interface shows an idle bar for the several minutes they take.
STAGE_MARKER = "[scene]"


def cam_to_world(point: np.ndarray) -> np.ndarray:
    """Camera space -> world space, the pipeline's convention."""
    return np.array([-point[2], point[0], -point[1]], dtype=np.float64)


CAMERA_TO_WORLD = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def model_pose_to_world_matrix(pose: dict[str, Any]) -> np.ndarray:
    """Convert SAM 3D Objects local-to-camera pose fields into object-to-world.

    This consumes the model output directly: no object-class, floor, upright
    axis, silhouette or size heuristic is involved.
    """
    def vector(name: str, length: int) -> np.ndarray:
        value = np.asarray(pose.get(name), dtype=np.float64).reshape(-1)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid model pose field {name}")
        return value

    translation = vector("translation_l2c", 3)
    quaternion = vector("rotation_quaternion_wxyz_l2c", 4)
    scale = vector("scale_l2c", 3)
    if np.any(scale <= 0):
        raise ValueError("model pose scale must be positive")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-12:
        raise ValueError("model pose rotation quaternion must be non-zero")
    quaternion /= quaternion_norm
    w, x, y, z = quaternion
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = CAMERA_TO_WORLD @ rotation @ np.diag(scale)
    matrix[:3, 3] = CAMERA_TO_WORLD @ translation
    return matrix


def upright_rotation(up_axis: int) -> np.ndarray:
    """Return the proper rotation that sends a mesh axis to world-up.

    The result maps a column vector in the mesh's current coordinate system to
    the canonical object frame: Z is up, and X/Y span the object's footprint.
    It is deliberately a rotation rather than an axis permutation so that
    choosing X or Y as up cannot mirror an asymmetric object.

    Args:
        up_axis: Index of the mesh axis that is vertical (0=X, 1=Y, 2=Z).

    Returns:
        A 3x3, right-handed rotation matrix.
    """
    if up_axis not in (0, 1, 2):
        raise ValueError(f"up_axis must be 0, 1, or 2; got {up_axis!r}")
    horizontal = [axis for axis in range(3) if axis != up_axis]
    permutation = (*horizontal, up_axis)
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[0, horizontal[0]] = -1.0 if inversions % 2 else 1.0
    rotation[1, horizontal[1]] = 1.0
    rotation[2, up_axis] = 1.0
    return rotation


def upright_flip_matrix(up_axis: int, flipped: bool) -> np.ndarray:
    """Return the proper rotation used when a candidate must be upside down.

    A vertical flip alone is a reflection. Negating the vertical axis and one
    horizontal axis instead is a 180-degree rotation, which preserves the
    shape's handedness. The returned matrix maps the original reconstructed
    mesh to the mesh written to the scene GLB.
    """
    if up_axis not in (0, 1, 2):
        raise ValueError(f"up_axis must be 0, 1, or 2; got {up_axis!r}")
    matrix = np.eye(3, dtype=np.float64)
    if not flipped:
        return matrix
    other_axis = next(axis for axis in range(3) if axis != up_axis)
    matrix[up_axis, up_axis] = -1.0
    matrix[other_axis, other_axis] = -1.0
    return matrix


def canonical_mesh_frame(
    mesh_vertices: np.ndarray,
    up_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical rotation and exact floor-contact pivot of a mesh.

    The pivot is expressed in the supplied mesh coordinate system. Applying
    the returned rotation to ``vertices - pivot`` produces a frame whose
    footprint is centred at its vertex mean and whose lowest vertex has Z=0.
    This is the one definition used for fitting, serialising and rendering;
    consumers must not replace it with a bounding-box centre.

    Args:
        mesh_vertices: Mesh vertices in the coordinate system to transform.
        up_axis: Index of the mesh axis that is vertical (0=X, 1=Y, 2=Z).

    Returns:
        ``(rotation_to_canonical, pivot_local)``.
    """
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh_vertices must be a non-empty Nx3 array")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh_vertices must contain only finite values")
    rotation = upright_rotation(up_axis)
    canonical = vertices @ rotation.T
    offset = np.array(
        [canonical[:, 0].mean(), canonical[:, 1].mean(), canonical[:, 2].min()],
        dtype=np.float64,
    )
    return rotation, rotation.T @ offset


def world_from_mesh_matrix(
    mesh_vertices: np.ndarray,
    up_axis: int,
    scale: float,
    position_world: np.ndarray | list[float] | tuple[float, float, float],
    yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical metric world transform for a scene mesh.

    The 4x4 matrix is row-major when serialised and acts on a column vector:
    ``world_h = matrix @ [mesh_x, mesh_y, mesh_z, 1]``. ``position_world`` is
    the floor-contact pivot, not a bounding-box centre. The supplied mesh is
    expected to be the one written to the GLB; use :func:`upright_flip_matrix`
    to compose a transform from the pre-flip reconstruction mesh.

    Args:
        mesh_vertices: Vertices in the source mesh coordinate system.
        up_axis: Index of the source mesh's vertical axis.
        scale: Metres per mesh unit.
        position_world: World position of the exact floor-contact pivot.
        yaw_rad: Heading about the world's positive Z axis.

    Returns:
        ``(world_from_mesh, mesh_pivot_local)``.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive; got {scale!r}")
    position = np.asarray(position_world, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world must contain three finite values")
    if not np.isfinite(yaw_rad):
        raise ValueError(f"yaw_rad must be finite; got {yaw_rad!r}")

    canonical_rotation, pivot_local = canonical_mesh_frame(mesh_vertices, up_axis)
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    heading_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    linear = heading_rotation @ canonical_rotation * float(scale)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = position - linear @ pivot_local
    return matrix, pivot_local


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an Nx3 point array.

    This small public helper keeps tests and any future validation on the same
    column-vector convention as the serialised scene-object contract.
    """
    transform = np.asarray(matrix, dtype=np.float64)
    vertices = np.asarray(points, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3:4]


def _homogeneous_rotation(rotation: np.ndarray) -> np.ndarray:
    """Embed a 3x3 rotation in a homogeneous transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix


def _transform_validation(
    mesh_vertices: np.ndarray,
    world_from_mesh: np.ndarray,
    mesh_pivot_local: np.ndarray,
    position_world: np.ndarray,
) -> dict[str, Any]:
    """Report algebraic transform checks without imposing a fit-quality cutoff."""
    transformed = transform_points(world_from_mesh, mesh_vertices)
    pivot_world = transform_points(world_from_mesh, mesh_pivot_local.reshape(1, 3))[0]
    scale = max(
        1.0,
        float(np.abs(transformed).max(initial=0.0)),
        float(np.abs(position_world).max(initial=0.0)),
    )
    tolerance = float(np.finfo(np.float64).eps * 128.0 * scale)
    floor_z = float(position_world[2])
    floor_residual = float(pivot_world[2] - floor_z)
    lowest_vertex_z = float(transformed[:, 2].min())
    checks = {
        "finite_matrix": bool(np.all(np.isfinite(world_from_mesh))),
        "positive_determinant": bool(np.linalg.det(world_from_mesh[:3, :3]) > 0.0),
        "pivot_maps_to_position": bool(np.all(np.abs(pivot_world - position_world) <= tolerance)),
        "mesh_is_on_or_above_floor": bool(lowest_vertex_z >= floor_z - tolerance),
    }
    issues = [name for name, passed in checks.items() if not passed]
    return {
        "status": "valid" if not issues else "invalid",
        "checks": checks,
        "issues": issues,
        "numerical_tolerance_m": tolerance,
        "floor_residual_m": floor_residual,
        "lowest_vertex_z_m": lowest_vertex_z,
        "world_pivot": [float(value) for value in pivot_world],
    }


def _rasterize_canonical_silhouette(
    canonical_vertices: np.ndarray,
    position_world: np.ndarray,
    yaw_rad: float,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project Z-up metric vertices to the sparse silhouette used by the fit."""
    import cv2

    cx, cy = width / 2.0, height / 2.0
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    rotated = np.stack(
        [
            canonical_vertices[:, 0] * cosine - canonical_vertices[:, 1] * sine,
            canonical_vertices[:, 0] * sine + canonical_vertices[:, 1] * cosine,
            canonical_vertices[:, 2],
        ],
        axis=1,
    )
    world = rotated + position_world
    # world -> camera is the inverse of cam_to_world = (-z, x, -y).
    camera = np.stack([world[:, 1], -world[:, 2], -world[:, 0]], axis=1)
    keep = camera[:, 2] > 1e-3
    canvas = np.zeros((height, width), dtype=np.uint8)
    if keep.sum() < 10:
        return canvas
    u = focal * camera[keep, 0] / camera[keep, 2] + cx
    v = focal * camera[keep, 1] / camera[keep, 2] + cy
    ui = np.clip(np.round(u).astype(np.int32), 0, width - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, height - 1)
    canvas[vi, ui] = 1
    # Splat the projected vertices rather than filling their convex hull: a
    # chair is mostly holes, and a hull would score those openings as solid.
    # A small dilation joins adjacent samples without swallowing the openings.
    return cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=2)


def rasterize_mesh_silhouette(
    mesh_vertices: np.ndarray,
    up_axis: int,
    scale: float,
    position_world: np.ndarray | list[float] | tuple[float, float, float],
    yaw_rad: float,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project a mesh using the same geometry contract as pose fitting.

    This is intentionally the fit's sparse vertex rasterisation, not a
    photorealistic renderer. It is useful for deterministic regression tests
    and for a future QA overlay because it reuses the exact pivot, up-axis and
    yaw conventions represented by :func:`world_from_mesh_matrix`.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive; got {scale!r}")
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"focal must be finite and positive; got {focal!r}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    position = np.asarray(position_world, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_world must contain three finite values")
    if not np.isfinite(yaw_rad):
        raise ValueError(f"yaw_rad must be finite; got {yaw_rad!r}")

    rotation, pivot_local = canonical_mesh_frame(mesh_vertices, up_axis)
    canonical_vertices = (np.asarray(mesh_vertices, dtype=np.float64) - pivot_local) @ rotation.T
    canonical_vertices *= float(scale)
    return _rasterize_canonical_silhouette(
        canonical_vertices, position, float(yaw_rad), float(focal), width, height
    )


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


def floor_height_for_run(metadata: dict[str, Any], records: list[dict]) -> tuple[float | None, str]:
    """Return the single floor reference shared by placement and the viewer.

    Current runs persist ``space_view.world_anchor.floor_y`` in camera space.
    It is the floor the viewer translates to Z=0, so a static object's metric
    floor must use its exact world equivalent (``-floor_y``). Older artifacts
    did not save an anchor; for those only, retain the robust foot estimate.

    Args:
        metadata: Parsed ``run_metadata.json`` payload.
        records: Valid per-frame reconstruction records.

    Returns:
        ``(floor_z_world, reference_name)``. The height is ``None`` when no
        reliable floor is available.
    """
    space_view = metadata.get("space_view")
    anchor = space_view.get("world_anchor") if isinstance(space_view, dict) else None
    floor_y = anchor.get("floor_y") if isinstance(anchor, dict) else None
    if isinstance(floor_y, (int, float)) and np.isfinite(float(floor_y)):
        return -float(floor_y), "world_anchor"
    return floor_height(records), "feet_quantile"


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


def _placement_quality(
    mask: np.ndarray,
    occluded: np.ndarray | None,
    fit: dict,
    placement: dict,
) -> dict[str, Any]:
    """Describe observable support for a placement without gating on a cutoff."""
    target = np.asarray(mask, dtype=bool)
    hidden = np.zeros_like(target, dtype=bool) if occluded is None else np.asarray(occluded, dtype=bool)
    if hidden.shape != target.shape:
        raise ValueError("occluded mask must have the same shape as mask")
    mask_pixels = int(target.sum())
    occluded_mask_pixels = int(np.logical_and(target, hidden).sum())
    judged_mask_pixels = mask_pixels - occluded_mask_pixels
    result: dict[str, Any] = {
        # These values are descriptive evidence, not a binary acceptance rule:
        # image quality, occlusion and object class decide what is sufficient.
        "reprojection_iou": float(fit["iou"]),
        "seed_reprojection_iou": float(fit["seed_iou"]),
        "reprojection_iou_gain": float(fit["iou"] - fit["seed_iou"]),
        "mask_pixels": mask_pixels,
        "occluded_mask_pixels": occluded_mask_pixels,
        "judged_mask_pixels": judged_mask_pixels,
        "occluded_mask_fraction": (
            float(occluded_mask_pixels / mask_pixels) if mask_pixels else None
        ),
        "floor_calibration": "feet" if placement.get("calibrated") else "recorded_focal",
        "floor_reference": str(placement.get("floor_reference") or "feet_quantile"),
    }
    if placement.get("calibration_spread") is not None:
        result["floor_calibration_spread"] = float(placement["calibration_spread"])
    return result


def fit_pose_to_silhouette(
    mesh_vertices: np.ndarray,
    up_axis: int,
    scale: float,
    mask: np.ndarray,
    focal: float,
    floor_z: float,
    width: int,
    height: int,
    occluded: np.ndarray | None = None,
    passes: int = 5,
) -> dict:
    """Find the floor position and heading whose silhouette matches the mask.

    Centring the mesh on the mask's bounding box is wrong twice over: the
    projection of a 3D centre is not the centre of its silhouette under
    perspective, and it says nothing at all about which way the object faces.

    Both fall out of one question — where must the object stand, and facing
    where, for its outline to be the one we observed. Height above the floor is
    not searched: the object rests on it, and the feet already fixed it.

    `occluded` marks pixels the subject stands in front of. There the object's
    state is unknown, not absent, so they are left out of the score entirely.
    Counting them as empty is what makes a chair come out two thirds of its
    real size when someone is sitting on it: the backrest is never visible, so
    every candidate large enough to have one is punished for the part hidden
    behind the person.

    Size is not searched here. It was once, as a fourth dimension of this same
    descent, and it settled well short of the optimum: scale trades against
    position, so narrowing both together walks into a local best. The caller
    scans sizes from outside instead, running this fit whole at each one.
    """
    cx, cy = width / 2.0, height / 2.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive; got {scale!r}")
    # Work in the exact same frame the serialised transform uses: Z is up,
    # the footprint's vertex mean is centred, and the lowest vertex is at the
    # origin plane. A viewer must consume that transform rather than deriving a
    # different pivot from its bounding box.
    canonical_rotation, mesh_pivot_local = canonical_mesh_frame(mesh_vertices, up_axis)
    local = (
        (np.asarray(mesh_vertices, dtype=np.float64) - mesh_pivot_local)
        @ canonical_rotation.T
        * float(scale)
    )

    target = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(target)
    if ys.size == 0:
        raise ValueError("empty mask")

    def silhouette(x_world: float, y_world: float, yaw: float) -> np.ndarray:
        return _rasterize_canonical_silhouette(
            local,
            np.array([x_world, y_world, floor_z], dtype=np.float64),
            yaw,
            focal,
            width,
            height,
        )

    # Pixels the subject covers say nothing about the object either way.
    judged = np.ones_like(target, bool) if occluded is None else ~occluded.astype(bool)

    def iou(x_world: float, y_world: float, yaw: float) -> float:
        got = silhouette(x_world, y_world, yaw) & judged
        seen = target & judged
        union = np.logical_or(got, seen).sum()
        return float(np.logical_and(got, seen).sum() / union) if union else 0.0

    # Seed from the naive placement, then coarse-to-fine over floor position and
    # heading together — they trade off against each other, so searching one at
    # a time would settle in the wrong place.
    v_bottom = float(ys.max()) - cy
    depth0 = abs(floor_z) * focal / max(v_bottom, 1e-6)
    x0 = -depth0
    y0 = (float(xs.mean()) - cx) / focal * depth0
    best = (iou(x0, y0, 0.0), x0, y0, 0.0)
    span_xy, span_yaw, steps = 0.9, np.pi, 7
    for _ in range(passes):
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
        "mesh_pivot_local": [float(value) for value in mesh_pivot_local],
    }


def place_object(
    run_dir: Path,
    mesh_path: Path,
    mask_path: Path,
    name: str = "object",
    log: Callable[[str], None] = print,
    occluded: np.ndarray | None = None,
) -> dict:
    """Solve an object's pose against a finished run and write it under scene/.

    Returns the record written to scene/<name>.json.
    """
    import cv2

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [r for r in (metadata.get("records") or []) if isinstance(r, dict)]

    focal = median_focal(records)
    floor_z, floor_reference = floor_height_for_run(metadata, records)
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
    placement["floor_reference"] = floor_reference
    if residual is not None:
        placement["calibration_spread"] = residual

    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(0) - vertices.min(0)

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{name}.glb"

    fit, up_axis, scale, upright, flipped = _fit_upright(
        vertices, extent, placement, mask > 0, floor_z, width_px, height_px, log,
        occluded,
    )
    # The fit may have found the object bigger than its visible mask suggested,
    # which happens whenever part of it never showed.
    # The solved size follows the size the fit measured, not the mask's extent.
    measured_height = float(extent[up_axis]) * scale
    if placement["height_m"] > 1e-9:
        ratio = measured_height / placement["height_m"]
        placement["width_m"] *= ratio
        placement["height_m"] = measured_height
        if abs(ratio - 1.0) > 0.03:
            placement["measured_beyond_mask"] = ratio
    # Write the mesh the way up the fit settled on. The canonical transform
    # below starts from these exact GLB coordinates, so the renderer does not
    # have to repeat the flip, axis conversion or pivot calculation.
    mesh.vertices = upright
    mesh.export(scene_dir / mesh_name)
    position_world = np.asarray(fit["position_world"], dtype=np.float64)
    object_to_world, mesh_pivot_local = world_from_mesh_matrix(
        upright, up_axis, scale, position_world, float(fit["yaw_rad"])
    )
    flip_matrix = upright_flip_matrix(up_axis, flipped)
    raw_mesh_to_world = object_to_world @ _homogeneous_rotation(flip_matrix)
    raw_mesh_pivot_local = flip_matrix.T @ mesh_pivot_local
    transform_validation = _transform_validation(
        upright, object_to_world, mesh_pivot_local, position_world
    )
    quality = _placement_quality(mask > 0, occluded, fit, placement)
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
        # Canonical contract for a renderer. The matrix is row-major and acts
        # on a column vector [mesh_x, mesh_y, mesh_z, 1]. Its source is the
        # exported GLB named above; it already includes every axis, flip,
        # scale, floor pivot and yaw decision made by the fit.
        "object_to_world": object_to_world.tolist(),
        "mesh_pivot_local": [float(value) for value in mesh_pivot_local],
        "world_pivot": [float(value) for value in position_world],
        "transform_contract": {
            "schema": TRANSFORM_SCHEMA,
            "matrix_layout": "row_major",
            "vector_convention": "column_vector",
            "source_mesh": "mesh",
        },
        # Retained for traceability: this matrix starts from the model's input
        # mesh before the proper flip baked into the exported GLB.
        "raw_mesh_to_world": raw_mesh_to_world.tolist(),
        "raw_mesh_pivot_local": [float(value) for value in raw_mesh_pivot_local],
        "orientation": {
            "up_axis": "XYZ"[up_axis],
            "flipped": flipped,
        },
        "quality": quality,
        "transform_validation": transform_validation,
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


def _fit_upright(
    vertices: np.ndarray,
    extent: np.ndarray,
    placement: dict,
    mask: np.ndarray,
    floor_z: float,
    width_px: int,
    height_px: int,
    log: Callable[[str], None],
    occluded: np.ndarray | None = None,
) -> tuple[dict, int, float, np.ndarray, bool]:
    """Work out which way up the object goes, by trying and measuring.

    The obvious guess — the longest dimension points up — holds for a chair and
    fails for a table, which is wider than it is tall: the table gets stood on
    its edge, and being wrong about up also makes the scale wrong, since the
    scale comes from matching the object's height. So each axis is tried, both
    ways up, and the one whose outline actually matches what was observed wins.

    Flipping negates TWO axes rather than one. Negating a single axis mirrors
    the object instead of turning it over, and a mirrored object can match the
    outline while facing the wrong way.
    """
    best: tuple[dict, int, float, np.ndarray, bool] | None = None
    for axis in range(3):
        if extent[axis] <= 1e-9:
            continue
        scale = placement["height_m"] / float(extent[axis])
        for flipped in (False, True):
            flip_matrix = upright_flip_matrix(axis, flipped)
            oriented = np.asarray(vertices, dtype=np.float64) @ flip_matrix.T
            fit = fit_pose_to_silhouette(
                oriented, axis, scale, mask, placement["focal_px"], floor_z,
                width_px, height_px, occluded,
            )
            if best is None or fit["iou"] > best[0]["iou"]:
                best = (fit, axis, scale, oriented, flipped)
    if best is None:
        raise ValueError("the mesh has no extent to stand on")

    # Which way up is settled; now measure the size. Taking the object's height
    # to be its mask's height assumes the whole of it showed. A seated person
    # hides a chair's back entirely, leaving floor-to-seat — barely half — so
    # the size has to be found rather than read off.
    fit, axis, scale, oriented, flipped = best
    grown = 1.0
    coarse = np.exp(np.linspace(np.log(0.6), np.log(2.2), 7))
    for stage, candidates in ((0, coarse), (1, None)):
        if stage == 1:
            step = coarse[1] / coarse[0]
            candidates = grown * np.exp(
                np.linspace(-np.log(step), np.log(step), 5)
            )
        for candidate in candidates:
            trial = fit_pose_to_silhouette(
                oriented, axis, scale * float(candidate), mask, placement["focal_px"],
                floor_z, width_px, height_px, occluded, passes=3 if stage == 0 else 5,
            )
            if trial["iou"] > fit["iou"]:
                fit, grown = trial, float(candidate)
    scale *= grown
    log(f"upright axis {'XYZ'[axis]} "
        f"({extent[axis] * scale * 100:.0f} cm tall), IoU {fit['iou']:.3f}"
        + (f", {grown:.2f}x what the mask alone showed" if abs(grown - 1) > 0.03 else ""))
    return fit, axis, scale, oriented, flipped


def parse_object_prompts(text: str | None) -> tuple[str, ...]:
    """Split the objects the user asked for, and ask for none when they did not.

    Deliberately unlike the subject prompt, which falls back to "person" when
    empty: an empty objects field means no objects, not an object called
    nothing.
    """
    raw = re.split(r"[,;|]", text or "")
    seen: dict[str, str] = {}
    for item in raw:
        cleaned = " ".join(item.split())
        if cleaned and cleaned.lower() not in seen:
            seen[cleaned.lower()] = cleaned
    return tuple(seen.values())[:MAX_OBJECTS]


def object_name(prompt: str) -> str:
    """A file-safe name for an object, from what it was called."""
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug or "object"


def _read_frames(video_path: Path, indices: list[int]) -> dict[int, Any]:
    """Read the requested frames, honouring the container's display rotation."""
    import cv2

    from .video_io import open_video

    frames: dict[int, Any] = {}
    capture = open_video(video_path)
    try:
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames[index] = frame
    finally:
        capture.release()
    return frames


def build_object_shapes(
    run_dir: Path,
    prompts: tuple[str, ...],
    video_path: Path,
    subject_track: Path | None = None,
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Find each named object and reconstruct its shape. No placement yet.

    This is the slow half — minutes per object — and it needs nothing from the
    subject's reconstruction, so it runs before it rather than after. Where the
    object then STANDS does depend on the subject: the floor is measured from
    their feet, and that shared floor is the whole reason the two end up in one
    space. So the shapes wait here and are placed the moment the run has feet
    to offer.
    """
    from . import object_masks, object_shapes
    from .sam3d_runtime import select_device, try_build_human_detector
    from .subject_preview import DEFAULT_SAM3_CODE_ROOT

    if not prompts:
        return {"shapes": [], "failures": [], "skipped": None}

    reason = object_shapes.unavailable_reason(object_shapes.objects_root(project_root))
    if reason:
        log(f"skipping scene objects: {reason}")
        return {"shapes": [], "failures": [], "skipped": reason}
    if not video_path.exists():
        log(f"skipping scene objects: source video missing at {video_path}")
        return {"shapes": [], "failures": [], "skipped": "source video missing"}

    subject_masks, width, height = _subject_coverage(
        run_dir, video_path, subject_track, log
    )
    if not subject_masks:
        log("skipping scene objects: nothing says where the subject is")
        return {"shapes": [], "failures": [], "skipped": "no subject coverage"}

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Spread the first looks across the clip so a subject parked in front of the
    # object early on cannot hide it from every seed.
    seed_frames = [
        subject_masks[round(i * (len(subject_masks) - 1) / max(SEED_FRAMES - 1, 1))][0]
        for i in range(min(SEED_FRAMES, len(subject_masks)))
    ]

    detector = try_build_human_detector(
        detector_name="sam3",
        device=select_device(),
        sam3_code_root=DEFAULT_SAM3_CODE_ROOT,
    )
    if detector is None:
        log("skipping scene objects: the detection model is unavailable")
        return {"shapes": [], "failures": [], "skipped": "detector unavailable"}

    # Segment everything first, then let the detection model go before the
    # shape model starts: each holds several gigabytes, and nothing needs them
    # both at once.
    found: list[dict] = []
    failures: list[dict] = []
    for position, prompt in enumerate(prompts, start=1):
        # Marker the job runner watches for, so the interface can say which
        # object is being worked on instead of showing a still bar for minutes.
        log(f"{STAGE_MARKER} looking for {prompt} ({position}/{len(prompts)})")
        try:
            found.append(_locate_object(
                prompt, scene_dir, video_path, subject_masks, seed_frames, detector, log,
            ))
        except (RuntimeError, ValueError, OSError) as error:
            log(f"could not add '{prompt}': {error}")
            failures.append({"prompt": prompt, "error": str(error)})

    del detector
    _release_memory()

    shaped: list[dict] = []
    for position, target in enumerate(found, start=1):
        log(f"{STAGE_MARKER} reconstructing {target['prompt']} "
            f"({position}/{len(found)}, a few minutes)")
        try:
            shaped.append(_shape_object(target, scene_dir, project_root, log))
        except (RuntimeError, ValueError, OSError) as error:
            log(f"could not add '{target['prompt']}': {error}")
            failures.append({"prompt": target["prompt"], "error": str(error)})

    object_shapes.clear_cache(scene_dir / ".cache")
    return {"shapes": shaped, "failures": failures, "skipped": None}


def place_built_shapes(
    run_dir: Path,
    log: Callable[[str], None] = print,
) -> dict:
    """Stand every reconstructed shape on the floor the subject's feet defined.

    Seconds of work, so it runs as soon as the reconstruction has produced
    enough frames rather than at the end of everything.
    """
    scene_dir = run_dir / "scene"
    pending_dir = scene_dir / PENDING_DIR
    waiting = sorted(pending_dir.glob("*.json")) if pending_dir.is_dir() else []
    if not waiting:
        return {"objects": [], "failures": [], "skipped": None}

    placed: list[dict] = []
    failures: list[dict] = []
    for position, entry in enumerate(waiting, start=1):
        target = json.loads(entry.read_text())
        log(f"{STAGE_MARKER} placing {target['prompt']} ({position}/{len(waiting)})")
        name = target["name"]
        try:
            pose_file = scene_dir / str(target.get("model_pose") or "")
            if pose_file.is_file():
                model_pose = json.loads(pose_file.read_text())
                object_to_world = model_pose_to_world_matrix(model_pose)
                record = {
                    "schema": SCHEMA,
                    "name": name,
                    "mesh": f"{name}.glb",
                    "object_to_world": object_to_world.tolist(),
                    "transform_contract": {
                        "schema": TRANSFORM_SCHEMA,
                        "matrix_layout": "row_major",
                        "vector_convention": "column_vector",
                        "source_mesh": "mesh",
                    },
                    "model_pose": model_pose,
                    "quality": {"source": "sam3d_objects_model_pose"},
                }
            else:
                record = place_object(
                    run_dir, scene_dir / f"{name}.glb", scene_dir / f"{name}_mask.png",
                    name, log,
                    occluded=_subject_on_frame(run_dir, target.get("source_frame")),
                )
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            log(f"could not place '{target['prompt']}': {error}")
            failures.append({"prompt": target["prompt"], "error": str(error)})
            continue
        record["prompt"] = target["prompt"]
        record["source_frame"] = target["source_frame"]
        record["detection_score"] = target["detection_score"]
        (scene_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
        entry.unlink()
        placed.append(record)
    return {"objects": placed, "failures": failures, "skipped": None}


def build_scene_objects(
    run_dir: Path,
    prompts: tuple[str, ...],
    project_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Reconstruct each named object and place it, against a finished run."""
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    built = build_object_shapes(
        run_dir, prompts, Path(metadata.get("video_input") or ""),
        subject_track=None, project_root=project_root, log=log,
    )
    if built["skipped"]:
        return {"objects": [], "failures": built["failures"], "skipped": built["skipped"]}
    placed = place_built_shapes(run_dir, log)
    return {
        "objects": placed["objects"],
        "failures": built["failures"] + placed["failures"],
        "skipped": None,
    }


def _subject_on_frame(run_dir: Path, video_frame: int | None) -> np.ndarray | None:
    """The subject's outline on one frame, to be excluded from the fit's score.

    Without it, everything the person hides counts as proof the object is not
    there.
    """
    if video_frame is None:
        return None
    from . import object_masks

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    width = int(metadata.get("video_width") or 0)
    height = int(metadata.get("video_height") or 0)
    if not width or not height:
        return None
    for record in metadata.get("records") or []:
        if not isinstance(record, dict) or record.get("video_frame") != video_frame:
            continue
        mesh_path, focal = record.get("mesh_path"), record.get("focal_length")
        if not mesh_path or not focal:
            return None
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except (ValueError, OSError):
            return None
        outline = object_masks.subject_silhouette(
            np.asarray(mesh.vertices, dtype=np.float64), float(focal), width, height
        )
        # Widen it a little: the reconstruction's edge and the person's real
        # edge differ by a pixel or two, and those pixels are the ones that
        # would otherwise argue hardest against a correct size.
        import cv2

        return cv2.dilate(
            outline.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)
    return None


def _subject_coverage(
    run_dir: Path,
    video_path: Path,
    subject_track: Path | None,
    log: Callable[[str], None],
) -> tuple[list[tuple[int, Any]], int, int]:
    """Where the subject is on each frame, by whichever means exist yet.

    Reconstructed outlines when the run has produced them, the detection step's
    boxes when it has not — the boxes are coarser, but they are available
    before the run starts, which is the point.
    """
    from . import object_masks

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [r for r in (metadata.get("records") or []) if isinstance(r, dict)]
    width = int(metadata.get("video_width") or 0)
    height = int(metadata.get("video_height") or 0)

    if records and width and height:
        silhouettes = object_masks.load_subject_silhouettes(records, width, height, log)
        if silhouettes:
            return silhouettes, width, height

    if subject_track and subject_track.exists():
        boxes, width, height = object_masks.load_subject_boxes(subject_track, log)
        if boxes:
            return boxes, width, height

    return [], width, height


def _release_memory() -> None:
    """Give the detection model's memory back before the shape model asks."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):  # pragma: no cover - depends on build
        pass


def _locate_object(
    prompt: str,
    scene_dir: Path,
    video_path: Path,
    subject_masks: list[tuple[int, Any]],
    seed_frames: list[int],
    detector: Any,
    log: Callable[[str], None],
) -> dict:
    """Find the object and the clearest view of it, and write both to disk.

    Raises when the object is nowhere to be seen, or is never clear of the
    subject at its base.
    """
    from . import object_masks

    name = object_name(prompt)

    # Where is it? Any decent look will do: this outline only has to say which
    # pixels to watch while the subject moves around.
    seeds = _read_frames(video_path, seed_frames)
    coarse = None
    for index in seed_frames:
        frame = seeds.get(index)
        if frame is None:
            continue
        found = object_masks.segment_object(frame, prompt, detector)
        if found is None:
            continue
        mask, score = found
        if coarse is None or mask.sum() > coarse.sum():
            coarse = mask
            log(f"'{prompt}' found on frame {index} "
                f"(score {score:.2f}, {int(mask.sum())} px)")
    if coarse is None:
        raise ValueError(f"'{prompt}' was not found anywhere in the clip")

    # When is it clearest? Ranking is free — the subject's whereabouts are
    # already in hand — so every frame is considered, not a sample.
    candidates = object_masks.rank_frames(
        coarse, subject_masks, limit=FINALIST_FRAMES
    )
    if not candidates:
        raise ValueError(
            f"the subject covers the base of '{prompt}' on every frame, "
            "so its depth cannot be measured"
        )
    log(f"'{prompt}': best frames "
        + ", ".join(f"{c['frame']} ({c['occlusion'] * 100:.1f}% hidden)"
                    for c in candidates))

    # Segment the finalists properly and keep the fullest outline.
    finalist_frames = _read_frames(
        video_path, [c["frame"] for c in candidates]
    )
    best: tuple[int, Any, float] | None = None
    for candidate in candidates:
        index = candidate["frame"]
        frame = finalist_frames.get(index)
        if frame is None:
            continue
        found = object_masks.segment_object(frame, prompt, detector)
        if found is None:
            continue
        mask, score = found
        if not object_masks.usable_for_depth(mask, mask.shape[0]):
            continue
        if best is None or mask.sum() > best[1].sum():
            best = (index, mask, score)
    if best is None:
        raise ValueError(f"'{prompt}' could not be segmented on any usable frame")

    frame_index, mask, score = best
    log(f"'{prompt}': using frame {frame_index} "
        f"(score {score:.2f}, {int(mask.sum())} px)")

    image_path = scene_dir / f"{name}_frame.png"
    mask_path = scene_dir / f"{name}_mask.png"
    import cv2

    cv2.imwrite(str(image_path), finalist_frames[frame_index])
    object_masks.write_mask(mask, mask_path)
    return {
        "prompt": prompt,
        "name": name,
        "image_path": image_path,
        "mask_path": mask_path,
        "source_frame": frame_index,
        "detection_score": score,
    }


def _shape_object(
    target: dict,
    scene_dir: Path,
    project_root: Path | None,
    log: Callable[[str], None],
) -> dict:
    """Reconstruct a located object's shape and note that it awaits a floor."""
    from . import object_shapes

    name = target["name"]
    object_shapes.reconstruct_mesh(
        image_path=target["image_path"],
        mask_path=target["mask_path"],
        output_glb=scene_dir / f"{name}.glb",
        output_pose=scene_dir / f"{name}_model_pose.json",
        cache_dir=scene_dir / ".cache" / name,
        root=object_shapes.objects_root(project_root),
        log=log,
    )
    # Kept out of the way of the viewer, which reads scene/*.json as objects it
    # can draw — this one has no position yet.
    pending_dir = scene_dir / PENDING_DIR
    pending_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt": target["prompt"],
        "name": name,
        "source_frame": target["source_frame"],
        "detection_score": target["detection_score"],
        "model_pose": f"{name}_model_pose.json",
    }
    (pending_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    return record
