"""Place reconstructed objects in the metric world of the reconstructed person.

SAM 3D Objects predicts each object's shape and pose against one full-scene
point map. All static objects use the same source image and that exact same
point map. SAM 3D Body is aligned once to the shared scene from the visible
subject, then the inverse similarity is applied unchanged to every object.
Nothing in this module invents a category size, orientation, floor offset, or
per-object correction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# COCO-WholeBody foot joints, as used everywhere else in the pipeline.
FOOT_JOINTS = [13, 15, 16, 17, 14, 18, 19, 20]
FLOOR_QUANTILE = 0.10

# The interaction pass evaluates an object's surface against a closed voxel
# occupancy made from the subject mesh. It is deliberately category-agnostic:
# every static reconstructed mesh follows exactly the same geometric test.
BODY_OCCUPANCY_PITCH_M = 0.02
MAX_BODY_OCCUPANCY_CELLS = 1_500_000
MAX_INTERACTION_SURFACE_SAMPLES = 16_000
INTERACTION_TRIGGER_FRACTION = 0.01
INTERACTION_CONTACT_QUANTILE = 0.05
MAX_INTERACTION_CONTACT_SAMPLES = 2_000
MIN_SCENE_ALIGNMENT_SAMPLES = 100

SCHEMA = "kinesia.scene_object.v1"
TRANSFORM_SCHEMA = "kinesia.scene_object_transform.v1"

# Static reconstruction needs one common scene image, not one image per object.
# These candidates only select that one anchor image; object motion is handled
# separately and is inferred on every reconstructed frame by default.
SCENE_FRAME_CANDIDATES = 9

# At most this many objects per run, whatever the prompt asks for: each one
# costs minutes and gigabytes, and a comma-separated list is easy to overfill.
MAX_OBJECTS = 4

# Shapes that are built but not yet standing anywhere.
PENDING_DIR = ".pending"
SCENE_FRAME_FILE = "scene_frame.png"
SCENE_POINTMAP_FILE = "scene_pointmap.npz"
SCENE_SUBJECT_MASK_FILE = "scene_subject_mask.png"
SCENE_ALIGNMENT_FILE = "scene_alignment.json"

# Validation only: these values never modify a pose. An alignment whose mean
# surface error exceeds this fraction of the reconstructed person's height is
# rejected instead of being displayed as a plausible scene.
MAX_BODY_SCENE_RMS_FRACTION = 0.20
MIN_OBJECT_REPROJECTION_IOU = 0.10

# Prefix the job runner recognises on a log line to show what is happening.
# The object steps have no frame count to drive a progress bar, so without
# this the interface shows an idle bar for the several minutes they take.
STAGE_MARKER = "[scene]"


def cam_to_world(point: np.ndarray) -> np.ndarray:
    """Camera space -> world space, the pipeline's convention."""
    return np.array([-point[2], point[0], -point[1]], dtype=np.float64)


CAMERA_TO_WORLD = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# SAM 3D Objects predicts pose in PyTorch3D camera coordinates: +X points left
# in the image and +Y points up. The body pipeline consumes image coordinates
# with +X right and +Y down. The runtime also exports its Z-up decoded mesh as
# a Y-up GLB using ``[x, z, -y]``. Applying both declared conversions is what
# preserves the orientation represented by the model quaternion in our world.
# PyTorch3D applies its pose matrices to row vectors. Our artifact contract uses
# column vectors, so the decoded quaternion matrix is transposed at this boundary.
MODEL_CAMERA_TO_WORLD = CAMERA_TO_WORLD @ np.diag([-1.0, -1.0, 1.0])
MODEL_LOCAL_TO_GLB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
GLB_TO_MODEL_LOCAL = MODEL_LOCAL_TO_GLB.T


def _model_pose_components(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and unpack the local-to-camera pose emitted by the object model."""
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
    return translation, rotation, scale


def model_pose_to_world_matrix(pose: dict[str, Any]) -> np.ndarray:
    """Convert an object-runtime GLB pose into this project's world frame.

    The returned transform applies the model quaternion exactly. It only
    converts the runtime's documented camera and exported-mesh coordinate
    conventions to the coordinate convention used by the person pipeline.
    """
    translation, rotation, scale = _model_pose_components(pose)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        MODEL_CAMERA_TO_WORLD @ rotation.T @ np.diag(scale) @ GLB_TO_MODEL_LOCAL
    )
    matrix[:3, 3] = MODEL_CAMERA_TO_WORLD @ translation
    return matrix


def calibrate_model_pose_to_subject_frame(
    pose: dict[str, Any],
    scene_alignment: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a model pose through the one shared scene-to-body transform.

    No object is independently resized, moved to a mask-derived depth, or
    snapped to the floor. The model's complete pose remains intact and every
    object receives the same scene transform.
    """
    if not isinstance(scene_alignment, dict):
        raise ValueError(
            "object pose has no shared Body/Object scene alignment; reconstruct the scene"
        )
    matrix = apply_scene_alignment_to_model_pose(pose, scene_alignment)
    return matrix, {
        "method": "shared_body_object_scene",
        "model_to_subject_scale": float(scene_alignment["scale"]),
        "floor_offset_m": 0.0,
        "pose_preserved": True,
        "scene_alignment": scene_alignment,
    }


def apply_scene_alignment_to_model_pose(
    pose: dict[str, Any],
    scene_alignment: dict[str, Any],
) -> np.ndarray:
    """Map a model pose into the metric body world with one scene similarity."""
    scale = float(scene_alignment["scale"])
    translation_camera = np.asarray(
        scene_alignment["translation_camera"], dtype=np.float64
    )
    if (
        not np.isfinite(scale)
        or scale <= 0
        or translation_camera.shape != (3,)
        or not np.all(np.isfinite(translation_camera))
    ):
        raise ValueError("invalid shared scene alignment")
    matrix = model_pose_to_world_matrix(pose)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = scale * matrix[:3, 3] + CAMERA_TO_WORLD @ translation_camera
    return matrix


def _world_mesh_silhouette(
    object_to_world: np.ndarray,
    mesh: trimesh.Trimesh,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a transformed object in the source camera for pose QA."""
    import cv2

    world = transform_points(
        object_to_world, np.asarray(mesh.vertices, dtype=np.float64)
    )
    camera = np.column_stack((world[:, 1], -world[:, 2], -world[:, 0]))
    in_front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    u = np.full(len(camera), -1.0, dtype=np.float64)
    v = np.full(len(camera), -1.0, dtype=np.float64)
    u[in_front] = focal * camera[in_front, 0] / camera[in_front, 2] + width / 2.0
    v[in_front] = focal * camera[in_front, 1] / camera[in_front, 2] + height / 2.0
    projected = np.column_stack((u, v))
    canvas = np.zeros((height, width), dtype=np.uint8)
    for face in np.asarray(mesh.faces, dtype=np.int64):
        if not np.all(in_front[face]):
            continue
        polygon = np.rint(projected[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].max() < 0
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(canvas, polygon, 1)
    return canvas.astype(bool)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return the intersection-over-union of two equally-sized masks."""
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


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


def _evenly_spaced_points(points: np.ndarray, limit: int) -> np.ndarray:
    """Keep a deterministic, bounded sample of an Nx3 point cloud."""
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _mesh_surface_samples(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return deterministic surface samples without depending on an acceleration index."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh must contain finite Nx3 vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh vertices must be finite")
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
        face_centres = vertices[faces].mean(axis=1)
        points = np.vstack((vertices, face_centres))
    else:
        points = vertices
    return _evenly_spaced_points(points, MAX_INTERACTION_SURFACE_SAMPLES)


def _subject_mesh_world_on_frame(
    run_dir: Path,
    video_frame: int | None,
) -> tuple[trimesh.Trimesh | None, float | None]:
    """Load the source-frame body mesh in the shared world coordinate system."""
    if video_frame is None:
        return None, None
    try:
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
    except (OSError, ValueError):
        return None, None
    for record in metadata.get("records") or []:
        if not isinstance(record, dict) or record.get("video_frame") != video_frame:
            continue
        mesh_path = record.get("mesh_path")
        if not mesh_path:
            return None, None
        path = Path(str(mesh_path))
        if not path.is_absolute():
            path = run_dir / path
        try:
            mesh = trimesh.load(path, force="mesh")
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
        except (OSError, ValueError, TypeError):
            return None, None
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return None, None
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            return None, None
        focal = record.get("focal_length")
        return (
            trimesh.Trimesh(
                vertices=vertices @ CAMERA_TO_WORLD.T,
                faces=faces,
                process=False,
            ),
            float(focal) if isinstance(focal, (int, float)) and focal > 0 else None,
        )
    return None, None


def align_body_to_scene_pointmap(
    pointmap: np.ndarray,
    subject_mesh_world: trimesh.Trimesh,
    focal: float,
    subject_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Align SAM 3D Body to the shared MoGe scene using Meta's contract.

    Meta's Body/Object notebook uses the visible human height and centre to map
    the body mesh into the MoGe point cloud. Kinesia keeps its body world as the
    public coordinate system, so this function also returns the exact inverse
    transform that maps every SAM 3D Objects pose into that body world. The
    transform is fitted once per scene frame and contains no object dimensions
    or category-dependent adjustment.
    """
    import cv2

    points = np.asarray(pointmap, dtype=np.float64)
    mask = np.asarray(subject_mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("pointmap must have shape HxWx3")
    if mask.shape != (image_height, image_width):
        raise ValueError("subject mask must match the scene image")
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("body focal length must be positive")

    world = np.asarray(subject_mesh_world.vertices, dtype=np.float64)
    faces = np.asarray(subject_mesh_world.faces, dtype=np.int64)
    if world.ndim != 2 or world.shape[1] != 3 or len(world) == 0:
        raise ValueError("subject mesh must contain Nx3 vertices")
    camera = np.column_stack((world[:, 1], -world[:, 2], -world[:, 0]))
    in_front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    u = np.full(len(camera), -1.0, dtype=np.float64)
    v = np.full(len(camera), -1.0, dtype=np.float64)
    u[in_front] = focal * camera[in_front, 0] / camera[in_front, 2] + image_width / 2.0
    v[in_front] = focal * camera[in_front, 1] / camera[in_front, 2] + image_height / 2.0

    projected = np.column_stack((u, v))
    face_indices = np.full((image_height, image_width), -1, dtype=np.int32)
    if faces.ndim == 2 and faces.shape[1] == 3:
        # Painter-style depth ordering is an inexpensive CPU equivalent of the
        # one-face-per-pixel rasterizer used by Meta's reference notebook.
        face_depth = camera[faces, 2].mean(axis=1)
        for face_index in np.argsort(face_depth)[::-1]:
            face = faces[face_index]
            if not np.all(in_front[face]):
                continue
            polygon = np.rint(projected[face]).astype(np.int32)
            if (
                polygon[:, 0].max() < 0
                or polygon[:, 0].min() >= image_width
                or polygon[:, 1].max() < 0
                or polygon[:, 1].min() >= image_height
            ):
                continue
            cv2.fillConvexPoly(face_indices, polygon, int(face_index))
    visible_pixels = (face_indices >= 0) & mask
    if int(visible_pixels.sum()) < MIN_SCENE_ALIGNMENT_SAMPLES:
        raise ValueError("too few visible subject pixels for Body/Object alignment")

    visible_faces = np.unique(face_indices[visible_pixels])
    visible_vertices = np.unique(faces[visible_faces].reshape(-1))
    body_visible = camera[visible_vertices]
    if len(body_visible) < MIN_SCENE_ALIGNMENT_SAMPLES:
        raise ValueError("too few visible body vertices for Body/Object alignment")

    pointmap_mask = cv2.resize(
        visible_pixels.astype(np.uint8),
        (points.shape[1], points.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    scene_visible = points[pointmap_mask].copy()
    scene_visible[:, :2] *= -1.0
    scene_visible = scene_visible[np.isfinite(scene_visible).all(axis=1)]
    if len(scene_visible) < MIN_SCENE_ALIGNMENT_SAMPLES:
        raise ValueError("too few finite MoGe subject points for Body/Object alignment")

    depth_range = float(np.ptp(scene_visible[:, 2]))
    depth_quantile = 0.90 if depth_range > 6.0 else 0.93 if depth_range > 2.0 else 0.95
    scene_visible = scene_visible[
        scene_visible[:, 2] <= np.quantile(scene_visible[:, 2], depth_quantile)
    ]
    body_height = float(np.ptp(body_visible[:, 1]))
    scene_height = float(np.ptp(scene_visible[:, 1]))
    if body_height <= 1e-9 or scene_height <= 1e-9:
        raise ValueError("degenerate visible height for Body/Object alignment")

    body_to_scene_scale = scene_height / body_height
    body_to_scene_translation = (
        scene_visible.mean(axis=0) - body_to_scene_scale * body_visible.mean(axis=0)
    )
    scene_to_body_scale = 1.0 / body_to_scene_scale
    scene_to_body_translation = -body_to_scene_translation / body_to_scene_scale

    body_tree = cKDTree(body_visible)
    scene_in_body = (
        scene_to_body_scale * scene_visible + scene_to_body_translation
    )
    bounded_scene = _evenly_spaced_points(scene_in_body, MAX_INTERACTION_SURFACE_SAMPLES)
    distances, _ = body_tree.query(bounded_scene, k=1)
    rms_m = float(np.sqrt(np.mean(np.square(distances))))
    normalized_rms = rms_m / body_height
    if not np.isfinite(normalized_rms) or normalized_rms > MAX_BODY_SCENE_RMS_FRACTION:
        raise ValueError(
            "Body/Object scene alignment is unreliable: "
            f"RMS {rms_m:.3f} m ({normalized_rms * 100:.1f}% of subject height)"
        )

    return {
        "method": "official_body_to_moge_height_center",
        "source_frame_shared": True,
        "body_to_scene": {
            "scale": float(body_to_scene_scale),
            "translation_camera": body_to_scene_translation.tolist(),
        },
        "scene_to_body": {
            "scale": float(scene_to_body_scale),
            "translation_camera": scene_to_body_translation.tolist(),
        },
        "sample_count": int(len(scene_visible)),
        "body_vertex_count": int(len(body_visible)),
        "rms_m": rms_m,
        "normalized_rms": float(normalized_rms),
        "subject_height_m": body_height,
    }


def _subject_occupancy(
    subject_mesh_world: trimesh.Trimesh,
) -> tuple[Any, float] | None:
    """Build a bounded, closed occupancy grid for an exact subject mesh."""
    if not subject_mesh_world.is_watertight:
        return None
    vertices = np.asarray(subject_mesh_world.vertices, dtype=np.float64)
    extents = np.ptp(vertices, axis=0)
    if not np.all(np.isfinite(extents)) or np.any(extents <= 1e-9):
        return None
    pitch = BODY_OCCUPANCY_PITCH_M
    cells = float(np.prod(np.ceil(extents / pitch) + 3))
    if cells > MAX_BODY_OCCUPANCY_CELLS:
        pitch *= float((cells / MAX_BODY_OCCUPANCY_CELLS) ** (1.0 / 3.0))
    try:
        occupancy = subject_mesh_world.voxelized(pitch).fill()
    except (ValueError, RuntimeError):
        return None
    if len(occupancy.points) == 0:
        return None
    return occupancy, float(pitch)


def _world_vertex_silhouette(
    world_vertices: np.ndarray,
    focal: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project world vertices into the source camera as a sparse silhouette."""
    import cv2

    world = np.asarray(world_vertices, dtype=np.float64)
    canvas = np.zeros((height, width), dtype=np.uint8)
    if world.ndim != 2 or world.shape[1] != 3 or len(world) == 0:
        return canvas.astype(bool)
    # This is the inverse of cam_to_world = (-camera_z, camera_x, -camera_y).
    camera = np.column_stack((world[:, 1], -world[:, 2], -world[:, 0]))
    visible = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    if not np.any(visible):
        return canvas.astype(bool)
    projected = camera[visible]
    u = focal * projected[:, 0] / projected[:, 2] + width / 2.0
    v = focal * projected[:, 1] / projected[:, 2] + height / 2.0
    valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return canvas.astype(bool)
    canvas[np.rint(v[valid]).astype(np.int32), np.rint(u[valid]).astype(np.int32)] = 1
    return cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=2).astype(bool)


def _visible_mask_iou(
    world_vertices: np.ndarray,
    mask: np.ndarray,
    focal: float,
    occluded: np.ndarray | None,
) -> float:
    """Score only object pixels that are observable outside the subject outline."""
    target = np.asarray(mask, dtype=bool)
    hidden = np.zeros_like(target, dtype=bool) if occluded is None else np.asarray(occluded, dtype=bool)
    if hidden.shape != target.shape:
        raise ValueError("occluded mask must have the same shape as mask")
    prediction = _world_vertex_silhouette(
        world_vertices,
        float(focal),
        int(target.shape[1]),
        int(target.shape[0]),
    )
    judged = ~hidden
    union = int(np.logical_and(np.logical_or(prediction, target), judged).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(np.logical_and(prediction, target), judged).sum() / union)


def _surface_contact_distance(
    surface_world: np.ndarray,
    subject_tree: cKDTree,
) -> float:
    """Return a robust near-surface distance between object and subject meshes."""
    samples = _evenly_spaced_points(surface_world, MAX_INTERACTION_CONTACT_SAMPLES)
    distances, _ = subject_tree.query(samples, k=1)
    return float(np.quantile(distances, INTERACTION_CONTACT_QUANTILE))


def resolve_floor_object_body_penetration(
    object_to_world: np.ndarray,
    mesh: trimesh.Trimesh,
    subject_mesh_world: trimesh.Trimesh | None,
    mask: np.ndarray,
    focal: float,
    occluded: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure mesh/body interaction without changing the calibrated pose.

    Overlap is diagnostic because body and object reconstructions have finite
    surface uncertainty and real contact can legitimately intersect slightly.
    This pass never changes scale, orientation, or position; those come only
    from the shared scene calibration.
    """
    matrix = np.asarray(object_to_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("object_to_world must be a finite 4x4 matrix")
    if subject_mesh_world is None:
        return matrix.copy(), {
            "status": "unavailable",
            "reason": "source_frame_subject_mesh_missing",
            "translation_world_m": [0.0, 0.0, 0.0],
            "scale_factor": 1.0,
            "pose_preserved": True,
            "pose_modified": False,
        }
    occupancy_result = _subject_occupancy(subject_mesh_world)
    if occupancy_result is None:
        return matrix.copy(), {
            "status": "unavailable",
            "reason": "source_frame_subject_mesh_not_closed",
            "translation_world_m": [0.0, 0.0, 0.0],
            "scale_factor": 1.0,
            "pose_preserved": True,
            "pose_modified": False,
        }
    occupancy, pitch = occupancy_result
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal must be finite and positive")
    surface_local = _mesh_surface_samples(mesh)
    surface_world = transform_points(matrix, surface_local)
    vertices_world = transform_points(matrix, np.asarray(mesh.vertices, dtype=np.float64))
    inside_before = np.asarray(occupancy.is_filled(surface_world), dtype=bool)
    sample_count = int(len(surface_world))
    hits_before = int(inside_before.sum())
    fraction_before = float(hits_before / max(sample_count, 1))
    visible_iou_before = _visible_mask_iou(vertices_world, mask, focal, occluded)
    subject_surface = _evenly_spaced_points(
        _mesh_surface_samples(subject_mesh_world),
        MAX_INTERACTION_CONTACT_SAMPLES,
    )
    subject_tree = cKDTree(subject_surface)
    contact_before = _surface_contact_distance(surface_world, subject_tree)
    status = (
        "clear"
        if fraction_before <= INTERACTION_TRIGGER_FRACTION
        else "overlap_detected"
    )
    return matrix.copy(), {
        "method": "subject_occupancy_contact_visible_mask",
        "status": status,
        "voxel_pitch_m": pitch,
        "surface_samples": sample_count,
        "penetrating_surface_samples_before": hits_before,
        "penetrating_surface_fraction_before": fraction_before,
        "visible_mask_iou_before": visible_iou_before,
        "contact_distance_before_m": contact_before,
        "pose_preserved": True,
        "penetrating_surface_samples_after": hits_before,
        "penetrating_surface_fraction_after": fraction_before,
        "visible_mask_iou_after": visible_iou_before,
        "contact_distance_after_m": contact_before,
        "translation_world_m": [0.0, 0.0, 0.0],
        "scale_factor": 1.0,
        "pose_modified": False,
    }


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
    # Splat the projected vertices rather than filling their convex hull:
    # hollow geometry would otherwise be scored as solid.
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
    biases, so it lands beside the subject rather than in the same space.

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
    Counting them as empty underestimates any substantially occluded geometry:
    every sufficiently large candidate is punished for the part hidden behind
    the person.

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


def _solve_static_placement(
    run_dir: Path,
    mask_path: Path,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], np.ndarray, int, int]:
    """Measure one static object's metric floor placement from its source mask."""
    import cv2

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    records = [record for record in (metadata.get("records") or []) if isinstance(record, dict)]
    focal = median_focal(records)
    floor_z, floor_reference = floor_height_for_run(metadata, records)
    if focal is None or floor_z is None:
        raise ValueError("the run has no focal length or no visible feet")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {mask_path}")
    width_px = int(metadata["video_width"])
    height_px = int(metadata["video_height"])

    calibration = calibrate_floor_from_feet(records, height_px)
    if calibration is not None:
        v_times_z, residual = calibration
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
    return placement, mask > 0, width_px, height_px


def place_object(
    run_dir: Path,
    mesh_path: Path,
    mask_path: Path,
    name: str = "object",
    log: Callable[[str], None] = print,
    occluded: np.ndarray | None = None,
    source_frame: int | None = None,
) -> dict:
    """Solve an object's pose against a finished run and write it under scene/.

    Returns the record written to scene/<name>.json.
    """
    placement, mask, width_px, height_px = _solve_static_placement(run_dir, mask_path, log)
    floor_z = float(placement["floor_z"])
    focal = float(placement["focal_px"])

    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(mesh.vertices)
    extent = vertices.max(0) - vertices.min(0)

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{name}.glb"

    fit, up_axis, scale, upright, flipped = _fit_upright(
        vertices, extent, placement, mask, floor_z, width_px, height_px, log,
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
    subject_mesh_world, source_focal = _subject_mesh_world_on_frame(run_dir, source_frame)
    object_to_world, interaction = resolve_floor_object_body_penetration(
        object_to_world,
        mesh,
        subject_mesh_world,
        mask,
        source_focal if source_focal is not None else focal,
        occluded,
    )
    position_world = position_world + np.asarray(interaction["translation_world_m"], dtype=np.float64)
    fit["position_world"] = [float(value) for value in position_world]
    flip_matrix = upright_flip_matrix(up_axis, flipped)
    raw_mesh_to_world = object_to_world @ _homogeneous_rotation(flip_matrix)
    raw_mesh_pivot_local = flip_matrix.T @ mesh_pivot_local
    transform_validation = _transform_validation(
        upright, object_to_world, mesh_pivot_local, position_world
    )
    quality = _placement_quality(mask, occluded, fit, placement)
    quality["subject_interaction"] = interaction["status"]
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
        "subject_interaction": interaction,
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
    if interaction["status"] == "overlap_detected":
        log(
            "subject interaction measured without changing pose: "
            f"{interaction['penetrating_surface_fraction_before'] * 100:.1f}% "
            "sampled overlap"
        )
    log(f"written: {scene_dir / f'{name}.json'}")
    return out


def place_model_pose_object(
    mesh_path: Path,
    mask_path: Path,
    model_pose: dict[str, Any],
    name: str,
    scene_alignment: dict[str, Any],
    alignment_evidence: dict[str, Any],
    subject_mesh_world: trimesh.Trimesh,
    focal: float,
    log: Callable[[str], None] = print,
    occluded: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply one already-solved Body/Object scene transform to an object."""
    import cv2

    loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if loaded_mask is None:
        raise OSError(f"could not read object mask: {mask_path}")
    mask = loaded_mask > 127
    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    object_to_world, calibration = calibrate_model_pose_to_subject_frame(
        model_pose, scene_alignment
    )
    object_to_world, interaction = resolve_floor_object_body_penetration(
        object_to_world,
        mesh,
        subject_mesh_world,
        mask,
        focal,
        occluded,
    )
    transformed = transform_points(object_to_world, vertices)
    scale = max(1.0, float(np.abs(transformed).max(initial=0.0)))
    tolerance = float(np.finfo(np.float64).eps * 128.0 * scale)
    hidden = np.zeros_like(mask, dtype=bool) if occluded is None else np.asarray(occluded, dtype=bool)
    if hidden.shape != mask.shape:
        raise ValueError("occluded mask must have the same shape as mask")
    projected = _world_mesh_silhouette(
        object_to_world,
        mesh,
        focal,
        int(mask.shape[1]),
        int(mask.shape[0]),
    )
    visible = ~hidden
    reprojection_iou = _mask_iou(projected & visible, mask & visible)
    quality = {
        "source": "sam3d_objects_model_pose_shared_scene",
        "orientation_source": "sam3d_objects_quaternion",
        "metric_alignment": calibration["method"],
        "mask_pixels": int(mask.sum()),
        "occluded_mask_pixels": int(np.logical_and(mask, hidden).sum()),
        "reprojection_iou": reprojection_iou,
        "subject_interaction": interaction["status"],
    }
    transform_checks = {
        "finite_matrix": bool(np.all(np.isfinite(object_to_world))),
        "positive_determinant": bool(np.linalg.det(object_to_world[:3, :3]) > 0.0),
        "shared_scene_alignment": bool(
            alignment_evidence.get("source_frame_shared") is True
            and float(alignment_evidence.get("normalized_rms", np.inf))
            <= MAX_BODY_SCENE_RMS_FRACTION
        ),
        "model_reprojects_to_mask": bool(reprojection_iou >= MIN_OBJECT_REPROJECTION_IOU),
    }
    issues = [check for check, passed in transform_checks.items() if not passed]
    if issues:
        raise ValueError(
            "model pose failed shared-scene validation: " + ", ".join(issues)
        )
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
        "model_pose_calibration": calibration,
        "body_object_alignment": alignment_evidence,
        "subject_interaction": interaction,
        "quality": quality,
        "transform_validation": {
            "status": "valid" if all(transform_checks.values()) else "invalid",
            "checks": transform_checks,
            "issues": issues,
            "numerical_tolerance_m": tolerance,
            "lowest_vertex_z_m": float(transformed[:, 2].min()),
        },
        "note": (
            "Orientation, scale, and position are the SAM 3D Objects pose after "
            "one shared Body-to-MoGe scene alignment. The same similarity is "
            "used for every object and interaction measurements never alter it."
        ),
    }
    log(f"model pose retained; shared scene scale {calibration['model_to_subject_scale']:.4f}")
    log(
        "one Body/Object alignment: "
        f"RMS {alignment_evidence['rms_m'] * 100:.1f} cm; "
        f"mask IoU {reprojection_iou:.3f}"
    )
    if interaction["status"] == "overlap_detected":
        log(
            "subject interaction measured without changing pose: "
            f"{interaction['penetrating_surface_fraction_before'] * 100:.1f}% "
            "sampled overlap"
        )
    return record


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

    The longest dimension is not generally the vertical one. Choosing the
    wrong up axis also changes the inferred scale, so each axis is tried both
    ways and the outline match decides.

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

    fit, axis, scale, oriented, flipped = best
    log(f"upright axis {'XYZ'[axis]} "
        f"({extent[axis] * scale * 100:.0f} cm tall), IoU {fit['iou']:.3f}")
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
    """Reconstruct all named objects from one shared scene observation.

    Every mask is produced by MLX SAM 3 after one image encoding. Every object
    is then reconstructed against the same SAM 3D Objects/MoGe point map. The
    finished body reconstruction is required because its silhouette identifies
    the subject in that same image without confusing it with another person.
    """
    from . import object_masks, object_shapes

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

    # A point map from a previous scene must never become the geometry of a new
    # image merely because its filename is stable.
    for stale in (scene_dir / SCENE_POINTMAP_FILE, scene_dir / SCENE_ALIGNMENT_FILE):
        if stale.exists():
            stale.unlink()

    try:
        processor = object_masks.build_sam3_processor(confidence=0.5)
        found = _select_shared_scene_frame(
            prompts,
            _subject_prompts_for_run(run_dir),
            scene_dir,
            video_path,
            subject_masks,
            processor,
            log,
        )
    except (RuntimeError, ValueError, OSError) as error:
        log(f"could not build a coherent scene: {error}")
        return {
            "shapes": [],
            "failures": [{"prompt": ", ".join(prompts), "error": str(error)}],
            "skipped": None,
        }

    del processor
    _release_memory()

    failures: list[dict] = []
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


def _build_shared_scene_alignment(
    run_dir: Path,
    source_frame: int,
) -> tuple[dict[str, Any], dict[str, Any], trimesh.Trimesh, float, np.ndarray]:
    """Solve and persist the one transform shared by every static object."""
    import cv2

    scene_dir = run_dir / "scene"
    pointmap_path = scene_dir / SCENE_POINTMAP_FILE
    subject_mask_path = scene_dir / SCENE_SUBJECT_MASK_FILE
    if not pointmap_path.is_file():
        raise ValueError("the shared scene point map is missing; reconstruct the scene")
    loaded_mask = cv2.imread(str(subject_mask_path), cv2.IMREAD_GRAYSCALE)
    if loaded_mask is None:
        raise ValueError("the shared subject mask is missing; reconstruct the scene")
    subject_mask = loaded_mask > 127
    subject_mesh_world, focal = _subject_mesh_world_on_frame(run_dir, source_frame)
    if subject_mesh_world is None or focal is None:
        raise ValueError("the shared scene frame has no reconstructed body mesh")
    with np.load(pointmap_path) as pointmap_file:
        pointmap = np.asarray(pointmap_file["pointmap"], dtype=np.float64)
    evidence = align_body_to_scene_pointmap(
        pointmap,
        subject_mesh_world,
        focal,
        subject_mask,
        int(subject_mask.shape[1]),
        int(subject_mask.shape[0]),
    )
    evidence["source_frame"] = int(source_frame)
    evidence["scene_frame"] = SCENE_FRAME_FILE
    evidence["scene_pointmap"] = SCENE_POINTMAP_FILE
    evidence["subject_mask"] = SCENE_SUBJECT_MASK_FILE
    (scene_dir / SCENE_ALIGNMENT_FILE).write_text(json.dumps(evidence, indent=1))
    return evidence["scene_to_body"], evidence, subject_mesh_world, focal, subject_mask


def place_built_shapes(
    run_dir: Path,
    log: Callable[[str], None] = print,
) -> dict:
    """Place every reconstructed shape in the subject's metric frame.

    Existing model-pose artifacts are deliberately refreshed too. This lets a
    corrected coordinate conversion be applied without repeating the expensive
    shape reconstruction.
    """
    scene_dir = run_dir / "scene"
    pending_dir = scene_dir / PENDING_DIR
    waiting = sorted(pending_dir.glob("*.json")) if pending_dir.is_dir() else []
    targets: list[tuple[dict[str, Any], Path | None]] = [
        (json.loads(entry.read_text()), entry) for entry in waiting
    ]
    seen_names = {str(target.get("name") or "") for target, _ in targets}
    for pose_file in sorted(scene_dir.glob("*_model_pose.json")):
        name = pose_file.stem.removesuffix("_model_pose")
        if not name or name in seen_names:
            continue
        record_path = scene_dir / f"{name}.json"
        if not record_path.is_file():
            continue
        existing = json.loads(record_path.read_text())
        if not isinstance(existing.get("model_pose"), dict):
            continue
        targets.append(({
            "prompt": str(existing.get("prompt") or name),
            "name": name,
            "source_frame": existing.get("source_frame"),
            "detection_score": existing.get("detection_score"),
            "model_pose": pose_file.name,
        }, None))
        seen_names.add(name)

    if not targets:
        return {"objects": [], "failures": [], "skipped": None}

    source_frames = {
        int(target["source_frame"])
        for target, _pending in targets
        if isinstance(target.get("source_frame"), int)
    }
    if len(source_frames) != 1 or any(
        not isinstance(target.get("source_frame"), int) for target, _pending in targets
    ):
        error = (
            "static objects do not share one source frame; reconstruct them together "
            "instead of recalibrating separate model spaces"
        )
        return {
            "objects": [],
            "failures": [
                {"prompt": str(target.get("prompt") or target.get("name") or "object"), "error": error}
                for target, _pending in targets
            ],
            "skipped": None,
        }
    source_frame = next(iter(source_frames))
    try:
        (
            scene_alignment,
            alignment_evidence,
            subject_mesh_world,
            focal,
            subject_mask,
        ) = _build_shared_scene_alignment(run_dir, source_frame)
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        return {
            "objects": [],
            "failures": [
                {
                    "prompt": str(target.get("prompt") or target.get("name") or "object"),
                    "error": str(error),
                }
                for target, _pending in targets
            ],
            "skipped": None,
        }

    placed: list[dict] = []
    failures: list[dict] = []
    for position, (target, pending_entry) in enumerate(targets, start=1):
        prompt = str(target.get("prompt") or target.get("name") or "object")
        log(f"{STAGE_MARKER} placing {prompt} ({position}/{len(targets)})")
        name = target["name"]
        record_path = scene_dir / f"{name}.json"
        if record_path.exists():
            record_path.unlink()
        try:
            pose_file = scene_dir / str(target.get("model_pose") or "")
            if pose_file.is_file():
                model_pose = json.loads(pose_file.read_text())
                record = place_model_pose_object(
                    scene_dir / f"{name}.glb",
                    scene_dir / f"{name}_mask.png",
                    model_pose,
                    name,
                    scene_alignment,
                    alignment_evidence,
                    subject_mesh_world,
                    focal,
                    log,
                    occluded=subject_mask,
                )
            else:
                raise ValueError(
                    "object has no model pose; reconstruct it instead of "
                    "estimating scale from its silhouette"
                )
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            log(f"could not place '{prompt}': {error}")
            failures.append({"prompt": prompt, "error": str(error)})
            continue
        record["prompt"] = prompt
        record["source_frame"] = target.get("source_frame")
        record["detection_score"] = target.get("detection_score")
        record_path.write_text(json.dumps(record, indent=1))
        if pending_entry is not None:
            pending_entry.unlink()
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


def _subject_prompts_for_run(run_dir: Path) -> tuple[str, ...]:
    """Return the exact open-vocabulary prompts used to select the subject."""
    try:
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
    except (OSError, ValueError):
        return ("person",)
    prompts = tuple(
        cleaned
        for value in (metadata.get("sam3_text_prompts") or [])
        if (cleaned := " ".join(str(value).split()))
    )
    return prompts or ("person",)


def _select_shared_scene_frame(
    prompts: tuple[str, ...],
    subject_prompts: tuple[str, ...],
    scene_dir: Path,
    video_path: Path,
    subject_masks: list[tuple[int, Any]],
    processor: Any,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Choose one frame containing the subject and every requested object."""
    from . import object_masks

    import cv2

    count = min(SCENE_FRAME_CANDIDATES, len(subject_masks))
    positions = sorted({
        round(index * (len(subject_masks) - 1) / max(count - 1, 1))
        for index in range(count)
    })
    candidates = [subject_masks[position] for position in positions]
    frames = _read_frames(video_path, [frame for frame, _mask in candidates])
    best: dict[str, Any] | None = None
    seen_counts = {prompt: 0 for prompt in prompts}
    for frame_index, reference_subject in candidates:
        frame = frames.get(frame_index)
        if frame is None:
            continue
        segmented = object_masks.segment_prompt_instances(
            frame,
            (*prompts, *subject_prompts),
            processor,
        )
        selected_subject = object_masks.select_mask_for_reference(
            (
                instance
                for subject_prompt in subject_prompts
                for instance in segmented.get(subject_prompt, [])
            ),
            np.asarray(reference_subject, dtype=bool),
        )
        if selected_subject is None:
            continue
        subject_mask, subject_score = selected_subject
        object_masks_by_prompt: dict[str, np.ndarray] = {}
        object_scores: dict[str, float] = {}
        for prompt in prompts:
            instances = segmented.get(prompt, [])
            if not instances:
                continue
            mask, score = instances[0]
            if mask.shape != subject_mask.shape or not np.any(mask):
                continue
            seen_counts[prompt] += 1
            object_masks_by_prompt[prompt] = mask
            object_scores[prompt] = score
        if len(object_masks_by_prompt) != len(prompts):
            continue

        reference = np.asarray(reference_subject, dtype=bool)
        subject_match = _mask_iou(subject_mask, reference)
        visible_fractions = [
            1.0 - float(np.logical_and(mask, subject_mask).sum()) / max(int(mask.sum()), 1)
            for mask in object_masks_by_prompt.values()
        ]
        rank = (
            min(object_scores.values()),
            min(visible_fractions),
            subject_match,
            float(np.mean(list(object_scores.values()))),
        )
        if best is None or rank > best["rank"]:
            best = {
                "rank": rank,
                "frame_index": frame_index,
                "frame": frame,
                "subject_mask": subject_mask,
                "subject_score": subject_score,
                "object_masks": object_masks_by_prompt,
                "object_scores": object_scores,
            }

    if best is None:
        missing = [prompt for prompt, count_seen in seen_counts.items() if count_seen == 0]
        detail = f"; never detected: {', '.join(missing)}" if missing else ""
        raise ValueError(
            "no single reconstructed frame contains the subject and every requested object"
            + detail
        )

    image_path = scene_dir / SCENE_FRAME_FILE
    if not cv2.imwrite(str(image_path), best["frame"]):
        raise OSError(f"could not write shared scene frame: {image_path}")
    object_masks.write_mask(
        best["subject_mask"], scene_dir / SCENE_SUBJECT_MASK_FILE
    )
    frame_index = int(best["frame_index"])
    log(
        f"{STAGE_MARKER} shared scene frame {frame_index}: subject and "
        f"{len(prompts)} object(s), one MLX image encoding"
    )
    targets: list[dict[str, Any]] = []
    for prompt in prompts:
        name = object_name(prompt)
        mask_path = scene_dir / f"{name}_mask.png"
        object_masks.write_mask(best["object_masks"][prompt], mask_path)
        score = float(best["object_scores"][prompt])
        log(f"'{prompt}' on shared frame {frame_index} (score {score:.2f})")
        targets.append({
            "prompt": prompt,
            "name": name,
            "image_path": image_path,
            "mask_path": mask_path,
            "source_frame": frame_index,
            "detection_score": score,
        })
    return targets


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
        pointmap_path=scene_dir / SCENE_POINTMAP_FILE,
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
