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

    scene_dir = run_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{name}.glb"

    fit, up_axis, scale, upright = _fit_upright(
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
    # Write the mesh the way up the fit settled on, so the viewer draws the
    # same object that was measured rather than the raw one.
    mesh.vertices = upright
    mesh.export(scene_dir / mesh_name)
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
) -> tuple[dict, int, float, np.ndarray]:
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
    best: tuple[dict, int, float, np.ndarray] | None = None
    for axis in range(3):
        if extent[axis] <= 1e-9:
            continue
        scale = placement["height_m"] / float(extent[axis])
        other = next(i for i in range(3) if i != axis)
        for flipped in (False, True):
            oriented = vertices.copy()
            if flipped:
                oriented[:, axis] *= -1.0
                oriented[:, other] *= -1.0
            fit = fit_pose_to_silhouette(
                oriented, axis, scale, mask, placement["focal_px"], floor_z,
                width_px, height_px, occluded,
            )
            if best is None or fit["iou"] > best[0]["iou"]:
                best = (fit, axis, scale, oriented)
    if best is None:
        raise ValueError("the mesh has no extent to stand on")

    # Which way up is settled; now measure the size. Taking the object's height
    # to be its mask's height assumes the whole of it showed. A seated person
    # hides a chair's back entirely, leaving floor-to-seat — barely half — so
    # the size has to be found rather than read off.
    fit, axis, scale, oriented = best
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
    return fit, axis, scale, oriented


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
    }
    (pending_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    return record
