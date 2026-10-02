"""One person's motion, from per-frame estimates to a smooth, grounded animation.

Steps, for each person (a chain of joined masklets):

1. **Identity.** Body shape and skeleton proportions are the median over the
   person's best-seen frames and held fixed, so nobody grows or shrinks
   between frames.
2. **Local pose.** Joint rotations are taken in each joint's parent frame,
   short gaps are filled by interpolation, single-frame flips are removed and
   the rest is smoothed with a zero-lag adaptive filter.
3. **Placement.** The pelvis is placed in the world (camera rotation and
   depth-to-floor correction applied) and its orientation smoothed the same
   way.
4. **Trajectory.** Planted feet are detected against the floor, and the
   pelvis trajectory is solved once for the whole segment under the physics
   of contact and flight (see ``trajectory``): planted feet stay put on the
   floor, a body in the air follows a free-fall arc, and the image of the
   person stays where the video shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import filters, quat, trajectory
from .body_model import BodyModel
from .ground import smooth_scales
from .keypoints import LEFT_FOOT, RIGHT_FOOT

PELVIS = 1  # MHR joint holding the global rotation; joint 0 is a fixed world root
FLIP = np.array([1.0, 0.0, 0.0, 0.0])  # 180 deg about x: MHR model axes -> camera axes

FILL_GAP_SECONDS = 0.35  # shorter absences are interpolated, longer ones split the track
MIN_SEGMENT_FRAMES = 3  # isolated glimpses shorter than this carry more noise than motion
POSE_FILTER = dict(min_cutoff=2.5, beta=0.4)
ROOT_TURN_FILTER = dict(min_cutoff=2.0, beta=0.5)
CONTACT_HEIGHT = 0.06  # metres above the floor
CONTACT_SPEED = 0.9  # m/s: a planted foot barely moves


@dataclass
class PersonFrames:
    """Per-frame estimates of one person, sorted by frame."""

    frames: np.ndarray  # (T,)
    model_params: np.ndarray  # (T, 204)
    shape: np.ndarray  # (T, 45)
    cam_t: np.ndarray  # (T, 3) camera coordinates, metres
    boxes: np.ndarray  # (T, 4)
    camera: np.ndarray  # (T, 4) camera-to-world rotations
    depth: np.ndarray  # (T,) raw depth-to-floor factors
    image_height: int


@dataclass
class PersonMotion:
    frames: np.ndarray  # (N,) every animated frame
    measured: np.ndarray  # (N,) False where interpolated
    segments: list[tuple[int, int]]  # visible spans, inclusive frame numbers
    shape: np.ndarray  # (45,)
    scales: np.ndarray  # (68,) skeleton proportions
    rest_vertices: np.ndarray  # (V, 3) centimetres, model space
    joint_scales: np.ndarray  # (127,) per-joint scale (constant)
    local_t: np.ndarray  # (N, 127, 3) centimetres
    local_q: np.ndarray  # (N, 127, 4)
    pelvis: np.ndarray  # (N, 8) world state of the pelvis joint
    joints: np.ndarray  # (N, 127, 3) world joint positions, metres
    keypoints: np.ndarray  # (N, 70, 3) world keypoints, metres
    contacts: np.ndarray  # (N, 2) left/right foot planted
    stats: dict = field(default_factory=dict)


def _identity(person: PersonFrames) -> tuple[np.ndarray, np.ndarray]:
    """Median shape and proportions over the frames where the person is seen best."""
    heights = person.boxes[:, 3] - person.boxes[:, 1]
    cut = (person.boxes[:, 1] <= 2) | (person.boxes[:, 3] >= person.image_height - 2)
    score = np.where(cut, 0.0, heights)
    best = score >= np.percentile(score, 50) if np.any(score > 0) else np.ones(len(score), bool)
    return np.median(person.shape[best], axis=0), np.median(person.model_params[best, 136:], axis=0)


def _placement(camera: np.ndarray, cam_t: np.ndarray, depth: np.ndarray, world_rotation: np.ndarray, world_origin: np.ndarray) -> np.ndarray:
    """Per-frame model-to-world similarity (T, 8) before smoothing."""
    world_q = np.broadcast_to(quat.from_matrix(world_rotation), camera.shape)
    rotation = quat.multiply(quat.multiply(world_q, camera), np.broadcast_to(FLIP, camera.shape))
    cam_rc = quat.rotate(camera, depth[:, None] * cam_t)  # rotation-compensated camera coords
    translation = (cam_rc - world_origin) @ world_rotation.T
    return np.concatenate([translation, rotation, (depth / 100.0)[:, None]], axis=1)


def _fill(frames: np.ndarray, values: np.ndarray, full: np.ndarray, rotations: bool) -> np.ndarray:
    """Values at every frame of ``full`` (a superset of ``frames``), interpolated."""
    index = np.searchsorted(frames, full)
    exact = (index < len(frames)) & (frames[np.minimum(index, len(frames) - 1)] == full)
    out = np.empty((len(full), *values.shape[1:]), dtype=np.float64)
    out[exact] = values[index[exact]]
    for i in np.flatnonzero(~exact):
        hi = index[i]
        lo = hi - 1
        t = (full[i] - frames[lo]) / (frames[hi] - frames[lo])
        out[i] = quat.slerp(values[lo], values[hi], t) if rotations else (1 - t) * values[lo] + t * values[hi]
    return out


def _world_joints(model: BodyModel, pelvis: np.ndarray, local_t: np.ndarray, local_q: np.ndarray, joint_scales: np.ndarray) -> np.ndarray:
    """Forward kinematics from the pelvis world state: (N, 127, 8)."""
    n = len(pelvis)
    local = np.concatenate([local_t, local_q, np.broadcast_to(joint_scales[None, :, None], (n, len(joint_scales), 1))], axis=2)
    states = np.zeros_like(local)
    states[:, PELVIS] = pelvis
    states[:, 0] = quat.compose(pelvis, _inverse(local[:, PELVIS]))
    for joint, parent in enumerate(model.parents):
        if joint > PELVIS:
            states[:, joint] = quat.compose(states[:, parent], local[:, joint])
    return states


def _inverse(state: np.ndarray) -> np.ndarray:
    q_inv = quat.conjugate(state[..., 3:7])
    s_inv = 1.0 / state[..., 7:8]
    t_inv = -quat.rotate(q_inv, state[..., :3]) * s_inv
    return np.concatenate([t_inv, q_inv, s_inv], axis=-1)


def _contacts(feet_world: np.ndarray, rate: float) -> np.ndarray:
    """Which feet are planted, from the height and speed of their lowest point."""
    n = len(feet_world[0])
    planted = np.zeros((n, 2), dtype=bool)
    for side, points in enumerate(feet_world):
        lowest = np.argmin(points[:, :, 2], axis=1)
        low = points[np.arange(n), lowest]
        speed = np.zeros(n)
        speed[1:] = np.linalg.norm(np.diff(points[:, :, :2].mean(axis=1), axis=0), axis=1) * rate
        speed = filters.rolling_median(speed, 3)
        planted[:, side] = (low[:, 2] < CONTACT_HEIGHT) & (speed < CONTACT_SPEED)
    # drop one-frame flickers
    for side in range(2):
        column = planted[:, side]
        isolated = column[1:-1] & ~column[:-2] & ~column[2:]
        column[1:-1][isolated] = False
    return planted


def animate(
    person: PersonFrames,
    model: BodyModel,
    rate: float,
    world_rotation: np.ndarray,
    world_origin: np.ndarray,
    camera_position: np.ndarray,
) -> PersonMotion:
    shape, scales = _identity(person)
    rest = model.rest_vertices(shape)
    params = person.model_params.astype(np.float64).copy()
    params[:, 136:] = scales
    local = model.local_states(params)
    joint_scales = np.median(local[:, :, 7], axis=0)

    frames = person.frames
    max_gap = int(round(FILL_GAP_SECONDS * rate))
    spans = [(lo, hi) for lo, hi in filters.segments(frames, max_gap) if hi - lo >= MIN_SEGMENT_FRAMES]
    if not spans:
        spans = [max(filters.segments(frames, max_gap), key=lambda span: span[1] - span[0])]
    out_frames, out_measured, pieces = [], [], []
    segments: list[tuple[int, int]] = []
    for lo, hi in spans:
        f = frames[lo:hi]
        full = np.arange(f[0], f[-1] + 1)
        segments.append((int(full[0]), int(full[-1])))
        measured = np.isin(full, f)

        depth = smooth_scales(person.depth[lo:hi], rate)
        placement = _placement(person.camera[lo:hi], person.cam_t[lo:hi], depth, world_rotation, world_origin)
        pelvis_raw = quat.compose(placement, local[lo:hi, PELVIS])

        lt = _fill(f, local[lo:hi, :, :3], full, rotations=False)
        lq = _fill(f, quat.align_hemisphere(local[lo:hi, :, 3:7]), full, rotations=True)
        pelvis_t = _fill(f, pelvis_raw[:, :3], full, rotations=False)
        pelvis_q = _fill(f, quat.align_hemisphere(pelvis_raw[:, 3:7]), full, rotations=True)
        pelvis_s = np.exp(_fill(f, np.log(pelvis_raw[:, 7:8]), full, rotations=False))

        lq, _ = filters.despike_quaternions(lq, np.deg2rad(45), np.deg2rad(15))
        lq = filters.smooth_quaternions(lq, rate, **POSE_FILTER)
        lt = filters.smooth(lt, rate, **POSE_FILTER)
        pelvis_q, _ = filters.despike_quaternions(pelvis_q[:, None], np.deg2rad(35), np.deg2rad(12))
        pelvis_q = filters.smooth_quaternions(pelvis_q[:, 0], rate, **ROOT_TURN_FILTER)
        pelvis_t, spikes = filters.despike(pelvis_t, window=5, factor=5.0, floor=0.4)
        trusted = measured & ~spikes
        first = trajectory.solve(pelvis_t, camera_position, trusted, rate)
        pelvis_s = np.exp(filters.smooth(np.log(pelvis_s), rate, min_cutoff=0.5, beta=0.0))
        pelvis = np.concatenate([first, pelvis_q, pelvis_s], axis=1)

        # Feet are judged on this first, unconstrained pass; then the
        # trajectory is solved again under contact and flight.
        states = _world_joints(model, pelvis, lt, lq, joint_scales)
        keypoints = model.keypoints(states, rest)
        feet = [keypoints[:, list(LEFT_FOOT)], keypoints[:, list(RIGHT_FOOT)]]
        planted = _contacts(feet, rate)
        body = trajectory.Body.from_keypoints(keypoints, first)
        airborne = ~planted.any(axis=1) & ((body.soles[..., 2] + first[None, :, 2]).min(axis=0) > CONTACT_HEIGHT)
        final = trajectory.solve(pelvis_t, camera_position, trusted, rate, body, planted, airborne)
        shift = final - first
        pelvis[:, :3] = final
        states[:, :, :3] += shift[:, None, :]
        keypoints = keypoints + shift[:, None, :]

        out_frames.append(full)
        out_measured.append(measured)
        pieces.append((lt, lq, pelvis, states[:, :, :3], keypoints, planted))

    lt, lq, pelvis, joints, keypoints, planted = (np.concatenate(parts) for parts in zip(*pieces))
    return PersonMotion(
        stats={"height_m": _standing_height(model, rest, scales, float(np.median(pelvis[:, 7])) / joint_scales[PELVIS])},
        frames=np.concatenate(out_frames),
        measured=np.concatenate(out_measured),
        segments=segments,
        shape=shape,
        scales=scales,
        rest_vertices=rest,
        joint_scales=joint_scales,
        local_t=lt,
        local_q=lq,
        pelvis=pelvis,
        joints=joints,
        keypoints=keypoints,
        contacts=planted,
    )


def _standing_height(model: BodyModel, rest: np.ndarray, scales: np.ndarray, world_per_cm: float) -> float:
    """Height of the person's mesh in the neutral pose, in world metres."""
    neutral = np.zeros((1, 204))
    neutral[0, 136:] = scales
    states = model.global_states(model.local_states(neutral))[0]
    vertices = model.skin(states, rest)
    return float(vertices[:, 1].max() - vertices[:, 1].min()) * world_per_cm

