"""The shared floor, and putting every person's feet on it.

A single image cannot tell a small person nearby from a tall one far away, so
each monocular reconstruction carries its own depth error: some people float,
others sink. With many people on one flat pitch the floor itself is well
determined, though. Kinesia fits one plane through everybody's lowest foot
point over the whole clip (RANSAC, so airborne feet do not count), then moves
each person along their camera ray until their feet meet it. Scaling a person
about the camera centre keeps their image exactly where it was; only their
distance and size change.

The per-frame factor is smoothed with a running median over about a second and
a half, so a jump (feet legitimately above the floor for a fraction of a
second) does not pull the jumper down: their image rises and the body with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters
from .keypoints import FEET

MIN_SCALE, MAX_SCALE = 0.6, 1.7


@dataclass
class Plane:
    normal: np.ndarray  # unit, pointing from the floor towards the camera
    offset: float  # camera height above the floor (metres): normal . X + offset = 0 on the floor

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset


def fit_plane(points: np.ndarray, threshold: float = 0.12, iterations: int = 800, seed: int = 0) -> tuple[Plane, np.ndarray]:
    """Robust plane through ``points`` (M, 3); returns the plane and its inliers."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("not enough foot points to find the floor")
    best_inliers, best_count = None, -1
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        distance = np.abs((points - a) @ normal)
        inliers = distance < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    chosen = points[best_inliers]
    centroid = chosen.mean(axis=0)
    _, _, vt = np.linalg.svd(chosen - centroid)
    normal = vt[-1]
    offset = -float(centroid @ normal)
    if offset < 0:  # orient the normal towards the camera (origin)
        normal, offset = -normal, -offset
    return Plane(normal, offset), best_inliers


def lowest_foot(points_cam: np.ndarray, down: np.ndarray) -> np.ndarray:
    """The lowest foot keypoint of each person-frame along ``down``: (N, 70, 3) -> (N, 3)."""
    feet = points_cam[:, FEET, :]
    depth = feet @ down
    pick = np.argmax(depth, axis=1)
    return feet[np.arange(len(feet)), pick]


def find_floor(points_world: np.ndarray, sample: int = 20000, seed: int = 0) -> Plane:
    """Fit the floor to foot keypoints (N, 70, 3), given in rotation-compensated camera coordinates.

    Camera images are close to level, so the first guess of "down" is image
    down (+y); the plane found with it gives a better "down" for a second pass.
    """
    rng = np.random.default_rng(seed)
    down = np.array([0.0, 1.0, 0.0])
    plane = None
    for _ in range(2):
        lowest = lowest_foot(points_world, down)
        if len(lowest) > sample:
            lowest = lowest[rng.choice(len(lowest), sample, replace=False)]
        plane, _ = fit_plane(lowest)
        down = -plane.normal
    assert plane is not None
    return plane


def floor_factors(lowest_world: np.ndarray, plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Per person-frame factor ``k`` that puts the lowest foot on the floor, and whether it is plausible.

    When no factor in ``[MIN_SCALE, MAX_SCALE]`` brings the feet to the floor
    (someone in the stands, on a stage, or a bad reconstruction), the frame is
    not *reachable*: its factor is clipped and should not be trusted.
    """
    along = lowest_world @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        k = -plane.offset / along
    solved = np.isfinite(k) & (k > 0)
    reachable = solved & (k >= MIN_SCALE) & (k <= MAX_SCALE)
    return np.clip(np.where(solved, k, 1.0), MIN_SCALE, MAX_SCALE), reachable


def depth_scales(lowest_world: np.ndarray, plane: Plane) -> np.ndarray:
    """Per person-frame factor ``k`` that puts the lowest foot on the floor."""
    return floor_factors(lowest_world, plane)[0]


def smooth_scales(k: np.ndarray, rate: float) -> np.ndarray:
    """Remove jumps and jitter from one person's depth factors (log domain)."""
    window = max(3, int(round(1.5 * rate)) | 1)
    median = filters.rolling_median(np.log(k), window)
    return np.exp(filters.smooth(median, rate, min_cutoff=0.4, beta=0.05))


@dataclass
class WorldFrame:
    """Z-up world with the floor at z = 0 and +y pointing away from the first view."""

    rotation: np.ndarray  # (3, 3) rows = world axes in compensated camera coordinates
    origin: np.ndarray  # (3,) world origin in compensated camera coordinates

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.rotation.T

    @property
    def camera_position(self) -> np.ndarray:
        return self.apply(np.zeros(3))

    @classmethod
    def from_floor(cls, plane: Plane, centre: np.ndarray) -> "WorldFrame":
        up = plane.normal
        forward = np.array([0.0, 0.0, 1.0]) - up * up[2]  # first view direction, on the floor
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 1.0, 0.0]) - up * up[1]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        origin = centre - up * (centre @ up + plane.offset)  # centre projected onto the floor
        return cls(np.stack([right, forward, up]), origin)
