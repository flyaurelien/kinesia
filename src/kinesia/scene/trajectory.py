"""Where each person is: one physically consistent trajectory per person.

SAM 3D Body sees each frame on its own. Across the picture it places a person
to within a few centimetres, but along the camera ray, their distance, it is
off by tens of centimetres, differently in every frame: taken as it is, the
trajectory shakes towards and away from the camera with accelerations no
human can produce. The trajectory is instead the one path that agrees best
with all frames at once (least squares), each frame trusted as much as it
deserves (closely across its ray, loosely along it), under the physics of a
body on a floor:

* a person changes pace with bounded accelerations;
* a planted foot does not slide, and stands on the floor;
* with no foot on the floor the centre of mass is in free fall: its
  horizontal velocity stays constant and it accelerates downwards at g.

These are the contact and flight conditions of physics-based monocular motion
capture (Rempe et al., "Contact and Human Dynamics from Monocular Video",
ECCV 2020; Shimada et al., "PhysCap", SIGGRAPH Asia 2020), kept to the centre
of mass so that they stay linear: one sparse least-squares problem per person.
The body's pose is not touched, and its image stays where the video shows it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keypoints import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_FOOT, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NECK, NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_FOOT, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)  # fmt: skip

GRAVITY = 9.81
LATERAL_NOISE = 0.002  # metres of sideways error per metre of distance (~3 px)
DEPTH_NOISE = 0.25  # metres of error along the camera ray, plus DEPTH_SHARE of the distance
DEPTH_SHARE = 0.02
HORIZONTAL_ACCEL = 5.0  # m/s^2 a player changes pace or direction with
VERTICAL_ACCEL = 14.0  # m/s^2: pushing off and landing
FREE_FALL_ACCEL = 1.0  # m/s^2: how far a centre of mass in the air may stray from free fall
SLIP_SPEED = 0.07  # m/s: how fast a planted foot may still creep
FLOOR_GAP = 0.02  # metres between a planted sole and the floor

# Mass shares of the body's segments (de Leva, J. Biomech. 1996), each between two keypoints.
SEGMENTS = (
    (0.0694, (NOSE,), (NECK,)),
    (0.4346, (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)),
    (0.0271, (LEFT_SHOULDER,), (LEFT_ELBOW,)),
    (0.0271, (RIGHT_SHOULDER,), (RIGHT_ELBOW,)),
    (0.0223, (LEFT_ELBOW,), (LEFT_WRIST,)),
    (0.0223, (RIGHT_ELBOW,), (RIGHT_WRIST,)),
    (0.1416, (LEFT_HIP,), (LEFT_KNEE,)),
    (0.1416, (RIGHT_HIP,), (RIGHT_KNEE,)),
    (0.0433, (LEFT_KNEE,), (LEFT_ANKLE,)),
    (0.0433, (RIGHT_KNEE,), (RIGHT_ANKLE,)),
    (0.0137, (LEFT_ANKLE,), LEFT_FOOT),
    (0.0137, (RIGHT_ANKLE,), RIGHT_FOOT),
)


def centre_of_mass(keypoints: np.ndarray) -> np.ndarray:
    """(N, 3) centre of mass from MHR70 keypoints (N, 70, 3): segment midpoints weighted by mass."""
    total = sum(mass for mass, _, _ in SEGMENTS)
    centre = np.zeros(keypoints.shape[:1] + (3,))
    for mass, a, b in SEGMENTS:
        centre += mass * 0.5 * (keypoints[:, list(a)].mean(axis=1) + keypoints[:, list(b)].mean(axis=1))
    return centre / total


@dataclass
class Body:
    """The pose around the pelvis, per frame: offsets from the pelvis in world axes (metres)."""

    feet: np.ndarray  # (2, N, 3) centre of the left and right foot
    soles: np.ndarray  # (2, N, 3) lowest point of each foot
    mass: np.ndarray  # (N, 3) centre of mass

    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray, pelvis: np.ndarray) -> Body:
        feet = np.stack([keypoints[:, list(side)] for side in (LEFT_FOOT, RIGHT_FOOT)])  # (2, N, 3, 3)
        lowest = np.take_along_axis(feet, feet[..., 2].argmin(axis=2)[..., None, None].repeat(3, axis=-1), axis=2)[:, :, 0]
        return cls(feet.mean(axis=2) - pelvis, lowest - pelvis, centre_of_mass(keypoints) - pelvis)


class _Normal:
    """The normal equations of a least-squares problem over N positions, as a symmetric band."""

    WIDTH = 6  # unknowns are (frame, axis); terms reach two frames away on the same axis

    def __init__(self, count: int):
        self.count = count
        self.band = np.zeros((3 * count, self.WIDTH + 1))  # band[i, k] = A[i, i - k]
        self.rhs = np.zeros(3 * count)

    def frames(self, frames: np.ndarray, weights: np.ndarray, targets: np.ndarray) -> None:
        """Residuals ``X[t] - target`` with a full 3x3 weight per frame."""
        for a in range(3):
            for c in range(a + 1):
                np.add.at(self.band, (3 * frames + a, a - c), weights[:, a, c])
        np.add.at(self.rhs, (3 * frames[:, None] + np.arange(3)).ravel(), np.einsum("tij,tj->ti", weights, targets).ravel())

    def differences(self, first: np.ndarray, coefficients: tuple[float, ...], axis: int, weight: float, targets: np.ndarray) -> None:
        """Residuals ``sum_k c_k X[first + k, axis] - target`` (consecutive frames, one axis)."""
        rows = [3 * (first + k) + axis for k in range(len(coefficients))]
        for k, ck in enumerate(coefficients):
            for m, cm in enumerate(coefficients[: k + 1]):
                np.add.at(self.band, (rows[k], 3 * (k - m)), weight * ck * cm)
            np.add.at(self.rhs, rows[k], weight * ck * targets)

    def solve(self) -> np.ndarray:
        """Banded Cholesky factorisation and the two triangular solves."""
        band, n, w = self.band, 3 * self.count, self.WIDTH
        low = np.zeros_like(band)  # low[i, k] = L[i, i - k]
        for i in range(n):
            start = max(0, i - w)
            for j in range(start, i + 1):
                m = np.arange(start, j)
                value = band[i, i - j] - low[i, i - m] @ low[j, j - m]
                low[i, i - j] = np.sqrt(max(value, 1e-12)) if j == i else value / low[j, 0]
        y = np.zeros(n)
        for i in range(n):
            m = np.arange(max(0, i - w), i)
            y[i] = (self.rhs[i] - low[i, i - m] @ y[m]) / low[i, 0]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            m = np.arange(i + 1, min(n, i + w + 1))
            x[i] = (y[i] - low[m, m - i] @ x[m]) / low[i, 0]
        return x.reshape(self.count, 3)


def solve(
    estimates: np.ndarray,
    camera: np.ndarray,
    trusted: np.ndarray,
    rate: float,
    body: Body | None = None,
    planted: np.ndarray | None = None,
    airborne: np.ndarray | None = None,
) -> np.ndarray:
    """The pelvis trajectory (N, 3) that best explains the per-frame ``estimates`` (N, 3).

    ``trusted`` frames are measurements, the others (filled gaps) carry no
    information. With ``body``, ``planted`` (N, 2) feet and ``airborne``
    (N,) frames, the contact and flight conditions apply; without, the
    result is a smoothing that knows depth is the uncertain direction.
    """
    count = len(estimates)
    if count < 3:
        return estimates.copy()
    dt = 1.0 / rate
    normal = _Normal(count)

    ray = estimates - camera
    distance = np.linalg.norm(ray, axis=1).clip(0.5)
    ray /= distance[:, None]
    along = ray[:, :, None] * ray[:, None, :]
    across = np.eye(3) - along
    weights = across / ((LATERAL_NOISE * distance) ** 2)[:, None, None] + along / ((DEPTH_NOISE + DEPTH_SHARE * distance) ** 2)[:, None, None]
    weights[~trusted] = np.eye(3) * 1e-6  # interpolated frames: only anchored, very loosely
    normal.frames(np.arange(count), weights, estimates)

    middle = np.arange(count - 2)  # second differences over frames t, t+1, t+2
    falling = np.zeros(count - 2, dtype=bool)
    if body is not None and airborne is not None:
        falling = airborne[:-2] & airborne[1:-1] & airborne[2:]
    for axis, accel in ((0, HORIZONTAL_ACCEL), (1, HORIZONTAL_ACCEL), (2, VERTICAL_ACCEL)):
        free = middle[~falling]
        normal.differences(free, (1.0, -2.0, 1.0), axis, 1.0 / (accel * dt * dt) ** 2, np.zeros(len(free)))
        if falling.any():
            # The centre of mass (pelvis + offset) moves as a thrown stone.
            t = middle[falling]
            mass = body.mass[:, axis]
            target = -(mass[t] - 2 * mass[t + 1] + mass[t + 2]) - (GRAVITY * dt * dt if axis == 2 else 0.0)
            normal.differences(t, (1.0, -2.0, 1.0), axis, 1.0 / (FREE_FALL_ACCEL * dt * dt) ** 2, target)

    if body is not None and planted is not None:
        for side in range(2):
            down = np.flatnonzero(planted[:, side])
            sole = body.soles[side, down]
            normal.differences(down, (1.0,), 2, 1.0 / FLOOR_GAP**2, -sole[:, 2])
            still = np.flatnonzero(planted[:-1, side] & planted[1:, side])
            foot = body.feet[side]
            for axis in range(3):
                target = -(foot[still + 1, axis] - foot[still, axis])
                normal.differences(still, (-1.0, 1.0), axis, 1.0 / (SLIP_SPEED * dt) ** 2, target)
    return normal.solve()
