"""The pelvis trajectory under the physics of contact and flight."""

import unittest

import numpy as np

from kinesia.scene import trajectory
from kinesia.scene.trajectory import GRAVITY, Body

RATE = 30.0
CAMERA = np.array([0.0, 0.0, 2.0])  # 2 m above the floor, looking along +y


def depth_noise(truth: np.ndarray, rng: np.random.Generator, along: float, across: float = 0.01) -> np.ndarray:
    """``truth`` seen through one camera: badly along each ray, well across it."""
    ray = truth - CAMERA
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    error = rng.normal(0, across, truth.shape)
    error -= (error * ray).sum(1, keepdims=True) * ray
    return truth + error + rng.normal(0, along, (len(truth), 1)) * ray


def standing_body(count: int, mass_height: float = 0.05) -> Body:
    """Feet 0.95 m below the pelvis, 20 cm apart; centre of mass a little above the pelvis."""
    feet = np.zeros((2, count, 3))
    feet[0, :, 0], feet[1, :, 0] = -0.1, 0.1
    feet[:, :, 2] = -0.95
    mass = np.zeros((count, 3))
    mass[:, 2] = mass_height
    return Body(feet, feet.copy(), mass)


def along_ray(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ray = reference - CAMERA
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    return ((points - reference) * ray).sum(1)


class TrajectoryTest(unittest.TestCase):
    def test_follows_a_sprint_and_trusts_the_precise_direction(self):
        rng = np.random.default_rng(4)
        t = np.arange(240) / RATE
        speed = np.clip(t - 2.0, 0, 3.0) * 2.5  # accelerate away from the camera, to 7.5 m/s
        truth = np.column_stack([0.3 * np.sin(t), 8 + np.cumsum(speed) / RATE, np.full(len(t), 0.95)])
        measured = depth_noise(truth, rng, along=0.35)
        solved = trajectory.solve(measured, CAMERA, np.ones(len(t), bool), RATE)
        error, raw = np.abs(along_ray(solved, truth)), np.abs(along_ray(measured, truth))
        self.assertLess(error.mean(), 0.4 * raw.mean())
        sideways = np.linalg.norm(solved - truth, axis=1) ** 2 - along_ray(solved, truth) ** 2
        self.assertLess(np.sqrt(sideways.clip(0)).mean(), 0.02)
        self.assertLess(float(np.percentile(np.linalg.norm(np.diff(solved, axis=0), axis=1) * RATE, 99)), 9.0)

    def test_a_planted_person_stands_still_on_the_floor(self):
        rng = np.random.default_rng(5)
        count = 90
        truth = np.tile([1.0, 8.0, 0.95], (count, 1))
        measured = depth_noise(truth, rng, along=0.3)
        body = standing_body(count)
        planted = np.ones((count, 2), bool)
        solved = trajectory.solve(measured, CAMERA, np.ones(count, bool), RATE, body, planted)
        self.assertLess(np.abs(along_ray(solved, truth)).max(), 0.05)  # the floor fixes the distance
        sole = solved[:, 2] + body.soles[0, :, 2]
        self.assertLess(np.abs(sole).max(), 0.03)
        foot = solved + body.feet[0]
        self.assertLess(np.linalg.norm(np.diff(foot, axis=0), axis=1).sum(), 0.1)  # no creeping

    def test_a_jump_towards_the_camera_flies_a_free_fall_arc(self):
        rng = np.random.default_rng(6)
        t = np.arange(45) / RATE  # 1.5 s: run, jump for 0.5 s, run
        up = (t >= 0.5) & (t <= 1.0)
        truth = np.column_stack([np.zeros(len(t)), 10.0 - 2.0 * t, np.full(len(t), 0.95)])
        flight = t[up] - 0.5
        truth[up, 2] += 2.45 * flight - 0.5 * GRAVITY * flight**2
        measured = depth_noise(truth, rng, along=0.3)
        body = standing_body(len(t), mass_height=0.0)
        body.feet[:, up, 2] = body.soles[:, up, 2] = -0.6  # knees tucked: feet well off the floor
        planted = np.zeros((len(t), 2), bool)
        solved = trajectory.solve(measured, CAMERA, np.ones(len(t), bool), RATE, body, planted, (up, up))
        arc = solved[up]
        acceleration = (arc[2:] - 2 * arc[1:-1] + arc[:-2]) * RATE**2
        self.assertLess(np.abs(acceleration[:, :2]).max(), 1.5)  # horizontal: constant velocity
        self.assertAlmostEqual(float(acceleration[:, 2].mean()), -GRAVITY, delta=1.0)
        self.assertLess(np.abs(along_ray(arc, truth[up])).mean(), 0.5 * np.abs(along_ray(measured[up], truth[up])).mean())

    def test_only_jumps_are_flights_and_only_true_falls_fall_at_g(self):
        rate = 30.0
        t = np.arange(90) / rate
        half_second, second = (t >= 0.5) & (t < 1.0), (t >= 0.5) & (t < 1.5)
        standing = np.full(90, 1.0)  # feet look lifted, but the body does not move
        real = np.full(90, 1.0)
        real[half_second] += 2.45 * (t[half_second] - 0.5) - 0.5 * GRAVITY * (t[half_second] - 0.5) ** 2
        slow = np.full(90, 1.0)  # the same jump played at half speed: twice as long, curving at g/4
        slow[second] += 1.225 * (t[second] - 0.5) - 0.5 * GRAVITY / 4 * (t[second] - 0.5) ** 2
        cases = ((standing, half_second, False, False), (real, half_second, True, True), (slow, second, True, False))
        for height, airborne, jump, fall in cases:
            jumping, falling = trajectory.flights(height, airborne, rate)
            self.assertEqual(jumping.any(), jump)
            self.assertEqual(falling.any(), fall)

    def test_centre_of_mass_of_a_standing_body_is_near_the_pelvis(self):
        keypoints = np.zeros((1, 70, 3))
        heights = {0: 1.65, 69: 1.5, 5: 1.45, 6: 1.45, 7: 1.15, 8: 1.15, 41: 0.85, 62: 0.85, 9: 0.95, 10: 0.95,
                   11: 0.5, 12: 0.5, 13: 0.08, 14: 0.08, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0}
        for index, z in heights.items():
            keypoints[0, index, 2] = z
        self.assertAlmostEqual(float(trajectory.centre_of_mass(keypoints)[0, 2]), 0.97, delta=0.05)


if __name__ == "__main__":
    unittest.main()
