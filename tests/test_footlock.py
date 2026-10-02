"""Bending the legs so that planted feet hold still."""

import unittest

import numpy as np

from kinesia.scene import footlock, quat

NAMES = ["root", "l_upleg", "l_lowleg", "l_foot", "r_upleg", "r_lowleg", "r_foot"]
PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5])


def world(local: np.ndarray) -> np.ndarray:
    out = local.copy()
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            out[..., joint, :] = quat.compose(out[..., parent, :], local[..., joint, :])
    return out


def standing(frames: int) -> np.ndarray:
    """Local states (N, 7, 8): hips 1 m up, knees slightly bent, z up."""
    local = np.zeros((frames, 7, 8))
    local[..., 6] = 1.0  # identity rotations
    local[..., 7] = 1.0  # unit scales
    local[:, 0, :3] = [0.0, 0.0, 1.0]
    for hip, side in ((1, -0.1), (4, 0.1)):
        local[:, hip, :3] = [side, 0.0, 0.0]
        local[:, hip + 1, :3] = [0.0, 0.03, -0.48]  # thigh, knee a little forward
        local[:, hip + 2, :3] = [0.0, -0.03, -0.47]  # shin
    return local


class FootLockTest(unittest.TestCase):
    def test_the_ankle_reaches_its_target_and_the_foot_keeps_its_turn(self):
        local = standing(1)
        local[0, 3, 3:7] = quat.axis_angle([0, 0, 1], 0.3)  # the foot turned out
        states = world(local)
        moves = np.zeros((2, 1, 3))
        moves[0, 0] = [0.04, -0.03, 0.05]
        bent = local.copy()
        bent[..., 3:7] = footlock.bend_legs(NAMES, PARENTS, states, local[..., 3:7], moves)
        after = world(bent)
        self.assertTrue(np.allclose(after[0, 3, :3], states[0, 3, :3] + moves[0, 0], atol=1e-6))
        self.assertLess(float(quat.angle(after[0, 3, 3:7], states[0, 3, 3:7])), 1e-6)
        self.assertTrue(np.allclose(after[0, 6], states[0, 6]))  # the other leg is untouched
        for parent, child in ((1, 2), (2, 3)):  # bones keep their length
            self.assertAlmostEqual(
                float(np.linalg.norm(after[0, child, :3] - after[0, parent, :3])),
                float(np.linalg.norm(states[0, child, :3] - states[0, parent, :3])),
                places=6,
            )

    def test_a_planted_foot_is_held_on_one_spot_and_lifted_out_of_the_floor(self):
        frames = 20
        feet = np.zeros((2, frames, 3, 3))
        feet[:, :, :, 0] = np.linspace(0.0, 0.1, frames)[None, :, None]  # creeping 10 cm
        feet[:, :, 0, 2] = -0.04  # toes 4 cm into the floor
        planted = np.zeros((frames, 2), bool)
        planted[5:15, 0] = True
        moves = footlock.shifts(feet, planted, 30.0)
        held = feet[0, 5:15].mean(axis=1) + moves[0, 5:15]
        self.assertLess(np.ptp(held[:, 0]), 1e-9)  # one spot during the contact
        self.assertTrue(np.allclose(feet[0, :, :, 2].min(axis=1) + moves[0, :, 2], 0.0))  # on the floor
        self.assertTrue(np.allclose(moves[1, :, :2], 0.0))  # the right foot is never planted
        self.assertEqual(moves[0, 0, 0], 0.0)  # far from the contact, nothing moves sideways
        self.assertGreater(abs(moves[0, 4, 0]), 0.0)  # but the correction fades in before it


if __name__ == "__main__":
    unittest.main()
