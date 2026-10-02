"""Quaternion helpers and temporal filters used to build the 3D scene."""

import unittest

import numpy as np

from kinesia.scene import filters, quat


def random_quats(rng, n):
    return quat.normalize(rng.normal(size=(n, 4)))


class QuaternionTest(unittest.TestCase):
    def test_matrix_round_trip(self):
        rng = np.random.default_rng(0)
        q = random_quats(rng, 50)
        back = quat.from_matrix(quat.to_matrix(q))
        self.assertTrue(np.allclose(np.abs(np.sum(q * back, axis=1)), 1.0, atol=1e-9))

    def test_rotate_matches_matrix(self):
        rng = np.random.default_rng(1)
        q = random_quats(rng, 20)
        v = rng.normal(size=(20, 3))
        expected = np.einsum("nij,nj->ni", quat.to_matrix(q), v)
        self.assertTrue(np.allclose(quat.rotate(q, v), expected))

    def test_compose_applies_right_then_left(self):
        rng = np.random.default_rng(2)
        a = np.concatenate([rng.normal(size=3), random_quats(rng, 1)[0], [1.7]])
        b = np.concatenate([rng.normal(size=3), random_quats(rng, 1)[0], [0.4]])
        p = rng.normal(size=(5, 3))
        self.assertTrue(np.allclose(quat.apply(quat.compose(a, b), p), quat.apply(a, quat.apply(b, p))))

    def test_slerp_endpoints_and_midpoint(self):
        a = quat.axis_angle([0, 0, 1], 0.0)
        b = quat.axis_angle([0, 0, 1], 1.0)
        self.assertTrue(np.allclose(quat.slerp(a, b, 0.0), a))
        self.assertTrue(np.allclose(quat.slerp(a, b, 1.0), b))
        self.assertAlmostEqual(float(quat.angle(quat.slerp(a, b, 0.5), a)), 0.5, places=6)

    def test_align_hemisphere_removes_sign_flips(self):
        q = np.tile(quat.axis_angle([1, 0, 0], 0.3), (6, 1))
        q[[1, 3]] *= -1
        aligned = quat.align_hemisphere(q)
        self.assertTrue(np.all(np.sum(aligned[1:] * aligned[:-1], axis=1) > 0))


class FilterTest(unittest.TestCase):
    def test_smooth_removes_jitter_without_lag(self):
        rng = np.random.default_rng(3)
        t = np.arange(300) / 30.0
        truth = np.sin(2 * np.pi * 0.2 * t)
        noisy = truth + rng.normal(0, 0.05, t.shape)
        smooth = filters.smooth(noisy, 30.0, min_cutoff=1.0, beta=0.1)
        self.assertLess(np.std(np.diff(smooth - truth)), 0.35 * np.std(np.diff(noisy - truth)))
        # zero-lag: the peak stays where it is
        self.assertLessEqual(abs(int(np.argmax(smooth[:120])) - int(np.argmax(truth[:120]))), 2)

    def test_despike_replaces_single_outliers(self):
        x = np.linspace(0, 1, 50)[:, None].repeat(3, axis=1)
        x[20] += 5.0
        cleaned, spikes = filters.despike(x, window=5, factor=4.0)
        self.assertEqual(np.flatnonzero(spikes).tolist(), [20])
        self.assertLess(np.abs(cleaned[20] - x[19]).max(), 0.1)

    def test_despike_quaternions_undoes_a_one_frame_flip(self):
        q = np.stack([quat.axis_angle([0, 1, 0], 0.01 * i) for i in range(30)])[:, None, :]
        q[10, 0] = quat.axis_angle([0, 1, 0], 2.5)
        cleaned, replaced = filters.despike_quaternions(q, np.deg2rad(45), np.deg2rad(15))
        self.assertEqual(replaced, 1)
        self.assertLess(float(quat.angle(cleaned[10, 0], quat.axis_angle([0, 1, 0], 0.1))), 0.01)

    def test_segments_split_on_long_gaps(self):
        frames = np.array([0, 1, 2, 5, 6, 20, 21])
        self.assertEqual(filters.segments(frames, max_gap=2), [(0, 5), (5, 7)])


if __name__ == "__main__":
    unittest.main()
