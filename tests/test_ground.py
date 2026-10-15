"""Floor fitting and depth-to-floor correction."""

import unittest

import numpy as np

from kinesia.scene.ground import Plane, WorldFrame, depth_scales, find_floor, fit_plane, smooth_scales
from kinesia.scene.keypoints import FEET


class GroundTest(unittest.TestCase):
    def test_fit_plane_ignores_airborne_points(self):
        rng = np.random.default_rng(0)
        # camera at the origin, floor 1.5 m below it (camera y points down)
        floor = np.column_stack([rng.uniform(-5, 5, 400), np.full(400, 1.5), rng.uniform(4, 30, 400)])
        floor += rng.normal(0, 0.02, floor.shape)
        jumps = floor[:40] - np.array([0, 0.4, 0])
        plane, inliers = fit_plane(np.concatenate([floor, jumps]))
        self.assertAlmostEqual(plane.offset, 1.5, delta=0.02)
        self.assertGreater(plane.normal @ np.array([0, -1.0, 0]), 0.999)
        self.assertFalse(inliers[-40:].any())

    def test_depth_scale_puts_the_foot_on_the_floor(self):
        plane = Plane(np.array([0.0, -1.0, 0.0]), 1.5)
        # a foot seen along a ray, reconstructed 20% too close
        true_foot = np.array([1.0, 1.5, 10.0])
        k = depth_scales((true_foot / 1.2)[None], plane)[0]
        self.assertAlmostEqual(k, 1.2, places=6)

    def test_smoothed_scales_ignore_a_short_jump(self):
        k = np.ones(90)
        k[40:52] = 1.4  # 0.4 s off the floor
        smoothed = smooth_scales(k, 30.0)
        self.assertLess(np.abs(smoothed - 1.0).max(), 0.03)

    def test_feet_along_one_line_take_the_floor_tilt_from_the_trunks(self):
        # Two people barely moving: the feet of one on the floor 1.5 m below
        # the camera, the other's raised on a machine, all along one line.
        rng = np.random.default_rng(1)
        points = np.zeros((200, 70, 3))
        depth = rng.uniform(3.0, 4.0, 200)
        height = np.where(np.arange(200) < 100, 1.5, 1.2)
        points[:, list(FEET)] = np.column_stack([np.zeros(200), height, depth])[:, None, :]
        points[:, list(FEET), 0] += rng.normal(0, 0.03, (200, len(FEET)))
        up = np.array([0.0, -1.0, 0.0])
        plane = find_floor(points, up=up)
        self.assertGreater(plane.normal @ up, 0.999)
        self.assertAlmostEqual(plane.offset, 1.5, delta=0.05)

    def test_world_frame_is_z_up_with_the_floor_at_zero(self):
        plane = Plane(np.array([0.0, -1.0, 0.0]), 1.5)
        world = WorldFrame.from_floor(plane, np.array([0.0, 1.5, 10.0]))
        on_floor = world.apply(np.array([[2.0, 1.5, 12.0]]))[0]
        self.assertAlmostEqual(on_floor[2], 0.0, places=9)
        self.assertAlmostEqual(world.camera_position[2], 1.5, places=9)
        self.assertTrue(np.allclose(np.linalg.det(world.rotation), 1.0))


if __name__ == "__main__":
    unittest.main()
