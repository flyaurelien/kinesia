"""Floor fitting, depth-to-floor correction and joining masklets into people."""

import unittest

import numpy as np

from kinesia.scene.ground import Plane, WorldFrame, depth_scales, fit_plane, smooth_scales
from kinesia.scene.identity import Fragment, link_fragments


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

    def test_world_frame_is_z_up_with_the_floor_at_zero(self):
        plane = Plane(np.array([0.0, -1.0, 0.0]), 1.5)
        world = WorldFrame.from_floor(plane, np.array([0.0, 1.5, 10.0]))
        on_floor = world.apply(np.array([[2.0, 1.5, 12.0]]))[0]
        self.assertAlmostEqual(on_floor[2], 0.0, places=9)
        self.assertAlmostEqual(world.camera_position[2], 1.5, places=9)
        self.assertTrue(np.allclose(np.linalg.det(world.rotation), 1.0))


def fragment(track, frames, direction, start_xy):
    frames = np.asarray(frames)
    embed = np.tile(direction / np.linalg.norm(direction), (len(frames), 1))
    positions = np.column_stack([np.full(len(frames), start_xy[0]), np.full(len(frames), start_xy[1]), np.zeros(len(frames))])
    return Fragment(track, frames, embed, positions)


class IdentityTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.looks = [rng.normal(size=16) for _ in range(3)]

    def test_rejoins_a_person_after_they_leave_and_return(self):
        a = fragment(0, range(0, 100), self.looks[0], (0, 0))
        b = fragment(1, range(0, 250), self.looks[1], (5, 5))  # someone else, always visible
        c = fragment(2, range(160, 250), self.looks[0] + 0.02, (3, 0))  # person 0 back
        result = link_fragments([a, b, c], rate=30.0)
        self.assertIn([0, 2], result.groups)
        self.assertIn([1], result.groups)

    def test_never_joins_fragments_seen_at_the_same_time(self):
        a = fragment(0, range(0, 100), self.looks[0], (0, 0))
        b = fragment(1, range(50, 150), self.looks[0], (0, 0))  # identical look, overlapping
        result = link_fragments([a, b], rate=30.0)
        self.assertEqual(sorted(result.groups), [[0], [1]])

    def test_refuses_a_teleport(self):
        a = fragment(0, range(0, 60), self.looks[0], (0, 0))
        other = fragment(5, range(0, 200), self.looks[1], (9, 9))
        b = fragment(1, range(62, 200), self.looks[0], (40, 0))  # 40 m away 2 frames later
        result = link_fragments([a, other, b], rate=30.0)
        self.assertIn([0], result.groups)
        self.assertIn([1], result.groups)


if __name__ == "__main__":
    unittest.main()
