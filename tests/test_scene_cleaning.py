"""Cleaning SAM 3.1 masklets before they become people: glitches, hand-overs, spectators."""

import unittest

import numpy as np

from kinesia.scene import build
from kinesia.scene.keypoints import LEFT_HIP, RIGHT_HIP

RATE = 25.0
FOCAL, CX, CY = 1500.0, 960.0, 540.0


def masklet(positions: np.ndarray, track: int = 0, first: int = 0) -> tuple[dict, np.ndarray]:
    """Rows of one masklet whose pelvis follows ``positions`` (camera coordinates, metres)."""
    n = len(positions)
    points = np.zeros((n, 70, 3))
    points[:, LEFT_HIP] = positions + [-0.1, 0, 0]
    points[:, RIGHT_HIP] = positions + [0.1, 0, 0]
    kp2d = np.zeros((n, 70, 2))
    for joint in (LEFT_HIP, RIGHT_HIP):
        kp2d[:, joint, 0] = CX + FOCAL * points[:, joint, 0] / points[:, joint, 2]
        kp2d[:, joint, 1] = CY + FOCAL * points[:, joint, 1] / points[:, joint, 2]
    height = FOCAL * 1.75 / positions[:, 2]
    x = CX + FOCAL * positions[:, 0] / positions[:, 2]
    box = np.column_stack([x - 0.2 * height, CY - 0.5 * height, x + 0.2 * height, CY + 0.5 * height])
    rows = {"track": np.full(n, track), "frame": np.arange(first, first + n), "kp2d": kp2d, "box": box}
    return rows, points


def walk(n: int, speed: float, depth: float = 15.0) -> np.ndarray:
    """Walking sideways at ``speed`` m/s, ``depth`` metres from the camera."""
    t = np.arange(n) / RATE
    return np.column_stack([speed * t - 3.0, np.full(n, 0.5), np.full(n, depth)])


class PiecesTest(unittest.TestCase):
    def test_a_sprinter_is_one_piece(self):
        rows, points = masklet(walk(80, speed=9.0))
        piece = build._pieces(rows, points, RATE)
        self.assertEqual(set(piece.tolist()), {0})

    def test_a_ballooning_mask_is_a_glitch_not_a_cut(self):
        positions = walk(80, speed=2.0)
        positions[40:42, 2] += 8.0  # the mask swallowed a neighbour for two frames
        rows, points = masklet(positions)
        piece = build._pieces(rows, points, RATE)
        self.assertEqual(piece[40:42].tolist(), [-1, -1])
        self.assertEqual(set(piece[piece >= 0].tolist()), {0})

    def test_a_hand_over_to_someone_far_behind_is_cut(self):
        positions = walk(100, speed=1.0, depth=20.0)
        positions[50:, 2] = 40.0  # now a spectator in the stands
        rows, points = masklet(positions)
        piece = build._pieces(rows, points, RATE)
        self.assertEqual(len(set(piece.tolist())), 2)
        self.assertTrue(np.all(piece[:48] == piece[0]) and np.all(piece[52:] == piece[-1]))

    def test_a_hand_over_to_a_neighbour_is_cut(self):
        positions = walk(100, speed=1.0)
        positions[50:, 0] += 4.0  # a player four metres to the side, same distance
        rows, points = masklet(positions)
        piece = build._pieces(rows, points, RATE)
        self.assertEqual(len(set(piece.tolist())), 2)

    def test_masklets_get_distinct_pieces(self):
        a, pa = masklet(walk(30, speed=1.0), track=3)
        b, pb = masklet(walk(30, speed=1.0, depth=12.0), track=8)
        rows = {key: np.concatenate([a[key], b[key]]) for key in a}
        piece = build._pieces(rows, np.concatenate([pa, pb]), RATE)
        self.assertEqual(len(set(piece[:30].tolist())), 1)
        self.assertEqual(len(set(piece[30:].tolist())), 1)
        self.assertNotEqual(piece[0], piece[30])


class FloorTest(unittest.TestCase):
    def test_someone_off_the_floor_is_left_out_and_gaps_are_filled(self):
        n = 40
        rows = {"frame": np.concatenate([np.arange(n), np.arange(n)])}
        piece = np.repeat([0, 1], n)
        depth = np.concatenate([np.linspace(0.9, 1.1, n), np.full(n, 1.7)])
        reachable = np.ones(2 * n, dtype=bool)
        reachable[5:8] = False  # piece 0: feet hidden for three frames
        reachable[n : n + 30] = False  # piece 1: in the stands
        depth[5:8] = 1.7
        cut = np.zeros(2 * n, dtype=bool)
        keep = build._keep_on_floor(rows, piece, depth, reachable, cut)
        self.assertTrue(keep[:n].all())
        self.assertFalse(keep[n:].any())
        np.testing.assert_allclose(depth[5:8], np.linspace(0.9, 1.1, n)[5:8], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
