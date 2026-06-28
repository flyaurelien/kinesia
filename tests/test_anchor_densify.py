"""The chosen-subject anchor track is sampled (detect stride) but the
reconstruction processes every frame; densify_anchor_track must fill the gaps
so the subject lock holds on EVERY frame of a covered span — otherwise the live
identity tracker re-tracks the in-between frames and swaps at crossings."""

import unittest

import numpy as np

from sam_3d_pose_estimation.pipeline import densify_anchor_track


def _box(v: float) -> np.ndarray:
    return np.array([v, v, v + 10, v + 10], dtype=np.float32)


class TestDensifyAnchorTrack(unittest.TestCase):
    def test_fills_every_frame_in_a_strided_span(self):
        # Anchors every 5 frames (the detect stride) over 0..95.
        track = [(i * 5, _box(float(i))) for i in range(20)]
        dense = densify_anchor_track(track, (1000, 1000, 3))
        frames = [f for f, _ in dense]
        # Every integer frame in the covered span now has an anchor.
        self.assertEqual(frames, list(range(0, 96)))
        # Original sample boxes are preserved exactly.
        d = dict(dense)
        self.assertTrue(np.allclose(d[0], _box(0.0)))
        self.assertTrue(np.allclose(d[5], _box(1.0)))
        # In-between boxes are interpolated and monotonic between the samples.
        self.assertTrue(d[0][0] < d[2][0] < d[5][0])

    def test_leaves_genuine_absences_uncovered(self):
        # Strided anchors 0..95, then the subject leaves and reappears at 400.
        track = [(i * 5, _box(float(i))) for i in range(20)]
        track.append((400, _box(0.0)))
        dense = densify_anchor_track(track, (1000, 1000, 3))
        frames = set(f for f, _ in dense)
        # The covered span is filled densely…
        self.assertTrue(all(f in frames for f in range(0, 96)))
        # …but the long gap (subject absent) is NOT interpolated — re-entry is
        # left to the tracker rather than extrapolating a stale box.
        self.assertNotIn(200, frames)
        self.assertIn(400, frames)

    def test_noop_on_trivial_tracks(self):
        self.assertEqual(densify_anchor_track([], (10, 10, 3)), [])
        one = [(7, _box(1.0))]
        self.assertEqual([f for f, _ in densify_anchor_track(one, (10, 10, 3))], [7])


if __name__ == "__main__":
    unittest.main()
