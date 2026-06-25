"""Tests for the offline global identity resolver.

The forward tracker now survives crossings, but the offline resolver is the
global safety net: it optimises the whole timeline jointly so a brief bystander
spike can't capture the track, and it blends multiple identity cues
(frozen-gallery appearance + lighting/pose-invariant body shape + a known-
distractor penalty) so a frame where appearance alone is ambiguous is still
resolved correctly. These drive ``resolve_identity_track`` directly with
synthetic ``detection_frames`` (no pipeline run needed).
"""

from __future__ import annotations

import unittest

import numpy as np

from sam_3d_pose_estimation.pipeline import resolve_identity_track

DIAG = float(np.hypot(854, 480))


def bb(cx, cy=240, w=120, h=300):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def resolve(frames, **kw):
    return resolve_identity_track(frames, frame_diag=DIAG, **kw)


class OfflineResolverTests(unittest.TestCase):
    """Global, multi-cue identity resolution over detector frames."""

    def test_brief_distractor_spike_does_not_capture_track(self):
        """A bystander whose gallery briefly, marginally beats the patient must not
        win globally.

        The patient walks across the frame (candidate 0); a distractor parked far
        away (candidate 1) spikes its gallery just above the patient's for two
        frames. A greedy per-frame pick flips to it on those frames; the global
        resolver keeps the patient because the small gain never outweighs the cost
        of leaving (and re-joining) the patient track.
        """
        frames = []
        greedy = []
        for i in range(20):
            patient = {"bbox": bb(100 + i * 30), "gallery": 0.72}
            distractor = {"bbox": bb(800), "gallery": 0.74 if 8 <= i <= 9 else 0.30}
            frames.append({"frame_idx": i, "candidates": [patient, distractor]})
            greedy.append(0 if patient["gallery"] >= distractor["gallery"] else 1)
        res = resolve(frames)
        # A greedy matcher would flip to the distractor on the spike frames...
        self.assertIn(1, greedy)
        # ...but the global resolver stays locked on the patient throughout.
        self.assertTrue(all(r["state"] == "present" for r in res))
        self.assertTrue(all(r["cand_idx"] == 0 for r in res), [r["cand_idx"] for r in res])

    def test_body_shape_rescues_ambiguous_appearance(self):
        """When appearance dips below baseline but body shape is clearly the patient,
        the body-shape cue keeps the frames resolved as present (vs absent without it)."""

        def build(with_shape):
            frames = []
            for i in range(16):
                gallery = 0.50 if 6 <= i <= 9 else 0.72  # ambiguous appearance mid-clip
                cand = {"bbox": bb(100 + i * 30), "gallery": gallery}
                if with_shape:
                    cand["shape"] = 0.9
                frames.append({"frame_idx": i, "candidates": [cand]})
            return frames

        present_no = sum(1 for r in resolve(build(False)) if r["state"] == "present")
        with_shape = resolve(build(True))
        present_with = sum(1 for r in with_shape if r["state"] == "present")
        self.assertGreater(present_with, present_no)  # shape rescues the dip frames
        self.assertTrue(all(r["state"] == "present" for r in with_shape))

    def test_distractor_affinity_enforces_mutual_exclusivity(self):
        """A co-located bystander with a HIGHER raw gallery is rejected when it
        clearly matches a known distractor identity (mutual exclusivity)."""
        frames = []
        for i in range(12):
            px = 100 + i * 25
            patient = {"bbox": bb(px), "gallery": 0.66, "distractor": 0.05}
            bystander = {"bbox": bb(px + 8), "gallery": 0.80, "distractor": 0.92}
            frames.append({"frame_idx": i, "candidates": [patient, bystander]})
        res = resolve(frames)
        self.assertTrue(all(r["cand_idx"] == 0 for r in res), [r["cand_idx"] for r in res])

    def test_gallery_only_candidates_unchanged(self):
        """Backward compatibility: candidates carrying only `gallery` resolve as before."""
        frames = [
            {"frame_idx": i, "candidates": [{"bbox": bb(100 + i * 20), "gallery": 0.7}]}
            for i in range(10)
        ]
        res = resolve(frames)
        self.assertTrue(all(r["state"] == "present" and r["cand_idx"] == 0 for r in res))


if __name__ == "__main__":
    unittest.main()
