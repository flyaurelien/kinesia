"""Tests for the mask-restricted appearance descriptor + embedding similarity.

A box always contains some background behind the person, which pollutes a plain
bbox colour histogram. Restricting the histogram to the SAM3 subject mask gives
a background-free identity cue. `embedding_similarity` is the cosine the tracker
uses when detections carry a learned SAM3 object embedding.
"""

from __future__ import annotations

import unittest

import numpy as np

from sam_3d_pose_estimation.pipeline import (
    appearance_similarity_score,
    embedding_similarity,
    extract_bbox_appearance_hist,
)

SHAPE = (360, 480, 3)
RED = (40, 40, 200)     # BGR background
BLUE = (200, 90, 40)    # BGR "person" clothing


def box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class AppearanceDescriptorTests(unittest.TestCase):
    def test_mask_restricts_histogram_to_the_subject(self):
        """A mask-restricted histogram drops the background, so it matches a clean
        reference of the subject's colour better than the background-polluted box."""
        # Pure-blue reference (the subject's clothing colour).
        blue_frame = np.full(SHAPE, BLUE, dtype=np.uint8)
        ref = extract_bbox_appearance_hist(blue_frame, box(120, 60, 260, 340))

        # A box that contains the blue person on the left and red background on the right.
        frame = np.full(SHAPE, RED, dtype=np.uint8)
        frame[60:340, 120:200] = BLUE
        mask = np.zeros(SHAPE[:2], dtype=bool)
        mask[60:340, 120:200] = True
        bbox = box(120, 60, 260, 340)

        h_nomask = extract_bbox_appearance_hist(frame, bbox)
        h_mask = extract_bbox_appearance_hist(frame, bbox, mask=mask)

        s_nomask = appearance_similarity_score(ref, h_nomask)
        s_mask = appearance_similarity_score(ref, h_mask)
        self.assertIsNotNone(s_mask)
        self.assertIsNotNone(s_nomask)
        # Background-free descriptor is a markedly cleaner match to the subject.
        self.assertGreater(s_mask, s_nomask)

    def test_embedding_similarity_cosine(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(embedding_similarity(a, a), 1.0, places=5)
        self.assertAlmostEqual(embedding_similarity(a, np.array([-1.0, 0.0, 0.0])), 0.0, places=5)
        self.assertAlmostEqual(embedding_similarity(a, np.array([0.0, 1.0, 0.0])), 0.5, places=5)
        self.assertIsNone(embedding_similarity(a, None))
        self.assertIsNone(embedding_similarity(a, np.array([1.0, 0.0])))  # size mismatch
        self.assertIsNone(embedding_similarity(a, np.zeros(3, dtype=np.float32)))  # zero vector


if __name__ == "__main__":
    unittest.main()
