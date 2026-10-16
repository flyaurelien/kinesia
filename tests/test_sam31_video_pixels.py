"""SAM 3.1 must see video frames in [-1, 1], as it sees images.

The OpenCV video loader of the pinned Meta revision normalises 0-255 pixels
with the 0.5 mean and std meant for 0-1 pixels, so a white frame reached the
model as 509. A tiny black-and-white clip goes through Meta's loader with
Kinesia's patch. Runs when ``SAM3_SOURCE`` points to a checkout of
facebookresearch/sam3 (skipped otherwise).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE = os.environ.get("SAM3_SOURCE", "")


@unittest.skipUnless(SOURCE and os.path.isdir(SOURCE), "set SAM3_SOURCE to a facebookresearch/sam3 checkout")
class VideoPixelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2

        sys.path.insert(0, SOURCE)
        from kinesia.inference.tracking import scale_video_pixels

        scale_video_pixels()
        cls.folder = tempfile.TemporaryDirectory()
        cls.video = str(Path(cls.folder.name) / "clip.mp4")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for value in (0, 255, 0, 255):
            writer.write(np.full((64, 64, 3), value, np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_frames_span_minus_one_to_one(self):
        from sam3.model.io_utils import load_resource_as_video_frames

        frames, height, width = load_resource_as_video_frames(
            self.video, image_size=32, offload_video_to_cpu=True, video_loader_type="cv2"
        )
        self.assertEqual((len(frames), height, width), (4, 64, 64))
        means = frames.float().mean(dim=(1, 2, 3)).tolist()
        np.testing.assert_allclose(means, [-1, 1, -1, 1], atol=0.03)

    def test_applying_twice_changes_nothing(self):
        from kinesia.inference.tracking import scale_video_pixels

        self.assertFalse(scale_video_pixels())


if __name__ == "__main__":
    unittest.main()
