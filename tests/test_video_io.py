"""Reading a rotated video must yield the picture a viewer sees.

Phones store portrait clips as landscape plus a rotation flag. Everything
downstream treats image "up" as world up, so reading the raw frames feeds the
models a sideways scene and silently invalidates the reconstruction and the
whole clinical gait layer.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2

from sam_3d_pose_estimation.video_io import display_rotation, open_video

WIDTH, HEIGHT = 320, 180  # stored landscape; displays as 180x320 once rotated


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def write_rotated_clip(path: Path, rotation: int) -> None:
    """A short clip stored WIDTHxHEIGHT carrying a display-rotation flag.

    The flag has to live in the stream's display matrix, the way a phone writes
    it — the legacy `rotate=` metadata tag is ignored by current readers. ffmpeg
    only stamps the matrix on a remux, hence the two steps.
    """
    plain = path.with_name(f"plain_{path.name}")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={WIDTH}x{HEIGHT}:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(plain),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(plain), "-c", "copy", str(path)],
        check=True,
        capture_output=True,
    )


@unittest.skipUnless(ffmpeg_available(), "ffmpeg is required to build the fixture")
class TestRotatedVideoReading(unittest.TestCase):
    def test_rotated_clip_is_read_upright(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "portrait.mp4"
            write_rotated_clip(clip, 90)
            if abs(display_rotation(clip)) < 1.0:
                self.skipTest("this ffmpeg/OpenCV pair does not carry the rotation flag")

            capture = open_video(clip)
            try:
                ok, frame = capture.read()
                self.assertTrue(ok)
                declared = (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            finally:
                capture.release()

            # The stored frame is landscape; read upright it must be portrait,
            # and the reported size must agree with the pixels handed back.
            self.assertEqual(frame.shape[:2], (WIDTH, HEIGHT))
            self.assertEqual(declared, (HEIGHT, WIDTH))

    def test_unrotated_clip_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "landscape.mp4"
            write_rotated_clip(clip, 0)
            capture = open_video(clip)
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (HEIGHT, WIDTH))


if __name__ == "__main__":
    unittest.main()
