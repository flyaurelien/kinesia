"""Video reading that respects the container's display rotation.

Phones record in landscape and store a rotation flag; players and browsers
honour it, so a portrait clip looks portrait everywhere a person checks it.
OpenCV does not honour it unless asked, and reading such a video raw hands the
models a sideways frame — people lying down.

That is not merely a quality loss. The whole pipeline treats image "up" as
world up: the reconstruction, the ground plane, the upright correction and the
clinical gait layer (which measures against world Z) are all wrong by 90
degrees, while the viewer plays the ORIGINAL file in a <video> element that
does honour the flag — so the boxes no longer sit on the people.

Every read path must therefore go through `open_video`.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Open a video with the display rotation applied.

    With auto-orientation on, both the frames and the reported frame size come
    back already rotated, so callers can treat the result as the picture a
    viewer sees. On an OpenCV build that ignores the flag the frames stay raw —
    the previous behaviour, not a new failure mode.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except (AttributeError, cv2.error):  # pragma: no cover - old OpenCV builds
        pass
    return capture


def display_rotation(video_path: str | Path) -> float:
    """The container's rotation flag in degrees (0 when absent/unsupported)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
    except (AttributeError, cv2.error):  # pragma: no cover - old OpenCV builds
        return 0.0
    finally:
        capture.release()
