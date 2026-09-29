"""Reading and preparing videos.

Every analysis works on one *normalized* copy of the upload: H.264, constant
frame rate, the display rotation baked into the pixels, at most 1920 pixels
wide and at most ``MAX_FPS`` frames per second. The GPU job, the scene build and
the browser then all agree on what "frame 812" is.

Phones store a display-rotation flag that OpenCV ignores unless asked, which
hands the models a sideways picture; normalizing with ffmpeg (which honours the
flag) removes that trap for every later reader.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

MAX_WIDTH = 1920
MAX_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float

    def to_json(self) -> dict:
        return asdict(self)


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required (install ffmpeg and put it on PATH)")
    return path


def probe(path: Path, *, count_frames: bool = False) -> VideoInfo:
    """Read size, frame rate, frame count and duration with ffprobe."""
    args = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        args.append("-count_frames")
    args += [
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration"
        ":stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]  # fmt: skip
    data = json.loads(subprocess.run(args, capture_output=True, check=True, text=True).stdout)
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or round(duration * fps))
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if rotation % 180:
        width, height = height, width
    return VideoInfo(width, height, fps, frames, duration)


def normalization_plan(info: VideoInfo) -> tuple[int, int, float]:
    """Target size (even numbers) and frame rate for the normalized copy."""
    scale = min(1.0, MAX_WIDTH / max(info.width, 1))
    width = int(round(info.width * scale / 2)) * 2
    height = int(round(info.height * scale / 2)) * 2
    fps = info.fps
    if fps > MAX_FPS:
        # Keep an integer decimation of the source rate (120 -> 30, 50 -> 25, 60 -> 30).
        step = int(np.ceil(fps / MAX_FPS - 1e-6))
        fps = fps / step
    return width, height, fps


def normalize(source: Path, target: Path) -> VideoInfo:
    """Write the normalized H.264 copy of ``source`` and return its exact info."""
    info = probe(source)
    width, height, fps = normalization_plan(info)
    partial = target.with_name(target.stem + ".part" + target.suffix)
    args = [
        _tool("ffmpeg"), "-y", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn",
        "-vf", f"fps={fps:.6f},scale={width}:{height}:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-g", "30", "-bf", "0", "-movflags", "+faststart",
        str(partial),
    ]  # fmt: skip
    subprocess.run(args, check=True)
    partial.replace(target)
    return probe(target, count_frames=True)


def write_poster(video: Path, target: Path, at_fraction: float = 0.3, width: int = 640) -> None:
    """Save one representative frame as a JPEG thumbnail."""
    capture = open_video(video)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(count * at_fraction)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return
    scale = width / frame.shape[1]
    frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def open_video(path: Path | str) -> cv2.VideoCapture:
    """Open a video with OpenCV, honouring the display rotation flag."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except (AttributeError, cv2.error):  # pragma: no cover - very old OpenCV
        pass
    return capture


def read_frames(path: Path, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, rgb_frame)`` for frames ``start`` .. ``stop - 1``, decoding in order."""
    capture = open_video(path)
    try:
        # Skip by decoding rather than seeking: seeks can land a frame off.
        for _ in range(start):
            if not capture.grab():
                return
        index = start
        while stop is None or index < stop:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
