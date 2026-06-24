"""Render the "Tracking box" MP4 — the source video with the subject's bounding
box drawn per frame, exactly like the viewer's *Tracking box* view (a dashed
yellow box, no label, no mesh). Reads a run's ``run_metadata.json`` for the source
video, fps, and the per-frame ``bbox_xyxy`` records, draws the box, and encodes a
browser-friendly H.264 MP4 at the run's output fps via ffmpeg.

    python -m sam_3d_pose_estimation.render_tracking_video --run-dir output/<run> \
        --out output/<run>/<name>_tracking.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Viewer's box style (app/globals.css .vo-box): dashed yellow #facc15, width 2.
_BOX_BGR = (21, 204, 250)  # #facc15 in BGR
_BOX_THICKNESS = 2
_DASH_ON, _DASH_OFF = 10, 7  # dashed look, scaled a touch up for video


def _valid_bbox(b) -> bool:
    return (
        isinstance(b, (list, tuple))
        and len(b) >= 4
        and all(isinstance(v, (int, float)) for v in b[:4])
        and float(b[2]) > float(b[0])
        and float(b[3]) > float(b[1])
    )


def _dashed_line(img, p1, p2) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1:
        return
    step = _DASH_ON + _DASH_OFF
    n = max(1, int(dist // step))
    for i in range(n + 1):
        a = (i * step) / dist
        b = min(1.0, (i * step + _DASH_ON) / dist)
        if a >= 1.0:
            break
        xa, ya = int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)
        xb, yb = int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)
        cv2.line(img, (xa, ya), (xb, yb), _BOX_BGR, _BOX_THICKNESS, cv2.LINE_AA)


def _draw_box(img, bbox) -> None:
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
    _dashed_line(img, (x1, y1), (x2, y1))
    _dashed_line(img, (x2, y1), (x2, y2))
    _dashed_line(img, (x2, y2), (x1, y2))
    _dashed_line(img, (x1, y2), (x1, y1))


def _frame_box_timeline(records: list[dict]) -> list[tuple[int, object]]:
    """[(video_frame, bbox_or_None)] sorted by frame; a box persists until the next
    record, so sparse runs (frame_step > 1) still show a steady box."""
    out: list[tuple[int, object]] = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        vf = r.get("video_frame")
        vf = int(vf) if isinstance(vf, (int, float)) else i
        b = r.get("bbox_xyxy")
        out.append((vf, b if _valid_bbox(b) else None))
    out.sort(key=lambda t: t[0])
    return out


def render(run_dir: Path, out_path: Path, video_override: str | None = None) -> int:
    """Render ``out_path`` (an H.264 MP4) from the run's source video + per-frame boxes.

    ``run_dir`` is the run output folder holding ``run_metadata.json`` (source video,
    fps, and the per-frame ``bbox_xyxy`` records). ``video_override`` forces a source
    video instead of the one recorded in the metadata. Returns a process exit code
    (0 on success, non-zero on a missing/unreadable input or ffmpeg failure)."""
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        print(f"error: {meta_path} not found", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text())

    video_path = Path(video_override or meta.get("video_input") or "")
    if not video_path.exists():
        print(f"error: source video not found: {video_path}", file=sys.stderr)
        return 2
    fps = float(meta.get("fps_output") or meta.get("fps_input") or 30.0)
    timeline = _frame_box_timeline(meta.get("records") or [])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        return 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or meta.get("video_width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or meta.get("video_height") or 0)
    if not width or not height:
        print("error: could not determine video dimensions", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "20", "-movflags", "+faststart",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
    )

    # Walk the source frames; hold the latest record's box until the next record.
    ti = 0
    cur_box = None
    n = 0
    drawn = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            while ti < len(timeline) and timeline[ti][0] <= n:
                cur_box = timeline[ti][1]
                ti += 1
            if cur_box is not None:
                _draw_box(frame, cur_box)
                drawn += 1
            ffmpeg.stdin.write(frame.tobytes())
            n += 1
    finally:
        cap.release()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()
        ffmpeg.wait()

    if ffmpeg.returncode != 0:
        print(f"error: ffmpeg exited {ffmpeg.returncode}", file=sys.stderr)
        return 1
    tmp_path.replace(out_path)
    print(json.dumps({"ok": True, "out": str(out_path), "frames": n, "boxed": drawn, "fps": fps}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="render_tracking_video", description=__doc__)
    p.add_argument("--run-dir", required=True, help="run output dir containing run_metadata.json")
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--video", default=None, help="override source video path")
    a = p.parse_args()
    return render(Path(a.run_dir), Path(a.out), a.video)


if __name__ == "__main__":
    raise SystemExit(main())
