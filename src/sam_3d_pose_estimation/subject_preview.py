from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pipeline import auto_initialize_patient_bbox, detect_sam3_prompt_candidates
from .sam3d_runtime import select_device, try_build_human_detector
from .workspace import project_root_from


DEFAULT_PROJECT_ROOT = project_root_from(Path(__file__))
DEFAULT_SAM3_CODE_ROOT = Path(os.environ.get("SAM3_CODE_ROOT", DEFAULT_PROJECT_ROOT / "vendor" / "sam3-main"))


def parse_prompts(raw: str) -> tuple[str, ...]:
    """Split a comma-separated prompt string, defaulting to ("person",) if empty."""
    prompts = tuple(item.strip() for item in raw.split(",") if item.strip())
    return prompts or ("person",)


def parse_float_list(raw: str | None) -> list[float]:
    """Parse a comma-separated list of floats, silently skipping invalid entries."""
    if not raw:
        return []
    out: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def parse_ranges(raw: str | None) -> list[tuple[float, float]]:
    """Parse a JSON list of [start, end] second pairs (kept regions)."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    ranges: list[tuple[float, float]] = []
    for entry in data if isinstance(data, list) else []:
        try:
            start, end = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            ranges.append((max(0.0, start), end))
    return ranges


def _video_meta(cap: cv2.VideoCapture, frame: np.ndarray | None = None) -> dict[str, Any]:
    """Read fps/frame-count/dimensions from a capture, falling back to a decoded
    frame's shape when the container does not report width/height."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame is not None:
        width = width or int(frame.shape[1])
        height = height or int(frame.shape[0])
    return {"fps": fps, "total_frames": total_frames, "video_width": width, "video_height": height}


def read_frame_at_time(video_input: Path, frame_sec: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode the frame nearest to frame_sec, falling back to frame 0 if the seek
    fails. Returns the BGR frame plus its resolved index/timing metadata."""
    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_input}")
    try:
        meta = _video_meta(cap)
        fps = meta["fps"]
        total_frames = meta["total_frames"]
        target_frame = int(round(max(0.0, frame_sec) * max(1.0, fps)))
        if total_frames > 0:
            target_frame = int(np.clip(target_frame, 0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            target_frame = 0
        if not ok:
            raise RuntimeError(f"Cannot read preview frame from video: {video_input}")
        return frame, {
            "fps": fps,
            "frame_index": target_frame,
            "frame_sec": target_frame / max(1.0, fps),
            "total_frames": total_frames,
            "video_width": meta["video_width"] or int(frame.shape[1]),
            "video_height": meta["video_height"] or int(frame.shape[0]),
        }
    finally:
        cap.release()


def bbox_payload(bbox: np.ndarray | None, width: int, height: int) -> dict[str, Any] | None:
    """Serialize an xyxy bbox for the UI as both pixel coords and a normalized
    (0..1) box; returns None when no bbox was detected."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox.tolist()]
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return {
        "xyxy": [x1, y1, x2, y2],
        "box": {
            "x": max(0.0, min(1.0, x1 / w)),
            "y": max(0.0, min(1.0, y1 / h)),
            "width": max(0.0, min(1.0, (x2 - x1) / w)),
            "height": max(0.0, min(1.0, (y2 - y1) / h)),
        },
    }


def _detect_at(
    frame: np.ndarray,
    *,
    detector: Any,
    auto_init_mode: str,
    auto_select_strategy: str,
    auto_detector_threshold: float,
    sam3_text_prompts: tuple[str, ...],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run patient auto-detection on a single frame (thin wrapper over the pipeline)."""
    return auto_initialize_patient_bbox(
        frame_bgr=frame,
        auto_init_mode=auto_init_mode,
        auto_select_strategy=auto_select_strategy,
        auto_detector_threshold=auto_detector_threshold,
        sam3_text_prompts=sam3_text_prompts,
        sam3_detector=detector,
    )


def _spread_pick(items: list[Any], k: int) -> list[Any]:
    """Pick up to k items spread evenly across the list (keeps first & last)."""
    n = len(items)
    if k <= 0:
        return []
    if n <= k:
        return items
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def build_scan_times(
    *,
    duration_sec: float,
    preferred: list[float],
    kept_ranges: list[tuple[float, float]],
    scan_step: float,
    scan_budget: int,
) -> list[float]:
    """Times to probe. When the caller passes explicit anchors (the wizard's
    numbered markers), probe EXACTLY those — the user picked those moments and
    expects each one checked, in order. Mixing in a grid (the old behaviour)
    scanned different frames than the markers shown, so a frame the user could
    see a subject on was never actually looked at. With no anchors, fall back to
    an even grid over the kept regions (the "scan the whole video" case)."""
    anchors = sorted({round(max(0.0, t), 3) for t in preferred})
    if anchors:
        return anchors[:scan_budget] if scan_budget > 0 else anchors

    ranges = kept_ranges or [(0.0, max(0.0, duration_sec))]
    kept = sum(max(0.0, e - s) for s, e in ranges) or max(0.001, duration_sec)
    step = scan_step if scan_step > 0 else max(1.0, kept / max(1, scan_budget))
    times: set[float] = set()
    for start, end in ranges:
        t = start
        while t < end:
            times.add(round(t, 3))
            t += step
    ordered = sorted(times)
    if len(ordered) <= scan_budget:
        return ordered
    return sorted(set(_spread_pick(ordered, scan_budget)))


def locate_subjects(
    video_input: Path,
    *,
    detector: Any,
    preferred_secs: list[float],
    kept_ranges: list[tuple[float, float]],
    scan_step: float,
    scan_budget: int,
    max_results: int,
    auto_init_mode: str,
    auto_select_strategy: str,
    auto_detector_threshold: float,
    sam3_text_prompts: tuple[str, ...],
) -> dict[str, Any]:
    """Scan the video at the planned probe times and return up to max_results
    evenly-spread detections (or probed misses, so the UI always has frames)."""
    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_input}")
    try:
        meta = _video_meta(cap)
        fps = meta["fps"]
        total_frames = meta["total_frames"]
        duration = (total_frames / fps) if total_frames > 0 else 0.0

        scan_times = build_scan_times(
            duration_sec=duration,
            preferred=preferred_secs,
            kept_ranges=kept_ranges,
            scan_step=scan_step,
            scan_budget=scan_budget,
        )

        width = meta["video_width"]
        height = meta["video_height"]
        hits: list[dict[str, Any]] = []
        misses: list[dict[str, Any]] = []
        for sec in scan_times:
            target_frame = int(round(sec * max(1.0, fps)))
            if total_frames > 0:
                target_frame = int(np.clip(target_frame, 0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok:
                continue
            if not width or not height:
                width, height = int(frame.shape[1]), int(frame.shape[0])
            # Multi-subject: return EVERY detected person on the frame (one SAM3
            # pass), not just the auto-selected "patient". The UI shows them all
            # as numbered boxes and the user picks. `detection` keeps the default
            # highlight (largest box) for backward compatibility.
            raw_candidates = (
                detect_sam3_prompt_candidates(
                    frame,
                    sam3_detector=detector,
                    auto_detector_threshold=auto_detector_threshold,
                    sam3_text_prompts=sam3_text_prompts,
                )
                if detector is not None
                else []
            )
            candidates = [
                payload
                for payload in (bbox_payload(c["bbox"], width, height) for c in raw_candidates)
                if payload is not None
            ]
            best = (
                max(candidates, key=lambda b: float(b["box"]["width"]) * float(b["box"]["height"]))
                if candidates
                else None
            )
            record = {
                "frame_sec": target_frame / max(1.0, fps),
                "frame_index": target_frame,
                "detection": best,
                "candidates": candidates,
                "info": {
                    "num_candidates": len(candidates),
                    "selected_source": (raw_candidates[0]["source"] if raw_candidates else None),
                },
            }
            (hits if candidates else misses).append(record)

        # Show one tile per requested sample: take the located hits first, then
        # top up with evenly-spread probed misses (each flagged no-subject) so the
        # preview never silently drops tiles when an anchor had no detection.
        # When there are already enough hits (the CUDA broad sweep), no misses are
        # added — behaviour is unchanged there.
        results = _spread_pick(hits, max_results)
        if len(results) < max_results and misses:
            results = results + _spread_pick(misses, max_results - len(results))
        results = sorted(results, key=lambda record: record["frame_sec"])

        return {
            "ok": any(r["detection"] is not None for r in results),
            "mode": "locate",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "video_width": width,
            "video_height": height,
            "scanned": len(scan_times),
            "hit_count": len(hits),
            "results": results,
        }
    finally:
        cap.release()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for single-frame preview and locate scans."""
    parser = argparse.ArgumentParser(description="Preview Kinesia patient auto-detection on video frames.")
    parser.add_argument("--video-input", type=Path, required=True)
    parser.add_argument("--mode", choices=["single", "locate"], default="single")
    parser.add_argument("--frame-sec", type=float, default=0.0)
    # locate mode: preferred anchor times + scan controls
    parser.add_argument("--frame-secs", type=str, default="")
    parser.add_argument("--kept-ranges", type=str, default="")
    parser.add_argument("--scan-step", type=float, default=0.0)
    parser.add_argument("--scan-budget", type=int, default=44)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--auto-init-mode", choices=["smart", "sam3"], default="sam3")
    parser.add_argument(
        "--auto-select-strategy",
        choices=["patient", "largest", "leftmost", "rightmost", "center", "tightest"],
        default="patient",
    )
    parser.add_argument("--auto-detector-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-code-root", type=Path, default=DEFAULT_SAM3_CODE_ROOT)
    parser.add_argument("--sam3-text-prompts", type=str, default="person")
    parser.add_argument("--force-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: detect the patient on a frame (or scan) and print one JSON
    line to stdout. Returns 0 on success, 1 on error (error JSON also printed)."""
    args = build_parser().parse_args(argv)
    prompts = parse_prompts(args.sam3_text_prompts)
    try:
        # Keep stdout clean for the final JSON: redirect the noisy model/runtime
        # logging to stderr while we build the result, then print on real stdout.
        with contextlib.redirect_stdout(sys.stderr):
            device = select_device(force_cpu=args.force_cpu)
            detector = None
            if args.auto_init_mode in {"smart", "sam3"}:
                detector = try_build_human_detector(
                    detector_name="sam3",
                    device=device,
                    sam3_code_root=args.sam3_code_root,
                )

            if args.mode == "locate":
                preferred_secs = parse_float_list(args.frame_secs)
                scan_budget = max(1, args.scan_budget)
                # SAM3 detection costs ~10s/frame off CUDA (on Apple Silicon, MPS
                # falls back to CPU for several ops), so the full ~44-frame sweep
                # takes many minutes and reads as "stuck / never finishes". Off
                # CUDA, scan the caller's anchors plus a small margin (enough
                # redundancy to still surface one hit per requested tile) so the
                # preview completes in ~1 min; CUDA keeps the broad, more-robust
                # sweep.
                if device.type != "cuda":
                    want = len(preferred_secs) or max(1, args.max_results)
                    scan_budget = min(scan_budget, want + 3)
                result = locate_subjects(
                    args.video_input,
                    detector=detector,
                    preferred_secs=preferred_secs,
                    kept_ranges=parse_ranges(args.kept_ranges),
                    scan_step=args.scan_step,
                    scan_budget=scan_budget,
                    # One tile per requested anchor (so testing more frames shows
                    # more tiles), falling back to the CLI default with no anchors.
                    max_results=max(len(preferred_secs), 1) if preferred_secs else max(1, args.max_results),
                    auto_init_mode=args.auto_init_mode,
                    auto_select_strategy=args.auto_select_strategy,
                    auto_detector_threshold=args.auto_detector_threshold,
                    sam3_text_prompts=prompts,
                )
            else:
                frame, frame_info = read_frame_at_time(args.video_input, args.frame_sec)
                bbox, info = _detect_at(
                    frame,
                    detector=detector,
                    auto_init_mode=args.auto_init_mode,
                    auto_select_strategy=args.auto_select_strategy,
                    auto_detector_threshold=args.auto_detector_threshold,
                    sam3_text_prompts=prompts,
                )
                result = {
                    "ok": bbox is not None,
                    **frame_info,
                    "detection": bbox_payload(
                        bbox,
                        int(frame_info["video_width"]),
                        int(frame_info["video_height"]),
                    ),
                    "info": info,
                }
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
