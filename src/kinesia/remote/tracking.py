"""SAM 3.1 Object Multiplex: detect and track every person through the video.

This is Meta's released video predictor used as published: one session over
the whole clip, a text prompt, and forward propagation. The model keeps a
memory per masklet, re-conditions tracks on fresh detections and confirms new
masklets before reporting them, so identities survive occlusions and crossings
without any association of our own. Only the object cap is raised: the public
builder stops at 16 objects, far fewer than the people on a football pitch.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable

import numpy as np

from .. import masks as mask_ops
from ..tracks import FrameTracks, TracksWriter

MIN_MASK_PIXELS = 64


def _accept_false_offload_keyword(predictor) -> None:
    """Let the session dispatcher pass ``offload_state_to_cpu=False``.

    At the pinned Meta revision the dispatcher always passes that keyword but
    the multiplex model's ``init_state`` does not accept it. Only the false
    default is dropped; asking for state offload is refused loudly.
    """
    original = predictor.model.init_state
    if "offload_state_to_cpu" in inspect.signature(original).parameters:
        return

    @wraps(original)
    def init_state(*args, offload_state_to_cpu: bool = False, **kwargs):
        if offload_state_to_cpu:
            raise ValueError("SAM 3.1 multiplex cannot offload its tracking state to the CPU")
        return original(*args, **kwargs)

    predictor.model.init_state = init_state


_removed_object: contextvars.ContextVar[int | None] = contextvars.ContextVar("kinesia_removed_object", default=None)


def _promote_shared_anchor(state: dict, removed: int) -> bool:
    """Make the earliest frame that holds another object the bucket's conditioning frame."""
    outputs = state["output_dict"]
    for candidate in sorted(outputs["non_cond_frame_outputs"]):
        members = outputs["non_cond_frame_outputs"][candidate].get("local_obj_id_to_idx", {})
        if any(other != removed for other in members):
            outputs["cond_frame_outputs"][candidate] = outputs["non_cond_frame_outputs"].pop(candidate)
            for per_object in state["output_dict_per_obj"].values():
                moved = per_object["non_cond_frame_outputs"].pop(candidate, None)
                if moved is not None:
                    per_object["cond_frame_outputs"][candidate] = moved
            return True
    return False


def keep_shared_conditioning() -> bool:
    """Stop the multiplex tracker from wiping objects that share a conditioning frame.

    Several masklets in one multiplex bucket can rely on a single conditioning
    frame created by another ("owner") masklet. When the owner is removed, the
    released tracker resets the whole bucket, and the next frame fails with
    "No points are provided" (facebookresearch/sam3 issue 572; it shows up on
    clips longer than about a minute). This applies the fix proposed upstream
    in pull request 573: promote the earliest remaining frame that holds
    another object to be the new conditioning frame, and reset only when no
    other object has memory left. Only resets triggered while removing an
    object's inputs are affected. Returns False if already applied.
    """
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo as Tracker

    if getattr(Tracker, "_kinesia_keeps_shared_conditioning", False):
        return False
    original_clear = Tracker.clear_all_points_in_frame
    original_reset = Tracker._reset_tracking_results

    @wraps(original_clear)
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id, *args, **kwargs):
        token = _removed_object.set(obj_id)
        try:
            return original_clear(self, inference_state, frame_idx, obj_id, *args, **kwargs)
        finally:
            _removed_object.reset(token)

    @wraps(original_reset)
    def _reset_tracking_results(self, inference_state):
        removed = _removed_object.get()
        if removed is not None and _promote_shared_anchor(inference_state, removed):
            return None
        return original_reset(self, inference_state)

    Tracker.clear_all_points_in_frame = clear_all_points_in_frame
    Tracker._reset_tracking_results = _reset_tracking_results
    Tracker._kinesia_keeps_shared_conditioning = True
    return True


def build_predictor(checkpoint: str, source: str, *, max_objects: int, prob_threshold: float):
    """Load the official multiplex video predictor on the GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 video tracking needs a CUDA GPU")
    if source not in sys.path:
        sys.path.insert(0, source)
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    # The builder prints its checkpoint report on stdout, which carries our events.
    with contextlib.redirect_stdout(sys.stderr):
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint,
            max_num_objects=max_objects,
            use_fa3=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            default_output_prob_thresh=prob_threshold,
        )
    # The predictor enters a process-wide bf16 autocast; keep it to our own calls.
    predictor.bf16_context.__exit__(None, None, None)
    _accept_false_offload_keyword(predictor)
    keep_shared_conditioning()
    return predictor


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def frame_record(index: int, outputs: dict, width: int, height: int) -> FrameTracks:
    """Convert one propagated frame into a compact record."""
    ids = _numpy(outputs["out_obj_ids"]).reshape(-1).astype(int)
    scores = _numpy(outputs["out_probs"]).reshape(-1).astype(float)
    masks = _numpy(outputs["out_binary_masks"]).astype(bool, copy=False).reshape(len(ids), height, width)
    record = FrameTracks(index, [], [], [], [])
    for track_id, score, mask in zip(ids, scores, masks):
        # Everything below works on the mask's extent: a person covers a few
        # percent of the frame, and this runs between GPU steps.
        bounds = mask_ops.extent(mask)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        if int(np.count_nonzero(mask[y0:y1, x0:x1])) < MIN_MASK_PIXELS:
            continue
        box = mask_ops.clean_box(mask, bounds)
        if box is None:
            continue
        record.ids.append(int(track_id))
        record.scores.append(float(score))
        record.boxes.append(list(box))
        record.rles.append(mask_ops.encode(mask, bounds))
    return record


def track_video(
    predictor,
    video: Path,
    output: Path,
    *,
    prompt: str,
    frame_count: int,
    width: int,
    height: int,
    header: dict,
    on_frame: Callable[[int, int], None] | None = None,
) -> dict:
    """Run one session over ``video`` and write every frame to ``output``."""
    import torch

    writer = TracksWriter(output, {**header, "width": width, "height": height, "prompt": prompt})
    seen: set[int] = set()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    session = None
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            session = predictor.handle_request(
                dict(type="start_session", resource_path=str(video), offload_video_to_cpu=True)
            )["session_id"]
            predictor.handle_request(
                dict(type="add_prompt", session_id=session, frame_index=0, text=prompt)
            )
            request = dict(
                type="propagate_in_video",
                session_id=session,
                propagation_direction="forward",
                start_frame_index=0,
            )
            for response in predictor.handle_stream_request(request):
                index = int(response["frame_index"])
                if index in seen:
                    raise RuntimeError(f"SAM 3.1 returned frame {index} twice")
                writer.write(frame_record(index, response["outputs"], width, height))
                seen.add(index)
                if on_frame:
                    on_frame(len(seen), frame_count)
        # Checked before the file is published: a resumed job reuses any
        # tracks file it finds.
        missing = sorted(set(range(frame_count)) - seen)
        if missing:
            raise RuntimeError(f"SAM 3.1 skipped {len(missing)} frames (first {missing[:5]})")
    except BaseException:
        writer.abort()
        raise
    finally:
        if session is not None:
            predictor.handle_request(dict(type="close_session", session_id=session))
    writer.close()
    torch.cuda.synchronize()
    return {
        "frames": len(seen),
        "seconds": round(time.monotonic() - started, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
