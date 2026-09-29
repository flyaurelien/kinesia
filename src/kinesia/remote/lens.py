"""The camera's focal length, from the picture alone (MoGe-2, on the GPU).

Kinesia films with a fixed camera, so the background never moves and gives
no cue about the lens. MoGe-2 estimates the camera intrinsics from a single
image; it is the field-of-view estimator SAM 3D Body's authors use. A few
frames spread over the clip are measured and the median focal length kept.
Without the model, SAM 3D Body's default (focal length = image diagonal) is
used and the result says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..video import read_frames

SAMPLES = 6


def default_focal(width: int, height: int) -> float:
    return float(np.hypot(width, height))


def estimate_focal(video: Path, frame_count: int, width: int, height: int, weights: Path | None) -> tuple[float, str, list[float]]:
    """``(focal in pixels, source, per-frame estimates)``."""
    if weights is None or not weights.is_file():
        return default_focal(width, height), "default", []
    import torch
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(str(weights)).cuda().eval()
    wanted = set(np.linspace(0, max(0, frame_count - 1), SAMPLES).astype(int).tolist())
    estimates = []
    with torch.inference_mode():
        for index, rgb in read_frames(video):
            if index not in wanted:
                continue
            image = torch.as_tensor(rgb, dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255.0
            intrinsics = model.infer(image)["intrinsics"].float().cpu().numpy()
            # Normalized intrinsics; SAM 3D Body keeps the vertical focal for both axes.
            estimates.append(float(intrinsics[1, 1] * height))
            if len(estimates) == len(wanted):
                break
    del model
    torch.cuda.empty_cache()
    if not estimates:
        return default_focal(width, height), "default", []
    return float(np.median(estimates)), "moge-2", [round(e, 1) for e in estimates]
