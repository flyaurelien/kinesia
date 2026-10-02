"""SAM 3D Body on every tracked person of every frame.

All people of one frame go through the model together, each prompted with its
SAM 3.1 box and mask: the mask tells the model which of several overlapping
players the crop is about. Results are written in chunks of frames, so a job
preempted on a shared GPU resumes where it stopped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..tracks import FrameTracks
from ..video import read_frames

CHUNK_FRAMES = 200
# Per person-frame arrays, in the order they are stored.
FIELDS = {
    "frame": np.int32,
    "track": np.int32,
    "box": np.float32,  # (4,) crop box in source pixels
    "model_params": np.float32,  # (204,) MHR pose angles and skeleton scales, as fed to MHR
    "shape": np.float32,  # (45,) identity blend shapes
    "scale": np.float32,  # (28,) skeleton proportion coefficients
    "cam_t": np.float32,  # (3,) body root in camera coordinates, metres
    "kp3d": np.float16,  # (70, 3) MHR70 keypoints relative to cam_t
    "kp2d": np.float16,  # (70, 2) the same keypoints in source pixels
}


def load_model(body_dir: str):
    """Load SAM 3D Body on the GPU, without the unused hand-crop decoder."""
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(
        checkpoint_path=f"{body_dir}/model.ckpt",
        device="cuda",
        mhr_path=f"{body_dir}/assets/mhr_model.pt",
    )
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return estimator


def infer_frame(estimator, image_rgb: np.ndarray, boxes: np.ndarray, masks: np.ndarray, cam_int: np.ndarray) -> dict:
    """Run the body decoder on all people of one frame; return per-person arrays."""
    import torch
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    model = estimator.model
    batch = prepare_batch(
        image_rgb,
        estimator.transform,
        boxes.astype(np.float32),
        masks[..., None].astype(np.uint8),
        np.ones(len(boxes), dtype=np.float32),
        cam_int=torch.as_tensor(cam_int[None], dtype=torch.float32),
    )
    batch = recursive_to(batch, "cuda")
    model._initialize_batch(batch)
    # The body model sets its own precision (bf16 backbone, fp32 decoder and MHR,
    # whose sparse pose correctives have no bf16 kernel): no ambient autocast.
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        output = model.run_inference(
            image_rgb,
            batch,
            inference_type="body",
            transform_hand=estimator.transform_hand,
            thresh_wrist_angle=estimator.thresh_wrist_angle,
        )
        mhr = output["mhr"]
    to_np = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
    return {
        "model_params": to_np(mhr["mhr_model_params"]),
        "shape": to_np(mhr["shape"]),
        "scale": to_np(mhr["scale"]),
        "cam_t": to_np(mhr["pred_cam_t"]),
        "kp3d": to_np(mhr["pred_keypoints_3d"]),
        "kp2d": to_np(mhr["pred_keypoints_2d"]),
    }


def select_people(record: FrameTracks, min_height: float) -> list[int]:
    """Indices of the tracked people large enough to reconstruct."""
    return [i for i, box in enumerate(record.boxes) if box[3] - box[1] >= min_height]


def reconstruct_bodies(
    estimator,
    video: Path,
    tracks: list[FrameTracks],
    out_dir: Path,
    *,
    width: int,
    height: int,
    cam_int: np.ndarray,
    min_height: float,
    on_frame: Callable[[int, int], None] | None = None,
) -> dict:
    """Reconstruct every selected person; write ``out_dir/chunk_XXXX.npz`` files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {record.frame: record for record in tracks}
    total = len(tracks)
    chunk_count = (total + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    started = time.monotonic()
    people = 0
    for chunk in range(chunk_count):
        target = out_dir / f"chunk_{chunk:04d}.npz"
        first, last = chunk * CHUNK_FRAMES, min(total, (chunk + 1) * CHUNK_FRAMES)
        if target.is_file():
            if on_frame:
                on_frame(last, total)
            continue
        rows: dict[str, list] = {key: [] for key in FIELDS}
        for index, image_rgb in read_frames(video, start=first, stop=last):
            record = by_frame.get(index)
            chosen = select_people(record, min_height) if record else []
            if chosen:
                boxes = np.array([record.boxes[i] for i in chosen], dtype=np.float32)
                masks = np.stack([record.mask(i, height, width) for i in chosen])
                result = infer_frame(estimator, image_rgb, boxes, masks, cam_int)
                rows["frame"].extend([index] * len(chosen))
                rows["track"].extend(record.ids[i] for i in chosen)
                rows["box"].extend(boxes)
                for key in FIELDS:
                    if key in result:
                        rows[key].extend(result[key])
                people += len(chosen)
            if on_frame:
                on_frame(index + 1, total)
        arrays = {
            key: np.asarray(values, dtype=dtype) if values else np.zeros((0,), dtype=dtype)
            for (key, dtype), values in zip(FIELDS.items(), rows.values())
        }
        partial = target.with_name(target.name + ".part.npz")
        np.savez(partial, **arrays)
        partial.replace(target)
    return {"person_frames": people, "seconds": round(time.monotonic() - started, 1)}


def merge_chunks(out_dir: Path, target: Path) -> int:
    """Concatenate the chunk files into one ``bodies.npz``; return the row count."""
    chunks = sorted(out_dir.glob("chunk_*.npz"))
    merged: dict[str, list[np.ndarray]] = {key: [] for key in FIELDS}
    for path in chunks:
        with np.load(path) as data:
            for key in FIELDS:
                if data[key].size:
                    merged[key].append(data[key])
    arrays = {
        key: np.concatenate(parts) if parts else np.zeros((0,), dtype=FIELDS[key])
        for key, parts in merged.items()
    }
    partial = target.with_name(target.name + ".part.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(target)
    return int(arrays["frame"].shape[0])
