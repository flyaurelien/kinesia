"""What each tracked person looks like, to tell people apart.

Each person is cut out with their SAM 3.1 mask and everything else is set to
the mean colour before the model sees the crop, so the description is of the
person alone (clothes, hair, build), not of the court behind them or of a
neighbour. For every body in ``bodies.npz``:

* ``dino``: DINOv3 ViT-H+, pretrained as released, on a 256 x 128 crop; its
  patch features averaged under the mask over three horizontal bands (head
  and shoulders, torso, legs), then reduced to the video's ``DIMS`` principal
  components;
* ``colour``: hue, saturation and value histograms of the upper and lower body;
* ``visible``: the share of the person's box their mask fills, and ``hidden``:
  the share covered by other people's masks.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..tracks import FrameTracks
from ..video import read_frames

CROP = (256, 128)  # height, width: a standing person, 16 x 8 patches
MARGIN = 0.05  # of the box height, around the box
MEAN = np.array([0.485, 0.456, 0.406])  # DINOv3's input normalisation
STD = np.array([0.229, 0.224, 0.225])
BINS = (8, 4, 4)  # hue, saturation, value
DIMS = 256
BATCH = 64
HUB = "facebookresearch_dinov3_main"
DTYPES = {"F32": np.float32, "F16": np.float16}


def read_safetensors(path: Path) -> dict:
    """The tensors of a ``.safetensors`` file (float32 or float16)."""
    import torch

    with path.open("rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(size))
        header.pop("__metadata__", None)
        tensors = {}
        for name, entry in header.items():
            start, end = entry["data_offsets"]
            file.seek(8 + size + start)
            data = np.frombuffer(file.read(end - start), dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])
            tensors[name] = torch.from_numpy(data.copy())
    return tensors


def _native_name(name: str) -> str:
    """timm's names for DINOv3's parameters, back to the original code's."""
    for old, new in (
        ("reg_token", "storage_tokens"),
        ("gamma_1", "ls1.gamma"),
        ("gamma_2", "ls2.gamma"),
        ("mlp.fc1_g", "mlp.w1"),
        ("mlp.fc1_x", "mlp.w2"),
        ("mlp.fc2", "mlp.w3"),
    ):
        name = name.replace(old, new)
    return name


def load_model(weights: Path, device: str, hub: Path | None = None):
    """DINOv3 ViT-H+ (LVD-1689M) in DINOv3's own code (``hub``), with timm's copy of the released weights."""
    hub = hub or Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache" / "torch") / "hub" / HUB
    if str(hub) not in sys.path:
        sys.path.insert(0, str(hub))
    from dinov3.hub.backbones import dinov3_vith16plus

    model = dinov3_vith16plus(pretrained=False)
    missing, unexpected = model.load_state_dict({_native_name(k): v for k, v in read_safetensors(weights).items()}, strict=False)
    # The released ViT-H+ has no query/key/value bias; the mask token and the
    # rotary periods are not weights.
    missing = [k for k in missing if not k.endswith(("attn.qkv.bias", "attn.qkv.bias_mask")) and k not in ("mask_token", "rope_embed.periods")]
    if missing or unexpected:
        raise ValueError(f"unexpected DINOv3 weights in {weights}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    for block in model.blocks:
        block.attn.qkv.bias.data.zero_()
    return model.eval().to(device)


def crop_person(image: np.ndarray, mask: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """The person alone on a ``CROP`` canvas: their box with a margin, scaled to fit, the rest the mean colour."""
    x0, y0, x1, y1 = (float(v) for v in box)
    pad = MARGIN * (y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    left, top = max(0, int(x0)), max(0, int(y0))
    right, bottom = min(image.shape[1], int(np.ceil(x1)) + 1), min(image.shape[0], int(np.ceil(y1)) + 1)
    height, width = CROP
    scale = min(height / (y1 - y0), width / (x1 - x0))
    warp = np.array(
        [[scale, 0.0, width / 2 - scale * ((x0 + x1) / 2 - left)], [0.0, scale, height / 2 - scale * ((y0 + y1) / 2 - top)]],
        dtype=np.float32,
    )
    fill = tuple(float(v) for v in MEAN * 255)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    picture = cv2.warpAffine(np.ascontiguousarray(image[top:bottom, left:right]), warp, (width, height), flags=interpolation, borderValue=fill)
    inside = cv2.warpAffine(mask[top:bottom, left:right].astype(np.float32), warp, (width, height), flags=cv2.INTER_LINEAR) > 0.5
    picture[~inside] = np.round(MEAN * 255).astype(np.uint8)
    return picture, inside


def colour_histogram(picture: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Hue, saturation and value histograms of the upper and lower body, as square roots of shares."""
    size = int(np.prod(BINS))
    out = np.zeros(2 * size, dtype=np.float32)
    rows = np.flatnonzero(inside.any(axis=1))
    if len(rows) < 4:
        return out
    hsv = cv2.cvtColor(picture, cv2.COLOR_RGB2HSV).astype(np.int64)
    middle = (rows[0] + rows[-1]) // 2
    for k, part in enumerate((slice(0, middle), slice(middle, None))):
        h, s, v = hsv[part][inside[part]].T
        index = (h * BINS[0] // 180) * BINS[1] * BINS[2] + (s * BINS[1] // 256) * BINS[2] + v * BINS[2] // 256
        counts = np.bincount(index, minlength=size)
        out[k * size : (k + 1) * size] = np.sqrt(counts / max(1, counts.sum()))
    return out


def band_features(model, pictures: np.ndarray, insides: np.ndarray, device: str) -> np.ndarray:
    """``(B, 3 * C)``: patch features averaged under the mask in three bands, each a unit vector."""
    import torch
    import torch.nn.functional as F

    mean = torch.tensor(MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    x = (torch.from_numpy(pictures).to(device).permute(0, 3, 1, 2).float() / 255 - mean) / std
    kind = torch.device(device).type
    with torch.inference_mode(), torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
        tokens = model.forward_features(x)["x_norm_patchtokens"]
    h, w = x.shape[2] // model.patch_size, x.shape[3] // model.patch_size
    tokens = tokens.float().view(len(x), h, w, -1)
    weights = F.adaptive_avg_pool2d(torch.from_numpy(insides).to(device).float()[:, None], (h, w))[:, 0]
    bands = []
    for top, bottom in ((0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)):
        share = weights[:, top:bottom]
        pooled = (tokens[:, top:bottom] * share[..., None]).sum(dim=(1, 2)) / share.sum(dim=(1, 2)).clamp_min(1e-3)[:, None]
        bands.append(F.normalize(pooled, dim=1))
    return torch.cat(bands, dim=1).cpu().numpy()


def principal_components(features: np.ndarray, dims: int = DIMS, sample: int = 20000) -> np.ndarray:
    """The features in their ``dims`` main directions (fitted on up to ``sample`` rows)."""
    import torch

    rows = np.random.default_rng(0).choice(len(features), min(sample, len(features)), replace=False)
    x = torch.from_numpy(np.ascontiguousarray(features[rows], dtype=np.float32))
    mean = x.mean(dim=0)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        _, _, basis = torch.pca_lowrank(x - mean, q=min(dims, *x.shape), center=False, niter=4)
    return ((torch.from_numpy(np.asarray(features, dtype=np.float32)) - mean) @ basis).numpy()


def describe_people(
    model,
    video: Path,
    tracks: list[FrameTracks],
    frames: np.ndarray,
    track_ids: np.ndarray,
    *,
    width: int,
    height: int,
    device: str,
    on_frame: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """The appearance of every ``(frame, track)`` body, in the order given."""
    by_frame = {record.frame: record for record in tracks}
    wanted: dict[int, list[int]] = {}
    for row, frame in enumerate(frames):
        wanted.setdefault(int(frame), []).append(row)
    count = len(frames)
    features = np.zeros((count, 0), dtype=np.float32)
    colour = np.zeros((count, 2 * int(np.prod(BINS))), dtype=np.float32)
    visible = np.zeros(count, dtype=np.float32)
    hidden = np.zeros(count, dtype=np.float32)
    batch: list[tuple[int, np.ndarray, np.ndarray]] = []
    described = np.zeros(count, dtype=bool)

    def flush() -> None:
        nonlocal features
        if not batch:
            return
        values = band_features(model, np.stack([b[1] for b in batch]), np.stack([b[2] for b in batch]), device)
        if features.shape[1] == 0:
            features = np.zeros((count, values.shape[1]), dtype=np.float32)
        features[[b[0] for b in batch]] = values
        batch.clear()

    last = int(frames.max()) + 1 if count else 0
    for index, image in read_frames(video, stop=last):
        if index in wanted:
            record = by_frame[index]
            masks = {t: record.mask(i, height, width) for i, t in enumerate(record.ids)}
            people = np.zeros((height, width), dtype=np.uint8)
            for mask in masks.values():
                people += mask
            for row in wanted[index]:
                track = int(track_ids[row])
                box = record.boxes[record.ids.index(track)]
                mask = masks[track]
                x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
                x1, y1 = min(width, int(np.ceil(box[2])) + 1), min(height, int(np.ceil(box[3])) + 1)
                area = max(1, (x1 - x0) * (y1 - y0))
                own = mask[y0:y1, x0:x1]
                visible[row] = own.sum() / area
                hidden[row] = ((people[y0:y1, x0:x1] - own) > 0).sum() / area
                picture, inside = crop_person(image, mask, box)
                colour[row] = colour_histogram(picture, inside)
                batch.append((row, picture, inside))
                described[row] = True
                if len(batch) >= BATCH:
                    flush()
        if on_frame:
            on_frame(index + 1, last)
    flush()
    if not described.all():
        raise ValueError(f"{int((~described).sum())} people are in frames the video does not have")
    return {
        "frame": np.asarray(frames, dtype=np.int32),
        "track": np.asarray(track_ids, dtype=np.int32),
        "dino": principal_components(features).astype(np.float16) if count else np.zeros((0, DIMS), np.float16),
        "colour": colour.astype(np.float16),
        "visible": visible.astype(np.float16),
        "hidden": hidden.astype(np.float16),
    }


def write_appearance(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_name(path.name + ".part.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def describe_run(model, run_dir: Path, *, device: str, on_frame: Callable[[int, int], None] | None = None) -> dict:
    """Describe every body of one analysis; write ``raw/appearance.npz``."""
    from ..tracks import read_tracks

    started = time.monotonic()
    raw = run_dir / "raw"
    info = json.loads((run_dir / "request.json").read_text())["video"]
    tracks = read_tracks(raw / "tracks.jsonl.gz")
    with np.load(raw / "bodies.npz") as bodies:
        frames, track_ids = bodies["frame"], bodies["track"]
    arrays = describe_people(
        model, run_dir / "video.mp4", tracks.frames, frames, track_ids,
        width=int(info["width"]), height=int(info["height"]), device=device, on_frame=on_frame,
    )  # fmt: skip
    write_appearance(raw / "appearance.npz", arrays)
    return {"people": int(len(frames)), "seconds": round(time.monotonic() - started, 1)}
