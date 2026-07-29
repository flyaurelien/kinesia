"""Segment an object named in plain text, and pick the frame worth segmenting.

The detection model the pipeline already loads is open-vocabulary: it locates
"chair" as readily as "person". Only the layers above it are person-only, and
they throw the masks away — the pipeline needs boxes, not outlines. An object
needs the outline, so this talks to the processor directly.

Which frame to segment matters more than it looks. The object's depth is read
from the BOTTOM row of its mask, so a subject standing in front of the object's
feet does not merely dent the outline: it moves the object. Frames are therefore
ranked by how much of the object the subject covers, and any frame where the
subject touches the object's ground contact is refused outright.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

# The bottom band of the object that fixes its depth. Fifteen rows is about the
# thickness of a chair leg's contact patch at the distances these clips are shot
# at — narrow enough to mean "the feet", wide enough to survive a ragged mask.
CONTACT_BAND_PX = 15

# Two candidate frames closer together than this show the same instant, so
# keeping both would waste the expensive re-segmentation passes on a duplicate.
MIN_CANDIDATE_GAP_FRAMES = 30


def segment_object(
    image_bgr: np.ndarray,
    prompt: str,
    detector: Any,
    confidence: float = 0.5,
) -> tuple[np.ndarray, float] | None:
    """Boolean (H, W) mask of the best instance of `prompt`, and its score.

    Returns None when nothing matches. `detector` is a built detector: loading
    one costs seconds and gigabytes, so callers hold it across every prompt and
    every pass rather than rebuilding it here.
    """
    import cv2
    from PIL import Image

    processor = detector.processor
    processor.confidence_threshold = float(confidence)
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=prompt.strip(), state=state)

    scores = state["scores"]
    if scores.numel() == 0:
        return None
    # The instances come back in detection order, not sorted by score.
    best = int(scores.argmax())
    mask = state["masks"][best, 0].detach().cpu().numpy()
    return np.ascontiguousarray(mask).astype(bool), float(scores[best])


def subject_silhouette(
    vertices: np.ndarray, focal: float, width: int, height: int
) -> np.ndarray:
    """Binary silhouette of a reconstructed body, from its camera-space mesh.

    The tracking box is not the subject: measured on a real run it covers three
    times the area the body actually occupies, which would report an occlusion
    wherever the subject merely passes nearby.
    """
    import cv2

    in_front = vertices[:, 2] > 1e-6
    canvas = np.zeros((height, width), np.uint8)
    if not in_front.any():
        return canvas.astype(bool)
    visible = vertices[in_front]
    u = focal * visible[:, 0] / visible[:, 2] + width / 2.0
    v = focal * visible[:, 1] / visible[:, 2] + height / 2.0
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not inside.any():
        return canvas.astype(bool)
    canvas[v[inside].astype(np.int32), u[inside].astype(np.int32)] = 1
    closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return closed.astype(bool)


def score_frame(subject: np.ndarray, obj: np.ndarray) -> dict | None:
    """How badly the subject spoils this frame for reconstructing the object.

    None means the frame is unusable: the subject stands on the object's ground
    contact, and the depth read from it would be the subject's, not the
    object's.
    """
    total = int(obj.sum())
    if total == 0:
        return None
    rows = np.nonzero(obj.any(axis=1))[0]
    bottom = int(rows.max())
    band = obj.copy()
    band[: max(bottom - CONTACT_BAND_PX, 0)] = False
    band_total = int(band.sum())
    if band_total and (subject & band).sum() > 0:
        return None
    return {
        "occlusion": float((subject & obj).sum() / total),
        "object_px": total,
        "bottom_row": bottom,
    }


def usable_for_depth(mask: np.ndarray, height: int) -> bool:
    """Whether this outline's bottom row can be trusted to carry the depth.

    Two ways it cannot: the object runs off the bottom of the picture, so its
    real contact with the floor was never seen; or it sits on the horizon, where
    the ray through it is parallel to the floor and the depth diverges.
    """
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return False
    bottom = int(rows.max())
    if bottom >= height - 1:
        return False
    return abs(bottom - height / 2.0) > 2.0


def rank_frames(
    object_mask: np.ndarray,
    silhouettes: Iterable[tuple[int, np.ndarray]],
    limit: int = 3,
) -> list[dict]:
    """Best frames to segment the object on, worst occlusion last.

    `object_mask` only has to be roughly right: it says WHERE the object is, so
    that the free per-frame silhouettes can say when the subject is out of the
    way. The chosen frames get segmented again for the mask that is actually
    used.
    """
    scored: list[dict] = []
    for index, silhouette in silhouettes:
        score = score_frame(silhouette, object_mask)
        if score is not None:
            scored.append({"frame": index, **score})
    scored.sort(key=lambda s: (s["occlusion"], -s["object_px"]))

    spread: list[dict] = []
    for candidate in scored:
        if all(abs(candidate["frame"] - kept["frame"]) >= MIN_CANDIDATE_GAP_FRAMES
               for kept in spread):
            spread.append(candidate)
        if len(spread) >= limit:
            break
    return spread


def write_mask(mask: np.ndarray, path: Path) -> None:
    """Save a mask as the single-channel PNG the reconstruction expects.

    Single channel on purpose: the object model reads the LAST channel of a
    multi-channel mask, so an RGB copy of the same picture would silently be
    reduced to its blue channel.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def load_subject_silhouettes(
    records: list[dict],
    width: int,
    height: int,
    log: Callable[[str], None] = print,
) -> list[tuple[int, np.ndarray]]:
    """Silhouette of the subject on every frame that reconstructed one.

    Costs no model time: the meshes were already exported by the run, and
    projecting them is arithmetic.
    """
    import trimesh

    silhouettes: list[tuple[int, np.ndarray]] = []
    for index, record in enumerate(records):
        mesh_path = record.get("mesh_path")
        focal = record.get("focal_length")
        if not mesh_path or not focal or not record.get("subject_present"):
            continue
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except (ValueError, OSError):
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            continue
        silhouettes.append(
            (index, subject_silhouette(vertices, float(focal), width, height))
        )
    log(f"subject silhouettes on {len(silhouettes)} of {len(records)} frames")
    return silhouettes
