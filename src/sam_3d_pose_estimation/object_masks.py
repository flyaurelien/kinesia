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

# How many rows the object's lowest point may be raised by the subject before
# the frame is refused. A couple of pixels is mask noise; more than that and the
# depth being read is the subject's, not the object's — a few pixels here is
# tens of centimetres out there.
CONTACT_TOLERANCE_PX = 3

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
    instances = segment_object_instances(image_bgr, prompt, detector, confidence)
    return instances[0] if instances else None


def segment_object_instances(
    image_bgr: np.ndarray,
    prompt: str,
    detector: Any,
    confidence: float = 0.5,
) -> list[tuple[np.ndarray, float]]:
    """Return every segmented instance of a prompt, highest confidence first.

    Static reconstruction needs the single strongest instance, while a temporal
    tracker must choose the instance that preserves an existing identity. Keeping
    this model interaction here avoids two subtly different mask conversions.

    Args:
        image_bgr: Source video frame in OpenCV BGR order.
        prompt: Open-vocabulary object description.
        detector: Loaded SAM 3 detector.
        confidence: SAM 3 confidence threshold.

    Returns:
        Boolean masks paired with detector confidence, sorted descending.
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
        return []
    order = scores.argsort(descending=True).detach().cpu().tolist()
    masks = state["masks"]
    return [
        (
            np.ascontiguousarray(masks[int(index), 0].detach().cpu().numpy()).astype(bool),
            float(scores[int(index)]),
        )
        for index in order
    ]


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


def box_silhouette(
    box: tuple[float, float, float, float], width: int, height: int
) -> np.ndarray:
    """Stand-in for the subject's outline, from the tracking box alone.

    A box covers about three times the area the body really occupies, so it
    reports the subject as being in the way more often than it is. That is the
    right way to be wrong when the reconstruction has not run yet and the real
    outlines do not exist: it passes over a usable frame now and then, never
    accepts one where the subject is actually blocking the object.
    """
    canvas = np.zeros((height, width), bool)
    x1, y1, x2, y2 = box
    left, right = sorted((int(round(x1)), int(round(x2))))
    top, bottom = sorted((int(round(y1)), int(round(y2))))
    canvas[
        max(top, 0) : min(bottom + 1, height),
        max(left, 0) : min(right + 1, width),
    ] = True
    return canvas


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

    # The depth comes from the object's lowest row, so that row is the only
    # part the subject genuinely must not stand on. An earlier version refused
    # any frame where the subject touched a band across the whole base, which
    # for a long object like a table means every frame — the band spans the
    # full width, and someone sitting at one end of it hides none of the
    # contact that matters.
    visible = obj & ~subject
    if not visible.any():
        return None
    visible_bottom = int(np.nonzero(visible.any(axis=1))[0].max())
    if bottom - visible_bottom > CONTACT_TOLERANCE_PX:
        return None
    return {
        "occlusion": float((subject & obj).sum() / total),
        "clearance": _gap_between(subject, obj),
        "object_px": total,
        "bottom_row": bottom,
    }


def _gap_between(subject: np.ndarray, obj: np.ndarray) -> float:
    """Pixels of clear space between the subject and the object, 0 if touching.

    Overlap alone cannot separate frames where the subject merely grazes the
    object from frames where they stand well away — and with boxes, which are
    much coarser than a body, whole runs of frames tie on it. How far away the
    subject is breaks those ties in the direction that is obviously better.
    """
    rows = np.nonzero(subject.any(axis=1))[0]
    cols = np.nonzero(subject.any(axis=0))[0]
    obj_rows = np.nonzero(obj.any(axis=1))[0]
    obj_cols = np.nonzero(obj.any(axis=0))[0]
    if rows.size == 0 or obj_rows.size == 0:
        return float("inf")
    horizontal = max(cols[0] - obj_cols[-1], obj_cols[0] - cols[-1], 0)
    vertical = max(rows[0] - obj_rows[-1], obj_rows[0] - rows[-1], 0)
    return float(np.hypot(horizontal, vertical))


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
    scored.sort(key=lambda s: (s["occlusion"], -s["clearance"], -s["object_px"]))

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
        frame = int(record.get("video_frame", index))
        silhouettes.append(
            (frame, subject_silhouette(vertices, float(focal), width, height))
        )
    log(f"subject silhouettes on {len(silhouettes)} of {len(records)} frames")
    return silhouettes


def load_subject_boxes(
    track_path: Path,
    log: Callable[[str], None] = print,
) -> tuple[list[tuple[int, np.ndarray]], int, int]:
    """Where the subject is on each frame, from the detection step's track.

    Available before any reconstruction has run, which is what lets an object's
    shape be built while the subject is still being reconstructed.
    """
    import json

    data = json.loads(track_path.read_text())
    width = int(data.get("videoWidth") or 0)
    height = int(data.get("videoHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"track file has no video size: {track_path}")

    covered: dict[int, np.ndarray] = {}
    for subject in data.get("subjects") or []:
        for key, box in (subject.get("frames") or {}).items():
            if not box or len(box) < 4:
                continue
            frame = int(key)
            mask = box_silhouette(tuple(box[:4]), width, height)
            # Several subjects on one frame all block the object.
            covered[frame] = covered[frame] | mask if frame in covered else mask
    log(f"subject boxes on {len(covered)} frames")
    return sorted(covered.items()), width, height
