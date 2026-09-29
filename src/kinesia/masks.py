"""Binary person masks: compact storage, clean boxes and outlines.

Masks are stored as COCO run-length encodings (``pycocotools``), which keeps a
full-resolution mask per person per frame to about a kilobyte.
"""

from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as coco_mask

SPECK_FRACTION = 0.01  # components smaller than this share of the mask are specks


Extent = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels, ends exclusive


def extent(mask: np.ndarray) -> Extent | None:
    """Pixel bounds of a mask, or None when it is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def encode(mask: np.ndarray, bounds: Extent | None = None) -> str:
    """Encode one ``(H, W)`` boolean mask as a compressed COCO RLE string.

    COCO runs follow the image column by column, and every column outside the
    mask's extent is empty, so only the columns it spans are scanned.
    """
    height, width = mask.shape
    bounds = bounds or extent(mask)
    if bounds is None:
        runs = [height * width]
    else:
        x0, _, x1, _ = bounds
        flat = np.ascontiguousarray(mask[:, x0:x1].T).reshape(-1)
        edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
        if flat[0]:
            runs.insert(0, 0)  # runs start with background
        runs[0] += x0 * height
        tail = (width - x1) * height
        if tail:
            if len(runs) % 2:  # the last run is background
                runs[-1] += tail
            else:
                runs.append(tail)
    rle = coco_mask.frPyObjects({"counts": runs, "size": [height, width]}, height, width)
    return rle["counts"].decode("ascii")


def decode(counts: str, height: int, width: int) -> np.ndarray:
    """Decode a compressed COCO RLE string back to an ``(H, W)`` boolean mask."""
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return coco_mask.decode(rle).astype(bool)


def area(counts: str, height: int, width: int) -> int:
    return int(coco_mask.area({"size": [height, width], "counts": counts.encode("ascii")}))


def clean_box(mask: np.ndarray, bounds: Extent | None = None) -> tuple[float, float, float, float] | None:
    """Tight ``x0, y0, x1, y1`` box of the mask without its stray specks.

    The tracker's own boxes enclose every mask pixel, and a few isolated pixels
    far from the person inflate them; the pose model then sees a badly framed
    crop. Components under ``SPECK_FRACTION`` of the mask area are ignored.
    """
    bounds = bounds or extent(mask)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    binary = mask[top:bottom, left:right].astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(1, SPECK_FRACTION * areas.sum())) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(areas)) + 1])
    x0 = stats[keep, cv2.CC_STAT_LEFT].min()
    y0 = stats[keep, cv2.CC_STAT_TOP].min()
    x1 = (stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max()
    y1 = (stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max()
    return float(left + x0), float(top + y0), float(left + x1), float(top + y1)


def outline(mask: np.ndarray, tolerance: float = 1.5, max_points: int = 96) -> list[list[int]]:
    """Simplified outer contours of a mask as flat ``[x0, y0, x1, y1, ...]`` lists."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = sum(cv2.contourArea(c) for c in contours) or 1.0
    polygons: list[list[int]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < SPECK_FRACTION * total or len(contour) < 3:
            continue
        epsilon = tolerance
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        while len(simplified) > max_points:
            epsilon *= 1.6
            simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) >= 3:
            polygons.append(simplified.reshape(-1).astype(int).tolist())
    return polygons


def touches_border(box: tuple[float, float, float, float], width: int, height: int, margin: int = 2) -> dict:
    """Which image edges a box touches (a person cut by the frame)."""
    x0, y0, x1, y1 = box
    return {
        "left": x0 <= margin,
        "top": y0 <= margin,
        "right": x1 >= width - margin,
        "bottom": y1 >= height - margin,
    }
