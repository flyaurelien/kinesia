"""The person tracks file written by SAM 3.1 and read by every later stage.

``tracks.jsonl.gz`` holds one JSON header line, then one line per video frame::

    {"frame": 12, "ids": [0, 3], "scores": [0.97, 0.88],
     "boxes": [[x0, y0, x1, y1], ...], "rles": ["<coco rle>", ...]}

``ids`` are SAM 3.1's own masklet identifiers; ``boxes`` are the tight boxes of
the masks without specks, in source pixels.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from . import masks as mask_ops

KIND = "kinesia.tracks"
VERSION = 1


@dataclass
class FrameTracks:
    frame: int
    ids: list[int]
    scores: list[float]
    boxes: list[list[float]]
    rles: list[str]

    def mask(self, index: int, height: int, width: int) -> np.ndarray:
        return mask_ops.decode(self.rles[index], height, width)


@dataclass
class TracksFile:
    header: dict
    frames: list[FrameTracks] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.header["width"])

    @property
    def height(self) -> int:
        return int(self.header["height"])

    def track_ids(self) -> list[int]:
        return sorted({i for f in self.frames for i in f.ids})


class TracksWriter:
    """Stream frame records to disk as they come out of the tracker."""

    def __init__(self, path: Path, header: dict):
        self.path = path
        self._partial = path.with_name(path.name + ".part")
        self._handle = gzip.open(self._partial, "wt", compresslevel=5)
        self._handle.write(json.dumps({"kind": KIND, "version": VERSION, **header}) + "\n")
        self.count = 0

    def write(self, record: FrameTracks) -> None:
        self._handle.write(
            json.dumps(
                {
                    "frame": record.frame,
                    "ids": record.ids,
                    "scores": [round(float(s), 4) for s in record.scores],
                    "boxes": [[round(float(v), 1) for v in box] for box in record.boxes],
                    "rles": record.rles,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self.count += 1

    def close(self) -> None:
        self._handle.close()
        self._partial.replace(self.path)

    def abort(self) -> None:
        self._handle.close()
        self._partial.unlink(missing_ok=True)


def iter_tracks(path: Path) -> Iterator[dict | FrameTracks]:
    with gzip.open(path, "rt") as handle:
        header = json.loads(handle.readline())
        if header.get("kind") != KIND:
            raise ValueError(f"{path} is not a Kinesia tracks file")
        yield header
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield FrameTracks(row["frame"], row["ids"], row["scores"], row["boxes"], row["rles"])


def read_tracks(path: Path) -> TracksFile:
    rows = iter_tracks(path)
    header = next(rows)
    assert isinstance(header, dict)
    frames = [row for row in rows if isinstance(row, FrameTracks)]
    frames.sort(key=lambda f: f.frame)
    return TracksFile(header=header, frames=frames)
