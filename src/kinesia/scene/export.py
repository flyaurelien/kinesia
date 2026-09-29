"""Files the viewer reads: ``scene.json`` plus packed binary sections.

Binary files are plain concatenations of little-endian typed arrays, each
section aligned to 8 bytes; ``scene.json`` lists every section's byte offset,
type and shape so the browser can wrap them without copying.

* ``mesh.bin``   shared topology and skinning (faces, four joint slots and
  weights per vertex, inverse bind pose, joint parents)
* ``people.bin`` per person: rest mesh, joint scales, and for every animated
  frame the pelvis world state, local joint rotations (int16), local joint
  offsets (int16, 0.01 cm), world keypoints (int16, 2 mm) and flags
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUAT_SCALE = 32767.0
LOCAL_T_UNIT = 0.01  # centimetres per int16 step
KEYPOINT_UNIT = 0.002  # metres per int16 step
FLAG_MEASURED, FLAG_LEFT_CONTACT, FLAG_RIGHT_CONTACT = 1, 2, 4


class SectionWriter:
    def __init__(self, path: Path):
        self.path = path
        self._handle = path.with_name(path.name + ".part").open("wb")
        self.offset = 0

    def add(self, array: np.ndarray, dtype: str) -> dict:
        data = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
        pad = (-self.offset) % 8
        if pad:
            self._handle.write(b"\0" * pad)
            self.offset += pad
        self._handle.write(data.tobytes())
        entry = {"offset": self.offset, "type": np.dtype(dtype).name, "shape": list(data.shape)}
        self.offset += data.nbytes
        return entry

    def close(self) -> None:
        name = self._handle.name
        self._handle.close()
        Path(name).replace(self.path)


def write_mesh(path: Path, model) -> dict:
    joints, weights = model.skinning
    writer = SectionWriter(path)
    layout = {
        "faces": writer.add(model.faces, "uint16"),
        "skin_index": writer.add(joints, "uint8"),
        "skin_weight": writer.add(weights, "float32"),
        "inverse_bind": writer.add(model.inverse_bind, "float32"),
        "parents": writer.add(model.parents, "int16"),
    }
    writer.close()
    return layout


def write_people(path: Path, motions: list) -> list[dict]:
    writer = SectionWriter(path)
    layouts = []
    for motion in motions:
        flags = motion.measured.astype(np.uint8) * FLAG_MEASURED
        flags |= motion.contacts[:, 0].astype(np.uint8) * FLAG_LEFT_CONTACT
        flags |= motion.contacts[:, 1].astype(np.uint8) * FLAG_RIGHT_CONTACT
        layouts.append(
            {
                "rest_vertices": writer.add(motion.rest_vertices, "float32"),
                "joint_scales": writer.add(motion.joint_scales, "float32"),
                "frames": writer.add(motion.frames, "int32"),
                "pelvis": writer.add(motion.pelvis, "float32"),
                "local_q": writer.add(np.round(motion.local_q * QUAT_SCALE), "int16"),
                "local_t": writer.add(np.round(motion.local_t / LOCAL_T_UNIT).clip(-32767, 32767), "int16"),
                "keypoints": writer.add(np.round(motion.keypoints / KEYPOINT_UNIT).clip(-32767, 32767), "int16"),
                "flags": writer.add(flags, "uint8"),
            }
        )
    writer.close()
    return layouts


def write_json(path: Path, data: dict) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, separators=(",", ":")))
    partial.replace(path)
