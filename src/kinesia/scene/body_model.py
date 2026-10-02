"""The Momentum Human Rig (MHR) body model that SAM 3D Body predicts.

MHR maps 204 model parameters (pose angles plus skeleton proportions) and 45
identity coefficients to a skinned mesh. Its TorchScript file exposes every
step; Kinesia uses them to

* hold each person's identity fixed (a body does not change shape mid-clip),
* smooth joint rotations in the skeleton's own local frames, and
* export exactly what the browser needs to skin the mesh itself: the rest
  mesh, the skinning weights (at most four joints per vertex, the same limit
  as three.js) and the inverse bind pose.

Units are MHR's: centimetres, Y up. Pose correctives (a learned per-pose mesh
offset) are not reproduced by the browser's linear blend skinning; the
difference is a few millimetres on most of the body.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from . import quat

NUM_JOINTS = 127
NUM_VERTICES = 18439
POSE_PARAMS = 136  # 3 translation x10, 3 root rotation, 130 body/hand angles
SCALE_PARAMS = 68


class BodyModel:
    def __init__(self, model_dir: Path):
        """``model_dir`` is the SAM 3D Body folder (``model.ckpt``, ``assets/mhr_model.pt``)."""
        import torch

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.module = torch.jit.load(str(self.model_dir / "assets" / "mhr_model.pt"), map_location="cpu").eval()
        character = self.module.character_torch
        self._character = character
        self.parents = character.skeleton.joint_parents.numpy().astype(np.int64)
        self.joint_names = list(character.skeleton.joint_names)
        if any(p >= i for i, p in enumerate(self.parents) if p >= 0):
            raise ValueError("MHR joints are expected in parent-before-child order")
        self.faces = character.mesh.faces.numpy().astype(np.int64)
        self.inverse_bind = character.linear_blend_skinning.inverse_bind_pose.numpy().astype(np.float64)

    # -- keypoints -----------------------------------------------------------------

    @cached_property
    def keypoint_mapping(self) -> np.ndarray:
        """(70, V + 127) linear map from mesh vertices and joints to the MHR70 keypoints."""
        torch = self._torch
        checkpoint = torch.load(self.model_dir / "model.ckpt", map_location="cpu", weights_only=False, mmap=True)
        state = checkpoint.get("state_dict", checkpoint)
        return state["head_pose.keypoint_mapping"][:70].numpy().astype(np.float64)

    def keypoints(self, states: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """World keypoints (N, 70, 3) from world joint states (N, 127, 8) and the rest mesh."""
        mapping = self.keypoint_mapping
        vertex_cols = np.flatnonzero(np.abs(mapping[:, :NUM_VERTICES]).sum(axis=0) > 0)
        joint_cols = np.flatnonzero(np.abs(mapping[:, NUM_VERTICES:]).sum(axis=0) > 0)
        joints, weights = self.skinning
        transforms = quat.compose(states, self.inverse_bind)  # (N, 127, 8)
        verts = np.zeros((len(states), len(vertex_cols), 3))
        for slot in range(4):
            j = joints[vertex_cols, slot]
            verts += weights[vertex_cols, slot][None, :, None] * quat.apply(transforms[:, j], rest[vertex_cols][None])
        return (
            np.einsum("kv,nvc->nkc", mapping[:, vertex_cols], verts)
            + np.einsum("kj,njc->nkc", mapping[:, NUM_VERTICES + joint_cols], states[:, joint_cols, :3])
        )

    # -- mesh ----------------------------------------------------------------------

    @cached_property
    def skinning(self) -> tuple[np.ndarray, np.ndarray]:
        """``(joints, weights)`` per vertex, four slots each (unused slots weigh 0)."""
        lbs = self._character.linear_blend_skinning
        vertices = lbs.vert_indices_flattened.numpy()
        joints = lbs.skin_indices_flattened.numpy()
        weights = lbs.skin_weights_flattened.numpy()
        slot_joints = np.zeros((NUM_VERTICES, 4), dtype=np.int64)
        slot_weights = np.zeros((NUM_VERTICES, 4), dtype=np.float64)
        fill = np.zeros(NUM_VERTICES, dtype=np.int64)
        for vertex, joint, weight in zip(vertices, joints, weights):
            slot = fill[vertex]
            if slot >= 4:
                raise ValueError("MHR vertex with more than four skinning influences")
            slot_joints[vertex, slot], slot_weights[vertex, slot] = joint, weight
            fill[vertex] += 1
        slot_weights /= slot_weights.sum(axis=1, keepdims=True).clip(1e-12)
        return slot_joints, slot_weights

    def rest_vertices(self, identity: np.ndarray) -> np.ndarray:
        """The unposed mesh of one identity (45 coefficients), centimetres."""
        torch = self._torch
        with torch.no_grad():
            coeffs = torch.as_tensor(np.asarray(identity, dtype=np.float32)[None])
            return self._character.blend_shape.forward(coeffs)[0].numpy().astype(np.float64)

    # -- skeleton ------------------------------------------------------------------

    def local_states(self, model_params: np.ndarray) -> np.ndarray:
        """``(N, 127, 8)`` joint states relative to each parent."""
        torch = self._torch
        params = np.asarray(model_params, dtype=np.float32)
        with torch.no_grad():
            full = torch.cat(
                [torch.as_tensor(params), torch.zeros(len(params), 45)], dim=1
            )
            joint_params = self._character.model_parameters_to_joint_parameters(full)
            local = self._character.skeleton.joint_parameters_to_local_skeleton_state(joint_params)
        return local.numpy().astype(np.float64)

    def global_states(self, local: np.ndarray) -> np.ndarray:
        """Forward kinematics: ``(..., 127, 8)`` local states to model-space states."""
        out = np.array(local, dtype=np.float64, copy=True)
        for joint, parent in enumerate(self.parents):
            if parent >= 0:
                out[..., joint, :] = quat.compose(out[..., parent, :], local[..., joint, :])
        return out

    def skin(self, global_state: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Linear blend skinning of ``rest`` (V, 3) by one frame's global states."""
        joints, weights = self.skinning
        transforms = quat.compose(global_state, self.inverse_bind)  # (127, 8)
        out = np.zeros_like(rest)
        for slot in range(4):
            out += weights[:, slot : slot + 1] * quat.apply(transforms[joints[:, slot]], rest)
        return out

    def reference_vertices(self, model_params: np.ndarray, identity: np.ndarray, correctives: bool) -> np.ndarray:
        """The TorchScript model's own skinned mesh, for validation."""
        torch = self._torch
        with torch.no_grad():
            verts, _ = self.module(
                torch.as_tensor(np.asarray(identity, dtype=np.float32)[None]),
                torch.as_tensor(np.asarray(model_params, dtype=np.float32)[None]),
                torch.zeros(1, 72),
                correctives,
            )
        return verts[0].numpy().astype(np.float64)
