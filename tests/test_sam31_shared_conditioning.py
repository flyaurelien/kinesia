"""The multiplex tracker must not wipe objects that share a conditioning frame.

Model-free reproduction of facebookresearch/sam3 issue 572: an "owner" object
holds a bucket's only conditioning frame and two other objects rely on it.
Removing the owner used to reset the whole bucket, and the next frame failed
with "No points are provided". Runs against Meta's code when ``SAM3_SOURCE``
points to a checkout of facebookresearch/sam3 (skipped otherwise).
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

SOURCE = os.environ.get("SAM3_SOURCE", "")


def _out(members):
    return {"local_obj_id_to_idx": {obj: i for i, obj in enumerate(members)}}


def _state(members_per_frame, cond_frame, owner, objects):
    index = {obj: i for i, obj in enumerate(objects)}
    per_object = {
        index[obj]: {
            "cond_frame_outputs": {cond_frame: _out([owner])} if obj == owner else {},
            "non_cond_frame_outputs": {
                f: _out(m) for f, m in members_per_frame.items() if f != cond_frame and obj in m
            },
        }
        for obj in objects
    }
    return {
        "multiplex_state": SimpleNamespace(total_valid_entries=len(objects)),
        "obj_id_to_idx": dict(index),
        "obj_idx_to_id": {i: obj for obj, i in index.items()},
        "obj_ids": list(objects),
        "tracking_has_started": True,
        "first_ann_frame_idx": cond_frame,
        "point_inputs_per_obj": {index[obj]: ({cond_frame: {}} if obj == owner else {}) for obj in objects},
        "mask_inputs_per_obj": {i: {} for i in index.values()},
        "temp_output_dict_per_obj": {i: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}} for i in index.values()},
        "output_dict_per_obj": per_object,
        "output_dict": {
            "cond_frame_outputs": {cond_frame: _out([owner])},
            "non_cond_frame_outputs": {f: _out(m) for f, m in members_per_frame.items() if f != cond_frame},
        },
        "consolidated_frame_inds": {"cond_frame_outputs": {cond_frame}, "non_cond_frame_outputs": set()},
        "frames_already_tracked": {f: {"reverse": False} for f in members_per_frame},
    }


@unittest.skipUnless(SOURCE and os.path.isdir(SOURCE), "set SAM3_SOURCE to a facebookresearch/sam3 checkout")
class SharedConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, SOURCE)
        from kinesia.inference.tracking import keep_shared_conditioning
        from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

        keep_shared_conditioning()
        cls.predictor = object.__new__(Sam3VideoTrackingMultiplexDemo)
        cls.predictor.is_dynamic_model = True

    def test_removing_the_owner_keeps_the_others(self):
        members = {0: [0], 1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 1, 2]}
        state = _state(members, cond_frame=0, owner=0, objects=[0, 1, 2])
        self.predictor.clear_all_points_in_frame(state, 0, 0, need_output=False)
        cond = state["output_dict"]["cond_frame_outputs"]
        self.assertEqual(list(cond), [2], "the first frame holding another object becomes the anchor")
        self.assertTrue(state["tracking_has_started"])
        for obj in (1, 2):
            memory = state["output_dict_per_obj"][state["obj_id_to_idx"][obj]]
            self.assertTrue(memory["cond_frame_outputs"] or memory["non_cond_frame_outputs"])

    def test_removing_the_only_object_still_resets(self):
        members = {0: [0], 1: [0], 2: [0]}
        state = _state(members, cond_frame=0, owner=0, objects=[0])
        self.predictor.clear_all_points_in_frame(state, 0, 0, need_output=False)
        self.assertEqual(state["output_dict"]["cond_frame_outputs"], {})
        self.assertFalse(state["tracking_has_started"])


if __name__ == "__main__":
    unittest.main()
