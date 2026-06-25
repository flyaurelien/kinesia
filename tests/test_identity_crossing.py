"""Crossing benchmark for the identity-locked subject tracker.

The hard failure the tracker must survive is two people with crossing
trajectories: when they overlap, their appearance crops blend and a greedy
single-target matcher can flip the lock onto the other person ("identity
swap"). This harness drives a deterministic two-person crossing and measures:

  * id_switches  — times the locked box flips between the two people,
  * on_patient   — fraction of unambiguous frames the box is on the patient,
  * ended_on_patient — whether the final lock is the patient.

It is the regression guard / measurement bed for the tracker-robustness work.
Synthetic frames use saturated colour blocks so the HSV appearance histogram is
discriminative; the closed loop feeds the previous output back as the pose
model's self-fed box (as the real pipeline does), which is what lets a crossing
drag the lock onto the wrong person.
"""

from __future__ import annotations

import unittest

import numpy as np

from sam_3d_pose_estimation.pipeline import (
    IdentityLockedBboxTracker,
    bbox_center,
    extract_bbox_appearance_hist,
)

SHAPE = (480, 854, 3)
PATIENT_BLUE = (200, 90, 40)  # BGR, saturated
DISTRACTOR_GREEN = (40, 190, 60)
BG = 128


def box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def det(b):
    return {"bbox": b, "score": 1.0, "source": "test"}


def make_frame(boxes_colors):
    """Grey background with filled colour boxes drawn in order (later overwrites)."""
    frame = np.full(SHAPE, BG, dtype=np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        frame[int(y1):int(y2), int(x1):int(x2)] = color
    return frame


def crossing_boxes(n_frames=48, box_w=130, box_h=320):
    """Patient (blue) left->right and distractor (green) right->left, crossing mid-clip."""
    w = SHAPE[1]
    y1 = 90
    y2 = y1 + box_h
    lo, hi = 50.0, float(w - 50 - box_w)
    seq = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        px = lo + (hi - lo) * t
        dx = hi + (lo - hi) * t
        seq.append((box(px, y1, px + box_w, y2), box(dx, y1, dx + box_w, y2)))
    return seq


def run_crossing(tracker_factory, n_frames=48, box_w=130, distractor_color=DISTRACTOR_GREEN, **factory_kw):
    """Drive a crossing and return metrics; closed-loop self-fed box = prev output."""
    seq = crossing_boxes(n_frames, box_w=box_w)
    p0, _ = seq[0]
    tracker = tracker_factory(p0, **factory_kw)

    prev_out = p0.copy()
    labels: list[str | None] = []
    present = 0
    for pbox, dbox in seq:
        # Patient drawn first, distractor second, so the overlap region blends
        # toward the distractor in BOTH crops — the realistic ambiguity.
        frame = make_frame([(pbox, PATIENT_BLUE), (dbox, distractor_color)])
        # Realistic self-fed box: the pose model re-fits each frame and snaps to
        # whichever person is nearest where it last sat — so it follows the
        # subject normally but can be hijacked onto the other person at a cross.
        proposed = min(
            (pbox, dbox),
            key=lambda b: abs(float(bbox_center(b)[0]) - float(bbox_center(prev_out)[0])),
        )
        out, info = tracker.update(frame, proposed, detections=[det(pbox), det(dbox)])
        prev_out = out.copy()
        if info["present"]:
            present += 1
        # Classify which person the locked box is on (skip ambiguous overlap).
        ox = float(bbox_center(out)[0])
        pcx = float(bbox_center(pbox)[0])
        dcx = float(bbox_center(dbox)[0])
        if abs(pcx - dcx) < box_w * 0.6:
            labels.append(None)  # overlapping: assignment is ambiguous
        else:
            labels.append("P" if abs(ox - pcx) <= abs(ox - dcx) else "D")

    seen = [l for l in labels if l is not None]
    switches = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
    on_patient = (sum(1 for l in seen if l == "P") / len(seen)) if seen else 0.0
    ended_on_patient = bool(seen and seen[-1] == "P")
    return {
        "id_switches": switches,
        "on_patient": on_patient,
        "ended_on_patient": ended_on_patient,
        "present_ratio": present / len(seq),
        "labels": "".join(l or "-" for l in labels),
    }


def default_tracker(patient_box, *, occlusion_hold_enabled=True, **overrides):
    """A tracker seeded + warmed up on the blue patient (settled lock)."""
    frame = make_frame([(patient_box, PATIENT_BLUE)])
    seed = extract_bbox_appearance_hist(frame, patient_box)
    params = dict(
        initial_bbox=patient_box,
        frame_shape=SHAPE,
        warmup_frames=3,
        gallery_floor=0.30,
        coast_gallery_floor=0.45,
        absence_patience=6,
        gallery_seeds=[seed] if seed is not None else None,
    )
    params.update(overrides)
    tracker = IdentityLockedBboxTracker(**params)
    tracker.occlusion_hold_enabled = occlusion_hold_enabled
    for _ in range(params["warmup_frames"] + 1):
        tracker.update(frame, patient_box, detections=[det(patient_box)])
    return tracker


class CrossingBenchmarkTests(unittest.TestCase):
    """Measure identity stability through a two-person crossing."""

    def test_crossing_metrics_are_reported(self):
        """The harness runs end-to-end and the subject is tracked throughout."""
        m = run_crossing(default_tracker)
        # The subject is visible every frame, so the tracker must stay present.
        self.assertGreater(m["present_ratio"], 0.85)
        # Sanity: at least some unambiguous frames were classified.
        self.assertGreater(len(m["labels"]), 0)

    def test_no_identity_swap_through_crossing(self):
        """The lock must stay on the patient through the crossing — no swap.

        Occlusion-aware hold carries the subject through the overlap on
        constant-velocity motion and re-anchors by the frozen gallery once the
        people separate, so the box never flips onto the other person.
        """
        m = run_crossing(default_tracker)
        self.assertEqual(m["id_switches"], 0)
        self.assertTrue(m["ended_on_patient"])

    def test_occlusion_hold_disabled_still_swaps(self):
        """Control: with the occlusion hold off, the greedy matcher still swaps."""
        m = run_crossing(default_tracker, occlusion_hold_enabled=False)
        self.assertFalse(m["ended_on_patient"])

    def test_no_swap_through_identical_appearance_crossing(self):
        """Hardest case: two identically-dressed people crossing — appearance
        cannot disambiguate, so only motion continuity can. The detection-driven
        occlusion hold carries the subject through and the nearest-detection
        re-anchor on exit keeps the lock on the patient."""
        m = run_crossing(default_tracker, distractor_color=PATIENT_BLUE)
        self.assertEqual(m["id_switches"], 0)
        self.assertTrue(m["ended_on_patient"])

    def test_identical_appearance_swaps_without_hold(self):
        """Control: the identical-appearance crossing swaps with the hold off."""
        m = run_crossing(default_tracker, distractor_color=PATIENT_BLUE, occlusion_hold_enabled=False)
        self.assertFalse(m["ended_on_patient"])


if __name__ == "__main__":
    import json

    print(json.dumps(run_crossing(default_tracker), indent=2))
