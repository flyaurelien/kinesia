"""Behavioural tests for the robust identity-locked subject tracker.

These exercise the three failure modes the tracker must survive:
  * the subject walking out of frame (-> explicit absence, no hallucinated mesh),
  * other people in frame (-> stay locked on the signalled subject),
  * the subject leaving and returning later (-> re-acquire on a gallery match).

Synthetic frames use saturated colour blocks so the HSV appearance histogram is
discriminative (blue patient vs red distractor over a desaturated background).
"""

import unittest

import numpy as np

from sam_3d_pose_estimation.pipeline import (
    IdentityLockedBboxTracker,
    extract_bbox_appearance_hist,
)

SHAPE = (720, 1280, 3)
PATIENT_BLUE = (200, 90, 40)  # BGR, high saturation
DISTRACTOR_RED = (40, 40, 200)


def make_frame(boxes_colors):
    """Desaturated grey background with filled colour boxes drawn on top."""
    frame = np.full(SHAPE, 128, dtype=np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        frame[int(y1):int(y2), int(x1):int(x2)] = color
    return frame


def box(x1, y1, x2, y2):
    """Build an [x1, y1, x2, y2] float32 bbox in the tracker's expected layout."""
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def det(b):
    """Wrap a bbox as a fully-trusted detector hit (max score, test source)."""
    return {"bbox": b, "score": 1.0, "source": "test"}


def build_tracker(patient_box, **overrides):
    """Construct a tracker seeded on the blue patient and run it through warmup.

    Returns a tracker that has already locked onto ``patient_box`` so each test
    starts from a settled state; ``overrides`` tweak individual constructor params.
    """
    frame = make_frame([(patient_box, PATIENT_BLUE)])
    seed = extract_bbox_appearance_hist(frame, patient_box)
    params = dict(
        initial_bbox=patient_box,
        frame_shape=SHAPE,
        warmup_frames=3,
        gallery_floor=0.30,
        coast_gallery_floor=0.45,
        absence_patience=4,
        gallery_seeds=[seed] if seed is not None else None,
    )
    params.update(overrides)
    tracker = IdentityLockedBboxTracker(**params)
    # Run warmup with the patient detected so the gallery and lock settle.
    for _ in range(params["warmup_frames"] + 1):
        tracker.update(frame, patient_box, detections=[det(patient_box)])
    return tracker


class IdentityTrackerScenarioTests(unittest.TestCase):
    """End-to-end scenarios for the identity-locked subject tracker."""

    def test_warmup_locks_and_reports_present(self):
        """After warmup the tracker is locked, holds a gallery, and is not lost."""
        patient = box(500, 100, 700, 600)
        tracker = build_tracker(patient)
        self.assertFalse(tracker.is_lost)
        self.assertGreaterEqual(len(tracker.fixed_gallery), 1)
        self.assertEqual(tracker.warmup_left, 0)

    def test_other_person_in_frame_does_not_steal_lock(self):
        """Self-fed box drifts onto a passer-by; detector evidence snaps back."""
        patient = box(500, 100, 700, 600)
        distractor = box(900, 120, 1080, 600)
        tracker = build_tracker(patient)

        frame = make_frame([(patient, PATIENT_BLUE), (distractor, DISTRACTOR_RED)])
        # The pose model's self-fed box has slid onto the red distractor, but the
        # detector still sees both people.
        out_bbox, info = tracker.update(
            frame, distractor, detections=[det(patient), det(distractor)]
        )

        self.assertTrue(info["present"])
        self.assertTrue(info["supported"])
        # Locked box must be the blue patient, not the red distractor.
        self.assertLess(abs(float(out_bbox[0]) - patient[0]), 60.0)
        self.assertGreaterEqual(tracker.total_distractor_blocks + len(tracker.distractors), 1)

    def test_subject_leaves_frame_becomes_absent(self):
        """No trusted patient detection -> explicit absence within patience."""
        patient = box(500, 100, 700, 600)
        distractor = box(900, 120, 1080, 600)
        tracker = build_tracker(patient)

        # Patient gone: only the red distractor remains visible.
        gone = make_frame([(distractor, DISTRACTOR_RED)])
        present_flags = []
        for _ in range(tracker.absence_patience + 2):
            _, info = tracker.update(gone, patient, detections=[det(distractor)])
            present_flags.append(info["present"])

        self.assertFalse(present_flags[-1])
        self.assertEqual(tracker.last_status, "absent")
        self.assertTrue(tracker.is_lost)

    def test_subject_returns_is_reacquired(self):
        """After going lost, a gallery-matching detection re-acquires the subject."""
        patient = box(500, 100, 700, 600)
        distractor = box(900, 120, 1080, 600)
        tracker = build_tracker(patient)

        gone = make_frame([(distractor, DISTRACTOR_RED)])
        for _ in range(tracker.absence_patience + 3):
            tracker.update(gone, patient, detections=[det(distractor)])
        self.assertTrue(tracker.is_lost)

        # Patient walks back in at a new location.
        back = box(300, 110, 500, 600)
        frame = make_frame([(back, PATIENT_BLUE), (distractor, DISTRACTOR_RED)])
        out_bbox, info = tracker.update(
            frame, patient, detections=[det(back), det(distractor)]
        )

        self.assertTrue(info["present"])
        self.assertEqual(info["status"], "reacquired_detector")
        self.assertLess(abs(float(out_bbox[0]) - back[0]), 60.0)
        self.assertFalse(tracker.is_lost)

    def test_self_fed_drift_without_detector_goes_lost_not_silent(self):
        """Between detector frames, a colour-mismatched drift must not be accepted."""
        patient = box(500, 100, 700, 600)
        tracker = build_tracker(patient)

        # Drift onto a red region with NO detector evidence this frame.
        drift = box(900, 120, 1080, 600)
        frame = make_frame([(drift, DISTRACTOR_RED)])
        _, info = tracker.update(frame, drift, detections=None)

        self.assertFalse(info["present"])
        self.assertEqual(info["status"], "lost_hold")

    def test_brief_identity_dip_is_held_then_lost_if_sustained(self):
        """A same-location appearance dip is graced briefly, then escalates."""
        patient = box(500, 100, 700, 600)
        tracker = build_tracker(patient)  # absence_patience=4
        # Geometrically continuous (same box) but appearance flips to white,
        # with no detector evidence: a mid-turn-style dip the tracker should
        # ride out for a few frames, then give up once it is clearly sustained.
        dip = make_frame([(patient, (235, 235, 235))])
        statuses = []
        for _ in range(tracker.absence_patience + 2):
            _, info = tracker.update(dip, patient, detections=None)
            statuses.append((info["present"], info["status"]))

        self.assertTrue(statuses[0][0])
        self.assertEqual(statuses[0][1], "coasting")
        self.assertFalse(statuses[-1][0])
        self.assertEqual(statuses[-1][1], "lost_hold")

    def test_coast_keeps_same_color_subject_between_detector_frames(self):
        """A small same-colour move with no detector frame is coasted, not dropped."""
        patient = box(500, 100, 700, 600)
        tracker = build_tracker(patient)

        moved = box(520, 100, 720, 600)
        frame = make_frame([(moved, PATIENT_BLUE)])
        _, info = tracker.update(frame, moved, detections=None)

        self.assertTrue(info["present"])
        self.assertIn(info["status"], {"tracked", "tracked_coast"})

    def test_anchor_reassertion_overrides_and_seeds_gallery(self):
        """An explicit anchor overrides tracking, snaps the box, and feeds the gallery."""
        patient = box(500, 100, 700, 600)
        tracker = build_tracker(patient)
        gallery_before = len(tracker.fixed_gallery)

        anchor = box(320, 130, 520, 600)
        frame = make_frame([(anchor, PATIENT_BLUE)])
        out_bbox, info = tracker.update(frame, None, anchor_bbox=anchor)

        self.assertTrue(info["present"])
        self.assertEqual(info["status"], "anchor")
        self.assertLess(abs(float(out_bbox[0]) - anchor[0]), 1.0)
        self.assertGreaterEqual(len(tracker.fixed_gallery), gallery_before)


if __name__ == "__main__":
    unittest.main()
