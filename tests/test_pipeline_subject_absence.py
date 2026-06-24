import unittest
from unittest.mock import patch

import numpy as np

from sam_3d_pose_estimation import pipeline
from sam_3d_pose_estimation.pipeline import append_subject_absent_record


class SubjectAbsenceRecordTests(unittest.TestCase):
    """Absence records must stay valid placeholders, never carry stale pose data."""

    def test_absent_record_has_no_mesh_or_joints(self) -> None:
        """An absent frame yields a marked-absent record with no mesh/bbox/joints."""
        records: list[dict] = []

        append_subject_absent_record(
            records,
            frame_idx=42,
            patient_bbox=None,
            reason="subject_not_initialized",
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["video_frame"], 42)
        self.assertIsNone(record["mesh_path"])
        self.assertIsNone(record["bbox_xyxy"])
        self.assertFalse(record["subject_present"])
        self.assertEqual(record["inference_status"], "subject_not_initialized")
        self.assertNotIn("joints_cam_xyz", record)
        self.assertNotIn("joints_space_cam_xyz", record)

    def test_absent_record_preserves_tracker_state_for_review(self) -> None:
        """Tracker/identity diagnostics are persisted on absent frames for later review."""
        records: list[dict] = []
        bbox = np.array([10, 20, 110, 220], dtype=np.float32)

        append_subject_absent_record(
            records,
            frame_idx=7,
            patient_bbox=bbox,
            reason="subject_lost",
            identity_info={
                "status": "lost_hold",
                "is_lost": True,
                "lost_frames": 5,
                "appearance_similarity": 0.25,
                "stability_score": 0.31,
                "reacquire_scanned": True,
                "reacquire_candidates": 2,
            },
        )

        record = records[0]
        self.assertEqual(record["bbox_xyxy"], [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(record["identity_lock_status"], "lost_hold")
        self.assertTrue(record["identity_is_lost"])
        self.assertEqual(record["identity_lost_frames"], 5)
        self.assertAlmostEqual(record["identity_appearance_similarity"], 0.25)
        self.assertAlmostEqual(record["identity_stability_score"], 0.31)
        self.assertTrue(record["identity_reacquire_scanned"])
        self.assertEqual(record["identity_reacquire_candidates"], 2)


class StrictSubjectDetectionTests(unittest.TestCase):
    """Auto-init relies solely on SAM3 prompts; no HOG/heuristic fallback guessing."""

    def test_auto_init_is_sam3_only_and_absent_when_no_prompt_match(self) -> None:
        """No SAM3 prompt match -> no box (subject absent), never a fallback guess."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        with patch.object(pipeline, "detect_sam3_prompt_candidates", return_value=[]):
            bbox, info = pipeline.auto_initialize_patient_bbox(
                frame_bgr=frame,
                auto_init_mode="sam3",
                auto_select_strategy="patient",
                auto_detector_threshold=0.5,
                sam3_text_prompts=("the patient",),
                sam3_detector=object(),
            )

        self.assertIsNone(bbox)
        self.assertEqual(info["num_candidates"], 0)
        self.assertIsNone(info["selected_source"])
        self.assertFalse(info["fallback_used"])

    def test_auto_init_selects_the_sam3_prompt_candidate(self) -> None:
        """A single SAM3 prompt match is selected and surfaced in the info payload."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        sam3_bbox = np.array([40, 10, 90, 110], dtype=np.float32)

        with patch.object(
            pipeline,
            "detect_sam3_prompt_candidates",
            return_value=[{"bbox": sam3_bbox, "score": None, "source": "sam3_prompt:the patient"}],
        ):
            bbox, info = pipeline.auto_initialize_patient_bbox(
                frame_bgr=frame,
                auto_init_mode="sam3",
                auto_select_strategy="patient",
                auto_detector_threshold=0.5,
                sam3_text_prompts=("the patient",),
                sam3_detector=object(),
            )

        self.assertIsNotNone(bbox)
        self.assertEqual(info["num_candidates"], 1)
        self.assertEqual(info["selected_source"], "sam3_prompt:the patient")

    def test_hog_fallback_code_is_removed(self) -> None:
        """Guards against re-introducing the old HOG people-detector fallback."""
        self.assertFalse(hasattr(pipeline, "detect_people_hog"))


if __name__ == "__main__":
    unittest.main()
