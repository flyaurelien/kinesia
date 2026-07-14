"""Multi-subject selections run once per subject: --subject-index picks which
subject of the chosen-subject track file a given run reconstructs, along with
its display label and palette colour."""

import json
import tempfile
import unittest
from pathlib import Path

from sam_3d_pose_estimation.cli import load_subject_track_anchors, select_subject_track


def _write(data: dict) -> str:
    path = Path(tempfile.mkdtemp()) / "chosen_subject_track.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestSubjectSelection(unittest.TestCase):
    def test_picks_subject_by_index_with_label_and_color(self):
        path = _write({
            "subjects": [
                {"subjectId": 0, "label": 1, "color": "#34d399",
                 "frames": {"0": [1, 2, 3, 4], "5": [2, 3, 4, 5]}},
                {"subjectId": 3, "label": 2, "color": "#60a5fa",
                 "frames": {"0": [9, 9, 20, 20]}},
            ],
        })
        s0 = select_subject_track(path, 0)
        s1 = select_subject_track(path, 1)
        self.assertEqual((s0["subject_id"], s0["label"], s0["color"]), ("0", "1", "#34d399"))
        self.assertEqual(len(s0["anchors"]), 2)
        self.assertEqual((s1["subject_id"], s1["label"], s1["color"]), ("3", "2", "#60a5fa"))
        # Out-of-range index clamps instead of crashing a queued job.
        self.assertEqual(select_subject_track(path, 9)["subject_id"], "3")
        self.assertEqual(len(load_subject_track_anchors(path, 1)), 1)

    def test_legacy_single_subject_format(self):
        path = _write({"frames": {"0": [1, 2, 3, 4]}})
        s = select_subject_track(path, 0)
        self.assertEqual(s["subject_id"], "0")
        self.assertEqual(len(s["anchors"]), 1)

    def test_missing_file_yields_none(self):
        self.assertIsNone(select_subject_track("/nonexistent/track.json", 0))
        self.assertEqual(load_subject_track_anchors("", 0), [])


if __name__ == "__main__":
    unittest.main()
