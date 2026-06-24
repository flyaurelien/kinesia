from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sam_3d_pose_estimation.evaluation import evaluate_dataset, evaluate_prediction


class EvaluationMetricsTest(unittest.TestCase):
    """Unit tests for the FoG episode-level evaluation metrics."""

    def test_event_metrics(self) -> None:
        """Predicted episodes overlapping ground-truth yield the expected precision/recall/F1 and onset latency."""
        predicted = {
            "episodes": [
                {"label": "fog", "start_ms": 1000, "end_ms": 2200},
                {"label": "fog", "start_ms": 5200, "end_ms": 6100},
            ]
        }
        labels = {
            "episodes": [
                {"label": "fog", "start_ms": 900, "end_ms": 2000},
                {"label": "fog", "start_ms": 5000, "end_ms": 6200},
                {"label": "fog", "start_ms": 9000, "end_ms": 9800},
            ]
        }
        metrics = evaluate_prediction(
            predicted_payload=predicted,
            labels_payload=labels,
            duration_ms=12000,
            needs_review=False,
        )
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertGreater(metrics["f1_event"], 0.79)
        self.assertEqual(metrics["false_negative_events"], 1)
        self.assertIsNotNone(metrics["mean_onset_latency_ms"])

    def test_dataset_evaluation_resolves_relative_label_paths(self) -> None:
        """Manifest label paths are resolved relative to the dataset dir and aggregated per split."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "output" / "datasets" / "demo"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            labels_dir = dataset_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = dataset_dir / "manifest.jsonl"
            label_path = labels_dir / "run_a_events.json"
            label_path.write_text(json.dumps({
                "episodes": [{"label": "fog", "start_ms": 1000, "end_ms": 1800}],
            }), encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "run_id": "run_a",
                "split": "holdout",
                "label_events_path": "labels/run_a_events.json",
            }), encoding="utf-8")

            summary = evaluate_dataset(
                dataset_manifest_path=manifest_path,
                preset="clinical_fog_v1",
                project_root=root,
                analysis_lookup=lambda _run_id, _preset: {
                    "events": {
                        "episodes": [{"label": "fog", "start_ms": 1000, "end_ms": 1800}],
                    },
                    "qa": {"needs_review": False, "status": "interpretable"},
                    "duration_ms": 4000,
                },
            )

            self.assertEqual(summary["dataset_id"], "demo")
            self.assertEqual(summary["runs_evaluated"], 1)
            self.assertAlmostEqual(summary["splits"]["holdout"]["f1_event"], 1.0)


if __name__ == "__main__":
    unittest.main()
