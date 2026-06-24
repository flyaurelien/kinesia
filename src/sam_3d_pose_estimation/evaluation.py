from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import now_iso, read_json, write_json
from .workspace import datasets_root, project_root_from


@dataclass(frozen=True)
class Episode:
    """A single labelled time interval (default a freezing-of-gait event)."""

    start_ms: int
    end_ms: int
    label: str = "fog"


def parse_episodes(payload: dict[str, Any]) -> list[Episode]:
    """Extract valid FoG episodes from a payload, dropping malformed/non-FoG entries."""
    raw = payload.get("episodes")
    if not isinstance(raw, list):
        return []
    episodes: list[Episode] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "fog").lower() != "fog":
            continue
        start_ms = int(item.get("start_ms") or 0)
        end_ms = int(item.get("end_ms") or 0)
        if end_ms <= start_ms:
            continue
        episodes.append(Episode(start_ms=start_ms, end_ms=end_ms))
    return episodes


def event_overlap_ms(a: Episode, b: Episode) -> int:
    """Return the overlapping duration (ms) between two episodes, 0 if disjoint."""
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def match_events(
    predicted: list[Episode],
    labels: list[Episode],
) -> tuple[int, int, int, list[int]]:
    """Greedily match predictions to labels by max overlap (each label used once).

    Returns (true positives, false positives, false negatives, onset errors in ms).
    """
    matched_labels: set[int] = set()
    matched_predictions = 0
    onset_errors: list[int] = []
    for pred in predicted:
        best_idx = None
        best_overlap = 0
        for idx, label in enumerate(labels):
            if idx in matched_labels:
                continue
            overlap = event_overlap_ms(pred, label)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx is None or best_overlap <= 0:
            continue
        matched_labels.add(best_idx)
        matched_predictions += 1
        onset_errors.append(abs(pred.start_ms - labels[best_idx].start_ms))
    false_positives = max(0, len(predicted) - matched_predictions)
    false_negatives = max(0, len(labels) - matched_predictions)
    return matched_predictions, false_positives, false_negatives, onset_errors


def evaluate_prediction(
    *,
    predicted_payload: dict[str, Any],
    labels_payload: dict[str, Any],
    duration_ms: int,
    needs_review: bool,
) -> dict[str, Any]:
    """Score one run's predicted episodes against labels (event P/R/F1, onset latency, FP rate)."""
    predicted = parse_episodes(predicted_payload)
    labels = parse_episodes(labels_payload)
    tp, fp, fn, onset_errors = match_events(predicted, labels)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall <= 1e-9 else (2.0 * precision * recall) / (precision + recall)
    fp_duration_ms = 0
    for pred in predicted:
        overlap_ms = sum(event_overlap_ms(pred, label) for label in labels)
        fp_duration_ms += max(0, (pred.end_ms - pred.start_ms) - overlap_ms)
    duration_min = max(duration_ms / 60000.0, 1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1_event": f1,
        "matched_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "mean_onset_latency_ms": (sum(onset_errors) / len(onset_errors)) if onset_errors else None,
        "false_positive_duration_per_min_s": (fp_duration_ms / 1000.0) / duration_min,
        "non_interpretable_rate": 1.0 if needs_review else 0.0,
    }


def resolve_dataset_path(dataset_manifest_path: Path, raw_path: str) -> Path:
    """Resolve a manifest-relative path against the manifest's directory (absolute paths pass through)."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (dataset_manifest_path.parent / candidate).resolve()


def evaluate_dataset(
    *,
    dataset_manifest_path: Path,
    preset: str,
    analysis_lookup: callable,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every run in a JSONL manifest, aggregate by split, and write summary.json/latest.json.

    ``analysis_lookup(run_id, preset)`` supplies each run's analysis payload (events, duration, qa).
    """
    project_root_resolved = project_root_from(project_root)
    dataset_text = dataset_manifest_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in dataset_text.splitlines() if line.strip()]
    tuning_scores: list[dict[str, Any]] = []
    holdout_scores: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id") or "").strip()
        label_events_path_raw = str(row.get("label_events_path") or "").strip()
        label_events_path = resolve_dataset_path(dataset_manifest_path, label_events_path_raw) if label_events_path_raw else Path()
        split = str(row.get("split") or "holdout").strip().lower()
        if not run_id or not label_events_path.exists():
            continue
        analysis_payload = analysis_lookup(run_id, preset)
        labels_payload = read_json(label_events_path)
        metrics = evaluate_prediction(
            predicted_payload=analysis_payload["events"],
            labels_payload=labels_payload,
            duration_ms=int(analysis_payload["duration_ms"]),
            needs_review=bool(analysis_payload["qa"].get("needs_review")),
        )
        item = {
            "run_id": run_id,
            "split": split,
            "metrics": metrics,
            "qa": analysis_payload["qa"],
        }
        per_run.append(item)
        if split == "tuning":
            tuning_scores.append(metrics)
        else:
            holdout_scores.append(metrics)

    summary = {
        "dataset_id": dataset_manifest_path.parent.name,
        "preset": preset,
        "generated_at": now_iso(),
        "project_root": str(project_root_resolved),
        "runs_evaluated": len(per_run),
        "splits": {
            "tuning": aggregate_metrics(tuning_scores),
            "holdout": aggregate_metrics(holdout_scores),
        },
        "per_run": per_run,
    }
    output_dir = datasets_root(project_root_resolved) / dataset_manifest_path.parent.name / "evaluations" / preset
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "latest.json", summary)
    return summary


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each numeric metric across runs, ignoring None/non-numeric values."""
    if not rows:
        return {
            "count": 0,
            "precision": None,
            "recall": None,
            "f1_event": None,
            "mean_onset_latency_ms": None,
            "false_positive_duration_per_min_s": None,
            "non_interpretable_rate": None,
        }
    keys = [
        "precision",
        "recall",
        "f1_event",
        "mean_onset_latency_ms",
        "false_positive_duration_per_min_s",
        "non_interpretable_rate",
    ]
    aggregated: dict[str, Any] = {"count": len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        aggregated[key] = (sum(values) / len(values)) if values else None
    return aggregated
