from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .workspace import DEFAULT_ANALYSIS_PRESET, DEFAULT_CONFIG_PROFILE, run_dir, sanitize_run_id

RUN_MANIFEST_VERSION = "sam3d.run.v1"
ANALYSIS_MANIFEST_VERSION = "sam3d.analysis.v1"
RUN_MANIFEST_FILE = "run_manifest.json"
RUN_METADATA_FILE = "run_metadata.json"
ANALYSIS_MANIFEST_FILE = "analysis_manifest.json"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a trailing 'Z'."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write `payload` as pretty-printed JSON, creating parent directories as needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(file_path: Path) -> dict[str, Any]:
    """Load a JSON object from `file_path`, raising if the top level is not a dict."""
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {file_path}")
    return payload


def run_manifest_path(run_id: str, project_root: Path | None = None) -> Path:
    """Return the path to a run's manifest file."""
    return run_dir(run_id, project_root) / RUN_MANIFEST_FILE


def run_metadata_path(run_id: str, project_root: Path | None = None) -> Path:
    """Return the path to a run's metadata file."""
    return run_dir(run_id, project_root) / RUN_METADATA_FILE


def ensure_run_layout(run_id: str, project_root: Path | None = None) -> Path:
    """Create the standard subdirectory layout for a run and return its root directory."""
    directory = run_dir(run_id, project_root)
    (directory / "meshes").mkdir(parents=True, exist_ok=True)
    (directory / "analysis").mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    return directory


def build_run_manifest(
    *,
    run_id: str,
    run_directory: Path,
    metadata: dict[str, Any],
    created_at: str | None = None,
    config_profile: str = DEFAULT_CONFIG_PROFILE,
) -> dict[str, Any]:
    """Assemble the top-level run manifest from pipeline metadata.

    Seeds the manifest with run identity, video/model info, and artifact paths;
    the analyses list starts empty and is populated by `append_analysis_to_run_manifest`.
    """
    processed_frames = int(metadata.get("total_frames_processed") or 0)
    fps = metadata.get("fps_output")
    run_manifest = {
        "schema_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "created_at": created_at or now_iso(),
        "updated_at": now_iso(),
        "source_video": str(metadata.get("video_input") or ""),
        "config_profile": config_profile,
        "inference_target": metadata.get("inference_target") or "body",
        "subject_count": 1,
        "frame_count": processed_frames,
        "processed_frames": processed_frames,
        "fps": float(fps) if isinstance(fps, (int, float)) else None,
        "video_width": metadata.get("video_width"),
        "video_height": metadata.get("video_height"),
        "model_versions": {
            "mhr_backend": metadata.get("mhr_backend"),
            "inference_precision": metadata.get("inference_precision_effective"),
            "mps_mode_requested": metadata.get("mps_mhr_mode_requested"),
            "checkpoint_path": metadata.get("checkpoint_path"),
            "mhr_path": metadata.get("mhr_path"),
        },
        "quality_summary": None,
        "artifacts": {
            "run_metadata": RUN_METADATA_FILE,
            "meshes": "meshes",
            "preview_video": _relative_if_in_dir(
                metadata.get("output_video"),
                run_directory,
            ),
            "logs": "logs",
        },
        "analysis_profiles": {"default": DEFAULT_ANALYSIS_PRESET},
        "analyses": [],
        "latest_analysis_id": None,
    }
    return run_manifest


def append_analysis_to_run_manifest(
    manifest: dict[str, Any],
    *,
    analysis_id: str,
    preset: str,
    parameters: dict[str, Any],
    qa_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy of `manifest` with the given analysis recorded.

    Replaces any existing entry with the same `analysis_id` (so re-runs are
    idempotent) and updates the latest-analysis pointer and quality summary.
    """
    next_manifest = deepcopy(manifest)
    analysis_rel_dir = f"analysis/{sanitize_run_id(analysis_id)}"
    entry = {
        "analysis_id": analysis_id,
        "preset": preset,
        "created_at": now_iso(),
        "parameters": parameters,
        "artifacts": {
            "manifest": f"{analysis_rel_dir}/{ANALYSIS_MANIFEST_FILE}",
            "signals": f"{analysis_rel_dir}/signals.json",
            "frames": f"{analysis_rel_dir}/frames.json",
            "qa": f"{analysis_rel_dir}/qa.json",
            "kinematics": f"{analysis_rel_dir}/kinematics.parquet",
        },
        "qa_summary": qa_summary,
    }
    analyses = list(next_manifest.get("analyses") or [])
    analyses = [item for item in analyses if item.get("analysis_id") != analysis_id]
    analyses.append(entry)
    next_manifest["analyses"] = analyses
    next_manifest["latest_analysis_id"] = analysis_id
    next_manifest["quality_summary"] = qa_summary
    next_manifest["updated_at"] = now_iso()
    return next_manifest


def build_analysis_manifest(
    *,
    run_id: str,
    analysis_id: str,
    preset: str,
    parameters: dict[str, Any],
    qa_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the standalone manifest stored inside a single analysis directory."""
    return {
        "schema_version": ANALYSIS_MANIFEST_VERSION,
        "run_id": run_id,
        "analysis_id": analysis_id,
        "preset": preset,
        "created_at": now_iso(),
        "parameters": parameters,
        "qa_summary": qa_summary,
        "artifacts": {
            "signals": "signals.json",
            "frames": "frames.json",
            "qa": "qa.json",
            "kinematics": "kinematics.parquet",
        },
    }


def _relative_if_in_dir(raw_value: Any, parent_dir: Path) -> str | None:
    """Make an absolute path relative to `parent_dir` when possible.

    Returns None for empty/non-string input, and leaves relative paths or paths
    outside `parent_dir` unchanged.
    """
    if not isinstance(raw_value, str) or not raw_value:
        return None
    path = Path(raw_value)
    if not path.is_absolute():
        return raw_value
    try:
        return str(path.relative_to(parent_dir))
    except ValueError:
        return raw_value
