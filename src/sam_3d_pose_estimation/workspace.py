"""Filesystem layout helpers for the Kinesia workspace.

Centralizes how run/upload/dataset directories are resolved so the pipeline and
viewer agree on paths. Locations default to ``<project>/output`` and
``<project>/input`` but can be redirected via the ``KINESIA_RUNS_ROOT`` and
``KINESIA_UPLOADS_ROOT`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CONFIG_PROFILE = "clinical_fog_workstation_v1"
DEFAULT_ANALYSIS_PRESET = "clinical_fog_v1"


def project_root_from(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or this file) to the dir holding ``pyproject.toml``.

    Falls back to the current working directory when no project marker is found.
    """
    base = (start or Path(__file__)).resolve()
    if base.is_file():
        base = base.parent
    for candidate in [base, *base.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def workspace_root(project_root: Path | None = None) -> Path:
    """Return the project's ``output`` directory (the workspace root)."""
    return project_root_from(project_root) / "output"


def _configured_project_path(env_name: str, fallback: Path, project_root: Path) -> Path:
    """Resolve an env-var override path, or ``fallback`` if it is unset.

    Relative override values are interpreted against ``project_root``; absolute
    ones are used as-is.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path


def runs_root(project_root: Path | None = None) -> Path:
    """Directory holding all run folders (``KINESIA_RUNS_ROOT`` or ``output``)."""
    root = project_root_from(project_root)
    return _configured_project_path("KINESIA_RUNS_ROOT", root / "output", root)


def uploads_root(project_root: Path | None = None) -> Path:
    """Directory holding uploaded inputs (``KINESIA_UPLOADS_ROOT`` or ``input``)."""
    root = project_root_from(project_root)
    return _configured_project_path("KINESIA_UPLOADS_ROOT", root / "input", root)


def datasets_root(project_root: Path | None = None) -> Path:
    """Directory holding generated datasets under the workspace root."""
    return workspace_root(project_root) / "datasets"


def run_dir(run_id: str, project_root: Path | None = None) -> Path:
    """Path to a single run's folder, with ``run_id`` validated/sanitized."""
    return runs_root(project_root) / sanitize_run_id(run_id)


def analysis_dir(run_id: str, analysis_id: str, project_root: Path | None = None) -> Path:
    """Path to one analysis subfolder of a run; both ids are sanitized."""
    return run_dir(run_id, project_root) / "analysis" / sanitize_run_id(analysis_id)


def sanitize_run_id(value: str) -> str:
    """Validate a run/analysis id, rejecting empty values and path separators.

    Guards against directory traversal and null bytes before the id is used to
    build a filesystem path.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("Run identifier cannot be empty")
    if "/" in text or "\\" in text or "\0" in text:
        raise ValueError("Invalid run identifier")
    return text
