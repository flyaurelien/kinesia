"""Where Kinesia keeps runs, uploads and local model files.

Every location defaults to a folder of the checkout and can be redirected with
an environment variable, so the web app, the command line and the tests agree.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return the checkout root: the nearest parent holding ``pyproject.toml``."""
    configured = os.environ.get("KINESIA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "kinesia").is_dir():
            return candidate
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return base


def _configured(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root() / path


def runs_root() -> Path:
    """Folder holding one sub-folder per analysis (``KINESIA_RUNS_ROOT``)."""
    return _configured("KINESIA_RUNS_ROOT", project_root() / "output")


def models_root() -> Path:
    """Folder holding local model files (``KINESIA_MODELS_ROOT``)."""
    return _configured("KINESIA_MODELS_ROOT", project_root() / "models")


def run_dir(run_id: str) -> Path:
    """Folder of one analysis, validating the identifier."""
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError(f"invalid run id: {run_id!r}")
    return runs_root() / run_id


def mhr_model_path() -> Path:
    """The Momentum Human Rig TorchScript model shipped with SAM 3D Body."""
    return _configured(
        "KINESIA_MHR_MODEL", models_root() / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    )
