"""Local settings: model locations and, optionally, a custom runner.

Everything has a default under ``models/``. To override, put ``KEY=VALUE``
lines in ``local.env`` at the project root (git-ignored; see
``local.env.example``) or set them in the environment, which wins.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import models_root, project_root


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines, ignoring blanks, comments and optional quotes."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_local_env(path: Path | None = None) -> None:
    """Apply ``local.env`` to the environment, without overriding what is set."""
    path = path or project_root() / "local.env"
    if path.is_file():
        for key, value in parse_env_file(path.read_text()).items():
            os.environ.setdefault(key, value)


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root() / path


def model_files() -> dict[str, Path]:
    """What the GPU stages need, by environment variable."""
    root = models_root()
    return {
        "SAM31_CHECKPOINT": _path("SAM31_CHECKPOINT", root / "sam3.1" / "sam3.1_multiplex.pt"),
        "SAM3D_BODY_DIR": _path("SAM3D_BODY_DIR", root / "sam-3d-body-dinov3"),
        "MOGE_WEIGHTS": _path("MOGE_WEIGHTS", root / "moge-2-vitl-normal" / "model.pt"),
        "DINOV3_WEIGHTS": _path("DINOV3_WEIGHTS", root / "dinov3-vith16plus" / "model.safetensors"),
    }


def inference_environment() -> dict[str, str]:
    """Environment for ``python -m kinesia.inference.job`` on this machine."""
    root = project_root()
    paths = [str(root / "src"), str(root / "vendor" / "sam-3d-body-main")]
    source = os.environ.get("SAM3_SOURCE", "").strip()
    if source:  # a facebookresearch/sam3 checkout instead of the installed package
        paths.append(str(_path("SAM3_SOURCE", Path(source))))
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = {
        **os.environ,
        **{key: str(value) for key, value in model_files().items()},
        "PYTHONPATH": os.pathsep.join(paths),
        "TORCH_HOME": str(models_root() / "torch"),  # DINOv3's code (Torch Hub), for SAM 3D Body and appearance
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MOMENTUM_ENABLED": "0",  # MHR's TorchScript model, not the optional Momentum build
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
    }
    return env
