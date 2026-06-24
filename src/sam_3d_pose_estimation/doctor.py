from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from .workspace import project_root_from


def run_doctor(
    *,
    checkpoint_path: Path,
    mhr_path: Path,
    sam3d_code_root: Path | None = None,
    sam3_code_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Collect a diagnostic snapshot of the runtime environment.

    Reports Python/torch versions, available compute devices, required CLI
    binaries, and whether model checkpoints and code roots are present, so
    setup problems can be spotted before running the pipeline.
    """
    project_root_resolved = project_root_from(project_root)
    sam3_cache_path: str | None = None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache("facebook/sam3", "sam3.pt")
        sam3_cache_path = str(cached) if isinstance(cached, str) else None
    except Exception:
        sam3_cache_path = None
    local_sam3_path = project_root_resolved / "models" / "sam3" / "sam3.pt"
    return {
        "project_root": str(project_root_resolved),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "binaries": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "uv": shutil.which("uv"),
        },
        "models": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
            "mhr_path": str(mhr_path),
            "mhr_exists": mhr_path.exists(),
            "sam3_checkpoint_cache": sam3_cache_path,
            "sam3_checkpoint_cached": sam3_cache_path is not None,
            "sam3_local_path": str(local_sam3_path),
            "sam3_local_exists": local_sam3_path.exists(),
        },
        "code_roots": {
            "sam3d_code_root": str(sam3d_code_root) if sam3d_code_root else None,
            "sam3d_code_exists": sam3d_code_root.exists() if sam3d_code_root else None,
            "sam3_code_root": str(sam3_code_root) if sam3_code_root else None,
            "sam3_code_exists": sam3_code_root.exists() if sam3_code_root else None,
        },
    }
