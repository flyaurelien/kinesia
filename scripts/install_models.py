"""Download every Kinesia model into the project-local models directory."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelSnapshot:
    """A Hugging Face snapshot and the files Kinesia needs from it."""

    repo_id: str
    directory: str
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


SNAPSHOTS = (
    ModelSnapshot(
        repo_id="facebook/sam-3d-body-dinov3",
        directory="sam-3d-body-dinov3",
        allow_patterns=("model_config.yaml", "model.ckpt", "assets/*"),
        required_files=("model_config.yaml", "model.ckpt", "assets/mhr_model.pt"),
    ),
    ModelSnapshot(
        repo_id="facebook/sam3",
        directory="sam3",
        allow_patterns=("sam3.pt", "config.json"),
        required_files=("sam3.pt", "config.json"),
    ),
    ModelSnapshot(
        repo_id="mlx-community/sam3-image",
        directory="sam3-mlx",
        allow_patterns=("*.safetensors", "*.json"),
        required_files=("model.safetensors", "model.safetensors.index.json"),
    ),
)


def snapshot_is_complete(root: Path, required_files: tuple[str, ...]) -> bool:
    """Return whether every required model file exists below ``root``."""
    return all((root / relative_path).is_file() for relative_path in required_files)


def cached_snapshot(spec: ModelSnapshot) -> Path | None:
    """Return the newest complete global-cache snapshot for ``spec``."""
    snapshots = Path(HF_HUB_CACHE) / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (entry for entry in snapshots.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    return next(
        (entry for entry in candidates if snapshot_is_complete(entry, spec.required_files)),
        None,
    )


def copy_cached_snapshot(source: Path, destination: Path, required_files: tuple[str, ...]) -> None:
    """Copy a symlinked Hub snapshot into ordinary project-local files."""
    for relative_path in required_files:
        source_file = source / relative_path
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = destination_file.with_name(f".{destination_file.name}.installing")
        shutil.copy2(source_file, temporary_file, follow_symlinks=True)
        temporary_file.replace(destination_file)


def install_snapshot(spec: ModelSnapshot, *, offline: bool) -> Path:
    """Materialize one model snapshot at its stable project-local path."""
    destination = MODELS_ROOT / spec.directory
    if snapshot_is_complete(destination, spec.required_files):
        print(f"ready: {spec.repo_id} -> {destination}")
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    cached = cached_snapshot(spec)
    if cached is not None:
        copy_cached_snapshot(cached, destination, spec.required_files)
        print(f"installed from cache: {spec.repo_id} -> {destination}")
        return destination

    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=destination,
        allow_patterns=list(spec.allow_patterns),
        local_files_only=offline,
    )
    if not snapshot_is_complete(destination, spec.required_files):
        missing = [name for name in spec.required_files if not (destination / name).is_file()]
        raise RuntimeError(f"incomplete {spec.repo_id} download: {', '.join(missing)}")
    print(f"installed: {spec.repo_id} -> {destination}")
    return destination


def install_cached_dinov3_source() -> Path:
    """Copy the cached DINOv3 source into the project-local Torch Hub directory.

    SAM 3D Body loads DINOv3 through Torch Hub even though its trained weights
    are already in the body checkpoint. Keeping the small source checkout here
    removes the final dependency on the user's global Torch cache.
    """
    destination = MODELS_ROOT / "torch" / "hub" / "facebookresearch_dinov3_main"
    if destination.is_dir():
        print(f"ready: DINOv3 source -> {destination}")
        return destination

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    source = torch_home / "hub" / "facebookresearch_dinov3_main"
    if not source.is_dir():
        print(
            "DINOv3 source is not cached yet; the first body run will download "
            f"it once into {destination.parent}."
        )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    print(f"installed: DINOv3 source -> {destination}")
    return destination


def main() -> int:
    """Install all first-party model artifacts required by the local pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only reuse files already present in the Hugging Face cache.",
    )
    args = parser.parse_args()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in SNAPSHOTS:
        install_snapshot(spec, offline=args.offline)
    install_cached_dinov3_source()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
