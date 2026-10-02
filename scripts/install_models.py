"""Put the model files in ``models/``.

* SAM 3.1 (``facebook/sam3.1``, ``sam3.1_multiplex.pt``): video tracking;
* SAM 3D Body (``facebook/sam-3d-body-dinov3``): the 3D bodies, and the
  Momentum Human Rig body model the 3D scene is built with;
* the DINOv3 source (Torch Hub), where SAM 3D Body loads its backbone code from;
* DINOv3 ViT-H+ (``timm/vit_huge_plus_patch16_dinov3.lvd1689m``, under the
  DINOv3 License): what each person looks like, to tell people apart;
* MoGe-2 (``Ruicheng/moge-2-vitl-normal``): the lens's focal length.

SAM 3.1 and SAM 3D Body are gated on Hugging Face: accept their licenses on
the model pages, then log in (``hf auth login``) before running this.

    uv run --no-sync python scripts/install_models.py [--offline] [--scene-only]

``--scene-only`` fetches just what building and viewing scenes needs (SAM 3D
Body's body model), for a machine without a CUDA GPU.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
BODY = MODELS_ROOT / "sam-3d-body-dinov3"
BODY_FILES = ("model_config.yaml", "model.ckpt", "assets/mhr_model.pt")
DINOV3 = MODELS_ROOT / "torch" / "hub" / "facebookresearch_dinov3_main"
SAM31 = MODELS_ROOT / "sam3.1"
MOGE = MODELS_ROOT / "moge-2-vitl-normal"
DINOV3_WEIGHTS = MODELS_ROOT / "dinov3-vith16plus"


def install_file(repo_id: str, filename: str, folder: Path, offline: bool, label: str) -> None:
    if (folder / filename).is_file():
        print(f"ready: {label} -> {folder / filename}")
        return
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=folder, local_files_only=offline)
    print(f"installed: {label} -> {folder / filename}")


def install_body(offline: bool) -> None:
    if all((BODY / name).is_file() for name in BODY_FILES):
        print(f"ready: SAM 3D Body -> {BODY}")
        return
    snapshot_download(
        repo_id="facebook/sam-3d-body-dinov3",
        local_dir=BODY,
        allow_patterns=["model_config.yaml", "model.ckpt", "assets/*"],
        local_files_only=offline,
    )
    missing = [name for name in BODY_FILES if not (BODY / name).is_file()]
    if missing:
        raise SystemExit(f"incomplete SAM 3D Body download: {', '.join(missing)}")
    print(f"installed: SAM 3D Body -> {BODY}")


def install_dinov3() -> None:
    if DINOV3.is_dir():
        print(f"ready: DINOv3 source -> {DINOV3}")
        return
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    cached = torch_home / "hub" / DINOV3.name
    if cached.is_dir():
        DINOV3.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cached, DINOV3)
        print(f"installed: DINOv3 source -> {DINOV3}")
        return
    import torch

    torch.hub.set_dir(str(DINOV3.parent))
    torch.hub.load("facebookresearch/dinov3", "dinov3_vith16plus", source="github", pretrained=False, trust_repo=True)
    print(f"installed: DINOv3 source -> {DINOV3}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="only reuse files already downloaded")
    parser.add_argument("--scene-only", action="store_true", help="skip the GPU-only models")
    args = parser.parse_args()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    install_body(args.offline)
    if not args.scene_only:
        install_file("facebook/sam3.1", "sam3.1_multiplex.pt", SAM31, args.offline, "SAM 3.1")
        install_file("Ruicheng/moge-2-vitl-normal", "model.pt", MOGE, args.offline, "MoGe-2")
        install_dinov3()
        install_file("timm/vit_huge_plus_patch16_dinov3.lvd1689m", "model.safetensors", DINOV3_WEIGHTS, args.offline, "DINOv3 ViT-H+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
