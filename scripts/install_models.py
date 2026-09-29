"""Put the local model files in place.

Inference runs on the cluster, where SAM 3.1 and SAM 3D Body already live.
This machine only needs:

* SAM 3D Body (``facebook/sam-3d-body-dinov3``): its Momentum Human Rig body
  model and keypoint regressor are used to build the 3D scene locally;
* the DINOv3 source (Torch Hub), uploaded once to the cluster, where SAM 3D
  Body loads its backbone code from.

    uv run python scripts/install_models.py [--offline]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
BODY = MODELS_ROOT / "sam-3d-body-dinov3"
BODY_FILES = ("model_config.yaml", "model.ckpt", "assets/mhr_model.pt")
DINOV3 = MODELS_ROOT / "torch" / "hub" / "facebookresearch_dinov3_main"


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
    args = parser.parse_args()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    install_body(args.offline)
    install_dinov3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
