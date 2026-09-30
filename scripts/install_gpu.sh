#!/usr/bin/env bash
# The Python packages of the GPU stages: SAM 3.1, SAM 3D Body and MoGe-2.
# Linux with an NVIDIA GPU and CUDA drivers; run after `uv sync`:
#
#   uv sync --frozen --no-editable
#   bash scripts/install_gpu.sh
#   uv run --no-sync python scripts/install_models.py
#
# Packages that depend on torch are installed without their dependencies, so
# the CUDA build of torch already in the environment is never replaced.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PIP=(uv pip install --python "$PY")

echo "=== packages without torch dependencies ==="
"${PIP[@]}" "ftfy==6.1.1" regex "iopath==0.1.10" portalocker safetensors pyyaml psutil \
  yacs omegaconf braceexpand einops lightning-utilities fsspec tqdm termcolor typing_extensions

echo "=== torch-dependent packages, without their dependencies ==="
"${PIP[@]}" --no-deps "timm==1.0.29" roma "pytorch-lightning==2.6.1" "torchmetrics==1.9.0" \
  "sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40" \
  "moge @ git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec" \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"

echo "=== verification ==="
PYTHONPATH="src:vendor/sam-3d-body-main" MOMENTUM_ENABLED=0 "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA GPU: check the NVIDIA driver and the torch build")
import sam3.model_builder  # noqa: F401
import sam_3d_body  # noqa: F401
from moge.model.v2 import MoGeModel  # noqa: F401
import kinesia.inference.job  # noqa: F401
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
print("GPU packages ok: now run scripts/install_models.py, then `kinesia doctor`")
PY
