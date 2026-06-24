# Kinesia

Kinesia is a local workstation app for **markerless 3D body reconstruction and
gait kinematics from ordinary video**. It uses Meta's **SAM 3D Body** to recover a
3D body mesh per frame, derives lower-body joint kinematics, and shows it all in a
browser-based 3D viewer that runs entirely on your machine.

> Kinesia is the reconstruction/kinematics core of an earlier freezing-of-gait
> tool, with the FoG probability, annotation, and ground-truth layers removed —
> focused purely on a fast, high-quality 3D reconstruction and clean kinematics.

## What you can do

- **Upload** a video.
- **Detect & pick the subject** — a streaming, text-prompted detector scans the
  whole video and previews every detected person; select the one to reconstruct
  (merge fragments of the same person into one subject if needed).
- **Reconstruct** with SAM 3D Body to get a per-frame 3D mesh and joint
  kinematics, streamed live into the 3D viewer.
- **Inspect** the reconstruction in 3D alongside the source video with the
  tracking box, and plot any joint signal.
- **Export** the full per-joint kinematics (CSV / JSON) and a tracking-box MP4.

## How it works

Kinesia is a **single Next.js app**: one process serves both the browser UI and
the backend API. When you process a video, the app spawns the Python pipeline
(`uv run sam3d …`) and reads/writes the repository's `input/` and `output/`
folders directly. Launch it **from the repository root** with `uv`, `ffmpeg`, and
the model weights available — `./dev.sh` wires it together.

> The fast in-viewer subject-detection preview uses an MLX build of SAM 3 that is
> Apple-Silicon-only; on Linux/Windows the reconstruction pipeline detects the
> subject itself (PyTorch SAM 3, CUDA/CPU). The 3D reconstruction (SAM 3D Body) is
> PyTorch and runs on every platform.

## Requirements

| Tool | Notes |
|------|-------|
| Python `>=3.12,<3.13` | provisioned automatically by `uv` |
| [`uv`](https://docs.astral.sh/uv/) | Python environment + runner |
| Node.js 18+ and npm | the web viewer |
| `ffmpeg` + `ffprobe` | must be on your `PATH` (video I/O) |
| Hugging Face account | the model weights are gated (see [step 2](#2-download-the-models)) |
| ~6 GB disk | for the model weights |

A GPU is optional — the pipeline picks the best device (NVIDIA CUDA → Apple
Silicon MPS → CPU); pass `--force-cpu` to override. Install `ffmpeg`:

```bash
brew install ffmpeg                 # macOS
sudo apt-get install -y ffmpeg      # Debian / Ubuntu
winget install --id Gyan.FFmpeg     # Windows (PowerShell) — ensure ffmpeg/ffprobe are on PATH
```

## 1. Install

```bash
uv sync                                  # creates .venv and installs the backend
cd web-viewer && npm install && cd ..    # installs the web viewer
```

`uv sync` registers the `sam3d` command used by the pipeline.

## 2. Download the models

Kinesia needs two **gated** Hugging Face models. Request access (one click) on
each page, log in with a token, then download (~6 GB):

```bash
uv run hf auth login    # token from https://huggingface.co/settings/tokens

# SAM 3D Body — 3D mesh recovery (~2.7 GB)
uv run hf download facebook/sam-3d-body-dinov3 --local-dir models/sam-3d-body-dinov3
# SAM 3 — open-vocabulary subject detector (~3.2 GB, into the HF cache)
uv run hf download facebook/sam3 sam3.pt config.json
```

The first command lands the files at `models/sam-3d-body-dinov3/{model.ckpt,
model_config.yaml,assets/mhr_model.pt}`. SAM 3 loads from the HF cache, and on
Apple Silicon the in-viewer detect preview auto-downloads the MLX SAM 3 weights
(`mlx-community/sam3-image`) on first use. After the one-time download the app
runs offline. Verify with:

```bash
uv run sam3d doctor --json
```

### Models, sources & licenses

All weights come from their original publishers — none are redistributed here.

| Model | Used for | Source | License |
|-------|----------|--------|---------|
| **SAM 3D Body** (`sam-3d-body-dinov3` + MHR head) | per-frame 3D body mesh | [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) | gated; see model card |
| **SAM 3** | subject detection (PyTorch, all platforms) | [facebook/sam3](https://huggingface.co/facebook/sam3) | gated; see model card |
| **SAM 3 (MLX)** | fast in-viewer detect preview (Apple Silicon) | [mlx-community/sam3-image](https://huggingface.co/mlx-community/sam3-image) | inherits SAM 3 |
| **DINOv3** backbone | image features in SAM 3D Body | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | see repo license |

Upstream code is vendored under `vendor/` so the pipeline runs straight after
clone, each under its own upstream license.

## 3. Run the app

```bash
./dev.sh
```

Then open <http://127.0.0.1:4001/>. Run it **from the repository root** — the app
locates `input/`, `output/`, and the Python environment relative to it.

Manual equivalent (any OS): `cd web-viewer && npm run dev -- --hostname 127.0.0.1 --port 4001`.

## Using the app

1. **Upload** a video.
2. **Detect & pick the subject** — enter a prompt (default `person`); the detector
   scans the whole video, you select the subject to reconstruct (merge fragments
   of one person if needed).
3. **Reconstruct** — the 3D viewer streams the mesh and kinematics as the job runs.
4. **Inspect & export** — view the 3D reconstruction with the tracking box, plot
   joint signals, and export the full per-joint kinematics (CSV/JSON) + a
   tracking-box MP4.

## Command-line usage (optional)

```bash
# reconstruct one video (automatic subject detection)
uv run sam3d run --video-input input/example.mp4 --run-id example_processed \
  --inference-target body --precision float32 --no-preview --output-codec h264

# derive kinematics for a run
uv run sam3d analyze --run-id example_processed
```

Artifacts are written under `output/<run_id>/`. On a machine with no GPU add `--force-cpu`.

## Configuration

Copy `web-viewer/.env.example` to `web-viewer/.env.local` to override defaults.

| Variable | Default | Purpose |
|----------|---------|---------|
| `KINESIA_RUNS_ROOT` | `output` | where processed runs are stored |
| `KINESIA_UPLOADS_ROOT` | `input` | where uploaded videos are stored |
| `NEXT_PUBLIC_KINESIA_BACKEND_URL` | empty | set only to split the frontend onto another host |
| `KINESIA_ALLOWED_ORIGINS` | `127.0.0.1:4001` | browser origins allowed to call the API |

## Validate

```bash
uv run sam3d doctor --json                       # environment + model files
uv run python -m unittest discover -s tests      # backend tests
cd web-viewer && npx tsc --noEmit && npm run build
```

## Layout

```text
kinesia/
  input/         source videos
  output/        processed runs + logs
  models/        local model weights (gitignored)
  src/           Python backend: detection, 3D reconstruction, kinematics (sam3d)
  tests/         backend tests
  web-viewer/    Next.js 3D viewer UI + API
  vendor/        SAM 3D Body, SAM 3, and MLX SAM 3 upstream code (required to run)
```
