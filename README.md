# Kinesia

**Multi-person 3D motion capture from a single ordinary video.**

Kinesia follows every person in a video — a football match, a training session,
a dance — reconstructs each of them as a 3D body on every frame, and plays them
back together in one animated 3D scene you can orbit, follow, view from above or
see through the original camera.

![Volleyball players reconstructed in one 3D scene](docs/viewer.png)

<sub>Eleven players of Paris Volley vs Resovia (2013), from one fixed camera. Source
video by Shev123, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/),
via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Paris_Volley_Resovia,_24_October_2013_-_20_-_Debut_Match.webm);
this image is shared under the same license.</sub>

## What it does

- **Tracks everyone, with memory.** SAM 3.1's video predictor (Object Multiplex)
  runs one session over the whole clip: each person is a *masklet* with its own
  memory, re-conditioned on fresh detections, so identities hold through
  crossings, occlusions and fast motion.
- **Brings people back.** Someone who leaves the picture and returns comes back
  as a new masklet; Kinesia joins the fragments using mask-pooled appearance
  features, a running-speed reachability test and a similarity threshold
  calibrated on each video (people visible together are certainly different).
- **Catches tracker slips.** A mask that swallows a neighbour for a frame or two
  is ignored for those frames; a masklet handed over to someone else (the
  body jumps further than anyone can run, sideways in the picture or in depth)
  is cut in two; people whose feet never meet the floor, such as spectators in
  the stands, are left out.
- **Reconstructs bodies.** SAM 3D Body (Momentum Human Rig) estimates every
  person on every frame, prompted with their SAM 3.1 mask so overlapping players
  are told apart.
- **One shared world.** The lens focal length is estimated from the image
  (MoGe-2); one floor is fitted under everybody's feet over the whole clip, and
  each person is moved along their camera ray until their feet meet it — which
  fixes the per-person depth errors of monocular reconstruction without changing
  where they appear in the image.
- **Fluid, grounded motion.** Body shape is held fixed per person; joint
  rotations are smoothed with a zero-lag adaptive filter; trajectories use a
  Kalman smoother that knows depth is the uncertain direction; planted feet are
  pinned to stop foot skating. Nothing is glued to the floor, so jumps keep their
  height.
- **A viewer built around people.** Presence lanes show when each person is on
  screen; a motion card gives distance, speed, jumps and joint angles with a
  chart; everything exports to CSV or JSON.

## How it works

```mermaid
flowchart LR
    A[Video] --> B["Normalize<br/>(ffmpeg, local)"]
    B --> C["remote GPU batch job<br/>SAM 3.1 tracking · lens · SAM 3D Body"]
    C --> D["Download<br/>the GPU is released"]
    D --> E["Scene build (local)<br/>floor · identities · motion · metrics"]
    E --> F["3D viewer"]
```

Inference runs on the **remote GPU server** (Kubernetes + the job scheduler) as a *batch*
workload: its pod ends with the computation, so the GPU is released — and the
billing stops — as soon as the results are written, if anything fails, or the
moment you cancel. A hard timeout bounds a job that hangs. Jobs are resumable
stage by stage, so a pre-empted job restarts where it stopped. Waiting for a
GPU costs nothing; when the preferred pool (B300) is full for four minutes the
job moves to the next one (A100).

The scene build needs only the CPU and runs on your machine after the download.

## Setup

**Cluster access** (network access): install `kubectl` and the the job scheduler CLI,
then `scheduler login`. Copy `cluster/config.env.example` to `cluster/config.env`
(git-ignored) and fill in your the job scheduler project, UID/GID, volume claim, and the
paths of the SAM 3.1 checkpoint (`facebook/sam3.1`, `sam3.1_multiplex.pt`), a
clean checkout of `facebookresearch/sam3` and SAM 3D Body
(`facebook/sam-3d-body-dinov3`) on the lab volume. Then, once (this also puts
MoGe-2 and its public weights on the volume):

```bash
uv sync --frozen --no-editable                    # local Python environment
uv run --no-sync python scripts/install_models.py # SAM 3D Body body model + DINOv3 code
uv run --no-sync kinesia cluster setup            # Python packages on the volume (CPU pod)
```

**Local tools:** [`uv`](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe` on the
`PATH`, and Node.js 24.

## Run

```bash
./dev.sh
```

Open <http://127.0.0.1:4001/>, choose **New analysis**, drop a video. The page
follows the job through its steps (upload, GPU, tracking, camera, bodies,
download, 3D scene) and opens the viewer when it is ready. Cancelling deletes
the cluster job immediately.

Everything is also available from the command line:

```bash
uv run --no-sync kinesia new input/match.mp4 --name "Sunday match" [--prompt "football player"]
uv run --no-sync kinesia process <run-id>                           # GPU job, download, scene
uv run --no-sync kinesia cancel <run-id>                            # delete the job, free the GPU
uv run --no-sync kinesia scene <run-id>                             # rebuild the 3D scene locally
```

Each analysis lives in `output/<run-id>/`: the normalized `video.mp4`, the raw
GPU results in `raw/` and the viewer files in `scene/`.

### Using the viewer

| Control | |
| --- | --- |
| Space · ← → · Shift+← → | play/pause · one frame · one second |
| O · F · T · C | orbit · follow the selected person · top view · through the recording camera, over the video |
| Click a body, a list row, a lane or an outline in the video | select that person |
| Double-click a name | rename |

The toolbar toggles body meshes, skeletons, motion trails, name tags and the
source video (picture-in-picture or side by side). Names and hidden people are
saved with the analysis.

## Capture advice and limits

- **The camera must be fixed** (tripod, stand, or rested on something): one
  camera frame serves the whole clip. A panning, zooming or hand-held camera is
  not supported.
- Works best when people are seen **whole** and at least ~50 pixels tall. The
  text prompt (default `person`) can be narrowed, e.g. `volleyball player`, so
  that spectators are not tracked.
- Depth from one camera is uncertain: distances and speeds are estimates, best
  compared between people of the same clip. Absolute size comes from SAM 3D
  Body's human prior, so unusually tall or short people are pulled towards
  average height, and their distances and speeds scale with it (professional
  volleyball players read about 1.6 m tall).
- Joining fragments is conservative: two players in the same kit who leave and
  re-enter at the same time may stay separate people rather than risk a swap.
- SAM 3.1's multiplex tracker at the pinned upstream revision crashes on long
  clips when an object sharing a conditioning frame is removed
  ([facebookresearch/sam3#572](https://github.com/facebookresearch/sam3/issues/572));
  Kinesia applies the fix proposed upstream (#573) at runtime.

## Development

```bash
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests   # backend, on the source tree
cd web-viewer && npx tsc --noEmit && npm test && npm run build
```

```text
kinesia/
  cluster/          GPU job and one-time setup scripts (run inside the pods)
  src/kinesia/
    cluster/        the job scheduler/kubectl orchestration from this machine
    remote/         code that runs on the GPU: SAM 3.1 tracking, lens, SAM 3D Body
    scene/          floor, identities, motion smoothing, metrics, export
  web-viewer/       Next.js app: library, processing status, 3D viewer
  vendor/           SAM 3D Body code (uploaded to the cluster with each job)
  tests/            backend tests
```

## Models

All model weights come from their original publishers; none are redistributed here.

| Model | Role | Paper | Weights / code | License |
| --- | --- | --- | --- | --- |
| **SAM 3.1** (Object Multiplex) | video detection, segmentation and tracking | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) · [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM License (gated) |
| **SAM 3D Body** | per-person 3D body ([MHR](https://github.com/facebookresearch/MHR)) | [arXiv:2602.15989](https://arxiv.org/abs/2602.15989) | [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) · [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body) | SAM License (gated) |
| **DINOv3** | image encoder inside SAM 3D Body | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 License |
| **MoGe-2** | camera focal length from a single image | [arXiv:2507.02546](https://arxiv.org/abs/2507.02546) | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) · [microsoft/MoGe](https://github.com/microsoft/MoGe) | MIT (code) / weights per model card |

## Citations

```bibtex
@misc{carion2025sam3segmentconcepts,
  title={SAM 3: Segment Anything with Concepts},
  author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman R{\"a}dle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Doll{\'a}r and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
  year={2025},
  eprint={2511.16719},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.16719},
}

@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}

@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```

## License

Kinesia's own code is dedicated to the **public domain** under
[CC0 1.0 Universal](LICENSE). The vendored `vendor/sam-3d-body-main` keeps Meta's
SAM License (included in that directory). Model weights are downloaded from their
original gated sources and remain subject to their own license terms.
`docs/viewer.png` is derived from a CC BY-SA 3.0 video (credited above) and is
shared under that license.
