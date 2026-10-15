<h1 align="center">Kinesia</h1>

<p align="center"><b>Multi-person 3D motion capture from a single, ordinary video.</b><br>
Drop a clip in the browser: everyone in it comes back as an animated 3D body, on one shared floor.</p>

<p align="center">
  <a href="https://github.com/flyaurelien/kinesia/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/flyaurelien/kinesia/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: CC0-1.0" src="https://img.shields.io/badge/license-CC0--1.0-lightgrey.svg"></a>
  <a href="#benchmark-mupots-3d-fixed-camera-several-people"><img alt="MuPoTS-3D: 3DPCK 93.4" src="https://img.shields.io/badge/MuPoTS--3D-3DPCK%2093.4-2b7bba.svg"></a>
  <a href="#benchmark-3dpw"><img alt="3DPW: jitter halved" src="https://img.shields.io/badge/3DPW-jitter%20%E2%88%9249%25-2b7bba.svg"></a>
</p>

<p align="center">
  <img src="docs/demo.webp" width="100%" alt="Left: a night streetball game in which every player's SAM 3.1 mask is outlined and numbered. Right: the same players reconstructed as 3D bodies in one scene, in the same colours, while the camera slowly orbits.">
</p>

<p align="center"><sub>
Streetball at night, filmed from one fixed camera: 18 s at 1080p, up to 13 people
at a time, each rebuilt in 3D on one shared floor. Colours match between the two
sides. Source video by Khanh Hoang Minh on
<a href="https://www.pexels.com/video/a-group-of-people-playing-basketball-at-night-19570048/">Pexels</a>.
</sub></p>

Kinesia follows every person in a video (a match, a dance, someone cooking),
reconstructs each of them as a 3D body on every frame, and plays them back
together in one animated scene you can orbit, follow, view from above or see
through the original camera. It turns Meta's per-image models
([SAM 3.1](https://github.com/facebookresearch/sam3),
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body),
[DINOv3](https://github.com/facebookresearch/dinov3)) into a video system:
one identity per person for the whole clip, motion that is smooth and
physically plausible, and a web app to explore it.

- **Drop a video, get a 3D scene**: a local web app runs the analysis step by
  step and opens the viewer when it is ready.
- **Everyone, all the time**: people are tracked through crossings and
  occlusions, and recognised by their appearance when they come back. On the
  MuPoTS-3D benchmark, 99.8% of the annotated people are found, with no
  identity switch.
- **Accurate 3D pose**: 93.4 3DPCK on MuPoTS-3D (several people, one fixed
  camera), above the published methods it is compared with (best: 89.0).
- **Motion you can trust**: jitter halved on the 3DPW benchmark, six times
  less foot sliding than SAM 3D Body alone on MuPoTS-3D, jumps that follow
  gravity.
- **Measurements**: per-person distance, speed, jumps and joint angles,
  exported as CSV or JSON.

## Examples

<table>
<tr>
<td width="50%"><img src="docs/dance-solo.webp" width="100%" alt="A dancer on a basketball court at night (left) and her 3D body orbiting (right)."></td>
<td width="50%"><img src="docs/dance-group.webp" width="100%" alt="Five dancers on a court at night (left) and their five 3D bodies in their own colours (right)."></td>
</tr>
<tr>
<td><sub><b>Street dance, solo.</b> Fast turns and kicks; the two painted figures on the wall are recognised as pictures and left out.</sub></td>
<td><sub><b>Street dance, five dancers.</b> Each keeps one identity for the whole clip.</sub></td>
</tr>
<tr>
<td><img src="docs/football.webp" width="100%" alt="An amateur football game with about fifteen players in bibs (left) and the players as 3D bodies (right)."></td>
<td><img src="docs/cooking.webp" width="100%" alt="A cook mixing a batter behind a kitchen counter (left) and his full 3D body, legs included (right)."></td>
</tr>
<tr>
<td><sub><b>Amateur football, about fifteen players.</b> Players crossing, hiding each other and running out of the picture.</sub></td>
<td><sub><b>Cooking.</b> The counter hides the legs: the body model predicts them, and the floor is placed under them.</sub></td>
</tr>
<tr>
<td><img src="docs/sprint-start.webp" width="100%" alt="Two sprinters leaving the starting position on a track (left) and their two 3D bodies (right)."></td>
<td><img src="docs/squat-coaching.webp" width="100%" alt="A woman squatting with a barbell while her coach watches from behind (left) and both 3D bodies (right)."></td>
</tr>
<tr>
<td><sub><b>Sports science: a sprint start.</b> Two athletes from the crouch to full speed, each with their own joint angles and speed.</sub></td>
<td><sub><b>Strength coaching: a barbell squat.</b> The coach stands behind the athlete, half hidden; knee and hip flexion through every repetition.</sub></td>
</tr>
<tr>
<td><img src="docs/rehab.webp" width="100%" alt="A patient with a prosthetic leg exercising on a machine while a physiotherapist holds his shoulder (left) and both 3D bodies (right)."></td>
<td><sub><b>Rehabilitation.</b> A patient with a leg prosthesis exercises seated on a machine, his physiotherapist at his side. With one person seated and both nearly still, the feet alone cannot set the floor's tilt; it is taken from how their trunks stand. His feet rest on the machine, not the floor, so his size and distance come from hers, not from a floor he does not touch.</sub></td>
</tr>
</table>

<sub>Source videos: street dance by Mixkit
(<a href="https://mixkit.co/free-stock-video/a-young-woman-wearing-urban-trendy-clothes-dances-on-the-51323/">solo</a>,
<a href="https://mixkit.co/free-stock-video/a-group-of-trendy-urban-young-people-dancing-on-a-51294/">group</a>,
<a href="https://mixkit.co/license/#videoFree">Mixkit license</a>); football by Usman AbdulrasheedGambo and cooking by Gustavo Fring on Pexels
(<a href="https://www.pexels.com/video/31370180/">football</a>,
<a href="https://www.pexels.com/video/8779935/">cooking</a>,
<a href="https://www.pexels.com/license/">Pexels license</a>); sprint by Yaroslav Shuraev, squat by Ketut Subiyanto and
rehabilitation by Kampus Production on Pexels
(<a href="https://www.pexels.com/video/8691728/">sprint</a>,
<a href="https://www.pexels.com/video/4859747/">squat</a>,
<a href="https://www.pexels.com/video/6111017/">rehabilitation</a>).</sub>

## Where it fits

Kinesia works **offline**, on recorded video. On an A100 it needs about
0.15 s per frame to follow everyone with SAM 3.1 and 0.06 to 0.1 s per person
and per frame for SAM 3D Body, so a 10-second clip of two people takes about
1.5 minutes of GPU time (see [compute time](#compute-time)). It is not a
real-time system. What it gives instead is every person, identified for the
whole clip, with motion clean enough to measure:

- **Sports science and coaching**: sprint starts, jumps, squats and lifts;
  joint angles, speed and jump height per athlete, several athletes at once.
- **Rehabilitation and clinical research**: exercises and gait filmed in a
  practice or a lab, patient and therapist together, from one phone or camera
  on a stand. Kinesia is a research tool, not a medical device, and it is not
  yet validated against marker-based motion capture.
- **Ergonomics, dance, animation**: posture over a work task, choreography
  with several dancers, motion for 3D characters.

The camera must not move, and people should be seen whole, feet included: in
a boxing clip framed above the knees, no floor could be found and the
analysis stopped. When the camera is calibrated, `kinesia new --focal
<pixels>` uses its focal length instead of estimating it from the picture,
which matters for where people are, not for how they move (see
[the benchmark](#benchmark-mupots-3d-fixed-camera-several-people)).

## Benchmark: 3DPW

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) (von Marcard et al., ECCV
2018) is the standard benchmark for 3D human pose in real-world video: people
filmed outdoors with a phone, with ground-truth 3D joints from body-worn
sensors. On the whole **test set** (24 sequences, 34,585 person-frames, 97% of
the frames where a subject is visible), Kinesia is compared with the model it
builds on, SAM 3D Body run on each frame independently:

| Method | MPJPE ↓ (mm) | PA-MPJPE ↓ (mm) | Accel. error ↓ (mm/frame²) |
| --- | ---: | ---: | ---: |
| SAM 3D Body, frame by frame | 60.4 | 40.9 | 13.3 |
| **Kinesia** (tracking, identities, temporal model) | **60.3** | **40.7** | **6.8** (−49%) |

Kinesia halves the jitter (acceleration error) while keeping, and slightly
improving, per-frame accuracy: the smoothing removes noise without lagging
behind the motion. It is better on the acceleration error in all 24 sequences.

<details>
<summary>Protocol</summary>

- Both methods are scored on the same frames and the same 12 joints that the
  SMPL skeleton of the ground truth and the MHR skeleton of SAM 3D Body share
  (shoulders, elbows, wrists, hips, knees, ankles), each pose centred between
  its hips. Joint sets differ from the 14-joint protocol of most papers, so
  these numbers compare the two rows above, not other leaderboards.
- MPJPE: mean joint error in the camera's orientation. PA-MPJPE: the same
  after the best rotation, scale and shift. Accel. error: error of the joints'
  acceleration (Kanazawa et al., CVPR 2019), the usual measure of jitter.
- Each ground-truth subject is matched to a detection by its 2D joints, and
  that detection to the scene person drawn over it.
- 3DPW is filmed with a moving phone, so the analyses use `--moving-camera`:
  the floor, foot-contact and gravity steps, which assume a still camera, are
  off. What is scored is tracking, identities and the temporal model of the
  pose. Reproduce with `scripts/benchmark_3dpw.py` (see its header).
- SAM 3.1 tracking memory grows with the number of people and the length of
  the clip: 10 of the 24 sequences, long and full of passers-by, needed an
  80 GB or 141 GB GPU instead of a 40 GB one.
- SAM 3D Body lists 3DPW among its training data (Table 1 of its paper), so
  3DPW is not held out for either row; MuPoTS-3D below is.

</details>

## Benchmark: MuPoTS-3D (fixed camera, several people)

[MuPoTS-3D](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)
(Mehta et al., 3DV 2018) is the standard benchmark for multi-person 3D pose
from one fixed camera: 20 sequences (5 indoors at 30 fps, 15 outdoors at
60 fps), 2 or 3 annotated people each, 8,370 frames, true 3D joints from
markerless multi-view capture. Neither SAM 3D Body nor SAM 3 was trained on
it. Everything below is run end to end on the raw videos: no ground-truth
boxes, no training on MuPoTS.

**What is compared.** Kinesia is not a new pose network: it is a video
pipeline around Meta's per-image models. Two of its rows are scored on the
same detections, so the difference between them is what the pipeline adds.
SAM-Body4D, the closest open-source pipeline built on the same models, is run
end to end in its own automatic mode⁴:

| Row | People found and followed | 3D body | Identity | Over time |
| --- | --- | --- | --- | --- |
| SAM 3D Body, frame by frame | SAM 3.1 in video mode, text prompt `person` | SAM 3D Body (DINOv3 backbone) on each frame alone | SAM 3.1's masklets | nothing |
| **Kinesia** | the same | the same | masklets cut where they change hands and joined again by appearance (DINOv3 ViT-H+, colours) | see below |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | ViTDet boxes on the first frame where it finds anyone, followed by SAM 3.1; nobody is added later | SAM 3D Body on each masklet, its own MoGe-2 lens | SAM 3.1's masklets | Kalman filter on body pose, one body shape, smoothed body rotation |

In Kinesia, SAM 3.1 detects and segments everyone the prompt `person` finds
(at least 48 px tall, up to 64 people) and carries each mask through the clip
with its video memory; MoGe-2 measures the lens (or `--focal` gives it); SAM
3D Body reconstructs each masklet on each frame with that lens. The scene
build then fixes each person's body shape (median over their clearest views),
removes single-frame flips and filters joint rotations with a zero-lag
adaptive filter (One Euro: 2.5 Hz, β 0.4; the body's turn 2 Hz, β 0.5), slides
everyone along their camera ray onto one shared floor, solves each path under
contact and flight (feet that stay put, free fall in the air), and bends the
legs by inverse kinematics so that planted feet hold still. Kinesia analyses
at most 30 frames per second: in the 60 fps sequences, the frames in between
are interpolated.

### 3D pose

The official protocol: 3DPCK, the share of joints within 15 cm once each pose
is centred on its pelvis and its bones are given the true lengths, averaged
over the 20 sequences. "All" counts a missed person as wrong, so it also
measures detection; "Matched" scores only the people found.

| Method | 3DPCK All ↑ | 3DPCK Matched ↑ |
| --- | ---: | ---: |
| LCR-Net (Rogez et al.) | 53.8 | 62.4 |
| Mehta et al. 2018 | 65.0 | 69.8 |
| ROMP | 69.9 | 72.2 |
| BEV | 70.2 | 75.2 |
| 3DCrowdNet | 72.7 | 73.3 |
| SMAP | 73.5 | 80.5 |
| Moon et al. (RootNet + PoseNet) | 81.8 | 82.5 |
| Multi-HMR (ViT-L, 896 px) | 85.0 | 89.3 |
| SAT-HMR | 89.0 | 90.1 |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | 92.9 | 93.8 |
| SAM 3D Body, frame by frame¹ | 93.6 | 93.7 |
| **Kinesia** | 93.4 | 93.6 |

Kinesia finds every annotated person in 99.8% of frames. On the 2-person
sequences it scores 93.3 (All), on the 3-person ones 93.6: crowding does not
cost it. Every MuPoTS sequence has 2 or 3 people, so a lone person cannot be
scored here. SAM-Body4D reconstructs the people it follows as well (93.8
Matched) but finds them in 98.3% of frames: it prompts the tracker once, at
the start, so someone it loses is not found again (on TS14, a person hidden
by another for about 300 frames).

### Under occlusion

MuPoTS labels each joint as visible or hidden (by the person or someone
else; about a quarter of all joints). 3DPCK, All:

| Method | Hidden joints ↑ | Visible joints ↑ |
| --- | ---: | ---: |
| Mehta et al. 2018 | 48.7 | 70.0 |
| XNect (stage II) | 58.8 | 76.0 |
| Dabral et al. 2019 | 61.0 | 75.6 |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | 87.2 | 94.8 |
| SAM 3D Body, frame by frame¹ | 89.0 | 94.9 |
| **Kinesia** | **89.7** | 94.5 |

Time helps where the picture does not: hidden joints gain 0.7 point over the
frame-by-frame model.

### Where people are

The same evaluation, without centring each pose on its pelvis: PCK_abs puts
the pose where the method placed it in the camera's space. Root error is the
mean 3D distance between the true and the predicted pelvis; PCOD (SMAP) the
share of pairs of people correctly ordered in depth. Most published methods
use the camera's true intrinsics; Kinesia estimates the focal length from the
picture, or takes it with `--focal`, here fitted per sequence from the
ground truth as a calibrated camera would give it.

| Method | Focal length | PCK_abs All / Matched ↑ | Root error ↓ | PCOD ↑ |
| --- | --- | ---: | ---: | ---: |
| Moon et al. (RootNet + PoseNet) | known | 31.5 / 31.8 | – | 92.6² |
| SMAP | known | 35.4 / 38.7 | 23.3 cm | 97.0 |
| DRM (Jin et al. 2022) | not stated | 39.3 / 41.0 | – | – |
| Cheng et al. 2021 | not stated | – / 48.0 | – | – |
| Multi-HMR | known (unknown) | – | – | 97.9 (95.1) |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | estimated | 5.0 / 5.0 | 35.8 cm | 99.3 |
| SAM 3D Body, frame by frame¹ | estimated | 10.9 / 10.9 | 37.4 cm | 99.4 |
| **Kinesia** | estimated | 7.0 / 7.0 | 36.2 cm | 99.4 |
| SAM 3D Body, frame by frame¹ | calibrated | 30.3 / 30.4 | 22.2 cm | 99.4 |
| **Kinesia** | calibrated | 22.1 / 22.1 | 24.9 cm | 99.4 |

Kinesia, like SAM 3D Body, orders people in depth better than any published
method, but places them less well in absolute terms. Most of the gap is the lens: MoGe-2's focal
length is 5 to 20% off on these clips, and a monocular depth is off by as
much (31 cm of depth error on average, 18 cm with the true focal length).
Even with the true focal length, Kinesia's depth error is 18 cm against 15 cm
for SAM 3D Body alone: the path model, which smooths depth over time, costs
absolute position here, so Kinesia scores below SAM 3D Body alone and below
the methods built for absolute depth. Without the shared-floor correction it
would do worse still (18.5 PCK_abs, 27.1 cm). A known limit.

### Motion-capture quality

What motion capture is judged on, for the people each row matches, on the
same frames (estimated focal length). No published method reports these on
MuPoTS; the comparison is with the frame-by-frame model and SAM-Body4D.

| | SAM 3D Body, frame by frame | SAM-Body4D (SAM 3.1)⁴ | **Kinesia** |
| --- | ---: | ---: | ---: |
| MPJPE, pelvis-centred, 14 joints (mm) ↓ | 94.0 | 95.3 | 95.2 |
| PA-MPJPE (mm) ↓ | 62.5 | 63.4 | 63.8 |
| Joint angle error, mean (°) ↓ | 10.2 | 10.4 | 10.8 |
| — elbow / knee / shoulder / hip (°) | 15.2 / 10.2 / 7.4 / 8.2 | 15.4 / 10.2 / 7.7 / 8.2 | 15.3 / 11.6 / 7.7 / 8.6 |
| Limb length error (mm) ↓ | 42.5 | 44.0 | **41.7** |
| Limb length change over time, std (mm) ↓ | 10.5 | 10.5 | **9.4** |
| Velocity error (mm/s) ↓ | 537 | 579 | **466** |
| Acceleration error (m/s²) ↓ | 19.7 | 22.3 | **15.0** |
| Foot sliding while the true foot is still (mm per 1/30 s) ↓ | 19.5 | 27.7 | **3.3** |
| Error on the distance between two people (mm) ↓ | 166 | 179 | 170 |

Kinesia gives up about 1 mm of joint position and half a degree of joint
angle, most of it at the knees, where legs are bent to hold planted feet, and
gains motion that is closer to the truth: 13% less velocity error, 24% less
acceleration error and six times less foot sliding. SAM-Body4D's smoothing
(a Kalman filter on body pose, one body shape per clip) does not show in these
measures: its velocity and acceleration errors are 8% and 13% above the
frame-by-frame model's and its feet slide 40% more; Kinesia has 20% less
velocity error, a third less acceleration error and eight times less foot
sliding than SAM-Body4D. MuPoTS has no ground-truth surface, so mesh error
(PVE) cannot be measured.

### Tracking

The protocol of T3DP and PHALP and their released ground-truth boxes
(IDs = identity switches):

| Method | IDs ↓ | MOTA ↑ | IDF1 ↑ | HOTA ↑ |
| --- | ---: | ---: | ---: | ---: |
| T3DP | 38 | 62.1 | 79.1 | 59.2 |
| PHALP | 22 | **66.2** | 81.4 | 59.4 |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | **0** | 60.1 | **83.1** | **63.5** |
| **Kinesia** | **0** | 45.0 | 78.4 | 61.6 |
| *annotated people only³:* | | | | |
| ByteTrack (as run by TRACE) | 15 | 73.3 | 84.6 | 63.5 |
| TRACE | 0 | 86.9 | 93.4 | 65.3 |
| SAM-Body4D (SAM 3.1 instead of SAM 3)⁴ | 0 | 94.3 | 97.1 | 73.7 |
| **Kinesia** | **0** | **97.3** | **98.6** | **74.2** |

Kinesia makes no identity switch and has a higher HOTA than T3DP and PHALP
under the strict protocol, but a lower MOTA and IDF1: it also follows the
people MuPoTS does not annotate (a third person indoors, someone hidden behind
a subject, passers-by far away), which the protocol counts as false positives
(precision 65%). Leaving those people out of the output would raise MOTA
without making Kinesia any better, so it does not. SAM-Body4D, which follows
only the people it saw at the start (precision 72%), scores higher under the
strict protocol and lower once unannotated people are left out, where the
people it lost count against it.

### Compute time

Measured on MuPoTS with one A100 (40 GB), 3.9 people per frame on average
(annotated or not):

| Step | Time |
| --- | ---: |
| SAM 3.1 tracking | 0.15 s per frame |
| SAM 3D Body | 0.06 s per person and frame (0.24 s per frame) |
| DINOv3 appearance | 0.05 s per frame |
| Scene build (CPU, laptop) | 0.03 s per frame |
| Whole GPU job, model loading included | 0.78 s per frame |

Published speeds, for scale only (different GPUs, and not all of them time
detection or tracking): XNect runs in real time (over 30 fps on a GTX 1080
Ti, 512×320 input); SAT-HMR takes 42 ms per image (RTX 3090) and Multi-HMR
about 120 ms (ViT-L, 896 px, V100), both per image without tracking; WHAM
takes 113 ms per frame for its whole pipeline (A100); CoMotion
176 ms and 4DHumans 2,163 ms per frame with tracking (V100, measured by
CoMotion); SAM 3D Body alone runs at 0.6 to 1.0 frames per second with its
own detector (RTX 6000 Ada, measured by Fast SAM 3D Body). SAM-Body4D, as run
here (SAM 3.1, no occlusion refiner), took 0.53 s per frame on an H100
(80 GB), model loading included. Kinesia sits with
the offline methods: it trades speed for identities, a shared world and
cleaner motion.

<details>
<summary>Protocol</summary>

- Literature rows are copied from the papers. 3D pose: Mehta et al. and
  LCR-Net from [Mehta et al. 2018](https://arxiv.org/abs/1712.03453) (Table 1),
  Moon et al. from [RootNet](https://arxiv.org/abs/1907.11346) (Table 5), SMAP
  from [SMAP](https://arxiv.org/abs/2008.11469) (Table 2), ROMP, BEV,
  3DCrowdNet and Multi-HMR from [Multi-HMR](https://arxiv.org/abs/2402.14654)
  (Table 5a), SAT-HMR from [SAT-HMR](https://arxiv.org/abs/2411.19824)
  (Table 2). Occlusion: Mehta et al. 2018 (§5.1), [XNect](https://arxiv.org/abs/1907.00837)
  (Table 9), [Dabral et al.](https://arxiv.org/abs/1909.10854) (§5.2).
  Position: RootNet (Table 2), SMAP (Table 2, supplementary §3.2; PCOD of
  Moon et al. as measured by SMAP²), [DRM](https://openaccess.thecvf.com/content/CVPR2022/papers/Jin_Single-Stage_Is_Enough_Multi-Person_Absolute_3D_Pose_Estimation_CVPR_2022_paper.pdf)
  (Table 1), [Cheng et al.](https://arxiv.org/abs/2104.01797) (Table 2),
  Multi-HMR (Table 6). Tracking: T3DP and PHALP from
  [PHALP](https://arxiv.org/abs/2112.04477) (Table 2), ByteTrack and TRACE
  from [TRACE](https://arxiv.org/abs/2306.02850) (Table 2). Speeds:
  [XNect](https://arxiv.org/abs/1907.00837) §7.1, SAT-HMR Table 1, Multi-HMR
  §4.1, [WHAM](https://arxiv.org/abs/2312.07531) Table 5,
  [CoMotion](https://arxiv.org/abs/2504.12186) Table 2,
  [Fast SAM 3D Body](https://arxiv.org/abs/2603.15603) Tables 1–2.
- 3D pose follows the official MATLAB evaluation line for line: predictions
  matched to the annotations by their 2D joints (40 px), poses centred on the
  pelvis, bones given the true lengths, 14 joints scored. PCK_abs is the same
  with the bones laid from the predicted pelvis in camera coordinates, as the
  official script's absolute mode does; PCOD follows SMAP's code (a pair is
  right if the depth order agrees or both gaps are under 30 cm). Joints come
  from the body mesh as for SMPL-based methods: Meta's barycentric map
  carries the MHR mesh onto the SMPL topology, then SPIN's Human3.6M regressor
  gives the 17 joints. Other papers derive their joints in their own way.
- Motion-capture measures are computed on the true metric skeleton
  (`annot3`), pelvis-centred, without rescaling: joint angles at elbows,
  knees, shoulders (arm against trunk) and hips (thigh against trunk); limb
  lengths of upper arms, forearms, thighs and shanks; velocity and
  acceleration from successive frames 1/30 s apart; foot sliding as the
  ankle's travel while the true ankle moves less than 1 cm.
- Tracking uses py-motmetrics with IoU matching for IDs, MOTA and IDF1 over
  all sequences, and TrackEval's HOTA averaged per sequence, as in PHALP's
  evaluation code. Kinesia's boxes are those of its video overlay: SAM 3.1's
  mask box of each person on every frame Kinesia shows them in.
- ¹ SAM 3D Body alone has no frames between Kinesia's analysed ones, so its
  rows are scored on the analysed frames only (scored the same way, Kinesia
  gets 93.5 / 93.6). Motion-capture measures use the analysed frames for both.
- ³ Not everyone in the picture is annotated. As TRACE does for its
  baselines, the tracks of unannotated people are left out: a track is kept
  if its box covers an annotated person (IoU ≥ 0.5) in at least half of the
  annotated frames where it appears. TRACE does not publish its exact rule,
  so this half of the table is indicative.
- ⁴ [SAM-Body4D](https://arxiv.org/abs/2512.08406) (Gao, Miao and Han,
  2025; [code](https://github.com/gaomingqi/sam-body4d), commit 21af102)
  publishes no numbers. It is run in its own automatic mode
  (`scripts/offline_app.py`) on the same videos, every frame: ViTDet looks
  for people from frame 10 on and stops at the first frame where it finds
  anyone, their boxes prompt the video segmenter, which follows them through
  the clip, and SAM 3D Body reconstructs every masklet with SAM-Body4D's
  smoothing and its own MoGe-2 lens. Its segmenter is SAM 3, whose checkpoint
  (`facebook/sam3`) is gated and was not available here, so SAM 3.1 (the
  checkpoint Kinesia uses) stands in for it, prompted with the same boxes
  (as SAM's two box corners); the standalone SAM 3.1 tracker of this
  revision also needed its video loader fixed (it does not scale pixels to
  0–1). The row is SAM-Body4D's method with SAM 3.1, not the published
  system. Its Diffusion-VAS occlusion refiner is off (`completion.enable:
  false`, as its authors advise unless occlusions are severe and long). With
  it, on the two most occluded sequences, 3DPCK All drops from 94.5 to 93.1
  (TS18) and from 94.5 to 93.0 (TS19), on hidden joints from 92.1 to 87.6,
  at 5.8 s per frame instead of 0.67 (after a one-line fix to a crash when an
  occlusion reaches the last frame of one of its 64-frame batches). As published, its code gives every person
  the body shape and scale of the first person it reconstructs (people are
  reconstructed one at a time and its per-person table is keyed by their
  index in the batch, always 0); 3DPCK's bone rescaling hides this, limb
  lengths do not. Only how its outputs are saved differs: one file of MHR
  parameters instead of meshes and rendered frames
  (`scripts/benchmark_mupots_sambody4d.py`), whose rebuilt meshes match its
  own within 2 mm, scored by the same code. Pose, position and tracking use
  all frames, motion-capture measures Kinesia's analysed frames. It ran on
  80 GB H100s.
- The calibrated rows are separate analyses made with `kinesia new --focal`,
  the focal length fitted per sequence to the ground truth's 2D and 3D joints
  (1,101 to 1,509 px).
- Reproduce with `scripts/benchmark_mupots.py`, and SAM-Body4D with
  `scripts/benchmark_mupots_sambody4d.py` (see their headers). Kinesia's GPU
  stages ran on 40 GB A100s.

</details>

## What is hard here, and what Kinesia does about it

| In the clip above | How Kinesia handles it |
| --- | --- |
| A dozen players crossing, screening and hiding each other | **SAM 3.1 video tracking** (Object Multiplex): one session over the whole clip, a memory per person, re-conditioned on fresh detections, so identities hold through crossings and occlusions. |
| One camera, so no depth | **One shared floor.** The lens is measured from the picture (MoGe-2), one floor is fitted under everybody's feet, and each person is slid along their camera ray until their feet meet it. Their image never moves; only the depth error goes away. Someone whose feet rest higher (raised on a machine, on a step) would be pushed away and blown up: a correction that would make a person taller than 2.1 m or shorter than 0.9 m is not trusted, and they take the others' correction instead. |
| People cut by the frame or seen from behind | **SAM 3D Body** (Momentum Human Rig) rebuilds the whole body from what is visible, prompted with each person's mask so overlapping players are told apart. Where the feet are cut off, their depth is interpolated from neighbouring frames. |
| Sprints, jumps, sudden turns | **Motion that respects physics.** Body shape is fixed per person and joint rotations are smoothed without lag. Each person's path is solved once for the whole clip, close to the picture but loose in depth (one camera judges distance poorly), with planted feet held still on the floor and the body in free fall while airborne: constant speed across the floor, gravity downwards, as in physics-based motion capture. The legs are then bent by inverse kinematics so that planted feet hold one spot and never sink into the floor; a foot that wanders more than 10 cm during a contact was pivoting or stepping, and is left as the video shows it. |
| Players leaving and coming back, or swapped while hidden | **Re-identification by appearance.** Each person is cut out with their mask, background greyed, and described by DINOv3 (head, torso and legs separately) and by the colours of their top and bottom. Each video calibrates its own score: the same masklet seconds apart is one person, people seen at the same time are two. A masklet that slides onto someone else during an occlusion is cut where it changes; pieces are then joined most-confident first, never two people seen together, never faster than a sprint, and never when a rival candidate scores nearly as well. |
| Tracker slips | **Masklet cleaning.** Frames where a mask swallows a neighbour are dropped, a track handed over to someone else is cut in two, people whose feet never reach the floor (spectators in the stands) are left out, and so are painted figures (murals, posters), which hold the same pose and the same pixels for seconds, and masks much wider than the body found in them (a bench, a banner, a group); someone lying down or doing push-ups fills their box and is kept. A person half hidden for a moment keeps their pose; only their distance, a guess, is taken from the frames around. |

**Measured on this clip**: 434 frames, focal length 1338 px estimated from the
image, heights between 1.58 and 1.78 m. On one NVIDIA B300, tracking took 92 s
and body reconstruction 104 s; the 3D scene was then built in 15 s. SAM 3.1's
20 masklet pieces belong to 14 people (checked by hand, frame by frame):
re-identification finds 15, with every return of a player who left the picture
or was hidden for 7 s, and the two players whose masklets swapped during a
screen given back their own identity; no two people are ever merged.

The viewer shows when each person is on screen (presence lanes), and gives
distance, speed, jumps and joint angles per person, with CSV and JSON export.


## How it works

```mermaid
flowchart LR
    A[Video] --> B["Normalize<br/>(ffmpeg)"]
    B --> C["GPU stages<br/>SAM 3.1 tracking · lens · SAM 3D Body · DINOv3 appearance"]
    C --> D["Scene build (CPU)<br/>floor · identities · motion · metrics"]
    D --> E["3D viewer"]
```

The GPU stages run on your NVIDIA GPU, in a child process of the analysis.
Each stage is saved as soon as it completes, so an interrupted analysis resumes
where it stopped, and cancelling stops the GPU work at once. The scene build
needs only the CPU.

## Setup

**Requirements:** Linux with an NVIDIA GPU and its CUDA driver, with about
32 GB of GPU memory for a short clip (see [Capture advice](#capture-advice-and-limits)
for longer ones); [`uv`](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe` on the
`PATH`, and Node.js 24 for the web app.

SAM 3.1 and SAM 3D Body are gated: accept their licenses on Hugging Face
([facebook/sam3.1](https://huggingface.co/facebook/sam3.1),
[facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3))
and log in with `hf auth login`. Then, once:

```bash
uv sync --frozen --no-editable                     # Python environment
bash scripts/install_gpu.sh                        # SAM 3.1, SAM 3D Body and MoGe-2 packages
uv run --no-sync python scripts/install_models.py  # model weights into models/
uv run --no-sync kinesia doctor                    # checks the GPU and the model files
```

Model files can live elsewhere: set their paths in `local.env` (see
`local.env.example`). Without a GPU, the web app still builds and plays scenes
made on another machine (`install_models.py --scene-only`).

## Run

```bash
./dev.sh
```

Open <http://127.0.0.1:4001/>, choose **New analysis**, drop a video. The page
follows the analysis through its steps (prepare, tracking, camera, bodies,
appearance, 3D scene) and opens the viewer when it is ready.

Everything is also available from the command line:

```bash
uv run --no-sync kinesia new input/match.mp4 --name "Sunday match"          # add --moving-camera for a moving camera,
                                                                    # --focal <pixels> for a calibrated one
uv run --no-sync kinesia process <run-id>                           # GPU stages, then the 3D scene
uv run --no-sync kinesia cancel <run-id>                            # stop it; the GPU is freed
uv run --no-sync kinesia scene <run-id>                             # rebuild the 3D scene
uv run --no-sync kinesia appearance <run-id> [--device mps]         # older analyses: add appearance, rebuild
```

Analyses made before the appearance stage existed still build, with SAM 3D
Body's own image features standing in (they tell people apart less well);
`kinesia appearance` adds the stage on any CUDA or Apple GPU.

Each analysis lives in `output/<run-id>/`: the normalized `video.mp4`, the raw
GPU results in `raw/`, the viewer files in `scene/` and the GPU log in `gpu.log`.

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
  not supported for the full pipeline: even a few degrees of panning slide
  everyone sideways in 3D. For such clips, `kinesia new --moving-camera`
  keeps tracking, identities and pose smoothing and skips the floor, foot
  and gravity steps (this is how the 3DPW benchmark is run).
- Works best when people are seen **whole**, feet included, and at least
  ~50 pixels tall. The floor is found from the feet: if they never show, the
  analysis stops with "not enough foot points to find the floor".
  SAM 3.1 is prompted with `person`, which finds the most people: on two test
  clips it found 90% and 97% of everyone that any prompt found, against 86%
  and 54% for `human`, and under 55% for `player` or `athlete`. Spectators in
  the stands are left out by the floor test, not by the prompt.
- Clips of up to 6,000 frames (about 3 minutes at 30 fps). SAM 3.1's state
  grows with every frame, by about 19 MB with 20 people: measured peaks were
  29 GB of GPU memory for 434 frames with up to 13 people, and 48 GB for 1,261
  frames with about 20. A clip too long for the GPU stops within the first
  minutes, with the reason, rather than near the end.
- Depth from one camera is uncertain: distances and speeds are estimates, best
  compared between people of the same clip. Absolute size comes from SAM 3D
  Body's human prior, so unusually tall or short people are pulled towards
  average height (tall athletes come out shorter than they are), and their
  distances and speeds scale with it.
- Joining fragments is conservative: when a returning player looks as much
  like two absent people (team-mates in the same kit), they stay a new person
  rather than risk a wrong identity. Appearance is all it goes on: shirt
  numbers are not read.
- SAM 3.1's multiplex tracker at the pinned upstream revision crashes on long
  clips when an object sharing a conditioning frame is removed
  ([facebookresearch/sam3#572](https://github.com/facebookresearch/sam3/issues/572));
  Kinesia applies the fix proposed upstream (#573) at runtime.

## Development

```bash
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests   # backend, on the source tree
uv run --no-sync python scripts/benchmark_3dpw.py <3DPW>/sequenceFiles/test <run-id>...   # 3DPW scores
uv run --no-sync --with scipy --with motmetrics --with "trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git" \
  python scripts/benchmark_mupots.py <MultiPersonTestSet> <mupots_gt.pickle> <mhr2smpl_mapping.npz> <J_regressor_h36m.npy> <run-id>...   # MuPoTS-3D scores
cd web-viewer && npx tsc --noEmit && npm test && npm run build
```

```text
kinesia/
  scripts/          install_gpu.sh (GPU packages), install_models.py (model weights),
                    benchmark_3dpw.py, benchmark_mupots.py (benchmark scores)
  src/kinesia/
    pipeline.py     the steps of an analysis and the runner that carries them out
    inference/      the GPU stages: SAM 3.1 tracking, lens, SAM 3D Body, appearance
    scene/          floor, identities, motion smoothing, metrics, export
  web-viewer/       Next.js app: library, processing status, 3D viewer
  vendor/           SAM 3D Body code (the GPU stages import it)
  tests/            backend tests
```

## Models

All model weights come from their original publishers; none are redistributed here.

| Model | Role | Paper | Weights / code | License |
| --- | --- | --- | --- | --- |
| **SAM 3.1** (Object Multiplex) | video detection, segmentation and tracking | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) · [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM License (gated) |
| **SAM 3D Body** | per-person 3D body ([MHR](https://github.com/facebookresearch/MHR)) | [arXiv:2602.15989](https://arxiv.org/abs/2602.15989) | [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) · [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body) | SAM License (gated) |
| **DINOv3** | image encoder inside SAM 3D Body; ViT-H+ alone describes each person's appearance | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | [timm/vit_huge_plus_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m) · [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 License |
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
The animations in `docs/` are made from videos on Pexels and Mixkit (credited
above), used under the [Pexels license](https://www.pexels.com/license/) and the
[Mixkit license](https://mixkit.co/license/#videoFree). 3DPW and MuPoTS-3D
are not redistributed; they are downloaded from their authors under their
research licenses.
