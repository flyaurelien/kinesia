"""Score Kinesia on the 3DPW test set against SAM 3D Body run frame by frame.

3DPW (von Marcard et al., ECCV 2018) films people outdoors with a moving
phone camera and gives their true 3D joints from body-worn sensors. Both
methods are scored on the same frames and the same 12 joints that the SMPL
skeleton of the ground truth and the MHR skeleton of SAM 3D Body share
(shoulders, elbows, wrists, hips, knees, ankles), each pose centred between
its hips:

  MPJPE      mean joint error, in the camera's orientation (mm)
  PA-MPJPE   the same after the best rotation, scale and shift (mm)
  Accel      error of the joints' acceleration, the usual jitter measure
             (mm/frame^2, Kanazawa et al. 2019)

"SAM 3D Body" is the raw estimate of each frame (``raw/bodies.npz``);
"Kinesia" is the person in the built scene (``scene/``): tracked, identified,
smoothed, put under contact physics. Each ground-truth subject is matched to
a detection by its 2D joints, and that detection to the scene person drawn
over it. The camera moves in 3DPW, so only body pose is
scored here, not the path through the world.

Usage: python scripts/benchmark_3dpw.py <3DPW sequenceFiles/test> <run id>... [--json out.json]
Each run must be an analysis of one test sequence, named after it
(``kinesia new <sequence>.mp4``), with its scene built.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinesia.paths import run_dir  # noqa: E402

# (SMPL joint, MHR70 keypoint): L/R shoulder, elbow, wrist, hip, knee, ankle
PAIRS = [(16, 5), (17, 6), (18, 7), (19, 8), (20, 62), (21, 41), (1, 9), (2, 10), (4, 11), (5, 12), (7, 13), (8, 14)]
SMPL = [s for s, _ in PAIRS]
MHR = [m for _, m in PAIRS]
HIPS = [6, 7]  # positions of the two hips in the 12-joint lists
MATCH = 0.25  # a detection matches a subject if its 2D joints are this close, in subject heights


def centred(joints: np.ndarray) -> np.ndarray:
    return joints - joints[..., HIPS, :].mean(axis=-2, keepdims=True)


def procrustes(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``pred`` moved onto ``truth`` by the best similarity transform (per frame)."""
    out = np.empty_like(pred)
    for i, (p, t) in enumerate(zip(pred, truth)):
        mp, mt = p.mean(0), t.mean(0)
        a, b = p - mp, t - mt
        u, s, vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(u @ vt))
        s[-1] *= d
        u[:, -1] *= d
        rotation = u @ vt
        scale = s.sum() / (a**2).sum()
        out[i] = scale * a @ rotation + mt
    return out


def read_scene(folder: Path) -> tuple[dict, dict]:
    """The built scene and, per person, frames and world keypoints (MHR70)."""
    scene = json.loads((folder / "scene" / "scene.json").read_text())
    raw = (folder / "scene" / scene["people_file"]).read_bytes()
    unit = scene["encoding"]["keypoint_unit"]
    people = {}
    for person in scene["people"]:
        layout = person["layout"]

        def section(name):
            d = layout[name]
            return np.frombuffer(raw, dtype=d["type"], count=int(np.prod(d["shape"])), offset=d["offset"]).reshape(d["shape"])

        people[person["id"]] = {
            "tracks": set(person["tracks"]),
            "frames": section("frames").astype(int),
            "keypoints": section("keypoints").astype(np.float64) * unit,
        }
    return scene, people


def world_to_camera(scene: dict) -> np.ndarray:
    """Rotation taking scene (world) directions to the camera's (x right, y down, z forward)."""
    x, y, z, w = scene["camera"]["rotation"]
    camera_to_world = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    return camera_to_world.T


def evaluate(sequence: Path, run: str) -> dict:
    gt = pickle.load(sequence.open("rb"), encoding="latin1")
    folder = run_dir(run)
    with np.load(folder / "raw" / "bodies.npz") as data:
        rows = {key: data[key] for key in ("frame", "track", "kp2d", "kp3d")}
    scene, people = read_scene(folder)
    rotation = world_to_camera(scene)
    position = np.asarray(scene["camera"]["position"], dtype=np.float64)
    focal = float(scene["camera"]["focal"])
    centre = np.array([scene["width"], scene["height"]]) / 2.0
    K = np.asarray(gt["cam_intrinsics"])
    results = []
    for subject in range(len(gt["poses"])):
        joints = np.asarray(gt["jointPositions"][subject]).reshape(-1, 24, 3)[:, SMPL]
        valid = np.asarray(gt["campose_valid"][subject]).astype(bool)
        truth, raw3d, ours, kept = [], [], [], []
        for frame in np.flatnonzero(valid):
            pose = np.asarray(gt["cam_poses"][frame])
            camera = joints[frame] @ pose[:3, :3].T + pose[:3, 3]
            uv = camera @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            height = np.ptp(uv[:, 1])
            index = np.flatnonzero(rows["frame"] == frame)
            if not len(index):
                continue
            distance = np.linalg.norm(rows["kp2d"][index][:, MHR, :2] - uv, axis=2).mean(axis=1)
            best = index[np.argmin(distance)]
            if distance.min() > MATCH * height:
                continue
            # The scene person drawn over that detection: the one whose body,
            # seen through the scene's camera, lands on it.
            found = None
            for person in people.values():
                at = np.flatnonzero(person["frames"] == frame)
                if not len(at):
                    continue
                body = (person["keypoints"][at[0]][MHR] - position) @ rotation.T
                seen = focal * body[:, :2] / body[:, 2:] + centre
                gap = np.linalg.norm(seen - rows["kp2d"][best][MHR, :2], axis=1).mean()
                if gap < MATCH * height and (found is None or gap < found[0]):
                    found = (gap, body)
            if found is None:
                continue
            truth.append(camera)
            raw3d.append(rows["kp3d"][best][MHR].astype(np.float64))
            ours.append(found[1])
            kept.append(frame)
        if not kept:
            continue
        truth, raw3d, ours, kept = centred(np.array(truth)), centred(np.array(raw3d)), centred(np.array(ours)), np.array(kept)
        results.append({"frames": kept, "truth": truth, "raw": raw3d, "ours": ours, "valid": int(valid.sum())})
    return {"sequence": sequence.stem, "subjects": results}


def accel(x: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Second differences over runs of three consecutive frames."""
    ok = (frames[2:] - frames[1:-1] == 1) & (frames[1:-1] - frames[:-2] == 1)
    return (x[2:] - 2 * x[1:-1] + x[:-2])[ok]


def score(evaluations: list[dict]) -> dict:
    sums: dict[str, list[float]] = {}
    covered = valid = 0
    for evaluation in evaluations:
        for s in evaluation["subjects"]:
            covered += len(s["frames"])
            valid += s["valid"]
            truth_accel = accel(s["truth"], s["frames"])
            for name in ("raw", "ours"):
                pred = s[name]
                sums.setdefault(f"{name}/mpjpe", []).extend(np.linalg.norm(pred - s["truth"], axis=2).mean(axis=1))
                sums.setdefault(f"{name}/pa", []).extend(np.linalg.norm(procrustes(pred, s["truth"]) - s["truth"], axis=2).mean(axis=1))
                error = np.linalg.norm(accel(pred, s["frames"]) - truth_accel, axis=2).mean(axis=1)
                sums.setdefault(f"{name}/accel", []).extend(error)
    table = {key: 1000.0 * float(np.mean(values)) for key, values in sums.items()}
    table["frames"] = covered
    table["coverage"] = covered / max(valid, 1)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequences", type=Path, help="3DPW sequenceFiles/test folder")
    parser.add_argument("runs", nargs="+", help="analysis ids, one per test sequence")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    evaluations = []
    for run in args.runs:
        meta = json.loads((run_dir(run) / "run.json").read_text())
        name = Path(meta["original_name"]).stem
        evaluations.append(evaluate(args.sequences / f"{name}.pkl", run))
        per = score(evaluations[-1:])
        print(f"{name:32s} frames {per['frames']:6d}  MPJPE {per['raw/mpjpe']:6.1f} -> {per['ours/mpjpe']:6.1f}  "
              f"PA {per['raw/pa']:5.1f} -> {per['ours/pa']:5.1f}  Accel {per['raw/accel']:5.1f} -> {per['ours/accel']:5.1f}")
    total = score(evaluations)
    print(json.dumps(total, indent=1))
    if args.json:
        args.json.write_text(json.dumps(total, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
