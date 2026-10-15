"""Score Kinesia on MuPoTS-3D: multi-person 3D pose and tracking, static camera.

MuPoTS-3D (Mehta et al., 3DV 2018) films 2 or 3 people at a time with one
fixed camera, 5 sequences indoors (30 fps) and 15 outdoors (60 fps), with
true 3D joints from markerless multi-view capture. It is scored here with
the two protocols the literature uses on it:

3D pose, the official protocol (``mpii_mupots_multiperson_eval.m``, ported
line for line): each frame's predictions are matched to the annotated people
by their 2D joints (40 px); each matched pose is centred on its pelvis, its
bones are scaled to the true bone lengths, and a joint counts as correct
within 15 cm (3DPCK). "All" counts every annotated person, a missed one
scoring zero, so it also measures detection; "Matched" scores only the
people found. Both are averaged over the 20 sequences, as in the papers.

Tracking, the protocol of T3DP (NeurIPS 2021) and PHALP (CVPR 2022), whose
ground-truth boxes are used as released (``_DATA/mupots_gt.pickle``):
py-motmetrics with IoU matching (``max_iou=0.99``) for ID switches, MOTA and
IDF1 over all sequences, and HOTA (TrackEval) averaged per sequence. Not
everyone in the picture is annotated, so the same metrics are also given
with the tracks of unannotated people left out, as TRACE (CVPR 2023) does.

Joints come from the body mesh, as for the published SMPL-based methods: the
MHR mesh is carried onto the SMPL topology by Meta's barycentric map
(``mhr2smpl_mapping.npz``, from facebookresearch/MHR), then the Human3.6M
regressor (``J_regressor_h36m.npy``, from SPIN) gives the 17 joints.

Two rows are scored on the same detections:

  SAM 3D Body   each frame on its own (``raw/bodies.npz``), with SAM 3.1's
                masklets as tracks
  Kinesia       the people of the built scene: identities, temporal model,
                floor and contact physics

Kinesia analyses at most 30 frames per second, so it sees every other frame
of the 60 fps outdoor sequences. The frames in between are linear
interpolations of their two neighbours (``--frames analysed`` scores only
the analysed frames instead).

Usage:
  uv run --no-sync --with scipy --with motmetrics \\
    --with "trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git" \\
    python scripts/benchmark_mupots.py <MultiPersonTestSet> <mupots_gt.pickle> \\
    <mhr2smpl_mapping.npz> <J_regressor_h36m.npy> <run id>... [--json out.json]

Each run must be an analysis of one sequence's frames as a video named
``mupots-ts<N>.mp4``, made at the sequence's frame rate, with its scene built.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinesia.paths import run_dir  # noqa: E402
from kinesia.scene import motion  # noqa: E402
from kinesia.scene.body_model import BodyModel  # noqa: E402
from kinesia.settings import load_local_env, model_files  # noqa: E402
from kinesia.video import probe  # noqa: E402

# Joints of SPIN's Human3.6M regressor (0 pelvis, 1-3 left leg, 4-6 right leg,
# 7 spine, 8 thorax, 9 nose, 10 head top, 11-13 left arm, 14-16 right arm; see
# SPIN's H36M_TO_J17) in MuPoTS's "relevant" order: head top, neck, R shoulder,
# elbow, wrist, L shoulder, elbow, wrist, R hip, knee, ankle, L hip, knee,
# ankle, pelvis, spine, head.
H36M_TO_MUPOTS = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
PARENTS = np.array([2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]) - 1  # mpii_get_joints o1
TRAVERSAL = np.array([16, 2, 1, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) - 1  # safe order, pelvis first
MATCHING = np.arange(1, 14)  # neck to ankles
PELVIS = 14
SCORED = 14  # head top to ankles
MATCH_PX = 40
PCK_MM = 150
AUC_MM = np.arange(0, 151, 5)
# MHR70 keypoints standing in for the matching joints on raw detections
# (neck = mid-shoulders): shoulders, elbows, wrists, hips, knees, ankles.
MHR_MATCHING = {2: 6, 3: 8, 4: 41, 5: 5, 6: 7, 7: 62, 8: 10, 9: 12, 10: 14, 11: 9, 12: 11, 13: 13}


def unpickle_numpy(path: Path):
    """Load a pickle that may only rebuild numpy arrays and scalars."""
    allowed = {"scalar", "_reconstruct", "dtype", "ndarray"}

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] == "numpy" and name in allowed:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"refusing {module}.{name}")

    with path.open("rb") as handle:
        return Restricted(handle).load()


class Joints:
    """Human3.6M joints of MHR meshes, through the SMPL topology."""

    def __init__(self, model: BodyModel, mapping: Path, regressor: Path):
        with np.load(mapping) as m:
            triangles = model.faces[m["triangle_ids"]]
            bary = m["baryc_coords"]
        h36m = np.load(regressor)  # (17, 6890)
        weights = np.zeros((17, model.faces.max() + 1))
        for corner in range(3):
            np.add.at(weights.T, triangles[:, corner], (h36m * bary[:, corner]).T)
        self.columns = np.flatnonzero(np.abs(weights).sum(axis=0) > 0)
        self.weights = weights[:, self.columns]
        self.model = model

    def from_vertices(self, vertices: np.ndarray) -> np.ndarray:
        return np.einsum("jv,...vc->...jc", self.weights, vertices[..., self.columns, :])

    def from_states(self, states: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """(N, 17, 3) joints of world joint states (N, 127, 8) skinning ``rest``."""
        joints, weights = self.model.skinning
        from kinesia.scene import quat

        transforms = quat.compose(states, self.model.inverse_bind)
        cols = self.columns
        verts = np.zeros((len(states), len(cols), 3))
        for slot in range(4):
            verts += weights[cols, slot][None, :, None] * quat.apply(transforms[:, joints[cols, slot]], rest[cols][None])
        return np.einsum("jv,nvc->njc", self.weights, verts)


# -- predictions ------------------------------------------------------------------


def scene_people(folder: Path, joints: Joints) -> tuple[dict, list[dict]]:
    """Each scene person's frames, camera-frame joints (m), 2D joints, and the
    boxes Kinesia shows them in (its video overlay: who is where, frame by
    frame, including frames whose 3D body comes from the frames around)."""
    scene = json.loads((folder / "scene" / "scene.json").read_text())
    raw = (folder / "scene" / scene["people_file"]).read_bytes()
    enc = scene["encoding"]
    x, y, z, w = scene["camera"]["rotation"]
    camera_to_world = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    position = np.asarray(scene["camera"]["position"], dtype=np.float64)
    focal = float(scene["camera"]["focal"])
    centre = np.array([scene["width"], scene["height"]]) / 2.0
    overlay = json.loads((folder / "scene" / "overlay.json").read_text())["frames"]
    shown: list[dict[int, np.ndarray]] = [{} for _ in scene["people"]]
    for frame, entries in enumerate(overlay):
        for entry in entries:
            shown[entry[0]][frame] = np.asarray(entry[1:5], dtype=np.float64)
    people = []
    for person in scene["people"]:
        layout = person["layout"]

        def section(name):
            d = layout[name]
            return np.frombuffer(raw, dtype=d["type"], count=int(np.prod(d["shape"])), offset=d["offset"]).reshape(d["shape"])

        states = motion._world_joints(
            joints.model,
            section("pelvis").astype(np.float64),
            section("local_t").astype(np.float64) * enc["local_t_unit"],
            section("local_q").astype(np.float64) / enc["quat_scale"],
            section("joint_scales").astype(np.float64),
        )
        world = joints.from_states(states, section("rest_vertices").astype(np.float64))
        camera = (world - position) @ camera_to_world  # world -> camera axes
        people.append({
            "frames": section("frames").astype(int),
            "joints": camera,
            "uv": (focal * camera[..., :2] / camera[..., 2:] + centre)[:, H36M_TO_MUPOTS],
            "boxes": shown[len(people)],
        })  # fmt: skip
    return scene, people


def raw_detections(folder: Path, model: BodyModel, joints: Joints) -> dict:
    """SAM 3D Body's own estimate of every detection, in camera axes."""
    with np.load(folder / "raw" / "bodies.npz") as data:
        rows = {key: data[key] for key in data.files}
    out = np.empty((len(rows["frame"]), 17, 3))
    for i in range(len(out)):
        verts = model.reference_vertices(rows["model_params"][i], rows["shape"][i], correctives=True) / 100.0
        out[i] = joints.from_vertices(verts * np.array([1.0, -1.0, -1.0]))  # MHR axes -> camera axes
    uv = np.zeros((len(out), 17, 2))
    kp2d = rows["kp2d"].astype(np.float64)
    for joint, keypoint in MHR_MATCHING.items():
        uv[:, joint] = kp2d[:, keypoint]
    uv[:, 1] = kp2d[:, [5, 6]].mean(axis=1)
    return {"frame": rows["frame"], "track": rows["track"], "box": rows["box"].astype(np.float64), "joints": out, "uv": uv}


def per_frame(people: list[dict], step: int, frames: int, scale: float, mode: str) -> list[list[dict]]:
    """Poses at every source frame: ``[{id, joints, uv}]``.

    ``step`` source frames per analysed frame; with ``mode == "all"``, the
    source frames in between are interpolated from the analysed neighbours.
    """
    out: list[list[dict]] = [[] for _ in range(frames)]
    for pid, person in enumerate(people):
        index = {f: i for i, f in enumerate(person["frames"])}
        for f, i in index.items():
            source = f * step
            if source < frames:
                out[source].append({"id": pid, "joints": person["joints"][i], "uv": person["uv"][i] * scale})
            if mode != "all" or step == 1 or f + 1 not in index:
                continue
            j = index[f + 1]
            for k in range(1, step):
                if source + k >= frames:
                    break
                a = k / step
                lerp = lambda key: (1 - a) * person[key][i] + a * person[key][j]  # noqa: E731
                out[source + k].append({"id": pid, "joints": lerp("joints"), "uv": lerp("uv") * scale})
    return out


def boxes_per_frame(people: list[dict], step: int, frames: int, scale: float, mode: str) -> list[list[dict]]:
    """Kinesia's boxes at every source frame: ``[{id, box}]``, interpolated
    between analysed frames like the poses."""
    out: list[list[dict]] = [[] for _ in range(frames)]
    for pid, person in enumerate(people):
        for f, box in person["boxes"].items():
            source = f * step
            if source < frames:
                out[source].append({"id": pid, "box": box * scale})
            if mode != "all" or step == 1 or f + 1 not in person["boxes"]:
                continue
            for k in range(1, min(step, frames - source)):
                a = k / step
                out[source + k].append({"id": pid, "box": ((1 - a) * box + a * person["boxes"][f + 1]) * scale})
    return out


def raw_per_frame(raw: dict, step: int, frames: int, scale: float) -> list[list[dict]]:
    out: list[list[dict]] = [[] for _ in range(frames)]
    for i, f in enumerate(raw["frame"]):
        if f * step < frames:
            out[f * step].append({"id": int(raw["track"][i]), "joints": raw["joints"][i], "uv": raw["uv"][i] * scale, "box": raw["box"][i] * scale})
    return out


# -- 3D pose: the official protocol ----------------------------------------------


def identity_matching(gt_2d: list[np.ndarray], pred_2d: list[np.ndarray], pred_visible: list[np.ndarray]) -> list[int]:
    """``mpii_multiperson_get_identity_matching``: greedy, in annotation order."""
    taken = np.zeros(len(pred_2d), dtype=bool)
    matching = []
    for gt in gt_2d:
        scores = np.zeros(len(pred_2d))
        for j, pred in enumerate(pred_2d):
            if taken[j]:
                continue
            close = np.all(np.abs(gt - pred) < MATCH_PX, axis=0)
            scores[j] = np.sum(close & pred_visible[j])
        best = int(np.argmax(scores)) if len(scores) else -1
        if best >= 0 and scores[best] > 0:
            matching.append(best)
            taken[best] = True
        else:
            matching.append(-1)
    return matching


def to_gt_bone_lengths(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``mpii_map_to_gt_bone_lengths`` (joints as rows)."""
    out = pred.copy()
    for idx in TRAVERSAL:
        parent = PARENTS[idx]
        bone = pred[idx] - pred[parent]
        out[idx] = out[parent] + bone * np.linalg.norm(gt[idx] - gt[parent]) / np.linalg.norm(bone).clip(1e-9)
    return out


def pose_errors(annotations: np.ndarray, predictions: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per-joint errors (n, 17) of every annotated person ("all" protocol), a
    flag for those matched to a prediction, and the people counts."""
    errors, matched = [], []
    annotated = undetected = 0
    for i in range(annotations.shape[0]):
        valid = [k for k in range(annotations.shape[1]) if int(np.asarray(annotations[i, k]["isValidFrame"][0, 0]).squeeze())]
        annotated += len(valid)
        if not valid:
            continue
        preds = predictions[i] if i < len(predictions) else []
        gt_2d = [np.asarray(annotations[i, k]["annot2"][0, 0])[:, MATCHING] for k in valid]
        pred_2d = [p["uv"][MATCHING].T for p in preds]
        visible = [~((p[0] == 0) & (p[1] == 0)) for p in pred_2d]
        matching = identity_matching(gt_2d, pred_2d, visible)
        for k, m in zip(valid, matching):
            truth = np.asarray(annotations[i, k]["univ_annot3"][0, 0]).T  # (17, 3) mm
            truth = truth - truth[PELVIS]
            if m < 0:
                undetected += 1
                errors.append(np.full(17, 1e5))
                matched.append(False)
                continue
            pred = 1000.0 * preds[m]["joints"][H36M_TO_MUPOTS]
            pred = to_gt_bone_lengths(pred - pred[PELVIS], truth)
            errors.append(np.linalg.norm(pred - truth, axis=1))
            matched.append(True)
    return np.array(errors).reshape(-1, 17), np.array(matched, dtype=bool), annotated, undetected


def pck_auc(errors: np.ndarray) -> tuple[float, float]:
    e = errors[:, :SCORED]
    if not len(e):
        return float("nan"), float("nan")
    return 100.0 * float(np.mean(e < PCK_MM)), 100.0 * float(np.mean([np.mean(e < t) for t in AUC_MM]))


# -- tracking: the T3DP / PHALP protocol ------------------------------------------


def _xywh(box: np.ndarray) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def annotated_only(gt: dict, predictions: list[list[dict]], mm) -> list[list[dict]]:
    """Leave out the tracks of people MuPoTS does not annotate.

    Not everyone in the picture is annotated (bystanders, a third person
    indoors), and tracking them counts as false positives. As TRACE (CVPR
    2023) does for its baselines, a track is kept only if its box covers an
    annotated person (IoU >= 0.5) in at least half of the annotated frames
    where it appears.
    """
    seen: dict[int, int] = {}
    covering: dict[int, int] = {}
    for key, (_, _, gt_boxes, gt_new) in gt.items():
        frame = int(re.search(r"(\d+)", key).group(1))
        if not len(gt_new) or frame >= len(predictions):
            continue
        preds = [p for p in predictions[frame] if np.all(np.isfinite(p["box"]))]
        if not preds:
            continue
        iou = 1.0 - np.nan_to_num(mm.distances.iou_matrix(np.asarray(gt_boxes, dtype=float).reshape(-1, 4), np.asarray([_xywh(p["box"]) for p in preds], dtype=float).reshape(-1, 4), max_iou=1.0), nan=1.0)
        for j, p in enumerate(preds):
            seen[p["id"]] = seen.get(p["id"], 0) + 1
            covering[p["id"]] = covering.get(p["id"], 0) + int(iou[:, j].max() >= 0.5)
    kept = {pid for pid, n in seen.items() if covering[pid] >= 0.5 * n}
    return [[p for p in frame if p["id"] in kept] for frame in predictions]


def tracking(sequences: dict[str, tuple[dict, list[list[dict]]]], filtered: bool = False) -> dict:
    import motmetrics as mm

    for alias in ("float", "int", "bool"):  # TrackEval still uses the removed numpy aliases
        if not hasattr(np, alias):
            setattr(np, alias, {"float": float, "int": int, "bool": bool}[alias])
    if not hasattr(np, "asfarray"):  # nor does py-motmetrics 1.4 run on NumPy 2 without it
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
    import trackeval

    accumulators, hota, det_a, ass_a = [], [], [], []
    for name, (gt, predictions) in sequences.items():
        if filtered:
            predictions = annotated_only(gt, predictions, mm)
        acc = mm.MOTAccumulator(auto_id=True)
        accumulators.append(acc)
        ids = sorted({p["id"] for frame in predictions for p in frame if np.all(np.isfinite(p["box"]))})
        column = {pid: c for c, pid in enumerate(ids)}
        gt_ids = sorted({g for frame in gt.values() for g in frame[3]})
        gt_col = {g: c for c, g in enumerate(gt_ids)}
        gt_present, pr_present, similarity = [], [], []
        for key in sorted(gt):
            frame = int(re.search(r"(\d+)", key).group(1))
            _, _, gt_boxes, gt_new = gt[key]
            if not len(gt_new):
                continue
            preds = [p for p in (predictions[frame] if frame < len(predictions) else []) if np.all(np.isfinite(p["box"]))]
            boxes = [_xywh(p["box"]) for p in preds]
            pids = [column[p["id"]] for p in preds]
            cost = mm.distances.iou_matrix(np.asarray(gt_boxes, dtype=float).reshape(-1, 4), np.asarray(boxes, dtype=float).reshape(-1, 4), max_iou=0.99)
            acc.update(list(gt_new), pids, cost)
            cost = np.nan_to_num(cost, nan=1.0)
            g = np.zeros(len(gt_ids), dtype=bool)
            p = np.zeros(len(ids), dtype=bool)
            s = np.zeros((len(gt_ids), len(ids)))
            for a, gid in enumerate(gt_new):
                g[gt_col[gid]] = True
                for b, pid in enumerate(pids):
                    p[pid] = True
                    s[gt_col[gid], pid] = 1.0 - cost[a, b]
            gt_present.append(g)
            pr_present.append(p)
            similarity.append(s)
        gp, pp = np.array(gt_present), np.array(pr_present).reshape(len(gt_present), len(ids))
        data = {
            "num_timesteps": len(gp),
            "num_gt_ids": int(np.sum(gp.any(axis=0))),
            "num_tracker_ids": int(np.sum(pp.any(axis=0))),
            "num_gt_dets": int(gp.sum()),
            "num_tracker_dets": int(pp.sum()),
            "gt_ids": [np.flatnonzero(row) for row in gp],
            "tracker_ids": [np.flatnonzero(row) for row in pp],
            "similarity_scores": [s[np.flatnonzero(g)][:, np.flatnonzero(p)] for s, g, p in zip(similarity, gp, pp)],
        }
        result = trackeval.metrics.HOTA().eval_sequence(data)
        hota.append(float(np.mean(result["HOTA"])))
        det_a.append(float(np.mean(result["DetA"])))
        ass_a.append(float(np.mean(result["AssA"])))
    mh = mm.metrics.create()
    summary = mh.compute_many(accumulators, metrics=["num_switches", "mota", "idf1", "precision", "recall"], generate_overall=True)
    overall = summary.loc["OVERALL"]
    return {
        "IDs": int(overall["num_switches"]),
        "MOTA": 100.0 * float(overall["mota"]),
        "IDF1": 100.0 * float(overall["idf1"]),
        "HOTA": 100.0 * float(np.mean(hota)),
        "DetA": 100.0 * float(np.mean(det_a)),
        "AssA": 100.0 * float(np.mean(ass_a)),
        "precision": 100.0 * float(overall["precision"]),
        "recall": 100.0 * float(overall["recall"]),
    }


# -- driver -----------------------------------------------------------------------


def main() -> int:
    from scipy.io import loadmat

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="MultiPersonTestSet folder (TS1..TS20 with annot.mat)")
    parser.add_argument("tracking_gt", type=Path, help="T3DP's _DATA/mupots_gt.pickle")
    parser.add_argument("mapping", type=Path, help="MHR's tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz")
    parser.add_argument("regressor", type=Path, help="SPIN's data/J_regressor_h36m.npy")
    parser.add_argument("runs", nargs="+", help="analysis ids, one per sequence")
    parser.add_argument("--frames", choices=("all", "analysed"), default="all")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    load_local_env()
    model = BodyModel(model_files()["SAM3D_BODY_DIR"])
    joints = Joints(model, args.mapping, args.regressor)
    tracking_gt = unpickle_numpy(args.tracking_gt)
    rows: dict[str, dict] = {"sam3d": {}, "kinesia": {}}
    tracks: dict[str, dict] = {"sam3d": {}, "kinesia": {}}
    for run in args.runs:
        folder = run_dir(run)
        meta = json.loads((folder / "run.json").read_text())
        seq = int(re.search(r"ts(\d+)", meta["original_name"].lower()).group(1))
        annotations = loadmat(args.dataset / f"TS{seq}" / "annot.mat")["annotations"]
        frames = annotations.shape[0]
        source = probe(folder / meta["source"])
        step = int(round(source.fps / meta["video"]["fps"]))
        scale = source.width / meta["video"]["width"]
        _, people = scene_people(folder, joints)
        ours = per_frame(people, step, frames, scale, args.frames)
        our_boxes = boxes_per_frame(people, step, frames, scale, args.frames)
        theirs = raw_per_frame(raw_detections(folder, model, joints), step, frames, scale)
        gt = tracking_gt[f"TS{seq}"]
        keep = np.ones(frames, dtype=bool)
        if args.frames == "analysed":  # only the annotations of analysed frames
            keep = np.arange(frames) % step == 0
            gt = {k: v for k, v in gt.items() if int(re.search(r"(\d+)", k).group(1)) % step == 0}
        for name, predictions in (("sam3d", theirs), ("kinesia", ours)):
            errors, matched, annotated, undetected = pose_errors(annotations[keep], [p for i, p in enumerate(predictions) if keep[i]])
            rows[name][seq] = {"errors": errors, "matched": matched, "annotated": annotated, "undetected": undetected,
                               "people": int(annotations.shape[1])}  # fmt: skip
            tracks[name][seq] = (gt, predictions if name == "sam3d" else our_boxes)
        r, k = rows["sam3d"][seq], rows["kinesia"][seq]
        print(f"TS{seq:<3d} {int(annotations.shape[1])} people  "
              f"3DPCK all {pck_auc(r['errors'])[0]:5.1f} -> {pck_auc(k['errors'])[0]:5.1f}  "
              f"matched {pck_auc(r['errors'][r['matched']])[0]:5.1f} -> {pck_auc(k['errors'][k['matched']])[0]:5.1f}  "
              f"missed {r['undetected']}/{r['annotated']} -> {k['undetected']}/{k['annotated']}")  # fmt: skip

    table: dict[str, dict] = {}
    for name in rows:
        seqs = sorted(rows[name])
        per_seq = {s: {"all": pck_auc(rows[name][s]["errors"]), "matched": pck_auc(rows[name][s]["errors"][rows[name][s]["matched"]])} for s in seqs}

        def mean(kind, index, subset=None):
            chosen = [s for s in seqs if subset is None or rows[name][s]["people"] == subset]
            return float(np.mean([per_seq[s][kind][index] for s in chosen])) if chosen else float("nan")

        annotated = sum(rows[name][s]["annotated"] for s in seqs)
        undetected = sum(rows[name][s]["undetected"] for s in seqs)
        table[name] = {
            "3DPCK all": mean("all", 0),
            "3DPCK matched": mean("matched", 0),
            "AUC all": mean("all", 1),
            "AUC matched": mean("matched", 1),
            "3DPCK all, 2-person sequences": mean("all", 0, 2),
            "3DPCK all, 3-person sequences": mean("all", 0, 3),
            "3DPCK matched, 2-person sequences": mean("matched", 0, 2),
            "3DPCK matched, 3-person sequences": mean("matched", 0, 3),
            "detection recall": 100.0 * (1 - undetected / max(annotated, 1)),
            "sequences": len(seqs),
            "per sequence": {f"TS{s}": {"all": per_seq[s]["all"][0], "matched": per_seq[s]["matched"][0]} for s in seqs},
            "tracking": tracking(tracks[name]),
            "tracking, annotated people only": tracking(tracks[name], filtered=True),
        }
    table["frames"] = args.frames
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per sequence"} if isinstance(v, dict) else v) for k, v in table.items()}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
