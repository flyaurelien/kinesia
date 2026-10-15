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

Where people are: PCK_abs (the official script's absolute mode), the pelvis
position error split into depth and lateral parts, and PCOD (SMAP's ordinal
depth between people). How they move, as motion capture is judged: MPJPE and
PA-MPJPE in mm, joint-angle and limb-length errors, limb-length drift,
velocity and acceleration errors, and foot sliding while the true foot is
still. Under occlusion: 3DPCK on the joints MuPoTS labels hidden or visible.

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
        out[i] = joints.from_vertices(verts * np.array([1.0, -1.0, -1.0])) + rows["cam_t"][i]  # MHR axes -> camera
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


def procrustes(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``pred`` (J, 3) moved onto ``truth`` by the best similarity transform."""
    mp, mt = pred.mean(0), truth.mean(0)
    a, b = pred - mp, truth - mt
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    u[:, -1] *= d
    return (s.sum() / (a**2).sum()) * a @ (u @ vt) + mt


# Joint angles, each the angle at the middle joint (MuPoTS joint numbers):
# elbows, knees, shoulders (arm against trunk) and hips (thigh against trunk).
ANGLES = {
    "elbow": [(2, 3, 4), (5, 6, 7)],
    "knee": [(8, 9, 10), (11, 12, 13)],
    "shoulder": [(8, 2, 3), (11, 5, 6)],
    "hip": [(1, 8, 9), (1, 11, 12)],
}
LIMBS = [(2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13)]  # upper arms, forearms, thighs, shanks
ANKLES = [10, 13]


def angles(joints: np.ndarray) -> np.ndarray:
    """(..., 8) joint angles in degrees, in ``ANGLES`` order, left and right."""
    out = []
    for triples in ANGLES.values():
        for a, b, c in triples:
            u, v = joints[..., a, :] - joints[..., b, :], joints[..., c, :] - joints[..., b, :]
            cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)).clip(1e-9)
            out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.stack(out, axis=-1)


def limbs(joints: np.ndarray) -> np.ndarray:
    return np.stack([np.linalg.norm(joints[..., a, :] - joints[..., b, :], axis=-1) for a, b in LIMBS], axis=-1)


def pose_errors(annotations: np.ndarray, occlusion: np.ndarray, predictions: list[list[dict]], dt: float = 1 / 30) -> dict:
    """Every measure of one sequence, people matched as the official script does.

    Per annotated person ("all" protocol): ``errors`` (n, 17), the official
    relative errors (pelvis-centred, true bone lengths, universal skeleton);
    ``absolute`` (n, 17), the same without centring on the pelvis: the bones
    are laid from the predicted pelvis, in camera coordinates (PCK_abs, as the
    official script's absolute mode does); ``matched``; ``hidden``, MuPoTS's
    per-joint occlusion labels.
    Per matched person: ``mpjpe`` and ``pa_mpjpe`` (mm, 14 joints, ``annot3``,
    pelvis-centred, no rescaling), ``root`` (3,) the pelvis position error in
    camera axes (mm). Per pair of matched people in a frame: ``order``
    whether their order in depth is right (PCOD, as SMAP's ``cal_ordinal.m``:
    the same sign, or both depth gaps under 30 cm) and ``spacing`` the error
    on the distance between their pelvises (mm). Per matched person and frame,
    ``angle`` (8,) joint-angle errors (degrees) and ``limb`` (8,) limb-length
    errors (mm). Over each annotated person's run of matched frames (``dt``
    seconds apart): ``accel`` (m/s^2) and ``velocity`` (mm/s) errors of the
    pelvis-centred joints, ``sliding``, how far an ankle moves (mm, per 1/30 s)
    while the true one stays within 1 cm, and ``drift`` the spread (std, mm)
    of each predicted limb's length.
    """
    out: dict[str, list] = {k: [] for k in ("errors", "absolute", "matched", "hidden", "mpjpe", "pa_mpjpe", "root", "order", "spacing",
                                            "angle", "limb", "accel", "velocity", "sliding", "drift")}  # fmt: skip
    runs: dict[int, list] = {}
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
        roots = []
        for k, m in zip(valid, matching):
            placed = np.asarray(annotations[i, k]["univ_annot3"][0, 0]).T  # (17, 3) mm, camera coordinates
            universal = placed - placed[PELVIS]
            metric = np.asarray(annotations[i, k]["annot3"][0, 0]).T  # (17, 3) mm, camera coordinates
            out["hidden"].append(np.asarray(occlusion[i, k]).reshape(17).astype(bool))
            if m < 0:
                undetected += 1
                out["errors"].append(np.full(17, 1e5))
                out["absolute"].append(np.full(17, 1e5))
                out["matched"].append(False)
                continue
            pred = 1000.0 * preds[m]["joints"][H36M_TO_MUPOTS]
            centred = pred - pred[PELVIS]
            out["errors"].append(np.linalg.norm(to_gt_bone_lengths(centred, universal) - universal, axis=1))
            truth = metric - metric[PELVIS]
            out["absolute"].append(np.linalg.norm(pred[PELVIS] + to_gt_bone_lengths(centred, universal) - placed, axis=1))
            out["mpjpe"].append(np.linalg.norm(centred[:SCORED] - truth[:SCORED], axis=1).mean())
            out["pa_mpjpe"].append(np.linalg.norm(procrustes(centred[:SCORED], truth[:SCORED]) - truth[:SCORED], axis=1).mean())
            out["root"].append(pred[PELVIS] - metric[PELVIS])
            out["angle"].append(np.abs(angles(centred) - angles(truth)))
            out["limb"].append(np.abs(limbs(centred) - limbs(truth)))
            out["matched"].append(True)
            roots.append((pred[PELVIS], metric[PELVIS]))
            runs.setdefault(k, []).append((i, centred[:SCORED], truth[:SCORED], pred[ANKLES], metric[ANKLES]))
        for a in range(len(roots)):
            for b in range(a + 1, len(roots)):
                (pa, ta), (pb, tb) = roots[a], roots[b]
                gap, true_gap = pa[2] - pb[2], ta[2] - tb[2]
                out["order"].append(np.sign(gap) == np.sign(true_gap) or (abs(gap) < 300 and abs(true_gap) < 300))
                out["spacing"].append(abs(np.linalg.norm(pa - pb) - np.linalg.norm(ta - tb)))
    per_30 = 1 / (30 * dt)  # displacements over 1/30 s
    for track in runs.values():
        frames = np.array([r[0] for r in track])
        pred, truth = np.array([r[1] for r in track]), np.array([r[2] for r in track])
        feet, true_feet = np.array([r[3] for r in track]), np.array([r[4] for r in track])
        out["drift"].append(limbs(pred).std(axis=0))
        step = np.flatnonzero(np.diff(frames) == 1)
        if len(step):
            out["velocity"].extend(np.linalg.norm((pred[step + 1] - pred[step]) - (truth[step + 1] - truth[step]), axis=2).mean(axis=1) / dt)
            moved = np.linalg.norm(feet[step + 1] - feet[step], axis=2) * per_30
            still = np.linalg.norm(true_feet[step + 1] - true_feet[step], axis=2) * per_30 < 10.0
            out["sliding"].extend(moved[still])
        three = np.flatnonzero((frames[2:] - frames[1:-1] == 1) & (frames[1:-1] - frames[:-2] == 1))
        if len(three):
            a = pred[three + 2] - 2 * pred[three + 1] + pred[three]
            b = truth[three + 2] - 2 * truth[three + 1] + truth[three]
            out["accel"].extend(np.linalg.norm(a - b, axis=2).mean(axis=1) / dt**2 / 1000.0)
    result = {k: np.array(v) for k, v in out.items()}
    for key in ("errors", "absolute", "hidden"):
        result[key] = result[key].reshape(-1, 17)
    result["matched"] = result["matched"].astype(bool)
    result["root"] = result["root"].reshape(-1, 3)
    for key in ("angle", "limb", "drift"):
        result[key] = result[key].reshape(-1, 8)
    result["annotated"], result["undetected"] = annotated, undetected
    return result


def pck_masked(errors: np.ndarray, mask: np.ndarray) -> float:
    """3DPCK over the joints ``mask`` selects (``mpii_evaluate_multiperson_errors_visibility_mask``)."""
    e, m = errors[:, :SCORED], mask[:, :SCORED]
    return 100.0 * float(np.mean(e[m] < PCK_MM)) if m.any() else float("nan")


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
    parser.add_argument("--cache", type=Path, help="folder keeping SAM 3D Body's own estimates between runs of this script")
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
        occlusion = loadmat(args.dataset / f"TS{seq}" / "occlusion.mat")["occlusion_labels"]
        frames = annotations.shape[0]
        source = probe(folder / meta["source"])
        step = int(round(source.fps / meta["video"]["fps"]))
        scale = source.width / meta["video"]["width"]
        _, people = scene_people(folder, joints)
        ours = per_frame(people, step, frames, scale, args.frames)
        our_boxes = boxes_per_frame(people, step, frames, scale, args.frames)
        cached = args.cache / f"{run}.npz" if args.cache else None
        if cached and cached.is_file():
            with np.load(cached) as data:
                raw = {key: data[key] for key in data.files}
        else:
            raw = raw_detections(folder, model, joints)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cached, **raw)
        theirs = raw_per_frame(raw, step, frames, scale)
        gt = tracking_gt[f"TS{seq}"]
        keep = np.ones(frames, dtype=bool)
        if args.frames == "analysed":  # only the annotations of analysed frames
            keep = np.arange(frames) % step == 0
            gt = {k: v for k, v in gt.items() if int(re.search(r"(\d+)", k).group(1)) % step == 0}
        for name, predictions in (("sam3d", theirs), ("kinesia", ours)):
            dt = (step if args.frames == "analysed" else 1) / source.fps
            rows[name][seq] = pose_errors(annotations[keep], occlusion[keep], [p for i, p in enumerate(predictions) if keep[i]], dt)
            rows[name][seq]["people"] = int(annotations.shape[1])
            tracks[name][seq] = (gt, predictions if name == "sam3d" else our_boxes)
        r, k = rows["sam3d"][seq], rows["kinesia"][seq]
        print(f"TS{seq:<3d} {int(annotations.shape[1])} people  "
              f"3DPCK all {pck_auc(r['errors'])[0]:5.1f} -> {pck_auc(k['errors'])[0]:5.1f}  "
              f"matched {pck_auc(r['errors'][r['matched']])[0]:5.1f} -> {pck_auc(k['errors'][k['matched']])[0]:5.1f}  "
              f"missed {r['undetected']}/{r['annotated']} -> {k['undetected']}/{k['annotated']}")  # fmt: skip

    table: dict[str, dict] = {name: summarise(rows[name], tracks[name]) for name in rows}
    table["frames"] = args.frames
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per sequence"} if isinstance(v, dict) else v) for k, v in table.items()}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1))
    return 0


def summarise(rows: dict[int, dict], tracks: dict[int, tuple]) -> dict:
    """One method's table: ``rows`` the ``pose_errors`` of each sequence (with
    ``people``), ``tracks`` each sequence's ``(tracking gt, per-frame boxes)``."""
    seqs = sorted(rows)
    per_seq = {s: {"all": pck_auc(rows[s]["errors"]), "matched": pck_auc(rows[s]["errors"][rows[s]["matched"]])} for s in seqs}

    def mean(kind, index, subset=None):
        chosen = [s for s in seqs if subset is None or rows[s]["people"] == subset]
        return float(np.mean([per_seq[s][kind][index] for s in chosen])) if chosen else float("nan")

    annotated = sum(rows[s]["annotated"] for s in seqs)
    undetected = sum(rows[s]["undetected"] for s in seqs)
    pooled = {key: np.concatenate([rows[s][key] for s in seqs])
              for key in ("mpjpe", "pa_mpjpe", "root", "order", "spacing", "angle", "limb", "accel", "velocity", "sliding", "drift")}  # fmt: skip
    root = pooled["root"]
    return {
        "3DPCK all": mean("all", 0),
        "3DPCK matched": mean("matched", 0),
        "AUC all": mean("all", 1),
        "AUC matched": mean("matched", 1),
        "3DPCK all, 2-person sequences": mean("all", 0, 2),
        "3DPCK all, 3-person sequences": mean("all", 0, 3),
        "3DPCK matched, 2-person sequences": mean("matched", 0, 2),
        "3DPCK matched, 3-person sequences": mean("matched", 0, 3),
        # Joints MuPoTS labels as hidden (by the person or someone else) or not,
        # 3DPCK per sequence then averaged, as for the totals.
        **{f"3DPCK {kind}, {which} joints": float(np.nanmean([
            pck_masked(r["errors"][r["matched"]] if kind == "matched" else r["errors"],
                       (r["hidden"][r["matched"]] if kind == "matched" else r["hidden"]) == (which == "occluded"))
            for r in (rows[s] for s in seqs)]))
           for kind in ("all", "matched") for which in ("occluded", "visible")},  # fmt: skip
        # Where people are, not only how they hold themselves (camera coordinates).
        "PCK_abs all": float(np.mean([pck_auc(rows[s]["absolute"])[0] for s in seqs])),
        "PCK_abs matched": float(np.mean([pck_auc(rows[s]["absolute"][rows[s]["matched"]])[0] for s in seqs])),
        "MPJPE (mm)": float(pooled["mpjpe"].mean()),
        "PA-MPJPE (mm)": float(pooled["pa_mpjpe"].mean()),
        "root error (mm)": float(np.linalg.norm(root, axis=1).mean()),
        "root depth error (mm)": float(np.abs(root[:, 2]).mean()),
        "root lateral error (mm)": float(np.linalg.norm(root[:, :2], axis=1).mean()),
        "root depth bias (mm)": float(root[:, 2].mean()),
        "roots within 25 cm (%)": 100.0 * float(np.mean(np.linalg.norm(root, axis=1) < 250)),
        "PCOD (%)": 100.0 * float(np.nanmean([rows[s]["order"].mean() if len(rows[s]["order"]) else np.nan for s in seqs])),
        "person spacing error (mm)": float(pooled["spacing"].mean()),
        # How the body moves, as motion capture is judged.
        **{f"{joint} angle error (deg)": float(pooled["angle"][:, 2 * n : 2 * n + 2].mean()) for n, joint in enumerate(ANGLES)},
        "joint angle error (deg)": float(pooled["angle"].mean()),
        "limb length error (mm)": float(pooled["limb"].mean()),
        "limb length drift (mm, std over time)": float(pooled["drift"].mean()),
        "velocity error (mm/s)": float(pooled["velocity"].mean()),
        "acceleration error (m/s^2)": float(pooled["accel"].mean()),
        "foot sliding (mm per 1/30 s)": float(pooled["sliding"].mean()),
        "detection recall": 100.0 * (1 - undetected / max(annotated, 1)),
        "sequences": len(seqs),
        "per sequence": {f"TS{s}": {"all": per_seq[s]["all"][0], "matched": per_seq[s]["matched"][0]} for s in seqs},
        "tracking": tracking(tracks),
        "tracking, annotated people only": tracking(tracks, filtered=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
