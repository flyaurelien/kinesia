"""Score SAM-Body4D on MuPoTS-3D with Kinesia's evaluator.

SAM-Body4D (Gao, Miao and Han, arXiv 2512.08406, github.com/gaomingqi/sam-body4d)
is the closest open-source relative of Kinesia: SAM 3 masklets, SAM 3D Body on
each masklet, then temporal smoothing of the body parameters (a Kalman filter on
pose, one body shape per person, smoothed global rotation) and, optionally, an
occlusion refiner (Diffusion-VAS amodal masks and content completion). It
publishes no quantitative results.

Its automatic mode (``scripts/offline_app.py``) is run unchanged: ViTDet finds
the people in the first frame (from frame 10) where it finds anyone, their boxes
prompt SAM 3, SAM 3 carries the masks through the clip, and SAM 3D Body
reconstructs every mask with SAM-Body4D's smoothing. Only its outputs are
captured differently: instead of a PLY mesh and a rendered picture per person
and frame, each person-frame's MHR parameters go to one ``bodies.npz`` with the
fields of Kinesia's own ``raw/bodies.npz``.

``--sam31`` puts SAM 3.1 (Object Multiplex, the checkpoint Kinesia uses) in
place of SAM 3, whose checkpoint is gated: SAM 3.1's tracker alone, prompted
with the same boxes. Its Diffusion-VAS refiner (``completion.enable`` in the
config) crashes when an occlusion reaches the last frame of one of its 64-frame
batches; it was run with ``end = min(<batch length>, end + 2)`` in place of
``min(<batch length> - 1, end + 2)`` in ``offline_app.py`` (commit 21af102).

Two steps:

  run    inside SAM-Body4D's environment (GPU), one sequence:
           python benchmark_mupots_sambody4d.py run <sam-body4d repo> <config.yaml> \\
             <mupots-tsN.mp4> <out folder> [--sam31 <sam3 checkout> <sam3.1_multiplex.pt>]
  score  here, all sequences, with ``scripts/benchmark_mupots.py``'s protocols:
           uv run --no-sync --with scipy --with motmetrics \\
             --with "trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git" \\
             python scripts/benchmark_mupots_sambody4d.py score <MultiPersonTestSet> \\
             <mupots_gt.pickle> <mhr2smpl_mapping.npz> <J_regressor_h36m.npy> <folder>... \\
             [--frames all|analysed] [--json out.json]

Each ``score`` folder holds the ``bodies.npz`` of one sequence (``run``'s
output). Joints are taken from the mesh exactly as for SAM 3D Body and Kinesia
(MHR mesh -> SMPL topology -> Human3.6M regressor); the matching joints come
from SAM-Body4D's own 2D keypoints; the tracking boxes are those of its masks.
SAM-Body4D reconstructs every frame, so nothing is interpolated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

FIELDS = ("frame", "track", "box", "model_params", "shape", "scale", "cam_t", "kp3d", "kp2d", "focal")


# -- run: SAM-Body4D's automatic mode, outputs captured ----------------------------


def run(repo: Path, config: Path, video: Path, out: Path) -> None:
    import importlib.util

    import types

    import cv2

    try:
        import pyrender  # noqa: F401
    except ImportError:  # no OpenGL on the GPU nodes; nothing is rendered here anyway
        stub = types.ModuleType("pyrender")
        def placeholder(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

        stub.__getattr__ = placeholder
        sys.modules["pyrender"] = stub
    sys.path[:0] = [str(repo), str(repo / "models" / "sam_3d_body"), str(repo / "models" / "diffusion_vas")]
    spec = importlib.util.spec_from_file_location("offline_app", repo / "scripts" / "offline_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    rows: list[dict] = []
    checks: list[np.ndarray] = []  # a few meshes as SAM-Body4D made them, to check the rebuild
    timing: dict[str, float] = {}

    def record(outputs, faces, save_dir, focal_dir, image_path, id_current):
        if outputs is None:
            return
        frame = int(Path(image_path).stem)
        for pid, person in enumerate(outputs):
            rows.append({
                "frame": frame,
                "track": int(id_current[pid]),
                "box": np.asarray(person["bbox"], dtype=np.float32).reshape(4),
                "model_params": person["mhr_model_params"],
                "shape": person["shape_params"],
                "scale": person["scale_params"],
                "cam_t": person["pred_cam_t"],
                "kp3d": person["pred_keypoints_3d"],
                "kp2d": person["pred_keypoints_2d"],
                "focal": np.float32(person["focal_length"]),
            })  # fmt: skip
            if len(checks) < 8:
                checks.append(np.asarray(person["pred_vertices"], dtype=np.float32))

    # Its outputs, not its pictures: no rendering, no PLY, no video.
    app.save_mesh_results = record
    app.visualize_sample_together = lambda img, *a, **k: np.zeros((2, 2, 3), np.uint8)
    app.visualize_sample = lambda *a, **k: []
    app.jpg_folder_to_mp4 = lambda *a, **k: None
    app.OfflineApp.__init__.__defaults__ = (str(config),)
    for stage in ("on_mask_generation", "on_4d_generation"):
        original = getattr(app.OfflineApp, stage)

        def timed(self, *a, _original=original, _stage=stage, **k):
            start = time.time()
            result = _original(self, *a, **k)
            timing[_stage] = time.time() - start
            timing["people"] = len(self.RUNTIME["out_obj_ids"])
            return result

        setattr(app.OfflineApp, stage, timed)

    work = out / "work"
    start = time.time()
    app.inference(argparse.Namespace(input_video=str(video), output_dir=str(work)))
    timing["total"] = time.time() - start

    capture = cv2.VideoCapture(str(video))
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    import torch

    arrays = {key: np.stack([np.asarray(r[key]) for r in rows]) for key in FIELDS}
    np.savez_compressed(
        out / "bodies.npz", **arrays, check_vertices=np.stack(checks), size=np.array([width, height]), frames=frames
    )
    (out / "run.json").write_text(json.dumps({
        "video": str(video),
        "config": Path(config).read_text(),
        "gpu": torch.cuda.get_device_name(0),
        "seconds": timing,
        "rows": len(rows),
    }, indent=1))  # fmt: skip
    print(f"{len(rows)} person-frames, {timing}")


# -- SAM 3.1 in place of SAM 3 (``run --sam31``) --------------------------------------


class Sam31Tracker:
    """SAM 3's tracker interface as SAM-Body4D calls it, on SAM 3.1."""

    def __init__(self, checkpoint: str):
        import torch

        from sam3.model_builder import build_sam3_multiplex_video_model

        # The released checkpoint holds the whole predictor: the tracker under
        # ``tracker.model.`` and the shared image encoder under ``detector.backbone.``.
        self.model = build_sam3_multiplex_video_model(checkpoint_path=None, load_from_HF=False, device="cpu")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        ckpt = ckpt.get("model", ckpt)
        state = {k[len("tracker.model."):]: v for k, v in ckpt.items() if k.startswith("tracker.model.")}
        state |= {k[len("detector."):]: v for k, v in ckpt.items() if k.startswith("detector.backbone.vision_backbone.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"SAM 3.1 tracker: {len(state)} tensors loaded, missing {missing}, unexpected {unexpected}")
        self.model.to("cuda").eval()
        # The tracker alone has no use for the detector's image features (and
        # this revision cannot carry them through ``forward_image``).
        forward_image = self.model.forward_image
        self.model.forward_image = lambda img, need_sam3_out=False, **kw: forward_image(img, need_sam3_out=False, **kw)

    def init_state(self, video_path):
        import sam3.model.video_tracking_multiplex_demo as demo
        import torch
        from sam3.model.io_utils import load_video_frames
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

        # This revision's demo passes loader options its loader no longer takes, and
        # its OpenCV loader normalises 0-255 pixels as if they were 0-1: scale the
        # mean and std instead, so frames come out in [-1, 1] as SAM 3's do.
        half = (127.5, 127.5, 127.5)
        demo.load_video_frames = lambda use_torchcodec=False, use_cv2=False, **kw: load_video_frames(**kw, img_mean=half, img_std=half)

        with torch.inference_mode():
            return VideoTrackingMultiplexDemo.init_state(self.model, video_path=video_path, offload_video_to_cpu=False, offload_state_to_cpu=False)

    def clear_all_points_in_video(self, state):
        import torch

        with torch.inference_mode():
            return self.model.clear_all_points_in_video(state)

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        import torch

        points = torch.as_tensor(box, dtype=torch.float32).reshape(2, 2)
        labels = torch.tensor([2, 3], dtype=torch.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):  # as SAM 3.1's predictor runs
            frame, _, low, high = self.model.add_new_points(inference_state, frame_idx, obj_id, points, labels, clear_old_points=True, rel_coordinates=True)
        return frame, list(inference_state["obj_ids"]), low, high  # SAM 3 returns every object so far

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse, propagate_preflight=True):
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.propagate_in_video_preflight(state)
            for out in self.model.propagate_in_video(state, start_frame_idx, max_frame_num_to_track, reverse):
                yield *out[:4], None, None


def use_sam31(source: Path, checkpoint: Path) -> None:
    """Make SAM-Body4D build SAM 3.1 where it builds SAM 3.

    It builds SAM 3 with ``from models.sam3.sam3.model_builder import
    build_sam3_video_model`` and keeps ``.tracker``; that module is replaced by
    one handing it SAM 3.1's tracker, prompted with the same boxes (as SAM's two
    box corners, labels 2 and 3).
    """
    import types

    sys.path.insert(0, str(source))
    builder = types.ModuleType("models.sam3.sam3.model_builder")
    builder.build_sam3_video_model = lambda checkpoint_path=None: types.SimpleNamespace(
        tracker=Sam31Tracker(str(checkpoint)), detector=types.SimpleNamespace(backbone=None)
    )
    for name in ("models.sam3", "models.sam3.sam3"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["models.sam3.sam3.model_builder"] = builder


# -- score: Kinesia's evaluator on SAM-Body4D's outputs ----------------------------


def predictions(folder: Path, model, joints, frames: int) -> list[list[dict]]:
    """Per-frame ``[{id, joints, uv, box}]`` as ``benchmark_mupots`` builds them
    (the videos are the frames at their own size: pixels are MuPoTS's)."""
    from benchmark_mupots import BODY_KEYPOINTS, MHR_MATCHING, body_box

    with np.load(folder / "bodies.npz") as data:
        rows = {key: data[key] for key in data.files}
    n = len(rows["frame"])
    out3d = np.empty((n, 17, 3))
    for i in range(n):
        verts = model.reference_vertices(rows["model_params"][i], rows["shape"][i], correctives=True) / 100.0
        verts = verts * np.array([1.0, -1.0, -1.0])  # MHR axes -> camera
        if i < len(rows["check_vertices"]):
            gap = np.abs(verts - rows["check_vertices"][i]).max()
            assert gap < 2e-3, f"{folder.name}: rebuilt mesh {gap * 1000:.1f} mm off SAM-Body4D's"
        out3d[i] = joints.from_vertices(verts) + rows["cam_t"][i]
    kp2d = rows["kp2d"].astype(np.float64)
    uv = np.zeros((n, 17, 2))
    for joint, keypoint in MHR_MATCHING.items():
        uv[:, joint] = kp2d[:, keypoint]
    uv[:, 1] = kp2d[:, [5, 6]].mean(axis=1)
    out: list[list[dict]] = [[] for _ in range(frames)]
    for i, f in enumerate(rows["frame"]):
        if f < frames:
            out[f].append({"id": int(rows["track"][i]), "joints": out3d[i], "uv": uv[i], "box": rows["box"][i].astype(np.float64),
                           "body_box": body_box(kp2d[i, BODY_KEYPOINTS])})  # fmt: skip
    return out


def score(args) -> int:
    from scipy.io import loadmat

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_mupots import Joints, pck_auc, pose_errors, summarise, unpickle_numpy, with_boxes

    from kinesia.settings import load_local_env, model_files
    from kinesia.scene.body_model import BodyModel

    load_local_env()
    model = BodyModel(model_files()["SAM3D_BODY_DIR"])
    joints = Joints(model, args.mapping, args.regressor)
    tracking_gt = unpickle_numpy(args.tracking_gt)
    rows: dict[int, dict] = {}
    tracks: dict[int, tuple] = {}
    seconds = frames_done = 0.0
    for folder in args.folders:
        meta = json.loads((folder / "run.json").read_text())
        seq = int(re.search(r"ts(\d+)", Path(meta["video"]).name.lower()).group(1))
        annotations = loadmat(args.dataset / f"TS{seq}" / "annot.mat")["annotations"]
        occlusion = loadmat(args.dataset / f"TS{seq}" / "occlusion.mat")["occlusion_labels"]
        frames = annotations.shape[0]
        fps = 30 if seq <= 5 else 60
        preds = predictions(folder, model, joints, frames)
        gt = tracking_gt[f"TS{seq}"]
        step = 1
        if args.frames == "analysed":  # only the frames Kinesia analyses (at most 30 per second)
            step = fps // 30
            preds = [p if i % step == 0 else [] for i, p in enumerate(preds)]
            gt = {k: v for k, v in gt.items() if int(re.search(r"(\d+)", k).group(1)) % step == 0}
        keep = np.arange(frames) % step == 0
        rows[seq] = pose_errors(annotations[keep], occlusion[keep], [p for i, p in enumerate(preds) if keep[i]], step / fps)
        rows[seq]["people"] = int(annotations.shape[1])
        tracks[seq] = (gt, with_boxes(preds, "body_box") if args.boxes == "body" else preds)
        seconds += meta["seconds"]["total"]
        frames_done += frames
        r = rows[seq]
        print(f"TS{seq:<3d} {r['people']} people  3DPCK all {pck_auc(r['errors'])[0]:5.1f}  "
              f"matched {pck_auc(r['errors'][r['matched']])[0]:5.1f}  missed {r['undetected']}/{r['annotated']}  "
              f"{meta['seconds'].get('people')} prompted, {meta['seconds']['total']:.0f} s on {meta['gpu']}")  # fmt: skip
    table = {"sambody4d": summarise(rows, tracks), "frames": args.frames, "seconds per frame": seconds / max(frames_done, 1)}
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per sequence"} if isinstance(v, dict) else v) for k, v in table.items()}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run SAM-Body4D's automatic mode on one sequence (its own environment, GPU)")
    r.add_argument("repo", type=Path)
    r.add_argument("config", type=Path, help="SAM-Body4D's configs/body4d.yaml with the checkpoint paths filled in")
    r.add_argument("video", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--sam31", type=Path, nargs=2, metavar=("SOURCE", "CHECKPOINT"),
                   help="SAM 3.1 in place of SAM 3: a facebookresearch/sam3 checkout and sam3.1_multiplex.pt")
    s = sub.add_parser("score", help="score the runs with benchmark_mupots.py's protocols")
    s.add_argument("dataset", type=Path, help="MultiPersonTestSet folder (TS1..TS20 with annot.mat)")
    s.add_argument("tracking_gt", type=Path, help="T3DP's _DATA/mupots_gt.pickle")
    s.add_argument("mapping", type=Path, help="MHR's mhr2smpl_mapping.npz")
    s.add_argument("regressor", type=Path, help="SPIN's J_regressor_h36m.npy")
    s.add_argument("folders", type=Path, nargs="+", help="run outputs, one per sequence")
    s.add_argument("--frames", choices=("all", "analysed"), default="all",
                   help="analysed: only the frames Kinesia analyses, as benchmark_mupots.py --frames analysed")
    s.add_argument("--json", type=Path)
    s.add_argument("--boxes", choices=("body", "mask"), default="body", help="tracking boxes: around the body's 2D keypoints (default) or its mask")
    args = parser.parse_args()
    if args.command == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        if args.sam31:
            use_sam31(*(path.resolve() for path in args.sam31))
        run(args.repo.resolve(), args.config.resolve(), args.video.resolve(), args.out.resolve())
        return 0
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
