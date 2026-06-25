"""Detection-only streaming preview: locate every person in a whole video with
SAM3 (run natively on Apple Silicon via the MLX port) and track identities with
a strong learned-embedding re-ID, writing results to disk *as each frame is
processed* so a UI can play an annotated preview that fills in progressively and
be stopped early.

Run as a module (the web layer spawns it):

    python -m sam_3d_pose_estimation.detect_stream \
        --video-input <video> --out-dir <scratch> [--prompt person] [--stride 5]

It writes three files under ``--out-dir``:
  - ``progress.json``  {status, pid, processed, total_to_process, total_frames,
                        last_frame, video_width, video_height, fps, stride}
  - ``frames.jsonl``   one line per *processed* frame:
                        {"f": idx, "t": sec, "dets": [{"id", "b":[x,y,w,h] norm, "s"}]}
  - ``tracks.json``    {"tracks": [{id, color, firstFrame, lastFrame, frameCount, repFrame}]}

Re-identification uses SAM3's own per-detection decoder embedding (256-d, cosine
similarity) — far more discriminative than colour, so a person who leaves and
returns keeps their track id instead of splitting into person 1/3/5.

This is a SEPARATE process from the torch run pipeline: it imports MLX + the
vendored ``sam3`` package only, so it stays light and the run's "detection is
strictly SAM3" policy is untouched (it is still SAM3 — just the MLX build).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# The vendored MLX SAM3 package (mirror of github.com/Deekshith-Dade/mlx_sam3).
# Also placed on PYTHONPATH by the spawner; add here too for direct invocation.
_VENDOR_MLX_SAM3 = Path(__file__).resolve().parents[2] / "vendor" / "mlx_sam3"
if _VENDOR_MLX_SAM3.is_dir() and str(_VENDOR_MLX_SAM3) not in sys.path:
    sys.path.insert(0, str(_VENDOR_MLX_SAM3))


# ── Geometry + embedding helpers ─────────────────────────────────────────────

def _clip_xyxy(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    x1 = min(max(x1, 0.0), width - 1.0)
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), height - 1.0)
    y2 = min(max(y2, 0.0), float(height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b[:4]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes (0 when disjoint)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 1e-6 else 0.0


# Distinct, high-contrast track colours (cycled by track index).
_PALETTE = [
    "#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa",
    "#fb7185", "#4ade80", "#38bdf8", "#facc15", "#c084fc",
]

# ── Tracking: PURE-APPEARANCE global gallery matching ────────────────────────
# Identity is decided ONLY by appearance, NEVER by position — so any camera angle,
# any cut, any re-arrangement of the people works. Each identity accumulates a
# GALLERY of ALL its raw 256-d SAM3 embeddings over the whole history (no average,
# no cap — RAM is cheap and detail matters). At frame t every detection is scored
# against EVERY identity's full gallery (the best-K matching exemplars) and matched
# globally (Hungarian). Because identity is recomputed from scratch every frame
# against the entire clean history, a handful of bad frames can never poison the
# rest of the video, and a person who left / changed pose / changed angle is still
# recognised the instant any of their past views matches again. A detection that
# overlaps another (a crossing) is matched but NOT banked — its embedding is
# contaminated by the other person, so it must never enter the clean gallery.

_MATCH_COS = 0.84        # accept a detection↔identity match at/above this top-K sim;
                         # below it the detection is a NEW identity (mergeable split,
                         # never a guessed swap). Same person ~0.86-0.99 (full gallery
                         # almost always finds a close past view) vs different ~0.72-0.83.
_BANK_MARGIN = 0.05      # a view is MEMORISED only if its identity clearly beats every
                         # other (margin >= this). An ambiguous detection (a crossing /
                         # occlusion blend — the swap source) is still displayed but NOT
                         # banked, so a real gallery never ingests the other person and
                         # the top-K match stays clean. Attachment itself is unchanged,
                         # so this never fragments. Measured: clean same-person margins
                         # ~0.13; contaminating/blend views live below ~0.05 (down to <0).
_GALLERY_TOPK = 5        # appearance = mean of the top-K best-matching gallery views
_DEDUP_COS = 0.992       # skip only a near-identical view (keeps every real pose/angle)
_N_INIT = 2              # detections before an identity is confirmed (surfaced)
_MIN_SURFACE_FRAMES = 2
# Motion continuity: a light position term added to the appearance assignment so
# that at a crossing — where blended embeddings match BOTH galleries equally and
# pure-appearance Hungarian can flip — each identity stays with the detection
# nearest its last box instead of teleporting onto the other person. It only
# breaks ties (appearance still must clear _MATCH_COS) and is disabled for stale
# tracks, so re-identification after a real absence remains appearance-only.
_MOTION_WEIGHT = 0.8
_MOTION_RECENCY_FRAMES = 30
_VEL_REF = 4.0           # px/frame at which an identity counts as "clearly moving";
                         # only moving identities get the motion override, so a swap
                         # during a walking crossing is fixed while a stationary
                         # close-encounter stays pure-appearance (no frame loss)


class _Track:
    """One identity = an unbounded GALLERY of all its raw unit embeddings (every
    distinct view it has ever shown), plus bookkeeping. Identity is decided by
    appearance; a last box + constant-velocity estimate are kept ONLY as a motion
    tie-break so two people can't swap identities at a crossing."""

    __slots__ = (
        "id", "color", "box", "vel", "gallery", "_gmat", "hits", "confirmed",
        "first_frame", "last_frame", "frame_count", "rep_frame", "rep_area",
    )

    def __init__(self, tid: int, frame_idx: int, box_xyxy: np.ndarray, feat):
        self.id = tid
        self.color = _PALETTE[tid % len(_PALETTE)]
        self.box = _xyxy_to_cxcywh(box_xyxy)
        self.vel = np.zeros(2, dtype=np.float32)  # per-frame centre velocity (cx, cy)
        # Gallery of RAW unit embeddings — every view, un-averaged, uncapped.
        self.gallery: list[np.ndarray] = []
        self._gmat: np.ndarray | None = None  # cached stack of `gallery` for matching
        if feat is not None:
            self.gallery.append(_l2norm(feat))
        self.hits = 1
        self.confirmed = False
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.frame_count = 1
        self.rep_frame = frame_idx
        self.rep_area = self._area()

    def _area(self) -> float:
        return float(max(0.0, self.box[2]) * max(0.0, self.box[3]))

    def _add(self, e: np.ndarray) -> None:
        # Keep every genuinely-distinct view; skip only a near-identical duplicate.
        if self.gallery:
            sims = self._matrix() @ e
            if sims.size and float(np.max(sims)) >= _DEDUP_COS:
                return
        self.gallery.append(e)
        self._gmat = None  # invalidate cache

    def _matrix(self) -> np.ndarray:
        if self._gmat is None:
            self._gmat = np.stack(self.gallery) if self.gallery else np.zeros((0, 256), np.float32)
        return self._gmat

    def appearance(self, e_unit) -> float:
        """Similarity to this identity = mean of the top-K best-matching gallery
        views (compare the new vector against the WHOLE history, take the K closest).
        `e_unit` must already be L2-normalised."""
        if e_unit is None or not self.gallery:
            return 0.0
        sims = self._matrix() @ e_unit               # cosine vs every stored view
        if sims.shape[0] <= _GALLERY_TOPK:
            return float(sims.mean())
        topk = np.partition(sims, -_GALLERY_TOPK)[-_GALLERY_TOPK:]
        return float(topk.mean())

    def predict(self, frame_idx: int) -> np.ndarray:
        """Constant-velocity prediction of this identity's box at ``frame_idx`` (xyxy).

        At a crossing the two people physically overlap, so matching to the *last*
        box can't separate them; extrapolating each track by its velocity does —
        the one moving left and the one moving right have distinct predicted boxes.
        """
        steps = max(0, int(frame_idx) - int(self.last_frame))
        cx = float(self.box[0]) + float(self.vel[0]) * steps
        cy = float(self.box[1]) + float(self.vel[1]) * steps
        w, h = float(self.box[2]), float(self.box[3])
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def update(self, frame_idx: int, box_xyxy: np.ndarray, e_unit) -> None:
        new_box = _xyxy_to_cxcywh(box_xyxy)
        # EMA the per-frame centre velocity from the observed displacement.
        step = max(1, int(frame_idx) - int(self.last_frame))
        inst_vel = (new_box[:2] - self.box[:2]) / float(step)
        self.vel = inst_vel.astype(np.float32) if self.frame_count <= 1 else (0.6 * self.vel + 0.4 * inst_vel).astype(np.float32)
        self.box = new_box
        if e_unit is not None:                        # bank every view (measured: a
            self._add(e_unit)                         # few degraded ones dilute, even
                                                      # enrich, and never erode the margin)
        self.hits += 1
        if self.hits >= _N_INIT:
            self.confirmed = True
        self.last_frame = frame_idx
        self.frame_count += 1
        a = self._area()
        if a > self.rep_area:
            self.rep_area = a
            self.rep_frame = frame_idx


class TrackManager:
    def __init__(
        self,
        min_surface_frames: int = _MIN_SURFACE_FRAMES,
        motion_weight: float = _MOTION_WEIGHT,
        motion_recency: int = _MOTION_RECENCY_FRAMES,
    ):
        self.tracks: list[_Track] = []
        self._next_id = 0
        # An identity must be present for at least this many detection-points to
        # surface (computed from a duration-in-seconds so it scales with fps/stride).
        self.min_surface_frames = max(1, int(min_surface_frames))
        # Motion-continuity tie-break (see constants); instance-level so it can be
        # tuned offline by replaying cached detections without re-running SAM3.
        self.motion_weight = float(motion_weight)
        self.motion_recency = int(motion_recency)

    def _new_track(self, frame_idx: int, box: np.ndarray, e_unit) -> _Track:
        t = _Track(self._next_id, frame_idx, box, e_unit)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def update(self, frame_idx: int, dets: list[dict]) -> list[tuple[int, np.ndarray]]:
        """Assign identities to this frame's detections — PURE APPEARANCE, no position.

        Every detection is scored against EVERY existing identity's full gallery
        (top-K best-matching views), then matched globally (Hungarian) so two people
        in the same frame always land on two different identities. A pair is accepted
        only if the appearance similarity reaches ``_MATCH_COS``; otherwise the
        detection starts a NEW identity (a mergeable split — never a guessed swap).
        Because each frame is decided from scratch against the whole history (the only
        state) and similarity is the mean of the top-K best gallery views, a handful of
        degraded frames can't poison anything — measured: banking every view (occluded
        / fade included) leaves the galleries unimodal and the margin unchanged."""
        assigned: dict[int, int] = {}
        unmatched_d = set(range(len(dets)))

        # Pre-normalise detection embeddings once.
        embs = [
            (_l2norm(d["emb"]) if d.get("emb") is not None else None)
            for d in dets
        ]

        def attach(t: _Track, j: int, bankable: bool) -> None:
            # bankable=False → the box/id are assigned and displayed, but the (blended,
            # ambiguous) embedding is NOT memorised, so the gallery stays clean.
            t.update(frame_idx, dets[j]["bbox"], embs[j] if bankable else None)
            assigned[j] = t.id
            unmatched_d.discard(j)

        # GLOBAL appearance assignment over ALL known identities × this frame's dets.
        ids = list(self.tracks)
        if ids and dets:
            sim = np.zeros((len(ids), len(dets)), dtype=np.float32)
            for i, t in enumerate(ids):
                for j in range(len(dets)):
                    sim[i, j] = t.appearance(embs[j]) if embs[j] is not None else 0.0
            # Decouple two decisions so motion fixes crossings WITHOUT fragmenting:
            #  (1) whether a detection is a NEW identity — appearance-only (its best
            #      gallery match must clear _MATCH_COS), exactly as before; motion
            #      never spawns identities.
            #  (2) WHICH known identity a matchable detection attaches to — appearance
            #      + a light motion-continuity term, so at a crossing (blended
            #      embeddings that match both galleries) each identity stays with the
            #      detection nearest its last box instead of teleporting.
            matchable = [j for j in range(len(dets)) if float(sim[:, j].max()) >= _MATCH_COS]
            if matchable:
                # Motion override is gated by VELOCITY CONFIDENCE: only a clearly-
                # moving identity gets it. At a walking crossing the blended embedding
                # can confidently match the WRONG gallery, but the two walkers' velocity
                # predictions are well-separated, so motion corrects it; a near-stationary
                # close-encounter (unreliable prediction) stays pure-appearance, so
                # identity continuity is never perturbed there (no frame loss).
                cost = np.zeros((len(ids), len(matchable)), dtype=np.float32)
                for i, t in enumerate(ids):
                    recent = (frame_idx - t.last_frame) <= self.motion_recency
                    vmag = float(np.hypot(t.vel[0], t.vel[1]))
                    vconf = max(0.0, min(1.0, vmag / _VEL_REF))
                    tb = t.predict(frame_idx) if (recent and vconf > 0.0) else None
                    for b, j in enumerate(matchable):
                        motion = (
                            self.motion_weight * vconf * _iou_xyxy(tb, dets[j]["bbox"])
                            if tb is not None else 0.0
                        )
                        cost[i, b] = sim[i, j] + motion
                rows, cols = linear_sum_assignment(-cost)  # maximise appearance + motion
                for i, b in zip(rows, cols):
                    i = int(i)
                    j = matchable[int(b)]
                    # The detection already appearance-matches SOME identity, so it
                    # attaches (never a spurious new track); motion only decided which.
                    s = float(sim[i, j])
                    rival = float(np.delete(sim[:, j], i).max()) if len(ids) > 1 else 0.0
                    attach(ids[i], j, bankable=(s - rival >= _BANK_MARGIN))

        # Anything not confidently matched to a known identity becomes a new one.
        for j in list(unmatched_d):
            t = self._new_track(frame_idx, dets[j]["bbox"], embs[j])
            assigned[j] = t.id

        return [(assigned[j], dets[j]["bbox"]) for j in range(len(dets))]

    def summary(self) -> list[dict]:
        # Only identities present long enough surface (short blips/hallucinations
        # stay hidden) — threshold is duration-based (self.min_surface_frames).
        return [
            {
                "id": t.id,
                "color": t.color,
                "firstFrame": int(t.first_frame),
                "lastFrame": int(t.last_frame),
                "frameCount": int(t.frame_count),
                "repFrame": int(t.rep_frame),
            }
            for t in sorted(self.tracks, key=lambda x: x.frame_count, reverse=True)
            if t.confirmed and t.frame_count >= self.min_surface_frames
        ]

# ── MLX SAM3 detection ───────────────────────────────────────────────────────

def _build_processor(confidence: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    return Sam3Processor(model, confidence_threshold=confidence)


def detect_persons(processor, frame_bgr: np.ndarray, prompt: str, text_cache=None):
    """Run SAM3 (MLX) on one frame; return (boxes_xyxy_pixels, scores, embeddings).

    The text prompt ("person") is identical every frame, so its encoding is
    computed once and reused (text_cache) instead of re-encoding per frame."""
    from PIL import Image

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    state = processor.set_image(img)
    if text_cache is not None:
        state["backbone_out"].update(text_cache)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = processor.model._get_dummy_prompt()
        state = processor._call_grounding(state)
    else:
        state = processor.set_text_prompt(prompt, state)
    boxes = state.get("boxes")
    scores = state.get("scores")
    embs = state.get("embeddings")
    if boxes is None:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 256), np.float32)
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.array(scores, dtype=np.float32).reshape(-1) if scores is not None else np.ones(len(boxes), np.float32)
    embs = np.array(embs, dtype=np.float32) if embs is not None else np.zeros((len(boxes), 256), np.float32)
    if embs.shape[0] != boxes.shape[0]:
        embs = np.zeros((len(boxes), embs.shape[-1] if embs.ndim == 2 else 256), np.float32)
    return boxes, scores, embs


# ── Streaming driver ─────────────────────────────────────────────────────────

_STOP = False


def _handle_term(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def _norm_xywh(bbox: np.ndarray, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = _clip_xyxy(bbox, w, h)
    return [round(x1 / w, 5), round(y1 / h, 5), round((x2 - x1) / w, 5), round((y2 - y1) / h, 5)]


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def run(args: argparse.Namespace) -> int:
    global _STOP
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    tracks_path = out_dir / "tracks.json"
    frames_path = out_dir / "frames.jsonl"

    cap = cv2.VideoCapture(str(args.video_input))
    if not cap.isOpened():
        _write_json(progress_path, {"status": "error", "error": "Could not open video.", "processed": 0})
        return 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(args.stride))
    max_frames = int(args.max_frames) if args.max_frames else total_frames
    if max_frames <= 0:
        max_frames = total_frames
    # When the container does not report a frame count, scan until decode ends.
    hard_cap = max_frames if max_frames > 0 else None
    total_to_process = (min(total_frames, max_frames) + stride - 1) // stride if total_frames else 0

    def write_progress(status: str, processed: int, last_frame: int) -> None:
        _write_json(progress_path, {
            "status": status,
            "pid": os.getpid(),
            "processed": processed,
            "total_to_process": total_to_process,
            "total_frames": total_frames,
            "last_frame": last_frame,
            "video_width": width,
            "video_height": height,
            "fps": fps,
            "stride": stride,
        })

    # Hallucination filter, expressed as a DURATION in seconds → frames, so it scales
    # with the real fps and stride (e.g. 1.0 s = 30 detection-points at 30 fps stride 1,
    # 6 at stride 5). A subject present for less time never surfaces.
    min_duration_sec = float(getattr(args, "min_duration_sec", 1.0) or 0.0)
    min_surface_frames = max(1, round(min_duration_sec * fps / stride)) if min_duration_sec > 0 else 1

    write_progress("loading", 0, -1)
    tm = TrackManager(min_surface_frames=min_surface_frames)
    processor = _build_processor(float(args.confidence))
    prompt = args.prompt or "person"
    # Encode the (constant) text prompt once; reused every frame.
    try:
        text_cache = processor.model.backbone.call_text([prompt])
    except Exception:
        text_cache = None
    write_progress("running", 0, -1)

    frames_file = frames_path.open("w", buffering=1)  # line-buffered
    processed = 0
    last_frame = -1
    frame_idx = -1
    t_start = time.perf_counter()
    try:
        while not _STOP:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_idx += 1
            if hard_cap is not None and frame_idx >= hard_cap:
                break
            if frame_idx % stride != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            boxes, scores, embs = detect_persons(processor, frame, prompt, text_cache)
            dets = []
            for i in range(len(boxes)):
                dets.append({
                    "bbox": _clip_xyxy(boxes[i], width, height),
                    "emb": embs[i] if i < len(embs) else None,
                    "score": float(scores[i]) if i < len(scores) else 1.0,
                })
            assigned = tm.update(frame_idx, dets)

            line = {
                "f": frame_idx,
                "t": round(frame_idx / fps, 3),
                "dets": [
                    {"id": tid, "b": _norm_xywh(bbox, width, height), "s": round(dets[i]["score"], 3)}
                    for i, (tid, bbox) in enumerate(assigned)
                ],
            }
            frames_file.write(json.dumps(line) + "\n")
            processed += 1
            last_frame = frame_idx
            if processed % 5 == 0:
                write_progress("running", processed, last_frame)
                _write_json(tracks_path, {"tracks": tm.summary()})
    finally:
        frames_file.flush()
        frames_file.close()
        cap.release()

    _write_json(tracks_path, {"tracks": tm.summary()})
    if os.environ.get("KINESIA_DUMP_GALLERIES") == "1":
        import numpy as _np
        _np.savez(
            Path(args.out_dir) / "galleries.npz",
            **{f"id{t.id}": _np.stack(t.gallery) for t in tm.tracks if t.gallery},
        )
    status = "stopped" if _STOP else "completed"
    write_progress(status, processed, last_frame)
    elapsed = time.perf_counter() - t_start
    print(json.dumps({
        "ok": True, "status": status, "processed": processed,
        "tracks": len(tm.summary()), "elapsed_sec": round(elapsed, 1),
        "per_frame_ms": round(1000 * elapsed / max(1, processed), 0),
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="detect_stream", description="Streaming SAM3 (MLX) person detection + embedding re-ID.")
    parser.add_argument("--video-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-duration-sec", type=float, default=1.0,
                        help="ignore subjects present for less than this many seconds (0 = no filter)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
