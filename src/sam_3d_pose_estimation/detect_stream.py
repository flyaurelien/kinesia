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

# ── Tracking: position-first cascade with appearance re-identification ──────
# Two people crossing paths is the case that matters, and appearance is the WORST
# signal exactly there: the boxes blend, the embeddings mix, and a confident-wrong
# match swaps the identities for the rest of the clip. Position is the OPPOSITE:
# on contiguous frames people cannot teleport, so a track's constant-velocity
# prediction overlaps its own person overwhelmingly better than the other one —
# crossings included. Hence a cascade (the BoT-SORT/ByteTrack recipe):
#
#   1. ACTIVE tracks (seen recently) are matched to detections by PREDICTED-BOX
#      IoU (Hungarian). Appearance only breaks ties, and only on detections that
#      are ISOLATED (not overlapping another detection) — in a crossing blend it
#      is contaminated, so there it is ignored entirely.
#   2. Detections no active track claims (and tracks whose IoU gate failed —
#      e.g. after a hard scene cut, where position is meaningless) fall back to
#      appearance-only re-identification against the full gallery history.
#   3. A leftover detection becomes a NEW identity only if it does not
#      substantially overlap any existing track — a crossing fragment (the box
#      hugging two half-people) must never be born as a ghost identity.
#
# Galleries stay unbounded (every distinct view of a person, no averaging), but a
# view is MEMORISED only when the detection is isolated — an overlapped view is
# displayed yet never banked, so a crossing can never poison the history that
# appearance re-ID depends on.

_MATCH_COS = 0.84        # appearance re-ID threshold (stage 2): same person
                         # ~0.86-0.99 vs different ~0.72-0.83 on this backbone.
_MATCH_COS_LOW = 0.76    # second-chance re-ID threshold for tracks lost only
                         # MOMENTS ago (within the active window) — right after a
                         # scene cut the same person can dip below the strict
                         # threshold (backlight, scale); a recently-present person
                         # is near-certainly still there, so the Hungarian-best
                         # pairing is accepted at this laxer bar WITH a margin.
_MATCH_LOW_MARGIN = 0.03 # ...but only when it beats the runner-up by this much.
_MATCH_COS_FLOOR = 0.60  # last-resort pairing floor (stage 2c): after a camera
                         # change the SAME person from a new angle (e.g. from
                         # behind) can sit well below the lax bar; when recently-
                         # seen tracks and unclaimed detections remain, pairing
                         # them beats fragmenting a real person into a ghost id.
                         # Never applied during a dissolve frame (blends could
                         # lock in an error) and never blindly: Hungarian-optimal
                         # with a per-track winner margin.
_TELEPORT_VEL = 40.0     # px/frame; an implausible per-frame displacement (scene
                         # cut) resets the velocity instead of poisoning the EMA.
_BANK_MARGIN = 0.05      # additionally require the matched identity to beat every
                         # rival by this margin before a view enters the gallery.
_GALLERY_TOPK = 5        # appearance = mean of the top-K best-matching gallery views
_DEDUP_COS = 0.992       # skip only a near-identical view (keeps every real pose/angle)
_N_INIT = 2              # detections before an identity is confirmed (surfaced)
_MIN_SURFACE_FRAMES = 2
_ACTIVE_RECENCY = 75     # VIDEO FRAMES since last sighting for a track to count as
                         # ACTIVE (position still meaningful, second-chance re-ID
                         # eligible): 3 s at 25 fps — enough to coast through a
                         # dissolve transition (~1.2 s) or a short occlusion.
                         # Beyond it, re-ID is strict appearance-only (stage 2).
_IOU_GATE = 0.10         # minimum predicted-box IoU for a stage-1 position match
_APP_TIEBREAK = 0.35     # appearance weight inside the stage-1 cost (isolated dets
                         # only) — position dominates, appearance orders near-ties.
_CROSS_IOU = 0.30        # detection-detection overlap at/above which the pair is a
                         # crossing blend: appearance is ignored and never banked.
_NEW_ID_MAX_IOU = 0.35   # a leftover detection overlapping an existing track above
                         # this spawns NO identity (crossing fragment — suppressed).
_SCENE_CUT_DIFF = 28.0   # mean |Δgray| (0-255) between consecutive processed frames
                         # (32x18 downsample) above which the frame is a hard cut:
                         # position is void there. Walking at stride 5 stays ~3-12;
                         # a white flash or a room change jumps to 60-120.


def _frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail used to detect hard scene cuts."""
    small = cv2.resize(frame_bgr, (32, 18), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _is_scene_change(sig: np.ndarray, prev_sig: np.ndarray | None) -> bool:
    if prev_sig is None:
        return False
    return float(np.abs(sig - prev_sig).mean()) >= _SCENE_CUT_DIFF


class _Track:
    """One identity = an unbounded GALLERY of all its raw unit embeddings (every
    distinct view it has ever shown), plus bookkeeping. Identity is decided by
    appearance; a last box + constant-velocity estimate are kept ONLY as a motion
    tie-break so two people can't swap identities at a crossing."""

    __slots__ = (
        "id", "color", "box", "vel", "pos_valid", "gallery", "_gmat", "tab",
        "hits", "confirmed", "first_frame", "last_frame", "frame_count",
        "rep_frame", "rep_area",
    )

    def __init__(self, tid: int, frame_idx: int, box_xyxy: np.ndarray, feat):
        self.id = tid
        self.color = _PALETTE[tid % len(_PALETTE)]
        self.box = _xyxy_to_cxcywh(box_xyxy)
        self.vel = np.zeros(2, dtype=np.float32)  # per-frame centre velocity (cx, cy)
        # False after a scene cut: the box belongs to the OLD scene, so position
        # matching is forbidden until appearance re-anchors this identity.
        self.pos_valid = True
        # Gallery of RAW unit embeddings — every view, un-averaged, uncapped.
        self.gallery: list[np.ndarray] = []
        self._gmat: np.ndarray | None = None  # cached stack of `gallery` for matching
        # Clothing-colour samples (trousers LAB a*/b*) from ISOLATED detections —
        # the cross-segment identity descriptor (see _trousers_ab).
        self.tab: list[tuple[float, float]] = []
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
        if float(np.hypot(inst_vel[0], inst_vel[1])) > _TELEPORT_VEL:
            # Implausible jump (scene cut / re-ID across a hard reposition):
            # position history is void — restart motion instead of poisoning it.
            self.vel = np.zeros(2, dtype=np.float32)
        elif self.frame_count <= 1:
            self.vel = inst_vel.astype(np.float32)
        else:
            self.vel = (0.6 * self.vel + 0.4 * inst_vel).astype(np.float32)
        self.box = new_box
        if e_unit is not None:                        # bank every view (measured: a
            self._add(e_unit)                         # few degraded ones dilute, even
                                                      # enrich, and never erode the margin)
        self.pos_valid = True  # an accepted match anchors the box in the current scene
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
        active_recency: int = _ACTIVE_RECENCY,
        app_tiebreak: float = _APP_TIEBREAK,
    ):
        self.tracks: list[_Track] = []
        self._next_id = 0
        # An identity must be present for at least this many detection-points to
        # surface (computed from a duration-in-seconds so it scales with fps/stride).
        self.min_surface_frames = max(1, int(min_surface_frames))
        # Cascade knobs (instance-level so cached detections can be replayed
        # offline with different values without re-running SAM3).
        self.active_recency = int(active_recency)
        self.app_tiebreak = float(app_tiebreak)

    def _new_track(self, frame_idx: int, box: np.ndarray, e_unit) -> _Track:
        t = _Track(self._next_id, frame_idx, box, e_unit)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def update(
        self,
        frame_idx: int,
        dets: list[dict],
        scene_change: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        """Assign identities to this frame's detections — position-first cascade.

        Stage 1: ACTIVE tracks (seen within ``active_recency`` detection-points)
        are matched by predicted-box IoU (Hungarian, gated at ``_IOU_GATE``);
        appearance only orders near-ties and only for ISOLATED detections — in a
        crossing blend it is contaminated, so position alone decides there.
        On a ``scene_change`` frame (hard cut) position is MEANINGLESS — two
        people can coincidentally land where the other stood — so stage 1 is
        skipped entirely and the frame is resolved by appearance alone.
        Stage 2: whatever stage 1 left (stale tracks, post-cut frames where every
        IoU gate fails) is matched by appearance against the full gallery history
        (strict ``_MATCH_COS``); a second pass re-attaches tracks lost only
        moments ago at the laxer ``_MATCH_COS_LOW`` with a winner margin — right
        after a cut the same person can dip below the strict bar, and losing them
        to a brand-new identity is the worse error.
        Stage 3: a leftover detection spawns a NEW identity only if it does not
        overlap an existing track above ``_NEW_ID_MAX_IOU``; otherwise it is a
        crossing fragment and is SUPPRESSED (returned with id -1, never written).

        Returns one ``(track_id, bbox)`` per detection; ``track_id`` is -1 for
        suppressed crossing fragments."""
        assigned: dict[int, int] = {}
        unmatched_d = set(range(len(dets)))

        # Pre-normalise detection embeddings once.
        embs = [
            (_l2norm(d["emb"]) if d.get("emb") is not None else None)
            for d in dets
        ]

        # A detection is ISOLATED when it does not overlap any other detection in
        # this frame — only then is its embedding trustworthy (tie-breaks) and
        # bankable (gallery growth). In a crossing, both blend and must be ignored.
        n = len(dets)
        isolated = [True] * n
        for a in range(n):
            for b in range(a + 1, n):
                if _iou_xyxy(dets[a]["bbox"], dets[b]["bbox"]) >= _CROSS_IOU:
                    isolated[a] = isolated[b] = False

        sim_cache: dict[tuple[int, int], float] = {}

        def sim(t: _Track, j: int) -> float:
            key = (t.id, j)
            if key not in sim_cache:
                sim_cache[key] = t.appearance(embs[j]) if embs[j] is not None else 0.0
            return sim_cache[key]

        def attach(t: _Track, j: int) -> None:
            # Bank the view only when the detection is isolated AND its identity
            # clearly beats every rival — the gallery must never ingest a blend.
            bankable = isolated[j] and not scene_change
            if bankable and len(self.tracks) > 1:
                s = sim(t, j)
                rival = max((sim(o, j) for o in self.tracks if o.id != t.id), default=0.0)
                bankable = (s - rival) >= _BANK_MARGIN
            t.update(frame_idx, dets[j]["bbox"], embs[j] if bankable else None)
            if not scene_change and dets[j].get("tab") is not None and _strip_clear(dets, j):
                t.tab.append(dets[j]["tab"])
            if scene_change:
                # A dissolve-frame box is a blend of two scenes: displaying it is
                # fine, but it must NOT re-validate position — otherwise stage 1
                # resumes on the first stable frame anchored to a blend and can
                # grab the wrong person. Position revalidates only from a stable
                # frame (update() above flips it back on the next clean attach).
                t.pos_valid = False
            assigned[j] = t.id
            unmatched_d.discard(j)

        # ── Stage 1: position continuity for ACTIVE tracks ──────────────────
        if scene_change:
            # Hard cut / dissolve frame: every stored box belongs to the OLD
            # scene. Invalidate positions — stage 1 stays forbidden for each
            # identity until appearance re-anchors it in the new scene (its
            # update() flips pos_valid back). This also covers the frames right
            # AFTER the cut, where stale predictions could coincidentally
            # overlap the other person and steal their identity.
            for t in self.tracks:
                t.pos_valid = False
        active = [
            t for t in self.tracks
            if t.pos_valid and (frame_idx - t.last_frame) <= self.active_recency
        ]
        if active and dets:
            preds = [t.predict(frame_idx) for t in active]
            iou = np.zeros((len(active), n), dtype=np.float32)
            for i in range(len(active)):
                for j in range(n):
                    iou[i, j] = _iou_xyxy(preds[i], dets[j]["bbox"])
            cost = np.full((len(active), n), 1e6, dtype=np.float32)
            for i, t in enumerate(active):
                for j in range(n):
                    if iou[i, j] >= _IOU_GATE:
                        tie = self.app_tiebreak * sim(t, j) if isolated[j] else 0.0
                        cost[i, j] = -(float(iou[i, j]) + tie)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[int(i), int(j)] < 1e5:  # gate passed
                    attach(active[int(i)], int(j))

        # ── Stage 2: appearance re-identification for everything left ───────
        # Covers stale tracks (person left and came back) and hard scene cuts
        # (active tracks whose IoU gate failed because the camera jumped).
        rem_tracks = [t for t in self.tracks if t.id not in assigned.values()]
        rem_dets = sorted(unmatched_d)
        if rem_tracks and rem_dets:
            app = np.zeros((len(rem_tracks), len(rem_dets)), dtype=np.float32)
            for i, t in enumerate(rem_tracks):
                for b, j in enumerate(rem_dets):
                    app[i, b] = sim(t, j)
            rows, cols = linear_sum_assignment(-app)
            for i, b in zip(rows, cols):
                if float(app[int(i), int(b)]) >= _MATCH_COS:
                    attach(rem_tracks[int(i)], rem_dets[int(b)])

        # ── Stage 2b: second-chance re-ID for JUST-LOST tracks ──────────────
        # A person present moments ago (within the active window) is near-
        # certainly still there; right after a hard cut their similarity can dip
        # below the strict bar (backlight, scale, new room). Accept the best
        # pairing at a laxer threshold, but only when it clearly beats the
        # runner-up — fragmenting a real person into a ghost id is the worse error.
        rem2_tracks = [
            t for t in self.tracks
            if t.id not in assigned.values()
            and (frame_idx - t.last_frame) <= self.active_recency
        ]
        rem2_dets = sorted(unmatched_d)
        if rem2_tracks and rem2_dets:
            app2 = np.zeros((len(rem2_tracks), len(rem2_dets)), dtype=np.float32)
            for i, t in enumerate(rem2_tracks):
                for b, j in enumerate(rem2_dets):
                    app2[i, b] = sim(t, j)
            rows, cols = linear_sum_assignment(-app2)
            for i, b in zip(rows, cols):
                s = float(app2[int(i), int(b)])
                runner_up = (
                    float(np.delete(app2[:, int(b)], int(i)).max())
                    if len(rem2_tracks) > 1 else 0.0
                )
                if s >= _MATCH_COS_LOW and (s - runner_up) >= _MATCH_LOW_MARGIN:
                    attach(rem2_tracks[int(i)], rem2_dets[int(b)])

        # ── Stage 2c: last-resort pairing after a camera change ─────────────
        # A person filmed from a NEW angle (e.g. from behind after a cut) can
        # fall below even the lax bar. If recently-seen tracks and unclaimed
        # detections still face each other on a STABLE frame, pair them
        # (Hungarian-optimal, floor + per-track margin) rather than fragment a
        # real person into a ghost identity. Skipped mid-dissolve: a blend
        # could anchor the wrong person and stage 1 would then lock the error.
        if not scene_change:
            rem3_tracks = [
                t for t in self.tracks
                if t.id not in assigned.values()
                and (frame_idx - t.last_frame) <= self.active_recency
            ]
            rem3_dets = sorted(unmatched_d)
            if rem3_tracks and rem3_dets:
                app3 = np.zeros((len(rem3_tracks), len(rem3_dets)), dtype=np.float32)
                for i, t in enumerate(rem3_tracks):
                    for b, j in enumerate(rem3_dets):
                        app3[i, b] = sim(t, j)
                rows, cols = linear_sum_assignment(-app3)
                for i, b in zip(rows, cols):
                    s = float(app3[int(i), int(b)])
                    row_margin = (
                        s - float(np.delete(app3[int(i), :], int(b)).max())
                        if len(rem3_dets) > 1 else 1.0
                    )
                    if s >= _MATCH_COS_FLOOR and row_margin >= _MATCH_LOW_MARGIN:
                        attach(rem3_tracks[int(i)], rem3_dets[int(b)])

        # ── Stage 3: births — never inside a crossing or a scene transition ─
        for j in list(unmatched_d):
            if scene_change:
                # Mid-dissolve frames are blends of two scenes: nobody genuinely
                # appears there, and a blend must never become a ghost identity.
                assigned[j] = -1
                unmatched_d.discard(j)
                continue
            overlap = max(
                (
                    _iou_xyxy(t.predict(frame_idx), dets[j]["bbox"])
                    for t in self.tracks
                    if t.pos_valid  # an old-scene box must not veto a birth
                ),
                default=0.0,
            )
            if overlap >= _NEW_ID_MAX_IOU:
                # Crossing fragment (a box hugging two half-people): suppressed —
                # it must never be born as a ghost identity.
                assigned[j] = -1
                unmatched_d.discard(j)
                continue
            t = self._new_track(frame_idx, dets[j]["bbox"], embs[j])
            assigned[j] = t.id
            unmatched_d.discard(j)

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


# ── Segment-level identity linking (montage-proof) ──────────────────────────
# A clinical video is often a MONTAGE: several takes joined by dissolves. Within
# one take, position continuity is near-infallible (people cannot teleport).
# ACROSS a dissolve, position is void, and single-frame appearance right after a
# cut can be actively misleading (backlight, tiny scale: measured wrong-person
# sims up to 0.96). What IS reliable is the AGGREGATE: a person's whole-segment
# gallery (dozens of views — close-ups included) matched against another
# whole-segment gallery separates the same pair by a wide margin (measured
# ~0.98 same vs ~0.85 cross on the reference clip).
#
# So: one fresh TrackManager per segment; when a segment closes, its tracklets
# are linked to the global identities by gallery-to-gallery similarity
# (Hungarian). The FIRST segment's ids stream live (they ARE the global ids —
# a video without cuts behaves exactly as before); later segments are buffered
# and flushed with FINAL global ids the moment the segment closes, so an id is
# never rewritten after being emitted. Dissolve frames emit nothing.

_LINK_MAX_DIST = 10.0    # max LAB (a*,b*) euclidean distance for a tracklet to
                         # link to an existing identity. Measured on the reference
                         # montage: correct links 0.4-8.9, a bystander fragment vs
                         # a main 11.4 — so 10 accepts every true link and rejects
                         # the impostor.
_IDENTITY_AB_CAP = 200   # keep identity colour-sample sets bounded.
_LINK_TARGET_MIN = 30    # an identity must carry at least this many frames of
                         # evidence to be a LINK TARGET — a 12-frame bystander
                         # blip with vaguely-similar colours must never attract a
                         # main subject away from their real identity.
_LINK_HARD_CAP = 20.0    # beyond this colour distance a link is never accepted.
_LINK_ASSIGN_MARGIN = 6.0  # a pair above _LINK_MAX_DIST is still accepted when
                         # the JOINT assignment demands it: banning the pair and
                         # re-solving must cost at least this much more. Scene
                         # lighting can shift denim's b* by >10 in absolute terms,
                         # but the alternative pairing (swapping the two people)
                         # stays far worse — measured margin 17.6 on the shifted
                         # segment vs 5.5 for a bystander fragment.

# Why COLOUR and not the SAM3 embeddings for cross-segment linking: measured on
# the reference montage, whole-tracklet SAM3-embedding similarity picked the
# WRONG person in 3 of 7 segments with high confidence (0.96-0.98) — the decoder
# queries encode pose/scale/context more than identity. The trousers-region LAB
# chroma (a*,b*) — khaki vs denim — separated every segment with a wide margin
# and is nearly invariant to the lighting changes between takes.


def _trousers_rect(box_xyxy: np.ndarray) -> tuple[float, float, float, float]:
    """The central lower strip of a person box sampled by _trousers_ab."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy[:4])
    h, w = y2 - y1, x2 - x1
    return (x1 + 0.30 * w, y1 + 0.55 * h, x2 - 0.30 * w, y1 + 0.92 * h)


def _strip_clear(dets: list[dict], j: int) -> bool:
    """True when detection ``j``'s trousers strip is not covered by any other
    detection — upper bodies may overlap, but a colour sample is only taken
    when the LEGS region itself is clearly this person's."""
    sx1, sy1, sx2, sy2 = _trousers_rect(dets[j]["bbox"])
    area = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
    for k, other in enumerate(dets):
        if k == j:
            continue
        ox1, oy1, ox2, oy2 = (float(v) for v in other["bbox"][:4])
        ix = max(0.0, min(sx2, ox2) - max(sx1, ox1))
        iy = max(0.0, min(sy2, oy2) - max(sy1, oy1))
        if ix * iy / area > 0.15:
            return False
    return True


def _trousers_ab(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> tuple[float, float] | None:
    """Median LAB (a*, b*) of the trousers region (central lower strip) of a
    person box — a small, lighting-robust clothing-colour descriptor."""
    x1, y1, x2, y2 = (int(v) for v in box_xyxy[:4])
    h, w = y2 - y1, x2 - x1
    if h < 40 or w < 12:
        return None
    ys, ye = y1 + int(0.55 * h), y1 + int(0.92 * h)  # trousers, avoid feet/floor
    xs, xe = x1 + int(0.30 * w), x2 - int(0.30 * w)  # central strip only
    crop = frame_bgr[max(0, ys):max(0, ye), max(0, xs):max(0, xe)]
    if crop.size < 300:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    return (
        float(np.median(lab[..., 1]) - 128.0),
        float(np.median(lab[..., 2]) - 128.0),
    )


class SegmentedIdentityTracker:
    """Drives one TrackManager per scene segment and links tracklets globally.

    ``process`` returns fully-FINAL jsonl lines ready to append (possibly none
    while buffering, several when a segment closes)."""

    def __init__(self, min_surface_frames: int, width: int, height: int, fps: float):
        self.min_surface_frames = max(1, int(min_surface_frames))
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self._tm = TrackManager(min_surface_frames=1)
        self._seg_idx = 0
        self._in_cut = False
        # Buffered (frame_idx, [(local_id, bbox, score), ...]) for segments > 0.
        self._buffer: list[tuple[int, list[tuple[int, np.ndarray, float]]]] = []
        # Global identities: gid -> {views, count, first, last, rep_frame, rep_area}
        self._ids: dict[int, dict] = {}
        self._next_gid = 0

    # ── internals ────────────────────────────────────────────────────────────
    def _line(self, frame_idx: int, entries: list[tuple[int, np.ndarray, float]]) -> str:
        return json.dumps({
            "f": frame_idx,
            "t": round(frame_idx / self.fps, 3),
            "dets": [
                {"id": int(g), "b": _norm_xywh(b, self.width, self.height), "s": round(s, 3)}
                for g, b, s in entries if g >= 0
            ],
        })

    def _absorb(self, gid: int, t: _Track) -> None:
        ident = self._ids.setdefault(gid, {
            "tab": [], "count": 0, "first": t.first_frame,
            "last": t.last_frame, "rep_frame": t.rep_frame, "rep_area": t.rep_area,
        })
        ident["tab"].extend(t.tab)
        if len(ident["tab"]) > _IDENTITY_AB_CAP:
            ident["tab"] = ident["tab"][::2]
        ident["count"] += t.frame_count
        ident["first"] = min(ident["first"], t.first_frame)
        ident["last"] = max(ident["last"], t.last_frame)
        if t.rep_area > ident["rep_area"]:
            ident["rep_area"] = t.rep_area
            ident["rep_frame"] = t.rep_frame

    def _close_segment(self) -> list[str]:
        """Link the finished segment's tracklets to global identities and flush
        its buffered lines with final ids."""
        tm, buf = self._tm, self._buffer
        self._tm = TrackManager(min_surface_frames=1)
        self._buffer = []
        seg_was_first = (self._seg_idx == 0)
        self._seg_idx += 1

        if seg_was_first:
            # Segment-0 local ids streamed live and ARE the global ids.
            for t in tm.tracks:
                self._absorb(t.id, t)
            self._next_gid = max(self._next_gid, tm._next_id)
            return []

        # Link tracklets → identities by whole-tracklet CLOTHING COLOUR (median
        # trousers a*/b*): the only signal measured to order every segment of
        # the reference montage correctly (SAM3 embeddings picked the wrong
        # person in 3/7 segments — see the note at _trousers_ab).
        tracklets = [t for t in tm.tracks]
        mapping: dict[int, int] = {}
        cand = [t for t in tracklets if len(t.tab) >= 3]
        usable = [
            g for g, ident in self._ids.items()
            if len(ident["tab"]) >= 3 and ident["count"] >= _LINK_TARGET_MIN
        ]
        if cand and usable:
            def med(samples: list[tuple[float, float]]) -> np.ndarray:
                arr = np.asarray(samples, dtype=np.float32)
                return np.median(arr, axis=0)
            dist = np.zeros((len(cand), len(usable)), dtype=np.float32)
            for i, t in enumerate(cand):
                ct = med(t.tab)
                for jj, g in enumerate(usable):
                    dist[i, jj] = float(np.linalg.norm(ct - med(self._ids[g]["tab"])))
            rows, cols = linear_sum_assignment(dist)
            opt_total = float(dist[rows, cols].sum())
            for i, jj in zip(rows, cols):
                d0 = float(dist[int(i), int(jj)])
                accept = d0 <= _LINK_MAX_DIST
                if not accept and d0 <= _LINK_HARD_CAP:
                    # Assignment-margin rescue: absolute colour can drift with a
                    # scene's lighting, but if forbidding this pair forces a much
                    # worse JOINT assignment, the pairing itself is unambiguous.
                    banned = dist.copy()
                    banned[int(i), int(jj)] = 1e6
                    r2, c2 = linear_sum_assignment(banned)
                    alt_total = float(banned[r2, c2].sum())
                    accept = (alt_total - opt_total) >= _LINK_ASSIGN_MARGIN
                if accept:
                    mapping[cand[int(i)].id] = usable[int(jj)]
        for t in tracklets:
            gid = mapping.get(t.id)
            if gid is None:
                # New identity — but only if it carries enough presence to ever
                # surface; a sub-threshold blip stays local (dropped from output).
                if t.frame_count < self.min_surface_frames:
                    mapping[t.id] = -1
                    continue
                gid = self._next_gid
                self._next_gid += 1
                mapping[t.id] = gid
            self._absorb(gid, t)

        lines = []
        for frame_idx, entries in buf:
            remapped = [(mapping.get(lid, -1), b, s) for lid, b, s in entries]
            lines.append(self._line(frame_idx, remapped))
        return lines

    # ── public API ────────────────────────────────────────────────────────────
    def process(
        self, frame_idx: int, dets: list[dict], scene_change: bool
    ) -> list[str]:
        out: list[str] = []
        if scene_change:
            # Dissolve frame: blends of two scenes — emit nothing, and remember
            # that the running segment must close at the next stable frame.
            self._in_cut = True
            return out
        if self._in_cut:
            self._in_cut = False
            out.extend(self._close_segment())
        res = self._tm.update(frame_idx, dets)
        entries = [
            (tid, bbox, float(dets[j].get("score", 1.0)))
            for j, (tid, bbox) in enumerate(res)
        ]
        if self._seg_idx == 0:
            out.append(self._line(frame_idx, entries))
        else:
            self._buffer.append((frame_idx, entries))
        return out

    def finalize(self) -> list[str]:
        return self._close_segment()

    def summary(self) -> list[dict]:
        # Closed/linked identities + (only while still in segment 0) the live
        # tracklets, whose local ids are already the global ids.
        rows: dict[int, dict] = {}
        for gid, ident in self._ids.items():
            rows[gid] = {
                "id": gid, "color": _PALETTE[gid % len(_PALETTE)],
                "firstFrame": int(ident["first"]), "lastFrame": int(ident["last"]),
                "frameCount": int(ident["count"]), "repFrame": int(ident["rep_frame"]),
            }
        if self._seg_idx == 0:
            for t in self._tm.tracks:
                rows[t.id] = {
                    "id": t.id, "color": _PALETTE[t.id % len(_PALETTE)],
                    "firstFrame": int(t.first_frame), "lastFrame": int(t.last_frame),
                    "frameCount": int(t.frame_count), "repFrame": int(t.rep_frame),
                }
        return [
            r for r in sorted(rows.values(), key=lambda r: -r["frameCount"])
            if r["frameCount"] >= max(self.min_surface_frames, _N_INIT)
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
    tracker = SegmentedIdentityTracker(
        min_surface_frames=min_surface_frames, width=width, height=height, fps=fps,
    )
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
    prev_sig: np.ndarray | None = None  # scene-cut detector state
    # Optional: capture raw SAM3 detections+embeddings per frame for offline
    # tracker development/validation (set KINESIA_DUMP_RAW=1).
    raw_dump = [] if os.environ.get("KINESIA_DUMP_RAW") == "1" else None
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

            sig = _frame_signature(frame)
            scene_change = _is_scene_change(sig, prev_sig)
            prev_sig = sig

            boxes, scores, embs = detect_persons(processor, frame, prompt, text_cache)
            dets = []
            for i in range(len(boxes)):
                bbox = _clip_xyxy(boxes[i], width, height)
                dets.append({
                    "bbox": bbox,
                    "emb": embs[i] if i < len(embs) else None,
                    "score": float(scores[i]) if i < len(scores) else 1.0,
                    # Clothing-colour descriptor for cross-segment identity linking.
                    "tab": _trousers_ab(frame, bbox),
                })
            if raw_dump is not None:
                raw_dump.append({
                    "f": frame_idx,
                    "cut": bool(scene_change),
                    "boxes": [np.asarray(d["bbox"], dtype=np.float32).tolist() for d in dets],
                    "embs": [
                        (np.asarray(d["emb"], dtype=np.float32).tolist() if d["emb"] is not None else None)
                        for d in dets
                    ],
                })

            for line in tracker.process(frame_idx, dets, scene_change):
                frames_file.write(line + "\n")
            processed += 1
            last_frame = frame_idx
            if processed % 5 == 0:
                write_progress("running", processed, last_frame)
                _write_json(tracks_path, {"tracks": tracker.summary()})
    finally:
        # Close the last open segment so every buffered line lands with its
        # FINAL linked identity before the terminal status is written.
        try:
            for line in tracker.finalize():
                frames_file.write(line + "\n")
        finally:
            frames_file.flush()
            frames_file.close()
            cap.release()

    _write_json(tracks_path, {"tracks": tracker.summary()})
    if raw_dump is not None:
        import pickle as _pickle
        with open(Path(args.out_dir) / "raw_dets.pkl", "wb") as _f:
            _pickle.dump(
                {"raw": raw_dump, "width": width, "height": height, "fps": fps, "stride": stride},
                _f,
            )
        print(json.dumps({"raw_dump_frames": len(raw_dump)}))
    status = "stopped" if _STOP else "completed"
    write_progress(status, processed, last_frame)
    elapsed = time.perf_counter() - t_start
    print(json.dumps({
        "ok": True, "status": status, "processed": processed,
        "tracks": len(tracker.summary()), "elapsed_sec": round(elapsed, 1),
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
