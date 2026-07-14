"""Deterministic crossing benchmark for the detect-stream TrackManager.

Reproduces the real failure: two people walk toward each other and cross; at
the overlap their SAM3 embeddings BLEND (cosine to both galleries above the
match threshold — a confident-wrong appearance signal), and at full occlusion
the detector may emit a single merged box. The tracker must keep both
identities on their own trajectories (no swap), suppress ghost births during
the overlap, and still re-identify by appearance after a genuine absence or a
hard scene cut."""

import unittest

import numpy as np

from sam_3d_pose_estimation.detect_stream import SegmentedIdentityTracker, TrackManager

W, H = 1280, 720
BOX_W, BOX_H = 60, 160
STRIDE = 5  # detection-point spacing in video frames


def _emb(base: np.ndarray) -> np.ndarray:
    return (base / np.linalg.norm(base)).astype(np.float32)


# Two people with different-person appearance (cos ~0.75, below the 0.84
# match threshold) and their crossing blend (cos ~0.94 to BOTH — above it).
_E_A = _emb(np.eye(256, dtype=np.float32)[0])
_E_B = _emb(0.75 * np.eye(256, dtype=np.float32)[0] + np.sqrt(1 - 0.75**2) * np.eye(256, dtype=np.float32)[1])
_E_BLEND = _emb(_E_A + _E_B)


def _box(cx: float, cy: float = 360.0) -> np.ndarray:
    return np.array(
        [cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2],
        dtype=np.float32,
    )


def _det(cx: float, emb: np.ndarray) -> dict:
    return {"bbox": _box(cx), "emb": emb, "score": 0.9}


class TestCrossingNoSwap(unittest.TestCase):
    def _walk_and_cross(self, tm: TrackManager):
        """A walks 300->900 (left to right), B walks 900->300. They meet at 600.
        Returns (id_A, id_B, per-frame assignments)."""
        out = []
        id_a = id_b = None
        speed = 15.0  # px per detection-point
        for k in range(41):
            f = k * STRIDE
            ax = 300.0 + speed * k
            bx = 900.0 - speed * k
            sep = abs(ax - bx)
            if sep < 30:
                # Full occlusion: ONE merged box with a blended embedding.
                dets = [_det((ax + bx) / 2, _E_BLEND)]
            elif sep < 90:
                # Partial overlap: two boxes, both embeddings contaminated.
                dets = [_det(ax, _E_BLEND), _det(bx, _E_BLEND)]
            else:
                dets = [_det(ax, _E_A), _det(bx, _E_B)]
            res = tm.update(f, dets)
            out.append((f, ax, bx, dets, res))
            if k == 0:
                id_a, id_b = res[0][0], res[1][0]
        return id_a, id_b, out

    def test_identities_never_swap_at_the_crossing(self):
        tm = TrackManager()
        id_a, id_b, out = self._walk_and_cross(tm)
        self.assertNotEqual(id_a, id_b)
        for f, ax, bx, dets, res in out:
            for (tid, bbox), det in zip(res, dets):
                if tid < 0:
                    continue  # suppressed fragment
                cx = float((bbox[0] + bbox[2]) / 2)
                # The detection nearest A's ground-truth position must carry
                # id_a — including every frame after the crossing.
                truth = id_a if abs(cx - ax) <= abs(cx - bx) else id_b
                if abs(ax - bx) >= 90:  # unambiguous frames only
                    self.assertEqual(
                        tid, truth,
                        f"identity swap at frame {f}: box at cx={cx:.0f} got id {tid}",
                    )

    def test_no_ghost_identity_is_born_during_the_overlap(self):
        tm = TrackManager()
        id_a, id_b, out = self._walk_and_cross(tm)
        surfaced = {t.id for t in tm.tracks}
        self.assertEqual(
            surfaced, {id_a, id_b},
            f"ghost identities born at the crossing: {surfaced - {id_a, id_b}}",
        )

    def test_reid_after_long_absence_is_appearance_based(self):
        tm = TrackManager()
        id_a = tm.update(0, [_det(300.0, _E_A)])[0][0]
        # A leaves; far beyond the active window, reappears elsewhere.
        f_back = (tm.active_recency + 200)
        res = tm.update(f_back, [_det(1000.0, _E_A)])
        self.assertEqual(res[0][0], id_a, "appearance re-ID after absence failed")

    def test_hard_scene_cut_reids_both_without_swapping(self):
        tm = TrackManager()
        r0 = tm.update(0, [_det(300.0, _E_A), _det(900.0, _E_B)])
        id_a, id_b = r0[0][0], r0[1][0]
        # Build a little clean gallery for both.
        for k in range(1, 4):
            tm.update(k * STRIDE, [_det(300.0 + 5 * k, _E_A), _det(900.0 - 5 * k, _E_B)])
        # Hard cut: camera jumps, the two are now on OPPOSITE sides.
        res = tm.update(4 * STRIDE, [_det(1000.0, _E_A), _det(200.0, _E_B)])
        self.assertEqual(res[0][0], id_a, "post-cut re-ID lost A")
        self.assertEqual(res[1][0], id_b, "post-cut re-ID lost B")


class TestSegmentedLinking(unittest.TestCase):
    """Montage behaviour: within a segment position rules; across a dissolve the
    tracker links whole tracklets by clothing colour (trousers LAB a*/b*) — the
    signal measured to survive scene changes where per-frame appearance fails."""

    KHAKI = (5.0, 17.0)
    DENIM = (1.0, -1.0)

    def _det(self, cx: float, emb: np.ndarray, tab) -> dict:
        return {"bbox": _box(cx), "emb": emb, "score": 0.9, "tab": tab}

    def _run_segment(self, trk, f0, n, pos_a, pos_b, swap_embs=False):
        """n frames with A (denim) at pos_a and B (khaki) at pos_b."""
        import json
        lines = []
        for k in range(n):
            f = f0 + k * STRIDE
            ea, eb = (_E_B, _E_A) if swap_embs else (_E_A, _E_B)
            lines += trk.process(f, [
                self._det(pos_a, ea, self.DENIM),
                self._det(pos_b, eb, self.KHAKI),
            ], scene_change=False)
        return [json.loads(l) for l in lines]

    def test_identities_survive_a_hard_scene_change_by_colour(self):
        import json
        trk = SegmentedIdentityTracker(min_surface_frames=2, width=W, height=H, fps=25.0)
        out = self._run_segment(trk, 0, 40, pos_a=300.0, pos_b=900.0)
        # Segment 0 streams live: A and B get ids immediately.
        first = out[0]["dets"]
        id_a = next(d["id"] for d in first if abs(d["b"][0] * W + d["b"][2] * W / 2 - 300) < 60)
        id_b = next(d["id"] for d in first if d["id"] != id_a)
        # Dissolve (positions void, nothing emitted)…
        for k in range(4):
            self.assertEqual(trk.process(200 * STRIDE + k * STRIDE, [], scene_change=True), [])
        # …new scene: SIDES SWAPPED and per-frame embeddings ADVERSARIAL
        # (each person's embedding now matches the OTHER's gallery).
        out2 = self._run_segment(trk, 210 * STRIDE, 40, pos_a=900.0, pos_b=300.0, swap_embs=True)
        out2 += [json.loads(l) for l in trk.finalize()]
        # Every post-cut frame: the DENIM person (A, now right) keeps id_a.
        for line in out2:
            for dd in line["dets"]:
                cx = (dd["b"][0] + dd["b"][2] / 2) * W
                want = id_a if abs(cx - 900) < abs(cx - 300) else id_b
                self.assertEqual(dd["id"], want, f"cross-segment identity flip at f{line['f']}")

    def test_dissolve_frames_emit_nothing(self):
        trk = SegmentedIdentityTracker(min_surface_frames=2, width=W, height=H, fps=25.0)
        self._run_segment(trk, 0, 6, pos_a=300.0, pos_b=900.0)
        got = trk.process(1000, [self._det(500.0, _E_BLEND, None)], scene_change=True)
        self.assertEqual(got, [])

    def test_short_blip_does_not_join_a_main_identity(self):
        import json
        trk = SegmentedIdentityTracker(min_surface_frames=2, width=W, height=H, fps=25.0)
        self._run_segment(trk, 0, 40, pos_a=300.0, pos_b=900.0)
        for k in range(4):
            trk.process(300 * STRIDE + k * STRIDE, [], scene_change=True)
        # New segment: only a brief denim-ish bystander (few frames).
        lines = []
        for k in range(4):
            lines += trk.process(320 * STRIDE + k * STRIDE, [self._det(600.0, _E_BLEND, (2.0, 4.0))], scene_change=False)
        lines += trk.finalize()
        parsed = [json.loads(l) for l in lines]
        blip_ids = {d["id"] for line in parsed if line["f"] >= 320 * STRIDE for d in line["dets"]}
        main_ids = {d["id"] for line in parsed if line["f"] < 300 * STRIDE for d in line["dets"]}
        self.assertTrue(blip_ids.isdisjoint(main_ids), f"blip merged into a main id: {blip_ids & main_ids}")


if __name__ == "__main__":
    unittest.main()
