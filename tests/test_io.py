"""Settings, progress events, run status and on-disk formats."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinesia import masks
from kinesia.pipeline import Status, parse_events
from kinesia.settings import parse_env_file
from kinesia.scene.export import SectionWriter
from kinesia.tracks import FrameTracks, TracksWriter, read_tracks
from kinesia.video import VideoInfo, normalization_plan


class SettingsTest(unittest.TestCase):
    def test_parse_env_file(self):
        text = '# comment\nA=1\nB = "two words"\n\nC=\'x=y\'\nnot a line\n'
        self.assertEqual(parse_env_file(text), {"A": "1", "B": "two words", "C": "x=y"})


class EventsTest(unittest.TestCase):
    def test_parse_events_ignores_other_output(self):
        log = "\n".join(
            [
                "INFO loading",
                'KINESIA {"event": "progress", "stage": "tracking", "done": 3, "total": 9}',
                "KINESIA {broken",
                '\x1b[0mKINESIA {"event": "done"}',
            ]
        )
        self.assertEqual([e["event"] for e in parse_events(log)], ["progress", "done"])


class StatusTest(unittest.TestCase):
    def test_a_cancel_wins_over_a_running_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            worker = Status(path)
            worker.update("running", "Tracking")
            Status(path).update("cancelled", "Cancelled")  # the web app cancels
            worker.update("running", "Tracking 50%")  # the worker reports once more
            self.assertEqual(json.loads(path.read_text())["state"], "cancelled")
            self.assertTrue(worker.cancelled)

    def test_a_new_worker_can_restart_a_cancelled_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            Status(path).update("cancelled", "Cancelled")
            restart = Status(path)
            restart.opened += 1.0  # started after the cancel
            restart.update("preparing", "Preparing")
            self.assertEqual(json.loads(path.read_text())["state"], "preparing")


class MaskAndTrackTest(unittest.TestCase):
    def test_clean_box_ignores_specks(self):
        mask = np.zeros((100, 100), bool)
        mask[20:80, 30:60] = True
        mask[2, 97] = True  # one stray pixel far away
        self.assertEqual(masks.clean_box(mask), (30.0, 20.0, 60.0, 80.0))

    def test_rle_matches_the_reference_encoder(self):
        from pycocotools import mask as coco_mask

        rng = np.random.default_rng(3)
        cases = [np.zeros((40, 30), bool), np.ones((40, 30), bool)]
        edge = np.zeros((40, 30), bool)
        edge[0, 0] = edge[39, 29] = True  # first and last pixel
        cases.append(edge)
        for _ in range(60):
            mask = np.zeros((40, 30), bool)
            y0, x0 = rng.integers(0, 40), rng.integers(0, 30)
            mask[y0 : y0 + rng.integers(1, 30), x0 : x0 + rng.integers(1, 20)] = True
            mask &= rng.random(mask.shape) > 0.2  # holes and specks
            cases.append(mask)
        for mask in cases:
            reference = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
            self.assertEqual(masks.encode(mask), reference)
            self.assertTrue(np.array_equal(masks.decode(masks.encode(mask), 40, 30), mask))

    def test_rle_round_trip_and_outline(self):
        mask = np.zeros((64, 48), bool)
        mask[10:50, 5:30] = True
        counts = masks.encode(mask)
        self.assertTrue(np.array_equal(masks.decode(counts, 64, 48), mask))
        self.assertEqual(masks.area(counts, 64, 48), int(mask.sum()))
        polygon = masks.outline(mask)[0]
        xs, ys = polygon[0::2], polygon[1::2]
        self.assertEqual((min(xs), max(xs), min(ys), max(ys)), (5, 29, 10, 49))

    def test_tracks_file_round_trip(self):
        mask = np.zeros((20, 30), bool)
        mask[5:15, 10:20] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl.gz"
            writer = TracksWriter(path, {"width": 30, "height": 20, "fps": 30.0})
            writer.write(FrameTracks(0, [4], [0.93], [[10, 5, 20, 15]], [masks.encode(mask)]))
            writer.write(FrameTracks(1, [], [], [], []))
            writer.close()
            tracks = read_tracks(path)
        self.assertEqual((tracks.width, tracks.height, len(tracks.frames)), (30, 20, 2))
        self.assertEqual(tracks.track_ids(), [4])
        self.assertTrue(np.array_equal(tracks.frames[0].mask(0, 20, 30), mask))


class ExportTest(unittest.TestCase):
    def test_sections_are_aligned_and_described(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SectionWriter(Path(tmp) / "data.bin")
            a = writer.add(np.arange(3), "uint8")
            b = writer.add(np.arange(6).reshape(2, 3), "float32")
            writer.close()
            data = (Path(tmp) / "data.bin").read_bytes()
        self.assertEqual(b["offset"] % 8, 0)
        self.assertEqual(b["shape"], [2, 3])
        values = np.frombuffer(data, dtype="<f4", count=6, offset=b["offset"])
        self.assertTrue(np.array_equal(values, np.arange(6)))
        self.assertEqual(json.loads(json.dumps(a))["type"], "uint8")


class VideoPlanTest(unittest.TestCase):
    def test_high_frame_rates_are_decimated_by_whole_steps(self):
        self.assertAlmostEqual(normalization_plan(VideoInfo(1920, 1080, 119.88, 7000, 58.6))[2], 29.97, places=2)
        self.assertEqual(normalization_plan(VideoInfo(1920, 1080, 50.0, 100, 2.0))[2], 25.0)
        self.assertEqual(normalization_plan(VideoInfo(1280, 720, 25.0, 100, 4.0)), (1280, 720, 25.0))

    def test_large_videos_are_scaled_to_even_sizes(self):
        width, height, _ = normalization_plan(VideoInfo(3840, 2160, 30.0, 10, 1.0))
        self.assertEqual((width, height), (1920, 1080))
        width, height, _ = normalization_plan(VideoInfo(2000, 1125, 30.0, 10, 1.0))
        self.assertEqual((width % 2, height % 2), (0, 0))


if __name__ == "__main__":
    unittest.main()


class TrackingMemoryTest(unittest.TestCase):
    def test_yielded_scores_are_forgotten_but_pending_ones_kept(self):
        from collections import defaultdict
        from types import SimpleNamespace

        from kinesia.inference.tracking import _forget_yielded_scores

        history = defaultdict(dict, {f: {1: 0.9} for f in range(40)})
        state = {"tracker_metadata": {"obj_id_to_sam2_score_frame_wise": history}}
        predictor = SimpleNamespace(_get_session=lambda session: {"state": state})
        _forget_yielded_scores(predictor, "s", 10)  # frame 10 just yielded, 11-39 still buffered
        self.assertEqual(sorted(history), list(range(10, 40)))

    def test_memory_guard_stops_a_clip_that_cannot_fit(self):
        from unittest import mock

        import torch

        from kinesia.inference.tracking import MemoryGuard

        allocated = {"bytes": 0}
        with mock.patch.object(torch.cuda, "get_device_properties", return_value=mock.Mock(total_memory=80e9)), \
             mock.patch.object(torch.cuda, "memory_allocated", side_effect=lambda: allocated["bytes"]):  # fmt: skip
            fits = MemoryGuard(frame_count=2000)
            for done in range(1, 2001):
                allocated["bytes"] = 10e9 + done * 20e6  # 20 MB per frame: 50 GB at the end
                fits(done)
            self.assertAlmostEqual(fits.per_frame, 20e6, delta=1e3)
            too_long = MemoryGuard(frame_count=6000)  # 130 GB at the end
            with self.assertRaisesRegex(RuntimeError, "GB of GPU memory"):
                for done in range(1, 6001):
                    allocated["bytes"] = 10e9 + done * 20e6
                    too_long(done)
            self.assertLess(done, 200)  # stopped early, not near the end
