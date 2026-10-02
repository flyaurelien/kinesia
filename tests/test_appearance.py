"""The person crops, colours and weights the appearance stage works with (no model needed)."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinesia.inference import appearance
from kinesia.scene import build


class CropTest(unittest.TestCase):
    def picture(self):
        image = np.full((400, 600, 3), 30, np.uint8)
        mask = np.zeros((400, 600), bool)
        mask[100:300, 250:330] = True  # a person, 200 px tall
        image[100:200, 250:330] = (220, 30, 30)  # red top
        image[200:300, 250:330] = (30, 30, 220)  # blue shorts
        image[100:300, 330:400] = (30, 220, 30)  # a green neighbour, inside the margin
        return image, mask, (250, 100, 330, 300)

    def test_the_person_is_alone_on_the_canvas(self):
        image, mask, box = self.picture()
        picture, inside = appearance.crop_person(image, mask, box)
        self.assertEqual(picture.shape, (*appearance.CROP, 3))
        self.assertEqual(inside.shape, appearance.CROP)
        mean = np.round(appearance.MEAN * 255).astype(np.uint8)
        self.assertTrue((picture[~inside] == mean).all())  # the neighbour is gone
        rows = np.flatnonzero(inside.any(axis=1))
        self.assertGreater(rows[-1] - rows[0], 0.85 * appearance.CROP[0])  # scaled to fill the height

    def test_colours_of_the_upper_and_lower_body(self):
        image, mask, box = self.picture()
        histogram = appearance.colour_histogram(*appearance.crop_person(image, mask, box))
        size = int(np.prod(appearance.BINS))
        upper, lower = histogram[:size], histogram[size:]
        self.assertAlmostEqual(float(upper @ upper), 1.0, places=5)  # square roots of shares
        self.assertAlmostEqual(float(lower @ lower), 1.0, places=5)
        self.assertLess(float(upper @ lower), 0.3)  # red and blue do not overlap

    def test_hidden_people_have_no_colours(self):
        picture = np.zeros((*appearance.CROP, 3), np.uint8)
        self.assertFalse(appearance.colour_histogram(picture, np.zeros(appearance.CROP, bool)).any())


class WeightsTest(unittest.TestCase):
    def test_reads_safetensors(self):
        tensors = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(4, dtype=np.float16)}
        header, blobs, offset = {}, [], 0
        for name, array in tensors.items():
            data = array.tobytes()
            header[name] = {"dtype": "F32" if array.dtype == np.float32 else "F16", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
            blobs.append(data)
            offset += len(data)
        text = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
            loaded = appearance.read_safetensors(path)
        self.assertTrue(np.array_equal(loaded["a"].numpy(), tensors["a"]))
        self.assertTrue(np.array_equal(loaded["b"].numpy(), tensors["b"]))

    def test_timm_names_map_to_dinov3_names(self):
        self.assertEqual(appearance._native_name("reg_token"), "storage_tokens")
        self.assertEqual(appearance._native_name("blocks.3.gamma_1"), "blocks.3.ls1.gamma")
        self.assertEqual(appearance._native_name("blocks.3.mlp.fc1_g.weight"), "blocks.3.mlp.w1.weight")
        self.assertEqual(appearance._native_name("blocks.3.mlp.fc1_x.bias"), "blocks.3.mlp.w2.bias")
        self.assertEqual(appearance._native_name("blocks.3.mlp.fc2.weight"), "blocks.3.mlp.w3.weight")
        self.assertEqual(appearance._native_name("blocks.3.attn.qkv.weight"), "blocks.3.attn.qkv.weight")

    def test_principal_components_keep_the_main_directions(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 2)) @ np.array([[10.0, 0, 0, 0], [0, 3.0, 0, 0]]) + 0.01 * rng.normal(size=(500, 4))
        z = appearance.principal_components(x.astype(np.float32), dims=2)
        self.assertEqual(z.shape, (500, 2))
        self.assertGreater(abs(np.corrcoef(z[:, 0], x[:, 0])[0, 1]), 0.999)


class SceneInputTest(unittest.TestCase):
    def rows(self, embed: bool) -> dict:
        rows = {
            "frame": np.array([0, 0, 1]),
            "track": np.array([3, 5, 3]),
            "box": np.array([[100, 100, 150, 300], [0, 100, 50, 300], [400, 100, 450, 130]], dtype=np.float32),
        }
        if embed:
            rows["embed"] = np.ones((3, 8), np.float16)
        return rows

    def test_features_are_matched_to_bodies_by_frame_and_track(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            np.savez(
                raw / "appearance.npz", frame=np.array([1, 0, 0]), track=np.array([3, 5, 3]),
                dino=np.arange(3, dtype=np.float16)[:, None] * np.ones((3, 4), np.float16),
                colour=np.ones((3, 6), np.float16), visible=np.array([0.6, 0.6, 0.1], np.float16),
                hidden=np.zeros(3, np.float16),
            )  # fmt: skip
            features, colour, clear = build._appearance(raw, self.rows(embed=False), 1920, 1080)
        self.assertEqual(features[:, 0].tolist(), [2.0, 1.0, 0.0])
        self.assertEqual(colour.shape, (3, 6))
        # its mask fills a tenth of its box / touching the picture's edge / too small
        self.assertEqual(clear.tolist(), [False, False, False])

    def test_older_analyses_fall_back_on_the_body_model_features(self):
        with tempfile.TemporaryDirectory() as folder:
            features, colour, clear = build._appearance(Path(folder), self.rows(embed=True), 1920, 1080)
            self.assertEqual(features.shape, (3, 8))
            self.assertIsNone(colour)
            self.assertEqual(clear.tolist(), [True, False, False])
            with self.assertRaises(FileNotFoundError):
                build._appearance(Path(folder), self.rows(embed=False), 1920, 1080)


if __name__ == "__main__":
    unittest.main()
