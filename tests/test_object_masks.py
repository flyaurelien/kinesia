"""Tests for category-agnostic MLX SAM 3 scene segmentation."""

from __future__ import annotations

import unittest

import numpy as np

from sam_3d_pose_estimation.object_masks import (
    segment_prompt_instances,
    select_mask_for_reference,
)


class FakeProcessor:
    """Minimal processor that records image and prompt model calls."""

    def __init__(self) -> None:
        self.confidence_threshold = 0.0
        self.image_calls = 0
        self.prompts: list[str] = []

    def set_image(self, _image: object) -> dict[str, object]:
        self.image_calls += 1
        return {"image": True}

    def set_text_prompt(self, prompt: str, state: dict[str, object]) -> dict[str, object]:
        self.prompts.append(prompt)
        offset = len(self.prompts)
        mask = np.zeros((1, 1, 12, 16), dtype=np.uint8)
        mask[0, 0, offset:offset + 3, offset:offset + 4] = 1
        return {**state, "scores": np.array([0.6 + offset / 10]), "masks": mask}


class ObjectMaskTests(unittest.TestCase):
    def test_several_prompts_share_one_mlx_image_encoding(self) -> None:
        processor = FakeProcessor()

        results = segment_prompt_instances(
            np.zeros((12, 16, 3), dtype=np.uint8),
            ("first arbitrary object", "second arbitrary object"),
            processor,
        )

        self.assertEqual(processor.image_calls, 1)
        self.assertEqual(
            processor.prompts,
            ["first arbitrary object", "second arbitrary object"],
        )
        self.assertEqual(set(results), set(processor.prompts))
        self.assertEqual(results["first arbitrary object"][0][0].dtype, np.bool_)

    def test_reference_selection_uses_geometry_not_instance_order(self) -> None:
        reference = np.zeros((20, 20), dtype=bool)
        reference[4:12, 5:13] = True
        wrong = np.zeros_like(reference)
        wrong[12:18, 12:18] = True
        matching = reference.copy()

        selected = select_mask_for_reference(
            [(wrong, 0.99), (matching, 0.55)], reference
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertTrue(np.array_equal(selected[0], matching))


if __name__ == "__main__":
    unittest.main()
