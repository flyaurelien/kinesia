"""Regression tests for the static-object geometry contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import trimesh

from sam_3d_pose_estimation.scene_objects import (
    FOOT_JOINTS,
    align_body_to_scene_pointmap,
    apply_scene_alignment_to_model_pose,
    cam_to_world,
    calibrate_model_pose_to_subject_frame,
    canonical_mesh_frame,
    fit_pose_to_silhouette,
    floor_height_for_run,
    model_pose_to_world_matrix,
    place_built_shapes,
    place_object,
    rasterize_mesh_silhouette,
    resolve_floor_object_body_penetration,
    transform_points,
    upright_flip_matrix,
    upright_rotation,
    world_from_mesh_matrix,
)


def _asymmetric_vertices() -> np.ndarray:
    """A deliberately non-box-shaped mesh, so pivot bugs cannot cancel out."""
    return np.array(
        [
            [-1.8, -0.7, -0.5],
            [2.7, -0.5, -0.5],
            [0.8, 1.9, -0.5],
            [-1.1, 0.9, -0.5],
            [-0.8, -0.4, 2.4],
            [1.1, -0.2, 1.7],
            [0.4, 1.4, 1.2],
            [-0.3, 0.6, 0.8],
        ],
        dtype=np.float64,
    )


def _dense_box(
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    samples: int,
) -> np.ndarray:
    """Return dense surface points for the sparse silhouette rasteriser."""
    xs = np.linspace(low[0], high[0], samples)
    ys = np.linspace(low[1], high[1], samples)
    zs = np.linspace(low[2], high[2], samples)
    points: list[list[float]] = []
    for z in zs:
        for y in ys:
            points.extend([[xs[0], y, z], [xs[-1], y, z]])
        for x in xs:
            points.extend([[x, ys[0], z], [x, ys[-1], z]])
    for x in xs:
        for y in ys:
            points.extend([[x, y, zs[0]], [x, y, zs[-1]]])
    return np.unique(np.asarray(points, dtype=np.float64), axis=0)


def _asymmetric_object_cloud() -> np.ndarray:
    """An L-shaped object with a heading visible in its silhouette."""
    return np.vstack(
        [
            _dense_box((-1.0, -0.45, 0.0), (0.65, 0.30, 1.55), 10),
            _dense_box((0.40, -0.30, 1.10), (1.25, 0.10, 2.20), 8),
        ]
    )


def _homogeneous(rotation: np.ndarray) -> np.ndarray:
    """Embed a rotation in a test-only homogeneous matrix."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix


def _angle_error(left: float, right: float) -> float:
    """Smallest angular separation in radians."""
    return abs(float(np.arctan2(np.sin(left - right), np.cos(left - right))))


class TestSceneObjectTransforms(unittest.TestCase):
    def test_model_pose_converts_runtime_camera_and_glb_axes_to_world(self):
        matrix = model_pose_to_world_matrix({
            "translation_l2c": [[1.0, 2.0, 3.0]],
            "rotation_quaternion_wxyz_l2c": [[1.0, 0.0, 0.0, 0.0]],
            "scale_l2c": [[2.0, 3.0, 4.0]],
        })
        np.testing.assert_allclose(matrix[:3, 3], [-3.0, -1.0, 2.0])
        np.testing.assert_allclose(matrix[:3, :3], np.array([
            [0.0, -4.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, -3.0],
        ]))

    def test_model_pose_transposes_the_runtime_row_vector_rotation(self):
        half_sqrt = float(np.sqrt(0.5))
        matrix = model_pose_to_world_matrix({
            "translation_l2c": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz_l2c": [half_sqrt, 0.0, 0.0, half_sqrt],
            "scale_l2c": [1.0, 1.0, 1.0],
        })

        # SAM 3D Objects uses PyTorch3D's row-vector convention. Its +90°
        # local Z rotation sends raw [1, 0, 0] to camera [0, -1, 0]. The GLB
        # conversion leaves this particular basis vector unchanged.
        np.testing.assert_allclose(
            transform_points(matrix, np.array([[1.0, 0.0, 0.0]])),
            [[0.0, 0.0, -1.0]],
            atol=1e-12,
        )

    def test_model_pose_requires_the_shared_scene_alignment(self):
        pose = {
            "translation_l2c": [0.0, -0.5, 2.0],
            "rotation_quaternion_wxyz_l2c": [1.0, 0.0, 0.0, 0.0],
            "scale_l2c": [1.0, 1.0, 1.0],
        }
        with self.assertRaisesRegex(ValueError, "shared Body/Object"):
            calibrate_model_pose_to_subject_frame(pose, None)  # type: ignore[arg-type]

    def test_shared_scene_alignment_sets_scale_without_changing_proportions(self):
        pose = {
            "translation_l2c": [0.1, -0.4, 2.0],
            "rotation_quaternion_wxyz_l2c": [1.0, 0.0, 0.0, 0.0],
            "scale_l2c": [0.5, 0.5, 0.5],
        }
        direct = model_pose_to_world_matrix(pose)
        alignment = {
            "method": "official_body_to_moge_height_center",
            "scale": 2.4,
            "translation_camera": [0.1, -0.2, 0.7],
        }

        matrix, calibration = calibrate_model_pose_to_subject_frame(
            pose, alignment
        )

        self.assertEqual(calibration["method"], "shared_body_object_scene")
        self.assertEqual(calibration["scene_alignment"], alignment)
        np.testing.assert_allclose(
            matrix[:3, :3] / alignment["scale"],
            direct[:3, :3],
            atol=1e-12,
        )
        self.assertEqual(calibration["floor_offset_m"], 0.0)

    def test_model_pose_rejects_an_invalid_rotation(self):
        with self.assertRaisesRegex(ValueError, "rotation quaternion"):
            model_pose_to_world_matrix({
                "translation_l2c": [0.0, 0.0, 0.0],
                "rotation_quaternion_wxyz_l2c": [0.0, 0.0, 0.0, 0.0],
                "scale_l2c": [1.0, 1.0, 1.0],
            })

    def test_run_anchor_is_the_floor_reference_for_new_object_placements(self):
        joints = [None] * 21
        for index in FOOT_JOINTS:
            joints[index] = [0.0, 1.04, 4.0]
        metadata = {
            "space_view": {"world_anchor": {"floor_y": 0.91}},
            "records": [{"joints_cam_xyz": joints}],
        }

        floor_z, reference = floor_height_for_run(metadata, metadata["records"])

        self.assertEqual(reference, "world_anchor")
        self.assertAlmostEqual(floor_z, -0.91)

    def test_canonical_frame_has_a_right_handed_up_axis_and_mean_pivot(self):
        vertices = _asymmetric_vertices()
        bbox_centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0

        for up_axis in range(3):
            with self.subTest(up_axis=up_axis):
                rotation, pivot = canonical_mesh_frame(vertices, up_axis)
                canonical = (vertices - pivot) @ rotation.T

                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
                np.testing.assert_allclose(canonical[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-12)
                self.assertAlmostEqual(float(canonical[:, 2].min()), 0.0, places=12)
                # The chosen contact pivot is intentionally not the bbox centre.
                self.assertFalse(np.allclose(pivot, bbox_centre))

    def test_flipped_raw_mesh_and_exported_mesh_have_the_same_world_pose(self):
        raw_vertices = _asymmetric_vertices()
        position = np.array([-4.2, 0.35, -1.1], dtype=np.float64)

        for up_axis in range(3):
            with self.subTest(up_axis=up_axis):
                flip = upright_flip_matrix(up_axis, flipped=True)
                self.assertAlmostEqual(float(np.linalg.det(flip)), 1.0, places=12)
                np.testing.assert_allclose(flip @ flip, np.eye(3), atol=1e-12)

                exported_vertices = raw_vertices @ flip.T
                object_to_world, exported_pivot = world_from_mesh_matrix(
                    exported_vertices,
                    up_axis,
                    scale=0.42,
                    position_world=position,
                    yaw_rad=0.47,
                )
                raw_to_world = object_to_world @ _homogeneous(flip)
                raw_pivot = flip.T @ exported_pivot

                np.testing.assert_allclose(
                    transform_points(raw_to_world, raw_vertices),
                    transform_points(object_to_world, exported_vertices),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    transform_points(raw_to_world, raw_pivot.reshape(1, 3))[0],
                    position,
                    atol=1e-12,
                )

    def test_transform_places_every_up_axis_on_the_same_floor(self):
        vertices = _asymmetric_vertices()
        position = np.array([-3.8, -0.25, -0.85], dtype=np.float64)
        bbox_centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0

        for up_axis in range(3):
            with self.subTest(up_axis=up_axis):
                object_to_world, pivot = world_from_mesh_matrix(
                    vertices,
                    up_axis,
                    scale=0.31,
                    position_world=position,
                    yaw_rad=-0.71,
                )
                world_vertices = transform_points(object_to_world, vertices)

                np.testing.assert_allclose(
                    transform_points(object_to_world, pivot.reshape(1, 3))[0],
                    position,
                    atol=1e-12,
                )
                self.assertFalse(np.allclose(
                    transform_points(object_to_world, bbox_centre.reshape(1, 3))[0],
                    position,
                ))
                self.assertAlmostEqual(float(world_vertices[:, 2].min()), position[2], places=12)
                self.assertTrue(np.all(world_vertices[:, 2] >= position[2] - 1e-12))

    def test_silhouette_fit_recovers_yaw_and_reprojects_an_asymmetric_object(self):
        canonical_vertices = _asymmetric_object_cloud()
        up_axis = 0
        # Invert the canonical rotation to make a source mesh whose X axis is up.
        source_vertices = canonical_vertices @ upright_rotation(up_axis)
        position = np.array([-4.6, 0.36, -1.2], dtype=np.float64)
        yaw = 0.62
        scale = 0.55
        mask = rasterize_mesh_silhouette(
            source_vertices,
            up_axis,
            scale,
            position,
            yaw,
            focal=300.0,
            width=320,
            height=240,
        )

        fit = fit_pose_to_silhouette(
            source_vertices,
            up_axis,
            scale,
            mask,
            focal=300.0,
            floor_z=position[2],
            width=320,
            height=240,
        )
        fitted_mask = rasterize_mesh_silhouette(
            source_vertices,
            up_axis,
            scale,
            fit["position_world"],
            fit["yaw_rad"],
            focal=300.0,
            width=320,
            height=240,
        )
        union = np.logical_or(mask, fitted_mask).sum()
        reprojection_iou = float(np.logical_and(mask, fitted_mask).sum() / union)

        self.assertGreater(fit["iou"], 0.95)
        self.assertGreater(reprojection_iou, 0.95)
        self.assertLess(_angle_error(fit["yaw_rad"], yaw), 0.05)
        np.testing.assert_allclose(fit["position_world"][:2], position[:2], atol=0.05)

    def test_place_object_serializes_the_canonical_transform_and_quality_evidence(self):
        raw_vertices = _asymmetric_vertices()
        up_axis = 1
        flip = upright_flip_matrix(up_axis, flipped=True)
        exported_vertices = raw_vertices @ flip.T
        fitted_position = [-3.7, 0.28, -1.0]
        fit = {
            "iou": 0.42,
            "seed_iou": 0.21,
            "position_world": fitted_position,
            "yaw_rad": 0.38,
            "mesh_pivot_local": [0.0, 0.0, 0.0],
        }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            joints = [None] * 21
            for index in FOOT_JOINTS:
                joints[index] = [0.0, 1.0, 4.0]
            (run_dir / "run_metadata.json").write_text(json.dumps({
                "video_width": 320,
                "video_height": 240,
                "records": [{"focal_length": 300.0, "joints_cam_xyz": joints}],
            }))
            source_mesh_path = run_dir / "source.ply"
            trimesh.Trimesh(
                vertices=raw_vertices,
                faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
                process=False,
            ).export(source_mesh_path)
            mask_path = run_dir / "mask.png"
            mask = np.zeros((240, 320), dtype=np.uint8)
            mask[90:190, 125:205] = 255
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            with mock.patch(
                "sam_3d_pose_estimation.scene_objects._fit_upright",
                return_value=(fit, up_axis, 0.35, exported_vertices, True),
            ):
                record = place_object(run_dir, source_mesh_path, mask_path, name="fixture", log=lambda _: None)
            written_mesh = trimesh.load(run_dir / "scene" / "fixture.glb", force="mesh")
            written_vertices = np.asarray(written_mesh.vertices, dtype=np.float64).copy()

        object_to_world = np.asarray(record["object_to_world"], dtype=np.float64)
        raw_to_world = np.asarray(record["raw_mesh_to_world"], dtype=np.float64)
        self.assertEqual(object_to_world.shape, (4, 4))
        self.assertEqual(record["transform_contract"]["matrix_layout"], "row_major")
        self.assertEqual(record["transform_contract"]["vector_convention"], "column_vector")
        self.assertEqual(record["orientation"], {"up_axis": "Y", "flipped": True})
        self.assertEqual(record["quality"]["floor_calibration"], "recorded_focal")
        self.assertEqual(record["transform_validation"]["status"], "valid")
        self.assertIn("scale", record)  # Legacy fields remain available.
        self.assertIn("centre_world", record)
        self.assertIn("position_world", record)
        np.testing.assert_allclose(
            transform_points(object_to_world, written_vertices),
            transform_points(raw_to_world, raw_vertices),
            atol=1e-12,
        )

    def test_every_static_object_receives_the_exact_same_scene_alignment(self):
        alignment = {"scale": 2.0, "translation_camera": [0.0, 0.0, 0.5]}
        evidence = {
            "method": "official_body_to_moge_height_center",
            "source_frame_shared": True,
            "scene_to_body": alignment,
            "normalized_rms": 0.01,
            "rms_m": 0.02,
        }
        received: list[dict] = []

        def place(*args: object, **_kwargs: object) -> dict:
            received.append(args[4])
            return {"schema": "fixture", "name": args[3]}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            scene_dir = run_dir / "scene"
            scene_dir.mkdir()
            pending = scene_dir / ".pending"
            pending.mkdir()
            for name in ("first_shape", "second_shape"):
                (pending / f"{name}.json").write_text(json.dumps({
                    "name": name,
                    "prompt": name,
                    "source_frame": 17,
                    "model_pose": f"{name}_model_pose.json",
                }))
                (scene_dir / f"{name}_model_pose.json").write_text("{}")
            with (
                mock.patch(
                    "sam_3d_pose_estimation.scene_objects._build_shared_scene_alignment",
                    return_value=(
                        alignment,
                        evidence,
                        mock.sentinel.subject_mesh,
                        200.0,
                        np.zeros((80, 100), dtype=bool),
                    ),
                ),
                mock.patch(
                    "sam_3d_pose_estimation.scene_objects.place_model_pose_object",
                    side_effect=place,
                ),
            ):
                result = place_built_shapes(run_dir, log=lambda _: None)

        self.assertEqual(result["failures"], [])
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual(received, [alignment, alignment])


class TestSharedSceneAlignment(unittest.TestCase):
    def test_recovers_official_height_and_center_body_alignment(self):
        grid_size = 30
        x, y = np.meshgrid(
            np.linspace(-0.8, 0.8, grid_size),
            np.linspace(-1.0, 1.0, grid_size),
        )
        body_camera = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.0)))
        body_world = np.asarray([cam_to_world(point) for point in body_camera])
        faces = []
        for row in range(grid_size - 1):
            for column in range(grid_size - 1):
                top_left = row * grid_size + column
                faces.extend([
                    [top_left, top_left + 1, top_left + grid_size],
                    [top_left + 1, top_left + grid_size + 1, top_left + grid_size],
                ])
        subject = trimesh.Trimesh(
            vertices=body_world,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        body_to_scene_scale = 1.7
        body_to_scene_translation = np.array([0.2, -0.3, 0.8])
        pointmap = np.full((240, 320, 3), np.nan, dtype=np.float64)
        subject_mask = np.zeros((240, 320), dtype=bool)
        for pixel_y in range(80, 161):
            for pixel_x in range(128, 193):
                body_point = np.array([
                    (pixel_x - 160.0) * 5.0 / 200.0,
                    (pixel_y - 120.0) * 5.0 / 200.0,
                    5.0,
                ])
                scene_point = body_to_scene_scale * body_point + body_to_scene_translation
                pointmap[pixel_y, pixel_x] = scene_point * np.array([-1.0, -1.0, 1.0])
                subject_mask[pixel_y, pixel_x] = True

        alignment = align_body_to_scene_pointmap(
            pointmap,
            subject,
            focal=200.0,
            subject_mask=subject_mask,
            image_width=320,
            image_height=240,
        )

        self.assertEqual(alignment["method"], "official_body_to_moge_height_center")
        self.assertAlmostEqual(alignment["body_to_scene"]["scale"], body_to_scene_scale, places=2)
        np.testing.assert_allclose(
            alignment["body_to_scene"]["translation_camera"],
            body_to_scene_translation,
            atol=0.08,
        )
        self.assertLess(alignment["normalized_rms"], 0.04)


class TestSubjectInteractionResolution(unittest.TestCase):
    def test_overlap_is_reported_without_modifying_the_calibrated_pose(self):
        subject = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        subject.apply_translation((-4.0, 0.0, 0.0))
        object_mesh = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
        object_to_world = np.eye(4, dtype=np.float64)
        object_to_world[:3, 3] = [-4.0, 0.0, 0.0]

        corrected, evidence = resolve_floor_object_body_penetration(
            object_to_world,
            object_mesh,
            subject,
            np.zeros((240, 320), dtype=bool),
            focal=200.0,
        )

        self.assertEqual(evidence["status"], "overlap_detected")
        self.assertGreater(evidence["penetrating_surface_fraction_before"], 0.5)
        self.assertEqual(
            evidence["penetrating_surface_fraction_after"],
            evidence["penetrating_surface_fraction_before"],
        )
        self.assertTrue(evidence["pose_preserved"])
        self.assertFalse(evidence["pose_modified"])
        self.assertEqual(evidence["scale_factor"], 1.0)
        self.assertEqual(evidence["translation_world_m"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(corrected, object_to_world)

    def test_clear_floor_object_is_left_at_its_model_position(self):
        subject = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        subject.apply_translation((-4.0, 0.0, 0.0))
        object_mesh = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
        object_to_world = np.eye(4, dtype=np.float64)
        object_to_world[:3, 3] = [-4.0, 2.0, 0.0]

        corrected, evidence = resolve_floor_object_body_penetration(
            object_to_world,
            object_mesh,
            subject,
            np.zeros((240, 320), dtype=bool),
            focal=200.0,
        )

        self.assertEqual(evidence["status"], "clear")
        self.assertEqual(evidence["translation_world_m"], [0.0, 0.0, 0.0])
        self.assertEqual(evidence["scale_factor"], 1.0)
        np.testing.assert_allclose(corrected, object_to_world)


if __name__ == "__main__":
    unittest.main()
