"""Regression tests for metric trajectories of known-size spherical objects."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from sam_3d_pose_estimation.dynamic_objects import (
    SphereCandidate,
    SphereIdentityTracker,
    assess_camera_stability,
    classify_camera_motion_samples,
    derive_velocities,
    effective_focal_from_subject_records,
    sphere_measurement_from_mask,
)


def disk_mask(
    width: int = 200,
    height: int = 160,
    center: tuple[int, int] = (100, 80),
    radius: int = 20,
) -> np.ndarray:
    """Build a binary circular mask with OpenCV, matching detector mask layout."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask.astype(bool)


class SphereMeasurementTests(unittest.TestCase):
    """Known-radius perspective lifting remains numerically and axis consistent."""

    def test_lifts_centered_sphere_to_expected_depth(self) -> None:
        measurement = sphere_measurement_from_mask(
            disk_mask(radius=20),
            focal_px=200.0,
            image_width=200,
            image_height=160,
            diameter_m=0.4,
            detection_score=0.9,
        )

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement["position_cam"][0], 0.0, places=3)
        self.assertAlmostEqual(measurement["position_cam"][1], 0.0, places=3)
        self.assertAlmostEqual(measurement["position_cam"][2], 2.0, places=2)
        self.assertAlmostEqual(measurement["position_world"][0], -2.0, places=2)
        self.assertGreater(measurement["circularity"], 0.9)

    def test_preserves_camera_to_world_axes_off_centre(self) -> None:
        measurement = sphere_measurement_from_mask(
            disk_mask(center=(120, 90), radius=20),
            focal_px=200.0,
            image_width=200,
            image_height=160,
            diameter_m=0.4,
            detection_score=1.0,
        )

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement["position_cam"][0], 0.2, places=2)
        self.assertAlmostEqual(measurement["position_cam"][1], 0.1, places=2)
        self.assertAlmostEqual(measurement["position_world"][0], -2.0, places=2)
        self.assertAlmostEqual(measurement["position_world"][1], 0.2, places=2)
        self.assertAlmostEqual(measurement["position_world"][2], -0.1, places=2)

    def test_rejects_unusable_mask(self) -> None:
        result = sphere_measurement_from_mask(
            np.zeros((20, 20), dtype=bool),
            focal_px=200.0,
            image_width=20,
            image_height=20,
            diameter_m=0.4,
            detection_score=1.0,
        )

        self.assertIsNone(result)


class SphereIdentityTests(unittest.TestCase):
    """The dynamic tracker favors temporal continuity over a distant distractor."""

    @staticmethod
    def candidate(center: tuple[float, float], score: float) -> SphereCandidate:
        return SphereCandidate(
            mask=disk_mask(center=(int(center[0]), int(center[1]),), radius=12),
            detector_score=score,
            center_px=np.asarray(center, dtype=np.float64),
            radius_px=12.0,
            circularity=1.0,
        )

    def test_keeps_nearby_identity_over_higher_scoring_distractor(self) -> None:
        tracker = SphereIdentityTracker()
        first = self.candidate((40.0, 60.0), 0.70)
        self.assertIs(tracker.choose([first], frame_index=0, image_width=200, image_height=160), first)

        nearby = self.candidate((48.0, 60.0), 0.68)
        distant = self.candidate((180.0, 140.0), 0.95)
        selected = tracker.choose(
            [distant, nearby], frame_index=1, image_width=200, image_height=160,
        )

        self.assertIs(selected, nearby)


class VelocityTests(unittest.TestCase):
    """Finite differences retain physical units and handle endpoint samples."""

    def test_derives_velocity_in_metres_per_second(self) -> None:
        poses = [
            {"time_s": 0.0, "position_world": [0.0, 0.0, 0.0]},
            {"time_s": 0.5, "position_world": [1.0, 0.0, 0.0]},
            {"time_s": 1.0, "position_world": [2.0, 0.0, 0.0]},
        ]

        derive_velocities(poses)

        self.assertEqual(poses[0]["velocity_world_m_s"], [2.0, 0.0, 0.0])
        self.assertEqual(poses[1]["velocity_world_m_s"], [2.0, 0.0, 0.0])
        self.assertEqual(poses[2]["velocity_world_m_s"], [2.0, 0.0, 0.0])


class DynamicSceneContractTests(unittest.TestCase):
    """Calibration and fixed-camera guards protect shared-world trajectories."""

    def test_uses_the_subject_floor_calibration_when_available(self) -> None:
        records = []
        for index in range(30):
            # The robust floor quantile sees the deeper 1.2 m contact, while
            # the median foot relation is 1.0 m. This makes the effective
            # focal deliberately differ from the model's recorded value.
            foot_y = 0.8 if index < 15 else 1.2
            joints = [[0.0, foot_y, 4.0] for _ in range(21)]
            records.append({"focal_length": 300.0, "joints_cam_xyz": joints})

        focal, evidence = effective_focal_from_subject_records(records, image_height=240)

        self.assertEqual(evidence["method"], "subject_feet")
        self.assertAlmostEqual(focal, 250.0, places=6)
        self.assertAlmostEqual(evidence["floor_z_m"], -1.2, places=6)

    def test_rejects_significant_global_camera_motion(self) -> None:
        result = classify_camera_motion_samples(
            [(0.0022, 0.01, 0.0002)] * 8,
        )

        self.assertEqual(result["status"], "moving")

    def test_keeps_stable_camera_samples_eligible(self) -> None:
        result = classify_camera_motion_samples(
            [(0.00005, 0.01, 0.0001)] * 8,
        )

        self.assertEqual(result["status"], "fixed")

    def test_detects_global_translation_from_video_features(self) -> None:
        class SyntheticCapture:
            def __init__(self, frames: list[np.ndarray]) -> None:
                self.frames = frames
                self.index = 0

            def get(self, property_id: int) -> float:
                if property_id == cv2.CAP_PROP_FPS:
                    return 4.0
                if property_id == cv2.CAP_PROP_FRAME_COUNT:
                    return float(len(self.frames))
                return 0.0

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index]
                self.index += 1
                return True, frame.copy()

            def release(self) -> None:
                return None

        source = np.zeros((120, 180, 3), dtype=np.uint8)
        for y in range(12, 120, 24):
            for x in range(12, 180, 24):
                cv2.circle(source, (x, y), 4, (255, 255, 255), thickness=-1)
        frames = [
            cv2.warpAffine(
                source,
                np.float32([[1.0, 0.0, 2.0 * index], [0.0, 1.0, 0.0]]),
                (180, 120),
            )
            for index in range(12)
        ]

        with mock.patch(
            "sam_3d_pose_estimation.dynamic_objects.open_video",
            return_value=SyntheticCapture(frames),
        ):
            result = assess_camera_stability(Path("synthetic.mp4"))

        self.assertEqual(result["status"], "moving")
        self.assertGreater(result["median_translation_diagonal_ratio_per_frame"], 0.001)
