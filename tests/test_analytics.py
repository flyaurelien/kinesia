from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from sam_3d_pose_estimation.analytics import AnalysisParams, analyze_run, stabilize_xy
from sam_3d_pose_estimation.artifacts import build_run_manifest, write_json
from sam_3d_pose_estimation.workspace import run_dir


def world_to_cam(x: float, y: float, z: float) -> list[float]:
    """Map a world point to the camera frame the pipeline expects (X right, Y down, Z forward)."""
    return [y, -z, -x]


def build_record(mesh_path: Path, frame_index: int, fps: float) -> dict:
    """Synthesize one per-frame pipeline record of a subject walking with a brief freeze around frames 12-18."""
    freeze = 12 <= frame_index <= 18
    pelvis_x = (0.02 * frame_index) if not freeze else (0.24 + 0.003 * (frame_index - 12))
    left_hip = (pelvis_x - 0.08, 0.0, 1.05)
    right_hip = (pelvis_x + 0.08, 0.0, 1.05)
    step = 0.18 if frame_index % 2 == 0 else 0.06
    left_ankle = (pelvis_x - step, 0.0, 0.05 + 0.01 * ((frame_index + 1) % 3))
    right_ankle = (pelvis_x + step, 0.0, 0.05 + 0.01 * (frame_index % 3))
    left_shoulder = (pelvis_x - 0.10, 0.0, 1.45)
    right_shoulder = (pelvis_x + 0.10, 0.0, 1.45)
    joints_world = [[pelvis_x, 0.0, 1.2] for _ in range(21)]
    joints_world[5] = list(left_shoulder)
    joints_world[6] = list(right_shoulder)
    joints_world[9] = list(left_hip)
    joints_world[10] = list(right_hip)
    joints_world[13] = list(left_ankle)
    joints_world[14] = list(right_ankle)
    joints_world[15] = [left_ankle[0] - 0.03, 0.02, left_ankle[2]]
    joints_world[16] = [left_ankle[0] - 0.02, -0.02, left_ankle[2]]
    joints_world[17] = [left_ankle[0] + 0.04, 0.0, left_ankle[2] - 0.01]
    joints_world[18] = [right_ankle[0] + 0.03, 0.02, right_ankle[2]]
    joints_world[19] = [right_ankle[0] + 0.02, -0.02, right_ankle[2]]
    joints_world[20] = [right_ankle[0] - 0.04, 0.0, right_ankle[2] - 0.01]
    joints_cam = [world_to_cam(*joint) for joint in joints_world]
    return {
        "video_frame": frame_index,
        "mesh_path": str(mesh_path),
        "bbox_xyxy": [120.0, 80.0, 320.0, 460.0],
        "camera_motion_ok": True,
        "camera_motion_inlier_ratio": 0.82,
        "camera_motion_scale_step": 1.0,
        "camera_comp_cam_xyz": [0.0, 0.0, 0.0],
        "identity_lock_status": "locked",
        "identity_is_lost": False,
        "identity_lost_frames": 0,
        "identity_appearance_similarity": 0.96,
        "identity_stability_score": 0.92,
        "identity_reacquire_scanned": False,
        "identity_reacquire_candidates": 0,
        "focal_length": 2400.0,
        "joints_space_cam_xyz": joints_cam,
        "joints_cam_xyz": joints_cam,
    }


class AnalyticsArtifactsTest(unittest.TestCase):
    """End-to-end checks on analytics output: artifact layout and root-XY stabilization."""

    def test_analyze_run_writes_versioned_artifacts(self) -> None:
        """Run analyze_run on a synthetic walk and assert the versioned analysis artifacts and kinematics columns are emitted."""
        with tempfile.TemporaryDirectory() as tmpdir_raw:
            tmpdir = Path(tmpdir_raw)
            (tmpdir / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
            run_id = "clinical_case_001"
            run_path = run_dir(run_id, tmpdir)
            (run_path / "meshes").mkdir(parents=True, exist_ok=True)

            fps = 30.0
            records = []
            for frame_index in range(30):
                mesh_path = run_path / "meshes" / f"frame_{frame_index:06d}.ply"
                mesh_path.write_text("ply\n", encoding="utf-8")
                records.append(build_record(mesh_path, frame_index, fps))

            metadata = {
                "video_input": str(tmpdir / "input.mp4"),
                "output_video": str(run_path / "patient_mesh_preview.mp4"),
                "mesh_dir": str(run_path / "meshes"),
                "mhr_backend": "native_mps_patched",
                "inference_precision_effective": "float32",
                "mps_mhr_mode_requested": None,
                "inference_target": "body",
                "video_width": 640,
                "video_height": 480,
                "fps_output": fps,
                "total_frames_processed": len(records),
                "space_view": {
                    "mode": "fixed_world_anchor",
                    "world_anchor": {"floor_y": 0.0, "center_x": 0.0, "center_z": 0.0},
                },
                "camera_motion_compensation": {
                    "requested": True,
                    "enabled": True,
                    "shift_px_step_mean_small": 1.5,
                },
                "records": records,
            }
            manifest = build_run_manifest(
                run_id=run_id,
                run_directory=run_path,
                metadata=metadata,
            )
            write_json(run_path / "run_manifest.json", manifest)
            write_json(run_path / "run_metadata.json", metadata)

            result = analyze_run(
                run_id=run_id,
                params=AnalysisParams(),
                project_root=tmpdir,
            )

            analysis_id = result["analysis_id"]
            analysis_path = run_path / "analysis" / analysis_id
            self.assertTrue((analysis_path / "signals.json").exists())
            self.assertTrue((analysis_path / "frames.json").exists())
            self.assertTrue((analysis_path / "qa.json").exists())
            self.assertTrue((analysis_path / "kinematics.parquet").exists())

            updated_manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_manifest["latest_analysis_id"], analysis_id)
            self.assertEqual(updated_manifest["quality_summary"]["status"], "interpretable")

            table = pq.read_table(analysis_path / "kinematics.parquet")
            self.assertEqual(table.num_rows, len(records))
            self.assertIn("root.stab.x", table.column_names)
            self.assertIn("root.xy_speed", table.column_names)
            self.assertIn("gait.ankle_relative_speed", table.column_names)
            self.assertIn("turn.yaw_rate", table.column_names)
            self.assertIn("step.cadence_hz", table.column_names)
            self.assertFalse(
                any(name.startswith("fog.") for name in table.column_names),
                msg="no FoG columns should remain",
            )

    def test_stabilize_xy_reduces_root_jitter_during_double_support(self) -> None:
        """When both feet are planted, stabilization should damp the injected per-frame root jitter."""
        fps = 30.0
        frames = []
        root_world_raw = []
        for frame_index in range(24):
            jitter_x = 0.025 if frame_index % 2 == 0 else -0.021
            jitter_y = 0.012 if frame_index % 3 == 0 else -0.011
            root_world_raw.append((jitter_x, jitter_y, 0.0))
            joints_world = [[0.0, 0.0, 1.0] for _ in range(21)]
            joints_world[13] = [-0.09, 0.00, 0.04]
            joints_world[14] = [0.09, 0.00, 0.04]
            joints_world[15] = [-0.12, 0.02, 0.03]
            joints_world[16] = [-0.11, -0.02, 0.03]
            joints_world[17] = [-0.05, 0.00, 0.02]
            joints_world[18] = [0.12, 0.02, 0.03]
            joints_world[19] = [0.11, -0.02, 0.03]
            joints_world[20] = [0.05, 0.00, 0.02]
            frames.append({
                "joints_cam": [world_to_cam(*joint) for joint in joints_world],
            })

        stabilized = stabilize_xy(frames, root_world_raw, fps)["root_world_stabilized"]
        raw_variation = sum(
            ((root_world_raw[i][0] - root_world_raw[i - 1][0]) ** 2 + (root_world_raw[i][1] - root_world_raw[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(root_world_raw))
        )
        stabilized_variation = sum(
            ((stabilized[i][0] - stabilized[i - 1][0]) ** 2 + (stabilized[i][1] - stabilized[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(stabilized))
        )
        self.assertLess(stabilized_variation, raw_variation * 0.55)

    def test_stabilize_xy_does_not_create_turning_drift_from_foot_motion(self) -> None:
        """Cyclic foot swing (no real root translation) must not leak into a drifting stabilized root."""
        fps = 30.0
        frames = []
        root_world_raw = []
        for frame_index in range(90):
            angle = frame_index * 0.06
            root_world_raw.append((0.01 if frame_index % 2 else -0.01, 0.006 if frame_index % 3 else -0.006, 1.0))
            joints_world = [[0.0, 0.0, 1.0] for _ in range(21)]
            left_ankle = (-0.10 + 0.08 * math.sin(angle), 0.05 * math.cos(angle), 0.04)
            right_ankle = (0.10 + 0.08 * math.sin(angle + 0.4), -0.05 * math.cos(angle), 0.04)
            joints_world[13] = list(left_ankle)
            joints_world[14] = list(right_ankle)
            joints_world[15] = [left_ankle[0] - 0.03, left_ankle[1] + 0.02, 0.03]
            joints_world[16] = [left_ankle[0] - 0.02, left_ankle[1] - 0.02, 0.03]
            joints_world[17] = [left_ankle[0] + 0.04, left_ankle[1], 0.02]
            joints_world[18] = [right_ankle[0] + 0.03, right_ankle[1] + 0.02, 0.03]
            joints_world[19] = [right_ankle[0] + 0.02, right_ankle[1] - 0.02, 0.03]
            joints_world[20] = [right_ankle[0] - 0.04, right_ankle[1], 0.02]
            frames.append({"joints_cam": [world_to_cam(*joint) for joint in joints_world]})

        stabilized = stabilize_xy(frames, root_world_raw, fps)["root_world_stabilized"]
        x_range = max(root[0] for root in stabilized) - min(root[0] for root in stabilized)
        y_range = max(root[1] for root in stabilized) - min(root[1] for root in stabilized)
        self.assertLess((x_range**2 + y_range**2) ** 0.5, 0.025)


if __name__ == "__main__":
    unittest.main()
