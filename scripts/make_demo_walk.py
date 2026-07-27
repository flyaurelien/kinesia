"""Generate a synthetic walking demo run so the viewer's gait-cycle report has
real cycles to render (all processed real runs so far are people standing).

The walker is the same ground-truth model the gait unit tests use: a subject
striding along world +Y at a known speed and cadence, with joints following
gait-typical sinusoids. Signals, QA and the gait layer are produced by the
real production code (build_analysis_payload / build_gait_analysis) — only the
joint positions and ground-truth foot contacts are synthetic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam_3d_pose_estimation.analytics import build_analysis_payload, AnalysisParams  # noqa: E402
from sam_3d_pose_estimation.artifacts import (  # noqa: E402
    build_analysis_manifest,
    build_run_manifest,
    append_analysis_to_run_manifest,
    write_json,
)
from sam_3d_pose_estimation.gait import build_gait_analysis  # noqa: E402

FPS = 30.0
STRIDE_S = 1.2
SPEED = 1.1
N_SECONDS = 14.0
RUN_ID = "demo-walk_synthetic"


L_THIGH = 0.45
L_SHANK = 0.42
STANCE_FRAC = 0.62  # heel-strike to toe-off, clinically ~60-62% of the cycle


def world_to_cam(p):
    return [p[1], -p[2], -p[0]]


def _circ_dist(frac, center):
    d = abs(frac - center)
    return min(d, 1.0 - d)


def _hip_flex_deg(frac):
    # Flexed at heel strike (~26 deg), extends through stance, flexes in swing.
    return 20.0 * math.cos(2 * math.pi * frac) + 6.0


def _knee_flex_deg(frac):
    # Small loading-response peak in early stance, large flexion peak in swing.
    loading = 15.0 * math.exp(-((_circ_dist(frac, 0.15) / 0.10) ** 2))
    swing = 58.0 * math.exp(-((_circ_dist(frac, 0.73) / 0.11) ** 2))
    return 4.0 + loading + swing


def _ankle_dorsi_deg(frac):
    # Neutral at strike, dorsiflexes through stance, plantarflexes at push-off.
    base = 6.0 * math.sin(2 * math.pi * (frac - 0.08)) - 1.0
    pushoff = -15.0 * math.exp(-((_circ_dist(frac, 0.60) / 0.055) ** 2))
    return base + pushoff


def _leg_joints(x_off, y, frac):
    """Sagittal-plane forward kinematics for one leg from target joint angles.

    Placing knee/ankle/foot from hip via these angles makes the reconstructed
    clinical angles reproduce the target curves exactly (see gait.py sign math),
    so the demo shows clinically shaped hip/knee/ankle curves.
    """
    th = math.radians(_hip_flex_deg(frac))       # thigh forward tilt
    tk = math.radians(_knee_flex_deg(frac))       # knee flexion (bends shank back)
    beta = math.radians(_ankle_dorsi_deg(frac)) + (th - tk)  # foot sagittal angle
    hip = (x_off, y, 0.95)
    knee = (x_off, y + L_THIGH * math.sin(th), 0.95 - L_THIGH * math.cos(th))
    ankle = (
        x_off,
        knee[1] + L_SHANK * math.sin(th - tk),
        knee[2] - L_SHANK * math.cos(th - tk),
    )
    fy, fz = math.cos(beta), math.sin(beta)
    toe = (x_off, ankle[1] + 0.15 * fy, ankle[2] - 0.03 + 0.15 * fz)
    heel = (x_off, ankle[1] - 0.05 * fy, ankle[2] - 0.03 - 0.05 * fz)
    return hip, knee, ankle, toe, heel


def build_records():
    rng = np.random.default_rng(11)
    n = int(N_SECONDS * FPS)
    records = []
    contacts = []
    for i in range(n):
        t = i / FPS
        y = SPEED * t
        frac_l = (t / STRIDE_S) % 1.0
        frac_r = (frac_l + 0.5) % 1.0
        joints = [None] * 21

        def sj(idx, p):
            joints[idx] = world_to_cam(p)

        sj(9, (-0.10, y, 0.95))
        sj(10, (0.10, y, 0.95))
        sj(5, (-0.16, y, 1.45))
        sj(6, (0.16, y, 1.45))
        sj(0, (0.0, y, 1.62))
        for x_off, frac, (knee_i, ankle_i, toe_i, heel_i) in (
            (-0.10, frac_l, (11, 13, 15, 17)),
            (0.10, frac_r, (12, 14, 18, 20)),
        ):
            _hip, knee, ankle, toe, heel = _leg_joints(x_off, y, frac)
            sj(knee_i, knee)
            sj(ankle_i, ankle)
            sj(toe_i, toe)
            sj(heel_i, heel)
        # The reconstruction always emits all 21 joints; fill the ones the gait
        # layer does not use (face/arms/small toes) with finite placeholders.
        sj(1, (-0.03, y, 1.66))
        sj(2, (0.03, y, 1.66))
        sj(3, (-0.07, y, 1.63))
        sj(4, (0.07, y, 1.63))
        sj(7, (-0.22, y, 1.15))
        sj(8, (0.22, y, 1.15))
        sj(16, (-0.10, y + 0.10, 0.03))
        sj(19, (0.10, y + 0.10, 0.03))
        for j, joint in enumerate(joints):
            if joint is None:
                joint = world_to_cam((0.0, y, 0.95))
            joints[j] = [v + rng.normal(0, 0.003) for v in joint]
        contact_l = frac_l < STANCE_FRAC
        contact_r = frac_r < STANCE_FRAC
        contacts.append(
            {
                "left": bool(contact_l),
                "right": bool(contact_r),
                "support": "both" if contact_l and contact_r else ("left" if contact_l else "right" if contact_r else "none"),
            }
        )
        records.append(
            {
                "video_frame": i,
                "mesh_path": "",
                "bbox_xyxy": [820.0, 300.0, 1100.0, 1040.0],
                "subject_present": True,
                "inference_status": "ok",
                "focal_length": 1400.0,
                "joints_cam_xyz": joints,
            }
        )
    return records, contacts


def main():
    project_root = Path(__file__).resolve().parents[1]
    records, contacts = build_records()
    metadata = {
        "records": records,
        "fps_output": FPS,
        "fps_input": FPS,
        "total_frames_processed": len(records),
        "video_input": "synthetic",
        "video_width": 1920,
        "video_height": 1080,
        "space_view": None,
        "subject": {"index": 0, "id": "demo", "label": "Walker", "color": "#22d3ee", "track_file": None},
    }

    params = AnalysisParams()
    payload = build_analysis_payload(run_id=RUN_ID, metadata=metadata, params=params)

    # Recompute the gait layer against ground-truth contacts so the demo has
    # guaranteed clean cycles regardless of the contact heuristic. This is the
    # same production build_gait_analysis, just fed the known contacts.
    gait_frames = []
    for i, frame in enumerate(payload["frames"]):
        gf = dict(frame)
        gf["foot_contact"] = contacts[i]
        gait_frames.append(gf)
    gait = build_gait_analysis(gait_frames, payload["fps"])
    payload["gait"] = {k: v for k, v in gait.items() if k != "angles"}

    analysis_id = f"{RUN_ID}_default_demo00000"
    run_dir = project_root / "output" / RUN_ID
    analysis_directory = run_dir / "analysis" / analysis_id
    analysis_directory.mkdir(parents=True, exist_ok=True)

    write_json(analysis_directory / "signals.json", {"analysis_id": analysis_id, "preset": params.preset, "signals": payload["signals"]})
    write_json(analysis_directory / "frames.json", {"analysis_id": analysis_id, "preset": params.preset, "fps": payload["fps"], "frames": payload["frames"]})
    write_json(analysis_directory / "qa.json", payload["qa"])
    write_json(analysis_directory / "gait.json", {"analysis_id": analysis_id, "preset": params.preset, "gait": payload["gait"]})
    analysis_manifest = build_analysis_manifest(run_id=RUN_ID, analysis_id=analysis_id, preset=params.preset, parameters=params.as_dict(), qa_summary=payload["qa"])
    write_json(analysis_directory / "analysis_manifest.json", analysis_manifest)

    manifest = build_run_manifest(run_id=RUN_ID, run_directory=run_dir, metadata=metadata)
    manifest = append_analysis_to_run_manifest(manifest, analysis_id=analysis_id, preset=params.preset, parameters=params.as_dict(), qa_summary=payload["qa"])
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "run_metadata.json", metadata)

    st = payload["gait"]["spatiotemporal"]
    total_cycles = sum(
        int((payload["gait"]["cycles"][side].get(sig, {}) or {}).get("n_cycles", 0) or 0)
        for side in ("left", "right")
        for sig in payload["gait"]["cycles"][side]
    )
    print(f"run: {RUN_ID}")
    print(f"walking_detected: {st['walking_detected']} | cadence: {st['cadence_steps_per_min']} | total cycles: {total_cycles}")
    print(f"signals: {len(payload['signals'])} | events: {len(payload['gait']['events'])}")
    print(f"stride_time mean: {st['stride_time_s']['mean']} | stride_length mean: {st['stride_length_m']['mean']} | speed mean: {st['walking_speed_m_s']['mean']}")


if __name__ == "__main__":
    main()
