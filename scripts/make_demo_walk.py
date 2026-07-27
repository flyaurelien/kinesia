"""Generate synthetic walking demo runs so the viewer's gait-cycle report has
real cycles to render (all processed real runs so far are people standing).

Each walker's legs are placed by forward kinematics from KNOWN hip/knee/ankle
angle curves, so whatever the report shows can be checked against the input.
Two subjects are written, sharing a track file so the viewer groups them as one
video: a reference walker, and a slower one with reduced knee flexion and a
longer stance — the asymmetry a clinician would look for. Signals, QA and the
gait layer all come from the production code (build_analysis_payload /
build_gait_analysis); only the joint positions and contacts are synthetic.
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
N_SECONDS = 14.0
RUN_ID = "demo-walk_synthetic"
TRACK_FILE = "demo-walk_subjects.json"

L_THIGH = 0.45
L_SHANK = 0.42
HIP_HEIGHT = 0.95  # nominal, before the stance leg plants the body (see below)


class Walker:
    """One subject's gait, defined by the parameters a clinician would report."""

    def __init__(self, *, key, label, color, stride_s, speed, stance_frac, knee_gain, lane_x):
        self.key = key
        self.label = label
        self.color = color
        self.stride_s = stride_s
        self.speed = speed
        self.stance_frac = stance_frac  # share of the cycle spent in contact
        self.knee_gain = knee_gain      # 1.0 = typical swing-phase knee flexion
        self.lane_x = lane_x            # side-by-side placement in the shared scene


WALKERS = [
    Walker(key="", label="Reference", color="#22d3ee", stride_s=1.2, speed=1.10,
           stance_frac=0.62, knee_gain=1.0, lane_x=0.0),
    Walker(key="_person2", label="Slower gait", color="#fb923c", stride_s=1.45, speed=0.72,
           stance_frac=0.70, knee_gain=0.55, lane_x=1.0),
]


def world_to_cam(p):
    return [p[1], -p[2], -p[0]]


def _circ_dist(frac, center):
    d = abs(frac - center)
    return min(d, 1.0 - d)


def _hip_flex_deg(frac):
    # Flexed at heel strike (~26 deg), extends through stance, flexes in swing.
    return 20.0 * math.cos(2 * math.pi * frac) + 6.0


def _knee_flex_deg(frac, gain=1.0):
    # Small loading-response peak in early stance, large flexion peak in swing.
    loading = 15.0 * math.exp(-((_circ_dist(frac, 0.15) / 0.10) ** 2))
    swing = 58.0 * math.exp(-((_circ_dist(frac, 0.73) / 0.11) ** 2))
    return 4.0 + gain * (loading + swing)


def _ankle_dorsi_deg(frac):
    # Neutral at strike, dorsiflexes through stance, plantarflexes at push-off.
    base = 6.0 * math.sin(2 * math.pi * (frac - 0.08)) - 1.0
    pushoff = -15.0 * math.exp(-((_circ_dist(frac, 0.60) / 0.055) ** 2))
    return base + pushoff


def _leg_joints(x_off, y, frac, knee_gain=1.0):
    """Sagittal-plane forward kinematics for one leg from target joint angles.

    Placing knee/ankle/foot from hip via these angles makes the reconstructed
    clinical angles reproduce the target curves exactly (see gait.py sign math),
    so the demo shows clinically shaped hip/knee/ankle curves.
    """
    th = math.radians(_hip_flex_deg(frac))       # thigh forward tilt
    tk = math.radians(_knee_flex_deg(frac, knee_gain))  # knee flexion (bends shank back)
    beta = math.radians(_ankle_dorsi_deg(frac)) + (th - tk)  # foot sagittal angle
    hip = (x_off, y, HIP_HEIGHT)
    knee = (x_off, y + L_THIGH * math.sin(th), HIP_HEIGHT - L_THIGH * math.cos(th))
    ankle = (
        x_off,
        knee[1] + L_SHANK * math.sin(th - tk),
        knee[2] - L_SHANK * math.cos(th - tk),
    )
    fy, fz = math.cos(beta), math.sin(beta)
    toe = (x_off, ankle[1] + 0.15 * fy, ankle[2] - 0.03 + 0.15 * fz)
    heel = (x_off, ankle[1] - 0.05 * fy, ankle[2] - 0.03 - 0.05 * fz)
    return hip, knee, ankle, toe, heel


def build_records(walker):
    rng = np.random.default_rng(11 + len(walker.key))
    n = int(N_SECONDS * FPS)
    records = []
    contacts = []
    for i in range(n):
        t = i / FPS
        y = walker.speed * t
        lane = walker.lane_x
        frac_l = (t / walker.stride_s) % 1.0
        frac_r = (frac_l + 0.5) % 1.0
        joints = [None] * 21

        # Real walking plants the stance foot and lets the pelvis rise and fall
        # over it. Holding the hip at a fixed height instead would lift the foot
        # off the floor at both ends of stance and produce two contacts per
        # cycle, so solve for the body height that keeps the stance foot down.
        legs = {}
        for side, x_off, frac in (("left", -0.10, frac_l), ("right", 0.10, frac_r)):
            legs[side] = (_leg_joints(x_off, y, frac, walker.knee_gain), frac)
        stance_sides = [s for s in ("left", "right") if legs[s][1] < walker.stance_frac]
        if stance_sides:
            body_z = -min(
                min(legs[s][0][3][2], legs[s][0][4][2])  # lowest of toe / heel
                for s in stance_sides
            )
        else:
            body_z = 0.0

        def sj(idx, p):
            joints[idx] = world_to_cam((p[0] + lane, p[1], p[2] + body_z))

        sj(9, (-0.10, y, HIP_HEIGHT))
        sj(10, (0.10, y, HIP_HEIGHT))
        sj(5, (-0.16, y, 1.45))
        sj(6, (0.16, y, 1.45))
        sj(0, (0.0, y, 1.62))

        loaded = {}
        for side, (knee_i, ankle_i, toe_i, heel_i) in (
            ("left", (11, 13, 15, 17)),
            ("right", (12, 14, 18, 20)),
        ):
            (_hip, knee, ankle, toe, heel), frac = legs[side]
            sj(knee_i, knee)
            sj(ankle_i, ankle)
            sj(toe_i, toe)
            sj(heel_i, heel)
            loaded[side] = frac < walker.stance_frac
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
                joint = world_to_cam((lane, y, 0.95))
            joints[j] = [v + rng.normal(0, 0.003) for v in joint]
        contact_l = loaded["left"]
        contact_r = loaded["right"]
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


def write_walker(project_root, walker, index):
    run_id = f"{RUN_ID}{walker.key}"
    records, contacts = build_records(walker)
    metadata = {
        "records": records,
        "fps_output": FPS,
        "fps_input": FPS,
        "total_frames_processed": len(records),
        "video_input": "synthetic",
        "video_width": 1920,
        "video_height": 1080,
        "space_view": None,
        # A shared track file is what makes the viewer treat these runs as the
        # subjects of one video rather than separate videos.
        "subject": {
            "index": index,
            "id": f"demo{index}",
            "label": walker.label,
            "color": walker.color,
            "track_file": TRACK_FILE,
        },
    }

    params = AnalysisParams()
    payload = build_analysis_payload(run_id=run_id, metadata=metadata, params=params)

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

    analysis_id = f"{run_id}_default_demo00000"
    run_dir = project_root / "output" / run_id
    analysis_directory = run_dir / "analysis" / analysis_id
    analysis_directory.mkdir(parents=True, exist_ok=True)

    write_json(analysis_directory / "signals.json", {"analysis_id": analysis_id, "preset": params.preset, "signals": payload["signals"]})
    write_json(analysis_directory / "frames.json", {"analysis_id": analysis_id, "preset": params.preset, "fps": payload["fps"], "frames": payload["frames"]})
    write_json(analysis_directory / "qa.json", payload["qa"])
    write_json(analysis_directory / "gait.json", {"analysis_id": analysis_id, "preset": params.preset, "gait": payload["gait"]})
    analysis_manifest = build_analysis_manifest(run_id=run_id, analysis_id=analysis_id, preset=params.preset, parameters=params.as_dict(), qa_summary=payload["qa"])
    write_json(analysis_directory / "analysis_manifest.json", analysis_manifest)

    manifest = build_run_manifest(run_id=run_id, run_directory=run_dir, metadata=metadata)
    manifest = append_analysis_to_run_manifest(manifest, analysis_id=analysis_id, preset=params.preset, parameters=params.as_dict(), qa_summary=payload["qa"])
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "run_metadata.json", metadata)

    st = payload["gait"]["spatiotemporal"]
    total_cycles = sum(
        int((payload["gait"]["cycles"][side].get(sig, {}) or {}).get("n_cycles", 0) or 0)
        for side in ("left", "right")
        for sig in payload["gait"]["cycles"][side]
    )
    print(f"{run_id}  [{walker.label}]")
    print(f"  walking={st['walking_detected']} cadence={st['cadence_steps_per_min']} cycles={total_cycles} events={len(payload['gait']['events'])}")
    print(f"  stride_time={st['stride_time_s']['mean']} stride_length={st['stride_length_m']['mean']} speed={st['walking_speed_m_s']['mean']} stance%={st['stance_pct']['mean']}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    for index, walker in enumerate(WALKERS):
        write_walker(project_root, walker, index)


if __name__ == "__main__":
    main()
