"""Validation of the clinical gait layer against a synthetic walker with a
KNOWN ground truth: cadence, stride length, speed, event times and joint
angles are all constructed, so every reported quantity can be checked."""

import math
import unittest

import numpy as np

from sam_3d_pose_estimation.gait import (
    build_gait_analysis,
    compute_clinical_angles,
    detect_gait_events,
    zero_phase_lowpass,
)

FPS = 60.0
STRIDE_S = 1.2  # per-side cycle duration -> cadence = 100 steps/min
SPEED = 1.25  # m/s forward
N_SECONDS = 12.0


def world_to_cam(p):
    # inverse of cam_to_world: world (x, y, z) -> cam (y, -z, -x)
    return [p[1], -p[2], -p[0]]


def synthetic_frames(noise_std: float = 0.0, seed: int = 7):
    """A walker moving along world +Y at SPEED, stride period STRIDE_S.

    Feet alternate contact half a cycle apart; joints follow simple sinusoids
    with amplitudes typical of gait. All positions in metres, world Z up.
    """
    rng = np.random.default_rng(seed)
    n = int(N_SECONDS * FPS)
    frames = []
    omega = 2 * math.pi / STRIDE_S
    for i in range(n):
        t = i / FPS
        y = SPEED * t
        phase_l = omega * t
        phase_r = phase_l + math.pi
        joints = [None] * 21

        def set_joint(index, x, yy, z):
            joints[index] = world_to_cam((x, yy, z))

        hip_z = 0.95
        set_joint(9, -0.10, y, hip_z)   # left hip
        set_joint(10, 0.10, y, hip_z)   # right hip
        set_joint(5, -0.15, y, 1.45)    # shoulders
        set_joint(6, 0.15, y, 1.45)
        for side, phase, x_off in (("l", phase_l, -0.10), ("r", phase_r, 0.10)):
            swing = math.sin(phase)
            forward = 0.30 * swing
            lift = max(0.0, 0.10 * math.sin(phase))  # off the floor half the cycle
            knee_i, ankle_i = (11, 13) if side == "l" else (12, 14)
            toe_i, heel_i = (15, 17) if side == "l" else (18, 20)
            set_joint(knee_i, x_off, y + forward * 0.5, 0.50 + 0.03 * swing)
            set_joint(ankle_i, x_off, y + forward, 0.08 + lift)
            set_joint(heel_i, x_off, y + forward - 0.08, 0.03 + lift)
            set_joint(toe_i, x_off, y + forward + 0.14, 0.03 + lift)
        if noise_std > 0:
            for j, joint in enumerate(joints):
                if joint is not None:
                    joints[j] = [v + rng.normal(0, noise_std) for v in joint]
        contact_l = math.sin(phase_l) <= 0.0
        contact_r = math.sin(phase_r) <= 0.0
        frames.append(
            {
                "index": i,
                "subject_present": True,
                "joints_cam": joints,
                "foot_contact": {"left": contact_l, "right": contact_r,
                                 "support": "both" if contact_l and contact_r else ("left" if contact_l else "right")},
            }
        )
    return frames


class TestZeroPhaseFilter(unittest.TestCase):
    def test_removes_noise_without_lag(self):
        t = np.arange(0, 10, 1 / FPS)
        clean = np.sin(2 * math.pi * 1.0 * t)  # 1 Hz gait-band signal
        rng = np.random.default_rng(3)
        noisy = clean + rng.normal(0, 0.2, clean.shape)
        filtered = zero_phase_lowpass(noisy, FPS, cutoff_hz=6.0)
        # Noise strongly reduced...
        self.assertLess(np.std(filtered - clean), 0.5 * np.std(noisy - clean))
        # ...and no temporal lag: cross-correlation peaks at zero shift.
        shifts = range(-5, 6)
        scores = [np.corrcoef(np.roll(filtered, s), clean)[0, 1] for s in shifts]
        self.assertEqual(list(shifts)[int(np.argmax(scores))], 0)

    def test_preserves_nan_gaps(self):
        x = np.sin(np.linspace(0, 6, 300))
        x[100:120] = np.nan
        y = zero_phase_lowpass(x, FPS)
        self.assertTrue(np.all(np.isnan(y[100:120])))
        self.assertTrue(np.all(np.isfinite(np.delete(y, slice(100, 120)))))


class TestGaitAnalysis(unittest.TestCase):
    def setUp(self):
        self.frames = synthetic_frames(noise_std=0.004)  # ~4 mm joint noise
        self.gait = build_gait_analysis(self.frames, FPS)

    def test_event_cadence_matches_ground_truth(self):
        events = self.gait["events"]
        hs_left = [e for e in events if e["side"] == "left" and e["type"] == "heel_strike"]
        # ~one heel strike per stride period over the clip
        expected = int(N_SECONDS / STRIDE_S)
        self.assertAlmostEqual(len(hs_left), expected, delta=1)
        # Inter-strike interval == stride period (tolerance 5%)
        intervals = np.diff([e["time_s"] for e in hs_left])
        self.assertLess(abs(float(np.median(intervals)) - STRIDE_S), 0.05 * STRIDE_S)

    def test_spatiotemporal_parameters(self):
        st = self.gait["spatiotemporal"]
        self.assertTrue(st["walking_detected"])
        # cadence: 2 steps per stride -> 60 * 2 / 1.2 = 100 steps/min
        self.assertLess(abs(st["cadence_steps_per_min"] - 100.0), 5.0)
        self.assertLess(abs(st["stride_time_s"]["mean"] - STRIDE_S), 0.06)
        # stride length = SPEED * STRIDE_S = 1.5 m
        self.assertLess(abs(st["stride_length_m"]["mean"] - SPEED * STRIDE_S), 0.15)
        self.assertLess(abs(st["walking_speed_m_s"]["mean"] - SPEED), 0.15)
        self.assertGreater(st["stance_pct"]["mean"], 30.0)
        self.assertLess(st["stance_pct"]["mean"], 70.0)

    def test_cycles_are_extracted_and_consistent(self):
        cycles = self.gait["cycles"]["left"]["gait.knee.left.flexion_deg"]
        self.assertGreaterEqual(cycles["n_cycles"], 7)
        self.assertEqual(len(cycles["mean"]), 101)
        # The synthetic knee angle is periodic: cycle-to-cycle SD stays small.
        self.assertLess(float(np.nanmean(cycles["sd"])), 6.0)

    def test_standing_subject_degrades_gracefully(self):
        # Same skeleton but no motion and permanent double support.
        frames = synthetic_frames(noise_std=0.002)
        for frame in frames:
            frame["foot_contact"] = {"left": True, "right": True, "support": "both"}
        gait = build_gait_analysis(frames, FPS)
        self.assertFalse(gait["spatiotemporal"]["walking_detected"])
        self.assertEqual(gait["cycles"]["left"]["gait.knee.left.flexion_deg"]["n_cycles"], 0)


class TestClinicalAngles(unittest.TestCase):
    def test_straight_leg_reads_near_zero(self):
        # A perfectly vertical, stationary leg: hip/knee ~0 deg, foot flat ~0.
        frames = []
        for i in range(120):
            joints = [None] * 21
            def w2c(p):
                return [p[1], -p[2], -p[0]]
            joints[9] = w2c((-0.1, 0.0, 1.0))
            joints[10] = w2c((0.1, 0.0, 1.0))
            joints[11] = w2c((-0.1, 0.0, 0.5))
            joints[12] = w2c((0.1, 0.0, 0.5))
            joints[13] = w2c((-0.1, 0.0, 0.1))
            joints[14] = w2c((0.1, 0.0, 0.1))
            joints[15] = w2c((-0.1, 0.15, 0.02))
            joints[17] = w2c((-0.1, -0.05, 0.02))
            joints[18] = w2c((0.1, 0.15, 0.02))
            joints[20] = w2c((0.1, -0.05, 0.02))
            frames.append({"index": i, "subject_present": True, "joints_cam": joints,
                           "foot_contact": {"left": True, "right": True, "support": "both"}})
        angles = compute_clinical_angles(frames, FPS)
        for key in ("hip.left", "knee.right", "ankle.left"):
            values = [v for v in angles[key] if v is not None]
            self.assertTrue(values, key)
            self.assertLess(abs(float(np.median(values))), 5.0, key)


if __name__ == "__main__":
    unittest.main()
