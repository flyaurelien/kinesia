"""Validation of the clinical gait layer against a synthetic walker with a
KNOWN ground truth: cadence, stride length, speed, event times and joint
angles are all constructed, so every reported quantity can be checked."""

import math
import unittest

import numpy as np

from sam_3d_pose_estimation.gait import (
    build_gait_analysis,
    compute_clinical_angles,
    compute_neutral_reference,
    detect_gait_events,
    find_static_frames,
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


def standing_frames(shank_tilt_deg: float, foot_pitch_deg: float, n: int = 240):
    """A subject standing still whose reconstruction carries a known posture
    bias: the shank tilted forward by `shank_tilt_deg` (ankle behind the knee)
    and the foot `foot_pitch_deg` toes-up. Both are anatomically ~0 in reality,
    so the neutral reference must measure exactly these values back.
    """
    tilt = math.radians(shank_tilt_deg)
    pitch = math.radians(foot_pitch_deg)
    frames = []
    for i in range(n):
        joints = [None] * 21
        for x_off, (knee_i, ankle_i, toe_i, heel_i) in (
            (-0.10, (11, 13, 15, 17)),
            (0.10, (12, 14, 18, 20)),
        ):
            hip = (x_off, 0.0, 0.95)
            knee = (x_off, 0.0, 0.50)
            ankle = (x_off, -0.42 * math.sin(tilt), 0.50 - 0.42 * math.cos(tilt))
            heel = (x_off, ankle[1] - 0.06, 0.02)
            toe = (x_off, heel[1] + 0.22 * math.cos(pitch), heel[2] + 0.22 * math.sin(pitch))
            joints[9 if x_off < 0 else 10] = world_to_cam(hip)
            joints[knee_i] = world_to_cam(knee)
            joints[ankle_i] = world_to_cam(ankle)
            joints[toe_i] = world_to_cam(toe)
            joints[heel_i] = world_to_cam(heel)
        joints[5] = world_to_cam((-0.15, 0.0, 1.45))
        joints[6] = world_to_cam((0.15, 0.0, 1.45))
        frames.append({"index": i, "subject_present": True, "joints_cam": joints,
                       "foot_contact": {"left": True, "right": True, "support": "both"}})
    return frames


class TestNeutralReference(unittest.TestCase):
    """The monocular reconstruction has a systematic standing-posture bias; the
    static calibration pose must measure it and cancel it."""

    BIAS_SHANK = 19.4  # measured on a real standing run
    BIAS_FOOT = 15.5

    def test_measures_and_cancels_a_known_standing_bias(self):
        frames = standing_frames(self.BIAS_SHANK, self.BIAS_FOOT)
        raw = compute_clinical_angles(frames, FPS)
        # Raw angles carry the bias: knee reads the shank tilt, ankle the sum.
        self.assertLess(abs(float(np.median([v for v in raw["knee.left"] if v is not None])) - self.BIAS_SHANK), 1.0)
        self.assertLess(
            abs(float(np.median([v for v in raw["ankle.left"] if v is not None]))
                - (self.BIAS_SHANK + self.BIAS_FOOT)),
            1.5,
        )
        # After calibration every clinical angle reads ~0 at quiet stance.
        gait = build_gait_analysis(frames, FPS)
        neutral = gait["neutral_reference"]
        self.assertTrue(neutral["applied"])
        self.assertGreater(neutral["static_duration_s"], 1.0)
        self.assertLess(abs(neutral["offsets_deg"]["ankle.left"] - (self.BIAS_SHANK + self.BIAS_FOOT)), 1.5)
        for key, values in gait["angles"].items():
            finite = [v for v in values if v is not None]
            self.assertLess(abs(float(np.median(finite))), 0.5, key)

    def test_walking_has_no_quiet_stance(self):
        # Double support during walking is brief and the pelvis is moving:
        # it must never be mistaken for a calibration pose.
        frames = synthetic_frames(noise_std=0.004)
        self.assertEqual(int(find_static_frames(frames, FPS).sum()), 0)
        angles = compute_clinical_angles(frames, FPS)
        neutral = compute_neutral_reference(frames, angles, FPS)
        self.assertFalse(neutral["applied"])
        self.assertEqual(neutral["offsets_deg"], {})

    def test_implausible_stance_is_rejected(self):
        # A crouch is not a neutral pose: refuse rather than zero it out.
        frames = standing_frames(60.0, 40.0)
        angles = compute_clinical_angles(frames, FPS)
        neutral = compute_neutral_reference(frames, angles, FPS)
        self.assertFalse(neutral["applied"])


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
