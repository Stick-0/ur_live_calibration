"""Controller braking, fresh calibration replacement, and bounded resumption.

Synthetic encoder states and recorded camera evidence exercise the real
controller/window/estimator. No RTDE interface or camera is constructed.
"""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_cooperative_controller import accept, packet, scene, settings
from urcal.control.policy import CalibrationFrame, ControlFault
from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController


def live_scene(**overrides):
    options = dict(
        live_calibration_recovery=True, base_motion_mode="pose", leader_response="bounded",
        stable_frames=1, target_tolerance_m=.05, target_tolerance_deg=10.,
        max_calibration_jump_m=.05, max_calibration_jump_deg=5.,
        max_calibration_age=.5, max_invalid_duration=.3, max_target_m=.2,
        max_accel=.1, max_angular_accel=.5, max_joint_accel=.75,
        calibration_recovery_timeout=2., calibration_recovery_max_shift_m=.15,
        calibration_recovery_max_shift_deg=15., calibration_blend_time=.5)
    options.update(overrides)
    engine, bases, states = scene(**options)
    accept(engine, bases, at=10., received=10.12)
    initial = engine.prepare_live(10.12, states, .04)
    engine.commit(initial)
    return engine, bases, states


def shifted_packet(engine, bases, sequence, capture, shift):
    moved = deepcopy(bases)
    moved[engine.follower][0, 3] += shift
    return packet(engine, moved, capture, sequence)


class LiveCalibrationControllerTests(unittest.TestCase):
    def test_sustained_outlier_brakes_and_good_frames_recover_without_arm(self):
        engine, bases, states = live_scene()
        follower, leader = states[engine.follower], states[engine.leader]
        for vector in (engine.previous_twist, engine.previous_qdot, follower.speed, follower.joint_velocity):
            vector[0] = .02
        leader.speed[0] = leader.joint_velocity[0] = .1
        self.assertFalse(engine.ingest(shifted_packet(engine, bases, 2, 10.1, .10), 10.22,
                                       active_recovery=True))
        self.assertFalse(engine.live_calibration.waiting)
        self.assertTrue(engine.check_live_calibration(10.22))
        self.assertFalse(engine.check_live_calibration(10.521))
        brake = engine.prepare_live(10.521, states, .04)
        self.assertTrue(brake.record["calibration_braking"])
        self.assertGreater(brake.decision.twist[0], 0.)
        self.assertLess(brake.decision.twist[0], .02)
        self.assertLessEqual(abs(brake.decision.twist[0] - .02), .1 * .04 + 1e-12)
        self.assertEqual(brake.next_progress, engine.progress)
        self.assertEqual(engine.gate.last_capture, 10.)
        # A return to the old estimate still needs an independent fresh
        # cluster; one subsequent plausible frame cannot instantly resume.
        for index, capture in enumerate((10.41, 10.49, 10.57), 3):
            result = engine.ingest(packet(engine, bases, capture, index), capture + .12,
                                   active_recovery=True)
            self.assertEqual(result, index == 5)
        self.assertFalse(engine.live_calibration.waiting)
        self.assertEqual(engine.live_calibration.reacquisitions, 1)
        self.assertTrue(engine.check_live_calibration(10.69))
        self.assertEqual(engine.gate.last_capture, 10.57)

    def test_stable_replacement_beyond_jump_limit_blends_without_capture_or_velocity_fabrication(self):
        # Isolate the existing recovery-window blend/rate contract. The normal
        # ACTIVE pose smoother has its own slower coordinate-update bounds.
        engine, bases, states = live_scene(base_pose_window=0)
        for index, capture in enumerate((10.1, 10.18, 10.26), 2):
            accepted = engine.ingest(shifted_packet(engine, bases, index, capture, .08), capture + .12,
                                     active_recovery=True)
            self.assertEqual(accepted, index == 4)
        self.assertTrue(engine.check_live_calibration(10.38))
        first = engine.state(10.38, active=True)
        np.testing.assert_allclose(first.base_poses[engine.follower], bases[engine.follower], atol=1e-12)
        self.assertEqual(first.measurement_times[engine.follower], 10.26)
        self.assertAlmostEqual(first.measurement_age[engine.follower], .12)
        previous = first
        moved = deepcopy(bases)
        moved[engine.follower][0, 3] += .08
        for index in range(1, 27):
            now = 10.38 + index * .04
            engine.ingest(packet(engine, moved, now - .08, index + 4), now, active_recovery=True)
            snapshot = engine.state(now, active=True)
            displacement = np.linalg.norm(snapshot.base_poses[engine.follower][:3, 3]
                                          - previous.base_poses[engine.follower][:3, 3])
            self.assertLessEqual(displacement, .1 * .04 + 1e-12)
            for name in bases:
                np.testing.assert_array_equal(snapshot.base_twists[name], np.zeros(6))
                self.assertAlmostEqual(snapshot.measurement_times[name], now - .08)
            previous = snapshot
        np.testing.assert_allclose(previous.base_poses[engine.follower], moved[engine.follower], atol=1e-10)
        self.assertFalse(engine.live_calibration.blending)
        result = engine.prepare_live(now, states, .04)
        self.assertFalse(result.record.get("calibration_braking", False))
        self.assertLess(result.decision.twist[0], 0., "tracking must use the recovered translated base")

    def test_alternating_or_excessive_new_estimates_cannot_extend_observation_deadline(self):
        for shifts in ((.08, -.08) * 10, (.3,) * 20):
            with self.subTest(shifts=shifts[:2]):
                engine, bases, _states = live_scene()
                for index, shift in enumerate(shifts, 2):
                    capture = 10.1 + (index - 2) * .08
                    self.assertFalse(engine.ingest(shifted_packet(engine, bases, index, capture, shift),
                                                   capture + .12, active_recovery=True))
                self.assertAlmostEqual(engine.live_calibration.started_at, 10.22)
                self.assertEqual(engine.gate.last_capture, 10.)
                self.assertFalse(engine.check_live_calibration(12.21))
                with self.assertRaisesRegex(ControlFault, "observation exceeded"):
                    engine.check_live_calibration(12.221)

    def test_no_new_frame_brakes_for_bounded_window_and_then_expires(self):
        engine, bases, states = live_scene()
        self.assertFalse(engine.check_live_calibration(10.51))
        self.assertTrue(engine.live_calibration.waiting)
        result = engine.prepare_live(10.55, states, .04)
        self.assertTrue(result.record["calibration_braking"])
        for name in bases:
            self.assertEqual(result.record["measurement_times"][name], 10.)
            self.assertAlmostEqual(result.record["measurement_age"][name], .55)
            np.testing.assert_array_equal(result.record["bases"][name], bases[name])
        self.assertEqual(engine.gate.last_capture, 10.)
        with self.assertRaisesRegex(ControlFault, "observation exceeded"):
            engine.prepare_live(12.511, states, .04)

    def test_late_good_image_is_rejected_before_it_can_reset_expired_window(self):
        engine, bases, _states = live_scene()
        engine.ingest(shifted_packet(engine, bases, 2, 10.1, .08), 10.22, active_recovery=True)
        with self.assertRaisesRegex(ControlFault, "observation exceeded"):
            engine.ingest(packet(engine, bases, 12.2, 3), 12.221, active_recovery=True)
        self.assertEqual(engine.gate.last_capture, 10.)
        self.assertTrue(engine.live_calibration.waiting)

    def test_duplicate_and_bad_frames_do_not_refresh_accepted_capture_or_window(self):
        engine, bases, _states = live_scene()
        original = shifted_packet(engine, bases, 2, 10.1, .08)
        engine.ingest(original, 10.22, active_recovery=True)
        engine.ingest(original, 10.24, active_recovery=True)
        self.assertEqual(engine.live_calibration.candidate.count, 1)
        engine.ingest(packet(engine, bases, 10.18, 3, reason="PnP failed"), 10.3, active_recovery=True)
        self.assertEqual(engine.live_calibration.candidate.count, 0)
        self.assertEqual(engine.gate.last_capture, 10.)
        self.assertIsNone(engine.live_calibration.started_at)
        self.assertFalse(engine.check_live_calibration(10.521))
        self.assertAlmostEqual(engine.live_calibration.started_at, 10.22)

    def test_robot_overspeed_and_collision_still_fail_during_calibration_wait(self):
        for fault in ("overspeed", "collision"):
            engine, bases, states = live_scene()
            engine.ingest(shifted_packet(engine, bases, 2, 10.1, .08), 10.22, active_recovery=True)
            if fault == "overspeed":
                states[engine.follower].joint_velocity[0] = engine.args.max_joint_speed * 2.
            else:
                engine.guard.allowed = False
            with self.subTest(fault=fault), self.assertRaises(ControlFault):
                engine.prepare_live(10.26, states, .04)

    def test_new_observation_after_recovery_has_its_own_fresh_budget(self):
        engine, bases, states = live_scene()
        for index, capture in enumerate((10.1, 10.18, 10.26), 2):
            engine.ingest(shifted_packet(engine, bases, index, capture, .08), capture + .12,
                          active_recovery=True)
        engine.prepare_live(10.38, states, .04)
        engine.ingest(shifted_packet(engine, bases, 5, 10.4, -.08), 10.52, active_recovery=True)
        self.assertFalse(engine.live_calibration.waiting)
        self.assertFalse(engine.check_live_calibration(10.761))
        self.assertTrue(engine.live_calibration.waiting)
        self.assertAlmostEqual(engine.live_calibration.started_at, 10.52)
        self.assertFalse(engine.check_live_calibration(12.5))

    def test_only_enabled_pose_bounded_control_uses_observation_window(self):
        for options in ({"live_calibration_recovery": False}, {"base_motion_mode": "velocity"},
                        {"leader_response": "strict"}, {"method": "direct"}):
            args = settings(live_calibration_recovery=True, base_motion_mode="pose", leader_response="bounded")
            for name, value in options.items():
                setattr(args, name, value)
            with self.subTest(options=options):
                self.assertIsNone(CooperativeController(args).live_calibration)
        engine, _bases, _states = live_scene()
        self.assertIsNotNone(engine.live_calibration)

    def test_cli_defaults_opt_out_and_positive_recovery_options(self):
        base = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]
        args = parse_args(base)
        self.assertTrue(args.live_calibration_recovery)
        self.assertEqual(args.calibration_recovery_timeout, 2.)
        self.assertFalse(parse_args(base + ["--no_live_calibration_recovery"]).live_calibration_recovery)
        for name in ("calibration_recovery_timeout", "calibration_recovery_max_shift_m",
                     "calibration_recovery_max_shift_deg", "calibration_blend_time"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(base + ["--" + name, value])

    def test_recorded_240ms_old_expiry_enters_observation_without_claiming_full_recovery(self):
        data = json.loads((Path(__file__).parent / "fixtures" / "live_calibration_rejections.json").read_text())
        engine, _bases, _states = scene(
            live_calibration_recovery=True, base_motion_mode="pose", leader_response="bounded",
            max_calibration_jump_m=.05, max_calibration_jump_deg=5., stable_frames=1,
            max_calibration_age=.5, max_invalid_duration=.3)

        def observed(row):
            return CalibrationFrame(row["sequence"], row["captured_at"],
                                    {name: np.asarray(value) for name, value in row["poses_camera_model"].items()},
                                    row["raw_mean_px"], row["raw_max_px"], row["valid_counts"])

        original = data["baseline"]
        self.assertTrue(engine.ingest(observed(original), original["received_at"]))
        cached = engine.state(original["received_at"], active=True)
        for row in data["samples"][:4]:
            self.assertFalse(engine.ingest(observed(row), row["received_at"], active_recovery=True))
        stop_at = data["samples"][3]["received_at"]
        first_rejection_at = data["samples"][0]["received_at"]
        self.assertAlmostEqual(stop_at - first_rejection_at, .24031496001407504)
        self.assertGreater(stop_at - original["captured_at"], .5)
        self.assertFalse(engine.check_live_calibration(stop_at))
        self.assertTrue(engine.live_calibration.waiting)
        self.assertEqual(engine.gate.last_capture, original["captured_at"])
        for name in cached.base_poses:
            np.testing.assert_array_equal(engine.live_calibration.frozen.base_poses[name], cached.base_poses[name])
        # The historical file has missing measurements during its terminal
        # stop. This test proves the old 240 ms cutoff is removed, not that
        # the entire historical run would recover within the new timeout.


if __name__ == "__main__":
    unittest.main()
