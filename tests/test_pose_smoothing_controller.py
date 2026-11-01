"""Live coordinate smoothing preserves encoder following and control lifecycles."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest

import numpy as np

from test_cooperative_controller import accept, scene
from test_live_calibration_controller import shifted_packet
from urcal.control.geometry import inverse
from urcal.control.runtime import parse_args


def active_scene(**overrides):
    options = dict(base_motion_mode="pose", leader_response="bounded",
                   target_tolerance_m=.05, target_tolerance_deg=10.,
                   max_calibration_jump_m=.05, max_calibration_jump_deg=5.,
                   max_invalid_duration=.3)
    options.update(overrides)
    engine, bases, states = scene(**options)
    accept(engine, bases)
    initial = engine.prepare_live(10., states, .04)
    engine.commit(initial)
    return engine, bases, states, initial


class PoseSmoothingControllerTests(unittest.TestCase):
    def test_cli_enables_robust_base_updates_and_zero_explicitly_disables_them(self):
        base = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]
        defaults = parse_args(base)
        self.assertEqual(defaults.base_pose_window, 5)
        self.assertEqual(defaults.base_pose_deadband_m, .005)
        self.assertEqual(defaults.base_pose_deadband_deg, .5)
        self.assertEqual(defaults.base_pose_update_speed, .02)
        self.assertEqual(defaults.base_pose_update_angular_speed, .05)
        enabled, *_ = active_scene()
        disabled, *_ = active_scene(base_pose_window=parse_args(base + ["--base_pose_window", "0"]).base_pose_window)
        self.assertIsNotNone(enabled.pose_smoother)
        self.assertIsNone(disabled.pose_smoother)
        invalid = (("base_pose_window", "2"), ("base_pose_window", "32"),
                   ("base_pose_window", "-1"), ("base_pose_smoothing_time", "0"),
                   ("base_pose_update_speed", "nan"), ("base_pose_update_angular_speed", "-1"),
                   ("base_pose_deadband_m", "-1"), ("base_pose_deadband_deg", "inf"))
        for option, value in invalid:
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + ["--" + option, value])

    def test_leader_encoder_translation_and_rotation_bypass_visual_pose_rate_limits(self):
        results = []
        for window in (0, 5):
            engine, bases, states, initial = active_scene(base_pose_window=window)
            leader = states[engine.leader]
            leader.joints[0] += .03
            leader.joints[4] += .04
            leader.tcp = engine.models[engine.leader].pose_and_jacobian(leader.joints)[0]
            leader.joint_velocity[[0, 4]] = [.10, .20]
            leader.speed = leader.joint_velocity.copy()
            # No visual update: the existing capture is still fresh. Encoder
            # motion must immediately affect both pose and velocity demand.
            result = engine.prepare_live(10.04, states, .04)
            target = result.record["target_tcp"]
            expected = inverse(bases[engine.follower]) @ bases[engine.leader] @ leader.tcp @ engine.path.pose(0.)
            np.testing.assert_allclose(target, expected, atol=1e-12)
            self.assertGreater(np.linalg.norm(target[:3, 3] - initial.record["target_tcp"][:3, 3]), .03)
            self.assertGreater(np.linalg.norm(result.record["leader_compensation"][:3]), .09)
            self.assertGreater(np.linalg.norm(result.record["leader_compensation"][3:]), .19)
            self.assertGreater(np.linalg.norm(result.decision.twist[:3]), .02)
            self.assertGreater(np.linalg.norm(result.decision.twist[3:]), .05)
            np.testing.assert_array_equal(result.record["base_compensation"], np.zeros(6))
            self.assertEqual(result.record["measurement_times"][engine.follower], 10.)
            results.append(result)
        np.testing.assert_allclose(results[0].record["target_tcp"], results[1].record["target_tcp"], atol=1e-12)
        np.testing.assert_allclose(results[0].record["leader_compensation"],
                                   results[1].record["leader_compensation"], atol=1e-12)
        np.testing.assert_allclose(results[0].decision.twist, results[1].decision.twist, atol=1e-12)

    def test_smoothed_geometry_does_not_extend_capture_freshness_or_avoid_braking(self):
        engine, bases, states, initial = active_scene(live_calibration_recovery=True)
        for at in (10.10, 10.20, 10.30, 10.49):
            proposal = engine.prepare_live(at, states, .04)
            self.assertFalse(proposal.record["calibration_braking"])
            self.assertEqual(proposal.record["measurement_times"][engine.follower], 10.)
            self.assertAlmostEqual(proposal.record["measurement_age"][engine.follower], at - 10.)
        self.assertEqual(engine.pose_smoother.diagnostics()["samples"], 1)
        expired = engine.prepare_live(10.501, states, .04)
        self.assertTrue(expired.record["calibration_braking"])
        self.assertTrue(engine.live_calibration.waiting)
        self.assertEqual(engine.gate.last_capture, 10.)
        self.assertAlmostEqual(expired.record["measurement_age"][engine.follower], .501)
        np.testing.assert_allclose(expired.record["bases"][engine.follower],
                                   initial.record["bases"][engine.follower], atol=1e-12)

    def test_live_replacement_keeps_filter_history_but_stop_starts_a_new_calibration_session(self):
        engine, bases, states, _initial = active_scene(live_calibration_recovery=True)
        modest = deepcopy(bases)
        modest[engine.follower][0, 3] += .012
        for index in range(1, 6):
            now = 10. + .04 * index
            accept(engine, modest, now, index + 1)
            engine.commit(engine.prepare_live(now, states, .04))
        smoother = engine.pose_smoother
        before = smoother.current.copy()
        old_estimator = engine.estimator
        for sequence, capture in enumerate((10.24, 10.32, 10.40), 7):
            accepted = engine.ingest(shifted_packet(engine, bases, sequence, capture, .08), capture,
                                     active_recovery=True)
            self.assertEqual(accepted, sequence == 9)
        self.assertIsNot(engine.estimator, old_estimator)
        self.assertIs(engine.pose_smoother, smoother)
        np.testing.assert_array_equal(smoother.current, before)
        replacement = engine.prepare_live(10.40, states, .04)
        self.assertEqual(replacement.record["measurement_times"][engine.follower], 10.40)
        self.assertLessEqual(np.linalg.norm(smoother.current[:3, 3] - before[:3, 3]),
                             smoother.linear_rate * smoother.max_dt + 1e-12)
        self.assertLess(smoother.current[0, 3], .01)
        engine.reset_after_stop()
        self.assertIsNone(engine.pose_smoother.current)
        self.assertEqual(engine.pose_smoother.diagnostics()["samples"], 0)
        fresh = deepcopy(bases)
        fresh[engine.follower][0, 3] += .08
        accept(engine, fresh, 10.44, 10)
        new_session = engine.state(10.44, active=True)
        np.testing.assert_allclose(new_session.base_poses[engine.follower], fresh[engine.follower], atol=1e-12)

    def test_clean_base_smoothing_does_not_slow_or_pause_gap_approach_progress(self):
        runs = [active_scene(base_pose_window=window) for window in (0, 5)]
        for engine, _bases, _states, _initial in runs:
            engine.handle_gap_approach("start", enabled=True)
        for tick in range(1, 81):
            results = []
            now = 10. + .04 * tick
            for engine, bases, states, _initial in runs:
                follower = states[engine.follower]
                follower.joint_velocity = engine.previous_qdot.copy()
                follower.speed = engine.previous_twist.copy()
                follower.joints += follower.joint_velocity * .04
                follower.tcp = engine.models[engine.follower].pose_and_jacobian(follower.joints)[0]
                accept(engine, bases, now, tick + 1)
                proposal = engine.prepare_live(now, states, .04)
                engine.commit(proposal)
                self.assertEqual(engine.gap_approach.state, "running")
                self.assertFalse(proposal.record["calibration_braking"])
                results.append(proposal)
            self.assertAlmostEqual(results[0].next_progress, results[1].next_progress, places=12)
            np.testing.assert_allclose(results[0].decision.twist, results[1].decision.twist, atol=1e-12)
        self.assertGreater(runs[1][0].progress, .05)
        self.assertLess(runs[1][0].path.pose(runs[1][0].progress)[2, 3], .3)


if __name__ == "__main__":
    unittest.main()
