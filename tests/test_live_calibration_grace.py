"""Transient visual outliers retain fresh geometry without braking the follower.

These tests exercise the production controller with encoder-only Cartesian
fixtures; there are no network, RTDE or camera connections.
"""

from copy import deepcopy
import unittest

import numpy as np

from test_cooperative_controller import accept, packet, scene
from test_live_calibration_controller import shifted_packet
from urcal.control.policy import ControlFault


def grace_scene(**overrides):
    options = dict(
        live_calibration_recovery=True, base_motion_mode="pose", leader_response="bounded",
        stable_frames=3, target_tolerance_m=.05, target_tolerance_deg=10.,
        max_calibration_jump_m=.05, max_calibration_jump_deg=5.,
        max_calibration_age=.5, max_invalid_duration=.3, max_target_m=.2,
        max_accel=.1, max_angular_accel=.5, max_joint_accel=.75,
        calibration_recovery_timeout=2., calibration_recovery_max_shift_m=.15,
        calibration_recovery_max_shift_deg=15., calibration_blend_time=.5)
    options.update(overrides)
    engine, bases, states = scene(**options)
    for sequence, capture in enumerate((10., 10.04, 10.08), 1):
        accept(engine, bases, at=capture, received=capture + .10, sequence=sequence)
    initial = engine.prepare_live(10.18, states, .04)
    engine.commit(initial)
    return engine, bases, states


class LiveCalibrationGraceTests(unittest.TestCase):
    def test_single_bad_frame_keeps_fresh_pose_and_next_good_frame_needs_no_restabilization(self):
        engine, bases, states = grace_scene()
        cached = deepcopy(engine._last_live_snapshot)
        bad = shifted_packet(engine, bases, 4, 10.12, .10)
        self.assertFalse(engine.ingest(bad, 10.22, active_recovery=True))
        self.assertTrue(engine.gate.ready)
        self.assertEqual(engine.gate.stable_count, 3)
        self.assertEqual(engine.gate._last_sequence, 4)
        self.assertEqual(engine.gate.last_capture, 10.08)
        self.assertFalse(engine.live_calibration.waiting)
        states[engine.leader].speed[0] = states[engine.leader].joint_velocity[0] = .01
        proposal = engine.prepare_live(10.26, states, .04)
        self.assertFalse(proposal.record.get("calibration_braking", False))
        self.assertTrue(proposal.record["live_calibration"]["grace"])
        self.assertGreater(proposal.twists[engine.follower][0], 0.)
        for name in bases:
            np.testing.assert_array_equal(proposal.record["bases"][name], cached.base_poses[name])
            self.assertEqual(proposal.record["measurement_times"][name], 10.08)
            self.assertAlmostEqual(proposal.record["measurement_age"][name], .18)
        self.assertTrue(engine.ingest(packet(engine, bases, 10.2, 5), 10.3, active_recovery=True))
        self.assertTrue(engine.gate.ready)
        self.assertTrue(engine.check_live_calibration(10.3))
        self.assertIsNone(engine.invalid_since)
        self.assertEqual(engine.live_calibration.reacquisitions, 0)
        self.assertEqual(engine.live_calibration.candidate.count, 0)

    def test_repeated_or_duplicate_rejections_do_not_refresh_grace_or_capture(self):
        engine, bases, states = grace_scene(max_calibration_age=1.)
        bad = shifted_packet(engine, bases, 4, 10.2, .30)
        for receipt in (10.30, 10.35, 10.45, 10.55):
            self.assertFalse(engine.ingest(bad, receipt, active_recovery=True))
            self.assertEqual(engine.invalid_since, 10.30)
            self.assertEqual(engine.gate.last_capture, 10.08)
        self.assertTrue(engine.check_live_calibration(10.59))
        self.assertFalse(engine.check_live_calibration(10.601))
        self.assertAlmostEqual(engine.live_calibration.started_at, 10.30)
        braking = engine.prepare_live(10.61, states, .04)
        self.assertTrue(braking.record["calibration_braking"])
        self.assertFalse(braking.record["live_calibration"]["grace"])
        with self.assertRaisesRegex(ControlFault, "observation exceeded"):
            engine.ingest(packet(engine, bases, 12.3, 5), 12.301, active_recovery=True)
        self.assertEqual(engine.gate.last_capture, 10.08)

    def test_original_capture_age_can_end_grace_before_invalid_duration(self):
        engine, bases, states = grace_scene()
        engine.ingest(shifted_packet(engine, bases, 4, 10.40, .30), 10.50,
                      active_recovery=True)
        self.assertTrue(engine.check_live_calibration(10.57))
        self.assertFalse(engine.check_live_calibration(10.581))
        self.assertLess(10.581 - engine.invalid_since, engine.args.max_invalid_duration)
        self.assertTrue(engine.prepare_live(10.59, states, .04).record["calibration_braking"])

    def test_late_good_frame_after_invalid_budget_requires_recovery_cluster(self):
        engine, bases, _states = grace_scene(max_calibration_age=1.)
        engine.ingest(shifted_packet(engine, bases, 4, 10.20, .30), 10.30,
                      active_recovery=True)
        self.assertFalse(engine.ingest(packet(engine, bases, 10.51, 5), 10.61,
                                       active_recovery=True))
        self.assertTrue(engine.live_calibration.waiting)
        self.assertEqual(engine.live_calibration.candidate.count, 1)
        self.assertEqual(engine.gate.last_capture, 10.08)

    def test_coherent_shift_during_grace_keeps_actual_capture_and_enters_smooth_blend(self):
        engine, bases, states = grace_scene()
        for sequence, capture in enumerate((10.12, 10.20, 10.28), 4):
            accepted = engine.ingest(shifted_packet(engine, bases, sequence, capture, .08),
                                     capture + .10, active_recovery=True)
            self.assertEqual(accepted, sequence == 6)
        self.assertEqual(engine.gate.last_capture, 10.28)
        self.assertEqual(engine.live_calibration.reacquisitions, 1)
        self.assertFalse(engine.live_calibration.waiting)
        self.assertTrue(engine.live_calibration.blending)
        result = engine.prepare_live(10.38, states, .04)
        self.assertFalse(result.record.get("calibration_braking", False))
        np.testing.assert_allclose(result.record["bases"][engine.follower], bases[engine.follower])
        self.assertEqual(result.record["measurement_times"][engine.follower], 10.28)

    def test_shorter_grace_preserves_real_candidate_receipts_into_observation(self):
        engine, bases, _states = grace_scene(max_invalid_duration=.10)
        for sequence, capture in enumerate((10.12, 10.20), 4):
            self.assertFalse(engine.ingest(shifted_packet(engine, bases, sequence, capture, .08),
                                           capture + .10, active_recovery=True))
        self.assertFalse(engine.check_live_calibration(10.321))
        self.assertEqual(engine.live_calibration.candidate.count, 2)
        np.testing.assert_allclose(
            [receipt for _frame, receipt in engine.live_calibration.candidate.samples],
            [10.22, 10.30], atol=1e-14, rtol=0.)
        self.assertTrue(engine.ingest(shifted_packet(engine, bases, 6, 10.28, .08),
                                      10.38, active_recovery=True))
        self.assertEqual(engine.gate.last_capture, 10.28)

    def test_approach_holds_on_one_outlier_and_continues_on_next_fresh_frame(self):
        engine, bases, states = grace_scene()
        engine.handle_gap_approach("start", enabled=True)
        engine.commit(engine.prepare_live(10.18, states, .04))
        progress = engine.progress
        self.assertEqual(engine.gap_approach.state, "running")
        # The reported run had a 55.3 mm outlier against a 50 mm jump limit.
        engine.ingest(shifted_packet(engine, bases, 4, 10.12, .0553), 10.22, active_recovery=True)
        self.assertEqual(engine.gap_approach.state, "running")
        self.assertFalse(engine.task_paused)
        self.assertTrue(engine.gap_approach_status(enabled=True)["temporary_hold"])
        held = engine.prepare_live(10.26, states, .04)
        self.assertEqual(held.decision.task_scale, 0.)
        self.assertEqual(held.next_progress, progress)
        self.assertFalse(held.record.get("calibration_braking", False))
        engine.commit(held)
        engine.ingest(packet(engine, bases, 10.20, 5), 10.30, active_recovery=True)
        self.assertEqual(engine.gap_approach.state, "running")
        self.assertFalse(engine.gap_approach_status(enabled=True)["temporary_hold"])
        self.assertGreater(engine.prepare_live(10.30, states, .04).next_progress, progress)

    def test_robot_collision_and_speed_checks_remain_active_during_grace(self):
        for fault in ("collision", "speed"):
            engine, bases, states = grace_scene()
            engine.ingest(shifted_packet(engine, bases, 4, 10.12, .10), 10.22, active_recovery=True)
            if fault == "collision":
                engine.guard.allowed = False
            else:
                states[engine.follower].joint_velocity[0] = engine.args.max_joint_speed * 2.
            with self.subTest(fault=fault), self.assertRaises(ControlFault):
                engine.prepare_live(10.26, states, .04)

    def test_preview_rejection_still_clears_stability(self):
        engine, bases, _states = grace_scene()
        engine.ingest(shifted_packet(engine, bases, 4, 10.12, .10), 10.22, active_recovery=False)
        self.assertFalse(engine.gate.ready)
        self.assertEqual(engine.gate.stable_count, 0)


if __name__ == "__main__":
    unittest.main()
