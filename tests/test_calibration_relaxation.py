"""Calibration acceptance and stationary handoff retries, without hardware IO."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import tests.test_live_alignment_runtime as lifecycle
from tests.test_active_calibration import packet
from urcal.control.geometry import flange_from_tcp, inverse, pose_delta, pose_matrix
from urcal.control.handoff import CalibrationHandoff
from urcal.control.policy import CalibrationFrame, CalibrationTracker, CalibrationUnavailable, ControlFault
from urcal.control.runtime import parse_args
from urcal.control.startup import check_handoff_snapshot
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.task import MotionInfeasible


CLI = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10", "--gap", ".3"]


def options(*extra, mode="follow"):
    return parse_args(["--mode", mode, *CLI, *extra])


class CalibrationRelaxationTests(unittest.TestCase):
    def test_live_defaults_separate_acceptance_from_task_and_motion_bounds(self):
        expected = dict(max_calibration_jump_m=.05, max_calibration_jump_deg=5.,
                        handoff_max_shift_m=.15, handoff_max_shift_deg=15., handoff_timeout=15.,
                        max_target_m=.2, target_tolerance_m=.05, target_tolerance_deg=10.,
                        stable_frames=3, max_calibration_age=.5, max_base_speed=.25,
                        max_base_angular_speed=.15, max_speed=.02, follow_max_speed=1.,
                        max_accel=.1, max_joint_speed=.2, max_joint_accel=.75,
                        max_travel=.30, min_gap=.15, base_motion_mode="pose")
        for mode in ("follow",):
            args = options(mode=mode)
            for key, value in expected.items():
                self.assertEqual(getattr(args, key), value, (mode, key))

    def test_invalid_handoff_options_rejected_and_explicit_old_limits_supported(self):
        for name in ("handoff_max_shift_m", "handoff_max_shift_deg", "handoff_timeout"):
            for value in ("nan", "inf", "-1"):
                with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        options(f"--{name}={value}")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            options("--handoff_timeout=0")
        args = options("--max_calibration_jump_m", ".015", "--max_calibration_jump_deg", "2",
                       "--handoff_max_shift_m", ".05", "--handoff_max_shift_deg", "10",
                       "--handoff_timeout", "5", "--max_target_m", ".1")
        self.assertEqual((args.handoff_max_shift_m, args.handoff_timeout, args.max_target_m), (.05, 5., .1))

    def test_jump_acceptance_still_checks_translation_rotation_and_relative_bases(self):
        for distance, degrees, accepted in ((.04, 0., True), (0., 4., True), (.051, 0., False), (0., 5.1, False), (.74, 0., False)):
            with self.subTest(distance=distance, degrees=degrees):
                engine = CooperativeController(options())
                engine.gate.update(packet(0, 10.), 10.)
                sample = packet(1, 10.4)
                sample.poses["ur5e"][0, 3] += distance
                sample.poses["ur5e"][:3, :3] = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(degrees)])[:3, :3]
                if accepted:
                    self.assertTrue(engine.gate.update(sample, 10.4))
                else:
                    with self.assertRaises(CalibrationUnavailable):
                        engine.gate.update(sample, 10.4)
                    self.assertEqual(engine.gate.last_capture, 10.)
        engine = CooperativeController(options())
        engine.gate.update(packet(0, 10.), 10.)
        sample = packet(1, 10.4)
        sample.poses["ur5e"][0, 3] -= .03
        sample.poses["ur10e"][0, 3] += .03
        with self.assertRaisesRegex(CalibrationUnavailable, "relative base"):
            engine.gate.update(sample, 10.4)

    def test_relaxed_jump_does_not_disable_observed_base_velocity_checks(self):
        engine = CooperativeController(options("--base_motion_mode", "velocity"))
        self.assertTrue(engine.ingest(packet(0, 10.), 10.))
        self.assertFalse(engine.ingest(packet(1, 10.04, .04), 10.04))
        self.assertIn("base linear speed", engine.last_rejection)

    def warm_handoff(self):
        handoff = CalibrationHandoff(CalibrationTracker(check_reprojection=False), 10.)
        for index in range(5):
            at = 10.04 + .04 * index
            self.assertTrue(handoff.ingest(packet(index, at), at))
        return handoff

    def test_validation_retry_requires_new_accepted_frame_and_retains_deadline(self):
        for failure in (CalibrationUnavailable("target shift"),
                        ResearchFault("tracking_error", "target error"),
                        MotionInfeasible("compensation acceleration", "compensation_infeasible")):
            handoff = self.warm_handoff()
            validate = Mock(side_effect=[failure, None])
            self.assertFalse(handoff.try_validate(10.21, validate))
            self.assertIn(str(failure), handoff.reason)
            self.assertFalse(handoff.try_validate(10.22, validate))
            validate.assert_called_once()
            self.assertTrue(handoff.ingest(packet(5, 10.24), 10.24))
            self.assertTrue(handoff.try_validate(10.24, validate))
            self.assertEqual(validate.call_count, 2)
            with self.assertRaisesRegex(CalibrationUnavailable, "wait exceeded 15s"):
                handoff.try_validate(25.01, validate)

    def test_hardware_encoder_and_unknown_faults_do_not_wait_for_vision(self):
        for failure in (ControlFault("robot moved during handoff"),
                        ResearchFault("tcp_speed", "actual overspeed"), RuntimeError("transport failed")):
            handoff = self.warm_handoff()
            with self.assertRaises(type(failure)) as caught:
                handoff.try_validate(10.21, Mock(side_effect=failure))
            self.assertIs(caught.exception, failure)

    def test_follow_waits_past_five_seconds_without_realigning(self):
        original = lifecycle.config
        def longer_wait(execute=False):
            args = original(execute)
            args.handoff_timeout = options().handoff_timeout
            return args
        for fixture in (lifecycle.LiveAlignmentRuntimeTests(),):
            kwargs = {"done_after": 1, "stop_after": 158,
                      "calibration_events": {index: 0. for index in range(150, 155)}}
            with self.subTest(mode=type(fixture).__name__), patch.object(lifecycle, "config", longer_wait):
                result = fixture.run_fake(**kwargs)
            self.assertIsNone(result.caught)
            self.assertEqual(result.alignment.commit.call_count, 1)
            self.assertGreater(result.alignment.propose_hold.call_count, 140)
            self.assertGreater(result.session.send.call_count, 0)

    def test_shift_failure_can_recover_at_rest_without_another_arm_or_alignment(self):
        for fixture in (lifecycle.LiveAlignmentRuntimeTests(),):
            kwargs = {"done_after": 1, "stop_after": 15,
                      "calibration_events": {**{i: .02 for i in range(3, 8)},
                                             **{i: 0. for i in range(8, 14)}}}
            result = fixture.run_fake(**kwargs)
            self.assertIsNone(result.caught)
            self.assertEqual(result.alignment.commit.call_count, 1)
            self.assertGreater(result.session.send.call_count, 0)

    def test_recorded_121mm_shift_is_accepted_but_infeasible_motion_waits(self):
        data = json.loads((Path(__file__).parent / "fixtures/calibration_handoff_shift.json").read_text())
        args = options("--base_motion_mode", "velocity", "--gap", ".30", "--max_travel", ".15",
                       "--stable_frames", "5", "--max_base_speed", ".15", "--max_base_angular_speed", ".5",
                       "--ur10e_output_correction")  # Recorded targets used the original corrected frame.
        engine = CooperativeController(args)
        handoff = CalibrationHandoff(engine.gate, data["started_at"], timeout=args.handoff_timeout,
                                     estimator_factory=engine._new_estimator, mappings=engine.mappings,
                                     leader=engine.leader)
        for row in data["frames"]:
            frame = CalibrationFrame(row["sequence"], row["captured_at"],
                                     {name: np.array(p) for name, p in row["poses_camera_model"].items()},
                                     row["raw_mean_px"], row["raw_max_px"], row["valid_counts"])
            self.assertTrue(handoff.ingest(frame, row["at"]))
        now = data["now"]
        states = {name: SimpleNamespace(joints=np.array(s["q"]), joint_velocity=np.array(s["qd"]),
                                       tcp=np.array(s["tcp"]), speed=np.array(s["speed"]))
                  for name, s in data["endpoint_robots"].items()}
        bases = handoff.estimator.estimate(now, leader=engine.leader).base_poses
        leader = bases[engine.leader] @ flange_from_tcp(states[engine.leader].tcp, engine.offsets[engine.leader])
        targets = {engine.follower: inverse(bases[engine.follower]) @ leader @ engine.path.pose(0.)}
        nominal = {name: np.array(p) for name, p in data["nominal_targets"].items()}
        dp, _ = pose_delta(nominal[engine.follower], targets[engine.follower])
        self.assertAlmostEqual(np.linalg.norm(dp), .121400977, places=8)
        original_bases = {name: np.array(p) for name, p in data["original_bases"].items()}
        def check():
            check_handoff_snapshot(args, original_bases, states, bases, states,
                                   offsets=engine.offsets, nominal_targets=nominal, live_targets=targets)
        check()
        args.handoff_max_shift_m = .05
        with self.assertRaisesRegex(CalibrationUnavailable, "121.4mm"):
            check()
        args.handoff_max_shift_m = .15
        for displacement, angle in ((.151, 0.), (0., 15.1)):
            outside = {engine.follower: nominal[engine.follower].copy()}
            outside[engine.follower][0, 3] += displacement
            outside[engine.follower][:3, :3] = (
                nominal[engine.follower][:3, :3]
                @ pose_matrix([0, 0, 0, 0, 0, np.deg2rad(angle)])[:3, :3])
            with self.subTest(displacement=displacement, angle=angle), self.assertRaises(CalibrationUnavailable):
                check_handoff_snapshot(args, original_bases, states, bases, states,
                                       offsets=engine.offsets, nominal_targets=nominal, live_targets=outside)
        engine.gate, engine.estimator = handoff.tracker, handoff.estimator
        engine.initial_tcp = np.array(data["initial_robots"][engine.follower]["tcp"])
        # Recorded estimator noise makes the first command infeasible. Keep
        # the completed alignment, rather than increasing acceleration limits.
        self.assertFalse(handoff.try_validate(now, lambda: engine.prepare(now, states, .04)))
        self.assertIn("compensation", handoff.reason)
        self.assertEqual(engine.progress, 0.)


if __name__ == "__main__":
    unittest.main()
