"""Pose-only calibration updates, real encoder limits, and recorded handoff.

All motion is planned offline. The compact recorded fixture contains only
calibration matrices, measured robot state and settings; no RTDE is created.
"""

from copy import deepcopy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_cooperative_controller import PermissiveGuard, accept, packet, scene
from test_follow_future_jacobian import ShrinkingTranslationModel
from urcal.control.geometry import inverse, pose_matrix
from urcal.control.policy import CalibrationFrame
from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.state import RelativeStateError, RelativeStateEstimator
from urcal.cooperative.task import MotionInfeasible


FIXTURE = Path(__file__).parent / "fixtures" / "follow_base_jitter.json"


def recorded_frame(row):
    return CalibrationFrame(
        sequence=row["sequence"], captured_at=row["captured_at"],
        poses={name: np.array(pose) for name, pose in row["poses_camera_model"].items()},
        raw_mean_px=row["raw_mean_px"], raw_max_px=row["raw_max_px"],
        valid_counts=row["valid_counts"])


def recorded_states(step):
    return {name: SimpleNamespace(
        joints=np.array(value["q"]), joint_velocity=np.array(value["qd"]),
        tcp=np.array(value["tcp"]), speed=np.array(value["tcp_speed"]),
        robot_timestamp=value["robot_timestamp"])
        for name, value in step["robots"].items()}


def recorded_controller(mode):
    data = json.loads(FIXTURE.read_text())
    args = SimpleNamespace(**data["settings"], base_motion_mode=mode)
    controller = CooperativeController(args)
    for sample in data["handoff_measurements"]:
        controller.ingest(recorded_frame(sample), sample["at"], allow_reacquire=True)
    first = data["first_control"]
    proposal = controller.prepare(first["at"], recorded_states(first), first["dt"], active=True)
    controller.commit(proposal)
    sample = data["live_measurement"]
    if not controller.ingest(recorded_frame(sample), sample["at"]):
        raise AssertionError(controller.last_rejection)
    return controller, data, proposal


class PoseBaseTrackingTests(unittest.TestCase):
    def assert_bounded(self, controller, proposal, states, dt, previous):
        qdot, twist = proposal.decision.qdot, proposal.decision.twist
        args = controller.args
        self.assertLessEqual(np.max(np.abs(qdot)), args.max_joint_speed + 1e-9)
        self.assertLessEqual(np.linalg.norm(twist[:3]), controller.limits.linear_speed + 1e-9)
        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed + 1e-9)
        for old_qdot, old_twist in (previous,
                                    (states[controller.follower].joint_velocity,
                                     states[controller.follower].speed)):
            self.assertLessEqual(np.max(np.abs(qdot - old_qdot)), args.max_joint_accel * dt + 1e-9)
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]), args.max_accel * dt + 1e-9)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]), args.max_angular_accel * dt + 1e-9)

    def test_cli_defaults_to_pose_with_explicit_legacy_velocity_option(self):
        base = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]
        self.assertEqual(parse_args(base).base_motion_mode, "pose")
        self.assertEqual(parse_args(base + ["--base_motion_mode", "velocity"]).base_motion_mode, "velocity")

    def test_cli_only_velocity_mode_requires_prediction_horizon_within_measurement_age(self):
        args = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10",
                "--max_calibration_age", "0.1", "--prediction_horizon", "0.5"]
        parsed = parse_args(args + ["--base_motion_mode", "pose"])
        self.assertGreater(parsed.prediction_horizon, parsed.max_calibration_age)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args(args + ["--base_motion_mode", "velocity"])
        self.assertIn("prediction_horizon must not exceed max_calibration_age", stderr.getvalue())

    def test_visual_update_filters_pose_without_velocity_rejection_or_prediction(self):
        initial = {"ur5e": pose_matrix([.4, 0, 1, 0, 0, 0]),
                   "ur10e": pose_matrix([-.4, 0, 1, 0, 0, 0])}
        estimator = RelativeStateEstimator(
            mode="pose", alpha=.2, max_linear_speed=.001, max_angular_speed=.001,
            max_prediction_age=.01, max_measurement_age=.5)
        estimator.update_pair(initial, 10., 10., 1)
        update = deepcopy(initial)
        update["ur5e"][0, 3] += .007
        estimator.update_pair(update, 10.01, 10.01, 2)
        now, later = estimator.estimate(10.01), estimator.estimate(10.4)
        self.assertAlmostEqual(now.base_poses["ur5e"][0, 3], .4014)
        for name in initial:
            np.testing.assert_array_equal(now.base_twists[name], np.zeros(6))
            np.testing.assert_array_equal(now.base_poses[name], later.base_poses[name])
            self.assertEqual(later.prediction_dt[name], 0.)
        np.testing.assert_array_equal(later.relative_twist, np.zeros(6))
        with self.assertRaisesRegex(RelativeStateError, "expired"):
            estimator.estimate(10.52)
        legacy = RelativeStateEstimator(mode="predict", max_linear_speed=.001)
        legacy.update_pair(initial, 10., 10., 1)
        with self.assertRaisesRegex(RelativeStateError, "base linear speed"):
            legacy.update_pair(update, 10.01, 10.01, 2)

    def test_shared_camera_motion_preserves_relative_pose_without_feedforward(self):
        estimator = RelativeStateEstimator(mode="pose", alpha=.2)
        initial = {"ur5e": pose_matrix([.4, -.1, 1, .1, 0, -.2]),
                   "ur10e": pose_matrix([-.4, .1, 1, -.1, .2, .1])}
        expected = inverse(initial["ur10e"]) @ initial["ur5e"]
        for index in range(15):
            change = pose_matrix([index * .002, 0, 0, 0, index * .002, 0])
            estimator.update_pair({name: change @ pose for name, pose in initial.items()},
                                  10. + .04 * index, 10. + .04 * index, index)
            snapshot = estimator.estimate(10. + .04 * index)
            np.testing.assert_allclose(snapshot.relative_pose, expected, atol=1e-12)
            np.testing.assert_array_equal(snapshot.relative_twist, np.zeros(6))

    def test_recorded_post_alignment_jitter_is_bounded_without_mandatory_compensation(self):
        controller, data, first = recorded_controller("pose")
        step = data["next_control"]
        states = recorded_states(step)
        for measured in states.values():
            self.assertLess(np.linalg.norm(measured.speed), 1e-15)
            self.assertLess(np.linalg.norm(measured.joint_velocity), 1e-15)
        proposal = controller.prepare(step["at"], states, step["dt"], active=True)
        self.assert_bounded(controller, proposal, states, step["dt"],
                            (first.decision.qdot, first.decision.twist))
        self.assertGreater(np.linalg.norm(proposal.record["target_tcp"][:3, 3]
                                         - first.record["target_tcp"][:3, 3]), .0001)
        np.testing.assert_array_equal(proposal.record["base_compensation"], np.zeros(6))
        self.assertEqual(proposal.record["base_motion_mode"], "pose")
        self.assertFalse(proposal.record["base_velocity_used"])

    def test_recorded_velocity_mode_retains_original_failure_for_research(self):
        controller, data, _ = recorded_controller("velocity")
        step = data["next_control"]
        with self.assertRaises(MotionInfeasible) as caught:
            controller.prepare(step["at"], recorded_states(step), step["dt"], active=True)
        self.assertEqual(str(caught.exception), data["expected_original_failure"])
        self.assertEqual(caught.exception.category, "compensation_infeasible")

    def test_base_pose_updates_and_encoder_leader_velocity_both_reach_following(self):
        controller, bases, states = scene(
            base_motion_mode="pose", leader_response="bounded", max_accel=.1,
            max_angular_accel=.5, max_joint_accel=.5, max_base_speed=.001,
            base_pose_window=0)
        accept(controller, bases)
        first = controller.prepare(10., states, .04, active=True)
        controller.commit(first)
        # 7 mm / 10 ms is a 0.7 m/s visual finite difference. This is geometry,
        # while the leader's encoder velocity is the actual motion feedforward.
        bases["ur10e"][0, 3] += .007
        states["ur10e"].joint_velocity[1] = .03
        states["ur10e"].speed[1] = .03
        accept(controller, bases, 10.01, 2)
        result = controller.prepare(10.04, states, .04, active=True)
        self.assertAlmostEqual(result.record["target_tcp"][0, 3], .007)
        self.assertGreater(result.record["leader_compensation"][1], .02)
        np.testing.assert_array_equal(result.record["base_compensation"], np.zeros(6))
        self.assertGreater(np.linalg.norm(result.decision.qdot), 0.)
        self.assert_bounded(controller, result, states, .04,
                            (first.decision.qdot, first.decision.twist))

    def test_pose_mode_keeps_raw_jump_and_measurement_age_guards(self):
        controller, bases, _ = scene(base_motion_mode="pose")
        accept(controller, bases)
        bases["ur5e"][0, 3] += .1
        self.assertFalse(controller.ingest(packet(controller, bases, 10.04, 2), 10.04))
        self.assertIn("calibration jump", controller.last_rejection)
        self.assertEqual(controller.gate.last_capture, 10.)
        with self.assertRaises(ResearchFault):
            controller.state(10.6, active=True)

    def test_sustained_base_shift_is_followed_after_delayed_reduced_robot_response(self):
        controller, bases, states = scene(
            base_motion_mode="pose", leader_response="bounded", filter_alpha=.2,
            max_accel=.1, max_angular_accel=.5, max_joint_accel=.5,
            target_tolerance_m=.05, target_tolerance_deg=10.,
            feedback_speed=.01, position_deadband=.001)
        accept(controller, bases)
        first = controller.prepare(10., states, .04, active=True)
        controller.commit(first)
        initial_target = first.record["target_tcp"].copy()
        reference = first.record["tracking_relative_target"].copy()
        previous = (first.decision.qdot, first.decision.twist)
        pending = first.decision.qdot.copy()
        start_base_x = bases["ur10e"][0, 3]
        nonzero = 0
        for tick in range(1, 201):
            now = 10. + tick * .04
            # The independent plant applies 60% of the previous command. The
            # leader does not move; its base transform shifts by 20 mm, then
            # stays there. Target tolerance must not absorb this later change.
            state = states["ur5e"]
            state.joint_velocity = pending * .6
            state.speed = state.joint_velocity.copy()
            state.joints += state.joint_velocity * .04
            state.tcp = controller.models["ur5e"].pose_and_jacobian(state.joints)[0]
            bases["ur10e"][0, 3] = start_base_x + min(tick, 10) * .002
            accept(controller, bases, now, tick + 1)
            proposal = controller.prepare(now, states, .04, active=True)
            self.assert_bounded(controller, proposal, states, .04, previous)
            np.testing.assert_array_equal(proposal.record["base_compensation"], np.zeros(6))
            np.testing.assert_allclose(proposal.record["tracking_relative_target"], reference, atol=1e-12)
            nonzero += np.linalg.norm(proposal.decision.qdot) > .0001
            controller.commit(proposal)
            pending = proposal.decision.qdot.copy()
            previous = (proposal.decision.qdot.copy(), proposal.decision.twist.copy())
        self.assertGreater(nonzero, 30)
        commanded_shift = proposal.record["target_tcp"][0, 3] - initial_target[0, 3]
        # The normal ACTIVE smoother deliberately leaves its visual-noise
        # deadband uncorrected. The remaining persistent shift must still
        # move the goal and the delayed follower must catch that goal.
        self.assertLessEqual(abs(commanded_shift - .02),
                             controller.args.base_pose_deadband_m + 1e-6)
        self.assertGreater(commanded_shift, .014)
        self.assertLess(abs(states["ur5e"].tcp[0, 3]
                            - proposal.record["target_tcp"][0, 3]), .002)

    def test_actual_following_sweep_can_be_valid_when_stationary_follower_would_collide(self):
        class FollowingClearanceGuard(PermissiveGuard):
            def check(self, bases, joints):
                # A translating leader can close clearance only if the
                # follower stays put; matched translation preserves it.
                clearance = .00002 - (joints["ur10e"][0] - joints["ur5e"][0])
                return clearance >= 0., "leader closes clearance", clearance

        for mode in ("pose", "velocity"):
            with self.subTest(mode=mode):
                controller, bases, states = scene(
                    base_motion_mode=mode, leader_response="bounded",
                    target_tolerance_m=.05, target_tolerance_deg=10.)
                controller.guard = FollowingClearanceGuard()
                accept(controller, bases)
                first = controller.prepare(10., states, .04, active=True)
                controller.commit(first)
                states["ur10e"].joint_velocity[0] = .002
                states["ur10e"].speed[0] = .002
                if mode == "velocity":
                    with self.assertRaisesRegex(ResearchFault, "mandatory compensation"):
                        controller.prepare(10.04, states, .04, active=True)
                    continue
                proposal = controller.prepare(10.04, states, .04, active=True)
                self.assertGreater(proposal.decision.qdot[0], .0015)
                for fraction in (.5, 1.):
                    trial = {name: state.joints + (proposal.decision.qdot if name == "ur5e"
                                                  else state.joint_velocity) * .04 * fraction
                             for name, state in states.items()}
                    self.assertTrue(controller.guard.check(bases, trial)[0])

    def test_pose_following_still_rejects_actual_predicted_collision(self):
        class ApproachingObstacleGuard(PermissiveGuard):
            def check(self, bases, joints):
                # This obstacle is in the leader's future path, so changing
                # the follower command cannot make the scene feasible.
                allowed = joints["ur10e"][0] <= .00002
                return allowed, "leader obstacle" if not allowed else "", 1.

        controller, bases, states = scene(
            base_motion_mode="pose", leader_response="bounded",
            target_tolerance_m=.05, target_tolerance_deg=10.)
        controller.guard = ApproachingObstacleGuard()
        accept(controller, bases)
        first = controller.prepare(10., states, .04, active=True)
        controller.commit(first)
        states["ur10e"].joint_velocity[0] = .002
        with self.assertRaises((ResearchFault, MotionInfeasible)) as caught:
            controller.prepare(10.04, states, .04, active=True)
        self.assertIn("clearance", str(caught.exception))

    def test_pose_mode_keeps_measured_joint_and_tcp_speed_guards(self):
        for field, index, speed, category in (("joint_velocity", 2, 1.21, "joint_speed"),
                                             ("speed", 0, .222, "tcp_speed"),
                                             ("speed", 3, .556, "tcp_speed")):
            with self.subTest(field=field, index=index):
                controller, bases, states = scene(base_motion_mode="pose")
                accept(controller, bases)
                getattr(states["ur5e"], field)[index] = speed
                with self.assertRaises(ResearchFault) as caught:
                    controller.prepare(10., states, .04, active=True)
                self.assertEqual(caught.exception.category, category)

    def test_zero_tolerance_pose_following_retains_future_inverse_jacobian_limits(self):
        for limit in ("speed", "acceleration"):
            with self.subTest(limit=limit):
                controller, bases, states = scene(
                    base_motion_mode="pose", leader_response="bounded",
                    target_tolerance_m=0., target_tolerance_deg=0.,
                    max_joint_speed=.003 if limit == "speed" else .2,
                    max_joint_accel=.1 if limit == "acceleration" else 1.)
                model = ShrinkingTranslationModel(controller.offsets["ur5e"])
                controller.models["ur5e"] = model
                states["ur10e"].joint_velocity[0] = .003
                accept(controller, bases)
                proposal = controller.prepare(10., states, .04, active=True)
                self.assertTrue(controller.bounded_leader)
                self.assertFalse(controller.approximate_targets)
                self.assertGreater(proposal.decision.twist[0], 0.)
                self.assertLess(proposal.decision.twist[0], .003 - 1e-9)
                for fraction in (.5, 1.):
                    trial_q = states["ur5e"].joints + proposal.decision.qdot * .04 * fraction
                    future_jacobian = model.pose_and_jacobian(trial_q)[1]
                    future_qdot = np.linalg.solve(future_jacobian, proposal.decision.twist)
                    self.assertLessEqual(np.max(np.abs(future_qdot)), controller.args.max_joint_speed + 1e-9)
                    self.assertLessEqual(np.max(np.abs(future_qdot)) / .04,
                                         controller.args.max_joint_accel + 1e-8)
                # Exact targets keep their native safety-check meaning. The
                # bounded command's endpoint must not replace this target.
                self.assertAlmostEqual(proposal.future_tcp_goals["ur5e"][0, 3], .003 * .04)


if __name__ == "__main__":
    unittest.main()
