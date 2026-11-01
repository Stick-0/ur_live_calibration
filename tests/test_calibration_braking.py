"""Frozen-geometry braking uses encoder motion and retains physical checks."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_bounded_follow import bounded_scene, move_joint, recorded_scene
from test_cooperative_controller import accept
from test_follow_future_jacobian import ShrinkingTranslationModel
from urcal.control.geometry import inverse
from urcal.control.policy import ControlFault
from urcal.cooperative.calibration_braking import frozen_braking_snapshot, prepare_calibration_braking
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.task import RelativePath


def braking_scene():
    controller, bases, states = bounded_scene(base_motion_mode="pose", follow_damping_time=.12,
                                              follow_max_speed=1., max_joint_accel=.75)
    accept(controller, bases)
    first = controller.prepare(10., states, .04, active=True)
    controller.commit(first)
    snapshot = controller.state(10., active=True)
    follower = states[controller.follower]
    follower.joint_velocity[0] = follower.speed[0] = .03
    controller.previous_twist = follower.speed.copy()
    controller.previous_qdot = follower.joint_velocity.copy()
    return controller, states, snapshot


class FrozenBrakingSnapshotTests(unittest.TestCase):
    def test_geometry_is_copied_without_visual_velocity_or_fake_freshness(self):
        controller, states, snapshot = braking_scene()
        original = deepcopy(snapshot)
        snapshot.base_twists["ur5e"][:] = .1
        snapshot.relative_twist[:] = .2
        result = frozen_braking_snapshot(snapshot, 11., leader=controller.leader)
        self.assertEqual(result.at, snapshot.at)
        self.assertEqual(result.measurement_times, snapshot.measurement_times)
        self.assertEqual(result.received_times, snapshot.received_times)
        self.assertAlmostEqual(result.measurement_age["ur5e"], 1.)
        for name in states:
            np.testing.assert_array_equal(result.base_twists[name], np.zeros(6))
            np.testing.assert_array_equal(result.base_poses[name], original.base_poses[name])
            self.assertEqual(result.prediction_dt[name], 0.)
        np.testing.assert_array_equal(result.relative_twist, np.zeros(6))
        result.base_poses["ur5e"][0, 3] += 99.
        result.measurement_times["ur5e"] = -99.
        np.testing.assert_array_equal(snapshot.base_poses["ur5e"], original.base_poses["ur5e"])
        self.assertEqual(snapshot.measurement_times, original.measurement_times)

    def test_invalid_geometry_or_future_timestamps_cannot_be_used_for_braking(self):
        controller, _states, snapshot = braking_scene()
        for field, value in (("measurement_times", np.inf), ("received_times", 10.1)):
            bad = deepcopy(snapshot)
            getattr(bad, field)["ur5e"] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                frozen_braking_snapshot(bad, 10., leader=controller.leader)
        bad = deepcopy(snapshot)
        bad.base_poses["ur5e"][0, 0] = 2.
        with self.assertRaises(ValueError):
            frozen_braking_snapshot(bad, 10., leader=controller.leader)
        for now in (9., np.nan, np.inf):
            with self.subTest(now=now), self.assertRaises(ValueError):
                frozen_braking_snapshot(snapshot, now, leader=controller.leader)

    def test_wrapper_proposes_only_and_preserves_saved_snapshot(self):
        controller, states, snapshot = braking_scene()
        sentinel = SimpleNamespace(record={})
        fake = SimpleNamespace(leader=controller.leader, prepare=Mock(return_value=sentinel), commit=Mock())
        result = prepare_calibration_braking(fake, 10.4, states, .04, snapshot)
        self.assertIs(result, sentinel)
        fake.commit.assert_not_called()
        args, kwargs = fake.prepare.call_args
        self.assertEqual(args, (10.4, states, .04))
        self.assertTrue(kwargs["active"])
        self.assertTrue(kwargs["calibration_braking"])
        self.assertIsNot(kwargs["snapshot_override"], snapshot)
        self.assertTrue(result.record["calibration_geometry_frozen"])
        self.assertEqual(result.record["calibration_geometry_at"], snapshot.at)


class CalibrationBrakingTests(unittest.TestCase):
    def assert_bounded(self, controller, states, proposal, dt):
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        limits = controller.limits
        self.assertLessEqual(np.linalg.norm(twist[:3]), limits.linear_speed + 1e-9)
        self.assertLessEqual(np.linalg.norm(twist[3:]), limits.angular_speed + 1e-9)
        self.assertTrue(np.all(abs(qdot) <= limits.joint_speed + 1e-9))
        for old_twist, old_qdot in ((controller.previous_twist, controller.previous_qdot),
                                   (states[controller.follower].speed,
                                    states[controller.follower].joint_velocity)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]), limits.linear_accel * dt + 1e-9)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]), limits.angular_accel * dt + 1e-9)
            self.assertTrue(np.all(abs(qdot - old_qdot) <= limits.joint_accel * dt + 1e-9))

    def test_braking_ignores_invalid_visual_target_but_keeps_measured_motion_limits(self):
        controller, states, snapshot = braking_scene()
        move_joint(controller, states, controller.leader, 0, .14)
        states[controller.leader].joint_velocity[0] = .06
        controller.previous_base_compensation[0] = .02
        controller.previous_base_compensation_qdot[0] = .02
        with self.assertRaisesRegex(ResearchFault, "relative target error"):
            controller.prepare(10., states, .04, active=True)
        with patch.object(controller, "state", side_effect=ResearchFault("measurement_expired", "camera expired")):
            proposal = prepare_calibration_braking(controller, 10.8, states, .04, snapshot)
        self.assert_bounded(controller, states, proposal, .04)
        self.assertAlmostEqual(proposal.decision.twist[0], .026)
        for name in ("feedback", "base_compensation", "leader_compensation", "task_twist_at_command"):
            np.testing.assert_allclose(proposal.record[name], np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(proposal.tcp_goals[controller.follower], states[controller.follower].tcp)
        self.assertEqual(proposal.next_progress, controller.progress)
        self.assertEqual(proposal.decision.task_scale, 0.)

    def test_repeated_braking_reaches_rest_without_recapturing_path_or_committing_failed_send(self):
        controller, states, snapshot = braking_scene()
        controller.path = RelativePath(controller.path.start, kind="line", delta=[.03, 0, 0, 0, 0, 0], duration=5.)
        controller.progress = .3
        reference, offset = controller.path.start.copy(), controller.tracking_offset.copy()
        first = prepare_calibration_braking(controller, 10.1, states, .04, snapshot)
        np.testing.assert_allclose(controller.previous_twist, states[controller.follower].speed)
        self.assertEqual(controller.progress, .3)
        for tick in range(20):
            proposal = prepare_calibration_braking(controller, 10.1 + tick * .04, states, .04, snapshot)
            self.assert_bounded(controller, states, proposal, .04)
            self.assertLessEqual(np.linalg.norm(proposal.decision.twist), np.linalg.norm(controller.previous_twist) + 1e-12)
            controller.commit(proposal)
            follower = states[controller.follower]
            follower.joint_velocity = proposal.decision.qdot.copy()
            follower.joints += follower.joint_velocity * .04
            follower.tcp, jacobian = controller.models[controller.follower].pose_and_jacobian(follower.joints)
            follower.speed = jacobian @ follower.joint_velocity
        np.testing.assert_allclose(controller.previous_twist, np.zeros(6), atol=1e-12)
        np.testing.assert_array_equal(controller.path.start, reference)
        np.testing.assert_array_equal(controller.tracking_offset, offset)
        self.assertEqual(controller.progress, .3)
        self.assertFalse(controller.finished)

    def test_braking_retains_frame_state_workspace_and_clearance_rejection(self):
        for failure in ("frame", "joint_speed", "tcp_speed", "workspace", "clearance"):
            with self.subTest(failure=failure):
                controller, states, snapshot = braking_scene()
                if failure == "frame":
                    states[controller.follower].tcp[0, 3] += .1
                elif failure == "joint_speed":
                    states[controller.follower].joint_velocity[0] = 1.
                elif failure == "tcp_speed":
                    states[controller.follower].speed[0] = 2.
                elif failure == "workspace":
                    controller.initial_tcp[0, 3] -= 1.
                else:
                    controller.guard.allowed = False
                with self.assertRaises(ControlFault):
                    prepare_calibration_braking(controller, 10.1, states, .04, snapshot)

    def test_leader_sweep_is_still_checked_while_leader_feedforward_is_disabled(self):
        controller, states, snapshot = braking_scene()
        states[controller.leader].joint_velocity[0] = .06
        initial_leader_q = states[controller.leader].joints.copy()
        def clearance(_bases, joints):
            allowed = joints[controller.leader][0] <= initial_leader_q[0] + .0001
            return allowed, "leader sweep enters obstacle" if not allowed else "", .1
        controller.guard.check = clearance
        with self.assertRaisesRegex(ControlFault, "sweep|clearance"):
            prepare_calibration_braking(controller, 10.1, states, .04, snapshot)

    def test_exact_target_braking_anchors_current_jacobian_and_logs_actual_command(self):
        controller, states, snapshot = braking_scene()
        controller.args.target_tolerance_m = controller.args.target_tolerance_deg = 0.
        controller.target_tolerance_m = controller.target_tolerance_rad = 0.
        controller.approximate_targets = False
        follower = states[controller.follower]
        # A small measured-vs-nominal TCP translation is accepted by the
        # existing frame check. Pure joint rotation moves that measured TCP
        # about a different point, so speedL needs the anchored current J.
        follower.tcp[0, 3] += .015
        nominal, jacobian = controller.models[controller.follower].pose_and_jacobian(follower.joints)
        anchored_jacobian = jacobian.copy()
        anchored_jacobian[:3] += np.cross(jacobian[3:].T, follower.tcp[:3, 3] - nominal[:3, 3]).T
        follower.joint_velocity[:] = [0., 0., 0., 0., 0., .05]
        follower.speed = anchored_jacobian @ follower.joint_velocity
        controller.previous_twist, controller.previous_qdot = follower.speed.copy(), follower.joint_velocity.copy()
        controller.models[controller.follower].inverse = Mock(
            side_effect=AssertionError("exact nominal target IK must not run during calibration braking"))
        proposal = prepare_calibration_braking(controller, 10.1, states, .04, snapshot)
        self.assert_bounded(controller, states, proposal, .04)
        np.testing.assert_allclose(anchored_jacobian @ proposal.decision.qdot,
                                   proposal.decision.twist, atol=1e-12)
        # This nonzero lever-arm contribution makes the regression distinguish
        # anchored Jacobian braking from merely using the nominal Jacobian.
        self.assertGreater(np.linalg.norm(jacobian @ proposal.decision.qdot - proposal.decision.twist), 1e-4)
        np.testing.assert_allclose(proposal.tcp_goals[controller.follower], follower.tcp)
        np.testing.assert_allclose(proposal.record["commanded_relative_pose"],
                                   proposal.record["estimated_relative_pose"])
        self.assertFalse(np.allclose(proposal.record["commanded_relative_pose"],
                                     proposal.record["desired_relative_pose"]))
        self.assertEqual(proposal.record["native_safety_target_mode"], "current_and_predicted_motion")
        controller.models[controller.follower].inverse.assert_not_called()

    def test_future_inverse_jacobian_is_not_bypassed_to_make_braking_feasible(self):
        controller, states, snapshot = braking_scene()
        controller.models[controller.follower] = ShrinkingTranslationModel(controller.offsets[controller.follower])
        states[controller.follower].joint_velocity[0] = states[controller.follower].speed[0] = .003
        controller.previous_qdot = states[controller.follower].joint_velocity.copy()
        controller.previous_twist = states[controller.follower].speed.copy()
        controller.args.max_joint_accel = .01
        controller.limits = replace(controller.limits, joint_accel=np.full(6, .01))
        with self.assertRaises(ControlFault):
            prepare_calibration_braking(controller, 10.1, states, .04, snapshot)

    def test_recorded_ur_geometry_brakes_with_current_and_future_joint_limits(self):
        controller, states, data = recorded_scene()
        snapshot = controller.state(data["control_at"], active=True)
        proposal = prepare_calibration_braking(controller, data["control_at"], states, data["dt"], snapshot)
        self.assert_bounded(controller, states, proposal, data["dt"])
        self.assertEqual(proposal.decision.task_scale, 0.)
        follower, model = controller.follower, controller.models[controller.follower]
        nominal = model.pose_and_jacobian(states[follower].joints)[0]
        correction = inverse(nominal) @ states[follower].tcp
        for fraction in (.5, 1.):
            q = states[follower].joints + proposal.decision.qdot * data["dt"] * fraction
            predicted, jacobian = model.pose_and_jacobian(q)
            anchored = predicted @ correction
            jacobian[:3] += np.cross(jacobian[3:].T, anchored[:3, 3] - predicted[:3, 3]).T
            future_qdot = np.linalg.solve(jacobian, proposal.decision.twist)
            self.assertTrue(np.all(abs(future_qdot) <= controller.limits.joint_speed + 1e-9))
            for previous in (controller.previous_qdot, states[follower].joint_velocity):
                self.assertTrue(np.all(abs(future_qdot - previous) <= controller.limits.joint_accel * data["dt"] + 1e-9))


if __name__ == "__main__":
    unittest.main()
