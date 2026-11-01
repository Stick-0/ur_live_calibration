"""Independent follow TCP speed limits, exercised without hardware interfaces."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest

import numpy as np

from test_bounded_follow import bounded_scene
from test_cooperative_controller import CartesianModel, accept
from test_live_alignment import fixture, make_plan
from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController, ResearchFault


CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2"]
UNCHANGED_LIMITS = {
    "max_speed": .02, "max_angular_speed": .1, "max_joint_speed": .2,
    "max_accel": .1, "max_angular_accel": .5,
}


class GrowingTranslationModel(CartesianModel):
    """Differentiable kinematics exposing a future Cartesian speed increase."""

    def __init__(self, offset, curvature):
        super().__init__(offset)
        self.curvature = curvature

    def pose_and_jacobian(self, joints, flange=False):
        pose, jacobian = super().pose_and_jacobian(joints, flange)
        q = float(joints[0])
        pose[0, 3] = q + .5 * self.curvature * q * q
        jacobian[0, 0] = 1. + self.curvature * q
        return pose, jacobian


def set_velocity(controller, states, name, velocity):
    state = states[name]
    state.joint_velocity = np.asarray(velocity, dtype=float).copy()
    state.tcp, jacobian = controller.models[name].pose_and_jacobian(state.joints)
    state.speed = jacobian @ state.joint_velocity


def seed_following(controller, states, follower_velocity, leader_velocity):
    set_velocity(controller, states, controller.follower, follower_velocity)
    set_velocity(controller, states, controller.leader, leader_velocity)
    controller.previous_qdot = states[controller.follower].joint_velocity.copy()
    controller.previous_twist = states[controller.follower].speed.copy()


class FollowSpeedLimitTests(unittest.TestCase):
    def assert_command_limits(self, controller, proposal, states, dt):
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        args = controller.args
        self.assertLessEqual(np.linalg.norm(twist[:3]), controller.limits.linear_speed + 1e-10)
        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed + 1e-10)
        self.assertLessEqual(np.max(np.abs(qdot)), args.max_joint_speed + 1e-10)
        for old_twist, old_qdot in (
                (controller.previous_twist, controller.previous_qdot),
                (states[controller.follower].speed, states[controller.follower].joint_velocity)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]),
                                 args.max_accel * dt + 1e-10)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]),
                                 args.max_angular_accel * dt + 1e-10)
            self.assertLessEqual(np.max(np.abs(qdot - old_qdot)),
                                 args.max_joint_accel * dt + 1e-10)

    def test_default_follow_tcp_limit_preserves_other_physical_motion_limits(self):
        for mode in ("follow",):
            args = parse_args(["--mode", mode] + CLI[2:])
            self.assertEqual(args.follow_max_speed, 1.)
            for name, value in UNCHANGED_LIMITS.items():
                self.assertEqual(getattr(args, name), value, (mode, name))

    def test_faster_follow_response_defaults(self):
        for mode, expected in (("follow", (.10, .15, 3., 3.)),):
            with self.subTest(mode=mode):
                args = parse_args(["--mode", mode] + CLI[2:])
                self.assertEqual((args.feedback_speed, args.feedback_angular_speed,
                                  args.kp_linear, args.kp_angular), expected)

    def test_explicit_response_tuning_wins(self):
        for mode in ("follow",):
            with self.subTest(mode=mode):
                args = parse_args(["--mode", mode] + CLI[2:] + [
                    "--feedback_speed", ".025", "--feedback_angular_speed", ".07",
                    "--kp_linear", "1.2", "--kp_angular", "1.5"])
                self.assertEqual((args.feedback_speed, args.feedback_angular_speed,
                                  args.kp_linear, args.kp_angular), (.025, .07, 1.2, 1.5))

    def test_partial_response_override_retains_other_defaults(self):
        args = parse_args(CLI + ["--feedback_speed", ".025"])
        self.assertEqual(args.feedback_speed, .025)
        self.assertEqual((args.feedback_angular_speed, args.kp_linear, args.kp_angular), (.15, 3., 3.))

    def test_response_tuning_still_requires_finite_positive_values(self):
        for name in ("feedback_speed", "feedback_angular_speed", "kp_linear", "kp_angular"):
            for value in ("0", "-0.1", "nan", "inf"):
                with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(CLI + ["--" + name + "=" + value])

    def test_cli_override_precedence_preserves_explicit_legacy_speed(self):
        cases = (
            (["--max_speed", ".08"], .08, .08),
            (["--follow_max_speed", ".5"], .02, .5),
            (["--follow_max_speed", ".3"], .02, .3),
            (["--max_speed", ".08", "--follow_max_speed", ".3"], .08, .3),
            (["--follow_max_speed", ".3", "--max_speed", ".08"], .08, .3),
        )
        for options, shared, following in cases:
            with self.subTest(options=options):
                args = parse_args(CLI + options)
                self.assertEqual(args.max_speed, shared)
                self.assertEqual(args.follow_max_speed, following)

    def test_cli_rejects_invalid_follow_and_shared_speed_limits(self):
        for option in ("--follow_max_speed", "--max_speed"):
            for value in ("0", "-.1", "nan", "inf", "-inf"):
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(CLI + [option + "=" + value])

    def test_missing_and_none_follow_limit_retain_saved_namespace_behavior(self):
        for missing in (False, True):
            controller, _, _ = bounded_scene(follow_max_speed=None, max_speed=.035)
            args = deepcopy(controller.args)
            if missing:
                del args.follow_max_speed
            restored = CooperativeController(args, models=controller.models, guard=controller.guard)
            self.assertEqual(restored.limits.linear_speed, .035)
            self.assertEqual(args.max_speed, .035)

    def test_policy_rejects_invalid_follow_override_without_cli_validation(self):
        for value in (0., -.1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bounded_scene(follow_max_speed=value)

    def test_follow_override_does_not_mutate_original_args_or_alignment_caps(self):
        values = fixture("follow")
        args = values[0]
        args.max_speed = .02
        args.follow_max_speed = .5
        before = deepcopy(vars(args))
        plan = make_plan(values)
        self.assertEqual(plan.limits["max_speed"], .02)
        self.assertEqual(vars(args), before)
        controller, _, _ = bounded_scene(follow_max_speed=.5)
        self.assertEqual(controller.args.max_speed, .02)
        self.assertEqual(controller.limits.linear_speed, .5)

    def test_measured_40_mm_per_second_and_its_future_command_are_accepted(self):
        controller, bases, states = bounded_scene(follow_max_speed=.5)
        seed_following(controller, states, [.04, 0, 0, 0, 0, 0], [.04, 0, 0, 0, 0, 0])
        accept(controller, bases)
        proposal = controller.prepare(10., states, .04, active=True)
        self.assertAlmostEqual(proposal.decision.twist[0], .04)
        self.assertGreater(proposal.decision.twist[0], controller.args.max_speed)
        self.assert_command_limits(controller, proposal, states, .04)
        self.assertEqual(proposal.record["follow_max_speed"], .5)

    def test_measured_tcp_speed_above_follow_limit_retains_stop_check(self):
        for cap in (.5, 1.):
            with self.subTest(cap=cap):
                controller, bases, states = bounded_scene(follow_max_speed=cap, max_joint_speed=2.)
                excessive = cap * 1.1 + .002
                seed_following(controller, states, [excessive, 0, 0, 0, 0, 0],
                               [excessive, 0, 0, 0, 0, 0])
                accept(controller, bases)
                with self.assertRaises(ResearchFault) as caught:
                    controller.prepare(10., states, .04, active=True)
                self.assertEqual(caught.exception.category, "tcp_speed")

    def test_high_demand_stays_at_configured_cap_when_other_limits_have_headroom(self):
        # Raise only this synthetic plant's joint speed so the TCP cap itself
        # can be reached; production joint limits remain tested separately.
        for cap in (.5, 1.):
            with self.subTest(cap=cap):
                controller, bases, states = bounded_scene(follow_max_speed=cap, max_joint_speed=2.)
                seed_following(controller, states, [cap, 0, 0, 0, 0, 0],
                               [cap + .2, 0, 0, 0, 0, 0])
                accept(controller, bases)
                proposal = controller.prepare(10., states, .04, active=True)
                self.assertAlmostEqual(proposal.decision.twist[0], cap)
                self.assertGreater(proposal.record["leader_compensation"][0], cap)
                self.assert_command_limits(controller, proposal, states, .04)

    def test_follow_from_rest_exceeds_old_limit_with_original_acceleration(self):
        controller, bases, states = bounded_scene(follow_max_speed=.5)
        dt = .04
        peak = 0.
        for tick in range(20):
            now = 10. + tick * dt
            set_velocity(controller, states, controller.leader, [.1, 0, 0, 0, 0, 0])
            accept(controller, bases, now, tick + 1)
            proposal = controller.prepare(now, states, dt, active=True)
            self.assert_command_limits(controller, proposal, states, dt)
            peak = max(peak, np.linalg.norm(proposal.decision.twist[:3]))
            controller.commit(proposal)
            for name, velocity in ((controller.leader, states[controller.leader].joint_velocity),
                                   (controller.follower, proposal.decision.qdot)):
                states[name].joints += velocity * dt
                set_velocity(controller, states, name, velocity)
        self.assertGreater(peak, .05)
        self.assertLessEqual(peak, .1 * 20 * dt + 1e-10)

    def test_joint_and_angular_caps_still_reduce_larger_leader_demands(self):
        cases = (
            ([.2, 0, 0, 0, 0, 0], [.4, 0, 0, 0, 0, 0], 0, .2),
            ([0, 0, 0, 0, 0, .1], [0, 0, 0, 0, 0, .3], 5, .1),
        )
        for current, leader, index, expected in cases:
            with self.subTest(index=index):
                controller, bases, states = bounded_scene(follow_max_speed=.5)
                seed_following(controller, states, current, leader)
                accept(controller, bases)
                proposal = controller.prepare(10., states, .04, active=True)
                self.assertAlmostEqual(proposal.decision.twist[index], expected)
                self.assertLess(proposal.decision.twist[index], leader[index])
                self.assert_command_limits(controller, proposal, states, .04)

    def test_growing_future_jacobian_uses_follow_cap_without_dropping_future_checks(self):
        cases = ((.05, .04, .06, 100.), (.5, .49, .6, 1.), (1., .99, 1.1, .1))
        for cap, measured, demand, curvature in cases:
            with self.subTest(cap=cap):
                # Give acceleration headroom so the rejection comes from
                # future TCP speed, independently of the default ramp tests.
                controller, bases, states = bounded_scene(
                    follow_max_speed=cap, max_joint_speed=1., max_accel=1.)
                model = GrowingTranslationModel(controller.offsets[controller.follower], curvature)
                controller.models[controller.follower] = model
                seed_following(controller, states, [measured, 0, 0, 0, 0, 0],
                               [demand, 0, 0, 0, 0, 0])
                accept(controller, bases)
                dt = .04
                unconstrained = min(cap, measured + controller.args.max_accel * dt)
                trial = states[controller.follower].joints.copy()
                trial[0] += unconstrained * dt
                raw_future_speed = model.pose_and_jacobian(trial)[1][0, 0] * unconstrained
                self.assertGreater(raw_future_speed, cap)
                proposal = controller.prepare(10., states, dt, active=True)
                self.assertGreater(proposal.decision.twist[0], controller.args.max_speed)
                self.assertLess(proposal.decision.twist[0], unconstrained - 1e-9)
                self.assert_command_limits(controller, proposal, states, dt)
                for fraction in (.5, 1.):
                    trial = states[controller.follower].joints + proposal.decision.qdot * dt * fraction
                    jacobian = model.pose_and_jacobian(trial)[1]
                    self.assertLessEqual(np.linalg.norm((jacobian @ proposal.decision.qdot)[:3]),
                                         cap + 1e-9)


if __name__ == "__main__":
    unittest.main()
