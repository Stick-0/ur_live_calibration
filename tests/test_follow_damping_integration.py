"""Full follower controller/governor with an independent delayed motion plant."""

from collections import deque
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import numpy as np

from test_approximate_tracking import OffsetCartesianModel
from test_bounded_follow import bounded_scene, move_joint
from test_cooperative_controller import accept
from urcal.control.runtime import parse_args
from urcal.cooperative.feedback import damped_feedback_velocity


def damping_scene(damping=.12, **overrides):
    settings = dict(base_motion_mode="pose", follow_damping_time=damping,
                    kp_linear=3., kp_angular=3., feedback_speed=.1,
                    feedback_angular_speed=.15, follow_max_speed=1.,
                    max_joint_accel=.75)
    settings.update(overrides)
    controller, bases, states = bounded_scene(**settings)
    accept(controller, bases)
    first = controller.prepare(10., states, .04, active=True)
    controller.commit(first)
    return controller, bases, states


class FollowDampingIntegrationTests(unittest.TestCase):
    def test_cli_damping_default_compatibility_switch_and_invalid_values(self):
        arguments = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]
        self.assertEqual(parse_args(arguments).follow_damping_time, .12)
        self.assertEqual(parse_args([*arguments, "--follow_damping_time", "0"]).follow_damping_time, 0.)
        for invalid in ("-0.01", "nan", "inf"):
            with self.subTest(value=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([*arguments, "--follow_damping_time", invalid])

    def assert_command_limits(self, controller, proposal, states, dt):
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        limits = controller.limits
        self.assertLessEqual(np.linalg.norm(twist[:3]), limits.linear_speed + 1e-10)
        self.assertLessEqual(np.linalg.norm(twist[3:]), limits.angular_speed + 1e-10)
        self.assertTrue(np.all(abs(qdot) <= limits.joint_speed + 1e-10))
        for old_twist, old_qdot in ((controller.previous_twist, controller.previous_qdot),
                                   (states[controller.follower].speed,
                                    states[controller.follower].joint_velocity)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]),
                                 limits.linear_accel * dt + 1e-10)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]),
                                 limits.angular_accel * dt + 1e-10)
            self.assertTrue(np.all(abs(qdot - old_qdot) <= limits.joint_accel * dt + 1e-10))

    def step_response(self, damping, dt, delay):
        controller, bases, states = damping_scene(damping)
        move_joint(controller, states, "ur5e", 0, -.05)
        queue = deque([np.zeros(6) for _ in range(delay)])
        errors, velocities = [], []
        for tick in range(1, round(7. / dt)):
            follower = states["ur5e"]
            change = queue.popleft() - follower.joint_velocity
            # This independent plant has command latency and a physical
            # velocity ramp. It never reuses the production governor.
            follower.joint_velocity += change * min(1., .1 * dt / max(np.linalg.norm(change), 1e-20))
            follower.joints += follower.joint_velocity * dt
            follower.tcp, jacobian = controller.models["ur5e"].pose_and_jacobian(follower.joints)
            follower.speed = jacobian @ follower.joint_velocity
            now = 10. + tick * dt
            accept(controller, bases, at=now, sequence=tick + 1)
            proposal = controller.prepare(now, states, dt, active=True)
            self.assert_command_limits(controller, proposal, states, dt)
            self.assertGreater(proposal.record["capsule_clearance_m"], 0.)
            errors.append(-follower.joints[0])  # Positive before, negative beyond target.
            velocities.append(follower.speed[0])
            controller.commit(proposal)
            queue.append(proposal.decision.qdot.copy())
        errors = np.asarray(errors)
        remaining_peak = np.maximum.accumulate(abs(errors[::-1]))[::-1]
        settled = np.flatnonzero(remaining_peak < .002)
        self.assertGreater(len(settled), 0, "follower failed to settle")
        return dict(overshoot=max(0., -errors.min()), settled_at=(settled[0] + 1) * dt,
                    peak_speed=max(abs(np.asarray(velocities))), integrated_error=sum(abs(errors)) * dt)

    def test_full_governor_removes_signed_overshoot_without_slow_catch_up(self):
        for dt in (.04, .08):
            for delay in (1, 2):
                with self.subTest(dt=dt, delay=delay):
                    legacy = self.step_response(0., dt, delay)
                    damped = self.step_response(.12, dt, delay)
                    self.assertGreater(legacy["overshoot"], .01)
                    self.assertLess(damped["overshoot"], .001)
                    self.assertLess(damped["settled_at"], legacy["settled_at"] * .8)
                    self.assertLess(damped["integrated_error"], legacy["integrated_error"] * .9)
                    self.assertGreater(damped["peak_speed"], .04)

    def test_calibration_pose_change_is_not_used_as_measured_feedback_velocity(self):
        # Isolate the feedback law from the separate multi-frame coordinate
        # smoother; this test deliberately applies one immediate pose change.
        controller, bases, states = damping_scene(base_pose_window=0)
        # Eight millimetres in 40 ms would look like 0.2 m/s if vision poses
        # were differentiated. Both encoder states are initially stationary.
        bases["ur5e"][0, 3] += .008
        accept(controller, bases, at=10.04, sequence=2)
        with patch("urcal.cooperative.controller.damped_feedback_velocity",
                   wraps=damped_feedback_velocity) as damp:
            stationary = controller.prepare(10.04, states, .04, active=True)
        np.testing.assert_array_equal(stationary.record["feedback_relative_tcp_velocity"], np.zeros(6))
        np.testing.assert_array_equal(damp.call_args_list[0].args[1], np.zeros(3))
        self.assertLess(stationary.feedback[0], 0.)  # Pose correction still follows the new calibration.
        for name, speed in (("ur5e", .01), ("ur10e", .004)):
            states[name].joint_velocity[0] = speed
            states[name].speed[0] = speed
        controller.previous_twist = states["ur5e"].speed.copy()
        controller.previous_qdot = states["ur5e"].joint_velocity.copy()
        moving = controller.prepare(10.04, states, .04, active=True)
        np.testing.assert_allclose(moving.record["feedback_relative_tcp_velocity"], [.006, 0., 0., 0., 0., 0.],
                                   atol=1e-12)

    def test_matching_leader_encoder_motion_keeps_feedforward_undamped(self):
        controller, bases, states = damping_scene()
        for name in states:
            move_joint(controller, states, name, 0, .0012)
            states[name].joint_velocity[0] = .03
            states[name].speed[0] = .03
        controller.previous_twist = states["ur5e"].speed.copy()
        controller.previous_qdot = states["ur5e"].joint_velocity.copy()
        proposal = controller.prepare(10., states, .04, active=True)
        np.testing.assert_allclose(proposal.feedback, 0., atol=1e-12)
        np.testing.assert_allclose(proposal.record["feedback_relative_tcp_velocity"], 0., atol=1e-12)
        np.testing.assert_allclose(proposal.decision.twist, [.03, 0., 0., 0., 0., 0.], atol=1e-12)
        self.assert_command_limits(controller, proposal, states, .04)

    def test_long_tcp_rotation_is_not_mistaken_for_flange_translation(self):
        controller, bases, states = damping_scene(ur5e_tcp_offset=[.4, .07, .1, .1, -.2, .05])
        controller.models = {name: OffsetCartesianModel(controller.offsets[name]) for name in states}
        move_joint(controller, states, "ur5e", 0, -.01)
        follower = states["ur5e"]
        follower.joint_velocity[5] = .02
        _, jacobian = controller.models["ur5e"].pose_and_jacobian(follower.joints)
        follower.speed = jacobian @ follower.joint_velocity
        self.assertGreater(np.linalg.norm(follower.speed[:3]), .004)
        controller.previous_twist, controller.previous_qdot = follower.speed.copy(), follower.joint_velocity.copy()
        with patch("urcal.cooperative.controller.damped_feedback_velocity",
                   wraps=damped_feedback_velocity) as damp:
            proposal = controller.prepare(10., states, .04, active=True)
        self.assertEqual(damp.call_count, 2)
        # Rotation moves the long tool's TCP, but the flange origin has zero
        # physical linear velocity. Damping must use that flange velocity.
        np.testing.assert_allclose(damp.call_args_list[0].args[1], np.zeros(3), atol=3e-9)
        np.testing.assert_allclose(damp.call_args_list[1].args[1], follower.speed[3:], atol=1e-12)
        self.assertGreater(proposal.feedback[0], .005)
        self.assert_command_limits(controller, proposal, states, .04)

    def test_zero_position_error_requests_braking_without_abrupt_velocity_step(self):
        controller, _bases, states = damping_scene()
        follower = states["ur5e"]
        follower.joint_velocity[0] = follower.speed[0] = .03
        controller.previous_qdot = follower.joint_velocity.copy()
        controller.previous_twist = follower.speed.copy()
        controller.previous_feedback[0] = .03  # Old correction must not wind up.
        proposal = controller.prepare(10., states, .04, active=True)
        np.testing.assert_allclose(proposal.feedback, 0., atol=1e-12)
        self.assertGreater(proposal.decision.twist[0], 0.)
        self.assertLess(proposal.decision.twist[0], .03)
        self.assert_command_limits(controller, proposal, states, .04)


if __name__ == "__main__":
    unittest.main()
