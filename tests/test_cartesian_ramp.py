"""Local Cartesian ramp prediction and SDK-free speedL transport checks."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from urcal.control.cartesian_ramp import bounded_speedl_acceleration
from urcal.control.policy import ControlFault
from urcal.control.runtime import MotionSession
from urcal.cooperative.kinematics import NominalKinematics
from urcal.cooperative.task import MotionLimits
from test_dual_control_runtime import args, controller


class CartesianRampTests(unittest.TestCase):
    def limits(self, **updates):
        values = dict(linear_accel=1., angular_accel=1., joint_accel=np.full(6, .5))
        values.update(updates)
        return MotionLimits(**values)

    def test_unchanged_velocity_needs_no_transition_and_preserves_positive_scalar(self):
        velocity = np.array([.01, 0., 0., 0., -.02, 0.])
        self.assertEqual(bounded_speedl_acceleration(
            np.eye(6), velocity, velocity, .04, self.limits(linear_accel=.1, angular_accel=.05)), .05)

    def test_longer_cartesian_ramp_takes_at_least_one_control_period(self):
        target = np.array([.002, 0., 0., 0., .004, 0.])
        result = bounded_speedl_acceleration(np.eye(6), np.zeros(6), target, .04, self.limits())
        self.assertAlmostEqual(result, .1)
        self.assertLess(np.linalg.norm(target[:3]) / result, .04)
        self.assertGreaterEqual(np.linalg.norm(target[3:]) / result, .04 - 1e-12)
        # A zero translation change must not prohibit a rotation-only ramp.
        target[:3] = 0.
        self.assertAlmostEqual(bounded_speedl_acceleration(
            np.eye(6), np.zeros(6), target, .04, self.limits()), .1)

    def test_tiny_angular_noise_does_not_stall_meaningful_translation(self):
        limits = self.limits(linear_accel=.1, angular_accel=.5)
        target = np.array([.004, 0., 0., 0., 1e-6, 0.])
        result = bounded_speedl_acceleration(np.eye(6), np.zeros(6), target, .04, limits)
        self.assertAlmostEqual(result, .1)
        self.assertAlmostEqual(np.linalg.norm(target[:3]) / result, .04)
        # Rotation finishes early; its standalone and combined acceleration
        # phases still stay below the configured per-joint limits.
        self.assertLessEqual(result, limits.joint_accel.min())

    def test_later_single_group_phase_is_checked_when_combined_joint_acceleration_cancels(self):
        inverse_jacobian = np.eye(6)
        inverse_jacobian[0, 0] = 10.
        inverse_jacobian[0, 3] = -10.
        J = np.linalg.inv(inverse_jacobian)
        result = bounded_speedl_acceleration(J, np.zeros(6), [.1, 0., 0., .2, 0., 0.], .04, self.limits())
        self.assertAlmostEqual(result, .05)
        self.assertAlmostEqual((inverse_jacobian @ np.array([1., 0., 0., 1., 0., 0.]))[0], 0.)
        self.assertAlmostEqual(result * abs(inverse_jacobian[0, 3]), .5)

    def test_combined_phase_is_checked_when_linear_and_angular_joint_accelerations_add(self):
        inverse_jacobian = np.eye(6)
        inverse_jacobian[0, 0] = 10.
        inverse_jacobian[0, 3] = 10.
        J = np.linalg.inv(inverse_jacobian)
        result = bounded_speedl_acceleration(J, np.zeros(6), [.1, 0., 0., .2, 0., 0.], .04, self.limits())
        self.assertAlmostEqual(result, .025)

    def test_independent_linear_angular_ramps_stay_within_local_joint_acceleration(self):
        inverse_jacobian = np.eye(6)
        inverse_jacobian[0, 0] = 10.
        inverse_jacobian[0, 3] = -10.
        J = np.linalg.inv(inverse_jacobian)
        target = np.array([.01, 0., 0., .03, 0., 0.])
        limits = self.limits()
        acceleration = bounded_speedl_acceleration(J, np.zeros(6), target, .04, limits)
        twist = np.zeros(6)
        period = .001
        saw_linear_finished = False
        for _ in range(1000):
            previous = twist.copy()
            for selection in (slice(0, 3), slice(3, 6)):
                change = target[selection] - twist[selection]
                length = np.linalg.norm(change)
                if length > 0.:
                    twist[selection] += change * min(1., acceleration * period / length)
            joint_acceleration = inverse_jacobian @ ((twist - previous) / period)
            self.assertTrue(np.all(np.abs(joint_acceleration) <= limits.joint_accel + 1e-11))
            if np.array_equal(twist[:3], target[:3]) and not np.array_equal(twist[3:], target[3:]):
                saw_linear_finished = True
        self.assertTrue(saw_linear_finished)
        np.testing.assert_allclose(twist, target, atol=1e-12)

    def test_recorded_leader_start_uses_a_ramp_that_accounts_for_joint_amplification(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "follow_leader_start.json").read_text())
        state = fixture["robots"]["ur5e"]
        pose, J = NominalKinematics("ur5e", np.eye(4)).pose_and_jacobian(np.asarray(state["q"]))
        J[:3] += np.cross(J[3:].T, np.asarray(state["tcp"])[:3, 3] - pose[:3, 3]).T
        measured = np.asarray(state["tcp_speed"])
        target = np.asarray(fixture["expected_original_failure"]["demanded_twist"])
        limits = MotionLimits()
        acceleration = bounded_speedl_acceleration(J, measured, target, fixture["dt"], limits)
        delta = target - measured
        linear = np.r_[delta[:3] / np.linalg.norm(delta[:3]), np.zeros(3)]
        angular = np.r_[np.zeros(3), delta[3:] / np.linalg.norm(delta[3:])]
        phases = np.linalg.solve(J, np.column_stack((linear, angular, linear + angular)))
        self.assertGreater(np.max(np.abs(phases) * limits.linear_accel), limits.joint_accel.max())
        self.assertTrue(np.all(np.abs(phases) * acceleration <= limits.joint_accel[:, None] + 1e-12))
        self.assertGreater(acceleration, 0.)
        self.assertLess(acceleration, limits.linear_accel)

    def test_tiny_nonzero_step_is_not_mistaken_for_zero_or_replaced_by_larger_floor(self):
        target = np.zeros(6)
        target[0] = 1e-300
        result = bounded_speedl_acceleration(np.eye(6), np.zeros(6), target, .1, self.limits())
        self.assertGreater(result, 0.)
        self.assertAlmostEqual(result / 1e-299, 1.)

    def test_bad_inputs_and_singular_jacobians_fail_before_transport(self):
        base = dict(J=np.eye(6), measured_twist=np.zeros(6), target_twist=np.ones(6) * .01,
                    dt=.04, limits=self.limits())
        malformed_limits = SimpleNamespace(linear_accel=.1, angular_accel=.5,
                                           joint_accel=np.array([.5, .5, 0., .5, .5, .5]),
                                           min_singular_value=.01)
        for update in (
                {"J": np.zeros((6, 6))}, {"J": np.eye(6) * .001}, {"J": np.eye(3)},
                {"J": np.full((6, 6), np.nan)}, {"dt": 0.}, {"dt": np.inf},
                {"dt": [1.]}, {"target_twist": [0.] * 5},
                {"target_twist": [np.inf] * 6}, {"measured_twist": [np.nan] * 6},
                {"limits": malformed_limits}, {"limits": SimpleNamespace()}):
            with self.subTest(update=update), self.assertRaises(ControlFault):
                bounded_speedl_acceleration(**(base | update))


class CartesianRampTransportTests(unittest.TestCase):
    def session(self):
        config = args()
        config.max_accel = .1
        config.max_angular_accel = .05
        controls = {name: controller() for name in ("ur5e", "ur10e")}
        session = MotionSession(config, tuple(controls), {name: np.eye(4) for name in controls},
                                factory=Mock(side_effect=list(controls.values())))
        self.addCleanup(session.close)
        poses = {name: np.eye(4) for name in controls}
        twists = {name: np.zeros(6) for name in controls}
        return session, controls, poses, twists

    def test_per_arm_ramps_reach_speedl_unchanged_without_extra_watchdog_kicks(self):
        session, controls, poses, twists = self.session()
        accelerations = {"ur5e": .013, "ur10e": .027}
        session.send(poses, twists, accelerations=accelerations)
        for name, control in controls.items():
            control.speedL.assert_called_once_with([0.] * 6, accelerations[name], .002)
            control.kickWatchdog.assert_not_called()

    def test_existing_callers_keep_their_prior_default(self):
        session, controls, poses, twists = self.session()
        session.send(poses, twists)
        for control in controls.values():
            control.speedL.assert_called_once_with([0.] * 6, .1, .002)

    def test_bad_second_arm_ramp_prevents_any_motion(self):
        for invalid in (0., -.01, .050001, np.nan, np.inf, [0.01], None, "bad"):
            with self.subTest(invalid=invalid):
                session, controls, poses, twists = self.session()
                with self.assertRaises(ControlFault):
                    session.send(poses, twists, accelerations={"ur5e": .01, "ur10e": invalid})
                for control in controls.values():
                    control.speedL.assert_not_called()
        session, controls, poses, twists = self.session()
        with self.assertRaises(ControlFault):
            session.send(poses, twists, accelerations={"ur5e": .01})
        for control in controls.values():
            control.speedL.assert_not_called()

    def test_freshness_and_native_safety_still_precede_all_sends(self):
        session, controls, poses, twists = self.session()
        order = []
        for name, control in controls.items():
            control.isPoseWithinSafetyLimits.side_effect = lambda _pose, name=name: order.append(name + " safety") or True
            control.speedL.side_effect = lambda *_args, name=name: order.append(name + " speed") or True
        session.send(poses, twists, check_fresh=lambda: order.append("fresh"), safety_targets=poses,
                     accelerations={"ur5e": .01, "ur10e": .02})
        self.assertEqual(order, ["ur5e safety", "ur5e safety", "ur10e safety", "ur10e safety",
                                 "fresh", "ur5e speed", "fresh", "ur10e speed"])


if __name__ == "__main__":
    unittest.main()
