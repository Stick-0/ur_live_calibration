"""Pure cooperative path derivatives, compensation, and pre-command limits."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.cooperative.task import MotionInfeasible, MotionLimits, RelativePath, govern_task, target_components
from urcal.control.geometry import facing_follower, inverse, pose_matrix, rotation_distance


def relaxed_limits(**changes):
    limits = MotionLimits(linear_speed=1., angular_speed=1., linear_accel=10., angular_accel=10.,
                          joint_speed=np.full(6, 10.), joint_accel=np.full(6, 100.), min_singular_value=0.001)
    return replace(limits, **changes)


def advance(transform, twist, dt):
    """Independent geometric differential: origin velocity + parent-frame omega."""
    result = transform.copy()
    result[:3, 3] += twist[:3] * dt
    result[:3, :3] = Rotation.from_rotvec(twist[3:] * dt).as_matrix() @ transform[:3, :3]
    return result


def central_twist(before, after, half_interval):
    linear = (after[:3, 3] - before[:3, 3]) / (2. * half_interval)
    angular = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec() / (2. * half_interval)
    return np.r_[linear, angular]


class RelativePathTests(unittest.TestCase):
    def test_start_is_copied_and_rendering_never_recaptures_or_advances_it(self):
        initial = facing_follower(np.eye(4), 0.3)
        expected = initial.copy()
        path = RelativePath(initial, "line", delta=[0.05, 0, 0, 0, 0, 0.1])
        initial[:3, 3] = 99
        returned = path.start
        returned[:3, 3] = -99
        middle = path.pose(0.5)
        middle[:3, 3] = 999
        for u in [1., 0.8, 0.1, 0., 0.5, 0.]:
            path.pose(u)
        np.testing.assert_array_equal(path.pose(0.), expected)
        np.testing.assert_array_equal(path.start, expected)

    def test_hold_has_fixed_pose_and_zero_derivative(self):
        start = pose_matrix([0.01, -0.02, 0.3, 0.2, 0.3, -0.1])
        path = RelativePath(start)
        for u in np.linspace(0, 1, 11):
            np.testing.assert_array_equal(path.pose(u), start)
            np.testing.assert_array_equal(path.derivative(u), np.zeros(6))

    def test_line_uses_leader_axes_and_left_rotation_with_zero_endpoint_speed(self):
        start = pose_matrix([0.02, 0.01, 0.3, 0.3, -0.2, 0.1])
        delta = np.array([0.04, -0.03, 0.02, 0.1, 0.03, -0.08])
        path = RelativePath(start, "line", delta=delta, duration=4.)
        goal = path.pose(1.)
        np.testing.assert_allclose(goal[:3, 3], start[:3, 3] + delta[:3])
        np.testing.assert_allclose(goal[:3, :3], Rotation.from_rotvec(delta[3:]).as_matrix() @ start[:3, :3])
        for u in (0., 1.):
            np.testing.assert_array_equal(path.derivative(u), np.zeros(6))
        np.testing.assert_allclose(path.derivative(0.5), delta * 1.875 / 4.)

    def test_circle_makes_one_closed_xy_lap_without_changing_orientation(self):
        start = pose_matrix([0.01, 0.02, 0.3, np.pi, 0, 0])
        path = RelativePath(start, "circle", radius=0.02, duration=5.)
        np.testing.assert_array_equal(path.pose(0.), start)
        np.testing.assert_array_equal(path.pose(1.), start)
        np.testing.assert_allclose(path.pose(0.5)[:3, 3], start[:3, 3] + [-0.04, 0, 0], atol=1e-12)
        for u in np.linspace(0, 1, 31):
            np.testing.assert_array_equal(path.pose(u)[:3, :3], start[:3, :3])
            center_delta = path.pose(u)[:3, 3] - start[:3, 3] + [0.02, 0, 0]
            self.assertAlmostEqual(np.linalg.norm(center_delta[:2]), 0.02)
        for u in (0., 1.):
            np.testing.assert_array_equal(path.derivative(u), np.zeros(6))

    def test_path_derivatives_match_finite_motion_per_second(self):
        start = pose_matrix([0.02, 0.01, 0.3, 0.5, -0.2, 2.9])
        for kind in ("line", "circle"):
            path = RelativePath(start, kind, delta=[0.03, -0.01, 0.02, 0.1, -0.2, 0.3], duration=7.)
            for u in (0.1, 0.3, 0.5, 0.8, 0.95):
                du = 1e-6
                numeric = central_twist(path.pose(u - du), path.pose(u + du), du * path.duration)
                np.testing.assert_allclose(path.derivative(u), numeric, atol=1e-9)

    def test_rotation_wrap_does_not_create_a_full_turn(self):
        start = pose_matrix([0, 0, 0.3, 0, 0, np.pi - 0.02])
        crossing = RelativePath(start, "line", delta=[0, 0, 0, 0, 0, 0.04])
        self.assertAlmostEqual(rotation_distance(start, crossing.pose(1)), 0.04)
        wrapped = RelativePath(start, "line", delta=[0, 0, 0, 0, 0, 2 * np.pi - 0.1])
        self.assertAlmostEqual(wrapped.delta[5], -0.1)
        self.assertAlmostEqual(rotation_distance(start, wrapped.pose(0.5)), 0.05)
        self.assertLess(wrapped.derivative(0.5)[5], 0.)

    def test_invalid_path_parameters_and_progress_are_rejected(self):
        for kwargs in ({"kind": "unknown"}, {"duration": 0}, {"radius": float("nan")},
                       {"delta": [0, 0, 0]}, {"duration": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RelativePath(np.eye(4), **kwargs)
        path = RelativePath(np.eye(4))
        for u in (-0.001, 1.001, float("nan")):
            with self.subTest(u=u), self.assertRaises(ValueError):
                path.pose(u)


class TargetComponentsTests(unittest.TestCase):
    def test_analytic_components_match_separate_and_combined_finite_differences(self):
        rng = np.random.default_rng(913)
        for _ in range(15):
            A, B, L, S, O = [pose_matrix(np.r_[rng.normal(0, .25, 3), rng.normal(0, .6, 3)]) for _ in range(5)]
            ta, tb, tl, ts = [rng.normal(0, .03, 6) for _ in range(4)]
            target, base, leader, task = target_components({"a": A, "b": B}, {"a": ta, "b": tb},
                                                         "a", "b", L, tl, S, ts, O)
            np.testing.assert_allclose(target, inverse(B) @ A @ L @ S @ O, atol=1e-12)
            dt = 1e-5
            for expected, include_base, include_leader, include_task in (
                    (base, True, False, False), (leader, False, True, False),
                    (task, False, False, True), (base + leader + task, True, True, True)):
                poses = []
                for sign in (-1., 1.):
                    At = advance(A, ta, sign * dt) if include_base else A
                    Bt = advance(B, tb, sign * dt) if include_base else B
                    Lt = advance(L, tl, sign * dt) if include_leader else L
                    St = advance(S, ts, sign * dt) if include_task else S
                    poses.append(inverse(Bt) @ At @ Lt @ St @ O)
                np.testing.assert_allclose(expected, central_twist(*poses, dt), atol=1e-9)

    def test_common_rigid_base_motion_cancels_from_relative_target(self):
        A = pose_matrix([-0.4, .1, .8, .1, -.2, .2])
        B = pose_matrix([.5, -.1, 1., -.1, .3, -.2])
        common_linear, omega = np.array([.02, -.01, .005]), np.array([.03, .04, -.02])
        twists = {name: np.r_[common_linear + np.cross(omega, T[:3, 3]), omega]
                  for name, T in {"a": A, "b": B}.items()}
        _target, base, _leader, _task = target_components(
            {"a": A, "b": B}, twists, "a", "b", pose_matrix([.1, .2, .3, .2, .1, 0]),
            np.zeros(6), facing_follower(np.eye(4), .3), np.zeros(6), pose_matrix([.01, .02, .08, 0, .2, 0]),
        )
        np.testing.assert_allclose(base, 0, atol=1e-12)

    def test_leader_and_task_rotation_use_different_tcp_lever_arms(self):
        relative = facing_follower(np.eye(4), .3)
        tcp_offset = pose_matrix([.05, 0, .1, .1, 0, 0])
        target, base, leader, task = target_components(
            {"a": np.eye(4), "b": np.eye(4)}, {"a": np.zeros(6), "b": np.zeros(6)},
            "a", "b", np.eye(4), [0, 0, 0, 0, .1, 0], relative, [0, 0, 0, 0, .1, 0], tcp_offset,
        )
        np.testing.assert_allclose(target, relative @ tcp_offset)
        np.testing.assert_allclose(base, 0)
        np.testing.assert_allclose(leader, [.02, 0, -.005, 0, .1, 0], atol=1e-12)
        np.testing.assert_allclose(task, [-.01, 0, -.005, 0, .1, 0], atol=1e-12)

    def test_follower_base_translation_requires_opposite_compensation(self):
        _target, base, leader, task = target_components(
            {"a": np.eye(4), "b": pose_matrix([1, 0, 0, 0, 0, 0])},
            {"a": np.zeros(6), "b": [.01, -.02, 0, 0, 0, 0]}, "a", "b", np.eye(4), np.zeros(6),
            facing_follower(np.eye(4), .3), np.zeros(6), np.eye(4),
        )
        np.testing.assert_allclose(base, [-.01, .02, 0, 0, 0, 0])
        np.testing.assert_array_equal(leader, np.zeros(6))
        np.testing.assert_array_equal(task, np.zeros(6))


class TaskGovernorTests(unittest.TestCase):
    def decide(self, compensation=None, feedback=None, task=None, J=None, previous_twist=None,
               previous_qdot=None, dt=.1, limits=None, **kwargs):
        return govern_task(np.zeros(6) if compensation is None else compensation,
                           np.zeros(6) if feedback is None else feedback,
                           np.zeros(6) if task is None else task,
                           np.eye(6) if J is None else J,
                           np.zeros(6) if previous_twist is None else previous_twist,
                           np.zeros(6) if previous_qdot is None else previous_qdot,
                           dt, relaxed_limits() if limits is None else limits, **kwargs)

    def test_zero_task_rate_keeps_compensation_and_feedback(self):
        compensation = np.array([.015, 0, 0, .04, 0, 0])
        feedback = np.array([0, .002, 0, 0, .002, 0])
        decision = self.decide(compensation, feedback, np.full(6, .1), rho_cap=0.)
        np.testing.assert_allclose(decision.twist, compensation + feedback)
        np.testing.assert_allclose(decision.qdot, compensation + feedback)
        np.testing.assert_allclose(decision.components["compensation"], compensation)
        np.testing.assert_array_equal(decision.components["task_applied"], np.zeros(6))
        self.assertEqual(decision.reason, "task_paused")

    def test_infeasible_compensation_is_not_hidden_by_cancelling_feedback(self):
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(compensation=[.03, 0, 0, 0, 0, 0], feedback=[-.02, 0, 0, 0, 0, 0],
                        limits=relaxed_limits(linear_speed=.02))
        self.assertEqual(caught.exception.category, "compensation_infeasible")

    def test_task_speed_saturation_scales_only_task_vector(self):
        compensation, feedback, task = (np.array(v) for v in (
            [.01, 0, 0, 0, 0, 0], [0, .005, 0, 0, 0, 0], [.04, 0, 0, 0, 0, 0]))
        decision = self.decide(compensation, feedback, task, limits=relaxed_limits(linear_speed=.02))
        expected = (np.sqrt(.02**2 - .005**2) - .01) / .04
        self.assertAlmostEqual(decision.task_scale, expected)
        np.testing.assert_allclose(decision.twist, compensation + feedback + expected * task)
        self.assertAlmostEqual(np.linalg.norm(decision.twist[:3]), .02)
        self.assertEqual(decision.reason, "task_slowed")

    def test_joint_speed_limit_reduces_task_without_cartesian_clipping(self):
        J = np.diag([.1, 1, 1, 1, 1, 1])
        decision = self.decide(task=[.04, 0, 0, 0, 0, 0], J=J,
                               limits=relaxed_limits(joint_speed=np.full(6, .2)))
        self.assertAlmostEqual(decision.task_scale, .5)
        self.assertAlmostEqual(decision.qdot[0], .2)
        np.testing.assert_allclose(J @ decision.qdot, decision.twist)

    def test_angular_and_joint_acceleration_constraints_limit_task(self):
        angular = self.decide(task=[0, 0, 0, 0, .4, 0], dt=.1,
                              limits=relaxed_limits(angular_accel=.5))
        self.assertAlmostEqual(angular.task_scale, .125)
        joint = self.decide(task=[.04, 0, 0, 0, 0, 0], J=np.diag([.1, 1, 1, 1, 1, 1]), dt=.1,
                            limits=relaxed_limits(joint_accel=np.full(6, .1)))
        self.assertAlmostEqual(joint.task_scale, .025)

    def test_compensation_acceleration_uses_its_own_history(self):
        comp = np.array([.01, 0, 0, 0, 0, 0])
        previous_total = np.array([.019, 0, 0, 0, 0, 0])
        limits = relaxed_limits(linear_speed=.02, linear_accel=.1)
        decision = self.decide(compensation=comp, task=[.009, 0, 0, 0, 0, 0],
                               previous_twist=previous_total, previous_qdot=previous_total, dt=.04,
                               previous_compensation=comp, previous_compensation_qdot=comp, limits=limits)
        self.assertAlmostEqual(decision.task_scale, 1.)
        np.testing.assert_allclose(decision.twist, previous_total)
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(compensation=comp, dt=.04, limits=limits)
        self.assertEqual(caught.exception.category, "compensation_infeasible")

    def test_established_task_can_have_positive_feasible_lower_rate(self):
        task = np.array([.01, 0, 0, 0, 0, 0])
        decision = self.decide(task=task, previous_twist=task, previous_qdot=task, dt=.04,
                               limits=relaxed_limits(linear_accel=.1))
        self.assertAlmostEqual(decision.task_scale, 1.)
        np.testing.assert_allclose(decision.components["feasible_task_interval"], [.6, 1.])

    def test_strict_immediate_pause_rejects_impossible_velocity_transition(self):
        task = np.array([.01, 0, 0, 0, 0, 0])
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(task=task, previous_twist=task, previous_qdot=task, dt=.04, rho_cap=0,
                        limits=relaxed_limits(linear_accel=.1))
        self.assertEqual(caught.exception.category, "transition_infeasible")

    def test_soft_pause_ramps_task_rate_to_zero_while_preserving_compensation(self):
        comp = np.array([.005, 0, 0, 0, 0, 0])
        task = np.array([.01, 0, 0, 0, 0, 0])
        previous = comp + task
        rates = []
        for _ in range(3):
            decision = self.decide(compensation=comp, task=task, previous_twist=previous,
                                   previous_qdot=previous, previous_compensation=comp,
                                   previous_compensation_qdot=comp, dt=.04, rho_cap=0,
                                   allow_deceleration=True, limits=relaxed_limits(linear_accel=.1))
            rates.append(decision.task_scale)
            self.assertLessEqual(np.linalg.norm(decision.twist - previous), .004 + 1e-12)
            np.testing.assert_allclose(decision.components["compensation"], comp)
            if decision.task_scale > 0:
                self.assertEqual(decision.reason, "task_decelerating")
            previous = decision.twist
        np.testing.assert_allclose(rates, [.6, .2, 0], atol=1e-12)
        np.testing.assert_allclose(previous, comp)
        self.assertEqual(decision.reason, "task_paused")

    def test_soft_pause_cannot_override_static_speed_infeasibility(self):
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(feedback=[.03, 0, 0, 0, 0, 0], task=[-.02, 0, 0, 0, 0, 0],
                        rho_cap=0., allow_deceleration=True, limits=relaxed_limits(linear_speed=.02))
        self.assertEqual(caught.exception.category, "tracking_infeasible")

    def test_singular_jacobian_rejected_even_for_zero_requested_command(self):
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(J=np.diag([1., 1., 1., 1., 1., 1e-5]))
        self.assertEqual(caught.exception.category, "singularity")

    def test_nonlinear_callback_uses_discrete_feasible_search_not_binary_assumption(self):
        def feasible(rho, _twist, _qdot):
            return rho <= .1 or .7 <= rho <= .75

        decision = self.decide(task=[.01, 0, 0, 0, 0, 0], extra_feasible=feasible, grid_samples=101)
        self.assertAlmostEqual(decision.task_scale, .75)
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(extra_feasible=lambda *_args: False)
        self.assertEqual(caught.exception.category, "tracking_infeasible")

    def test_callback_cannot_mutate_commands_and_deceleration_search_is_ascending(self):
        task = np.array([.01, 0, 0, 0, 0, 0])

        def feasible(rho, twist, qdot):
            twist[:] = 999
            qdot[:] = -999
            return rho >= .7

        decision = self.decide(task=task, previous_twist=task, previous_qdot=task, dt=.04, rho_cap=0.,
                               allow_deceleration=True, limits=relaxed_limits(linear_accel=.1),
                               extra_feasible=feasible, grid_samples=101)
        self.assertAlmostEqual(decision.task_scale, .7)
        np.testing.assert_allclose(decision.twist, [.007, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(decision.qdot, decision.twist)

    def test_convex_interval_solution_agrees_with_independent_dense_rate_scan(self):
        rng = np.random.default_rng(751)
        limits = relaxed_limits(linear_speed=.04, angular_speed=.08, linear_accel=.2,
                                angular_accel=.4, joint_speed=np.full(6, .2), joint_accel=np.full(6, .6))
        dt = .04
        rates = np.linspace(0, 1, 2001)
        for _ in range(60):
            U, _unused = np.linalg.qr(rng.normal(size=(6, 6)))
            V, _unused = np.linalg.qr(rng.normal(size=(6, 6)))
            J = U @ np.diag(rng.uniform(.4, 1.2, 6)) @ V.T
            comp = rng.uniform(-.004, .004, 6)
            feedback = rng.uniform(-.006, .006, 6)
            task = rng.uniform(-.04, .04, 6)
            previous = comp + feedback + rng.uniform(0, 1) * task
            previous_qdot = np.linalg.solve(J, previous)
            samples = comp + feedback + rates[:, None] * task
            qdots = np.linalg.solve(J, samples.T).T
            permitted = (
                (np.linalg.norm(samples[:, :3], axis=1) <= limits.linear_speed)
                & (np.linalg.norm(samples[:, 3:], axis=1) <= limits.angular_speed)
                & (np.linalg.norm(samples[:, :3] - previous[:3], axis=1) <= limits.linear_accel * dt)
                & (np.linalg.norm(samples[:, 3:] - previous[3:], axis=1) <= limits.angular_accel * dt)
                & (np.abs(qdots) <= limits.joint_speed).all(axis=1)
                & (np.abs(qdots - previous_qdot) <= limits.joint_accel * dt).all(axis=1)
            )
            try:
                decision = self.decide(comp, feedback, task, J, previous, previous_qdot,
                                       dt=dt, limits=limits, previous_compensation=comp,
                                       previous_compensation_qdot=np.linalg.solve(J, comp))
            except MotionInfeasible:
                self.assertFalse(permitted.any(), "governor rejected an independently feasible command")
                continue
            if permitted.any():
                self.assertGreaterEqual(decision.task_scale + 1e-12, rates[permitted][-1])
                self.assertLessEqual(decision.task_scale - rates[permitted][-1], .0005 + 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist[:3]), limits.linear_speed + 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist[3:]), limits.angular_speed + 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist[:3] - previous[:3]), limits.linear_accel * dt + 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist[3:] - previous[3:]), limits.angular_accel * dt + 1e-12)
            self.assertTrue((np.abs(decision.qdot) <= limits.joint_speed + 1e-12).all())
            self.assertTrue((np.abs(decision.qdot - previous_qdot) <= limits.joint_accel * dt + 1e-12).all())

    def test_input_limits_and_histories_are_validated_without_mutation(self):
        bad_limits = ({"linear_speed": 0}, {"joint_accel": np.zeros(6)},
                      {"joint_speed": np.full(6, float("nan"))}, {"joint_speed": [1, 2]})
        for kwargs in bad_limits:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionLimits(**kwargs)
        for kwargs in ({"dt": 0}, {"rho_cap": 1.1}, {"task": [0, 0]},
                       {"previous_compensation": np.full(6, float("inf"))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.decide(**kwargs)
        original = np.full(6, .2)
        limits = MotionLimits(joint_speed=original)
        original[:] = 999
        np.testing.assert_allclose(limits.joint_speed, .2)


if __name__ == "__main__":
    unittest.main()
