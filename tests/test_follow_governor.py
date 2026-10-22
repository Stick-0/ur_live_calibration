"""Physical bounds on leader starts, stops, reversals, and servo lag."""

import unittest

import numpy as np

from urcal.cooperative.follow import govern_follow
from urcal.cooperative.task import MotionInfeasible, MotionLimits


def velocity(x=0., angular=0.):
    return np.array([x, 0., 0., 0., 0., angular])


class FollowGovernorTests(unittest.TestCase):
    def decide(self, tracking, *, base=None, task=None, previous=None, measured=None,
               previous_base=None, J=None, previous_qdot=None, measured_qdot=None,
               limits=None, **kwargs):
        base = np.zeros(6) if base is None else base
        task = np.zeros(6) if task is None else task
        previous = np.zeros(6) if previous is None else previous
        measured = previous if measured is None else measured
        previous_base = np.zeros(6) if previous_base is None else previous_base
        J = np.eye(6) if J is None else J
        return govern_follow(
            base, tracking, task, J, measured,
            np.linalg.solve(J, measured) if measured_qdot is None else measured_qdot,
            previous, np.linalg.solve(J, previous) if previous_qdot is None else previous_qdot,
            .04, limits or MotionLimits(), previous_base=previous_base,
            previous_base_qdot=np.linalg.solve(J, previous_base), **kwargs)

    def test_start_ramps_instead_of_rejecting_valid_speed_demand(self):
        decision = self.decide(velocity(.015))
        self.assertAlmostEqual(decision.twist[0], .004)
        self.assertEqual(decision.reason, "tracking_limited")
        self.assertAlmostEqual(decision.components["tracking_scale"], .004 / .015)
        self.assertEqual(decision.task_scale, 1.)

    def test_stop_and_reverse_decelerate_from_previous_command(self):
        for requested in (0., -.015):
            decision = self.decide(velocity(requested), previous=velocity(.015))
            self.assertAlmostEqual(decision.twist[0], .011)
            self.assertTrue(decision.components["tracking_limited"])
        previous = velocity(.015)
        outputs = []
        for _ in range(9):
            decision = self.decide(velocity(-.015), previous=previous)
            outputs.append(decision.twist[0])
            self.assertLessEqual(np.linalg.norm(decision.twist[:3] - previous[:3]), .004 + 1e-12)
            previous = decision.twist
        self.assertLess(outputs[4], 0.)
        self.assertAlmostEqual(outputs[-1], -.015)

    def test_measured_lag_and_command_history_both_bound_acceleration(self):
        decision = self.decide(velocity(.02), previous=velocity(.015), measured=velocity(.012))
        self.assertAlmostEqual(decision.twist[0], .016)
        self.assertLessEqual(abs(decision.twist[0] - .015), .004 + 1e-12)
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(velocity(.02), previous=velocity(.015), measured=velocity(0.))
        self.assertEqual(caught.exception.category, "transition_infeasible")

    def test_measured_anchor_can_recover_a_noncollinear_feasible_transition(self):
        previous = velocity(.01)
        measured = previous.copy()
        measured[1] = .006
        requested = previous.copy()
        requested[1] = -.01
        decision = self.decide(requested, previous=previous, measured=measured)
        self.assertEqual(decision.components["tracking_origin"], "measured_motion")
        self.assertAlmostEqual(decision.twist[1], .002)
        self.assertLessEqual(np.linalg.norm(decision.twist[:3] - previous[:3]), .004 + 1e-12)
        self.assertLessEqual(np.linalg.norm(decision.twist[:3] - measured[:3]), .004 + 1e-12)

    def test_over_speed_request_can_turn_without_sticking_at_old_speed_boundary(self):
        previous = velocity(.02)
        requested = np.array([.04, .02, 0., 0., 0., 0.])
        decision = self.decide(requested, previous=previous)
        self.assertGreater(decision.twist[1], .001)
        self.assertLess(decision.twist[0], .02)
        self.assertLessEqual(np.linalg.norm(decision.twist[:3]), .02 + 1e-12)
        self.assertLessEqual(np.linalg.norm(decision.twist[:3] - previous[:3]), .004 + 1e-12)
        self.assertLess(decision.components["tracking_speed_scale"], 1.)

    def test_all_tcp_joint_limits_hold_with_nontrivial_jacobian(self):
        J = np.array([[.1, .02, 0, 0, 0, 0], [0, .08, .01, 0, 0, 0],
                      [0, 0, .12, 0, 0, 0], [0, 0, 0, .7, .1, 0],
                      [0, 0, 0, 0, .8, .1], [0, 0, 0, 0, 0, .6]])
        previous = np.zeros(6)
        limits = MotionLimits()
        for requested in ([.01, .012, .008, .09, .08, .07],
                          [-.03, -.02, 0, -.1, .02, -.09], [0.] * 6):
            for _ in range(12):
                decision = self.decide(np.array(requested), previous=previous, J=J)
                np.testing.assert_allclose(J @ decision.qdot, decision.twist, atol=1e-12)
                self.assertLessEqual(np.linalg.norm(decision.twist[:3]), limits.linear_speed + 1e-12)
                self.assertLessEqual(np.linalg.norm(decision.twist[3:]), limits.angular_speed + 1e-12)
                self.assertTrue(np.all(np.abs(decision.qdot) <= limits.joint_speed + 1e-12))
                self.assertLessEqual(np.linalg.norm(decision.twist[:3] - previous[:3]), limits.linear_accel * .04 + 1e-12)
                self.assertLessEqual(np.linalg.norm(decision.twist[3:] - previous[3:]), limits.angular_accel * .04 + 1e-12)
                self.assertTrue(np.all(np.abs(decision.qdot - np.linalg.solve(J, previous)) <= limits.joint_accel * .04 + 1e-12))
                previous = decision.twist

    def test_joint_acceleration_is_respected_when_cartesian_speed_is_small(self):
        J = np.diag([.05, 1, 1, 1, 1, 1])
        decision = self.decide(velocity(.003), J=J)
        self.assertAlmostEqual(decision.twist[0], .001)
        self.assertAlmostEqual(decision.qdot[0], .02)

    def test_base_is_retained_and_infeasible_base_cannot_be_cancelled(self):
        decision = self.decide(velocity(.03), base=velocity(.002))
        np.testing.assert_array_equal(decision.components["compensation"], velocity(.002))
        np.testing.assert_allclose(decision.twist, velocity(.002) + decision.components["tracking_applied"])
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(velocity(-.01), base=velocity(.01))
        self.assertEqual(caught.exception.category, "compensation_infeasible")
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(velocity(-.03), base=velocity(.03), previous_base=velocity(.03), previous=velocity(.03))
        self.assertEqual(caught.exception.category, "compensation_infeasible")

    def test_task_progress_uses_only_capacity_remaining_after_tracking(self):
        decision = self.decide(velocity(.015), task=velocity(.02), previous=velocity(.015))
        self.assertAlmostEqual(decision.components["tracking_applied"][0], .015)
        self.assertAlmostEqual(decision.task_scale, .2)
        self.assertAlmostEqual(decision.twist[0], .019)
        paused = self.decide(velocity(.015), task=velocity(.02), previous=velocity(.015), rho_cap=0.)
        self.assertEqual(paused.task_scale, 0.)
        self.assertAlmostEqual(paused.twist[0], .015)

    def test_future_callback_can_select_interior_tracking_candidate(self):
        inspected = []

        def feasible(rho, twist, qdot):
            inspected.append(twist.copy())
            return twist[0] <= .002 + 1e-12

        decision = self.decide(velocity(.015), extra_feasible=feasible)
        self.assertTrue(any(twist[0] > .002 for twist in inspected))
        self.assertGreater(decision.twist[0], 0.)
        self.assertLessEqual(decision.twist[0], .002 + 1e-12)
        self.assertTrue(decision.components["tracking_limited"])
        with self.assertRaises(MotionInfeasible):
            self.decide(velocity(.015), extra_feasible=lambda *args: False)

    def test_callback_gets_actual_final_command_and_cannot_mutate_it(self):
        seen = []

        def feasible(rho, twist, qdot):
            seen.append((rho, twist.copy(), qdot.copy()))
            twist[:] = 999
            qdot[:] = 999
            return True

        decision = self.decide(velocity(.002), task=velocity(.01), extra_feasible=feasible)
        np.testing.assert_allclose(seen[-1][1], decision.twist)
        np.testing.assert_allclose(seen[-1][2], decision.qdot)
        self.assertLess(decision.twist.max(), .01)

    def test_duplicate_rejected_candidates_do_not_repeat_expensive_geometry_checks(self):
        seen = []

        def reject(rho, twist, qdot):
            seen.append((float(rho), twist.tobytes(), qdot.tobytes()))
            return False

        with self.assertRaises(MotionInfeasible):
            self.decide(velocity(.015), extra_feasible=reject, rho_cap=0.)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLessEqual(len(seen), 9)

    def test_invalid_inputs_and_singular_jacobian_are_rejected(self):
        with self.assertRaises(ValueError):
            self.decide(np.full(6, np.nan))
        with self.assertRaises(MotionInfeasible) as caught:
            self.decide(velocity(.01), J=np.diag([.001, 1, 1, 1, 1, 1]))
        self.assertEqual(caught.exception.category, "singularity")


if __name__ == "__main__":
    unittest.main()
