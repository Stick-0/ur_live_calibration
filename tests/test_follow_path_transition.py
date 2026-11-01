"""Optional path velocity must not accumulate in a bounded tracking ramp."""

import unittest
import numpy as np

from urcal.cooperative.follow import govern_follow
from urcal.cooperative.task import MotionLimits


class FollowPathTransitionTests(unittest.TestCase):
    def decide(self, previous, tracking, task, *, rho_cap=1., callback=None):
        z = np.zeros(6)
        return govern_follow(z, tracking, task, np.eye(6), previous, previous, previous, previous,
                             .04, MotionLimits(linear_speed=.1, linear_accel=.1,
                                               joint_speed=np.full(6, .2), joint_accel=np.full(6, .75)),
                             previous_base=z, previous_base_qdot=z, rho_cap=rho_cap,
                             extra_feasible=callback)

    def test_steady_path_larger_than_one_tick_acceleration_does_not_accumulate(self):
        task = np.array([-.005, 0., 0., 0., 0., 0.])
        previous = task.copy()
        for _ in range(30):
            decision = self.decide(previous, np.zeros(6), task)
            np.testing.assert_allclose(decision.twist, task, atol=1e-12)
            self.assertEqual(decision.task_scale, 1.)
            previous = decision.twist

    def test_opposing_feedback_brakes_instead_of_adding_task_again(self):
        previous = np.array([-.010, 0., 0., 0., 0., 0.])
        tracking = np.array([.030, 0., 0., 0., 0., 0.])
        task = np.array([-.005, 0., 0., 0., 0., 0.])
        rates = []
        for _ in range(20):
            decision = self.decide(previous, tracking, task)
            self.assertGreaterEqual(decision.twist[0], previous[0] - 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist - previous), .004 + 1e-12)
            rates.append(decision.task_scale)
            previous = decision.twist
        self.assertIn(0., rates)
        self.assertAlmostEqual(previous[0], .025)
        self.assertEqual(rates[-1], 1.)

    def test_pause_brakes_residual_task_and_keeps_progress_frozen(self):
        previous = np.array([-.005, 0., 0., 0., 0., 0.])
        task = previous.copy()
        for _ in range(5):
            decision = self.decide(previous, np.zeros(6), task, rho_cap=0.)
            self.assertEqual(decision.task_scale, 0.)
            self.assertGreaterEqual(decision.twist[0], previous[0] - 1e-12)
            self.assertLessEqual(np.linalg.norm(decision.twist - previous), .004 + 1e-12)
            previous = decision.twist
        np.testing.assert_allclose(previous, 0., atol=1e-12)

    def test_combined_endpoint_cannot_bypass_geometry_rejection(self):
        task = np.array([-.005, 0., 0., 0., 0., 0.])
        checked = []
        def reject_progress(rho, twist, qdot):
            checked.append(rho)
            return rho == 0.
        decision = self.decide(task.copy(), np.zeros(6), task, callback=reject_progress)
        self.assertEqual(decision.task_scale, 0.)
        self.assertTrue(any(rho > 0. for rho in checked))
        self.assertLessEqual(np.linalg.norm(decision.twist-task), .004+1e-12)


if __name__ == "__main__":
    unittest.main()
