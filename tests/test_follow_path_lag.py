"""Optional approach remains bounded when measured motion lags a reversal."""

import unittest

import numpy as np

from urcal.cooperative.follow import govern_follow
from urcal.cooperative.task import MotionLimits


class FollowPathLagTests(unittest.TestCase):
    def test_slow_measured_response_pauses_path_then_resumes_after_leader_reversal(self):
        dt = .04
        limits = MotionLimits(linear_speed=.1, linear_accel=.1,
                              joint_speed=np.full(6, .2), joint_accel=np.full(6, .75))
        zero = np.zeros(6)
        task = np.array([-.005, 0., 0., 0., 0., 0.])
        previous, measured = zero.copy(), zero.copy()
        for leader_speed in (.03, -.03):
            rates = []
            for tick in range(100):
                tracking = np.array([leader_speed, 0., 0., 0., 0., 0.])
                decision = govern_follow(
                    zero, tracking, task, np.eye(6), measured, measured,
                    previous, previous, dt, limits,
                    previous_base=zero, previous_base_qdot=zero)
                for origin in (previous, measured):
                    self.assertLessEqual(np.linalg.norm(decision.twist-origin), limits.linear_accel*dt+1e-12)
                self.assertLessEqual(np.linalg.norm(decision.twist), limits.linear_speed+1e-12)
                rates.append(decision.task_scale)
                # Independent actuator lag: the physical velocity only closes
                # 60% of its difference to the new command each control tick.
                measured += .6*(decision.twist-measured)
                previous = decision.twist.copy()
            self.assertIn(0., rates)
            self.assertEqual(rates[-1], 1.)
            self.assertAlmostEqual(previous[0], leader_speed+task[0], places=12)
            self.assertAlmostEqual(measured[0], leader_speed+task[0], places=12)


if __name__ == "__main__":
    unittest.main()
