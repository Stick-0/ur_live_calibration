"""Constant relative paths avoid duplicate geometry without dropping checks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.cooperative.follow import govern_follow
from urcal.cooperative.task import MotionInfeasible, MotionLimits


def velocity(x):
    return np.array([x, 0., 0., 0., 0., 0.])


def decide(callback, *, rate_dependent=True, previous=None, task=None, tracking=None):
    previous = np.zeros(6) if previous is None else previous
    return govern_follow(
        np.zeros(6), velocity(.015) if tracking is None else tracking,
        np.zeros(6) if task is None else task, np.eye(6), previous, previous,
        previous, previous, .04, MotionLimits(), previous_base=np.zeros(6),
        previous_base_qdot=np.zeros(6), extra_feasible=callback,
        extra_feasible_depends_on_rho=rate_dependent)


class FollowFeasibilityCacheTests(unittest.TestCase):
    def test_constant_path_rejects_all_candidates_with_nine_times_fewer_geometry_calls(self):
        calls = {}
        for rate_dependent in (True, False):
            seen = []

            def reject(rho, twist, qdot):
                seen.append((rho, twist.tobytes(), qdot.tobytes()))
                return False

            with self.assertRaises(MotionInfeasible):
                decide(reject, rate_dependent=rate_dependent)
            calls[rate_dependent] = seen
        self.assertEqual(len(calls[True]), 81)
        self.assertEqual(len(calls[False]), 9)
        # The distinct motion candidates are identical: only redundant task
        # rates for an unchanging relative path disappear from the checks.
        self.assertEqual({item[1:] for item in calls[True]},
                         {item[1:] for item in calls[False]})

    def test_interior_feasible_motion_and_selected_callback_rate_are_unchanged(self):
        decisions, calls = {}, {}
        for rate_dependent in (True, False):
            seen, targets = [], {}

            def feasible(rho, twist, qdot):
                seen.append((rho, twist.copy(), qdot.copy()))
                if twist[0] > .002 + 1e-12:
                    return False
                targets[rho] = twist.copy()
                return True

            result = decide(feasible, rate_dependent=rate_dependent)
            decisions[rate_dependent], calls[rate_dependent] = result, seen
            np.testing.assert_array_equal(targets[result.task_scale], result.twist)
            np.testing.assert_array_equal(seen[-1][1], result.twist)
            self.assertEqual(seen[-1][0], result.task_scale)
        np.testing.assert_array_equal(decisions[True].twist, decisions[False].twist)
        np.testing.assert_array_equal(decisions[True].qdot, decisions[False].qdot)
        self.assertEqual(decisions[True].task_scale, decisions[False].task_scale)
        self.assertEqual(decisions[True].reason, decisions[False].reason)
        self.assertLess(len(calls[False]), len(calls[True]))

    def test_default_preserves_rate_sensitive_callback_with_zero_task_twist(self):
        seen = []

        def feasible(rho, twist, qdot):
            seen.append(rho)
            return rho < .5

        result = decide(feasible)
        self.assertEqual(result.task_scale, .375)
        self.assertEqual(seen, [1., .875, .75, .625, .5, .375])

    def test_checked_braking_preserves_command_and_pauses_relative_task(self):
        previous = np.array([.01, .01, 0., 0., 0., 0.])
        outputs = []
        for rate_dependent in (True, False):
            result = decide(lambda rho, twist, qdot: twist[0] <= .009,
                            previous=previous, tracking=velocity(.02), task=velocity(.005),
                            rate_dependent=rate_dependent)
            self.assertEqual(result.reason, "tracking_braking")
            self.assertEqual(result.task_scale, 0.)
            self.assertLess(np.linalg.norm(result.twist), np.linalg.norm(previous))
            self.assertLessEqual(np.linalg.norm(result.twist - previous), .004 + 1e-12)
            outputs.append(result)
        np.testing.assert_array_equal(outputs[0].twist, outputs[1].twist)
        np.testing.assert_array_equal(outputs[0].qdot, outputs[1].qdot)

    def test_new_cycle_rechecks_changed_geometry(self):
        allowed = False
        with self.assertRaises(MotionInfeasible):
            decide(lambda *args: allowed, rate_dependent=False)
        allowed = True
        result = decide(lambda *args: allowed, rate_dependent=False)
        self.assertGreater(result.twist[0], 0.)


if __name__ == "__main__":
    unittest.main()
