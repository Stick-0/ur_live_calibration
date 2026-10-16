"""Integration regressions found while auditing the research implementation.

All robot and geometry interfaces are injected numerical fakes; no hardware is
opened. The shared scene helpers use production planning and a Cartesian plant.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from urcal.control.runtime import MotionSession
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.task import MotionInfeasible
from urcal.control.geometry import inverse, matrix_pose, pose_matrix
from urcal.control.policy import ControlFault
from test_cooperative_controller import ROBOTS, accept, packet, scene
from test_dual_control_runtime import controller as fake_rtde_control


class CalibrationAtomicityRegressionTests(unittest.TestCase):
    def test_velocity_rejection_does_not_poison_jump_gate_or_valid_recovery(self):
        engine, bases, _states = scene()
        accept(engine, bases, at=10., sequence=1)
        accepted_poses = engine.gate.poses
        bad = {name: value.copy() for name, value in bases.items()}
        bad[engine.follower][0, 3] += .014  # <15mm jump, but .7m/s over 20ms
        self.assertFalse(engine.ingest(packet(engine, bad, at=10.02, sequence=2), 10.02))
        self.assertIn("speed", engine.last_rejection)
        self.assertEqual(engine.gate.last_capture, 10.)
        self.assertEqual(engine.gate.stable_count, 0)
        for name in ROBOTS:
            np.testing.assert_array_equal(engine.gate.poses[name], accepted_poses[name])
        self.assertEqual(engine.estimator.estimate(10.02).measurement_times, {name: 10. for name in ROBOTS})
        # Reusing a rejected sequence cannot relabel it as a fresh measurement.
        self.assertFalse(engine.ingest(packet(engine, bases, at=10.02, sequence=2), 10.03))
        self.assertEqual(engine.gate.last_capture, 10.)
        recovery = {name: value.copy() for name, value in bases.items()}
        recovery[engine.follower][0, 3] -= .002  # valid from x0; 16mm from REJECTED sample
        self.assertTrue(engine.ingest(packet(engine, recovery, at=10.1, sequence=3), 10.1))
        self.assertEqual(engine.gate.last_capture, 10.1)
        self.assertIsNone(engine.invalid_since)
        self.assertEqual(engine.state(10.1, active=True).measurement_times, {name: 10.1 for name in ROBOTS})


class MeasuredMotionRegressionTests(unittest.TestCase):
    def test_actual_velocity_transition_is_checked_even_if_previous_command_was_zero(self):
        engine, bases, states = scene(max_accel=.1, max_joint_accel=.2)
        accept(engine, bases)
        states[engine.follower].speed[0] = .05
        states[engine.follower].joint_velocity[0] = .05
        np.testing.assert_array_equal(engine.previous_twist, np.zeros(6))
        with self.assertRaises(MotionInfeasible) as caught:
            engine.prepare(10., states, .04, active=True)
        self.assertEqual(caught.exception.category, "transition_infeasible")
        self.assertEqual(engine.progress, 0.)

    def test_previous_command_slew_is_still_checked_when_measured_motion_is_zero(self):
        engine, bases, states = scene(max_accel=.1, max_joint_accel=.2)
        accept(engine, bases)
        engine.previous_twist[0] = .05
        engine.previous_qdot[0] = .05
        with self.assertRaises((MotionInfeasible, ResearchFault)):
            engine.prepare(10., states, .04, active=True)
        self.assertEqual(engine.progress, 0.)

    def test_measured_tcp_speed_is_guarded_independently_of_joint_speed(self):
        for component, value in ((0, .03), (4, .2)):
            engine, bases, states = scene(max_speed=.02, max_angular_speed=.1)
            accept(engine, bases)
            states[engine.follower].speed[component] = value
            with self.subTest(component=component), self.assertRaises(ResearchFault) as caught:
                engine.prepare(10., states, .04, active=True)
            self.assertEqual(caught.exception.category, "tcp_speed")

    def test_compensation_only_swept_clearance_failure_has_mandatory_motion_category(self):
        engine, bases, states = scene()
        accept(engine, bases)
        states[engine.leader].joint_velocity[0] = .02

        def clearance(_bases, joints):
            if joints[engine.follower][0] > .0001:
                return False, "synthetic obstacle in mandatory compensation direction", -.001
            return True, "clear", 1.

        engine.guard.check = clearance
        with self.assertRaises((MotionInfeasible, ResearchFault)) as caught:
            engine.prepare(10., states, .1, active=True)
        self.assertEqual(caught.exception.category, "compensation_infeasible")
        self.assertEqual(engine.progress, 0.)


class FutureTargetRegressionTests(unittest.TestCase):
    def test_proposal_keeps_current_reference_and_separate_future_safety_target(self):
        engine, bases, states = scene()
        accept(engine, bases)
        states[engine.leader].joint_velocity[0] = .02
        proposal = engine.prepare(10., states, .1, active=True)
        current = proposal.tcp_goals[engine.follower]
        future = proposal.future_tcp_goals[engine.follower]
        self.assertAlmostEqual(future[0, 3] - current[0, 3], .002)
        np.testing.assert_allclose(proposal.record["target_tcp"], current)
        np.testing.assert_allclose(proposal.record["next_target_tcp"], future)
        leader = states[engine.leader].tcp @ inverse(engine.offsets[engine.leader])
        relation = inverse(bases[engine.leader] @ leader) @ bases[engine.follower] @ current @ inverse(engine.offsets[engine.follower])
        np.testing.assert_allclose(relation, engine.path.pose(engine.progress), atol=1e-12)

    def test_future_goal_failure_prevents_either_arm_command_after_current_goals_pass(self):
        engine, _bases, _states = scene()
        controls = [fake_rtde_control(), fake_rtde_control()]
        session = MotionSession(engine.args, ROBOTS, engine.offsets, factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        controls[0].isPoseWithinSafetyLimits.side_effect = [True, True]
        controls[1].isPoseWithinSafetyLimits.side_effect = [True, False]
        current = {name: pose_matrix([0., 0, .7, 0, 0, 0]) for name in ROBOTS}
        future = {name: pose_matrix([.002, 0, .7, 0, 0, 0]) for name in ROBOTS}
        with self.assertRaisesRegex(ControlFault, "next goal"):
            session.send(current, {name: np.zeros(6) for name in ROBOTS}, safety_targets=future)
        for name, control in zip(ROBOTS, controls):
            self.assertEqual(control.isPoseWithinSafetyLimits.call_count, 2)
            np.testing.assert_allclose(control.isPoseWithinSafetyLimits.call_args_list[0].args[0], matrix_pose(current[name]))
            np.testing.assert_allclose(control.isPoseWithinSafetyLimits.call_args_list[1].args[0], matrix_pose(future[name]))
            control.speedL.assert_not_called()


class CompletionRegressionTests(unittest.TestCase):
    def test_path_end_requires_committed_tracking_dwell_and_preview_does_not_accumulate(self):
        engine, bases, states = scene(method="direct", path="line", path_delta=[.02, 0, 0, 0, 0, 0],
                                      finish_position_m=.001, finish_rotation_deg=.2, finish_dwell=.2)
        accept(engine, bases)
        engine.progress = 1.0  # the reference finished, but the real follower has not arrived
        lagging = engine.prepare(10., states, .1, active=True)
        self.assertTrue(lagging.record["path_finished"])
        self.assertFalse(lagging.record["within_finish_tolerance"])
        engine.commit(lagging)
        self.assertFalse(engine.finished)
        self.assertEqual(engine.settle_elapsed, 0.)
        follower = engine.follower
        states[follower].joints[0] = .02
        states[follower].tcp = engine.models[follower].pose_and_jacobian(states[follower].joints)[0]
        for _ in range(5):
            preview = engine.prepare(10.1, states, .1, active=False)
            self.assertTrue(preview.record["within_finish_tolerance"])
        self.assertEqual(engine.settle_elapsed, 0.)
        self.assertFalse(engine.finished)
        engine.commit(engine.prepare(10.1, states, .1, active=True))
        self.assertFalse(engine.finished)
        self.assertAlmostEqual(engine.settle_elapsed, .1)
        # Leaving tolerance resets consecutive dwell instead of accumulating isolated good samples.
        states[follower].joints[0] = .015
        states[follower].tcp = engine.models[follower].pose_and_jacobian(states[follower].joints)[0]
        engine.commit(engine.prepare(10.2, states, .1, active=True))
        self.assertEqual(engine.settle_elapsed, 0.)
        states[follower].joints[0] = .02
        states[follower].tcp = engine.models[follower].pose_and_jacobian(states[follower].joints)[0]
        engine.commit(engine.prepare(10.3, states, .1, active=True))
        self.assertFalse(engine.finished)
        engine.commit(engine.prepare(10.4, states, .1, active=True))
        self.assertTrue(engine.finished)
        self.assertAlmostEqual(engine.settle_elapsed, .2)


if __name__ == "__main__":
    unittest.main()
