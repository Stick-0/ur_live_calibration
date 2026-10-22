"""Completed-alignment watchdog holds using virtual encoder states only."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from test_live_alignment import fixture, make_plan
from urcal.control.alignment import AlignmentFault


class AlignmentHandoffHoldTests(unittest.TestCase):
    def settled(self, mode="follow"):
        values = fixture(mode, already_at_goal=True)
        plan = make_plan(values)
        plant = values[2]
        for _ in range(5):
            plan.propose(plant.states(100.), plant.bases, .04)
            plan.commit()
        self.assertTrue(plan.done)
        return values, plan

    def test_only_zero_commands_for_commanded_arms_without_commit_or_progress_changes(self):
        for mode in ("follow", "hold"):
            with self.subTest(mode=mode):
                values, plan = self.settled(mode)
                plant, elapsed, settled = values[2], plan.elapsed, plan._settled
                before_commands = deepcopy(plan.previous_commands)
                before_twists = deepcopy(plan.previous_twists)
                for tick in range(6):
                    states = plant.states(101. + tick * .04)
                    before = deepcopy(states)
                    commands = plan.propose_hold(states, plant.bases, .04)
                    self.assertEqual(set(commands), set(plan.commanded))
                    for name, command in commands.items():
                        np.testing.assert_array_equal(command, np.zeros(6))
                        np.testing.assert_array_equal(plan.previous_commands[name], before_commands[name])
                        np.testing.assert_array_equal(plan.previous_twists[name], before_twists[name])
                        self.assertEqual(plan.joint_accelerations[name], 1e-6)
                    for name in states:
                        np.testing.assert_array_equal(plan.predicted_joints[name], states[name].joints)
                        for field in ("joints", "joint_velocity", "tcp", "speed"):
                            np.testing.assert_array_equal(getattr(states[name], field), getattr(before[name], field))
                    self.assertEqual(plan.elapsed, elapsed)
                    self.assertEqual(plan.progress, 1.)
                    self.assertEqual(plan._settled, settled)
                    self.assertTrue(plan.done)
                with self.assertRaisesRegex(AlignmentFault, "successful proposal"):
                    plan.commit()

    def test_incomplete_alignment_cannot_enter_stationary_hold(self):
        values = fixture("follow", already_at_goal=True)
        plan = make_plan(values)
        with self.assertRaisesRegex(AlignmentFault, "must be complete"):
            plan.propose_hold(values[2].states(0.), values[2].bases, .04)

    def test_invalid_and_stale_periods_are_rejected(self):
        values, plan = self.settled()
        for dt in (0., -.01, np.nan, np.inf, values[0].max_robot_age + .001):
            with self.subTest(dt=dt), self.assertRaisesRegex(AlignmentFault, "invalid or stale"):
                plan.propose_hold(values[2].states(101.), values[2].bases, dt)

    def test_fixed_leader_motion_and_changed_captured_bases_are_rejected(self):
        for change in ("leader", "base"):
            with self.subTest(change=change):
                values, plan = self.settled()
                plant = values[2]
                bases = deepcopy(plant.bases)
                if change == "leader":
                    plant.joints["ur10e"][5] += .011
                    reason = "fixed leader moved"
                else:
                    bases["ur5e"][0, 3] += .011
                    reason = "base moved"
                with self.assertRaisesRegex(AlignmentFault, reason):
                    plan.propose_hold(plant.states(101.), bases, .04)

    def test_endpoint_and_corridor_drift_are_rejected_without_corrective_motion(self):
        for change in ("joint_endpoint", "tcp_endpoint", "corridor"):
            with self.subTest(change=change):
                values, plan = self.settled()
                plant = values[2]
                if change == "joint_endpoint":
                    plant.joints["ur5e"][5] += .015
                elif change == "corridor":
                    plant.joints["ur5e"][5] += .04
                states = plant.states(101.)
                if change == "tcp_endpoint":
                    states["ur5e"].tcp[0, 3] += .003
                reason = "corridor" if change == "corridor" else "endpoint"
                with self.assertRaisesRegex(AlignmentFault, reason):
                    plan.propose_hold(states, plant.bases, .04)
                self.assertIsNone(plan._pending)

    def test_encoder_invalidity_motion_and_kinematic_mismatch_are_rejected(self):
        for change, reason in (("nonfinite", "finite"), ("motion", "stationary"),
                               ("speed", "speed exceeded"), ("kinematics", "kinematics")):
            with self.subTest(change=change):
                values, plan = self.settled()
                plant = values[2]
                states = plant.states(101.)
                if change == "nonfinite":
                    states["ur5e"].joint_velocity[0] = np.nan
                elif change == "motion":
                    states["ur5e"].joint_velocity[5] = .006
                elif change == "speed":
                    states["ur5e"].speed[0] = values[0].max_speed + .01
                else:
                    states["ur5e"].tcp[0, 3] += values[0].frame_check_m + .01
                with self.assertRaisesRegex(AlignmentFault, reason):
                    plan.propose_hold(states, plant.bases, .04)

    def test_current_geometry_and_joint_limits_remain_checked(self):
        values, plan = self.settled()
        states = values[2].states(101.)
        with patch.object(plan.guard, "check", return_value=(False, "blocked capsule", 0.)):
            with self.assertRaisesRegex(AlignmentFault, "blocked capsule"):
                plan.propose_hold(states, values[2].bases, .04)
        plan.guard.upper["ur5e"][0] = states["ur5e"].joints[0] - .01
        with self.assertRaisesRegex(AlignmentFault, "joint position limit"):
            plan.propose_hold(states, values[2].bases, .04)

    def test_small_measured_velocity_has_consistent_ramp_including_tiny_noise(self):
        for measured in (.003, 1e-14):
            with self.subTest(measured=measured):
                values, plan = self.settled()
                states = values[2].states(101.)
                states["ur5e"].joint_velocity[5] = measured
                jacobian = plan.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[1]
                states["ur5e"].speed = jacobian @ states["ur5e"].joint_velocity
                command = plan.propose_hold(states, values[2].bases, .04)["ur5e"]
                np.testing.assert_array_equal(command, np.zeros(6))
                ramp = plan.joint_accelerations["ur5e"]
                self.assertGreater(ramp, 0.)
                self.assertAlmostEqual(ramp, measured / .04)
                self.assertLessEqual(ramp, values[0].max_joint_accel)
                direction = jacobian @ (-states["ur5e"].joint_velocity / measured)
                self.assertLessEqual(ramp * np.linalg.norm(direction[:3]), values[0].max_accel)
                self.assertLessEqual(ramp * np.linalg.norm(direction[3:]), values[0].max_angular_accel)

    def test_rest_threshold_does_not_bypass_measured_acceleration_limit(self):
        values, plan = self.settled()
        states = values[2].states(101.)
        states["ur5e"].joint_velocity[5] = .004
        jacobian = plan.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[1]
        states["ur5e"].speed = jacobian @ states["ur5e"].joint_velocity
        self.assertTrue(plan._at_rest(states["ur5e"]))
        with self.assertRaisesRegex(AlignmentFault, "acceleration_from_measured"):
            plan.propose_hold(states, values[2].bases, .001)


if __name__ == "__main__":
    unittest.main()
