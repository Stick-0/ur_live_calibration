"""Recorded alignment failures replayed with real kinematics and no transport.

The fixture records the approximation of the unlogged Arm snapshot and how
the control period was inferred. Tests need no runs/ logs, camera, or UR SDK.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.alignment import AlignmentFault, LiveAlignment


FIXTURE = Path(__file__).parent / "fixtures" / "alignment_governor_cases.json"


def states(values):
    return {name: SimpleNamespace(joints=np.array(state["q"]), tcp=np.array(state["tcp"]),
                                  joint_velocity=np.array(state["qd"]), speed=np.array(state["speed"]))
            for name, state in values.items()}


class RecordedAlignmentGovernorTests(unittest.TestCase):
    def replay(self, name):
        cases = json.loads(FIXTURE.read_text())["cases"]
        data = next(case for case in cases if case["name"] == name)
        args = SimpleNamespace(**data["settings"])
        arrays = lambda values: {name: np.array(value) for name, value in values.items()}
        bases = arrays(data["bases"])
        alignment = LiveAlignment(args, arrays(data["offsets"]), bases, states(data["initial_states"]),
                                  arrays(data["targets"]), arrays(data["goal_joints"]))
        alignment.previous_commands = arrays(data["previous_commands"])
        alignment.previous_twists = arrays(data["previous_twists"])
        alignment.progress = data["previous_progress"]
        return alignment, states(data["states"]), bases, data["dt"], data

    def assert_command_bounds(self, alignment, measured, commands, dt):
        limits, tolerance = alignment.limits, 2e-9
        self.assertEqual(set(commands), {"ur5e"})  # The measured UR10e leader is never commanded.
        for name, command in commands.items():
            state = measured[name]
            previous = alignment.previous_commands[name]
            jacobian = alignment.models[name].pose_and_jacobian(state.joints)[1]
            predicted = state.joints + command * dt
            np.testing.assert_allclose(alignment.predicted_joints[name], predicted, atol=1e-14)
            current_twist = jacobian @ command
            future_twist = alignment.models[name].pose_and_jacobian(predicted)[1] @ command
            self.assertLessEqual(np.max(np.abs(command)), .5 * limits["max_joint_speed"] + tolerance)
            self.assertLessEqual(np.max(np.abs(command - previous)), .5 * limits["max_joint_accel"] * dt + tolerance)
            self.assertLessEqual(np.max(np.abs(command - state.joint_velocity)), limits["max_joint_accel"] * dt + tolerance)
            for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                       (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                with self.subTest(robot=name, component=speed):
                    self.assertLessEqual(np.linalg.norm(current_twist[part]), .5 * limits[speed] + tolerance)
                    self.assertLessEqual(np.linalg.norm(current_twist[part] - alignment.previous_twists[name][part]),
                                         .5 * limits[accel] * dt + tolerance)
                    self.assertLessEqual(np.linalg.norm(current_twist[part] - state.speed[part]),
                                         limits[accel] * dt + tolerance)
                    self.assertLessEqual(np.linalg.norm(future_twist[part]), limits[speed] + tolerance)
                    self.assertLessEqual(np.linalg.norm(future_twist[part] - state.speed[part]),
                                         limits[accel] * dt + tolerance)
            acceleration = alignment.joint_accelerations[name]
            self.assertGreater(acceleration, 0.)
            self.assertLessEqual(acceleration, limits["max_joint_accel"] + tolerance)
            peak = np.max(np.abs(command - state.joint_velocity))
            if peak > 0.:
                self.assertLessEqual(acceleration, peak / dt + tolerance)
                local_ramp = jacobian @ ((command - state.joint_velocity) / peak) * acceleration
                self.assertLessEqual(np.linalg.norm(local_ramp[:3]), limits["max_accel"] + tolerance)
                self.assertLessEqual(np.linalg.norm(local_ramp[3:]), limits["max_angular_accel"] + tolerance)
        np.testing.assert_array_equal(alignment.predicted_joints["ur10e"], measured["ur10e"].joints)

    def propose_checked(self, alignment, measured, bases, dt):
        limits = deepcopy(alignment.limits)
        previous = deepcopy(alignment.previous_commands)
        previous_twists = deepcopy(alignment.previous_twists)
        elapsed, progress = alignment.elapsed, alignment.progress
        commands = alignment.propose(measured, bases, dt)
        self.assert_command_bounds(alignment, measured, commands, dt)
        self.assertEqual(alignment.limits, limits)
        self.assertEqual((alignment.elapsed, alignment.progress), (elapsed, progress))
        for name in commands:
            np.testing.assert_array_equal(alignment.previous_commands[name], previous[name])
            np.testing.assert_array_equal(alignment.previous_twists[name], previous_twists[name])
            self.assertGreater(command_progress := commands[name] @ (alignment.goal_joints[name] - measured[name].joints), 0.)
            self.assertTrue(np.isfinite(command_progress))
        return commands

    def test_future_jacobian_boundary_uses_a_valid_interior_command_without_relaxing_acceleration(self):
        alignment, measured, bases, dt, data = self.replay("future_jacobian_boundary")
        with patch.object(alignment, "_predicted_motion", wraps=alignment._predicted_motion) as predictions:
            commands = self.propose_checked(alignment, measured, bases, dt)
        self.assertGreater(predictions.call_count, 1)  # The previously selected boundary fails.
        self.assertLessEqual(predictions.call_count, 20)
        self.assertEqual(alignment.limits["max_angular_accel"], .5)
        alignment.commit()
        np.testing.assert_array_equal(alignment.previous_commands["ur5e"], commands["ur5e"])
        self.assertAlmostEqual(alignment.elapsed, dt)

    def test_measured_velocity_lag_can_reduce_the_command_with_all_history_constraints_intact(self):
        alignment, measured, bases, dt, data = self.replay("measured_velocity_lag")
        jacobian = alignment.models["ur5e"].pose_and_jacobian(measured["ur5e"].joints)[1]
        old_command = alignment.previous_commands["ur5e"].copy()
        commands = self.propose_checked(alignment, measured, bases, dt)
        self.assertLess(np.linalg.norm((jacobian @ commands["ur5e"])[3:]),
                        np.linalg.norm((jacobian @ old_command)[3:]))
        self.assertLess(commands["ur5e"][5], old_command[5])

    def test_successful_normal_start_needs_one_prediction(self):
        alignment, measured, bases, dt, data = self.replay("future_jacobian_boundary")
        measured = states(data["initial_states"])
        alignment.previous_commands = {name: np.zeros(6) for name in alignment.commanded}
        alignment.previous_twists = {name: np.zeros(6) for name in alignment.commanded}
        with patch.object(alignment, "_predicted_motion", wraps=alignment._predicted_motion) as predictions:
            self.propose_checked(alignment, measured, bases, dt)
        self.assertEqual(predictions.call_count, 1)

    def test_incompatible_measured_and_previous_joint_acceleration_bounds_remain_fatal(self):
        alignment, measured, bases, dt, data = self.replay("measured_velocity_lag")
        measured["ur5e"].joint_velocity = -alignment.previous_commands["ur5e"].copy()
        jacobian = alignment.models["ur5e"].pose_and_jacobian(measured["ur5e"].joints)[1]
        measured["ur5e"].speed = jacobian @ measured["ur5e"].joint_velocity
        before = deepcopy(alignment.previous_commands)
        limits = deepcopy(alignment.limits)
        with self.assertRaises(AlignmentFault):
            alignment.propose(measured, bases, dt)
        self.assertIsNone(alignment._pending)
        with self.assertRaises(AlignmentFault):
            alignment.commit()
        self.assertEqual(alignment.limits, limits)
        self.assertEqual(alignment.elapsed, 0.)
        for name in before:
            np.testing.assert_array_equal(alignment.previous_commands[name], before[name])

    def test_unusable_future_predictions_exhaust_a_bounded_search_without_a_pending_command(self):
        alignment, measured, bases, dt, data = self.replay("future_jacobian_boundary")
        before = deepcopy(alignment.previous_commands)
        rejected = AlignmentFault("synthetic future-J constraint failure", {"metric": "angular_acceleration"})
        with patch.object(alignment, "_predicted_motion", side_effect=rejected) as predictions:
            with self.assertRaises(AlignmentFault):
                alignment.propose(measured, bases, dt)
        self.assertGreater(predictions.call_count, 1)
        self.assertLessEqual(predictions.call_count, 20)
        self.assertIsNone(alignment._pending)
        with self.assertRaises(AlignmentFault):
            alignment.commit()
        for name in before:
            np.testing.assert_array_equal(alignment.previous_commands[name], before[name])


if __name__ == "__main__":
    unittest.main()
