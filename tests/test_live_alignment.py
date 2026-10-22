"""Live-alignment policy tested against virtual joints, without robot drivers."""

import copy
import subprocess
import sys
import unittest

import numpy as np

from urcal.control.alignment import AlignmentFault, LiveAlignment
from urcal.control.geometry import pose_matrix, rotation_distance
from urcal.control.runtime import parse_args
from urcal.cooperative.kinematics import NominalKinematics
from urcal.simulation.plant import JointPlant


ROBOTS = ("ur5e", "ur10e")


def fixture(mode="hold", already_at_goal=False):
    args = parse_args(["--mode", mode, "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2",
                       "--align_speed_scale", "1",
                       "--max_speed", ".1", "--max_angular_speed", ".4", "--max_accel", ".4",
                       "--max_angular_accel", "1", "--max_joint_speed", ".8", "--max_joint_accel", "1"])
    offsets = {name: pose_matrix([.02, 0, .04, .1, -.2, .3]) for name in ROBOTS}
    initial = {"bases": {"ur5e": pose_matrix([-2, 0, 1, 0, 0, .2]),
                         "ur10e": pose_matrix([2, 0, 1, 0, 0, -.2])}, "robots": {}}
    for name in ROBOTS:
        q = np.array([.2, -1.1, 1.2, -.6, .7, -.4])
        tcp = NominalKinematics(name, offsets[name]).pose_and_jacobian(q)[0]
        initial["robots"][name] = {"q": q, "tcp": tcp, "tcp_offset": offsets[name]}
    plant = JointPlant(initial, args)
    commanded = ROBOTS if mode == "hold" else ("ur5e",)
    goals, targets = {}, {}
    for name in commanded:
        goals[name] = plant.joints[name].copy()
        if not already_at_goal:
            goals[name][5] += 1.2 if name == "ur5e" else -1.
        targets[name] = plant.models[name].pose_and_jacobian(goals[name])[0]
    return args, offsets, plant, targets, goals


def make_plan(values, states=None, guard=None):
    args, offsets, plant, targets, goals = values
    return LiveAlignment(args, offsets, plant.bases, plant.states(1000.) if states is None else states,
                         targets, goals, guard=guard)


class LiveAlignmentTests(unittest.TestCase):
    def test_large_orientation_alignment_converges_with_ideal_and_lagging_measured_joints(self):
        for mode, response, speed_scale in (("hold", 1., 1.), ("follow", .6, 1.), ("hold", 1., 2.)):
            with self.subTest(mode=mode, response=response, speed_scale=speed_scale):
                values = fixture(mode)
                args, offsets, plant, targets, goals = values
                args.align_speed_scale = speed_scale
                plan = make_plan(values)
                self.assertEqual(len(plan.samples), 81)
                self.assertGreater(rotation_distance(plant.states(0.)["ur5e"].tcp, targets["ur5e"]),
                                   np.deg2rad(args.max_target_deg))
                dt = .04
                previous = {name: np.zeros(6) for name in plan.commanded}
                original_leader = plant.joints["ur10e"].copy()
                for index in range(1000):
                    states = plant.states(1000. + index * dt)
                    commands = plan.propose(states, plant.bases, dt)
                    self.assertEqual(set(commands), set(plan.commanded))
                    for name, command in commands.items():
                        np.testing.assert_allclose(plan.predicted_joints[name], states[name].joints + command * dt)
                        self.assertLessEqual(np.max(np.abs(command)), args.max_joint_speed * .5 * speed_scale + 1e-9)
                        self.assertLessEqual(np.max(np.abs(command - previous[name])), args.max_joint_accel * .5 * dt + 1e-9)
                        _, jacobian = plan.models[name].pose_and_jacobian(states[name].joints)
                        twist = jacobian @ command
                        self.assertLessEqual(np.linalg.norm(twist[:3]), args.max_speed * .5 * speed_scale + 1e-9)
                        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed * .5 * speed_scale + 1e-9)
                    applied = {name: response * command + (1 - response) * states[name].joint_velocity
                               for name, command in commands.items()}
                    plant.step(applied, dt)
                    plan.commit()
                    previous = commands
                    if plan.done:
                        break
                self.assertTrue(plan.done, plan.message)
                self.assertEqual(plan.progress, 1.)
                self.assertLess(plan.elapsed, plan.timeout)
                for name, target in targets.items():
                    state = plant.states(2000.)[name]
                    self.assertLessEqual(np.linalg.norm(state.tcp[:3, 3] - target[:3, 3]), .002)
                    self.assertLessEqual(rotation_distance(state.tcp, target), np.deg2rad(1.))
                    self.assertLessEqual(np.max(np.abs(state.joints - goals[name])), .01)
                    self.assertTrue(plan._at_rest(state))
                if mode == "follow":
                    np.testing.assert_array_equal(plant.joints["ur10e"], original_leader)

    def test_no_progress_is_committed_without_a_successful_send(self):
        values = fixture()
        plan = make_plan(values)
        before = copy.deepcopy(values[2].joints)
        with self.assertRaisesRegex(AlignmentFault, "successful proposal"):
            plan.commit()
        first = plan.propose(values[2].states(0.), values[2].bases, .04)
        second = plan.propose(values[2].states(0.), values[2].bases, .04)
        self.assertEqual(plan.elapsed, 0.)
        self.assertEqual(plan.progress, 0.)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(values[2].joints[name], before[name])
        values[2].step(second, .04)
        plan.commit()
        self.assertEqual(plan.elapsed, .04)
        with self.assertRaisesRegex(AlignmentFault, "successful proposal"):
            plan.commit()

    def test_speedj_ramp_matches_measured_velocity_delta_without_mutating_state(self):
        values = fixture()
        args, _, plant, _, _ = values
        plan = make_plan(values)
        states = plant.states(0.)
        # Feedback may lag the previous command; acceleration must start at
        # this measured velocity, rather than the command history.
        states["ur5e"].joint_velocity[5] = .003
        states["ur5e"].speed = (plan.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[1]
                                @ states["ur5e"].joint_velocity)
        before = copy.deepcopy(states)
        dt = .04
        commands = plan.propose(states, plant.bases, dt)
        for name, command in commands.items():
            difference = command - states[name].joint_velocity
            peak = float(np.max(np.abs(difference)))
            self.assertGreater(peak, 0.)
            acceleration = plan.joint_accelerations[name]
            self.assertTrue(np.isfinite(acceleration))
            self.assertGreater(acceleration, 0.)
            self.assertLessEqual(acceleration, peak / dt + 1e-12)
            self.assertAlmostEqual(acceleration, peak / dt)
            self.assertLessEqual(acceleration, args.max_joint_accel)
            jacobian = plan.models[name].pose_and_jacobian(states[name].joints)[1]
            direction = jacobian @ (difference / peak)
            self.assertLessEqual(acceleration * np.linalg.norm(direction[:3]), args.max_accel + 1e-12)
            self.assertLessEqual(acceleration * np.linalg.norm(direction[3:]), args.max_angular_accel + 1e-12)
            for attribute in ("joints", "joint_velocity", "tcp", "speed"):
                np.testing.assert_array_equal(getattr(states[name], attribute), getattr(before[name], attribute))
            np.testing.assert_array_equal(plan.previous_commands[name], np.zeros(6))

    def test_tiny_nonzero_speedj_ramp_is_not_floored_and_zero_delta_stays_positive(self):
        values = fixture(already_at_goal=True)
        plan = make_plan(values)
        self.assertEqual(plan.joint_accelerations, {name: 1e-6 for name in ROBOTS})
        states = values[2].states(0.)
        states["ur5e"].joint_velocity[5] = 1e-14
        states["ur5e"].speed = (plan.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[1]
                                @ states["ur5e"].joint_velocity)
        commands = plan.propose(states, values[2].bases, .04)
        for command in commands.values():
            np.testing.assert_array_equal(command, np.zeros(6))
        self.assertEqual(plan.joint_accelerations["ur5e"], 1e-14 / .04)
        self.assertLess(plan.joint_accelerations["ur5e"], 1e-6)
        self.assertEqual(plan.joint_accelerations["ur10e"], 1e-6)
        self.assertEqual(states["ur5e"].joint_velocity[5], 1e-14)

    def test_completion_requires_five_zero_command_and_measured_rest_ticks(self):
        values = fixture(already_at_goal=True)
        plan = make_plan(values)
        for count in range(5):
            commands = plan.propose(values[2].states(0.), values[2].bases, .04)
            for command in commands.values():
                np.testing.assert_array_equal(command, np.zeros(6))
            plan.commit()
            self.assertEqual(plan.done, count == 4)

    def test_native_joint_endpoint_does_not_replace_measured_tcp_verification(self):
        values = fixture(already_at_goal=True)
        plan = make_plan(values)
        states = values[2].states(0.)
        states["ur5e"].tcp[0, 3] += .01  # Inside model tolerance, outside final TCP tolerance.
        with self.assertRaisesRegex(AlignmentFault, "measured TCP does not match"):
            for _ in range(30):
                plan.propose(states, values[2].bases, .04)
                plan.commit()
        self.assertFalse(plan.done)

    def test_initial_rest_accepts_small_measured_noise_but_rejects_motion(self):
        values = fixture()
        states = values[2].states(0.)
        states["ur5e"].joint_velocity[5] = .004
        states["ur5e"].speed = values[2].models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[1] @ states["ur5e"].joint_velocity
        make_plan(values, states)
        states["ur5e"].joint_velocity[5] = .006
        with self.assertRaisesRegex(AlignmentFault, "stop the robot"):
            make_plan(values, states)

    def test_cumulative_base_translation_and_rotation_drift_stop_alignment(self):
        for kind in ("translation", "rotation"):
            with self.subTest(kind=kind):
                values = fixture("follow")
                plan = make_plan(values)
                bases = copy.deepcopy(values[2].bases)
                if kind == "translation":
                    bases["ur5e"][0, 3] += .006
                    plan.propose(values[2].states(0.), bases, .04)
                    bases["ur5e"][0, 3] += .006
                else:
                    bases["ur5e"] = bases["ur5e"] @ pose_matrix([0, 0, 0, 0, 0, np.deg2rad(1.1)])
                with self.assertRaisesRegex(AlignmentFault, "base moved"):
                    plan.propose(values[2].states(0.), bases, .04)
                with self.assertRaises(AlignmentFault):
                    plan.commit()

    def test_fixed_leader_motion_position_or_speed_is_rejected(self):
        for change in ("position", "speed"):
            values = fixture("follow")
            plan = make_plan(values)
            if change == "position":
                values[2].joints["ur10e"][0] += .011
            else:
                values[2].joint_velocity["ur10e"][5] = values[0].align_leader_max_joint_speed + .001
            with self.assertRaisesRegex(AlignmentFault, "fixed leader moved"):
                plan.propose(values[2].states(0.), values[2].bases, .04)

    def test_measured_joints_outside_the_planned_corridor_are_rejected(self):
        values = fixture("follow")
        plan = make_plan(values)
        values[2].joints["ur5e"][0] += .04
        with self.assertRaisesRegex(AlignmentFault, "left the.*corridor"):
            plan.propose(values[2].states(0.), values[2].bases, .04)

    def test_native_endpoint_model_mismatch_and_travel_envelope_fail_before_commands(self):
        values = fixture()
        values[3]["ur5e"][0, 3] += .04
        with self.assertRaisesRegex(AlignmentFault, "native IK endpoint disagrees"):
            make_plan(values)
        values = fixture()
        values[4]["ur5e"][1] += .7
        values[3]["ur5e"] = values[2].models["ur5e"].pose_and_jacobian(values[4]["ur5e"])[0]
        with self.assertRaisesRegex(AlignmentFault, "travel envelope"):
            make_plan(values)

    def test_entire_native_joint_path_is_checked_before_start(self):
        values = fixture()
        guard = values[2].guard
        original = guard.check
        guard.check = lambda bases, joints: ((False, "path obstruction", None)
                                            if -.2 < joints["ur5e"][5] < 0 else original(bases, joints))
        with self.assertRaisesRegex(AlignmentFault, "path obstruction"):
            make_plan(values, guard=guard)

    def test_timeout_invalid_period_nonfinite_state_and_measured_overspeed_fail_closed(self):
        for fault in ("timeout", "period", "nan", "speed"):
            values = fixture()
            plan = make_plan(values)
            states, dt = values[2].states(0.), .04
            if fault == "timeout":
                plan.elapsed = plan.timeout
            elif fault == "period":
                dt = .2
            elif fault == "nan":
                states["ur5e"].tcp[0, 3] = np.nan
            else:
                states["ur5e"].joint_velocity[5] = 10.
            with self.subTest(fault=fault), self.assertRaises(AlignmentFault):
                plan.propose(states, values[2].bases, dt)
            with self.assertRaises(AlignmentFault):
                plan.commit()
            self.assertFalse(plan.done)

    def test_goal_and_capture_arrays_are_copied(self):
        values = fixture("follow")
        states = values[2].states(0.)
        plan = make_plan(values, states)
        targets = copy.deepcopy(plan.targets)
        goals = copy.deepcopy(plan.goal_joints)
        values[3]["ur5e"][0, 3] = 10.
        values[4]["ur5e"][:] = 10.
        states["ur5e"].joints[:] = 10.
        values[2].bases["ur5e"][0, 3] = 10.
        np.testing.assert_array_equal(plan.targets["ur5e"], targets["ur5e"])
        np.testing.assert_array_equal(plan.goal_joints["ur5e"], goals["ur5e"])
        self.assertEqual(plan.initial_joints["ur5e"][0], .2)
        self.assertEqual(plan.initial_bases["ur5e"][0, 3], -2.)

    def test_import_never_loads_robot_or_camera_drivers(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import urcal.control.alignment; "
                                 "assert not {'rtde_control','rtde_receive','pyrealsense2'} & set(sys.modules)"],
                                capture_output=True, text=True, timeout=10.)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
