"""Role-specific leader rest thresholds with recorded and virtual encoders."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tests.test_live_alignment import fixture
from urcal.control.alignment import AlignmentFault, LiveAlignment
from urcal.control.policy import ControlFault, stationary_velocity_limits
from urcal.control.runtime import parse_args
from urcal.control.startup import check_handoff_snapshot, check_start_snapshot, check_stationary


CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]


class LeaderRestLimitTests(unittest.TestCase):
    def test_defaults_and_override_validation(self):
        args = parse_args(CLI)
        self.assertEqual(stationary_velocity_limits(args, leader=True), (.025, .01, .05))
        self.assertEqual(stationary_velocity_limits(args), (.005, .002, .01))
        old = parse_args(CLI + ["--align_leader_max_speed", ".002",
                               "--align_leader_max_joint_speed", ".005",
                               "--align_leader_max_angular_speed", ".01"])
        self.assertEqual(stationary_velocity_limits(old, leader=True), (.005, .002, .01))
        for name in ("align_leader_max_speed", "align_leader_max_joint_speed", "align_leader_max_angular_speed"):
            for value in ("0", "-1", "nan", "inf"):
                with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(CLI + [f"--{name}={value}"])

    def test_saved_namespaces_keep_original_limits_and_reject_bad_values(self):
        self.assertEqual(stationary_velocity_limits(SimpleNamespace(), leader=True), (.005, .002, .01))
        for value in (0., -.1, np.nan, np.inf):
            with self.assertRaises(ControlFault):
                stationary_velocity_limits(SimpleNamespace(align_leader_max_speed=value), leader=True)

    def plan(self, mode="follow", leader="ur10e"):
        args, offsets, plant, targets, goals = fixture("hold", already_at_goal=True)
        args.mode, args.leader = mode, leader
        if mode == "follow":
            follower = next(n for n in targets if n != leader)
            targets, goals = {follower: targets[follower]}, {follower: goals[follower]}
        plan = LiveAlignment(args, offsets, plant.bases, plant.states(100.), targets, goals)
        for _ in range(5):
            plan.propose(plant.states(100.), plant.bases, .04)
            plan.commit()
        self.assertTrue(plan.done)
        return args, plant, plan

    def test_both_leader_choices_pass_new_limits_in_handoff_and_startup(self):
        for leader in ("ur5e", "ur10e"):
            args, plant, plan = self.plan(leader=leader)
            states = plant.states(101.)
            states[leader].joint_velocity[0] = .024
            states[leader].speed[:] = [.009, 0, 0, 0, 0, .049]
            check_stationary(states, args)
            check_start_snapshot(args, plant.bases, states, plant.bases, states)
            commands = plan.propose_hold(states, plant.bases, .04)
            self.assertNotIn(leader, commands)
            for command in commands.values():
                np.testing.assert_array_equal(command, np.zeros(6))

    def test_speed_limits_still_reject_above_each_new_threshold(self):
        for field, index, speed in (("joint_velocity", 0, .02501), ("speed", 0, .01001), ("speed", 5, .05001)):
            args, plant, plan = self.plan()
            states = plant.states(101.)
            getattr(states[args.leader], field)[index] = speed
            with self.assertRaises(ControlFault):
                check_stationary(states, args)
            with self.assertRaisesRegex(AlignmentFault, "actual/limit") as caught:
                plan.propose_hold(states, plant.bases, .04)
            self.assertEqual(caught.exception.details["linear_limit"], .01)

    def test_follower_and_both_hold_arms_keep_original_settling_threshold(self):
        for mode, name in (("follow", "ur5e"), ("hold", "ur5e"), ("hold", "ur10e")):
            args, plant, plan = self.plan(mode)
            states = plant.states(101.)
            states[name].speed[0] = .003
            with self.assertRaises(ControlFault):
                check_stationary(states, args)
            with self.assertRaisesRegex(AlignmentFault, "stationary"):
                plan.propose_hold(states, plant.bases, .04)

    def test_relaxed_velocity_does_not_allow_leader_position_drift(self):
        args, plant, plan = self.plan()
        initial, states = plant.states(100.), plant.states(101.)
        states[args.leader].tcp[0, 3] += .0031
        states[args.leader].speed[0] = .005
        with self.assertRaisesRegex(AlignmentFault, "fixed leader moved"):
            plan.propose_hold(states, plant.bases, .04)
        with self.assertRaisesRegex(ControlFault, "robot moved"):
            check_start_snapshot(args, plant.bases, initial, plant.bases, states)

    def test_recorded_2_16mm_per_second_sample_passes_zero_command_and_handoff_checks(self):
        data = json.loads((Path(__file__).parent / "fixtures/alignment_leader_rest_speed.json").read_text())
        args = parse_args(CLI)
        for key, value in data["settings"].items():
            setattr(args, key, value)
        def states(field):
            return {name: SimpleNamespace(joints=np.array(s["q"]), joint_velocity=np.array(s["qd"]),
                                         tcp=np.array(s["tcp"]), speed=np.array(s.get("speed", s.get("tcp_speed"))))
                    for name, s in data[field].items()}
        bases = {name: np.array(pose) for name, pose in data["initial_bases"].items()}
        targets = {name: np.array(pose) for name, pose in data["targets"].items()}
        goals = {name: np.array(data["goal_joints"][name]) for name in targets}
        offsets = {name: np.eye(4) for name in bases}
        measured = states("fault_robots")
        plan = LiveAlignment(args, offsets, bases, states("initial_robots"), targets, goals)
        plan.done, plan.progress = True, 1.
        before = deepcopy(measured)
        self.assertAlmostEqual(np.linalg.norm(measured[args.leader].speed[:3]), .0021588281247)
        self.assertFalse(plan._at_rest(measured[args.leader]))
        commands = plan.propose_hold(measured, bases, .04)
        np.testing.assert_array_equal(commands["ur5e"], np.zeros(6))
        check_handoff_snapshot(args, bases, states("endpoint_robots"), bases, measured,
                               offsets=offsets, nominal_targets=targets, live_targets=targets)
        # Isolate the old 2 mm/s limit: it reproduces the reported rejection.
        plan.leader_rest_limits = (.025, .002, .05)
        with self.assertRaisesRegex(AlignmentFault, "fixed leader moved"):
            plan.propose_hold(measured, bases, .04)
        for name in measured:
            np.testing.assert_array_equal(measured[name].tcp, before[name].tcp)
            np.testing.assert_array_equal(measured[name].speed, before[name].speed)


if __name__ == "__main__":
    unittest.main()
