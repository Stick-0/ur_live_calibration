"""Initial-approach path/time boundaries with an analytic virtual test model."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from urcal.control.geometry import pose_matrix, rotation_distance
from urcal.simulation.approach import InitialApproach
from urcal.simulation.plant import JointPlant, SimulationFault
from urcal.tools.simulate import CartesianModel


ROBOTS = ("ur5e", "ur10e")


class Guard:
    """Empty test workspace with optional analytic blocked configurations."""

    def __init__(self, blocked=None):
        self.lower = {name: np.full(6, -2 * np.pi) for name in ROBOTS}
        self.upper = {name: np.full(6, 2 * np.pi) for name in ROBOTS}
        self.blocked = blocked
        self.checks = 0

    def pose_allowed(self, name, pose):
        return True

    def check(self, bases, joints):
        self.checks += 1
        return not (self.blocked and self.blocked(joints)), "test obstacle", None


def fixture():
    args = SimpleNamespace(control_hz=25., max_joint_speed=.5, max_joint_accel=1.,
                           max_speed=.05, max_angular_speed=.2, max_accel=.2, max_angular_accel=.6,
                           min_gap=.15, max_travel=.2, max_target_m=.1, max_target_deg=25., workspace_config=None)
    models = {name: CartesianModel(np.eye(3)) for name in ROBOTS}
    for model in models.values():
        model.tcp_offset = np.eye(4)
    initial = {"bases": {"ur5e": pose_matrix([-.5, 0, 1, 0, 0, 0]),
                          "ur10e": pose_matrix([.5, 0, 1, 0, 0, 0])}, "robots": {}}
    for name, model in models.items():
        q = np.array([0., 0., .2, 0., 0., 0.])
        initial["robots"][name] = {"q": q, "tcp": model.pose_and_jacobian(q)[0], "tcp_offset": np.eye(4)}
    guard = Guard()
    plant = JointPlant(initial, args, models=models, guard=guard)
    targets = {"ur5e": pose_matrix([.06, .01, .2, 1.5, 0, 0]),
               "ur10e": pose_matrix([0, -.03, .2, 0, 0, 0])}
    return args, models, guard, plant, targets


class InitialApproachTests(unittest.TestCase):
    def plan(self, fixture_values=None):
        args, models, guard, plant, targets = fixture_values or fixture()
        return InitialApproach(models, guard, args, plant.bases, plant.states(1000.), targets)

    def test_full_trajectory_preserves_final_targets_and_every_motion_limit(self):
        args, models, guard, plant, targets = fixture()
        original_q = copy.deepcopy(plant.joints)
        original_targets = copy.deepcopy(targets)
        approach = self.plan((args, models, guard, plant, targets))
        self.assertGreater(rotation_distance(plant.states(1000.)["ur5e"].tcp, targets["ur5e"]),
                           np.deg2rad(args.max_target_deg))
        self.assertGreaterEqual(guard.checks, 81)
        count = 0
        while not approach.done:
            before = plant.states(1000. + count * approach.dt)
            qdots = approach.propose(before)
            for name in ROBOTS:
                self.assertLessEqual(np.max(np.abs(qdots[name])), args.max_joint_speed + 1e-9)
                self.assertLessEqual(np.max(np.abs(qdots[name] - before[name].joint_velocity)),
                                     args.max_joint_accel * approach.dt + 1e-9)
            plant.step(qdots, approach.dt)
            after = plant.states(1000. + (count + 1) * approach.dt)
            for name in ROBOTS:
                for part, speed, accel in ((slice(0, 3), args.max_speed, args.max_accel),
                                           (slice(3, 6), args.max_angular_speed, args.max_angular_accel)):
                    self.assertLessEqual(np.linalg.norm(after[name].speed[part]), speed + 1e-9)
                    self.assertLessEqual(np.linalg.norm(after[name].speed[part] - before[name].speed[part]),
                                         accel * approach.dt + 1e-9)
            approach.commit()
            count += 1
        self.assertAlmostEqual(approach.duration, count * approach.dt)
        self.assertEqual(approach.progress, 1.)
        for name in ROBOTS:
            np.testing.assert_array_equal(plant.joint_velocity[name], np.zeros(6))
            np.testing.assert_allclose(plant.states(2000.)[name].tcp, original_targets[name], atol=1e-12)
            np.testing.assert_array_equal(approach.initial_joints[name], original_q[name])
            np.testing.assert_array_equal(targets[name], original_targets[name])

    def test_propose_is_repeatable_and_only_successful_commit_advances_progress(self):
        args, models, guard, plant, targets = fixture()
        approach = self.plan((args, models, guard, plant, targets))
        with self.assertRaisesRegex(SimulationFault, "successful proposal"):
            approach.commit()
        before = copy.deepcopy(plant.joints)
        first = approach.propose(plant.states(1000.))
        second = approach.propose(plant.states(1000.))
        self.assertEqual(approach.progress, 0.)
        for name in ROBOTS:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(plant.joints[name], before[name])
        plant.step(second, approach.dt)
        approach.commit()
        self.assertGreater(approach.progress, 0.)
        with self.assertRaisesRegex(SimulationFault, "successful proposal"):
            approach.commit()

    def test_follow_approach_keeps_the_uncommanded_leader_at_its_captured_pose(self):
        args, models, guard, plant, targets = fixture()
        targets = {"ur5e": targets["ur5e"]}
        approach = self.plan((args, models, guard, plant, targets))
        before = plant.joints["ur10e"].copy()
        for _ in range(5):
            commands = approach.propose(plant.states(1000.))
            np.testing.assert_array_equal(commands["ur10e"], np.zeros(6))
            plant.step(commands, approach.dt)
            approach.commit()
        np.testing.assert_array_equal(approach.goal_joints["ur10e"], before)
        np.testing.assert_array_equal(plant.joints["ur10e"], before)

    def test_source_arrays_are_copied_and_do_not_retarget_the_plan(self):
        args, models, guard, plant, targets = fixture()
        states = plant.states(1000.)
        bases = copy.deepcopy(plant.bases)
        approach = InitialApproach(models, guard, args, bases, states, targets)
        targets["ur5e"][0, 3] = 10.
        bases["ur5e"][0, 3] = 10.
        states["ur5e"].joints[:] = 10.
        states["ur5e"].tcp[0, 3] = 10.
        self.assertEqual(approach.targets["ur5e"][0, 3], .06)
        self.assertEqual(approach.bases["ur5e"][0, 3], -.5)
        self.assertEqual(approach.initial_joints["ur5e"][0], 0.)
        self.assertEqual(approach.initial_tcp["ur5e"][0, 3], 0.)

    def test_motion_starts_at_rest_and_invalid_limits_are_rejected(self):
        args, models, guard, plant, targets = fixture()
        states = plant.states(1000.)
        states["ur5e"].joint_velocity[0] = .01
        with self.assertRaisesRegex(SimulationFault, "start from rest"):
            InitialApproach(models, guard, args, plant.bases, states, targets)
        args.max_joint_accel = 0.
        with self.assertRaisesRegex(SimulationFault, "finite and positive"):
            self.plan((args, models, guard, plant, targets))

    def test_an_obstructed_middle_of_the_joint_path_is_rejected_without_moving(self):
        args, models, guard, plant, targets = fixture()
        guard.blocked = lambda joints: .02 < joints["ur5e"][0] < .04
        before = copy.deepcopy(plant.joints)
        with self.assertRaisesRegex(SimulationFault, "no feasible.*test obstacle"):
            self.plan((args, models, guard, plant, targets))
        for name in ROBOTS:
            np.testing.assert_array_equal(plant.joints[name], before[name])

    def test_no_ik_solution_reports_which_arm_failed(self):
        args, models, guard, plant, targets = fixture()
        models["ur10e"].inverse = lambda *arguments: None
        with self.assertRaisesRegex(SimulationFault, "ur10e: initial approach IK failed"):
            self.plan((args, models, guard, plant, targets))

    def test_minimum_flange_gap_is_checked_along_the_pair_path(self):
        args, models, guard, plant, targets = fixture()
        plant.bases["ur5e"][:3, 3] = [0, 0, 1]
        plant.bases["ur10e"][:3, 3] = [.3, 0, 1]
        targets = {"ur5e": pose_matrix([.1, 0, .2, 0, 0, 0]),
                   "ur10e": pose_matrix([-.1, 0, .2, 0, 0, 0])}
        with self.assertRaisesRegex(SimulationFault, "flange separation below min_gap"):
            self.plan((args, models, guard, plant, targets))

    def test_tcp_and_flange_start_envelopes_are_checked_separately(self):
        for kind in ("TCP", "flange"):
            with self.subTest(kind=kind):
                args, models, guard, plant, targets = fixture()
                args.max_travel = .03 if kind == "TCP" else .15
                if kind == "flange":
                    models["ur5e"].tcp_offset = pose_matrix([0, 0, .1, 0, 0, 0])
                    targets["ur5e"] = pose_matrix([0, 0, .2, 2.5, 0, 0])
                with self.assertRaisesRegex(SimulationFault, kind + " travel envelope"):
                    self.plan((args, models, guard, plant, targets))

    def test_changed_measured_state_cannot_bypass_joint_slew_limit_or_commit(self):
        args, models, guard, plant, targets = fixture()
        approach = self.plan((args, models, guard, plant, targets))
        plant.joints["ur5e"][0] -= .1
        with self.assertRaisesRegex(SimulationFault, "joint speed/acceleration"):
            approach.propose(plant.states(1000.))
        self.assertEqual(approach.progress, 0.)
        with self.assertRaisesRegex(SimulationFault, "successful proposal"):
            approach.commit()

    def test_reducing_speed_limits_extends_duration_instead_of_changing_targets(self):
        values = fixture()
        fast = self.plan(values)
        values[0].max_angular_speed /= 2
        slow = self.plan(values)
        self.assertGreater(slow.duration, fast.duration * 1.9)
        for name in ROBOTS:
            np.testing.assert_array_equal(slow.targets[name], fast.targets[name])
            np.testing.assert_array_equal(slow.goal_joints[name], fast.goal_joints[name])


if __name__ == "__main__":
    unittest.main()
