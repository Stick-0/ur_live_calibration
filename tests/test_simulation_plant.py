"""Captured-pose joint simulation checks; no camera or robot is accessed."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, pose_matrix, rotation_distance
from urcal.cooperative.kinematics import NominalKinematics
from urcal.simulation.plant import JointPlant, SeededKinematics, SimulationFault


ROBOTS = ("ur5e", "ur10e")
Q0 = np.array([.2, -1.1, 1.2, -.6, .7, -.4])
OFFSET = pose_matrix([.08, -.04, .12, .3, -.2, .4])
CORRECTION = pose_matrix([.003, -.004, .006, .02, -.01, .03])


def captured_initial():
    initial = {"bases": {"ur5e": pose_matrix([-2., 0, 1., 0, 0, .2]),
                         "ur10e": pose_matrix([2., 0, 1., 0, 0, -.2])}, "robots": {}}
    for name in ROBOTS:
        flange, _ = NominalKinematics(name).pose_and_jacobian(Q0, flange=True)
        initial["robots"][name] = {"q": Q0.copy(), "tcp": flange @ CORRECTION @ OFFSET,
                                   "tcp_offset": OFFSET.copy(), "joint_velocity": np.full(6, .04)}
    return initial


def args():
    return SimpleNamespace(max_joint_speed=.2, max_joint_accel=.5, workspace_config=None)


class RecordingGuard:
    def __init__(self, reject=None):
        self.lower = {name: np.full(6, -2 * np.pi) for name in ROBOTS}
        self.upper = {name: np.full(6, 2 * np.pi) for name in ROBOTS}
        self.calls = []
        self.reject = reject

    def check(self, bases, joints):
        self.calls.append((copy.deepcopy(bases), copy.deepcopy(joints)))
        return len(self.calls) != self.reject, "test obstruction", None


class SeededKinematicsTests(unittest.TestCase):
    def test_initial_measured_tcp_and_corrected_flange_preserved_for_both_robots(self):
        for name, row in captured_initial()["robots"].items():
            with self.subTest(robot=name):
                model = SeededKinematics(name, row["q"], row["tcp"], row["tcp_offset"])
                tcp, _ = model.pose_and_jacobian(Q0)
                flange, _ = model.pose_and_jacobian(Q0, flange=True)
                np.testing.assert_allclose(tcp, row["tcp"], atol=1e-12)
                np.testing.assert_allclose(flange, row["tcp"] @ inverse(OFFSET), atol=1e-12)
                np.testing.assert_allclose(model.flange_correction, CORRECTION, atol=1e-12)

    def test_corrected_jacobians_match_independent_pose_differences_away_from_seed(self):
        epsilon = 1e-6
        for name, row in captured_initial()["robots"].items():
            model = SeededKinematics(name, row["q"], row["tcp"], row["tcp_offset"])
            nominal = NominalKinematics(name)
            for flange in (False, True):
                for q in (Q0, Q0 + [.1, -.08, .03, .09, -.13, .05]):
                    with self.subTest(robot=name, flange=flange, joints=q):
                        _, jacobian = model.pose_and_jacobian(q, flange=flange)
                        finite = np.empty((6, 6))
                        for index in range(6):
                            delta = np.eye(6)[index] * epsilon
                            plus, _ = nominal.pose_and_jacobian(q + delta, flange=True)
                            minus, _ = nominal.pose_and_jacobian(q - delta, flange=True)
                            local = CORRECTION @ (np.eye(4) if flange else OFFSET)
                            plus, minus = plus @ local, minus @ local
                            finite[:3, index] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
                            finite[3:, index] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon)
                        np.testing.assert_allclose(jacobian, finite, atol=2e-8, rtol=2e-7)

    def test_nominal_link_points_retained_with_corrected_flange_and_tcp_appendages(self):
        for name, row in captured_initial()["robots"].items():
            model = SeededKinematics(name, row["q"], row["tcp"], row["tcp_offset"])
            nominal = NominalKinematics(name, OFFSET)
            points = model.points(Q0)
            np.testing.assert_allclose(points[:-2], nominal.points(Q0)[:-1], atol=1e-12)
            np.testing.assert_allclose(points[-2], (row["tcp"] @ inverse(OFFSET))[:3, 3], atol=1e-12)
            np.testing.assert_allclose(points[-1], row["tcp"][:3, 3], atol=1e-12)
            for key, link in nominal.links(Q0).items():
                np.testing.assert_allclose(model.links(Q0)[key], link, atol=1e-12)

    def test_inverse_uses_corrected_model_with_nonzero_offset(self):
        for name, row in captured_initial()["robots"].items():
            model = SeededKinematics(name, row["q"], row["tcp"], row["tcp_offset"])
            goal, _ = model.pose_and_jacobian(Q0 + [.015, -.02, .025, -.01, .02, -.02])
            solution = model.inverse(goal, Q0, np.full(6, -6.), np.full(6, 6.))
            self.assertIsNotNone(solution)
            result, _ = model.pose_and_jacobian(solution)
            self.assertLess(np.linalg.norm(result[:3, 3] - goal[:3, 3]), .0005)
            self.assertLess(rotation_distance(result, goal), .002)

    def test_inverse_fallback_recovers_reachable_near_wrist_goal_after_short_solver_exhaustion(self):
        # Invented nominal poses exercise a reachable wrist-flip goal; these
        # are not copied from the user's captured robot/calibration data.
        seed = np.array([.4, -1.2, 1.4, -.5, .08, 1.])
        target = np.array([.5, -1., 1.2, .6, -.2, .4])
        lower, upper = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
        for name in ROBOTS:
            with self.subTest(robot=name):
                nominal, _ = NominalKinematics(name).pose_and_jacobian(seed)
                model = SeededKinematics(name, seed, nominal @ CORRECTION @ OFFSET, OFFSET)
                goal, _ = model.pose_and_jacobian(target)
                self.assertIsNone(NominalKinematics.inverse(model, goal, seed, lower, upper))
                self.assertIsNotNone(NominalKinematics.inverse(model, goal, seed, lower, upper, max_iterations=256))
                result = model.inverse(goal, seed, lower, upper)
                self.assertIsNotNone(result)
                achieved, _ = model.pose_and_jacobian(result)
                self.assertLess(np.linalg.norm(achieved[:3, 3] - goal[:3, 3]), .0005)
                self.assertLess(rotation_distance(achieved, goal), .002)
                self.assertTrue(np.all(result >= lower) and np.all(result <= upper))
                np.testing.assert_array_equal(seed, [.4, -1.2, 1.4, -.5, .08, 1.])

    def test_inverse_fallback_recovers_when_unbounded_local_branch_exits_joint_limits(self):
        seed = np.array([-.5, -1.3, -1.9, -1.9, -.05, 5.1])
        target = np.array([-.55, -1.5, -2., .4, .4, 5.5])
        lower, upper = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
        for name in ROBOTS:
            with self.subTest(robot=name):
                nominal, _ = NominalKinematics(name).pose_and_jacobian(seed)
                model = SeededKinematics(name, seed, nominal @ CORRECTION @ OFFSET, OFFSET)
                goal, _ = model.pose_and_jacobian(target)
                self.assertIsNone(NominalKinematics.inverse(model, goal, seed, lower, upper, max_iterations=256))
                result = model.inverse(goal, seed, lower, upper)
                self.assertIsNotNone(result)
                achieved, _ = model.pose_and_jacobian(result)
                self.assertLess(np.linalg.norm(achieved[:3, 3] - goal[:3, 3]), .0005)
                self.assertLess(rotation_distance(achieved, goal), .002)
                self.assertTrue(np.all(result >= lower) and np.all(result <= upper))

    def test_inverse_fallback_rejects_unreachable_pose_and_joint_box_despite_solver_termination(self):
        for name, row in captured_initial()["robots"].items():
            with self.subTest(robot=name):
                model = SeededKinematics(name, row["q"], row["tcp"], row["tcp_offset"])
                unreachable = row["tcp"].copy()
                unreachable[0, 3] += 10.
                self.assertIsNone(model.inverse(unreachable, Q0, np.full(6, -6.), np.full(6, 6.)))
                shifted, _ = model.pose_and_jacobian(Q0 + [.2, 0, 0, 0, 0, 0])
                self.assertIsNone(model.inverse(shifted, Q0, Q0 - .001, Q0 + .001))
                self.assertIsNone(model.inverse(row["tcp"], Q0, Q0 + .01, Q0 + .02))

    def test_inverse_rejects_malformed_transform_or_bounds(self):
        row = captured_initial()["robots"]["ur5e"]
        model = SeededKinematics("ur5e", row["q"], row["tcp"], row["tcp_offset"])
        malformed = row["tcp"].copy()
        malformed[0, 0] = 10.
        with self.assertRaises(ValueError):
            model.inverse(malformed, Q0, np.full(6, -6.), np.full(6, 6.))
        with self.assertRaises(ValueError):
            model.inverse(row["tcp"], Q0, np.ones(6), np.zeros(6))


class JointPlantTests(unittest.TestCase):
    def setUp(self):
        self.initial = captured_initial()
        self.guard = RecordingGuard()
        self.plant = JointPlant(self.initial, args(), guard=self.guard)

    def assert_unchanged(self, before):
        for name in ROBOTS:
            np.testing.assert_array_equal(self.plant.joints[name], before[0][name])
            np.testing.assert_array_equal(self.plant.joint_velocity[name], before[1][name])
            np.testing.assert_array_equal(self.plant.bases[name], before[2][name])

    def snapshot(self):
        return copy.deepcopy((self.plant.joints, self.plant.joint_velocity, self.plant.bases))

    def test_loading_starts_at_rest_and_states_do_not_alias_capture_or_plant(self):
        state = self.plant.states(123.)
        for name in ROBOTS:
            np.testing.assert_allclose(state[name].tcp, self.initial["robots"][name]["tcp"], atol=1e-12)
            np.testing.assert_array_equal(state[name].joint_velocity, np.zeros(6))
            np.testing.assert_array_equal(state[name].speed, np.zeros(6))
            self.assertEqual(state[name].robot_timestamp, 123.)
            state[name].joints[:] = 100.
            state[name].tcp[:] = 100.
            state[name].joint_velocity[:] = 100.
            self.initial["robots"][name]["q"][:] = 100.
            np.testing.assert_array_equal(self.plant.joints[name], Q0)
        with self.assertRaises(SimulationFault):
            self.plant.states(float("nan"))

    def test_step_integrates_joints_and_reports_tcp_velocity_consistently(self):
        velocity = np.array([.02, -.03, .01, -.02, .025, .01])
        self.plant.step({"ur5e": velocity}, .1)
        np.testing.assert_allclose(self.plant.joints["ur5e"], Q0 + velocity * .1)
        np.testing.assert_array_equal(self.plant.joints["ur10e"], Q0)
        states = self.plant.states(10.)
        tip, jacobian = self.plant.models["ur5e"].pose_and_jacobian(Q0 + velocity * .1)
        np.testing.assert_allclose(states["ur5e"].tcp, tip)
        np.testing.assert_allclose(states["ur5e"].speed, jacobian @ velocity)
        self.assertEqual(len(self.guard.calls), 2)

    def test_guard_checks_midpoint_and_endpoint_with_interpolated_camera_bases(self):
        target = copy.deepcopy(self.plant.bases)
        before = target["ur5e"].copy()
        target["ur5e"][:3, 3] += [.1, -.2, .3]
        delta = Rotation.from_rotvec([.2, -.1, .3])
        target["ur5e"][:3, :3] = delta.as_matrix() @ before[:3, :3]
        command = np.full(6, .02)
        self.plant.step({"ur5e": command}, .1, bases=target)
        middle_bases, middle_joints = self.guard.calls[0]
        np.testing.assert_allclose(middle_bases["ur5e"][:3, 3], before[:3, 3] + [.05, -.1, .15])
        np.testing.assert_allclose(middle_bases["ur5e"][:3, :3], Rotation.from_rotvec(delta.as_rotvec() * .5).as_matrix() @ before[:3, :3])
        np.testing.assert_allclose(middle_joints["ur5e"], Q0 + command * .05)
        np.testing.assert_allclose(self.guard.calls[1][0]["ur5e"], target["ur5e"])
        target["ur5e"][:] = 9.
        self.assertTrue(np.isfinite(self.plant.bases["ur5e"]).all())
        self.assertNotEqual(self.plant.bases["ur5e"][0, 0], 9.)

    def test_rejected_pair_is_atomic_for_invalid_velocity_speed_acceleration_and_bounds(self):
        before = self.snapshot()
        commands = [({"ur5e": np.zeros(6), "ur10e": [0, 0, 0, 0, 0, np.nan]}, .1),
                    ({"ur5e": np.full(6, .21)}, 1.),
                    ({"ur5e": np.full(6, .1)}, .01),
                    ({"ur5e": [0] * 5}, .1), ({"unknown": np.zeros(6)}, .1),
                    ({}, 0.), ({}, -1.), ({}, np.nan)]
        for command, dt in commands:
            with self.subTest(command=command, dt=dt), self.assertRaises(SimulationFault):
                self.plant.step(command, dt)
            self.assert_unchanged(before)
        self.guard.upper["ur10e"] = Q0 + .001
        with self.assertRaisesRegex(SimulationFault, "position limit"):
            self.plant.step({name: np.full(6, .02) for name in ROBOTS}, .1)
        self.assert_unchanged(before)

    def test_guard_rejection_at_either_sample_preserves_all_state_and_bases(self):
        for reject in (1, 2):
            with self.subTest(reject=reject):
                guard = RecordingGuard(reject=reject)
                self.plant = JointPlant(self.initial, args(), guard=guard)
                before = self.snapshot()
                bases = copy.deepcopy(self.plant.bases)
                bases["ur10e"][0, 3] += .01
                with self.assertRaisesRegex(SimulationFault, "test obstruction"):
                    self.plant.step({"ur5e": np.full(6, .02)}, .1, bases)
                self.assert_unchanged(before)
                self.assertEqual(len(guard.calls), reject)

    def test_default_workspace_guard_rejects_collocated_arms(self):
        plant = JointPlant(self.initial, args())
        collocated = {name: np.eye(4) for name in ROBOTS}
        before = copy.deepcopy(plant.bases)
        with self.assertRaisesRegex(SimulationFault, "inter-arm capsule clearance"):
            plant.step({}, .1, collocated)
        for name in ROBOTS:
            np.testing.assert_array_equal(plant.bases[name], before[name])

    def test_stop_and_reset_freeze_virtual_motion_then_restore_capture(self):
        bases = copy.deepcopy(self.plant.bases)
        bases["ur5e"][0, 3] += .01
        self.plant.step({"ur5e": np.full(6, .02)}, .1, bases)
        moved = self.plant.joints["ur5e"].copy()
        self.plant.stop()
        np.testing.assert_array_equal(self.plant.joints["ur5e"], moved)
        np.testing.assert_array_equal(self.plant.states(4.)["ur5e"].speed, np.zeros(6))
        self.initial["bases"]["ur5e"][:] = 500.
        self.plant.reset()
        np.testing.assert_array_equal(self.plant.joints["ur5e"], Q0)
        np.testing.assert_array_equal(self.plant.bases["ur5e"], captured_initial()["bases"]["ur5e"])

    def test_virtual_hold_servo_maps_twist_then_limits_slew_without_mutation(self):
        before = self.snapshot()
        model = self.plant.models["ur5e"]
        _, jacobian = model.pose_and_jacobian(Q0)
        desired_velocity = np.array([.01, -.02, .01, .015, -.025, .005])
        command = self.plant.twists_to_qdots({"ur5e": jacobian @ desired_velocity}, .1)
        np.testing.assert_allclose(command["ur5e"], desired_velocity, atol=1e-7)
        np.testing.assert_array_equal(command["ur10e"], np.zeros(6))
        self.assert_unchanged(before)
        self.plant.step(command, .1)
        old_velocity = self.plant.joint_velocity["ur5e"].copy()
        limited = self.plant.twists_to_qdots({"ur5e": np.full(6, 100.)}, .01)
        self.assertTrue(np.all(np.abs(limited["ur5e"]) <= .2))
        self.assertTrue(np.all(np.abs(limited["ur5e"] - old_velocity) <= .005 + 1e-12))
        self.plant.step(limited, .01)

    def test_bad_initial_pose_or_velocity_configuration_is_rejected(self):
        bad = copy.deepcopy(self.initial)
        bad["robots"]["ur10e"]["tcp"][0, 0] = 10.
        with self.assertRaises(SimulationFault):
            JointPlant(bad, args(), guard=self.guard)
        for value in (0., -1., np.nan):
            bad_args = args()
            bad_args.max_joint_accel = value
            with self.assertRaises(SimulationFault):
                JointPlant(self.initial, bad_args, guard=self.guard)

    def test_joint_slew_limiting_preserves_cartesian_command_direction_at_rest(self):
        # This wrist configuration used to enlarge the requested translational
        # speed when each joint was clipped independently, blocking hold mode.
        q = np.array([-.61942049, .22685986, .50310870, -.00980895, .89066485, -.97300499])
        self.plant.joints["ur5e"] = q
        twist = np.array([-.00294497, -.00090988, -.00254937, -.0008544, -.01854561, .00743843])
        command = self.plant.twists_to_qdots({"ur5e": twist}, .04)["ur5e"]
        _, jacobian = self.plant.models["ur5e"].pose_and_jacobian(q)
        raw = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), twist)
        ratio = command[np.argmax(np.abs(raw))] / raw[np.argmax(np.abs(raw))]
        self.assertGreater(ratio, 0.)
        self.assertLess(ratio, 1.)
        np.testing.assert_allclose(command, ratio * raw, atol=1e-12)
        self.assertLessEqual(np.linalg.norm((jacobian @ command)[:3]), np.linalg.norm(twist[:3]) + 1e-8)

    def test_joint_speed_and_slew_use_one_fraction_from_previous_velocity(self):
        previous = np.array([.19, -.19, .04, -.02, .01, -.01])
        self.plant.joint_velocity["ur5e"] = previous
        _, jacobian = self.plant.models["ur5e"].pose_and_jacobian(Q0)
        desired = np.array([.8, -.9, -.7, .4, -.5, .7])
        twist = jacobian @ desired
        actual = self.plant.twists_to_qdots({"ur5e": twist}, .1)["ur5e"]
        solved = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), twist)
        delta = solved - previous
        ratio = (actual - previous)[np.argmax(np.abs(delta))] / delta[np.argmax(np.abs(delta))]
        np.testing.assert_allclose(actual - previous, ratio * delta, atol=1e-12)
        self.assertTrue(np.all(np.abs(actual) <= .2 + 1e-12))
        self.assertTrue(np.all(np.abs(actual - previous) <= .05 + 1e-12))


if __name__ == "__main__":
    unittest.main()
