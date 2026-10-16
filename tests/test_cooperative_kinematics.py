"""Independent kinematics/geometry checks without camera or robot access."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard, segment_distance
from urcal.control.geometry import pose_matrix, rotation_distance


def dh_flange(robot, joints):
    """Independent standard-DH FK, rather than the production URDF link tree."""
    if robot == "ur5e":
        a, d = [0, -.425, -.3922, 0, 0, 0], [.1625, 0, 0, .1333, .0997, .0996]
    else:
        a, d = [0, -.6127, -.57155, 0, 0, 0], [.1807, 0, 0, .17415, .11985, .11655]
    alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    result = np.eye(4)
    for theta, length, height, twist in zip(joints, a, d, alpha):
        c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(twist), np.sin(twist)
        result = result @ np.array([[c, -s*ca, s*sa, length*c],
                                    [s, c*ca, -c*sa, length*s],
                                    [0, sa, ca, height], [0, 0, 0, 1]])
    return result


class NominalKinematicsTests(unittest.TestCase):
    def test_jacobian_matches_independent_dh_finite_differences_both_robots_and_tcp_offsets(self):
        generator = np.random.default_rng(8021)
        poses = [np.zeros(6), np.array([.2, -1.1, 1.2, -.6, .7, -.4]),
                 *generator.uniform(-2., 2., (5, 6))]
        offsets = [np.eye(4), pose_matrix([.08, -.04, .12, .3, -.2, .4])]
        epsilon = 1e-6
        for robot in ("ur5e", "ur10e"):
            for offset in offsets:
                model = NominalKinematics(robot, offset)
                for q in poses:
                    with self.subTest(robot=robot, offset=offset[:3, 3], q=q):
                        tip, jacobian = model.pose_and_jacobian(q)
                        expected = dh_flange(robot, q) @ offset
                        np.testing.assert_allclose(tip, expected, atol=1e-12)
                        finite = np.empty((6, 6))
                        for index in range(6):
                            delta = np.zeros(6)
                            delta[index] = epsilon
                            plus, minus = dh_flange(robot, q + delta) @ offset, dh_flange(robot, q - delta) @ offset
                            finite[:3, index] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
                            finite[3:, index] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon)
                        np.testing.assert_allclose(jacobian, finite, atol=2e-8, rtol=2e-7)

    def test_flange_jacobian_ignores_active_tcp_offset(self):
        q = np.array([.2, -.7, 1.3, .4, .8, -.3])
        for robot in ("ur5e", "ur10e"):
            offset = pose_matrix([.12, -.03, .07, -.1, .4, -.2])
            actual, jacobian = NominalKinematics(robot, offset).pose_and_jacobian(q, flange=True)
            expected, flange_jacobian = NominalKinematics(robot).pose_and_jacobian(q)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_allclose(jacobian, flange_jacobian, atol=1e-12)

    def test_inverse_finds_nearby_reachable_pose_with_nonzero_tcp(self):
        seed = np.array([-.3, -1., 1.2, -.7, 1.1, .2])
        desired_q = seed + [.03, -.02, .025, -.015, .02, -.03]
        lower, upper = np.full(6, -np.pi), np.full(6, np.pi)
        offset = pose_matrix([.08, -.04, .12, .3, -.2, .4])
        for robot in ("ur5e", "ur10e"):
            with self.subTest(robot=robot):
                model = NominalKinematics(robot, offset)
                goal = dh_flange(robot, desired_q) @ offset
                solution = model.inverse(goal, seed, lower, upper)
                self.assertIsNotNone(solution)
                achieved = dh_flange(robot, solution) @ offset
                self.assertLess(np.linalg.norm(achieved[:3, 3] - goal[:3, 3]), .0005)
                self.assertLess(rotation_distance(achieved, goal), .002)
                self.assertTrue(np.all(solution >= lower) and np.all(solution <= upper))
                np.testing.assert_allclose(seed, [-.3, -1., 1.2, -.7, 1.1, .2])

    def test_unreachable_pose_returns_no_solution(self):
        seed = np.array([-.3, -1., 1.2, -.7, 1.1, .2])
        for robot in ("ur5e", "ur10e"):
            model = NominalKinematics(robot)
            goal = dh_flange(robot, seed)
            goal[0, 3] += 10.
            self.assertIsNone(model.inverse(goal, seed, np.full(6, -2*np.pi), np.full(6, 2*np.pi)))

    def test_inverse_rejects_goal_or_solution_outside_joint_bounds(self):
        seed = np.array([-.3, -1., 1.2, -.7, 1.1, .2])
        for robot in ("ur5e", "ur10e"):
            model = NominalKinematics(robot)
            exact = dh_flange(robot, seed)
            lower, upper = seed - .01, seed + .01
            upper[0] = seed[0] - .001
            self.assertIsNone(model.inverse(exact, seed, lower, upper))
            target_q = seed.copy()
            target_q[0] += .2
            self.assertIsNone(model.inverse(dh_flange(robot, target_q), seed, seed - .01, seed + .01))

    def test_capsule_points_include_real_forearm_bend_and_active_tcp(self):
        q = np.array([.2, -.7, 1.3, .4, .8, -.3])
        offset = pose_matrix([.08, -.04, .12, .3, -.2, .4])
        for robot, length in (("ur5e", .3922), ("ur10e", .57155)):
            model = NominalKinematics(robot, offset)
            points, links = model.points(q), model.links(q)
            self.assertEqual(points.shape, (8, 3))
            np.testing.assert_allclose(points[0], [0, 0, 0], atol=1e-12)
            np.testing.assert_allclose(points[3], (links["forearm_link"] @ [-length, 0, 0, 1])[:3], atol=1e-12)
            np.testing.assert_allclose(points[-1], (dh_flange(robot, q) @ offset)[:3, 3], atol=1e-12)
            self.assertGreater(np.linalg.norm(points[3] - points[4]), .1)

    def test_nonfinite_and_wrong_shape_joints_are_rejected(self):
        model = NominalKinematics("ur5e")
        for joints in ([0]*5, [0, 0, 0, 0, 0, np.nan], [[0]*6]):
            with self.assertRaises(ValueError):
                model.pose_and_jacobian(joints)


class SegmentDistanceTests(unittest.TestCase):
    def test_crossed_parallel_skew_disjoint_and_degenerate_segments(self):
        cases = [
            ([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], 0.),
            ([0, 0, 0], [2, 0, 0], [.5, 1, 0], [1.5, 1, 0], 1.),
            ([-1, 0, 0], [1, 0, 0], [0, -1, 2], [0, 1, 2], 2.),
            ([0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], np.sqrt(2)),
            ([0, 0, 0], [0, 0, 0], [0, 3, 4], [0, 3, 4], 5.),
            ([.5, 2, 0], [.5, 2, 0], [0, 0, 0], [1, 0, 0], 2.),
            ([2, 2, 0], [2, 2, 0], [0, 0, 0], [1, 0, 0], np.sqrt(5)),
            ([0, 0, 0], [1, 0, 0], [.5, 0, 0], [2, 0, 0], 0.),
        ]
        for a, b, c, d, expected in cases:
            for endpoints in ((a, b, c, d), (c, d, a, b), (b, a, d, c)):
                with self.subTest(endpoints=endpoints):
                    self.assertAlmostEqual(segment_distance(*endpoints), expected, places=12)

    def test_distance_is_invariant_to_common_camera_rotation_translation(self):
        change = pose_matrix([4, -2, 3, .7, -.3, 1.2])
        points = np.array([[0, 0, 0], [1, .2, .3], [0, .5, 1.], [1, -.1, .8]])
        changed = points @ change[:3, :3].T + change[:3, 3]
        self.assertAlmostEqual(segment_distance(*points), segment_distance(*changed), places=12)

    def test_nonfinite_or_wrong_shape_endpoints_fail_closed(self):
        for bad in ([np.nan, 0, 0], [0, np.inf, 0], [0, 0], [[0, 0, 0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                segment_distance(bad, [1, 0, 0], [0, 1, 0], [1, 1, 0])


class StaticModel:
    def __init__(self, points, tip=None):
        self._points = np.asarray(points, dtype=float)
        self.tip = pose_matrix([*self._points[-1], 0, 0, 0]) if tip is None else tip

    def points(self, joints):
        return self._points.copy()

    def pose_and_jacobian(self, joints):
        return self.tip.copy(), np.eye(6)


class WorkspaceGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_path = Path(self.directory.name) / "workspace.json"
        self.models = {name: StaticModel([[0, 0, 0], [1, 0, 0], [1.2, 0, 0]]) for name in ("ur5e", "ur10e")}
        self.joints = {name: np.zeros(6) for name in self.models}
        self.bases = {"ur5e": np.eye(4), "ur10e": pose_matrix([0, .5, 0, 0, 0, 0])}

    def guard(self, **config):
        self.config_path.write_text(json.dumps(config))
        return WorkspaceGuard(self.models, self.config_path)

    def test_interarm_capsule_clearance_includes_both_radii_and_margin(self):
        guard = self.guard(link_radius={"ur5e": .06, "ur10e": .08}, margin=.02)
        allowed, reason, clearance = guard.check(self.bases, self.joints)
        self.assertTrue(allowed)
        self.assertEqual(reason, "feasible")
        self.assertAlmostEqual(clearance, .5 - .06 - .08)
        self.bases["ur10e"][1, 3] = .15
        allowed, reason, clearance = guard.check(self.bases, self.joints)
        self.assertFalse(allowed)
        self.assertEqual(reason, "inter-arm capsule clearance")
        self.assertAlmostEqual(clearance, .15 - .06 - .08)

    def test_tool_capsule_radius_controls_tool_clearance(self):
        self.models = {"ur5e": StaticModel([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                       "ur10e": StaticModel([[0, 0, 0], [1, 0, 0], [2, 0, 0]])}
        bases = {"ur5e": np.eye(4), "ur10e": pose_matrix([3, .2, 0, 0, 0, np.pi])}
        guard = self.guard(link_radius={"ur5e": .01, "ur10e": .01},
                           tool_radius={"ur5e": .12, "ur10e": .12}, margin=.01)
        allowed, reason, clearance = guard.check(bases, self.joints)
        self.assertFalse(allowed)
        self.assertEqual(reason, "inter-arm capsule clearance")
        self.assertLess(clearance, 0.)
        guard = self.guard(link_radius={"ur5e": .01, "ur10e": .01},
                           tool_radius={"ur5e": .02, "ur10e": .02}, margin=.01)
        self.assertTrue(guard.check(bases, self.joints)[0])

    def test_common_camera_transform_preserves_collision_and_clearance(self):
        change = pose_matrix([1., -.7, 2., .8, -.2, .4])
        guard = self.guard()
        for separation in (.5, .15):
            self.bases["ur10e"][1, 3] = separation
            original = guard.check(self.bases, self.joints)
            changed = guard.check({name: change @ base for name, base in self.bases.items()}, self.joints)
            self.assertEqual(original[:2], changed[:2])
            self.assertAlmostEqual(original[2], changed[2], places=12)

    def test_camera_frame_obstacle_clearance_includes_obstacle_and_link_radius(self):
        guard = self.guard(obstacles=[{"center": [.5, -.15, 0], "radius": .08}], margin=.02)
        allowed, reason, clearance = guard.check(self.bases, self.joints)
        self.assertFalse(allowed)
        self.assertIn("obstacle", reason)
        self.assertAlmostEqual(clearance, .15 - .08 - .06)

    def test_obstacles_transform_with_camera_but_tcp_box_stays_in_robot_base(self):
        center = np.array([.5, -1., 0])
        config = {"tcp_boxes": {"ur5e": [[1., -.1, -.1], [1.3, .1, .1]]},
                  "obstacles": [{"center": center.tolist(), "radius": .08}]}
        guard = self.guard(**config)
        original = guard.check(self.bases, self.joints)
        change = pose_matrix([1., -.7, 2., .8, -.2, .4])
        config["obstacles"][0]["center"] = (change[:3, :3] @ center + change[:3, 3]).tolist()
        changed_guard = self.guard(**config)
        changed = changed_guard.check({name: change @ base for name, base in self.bases.items()}, self.joints)
        self.assertTrue(original[0])
        self.assertEqual(original[:2], changed[:2])
        self.assertAlmostEqual(original[2], changed[2], places=12)

    def test_joint_limits_apply_to_both_robots(self):
        guard = self.guard(joint_lower={name: [-.5]*6 for name in self.models},
                           joint_upper={name: [.5]*6 for name in self.models})
        self.assertTrue(guard.check(self.bases, self.joints)[0])
        for name in self.models:
            joints = {robot: values.copy() for robot, values in self.joints.items()}
            joints[name][4] = .6
            allowed, reason, clearance = guard.check(self.bases, joints)
            self.assertFalse(allowed)
            self.assertIn(name, reason)
            self.assertIn("joint", reason)
            self.assertIsNone(clearance)

    def test_tcp_workspace_box_rejects_current_and_candidate_pose(self):
        guard = self.guard(tcp_boxes={"ur5e": [[0, -.1, -.1], [1.1, .1, .1]]})
        allowed, reason, clearance = guard.check(self.bases, self.joints)
        self.assertFalse(allowed)
        self.assertIn("TCP workspace box", reason)
        self.assertIsNone(clearance)
        self.assertTrue(guard.pose_allowed("ur5e", pose_matrix([.5, 0, 0, 0, 0, 0])))
        self.assertFalse(guard.pose_allowed("ur5e", pose_matrix([.5, .2, 0, 0, 0, 0])))

    def test_invalid_base_transform_fails_closed_before_collision_math(self):
        guard = self.guard()
        for bad in (np.full((4, 4), np.nan), np.diag([2., 1., 1., 1.]), np.zeros((3, 3))):
            bases = dict(self.bases, ur10e=bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                guard.check(bases, self.joints)

    def test_invalid_workspace_configuration_is_rejected(self):
        cases = [{"unknown": True}, {"margin": -.01}, {"tool_radius": {"ur5e": 0.}},
                 {"joint_lower": {"ur5e": [1.]*6}, "joint_upper": {"ur5e": [0.]*6}},
                 {"tcp_boxes": {"ur5e": [[1, 0, 0], [0, 1, 1]]}},
                 {"obstacles": [{"center": [0, 0, 0], "radius": -1.}]}]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.guard(**config)


if __name__ == "__main__":
    unittest.main()
