"""Frame conventions and flange goals, without camera or robot connections."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
import urcal.control.geometry as geometry


def nominal_dh_flange(robot, q):
    """Independent standard-DH controller-base FK (not the dataset link tree)."""
    if robot == "ur5e":
        a = [0, -0.425, -0.3922, 0, 0, 0]
        d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    else:
        a = [0, -0.6127, -0.57155, 0, 0, 0]
        d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
    alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    result = np.eye(4)
    for angle, length, offset, twist in zip(q, a, d, alpha):
        c, s = np.cos(angle), np.sin(angle)
        ca, sa = np.cos(twist), np.sin(twist)
        result = result @ np.array([
            [c, -s * ca, s * sa, length * c],
            [s, c * ca, -c * sa, length * s],
            [0, sa, ca, offset],
            [0, 0, 0, 1],
        ])
    return result


class DualControlGeometryTests(unittest.TestCase):
    def setUp(self):
        self.first = geometry.pose_matrix([0.2, -0.3, 0.4, 0.2, 0.3, -0.1])
        self.second = geometry.pose_matrix([0.8, 0.1, 0.7, -0.3, 0.6, 0.5])
        self.change = geometry.pose_matrix([-0.5, 0.2, 0.7, -1.0, 0.2, 0.6])

    def assert_facing(self, first, second, gap):
        direction = (second[:3, 3] - first[:3, 3]) / gap
        np.testing.assert_allclose(first[:3, 2], direction, atol=1e-12)
        np.testing.assert_allclose(second[:3, 2], -direction, atol=1e-12)
        np.testing.assert_allclose(first[:3, 0], second[:3, 0], atol=1e-12)
        np.testing.assert_allclose(first[:3, 1], -second[:3, 1], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(second[:3, :3]), 1.0)

    def test_pose_roundtrip_near_pi_and_noncanonical_rotvec(self):
        for rotvec in ([0, 0, 0], [np.pi, 0, 0], [0, 2 * np.pi + 0.2, 0], [0.3, -0.5, 2.8]):
            matrix = geometry.pose_matrix([0.2, 0.3, -0.7, *rotvec])
            np.testing.assert_allclose(geometry.pose_matrix(geometry.matrix_pose(matrix)), matrix, atol=1e-12)

    def test_inverse_and_tcp_offset_roundtrip_include_rotated_translation(self):
        offset = geometry.pose_matrix([0.08, -0.02, 0.12, 0.2, -0.3, 0.4])
        tcp = self.first @ offset
        np.testing.assert_allclose(geometry.flange_from_tcp(tcp, offset), self.first, atol=1e-12)
        np.testing.assert_allclose(self.first @ geometry.inverse(self.first), np.eye(4), atol=1e-12)

    def test_rotation_distance_ignores_translation_and_uses_shortest_angle(self):
        translated = self.first.copy()
        translated[:3, 3] += 10
        self.assertAlmostEqual(geometry.rotation_distance(self.first, translated), 0.0)
        self.assertAlmostEqual(geometry.rotation_distance(np.eye(3), Rotation.from_rotvec([0, 0, 1.8 * np.pi]).as_matrix()), 0.2 * np.pi)
        self.assertAlmostEqual(geometry.rotation_distance(self.first, self.second), geometry.rotation_distance(self.second, self.first))

    def test_follower_normal_and_roll(self):
        follower = geometry.facing_follower(self.first, 0.25)
        self.assert_facing(self.first, follower, 0.25)
        np.testing.assert_allclose(geometry.facing_follower(follower, 0.25), self.first, atol=1e-12)

    def test_follower_is_rigidly_equivariant(self):
        np.testing.assert_allclose(
            geometry.facing_follower(self.change @ self.first, 0.23),
            self.change @ geometry.facing_follower(self.first, 0.23), atol=1e-12,
        )

    def test_fixed_defaults_keep_positions_and_choose_opposite_normals(self):
        targets = geometry.fixed_facing_targets(self.first, self.second)
        gap = np.linalg.norm(self.second[:3, 3] - self.first[:3, 3])
        self.assert_facing(targets["ur5e"], targets["ur10e"], gap)
        np.testing.assert_allclose(targets["ur5e"][:3, 3], self.first[:3, 3], atol=1e-12)
        np.testing.assert_allclose(targets["ur10e"][:3, 3], self.second[:3, 3], atol=1e-12)
        normal = targets["ur5e"][:3, 2]
        expected_x = self.first[:3, 0] - normal * np.dot(normal, self.first[:3, 0])
        np.testing.assert_allclose(targets["ur5e"][:3, 0], expected_x / np.linalg.norm(expected_x), atol=1e-12)

    def test_fixed_custom_gap_preserves_midpoint(self):
        targets = geometry.fixed_facing_targets(self.first, self.second, gap=0.31)
        self.assert_facing(targets["ur5e"], targets["ur10e"], 0.31)
        np.testing.assert_allclose(targets["ur5e"][:3, 3] + targets["ur10e"][:3, 3], self.first[:3, 3] + self.second[:3, 3], atol=1e-12)

    def test_fixed_goals_are_rigidly_equivariant_including_roll_fallback(self):
        for second in (self.second, self.first.copy()):
            if np.array_equal(second, self.first):
                second[:3, 3] += 0.3 * self.first[:3, 0]
            original = geometry.fixed_facing_targets(self.first, second, gap=0.2)
            changed = geometry.fixed_facing_targets(self.change @ self.first, self.change @ second, gap=0.2)
            for robot in original:
                np.testing.assert_allclose(changed[robot], self.change @ original[robot], atol=1e-12)
            self.assert_facing(original["ur5e"], original["ur10e"], 0.2)

    def test_explicit_center_pose_orientation_and_gap(self):
        for center in (self.change, geometry.matrix_pose(self.change)):
            targets = geometry.fixed_facing_targets(self.first, self.second, gap=0.4, center_pose=center)
            self.assert_facing(targets["ur5e"], targets["ur10e"], 0.4)
            np.testing.assert_allclose(targets["ur5e"][:3, :3], self.change[:3, :3], atol=1e-12)
            np.testing.assert_allclose(0.5 * (targets["ur5e"][:3, 3] + targets["ur10e"][:3, 3]), self.change[:3, 3], atol=1e-12)

    def test_coincident_positions_require_both_gap_and_center(self):
        for kwargs in ({}, {"gap": 0.2}, {"center_pose": self.change}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                geometry.fixed_facing_targets(self.first, self.first, **kwargs)
        targets = geometry.fixed_facing_targets(self.first, self.first, gap=0.2, center_pose=self.change)
        self.assert_facing(targets["ur5e"], targets["ur10e"], 0.2)

    def test_pose_delta_is_expressed_in_shared_parent_frame(self):
        linear, angular = geometry.pose_delta(self.first, self.second)
        np.testing.assert_allclose(self.first[:3, 3] + linear, self.second[:3, 3], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_rotvec(angular).as_matrix() @ self.first[:3, :3], self.second[:3, :3], atol=1e-12)
        changed_linear, changed_angular = geometry.pose_delta(self.change @ self.first, self.change @ self.second)
        np.testing.assert_allclose(changed_linear, self.change[:3, :3] @ linear, atol=1e-12)
        np.testing.assert_allclose(changed_angular, self.change[:3, :3] @ angular, atol=1e-12)

    def test_nominal_mapping_matches_controller_dh_across_configurations(self):
        random = np.random.default_rng(31)
        for robot in ("ur5e", "ur10e"):
            mapping = geometry.model_from_controller_base(robot)
            for q in [np.zeros(6), *random.uniform(-np.pi, np.pi, (12, 6))]:
                with self.subTest(robot=robot, joints=q):
                    model = geometry.model_flange_pose(robot, q)
                    np.testing.assert_allclose(geometry.inverse(mapping) @ model, nominal_dh_flange(robot, q), atol=1e-12)
        np.testing.assert_array_equal(geometry.model_from_controller_base("ur5e"), np.eye(4))
        np.testing.assert_array_equal(geometry.model_from_controller_base("ur10e")[:3, 3], [0, 0, 0.005])

    def test_bad_pose_joint_and_model_inputs_are_rejected(self):
        for invalid in ([0] * 5, [[0] * 6], [0, 0, 0, np.nan, 0, 0], [np.inf] * 6, "bad"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                geometry.pose_matrix(invalid)
            with self.subTest(joints=invalid), self.assertRaises(ValueError):
                geometry.model_flange_pose("ur5e", invalid)
        with self.assertRaises(ValueError):
            geometry.model_from_controller_base("ur3e")
        with self.assertRaises(ValueError):
            geometry.model_flange_pose("ur3e", np.zeros(6))

    def test_malformed_se3_is_not_silently_projected_to_a_rotation(self):
        reflection = np.eye(4)
        reflection[0, 0] = -1
        scaling = np.eye(4)
        scaling[1, 1] = 1.02
        bad_last_row = np.eye(4)
        bad_last_row[3, 0] = 0.1
        nonfinite = np.eye(4)
        nonfinite[2, 3] = np.nan
        for invalid in (reflection, scaling, bad_last_row, nonfinite, np.eye(3)):
            operations = (
                geometry.matrix_pose, geometry.inverse,
                lambda value: geometry.facing_follower(value, 0.2),
                lambda value: geometry.flange_from_tcp(value, self.first),
                lambda value: geometry.flange_from_tcp(self.first, value),
                lambda value: geometry.pose_delta(self.first, value),
                lambda value: geometry.fixed_facing_targets(value, self.second),
            )
            for operation in operations:
                with self.subTest(value=invalid, operation=operation), self.assertRaises(ValueError):
                    operation(invalid)
        for invalid in (reflection, scaling, bad_last_row, nonfinite):
            with self.assertRaises(ValueError):
                geometry.rotation_distance(self.first, invalid)

    def test_bad_gaps_and_center_poses_are_rejected(self):
        for gap in (0, -0.01, np.nan, np.inf, "invalid"):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                geometry.facing_follower(self.first, gap)
            with self.subTest(fixed_gap=gap), self.assertRaises(ValueError):
                geometry.fixed_facing_targets(self.first, self.second, gap=gap)
        for center in ([1, 2], np.zeros((4, 4)), [0, 0, 0, np.inf, 0, 0]):
            with self.subTest(center=center), self.assertRaises(ValueError):
                geometry.fixed_facing_targets(self.first, self.second, gap=0.2, center_pose=center)


if __name__ == "__main__":
    unittest.main()
