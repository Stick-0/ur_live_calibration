"""Geometry batching keeps every scalar capsule and workspace decision."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.control.geometry import inverse, pose_matrix
from urcal.cooperative.kinematics import (
    NominalKinematics, WorkspaceGuard, _segment_distances, segment_distance,
)


def scalar_guard_check(guard, bases, joints):
    """Original scalar algorithm, independent of the optimized guard path."""
    segments, minimum = {}, float("inf")
    for name, model in guard.models.items():
        inverse(bases[name])
        q = guard._vector(joints[name], 6)
        if np.any(q < guard.lower[name]) or np.any(q > guard.upper[name]):
            return False, f"{name}: joint position limit", None
        tip, _ = model.pose_and_jacobian(q)
        if not guard.pose_allowed(name, tip):
            return False, f"{name}: TCP workspace box", None
        points = model.points(q) @ bases[name][:3, :3].T + bases[name][:3, 3]
        segments[name] = [(a, b, guard.tool_radii[name] if i == len(points)-2 else guard.radii[name])
                          for i, (a, b) in enumerate(zip(points[:-1], points[1:]))]
        for a, b, radius in segments[name]:
            for center, obstacle_radius in guard.obstacles:
                clearance = segment_distance(a, b, center, center) - radius - obstacle_radius
                minimum = min(minimum, clearance)
                if clearance < guard.margin:
                    return False, f"{name}: obstacle clearance", clearance
    for i, name in enumerate(segments):
        for other in list(segments)[i+1:]:
            for a, b, ra in segments[name]:
                for c, d, rb in segments[other]:
                    clearance = segment_distance(a, b, c, d) - ra - rb
                    minimum = min(minimum, clearance)
                    if clearance < guard.margin:
                        return False, "inter-arm capsule clearance", clearance
    return True, "feasible", None if np.isinf(minimum) else minimum


class BatchedSegmentTests(unittest.TestCase):
    def test_random_pairs_match_scalar_reference_with_broadcasting(self):
        rng = np.random.default_rng(647)
        a, b = rng.normal(size=(2, 23, 1, 3))
        c, d = rng.normal(size=(2, 1, 31, 3))
        actual = _segment_distances(a, b, c, d)
        expected = [[segment_distance(a[i, 0], b[i, 0], c[0, j], d[0, j])
                     for j in range(31)] for i in range(23)]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_points_parallel_overlapping_and_tiny_segments_match_scalar(self):
        cases = np.array([
            [[0, 0, 0], [0, 0, 0], [2, 3, 4], [2, 3, 4]],
            [[2, 2, 0], [2, 2, 0], [0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [.5, 2, 0], [.5, 2, 0]],
            [[0, 0, 0], [1, 0, 0], [.5, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]],
            [[0, 0, 0], [1e-10, 0, 0], [.5, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [.5, 1, 0], [.5+1e-10, 1, 0]],
            [[0, 0, 0], [1e-10, 0, 0], [0, 1, 0], [1e-10, 1, 0]],
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [.5, 1, 0], [2, 1+1e-10, 0]],
        ])
        for order in ((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2)):
            endpoints = cases[:, order]
            actual = _segment_distances(*np.moveaxis(endpoints, 1, 0))
            expected = [segment_distance(*pair) for pair in endpoints]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_endpoints_fail_closed(self):
        for bad in ([np.nan, 0, 0], [[0, np.inf, 0]], [1, 2], 1.):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _segment_distances(bad, [1, 0, 0], [0, 1, 0], [1, 1, 0])


class BatchedWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.models = {name: NominalKinematics(name, pose_matrix([.03, -.02, .08, .1, -.2, .3]))
                       for name in ("ur5e", "ur10e")}

    def assert_result_equal(self, actual, expected):
        self.assertEqual(actual[:2], expected[:2])
        if expected[2] is None:
            self.assertIsNone(actual[2])
        else:
            self.assertAlmostEqual(actual[2], expected[2], places=12)

    def test_combined_geometry_matches_tcp_and_original_forearm_vertices(self):
        rng = np.random.default_rng(91)
        for name, model in self.models.items():
            for _ in range(12):
                q = rng.uniform(-np.pi, np.pi, size=6)
                links = model.links(q)
                elbow = links["forearm_link"] @ pose_matrix([-model.forearm_length, 0, 0, 0, 0, 0])
                tip, points = model.pose_and_points(q)
                np.testing.assert_allclose(tip, model.pose_and_jacobian(q)[0], atol=1e-14)
                np.testing.assert_allclose(points, model.points(q), atol=1e-14)
                np.testing.assert_allclose(points[3], elbow[:3, 3], atol=1e-14)
            with patch.object(model, "fk", wraps=model.fk) as fk:
                model.pose_and_points(q)
                self.assertEqual(fk.call_count, 1)

    def test_random_real_robot_scenes_preserve_scalar_result_and_first_violation(self):
        rng = np.random.default_rng(812)
        configurations = [{}, {"obstacles": [
            {"center": [.2, -.1, .3], "radius": .2},
            {"center": [-.2, .5, .5], "radius": .1}]},
            {"tcp_boxes": {"ur5e": [[-1, -1, -.2], [1, 1, 1]]}}]
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "workspace.json"
            for config in configurations:
                config_path.write_text(json.dumps(config))
                guard = WorkspaceGuard(self.models, config_path)
                for _ in range(70):
                    joints = {name: rng.uniform(-np.pi, np.pi, size=6) for name in self.models}
                    bases = {"ur5e": np.eye(4), "ur10e": pose_matrix(
                        [*rng.uniform(-1, 1, size=3), *rng.uniform(-1, 1, size=3)])}
                    self.assert_result_equal(guard.check(bases, joints),
                                             scalar_guard_check(guard, bases, joints))

    def test_each_robot_fk_runs_once_per_guard_check(self):
        guard = WorkspaceGuard(self.models)
        bases = {"ur5e": np.eye(4), "ur10e": pose_matrix([3, 0, 0, 0, 0, 0])}
        joints = {name: np.zeros(6) for name in self.models}
        with patch.object(self.models["ur5e"], "fk", wraps=self.models["ur5e"].fk) as fk5, \
                patch.object(self.models["ur10e"], "fk", wraps=self.models["ur10e"].fk) as fk10:
            self.assertTrue(guard.check(bases, joints)[0])
            self.assertEqual((fk5.call_count, fk10.call_count), (1, 1))

    def test_captured_and_simulated_models_retain_their_corrected_tool_geometry(self):
        from urcal.control.alignment import CapturedKinematics
        from urcal.simulation.plant import SeededKinematics
        q = np.array([.3, -1.1, 1.4, -.7, .5, -.2])
        offset = pose_matrix([0, 0, .1, 0, 0, 0])
        nominal, _ = NominalKinematics("ur5e", offset).pose_and_jacobian(q)
        captured = nominal @ pose_matrix([.04, -.02, .01, .1, -.1, .1])
        for model_type in (CapturedKinematics, SeededKinematics):
            model = model_type("ur5e", q, captured, offset)
            actual_pose, actual_points = model.pose_and_points(q)
            np.testing.assert_allclose(actual_pose, captured, atol=1e-14)
            np.testing.assert_allclose(actual_points, model.points(q), atol=1e-14)
            self.assertEqual(len(actual_points), 9)
            guard = WorkspaceGuard({"ur5e": model})
            bases, joints = {"ur5e": np.eye(4)}, {"ur5e": q}
            guard.boxes["ur5e"] = (captured[:3, 3] - .001, captured[:3, 3] + .001)
            self.assert_result_equal(guard.check(bases, joints), scalar_guard_check(guard, bases, joints))
            self.assertTrue(guard.check(bases, joints)[0])


if __name__ == "__main__":
    unittest.main()
