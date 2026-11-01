"""Shared-plane estimates reach rendering/control without changing detections or time."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose, model_from_controller_base, pose_matrix
from urcal.control.runtime import parse_args
from urcal.control.vision import calibration_frame
from urcal.inference import infer_live_dual as dual


class CommonBasePlanePipelineTests(unittest.TestCase):
    def setUp(self):
        self.names = ("ur5e", "ur10e")
        self.K = np.array([[600., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        base = pose_matrix([-.4, .2, 2.5, 1.2, .1, .2])
        self.bases = {"ur5e": base, "ur10e": base @ pose_matrix([.8, .1, 0., 0., 0., 1.1])}
        X = np.array([[0., 0., .1], [.1, .05, .2], [.12, .3, .25], [-.1, .2, .4],
                      [.25, .1, .3], [.05, -.1, .3], [.12, -.04, .5]])
        self.robots, self.results = [], []
        self.joints = [[0.] * 6, [.1] * 6]
        for index, name in enumerate(self.names):
            M = model_from_controller_base(name)
            model_points = X @ M[:3, :3].T + M[:3, 3]
            truth = self.bases[name] @ inverse(M)
            pose = matrix_pose(truth)
            points = dual.cv2.projectPoints(model_points, pose[3:], pose[:3], self.K, self.dist)[0].reshape(-1, 2)
            seed_base = self.bases[name] @ pose_matrix([.01, -.01, .025 * (index + 1), .025, -.015, 0.])
            seed = matrix_pose(seed_base @ inverse(M))
            valid = np.ones(7, dtype=bool)
            valid[2], points[2] = False, np.nan
            result = dual.RobotResult(points, valid, None, True, seed[3:], seed[:3], 0.)
            self.results.append(result)
            self.robots.append(SimpleNamespace(
                name=name, key_names=[f"{name}_link{i}" for i in range(7)],
                infer=Mock(return_value=result), fk=Mock(return_value=model_points),
                prev_rvec=np.zeros(3), prev_tvec=np.ones(3)))

    def process(self, enabled=True):
        return dual.process_frame(np.zeros((480, 640, 3), np.uint8), self.joints, self.robots,
                                  self.K, self.dist, .5, None, None, None, common_base_plane=enabled)

    def test_joint_fit_uses_original_votes_and_publishes_one_consistent_pair(self):
        original_points = [result.points.copy() for result in self.results]
        regions, results = self.process()
        packet = calibration_frame(17, 123., results, self.robots, self.joints,
                                   self.K, self.dist, .1, .02, regions=regions)
        self.assertEqual(packet.sequence, 17)
        self.assertEqual(packet.captured_at, 123.)
        self.assertFalse(packet.reason)
        self.assertTrue(packet.pose_diagnostics["common_base_plane"]["applied"])
        for name, robot, before, result, points in zip(self.names, self.robots, self.results, results, original_points):
            np.testing.assert_array_equal(result.points, points)
            np.testing.assert_array_equal(result.valid, before.valid)
            np.testing.assert_array_equal(result.empnp_rvec, before.rvec)
            np.testing.assert_array_equal(result.empnp_tvec, before.tvec)
            np.testing.assert_allclose(packet.poses[name] @ model_from_controller_base(name), self.bases[name], atol=1e-6)
            np.testing.assert_array_equal(robot.prev_rvec, result.rvec)
            np.testing.assert_array_equal(robot.prev_tvec, result.tvec)
            self.assertLess(packet.raw_mean_px[name], 1e-5)
            diagnostic = packet.pose_diagnostics["robots"][name]
            np.testing.assert_array_equal(diagnostic["empnp_rvec"], before.rvec)
            np.testing.assert_array_equal(diagnostic["output_rvec"], result.rvec)

    def test_disable_preserves_independent_output(self):
        with patch("urcal.inference.coplanar.refine_coplanar_pair") as fit:
            _, results = self.process(False)
        fit.assert_not_called()
        self.assertIs(results[0], self.results[0])
        self.assertIsNone(results[0].base_plane)

    def test_missing_robot_does_not_fabricate_a_pose_or_joint_fit(self):
        self.robots[1].infer.return_value = replace(self.results[1], ok=False, rvec=None, tvec=None)
        with patch("urcal.inference.coplanar.refine_coplanar_pair") as fit:
            _, results = self.process()
        fit.assert_not_called()
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)
        self.assertFalse(results[0].base_plane["applied"])
        packet = calibration_frame(18, 124., results, self.robots, self.joints, self.K, self.dist, .1, .02)
        self.assertIn("ur10e", packet.reason)
        self.assertNotIn("ur10e", packet.poses)

    def test_invalid_constrained_pair_does_not_publish_success_or_poison_previous_pose(self):
        priors = [(robot.prev_rvec.copy(), robot.prev_tvec.copy()) for robot in self.robots]
        with patch("urcal.inference.coplanar.refine_coplanar_pair", side_effect=ValueError("invalid projected depth")):
            _, results = self.process()
        self.assertTrue(all(not result.ok for result in results))
        for robot, (rvec, tvec) in zip(self.robots, priors):
            np.testing.assert_array_equal(robot.prev_rvec, rvec)
            np.testing.assert_array_equal(robot.prev_tvec, tvec)
        packet = calibration_frame(19, 125., results, self.robots, self.joints, self.K, self.dist, .1, .02)
        self.assertEqual(packet.poses, {})
        self.assertIn("common base plane: invalid projected depth", packet.reason)

    def test_both_live_launchers_default_on_and_allow_explicit_off(self):
        for parser, sources in ((parse_args, ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]),
                                (dual.parse_args, ["--ur5e_joints", *(["0"] * 6),
                                                   "--ur10e_joints", *(["0"] * 6)])):
            with self.subTest(parser=parser.__module__):
                self.assertTrue(parser(sources).common_base_plane)
                self.assertFalse(parser(sources + ["--no_common_base_plane"]).common_base_plane)
                self.assertTrue(parser(sources + ["--common_base_plane"]).common_base_plane)


if __name__ == "__main__":
    unittest.main()
