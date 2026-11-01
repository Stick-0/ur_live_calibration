"""Common mounting plane and joint pixel refinement without hardware/weights."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.control.geometry import inverse, model_from_controller_base, pose_matrix, rotation_distance
from urcal.inference.coplanar import project_coplanar_bases, refine_coplanar_pair


ROBOTS = ("ur5e", "ur10e")


def bases():
    first = pose_matrix([-.4, -.1, 2.1, .35, -.2, .1])
    second = first @ pose_matrix([.93, .18, 0., 0., 0., 2.6])
    return {"ur5e": first, "ur10e": second}


def observations(noise=0.):
    truth = bases()
    mappings = {robot: model_from_controller_base(robot) for robot in ROBOTS}
    poses = {robot: truth[robot] @ inverse(mappings[robot]) for robot in ROBOTS}
    model = np.array([[0., 0., 0.], [0., 0., .16], [-.24, .01, .48],
                      [-.44, -.04, .59], [-.33, .08, .75], [-.22, .15, .67],
                      [-.18, .05, .79]])
    points = {robot: model.copy() for robot in ROBOTS}
    points["ur10e"] *= 1.2
    K = np.array([[610., 0., 320.], [0., 615., 240.], [0., 0., 1.]])
    dist = np.array([.01, -.005, .0002, -.0001, .0003])
    random = np.random.default_rng(17)
    images = {robot: cv2.projectPoints(points[robot],
                                      Rotation.from_matrix(poses[robot][:3, :3]).as_rotvec(),
                                      poses[robot][:3, 3], K, dist)[0].reshape(-1, 2)
              + random.normal(0., noise, (len(points[robot]), 2)) for robot in ROBOTS}
    return truth, poses, points, images, K, dist, mappings


class CoplanarProjectionTests(unittest.TestCase):
    def assert_coplanar(self, value, atol=2e-12):
        first, second = (value[robot] for robot in ROBOTS)
        np.testing.assert_allclose(first[:3, 2], second[:3, 2], atol=atol)
        self.assertLess(abs(np.dot(first[:3, 2], second[:3, 3] - first[:3, 3])), atol)
        for pose in value.values():
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=atol)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1., places=12)

    def test_planar_pair_remains_unchanged_with_independent_yaw_and_spacing(self):
        original = bases()
        result = project_coplanar_bases(original)
        self.assert_coplanar(result)
        for robot in ROBOTS:
            np.testing.assert_allclose(result[robot], original[robot], atol=1e-15)
            self.assertFalse(np.shares_memory(result[robot], original[robot]))
        relative = inverse(result["ur5e"]) @ result["ur10e"]
        np.testing.assert_allclose(relative[:3, 3], [.93, .18, 0.], atol=1e-15)
        self.assertAlmostEqual(Rotation.from_matrix(relative[:3, :3]).as_rotvec()[2], 2.6)

    def test_tilt_and_height_use_symmetric_midpoint_plane_without_input_changes(self):
        original = bases()
        original["ur5e"] = original["ur5e"] @ pose_matrix([.01, .02, .04, .1, -.08, 0.])
        original["ur10e"] = original["ur10e"] @ pose_matrix([.01, .02, -.08, -.05, .02, 0.])
        saved = copy.deepcopy(original)
        normal = original["ur5e"][:3, 2] + original["ur10e"][:3, 2]
        normal /= np.linalg.norm(normal)
        midpoint = (original["ur5e"][:3, 3] + original["ur10e"][:3, 3]) / 2
        result = project_coplanar_bases(original)
        self.assert_coplanar(result)
        np.testing.assert_allclose(result["ur5e"][:3, 2], normal, atol=2e-15)
        np.testing.assert_allclose((result["ur5e"][:3, 3] + result["ur10e"][:3, 3]) / 2,
                                   midpoint, atol=1e-15)
        for robot in ROBOTS:
            delta = result[robot][:3, 3] - original[robot][:3, 3]
            np.testing.assert_allclose(np.cross(delta, normal), 0., atol=1e-15)
            np.testing.assert_array_equal(original[robot], saved[robot])

    def test_arbitrary_camera_transform_commutes_with_geometric_projection(self):
        original = bases()
        original["ur10e"] = original["ur10e"] @ pose_matrix([.02, .01, .08, .11, -.04, .01])
        camera = pose_matrix([.3, -.8, .5, -.8, .3, .5])
        plain = project_coplanar_bases(original)
        transformed = project_coplanar_bases({robot: camera @ original[robot] for robot in ROBOTS})
        for robot in ROBOTS:
            np.testing.assert_allclose(transformed[robot], camera @ plain[robot], atol=2e-15)

    def test_exchanging_robot_names_keeps_projection_symmetric(self):
        original = bases()
        original["ur10e"] = original["ur10e"] @ pose_matrix([0., 0., .08, .1, .2, 0.])
        normal = project_coplanar_bases(original)
        swapped = project_coplanar_bases({"ur5e": original["ur10e"], "ur10e": original["ur5e"]})
        np.testing.assert_allclose(normal["ur5e"], swapped["ur10e"], atol=1e-15)
        np.testing.assert_allclose(normal["ur10e"], swapped["ur5e"], atol=1e-15)

    def test_antiparallel_normals_rejected_instead_of_inventing_plane(self):
        original = bases()
        original["ur10e"] = original["ur5e"] @ pose_matrix([1., 0., 0., np.pi, 0., 0.])
        with self.assertRaisesRegex(ValueError, "opposing"):
            project_coplanar_bases(original)

    def test_non_se3_or_missing_inputs_rejected(self):
        for invalid in (None, {}, {"ur5e": np.eye(4)}):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                project_coplanar_bases(invalid)
        for bad in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([-1., 1., 1., 1.])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                project_coplanar_bases({"ur5e": bad, "ur10e": np.eye(4)})


class CoplanarRefinementTests(unittest.TestCase):
    assert_coplanar = CoplanarProjectionTests.assert_coplanar
    def test_joint_noisy_refinement_recovers_common_tilt_height_and_free_yaw(self):
        truth, poses, points, images, K, dist, mappings = observations(.15)
        raw = copy.deepcopy(poses)
        raw["ur5e"] = raw["ur5e"] @ pose_matrix([.02, -.015, .035, .025, -.035, .02])
        raw["ur10e"] = raw["ur10e"] @ pose_matrix([-.015, .02, -.04, -.03, .025, -.015])
        saved = copy.deepcopy((raw, points, images, K, dist, mappings))
        result, info = refine_coplanar_pair(raw, points, images, K, dist, mappings)
        final_bases = {robot: result[robot] @ mappings[robot] for robot in ROBOTS}
        self.assert_coplanar(final_bases)
        self.assertTrue(info["applied"])
        self.assertFalse(info["fallback"])
        self.assertLess(info["final_plane"]["height_difference_m"], 1e-12)
        self.assertLess(info["final_plane"]["normal_angle_deg"], 1e-10)
        self.assertGreater(info["raw_plane"]["height_difference_m"], .01)
        for robot in ROBOTS:
            self.assertLess(np.linalg.norm(final_bases[robot][:3, 3] - truth[robot][:3, 3]), .008)
            self.assertLess(rotation_distance(final_bases[robot], truth[robot]), .015)
            self.assertLess(info["reprojection"][robot]["after_mean_px"], .4)
            self.assertLess(info["reprojection"][robot]["after_mean_px"],
                            info["reprojection"][robot]["before_mean_px"])
            for current, previous in zip((raw, points, images, mappings), (saved[0], saved[1], saved[2], saved[5])):
                np.testing.assert_array_equal(current[robot], previous[robot])
        np.testing.assert_array_equal(K, saved[3])
        np.testing.assert_array_equal(dist, saved[4])

    def test_exact_planar_truth_and_model_height_mapping_remain_unchanged(self):
        truth, poses, points, images, K, dist, mappings = observations()
        result, info = refine_coplanar_pair(poses, points, images, K, dist, mappings)
        for robot in ROBOTS:
            np.testing.assert_allclose(result[robot], poses[robot], atol=2e-13)
            np.testing.assert_allclose(result[robot] @ mappings[robot], truth[robot], atol=2e-13)
        # Raw model origins intentionally differ by 5 mm along the true normal;
        # the constraint belongs to physical controller bases, not model origins.
        raw_height = np.dot(truth["ur5e"][:3, 2], result["ur10e"][:3, 3] - result["ur5e"][:3, 3])
        self.assertAlmostEqual(raw_height, -.005, places=12)
        self.assertLess(info["correction"]["ur10e"]["translation_m"], 1e-12)

    def test_solver_exception_returns_constrained_seed_and_reports_fallback(self):
        _, poses, points, images, K, dist, mappings = observations()
        poses["ur10e"] = poses["ur10e"] @ pose_matrix([0., 0., .06, .1, 0., 0.])
        expected = project_coplanar_bases({robot: poses[robot] @ mappings[robot] for robot in ROBOTS})
        with patch("urcal.inference.coplanar.least_squares", side_effect=ValueError("synthetic solver failure")):
            result, info = refine_coplanar_pair(poses, points, images, K, dist, mappings)
        self.assertTrue(info["fallback"])
        self.assertFalse(info["solver_success"])
        self.assertIn("synthetic solver failure", info["solver_message"])
        for robot in ROBOTS:
            np.testing.assert_allclose(result[robot] @ mappings[robot], expected[robot], atol=1e-15)

    def test_one_evaluation_budget_still_returns_exact_common_plane(self):
        _, poses, points, images, K, dist, mappings = observations()
        poses["ur10e"] = poses["ur10e"] @ pose_matrix([0., 0., .06, .1, 0., 0.])
        result, info = refine_coplanar_pair(poses, points, images, K, dist, mappings, max_nfev=1)
        self.assert_coplanar({robot: result[robot] @ mappings[robot] for robot in ROBOTS})
        self.assertEqual(info["nfev"], 1)

    def test_unusable_solver_result_cannot_escape_to_unconstrained_raw_pose(self):
        _, poses, points, images, K, dist, mappings = observations()
        poses["ur10e"] = poses["ur10e"] @ pose_matrix([0., 0., .06, .1, 0., 0.])
        for translation in ([0., 0., -3.], [4., 4., 3.], [float("nan"), 0., 3.]):
            candidate = SimpleNamespace(success=True, status=1, message="synthetic candidate", nfev=2,
                                        x=np.array([*translation, 0., 0., 0., .9, 0., 0.]))
            with self.subTest(translation=translation), patch(
                    "urcal.inference.coplanar.least_squares", return_value=candidate):
                result, info = refine_coplanar_pair(poses, points, images, K, dist, mappings)
            self.assertTrue(info["fallback"])
            self.assert_coplanar({robot: result[robot] @ mappings[robot] for robot in ROBOTS})

    def test_four_original_correspondences_per_robot_are_supported(self):
        truth, poses, points, images, K, dist, mappings = observations()
        points = {robot: points[robot][[0, 2, 4, 6]] for robot in ROBOTS}
        images = {robot: images[robot][[0, 2, 4, 6]] for robot in ROBOTS}
        result, _ = refine_coplanar_pair(poses, points, images, K, dist, mappings)
        for robot in ROBOTS:
            np.testing.assert_allclose(result[robot] @ mappings[robot], truth[robot], atol=2e-13)

    def test_invalid_correspondences_and_camera_are_rejected(self):
        _, poses, points, images, K, dist, mappings = observations()
        cases = [dict(points_model={"ur5e": points["ur5e"][:3], "ur10e": points["ur10e"]}),
                 dict(points_image={"ur5e": images["ur5e"][:-1], "ur10e": images["ur10e"]}),
                 dict(points_image={"ur5e": np.full_like(images["ur5e"], np.nan), "ur10e": images["ur10e"]}),
                 dict(K=np.eye(4)), dict(dist=[1., 2.]), dict(max_nfev=0), dict(max_nfev=True)]
        kwargs = dict(poses_model=poses, points_model=points, points_image=images, K=K,
                      dist=dist, mappings=mappings)
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                refine_coplanar_pair(**(kwargs | changes))

    def test_input_keypoints_behind_camera_are_rejected(self):
        _, poses, points, images, K, dist, mappings = observations()
        poses["ur5e"][2, 3] = -3.
        with self.assertRaisesRegex(ValueError, "in front"):
            refine_coplanar_pair(poses, points, images, K, dist, mappings)

    def test_projected_seed_behind_camera_is_rejected_not_returned(self):
        poses = {"ur5e": pose_matrix([5., 0., .01, 0., np.pi / 4, 0.]),
                 "ur10e": pose_matrix([-5., 0., .01, 0., np.pi / 4, 0.])}
        points = {robot: np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.], [0., 0., .001]])
                  for robot in ROBOTS}
        images = {robot: np.zeros((4, 2)) for robot in ROBOTS}
        mappings = {robot: np.eye(4) for robot in ROBOTS}
        with self.assertRaisesRegex(ValueError, "projection puts"):
            refine_coplanar_pair(poses, points, images, np.eye(3), None, mappings)


if __name__ == "__main__":
    unittest.main()
