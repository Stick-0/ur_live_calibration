"""Raw residual diagnostics and explicit snapshot files, without hardware."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.diagnostics import (
    calibration_debug_record, cv2, diagnostic_lines, draw_calibration_debug,
    pose_input_diagnostics, save_calibration_debug,
)
from urcal.control.vision import calibration_frame


class CalibrationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
        self.dist = np.zeros(5)
        self.points3d = np.array([[-.12, -.10, 0], [.12, -.10, .04], [.16, .12, .02],
                                  [-.13, .13, -.03], [.02, -.02, .12], [.08, .07, .07], [-.09, .04, .05]])
        self.regions = ((0, 0, 320, 480), (320, 0, 640, 480))
        self.joints = [np.zeros(6), np.ones(6) * .2]
        self.results, self.estimators = [], []
        for name, x in (("ur5e", -.2), ("ur10e", .2)):
            rvec, tvec = np.array([.12, -.18, .07]), np.array([x, 0, 1.2])
            points = cv2.projectPoints(self.points3d, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
            self.results.append(SimpleNamespace(ok=True, points=points, valid=np.ones(7, dtype=bool),
                                                rvec=rvec, tvec=tvec))
            self.estimators.append(SimpleNamespace(fk=lambda q: self.points3d.copy(),
                                                   key_names=[name + f"_link{i}" for i in range(7)]))

    def record(self, **options):
        self.frame = calibration_frame(7, 100., self.results, self.estimators, self.joints,
                                       self.K, self.dist, .08, .04)
        return calibration_debug_record(self.frame, self.results, self.estimators, self.joints,
                                        self.K, self.dist, self.regions, **options)

    def pose_inputs(self):
        return pose_input_diagnostics(self.results, self.estimators, self.joints,
                                      self.K, self.dist, self.regions)

    def test_pnp_input_rows_match_actual_solver_filtering(self):
        from urcal.inference.infer_live import solve_pose_empnp

        self.results[0].points[1] = np.nan
        self.points3d[2] = np.inf
        self.results[0].valid[5] = False
        record = self.pose_inputs()
        robot = record["robots"]["ur5e"]
        indices = robot["solver_indices"]
        np.testing.assert_array_equal(indices, [0, 3, 4, 6])
        # Intercept the real solver wrapper's fit inputs. No PnP runs here.
        args = SimpleNamespace(empnp_use_prev=False, empnp_prior=.9, empnp_sigma=3.,
                               camera_w=640, camera_h=480, empnp_max_iter=50,
                               empnp_tol=1e-6, empnp_lambda=.5)
        valid = self.results[0].valid
        covariances = np.tile(np.eye(2), (7, 1, 1))
        with patch("urcal.inference.infer_live.EMPnP") as solver:
            solver.return_value.fit.return_value = (np.zeros(3), np.array([0., 0., 1.]),
                                                   np.ones(4), np.zeros(4))
            ok, *_ = solve_pose_empnp(self.points3d[valid], self.results[0].points[valid],
                                     covariances[valid], self.K, self.dist, args, None, None, None)
            self.assertTrue(ok)
            fit_args = solver.return_value.fit.call_args.args
            np.testing.assert_array_equal(robot["points_3d_base_m"][indices], fit_args[0])
            np.testing.assert_array_equal(robot["detected_px"][indices], fit_args[1])
        # Preserve the original vote validity separately from finite filtering.
        self.assertTrue(robot["valid"][1])
        self.assertTrue(robot["valid"][2])

    def test_pnp_inputs_preserve_failed_result_without_running_solver(self):
        self.results[0].ok = False
        self.results[0].rvec = self.results[0].tvec = None
        with patch("urcal.control.diagnostics.cv2.solvePnPRansac") as solver:
            record = self.pose_inputs()
            solver.assert_not_called()
        self.assertEqual(record["schema"], "urcal.pnp-initialization.v1")
        self.assertEqual(record["initialization"]["reprojection_error_px"], 3.)
        self.assertEqual(record["initialization"]["opencv_version"], cv2.__version__)
        robot = record["robots"]["ur5e"]
        self.assertFalse(robot["empnp_ok"])
        np.testing.assert_array_equal(robot["detected_px"], self.results[0].points)
        np.testing.assert_array_equal(robot["solver_indices"], np.arange(7))
        np.testing.assert_array_equal(robot["q_rad"], self.joints[0])
        self.assertEqual(robot["region"], list(self.regions[0]))

    def test_pnp_inputs_are_isolated_from_source_and_other_snapshots(self):
        record = self.pose_inputs()
        sibling = self.pose_inputs()
        self.K[:] = 0
        self.dist[:] = 1
        self.joints[0][:] = 2
        self.results[0].points[:] = 3
        self.results[0].valid[:] = False
        self.points3d[:] = 4
        self.estimators[0].key_names[0] = "changed"
        self.assertEqual(record["K"][0, 0], 600.)
        np.testing.assert_array_equal(record["dist"], np.zeros(5))
        robot = record["robots"]["ur5e"]
        np.testing.assert_array_equal(robot["q_rad"], np.zeros(6))
        self.assertTrue(robot["valid"].all())
        self.assertFalse(np.all(robot["detected_px"] == 3))
        self.assertFalse(np.all(robot["points_3d_base_m"] == 4))
        self.assertEqual(robot["key_names"][0], "ur5e_link0")
        # Consumer mutation must not corrupt later snapshots, another robot,
        # or objects still held by the capture/inference process.
        record["K"][:] = 99
        robot["q_rad"][:] = 99
        robot["points_3d_base_m"][:] = 99
        robot["detected_px"][:] = 99
        robot["valid"][:] = False
        robot["solver_indices"][:] = 99
        robot["key_names"][0] = "consumer"
        robot["region"][0] = 99
        other = sibling["robots"]["ur5e"]
        self.assertEqual(sibling["K"][0, 0], 600.)
        np.testing.assert_array_equal(other["q_rad"], np.zeros(6))
        self.assertTrue(other["valid"].all())
        np.testing.assert_array_equal(other["solver_indices"], np.arange(7))
        self.assertEqual(other["key_names"][0], "ur5e_link0")
        self.assertEqual(other["region"], list(self.regions[0]))
        self.assertFalse(np.all(other["detected_px"] == 99))
        self.assertFalse(np.all(other["points_3d_base_m"] == 99))
        self.assertFalse(np.all(record["robots"]["ur10e"]["points_3d_base_m"] == 99))
        np.testing.assert_array_equal(self.joints[0], np.full(6, 2.))

    def test_pnp_inputs_reject_inconsistent_shapes_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            pose_input_diagnostics(self.results[:1], self.estimators, self.joints,
                                   self.K, self.dist, self.regions)
        self.results[0].valid = np.ones(6, dtype=bool)
        with self.assertRaisesRegex(ValueError, "inconsistent.*shapes"):
            self.pose_inputs()

    def test_disabled_limits_remain_visible_with_raw_numbers_and_snapshot_setting(self):
        self.results[0].points[3] += [12, 16]
        record = self.record(check_reprojection=False)
        self.assertFalse(record["limits"]["check_reprojection"])
        text, green = diagnostic_lines(record)[0]
        self.assertIn("RAW limits OFF", text)
        self.assertIn("max=20.0px", text)
        self.assertNotIn("/8.0", text)
        self.assertFalse(green)
        image = np.zeros((480, 640, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = save_calibration_debug(folder, image, image, record)
            payload = json.loads(path.read_text())
            self.assertFalse(payload["limits"]["check_reprojection"])
            self.assertAlmostEqual(payload["robots"]["ur5e"]["raw_max_px"], 20.)

    def test_disabled_limits_do_not_hide_pnp_failure(self):
        self.results[0].ok = False
        record = self.record(check_reprojection=False)
        text, green = diagnostic_lines(record)[0]
        self.assertIn("RAW limits OFF", text)
        self.assertIn("PnP failed", text)
        self.assertFalse(green)

    def test_reports_original_outlier_and_correct_landmark_without_changing_source(self):
        self.results[0].points[3] += [12, 16]
        before = self.results[0].points.copy()
        record = self.record()
        robot = record["robots"]["ur5e"]
        self.assertEqual(robot["worst_index"], 3)
        self.assertAlmostEqual(robot["errors_px"][3], 20.)
        self.assertAlmostEqual(robot["raw_mean_px"], 20./7)
        self.assertAlmostEqual(robot["raw_max_px"], 20.)
        self.assertIn("worst=link3", diagnostic_lines(record)[0][0])
        self.assertFalse(diagnostic_lines(record)[0][1])
        self.assertTrue(diagnostic_lines(record)[1][1])
        np.testing.assert_array_equal(self.results[0].points, before)
        robot["camera_from_base"][:] = 0
        self.assertFalse(np.all(self.frame.poses["ur5e"] == 0))

    def test_failed_pnp_still_reports_valid_detection_count_without_invented_residual(self):
        self.results[0].ok = False
        self.results[0].rvec = self.results[0].tvec = None
        record = self.record()
        robot = record["robots"]["ur5e"]
        self.assertEqual(robot["count"], 7)
        self.assertIsNone(robot["raw_mean_px"])
        self.assertTrue(np.isnan(robot["errors_px"]).all())
        self.assertIn("PnP failed", diagnostic_lines(record)[0][0])

    def test_invalid_detection_never_becomes_worst_point_or_residual_line(self):
        self.results[0].points[2] = np.nan
        self.results[0].valid[5] = False
        self.results[0].points[5] = 10000
        record = self.record()
        robot = record["robots"]["ur5e"]
        self.assertEqual(robot["count"], 5)
        self.assertTrue(np.isnan(robot["errors_px"][[2, 5]]).all())
        self.assertNotIn(robot["worst_index"], (2, 5))
        with patch("urcal.control.diagnostics.cv2.drawMarker") as marker:
            draw_calibration_debug(np.zeros((480, 640, 3), np.uint8), record)
            self.assertEqual(marker.call_count, 12)

    def test_hidden_keypoints_hide_projection_marks_but_keep_metrics(self):
        record = self.record()
        with patch("urcal.control.diagnostics.cv2.drawMarker") as marker, \
                patch("urcal.control.diagnostics.cv2.putText", wraps=cv2.putText) as put_text:
            draw_calibration_debug(np.zeros((480, 640, 3), np.uint8), record, False)
            marker.assert_not_called()
            self.assertTrue(any("ur5e RAW" in call.args[1] for call in put_text.call_args_list))

    def test_live_overlay_contains_geometry_without_text_and_hidden_points_leave_image_clean(self):
        record = self.record()
        self.results[0].points[0] += [3., 4.]
        record = self.record()
        image = np.zeros((480, 640, 3), np.uint8)
        with patch("urcal.control.diagnostics.cv2.putText") as text, \
                patch("urcal.control.diagnostics.cv2.drawMarker", wraps=cv2.drawMarker) as marker, \
                patch("urcal.control.diagnostics.cv2.line", wraps=cv2.line) as line:
            draw_calibration_debug(image, record, show_text=False)
            text.assert_not_called()
            self.assertTrue(marker.called)
            self.assertTrue(line.called)
            self.assertTrue(np.any(image))
        blank = np.zeros_like(image)
        draw_calibration_debug(blank, record, show_keypoints=False, show_text=False)
        np.testing.assert_array_equal(blank, np.zeros_like(image))

    def test_extreme_rejected_projection_does_not_overflow_opencv(self):
        record = self.record()
        record["robots"]["ur5e"]["projected_px"][0] = [1e100, -1e100]
        image = np.zeros((480, 640, 3), np.uint8)
        self.assertIs(draw_calibration_debug(image, record), image)

    def test_snapshots_are_replayable_json_and_distinct_pngs_with_no_nan_tokens(self):
        self.results[0].points[1] = np.nan
        record = self.record()
        raw = np.zeros((480, 640, 3), np.uint8)
        overlay = draw_calibration_debug(raw.copy(), record)
        with tempfile.TemporaryDirectory() as folder:
            first = save_calibration_debug(folder, raw, overlay, record, "WAIT: test")
            second = save_calibration_debug(folder, raw, overlay, record, "WAIT: test")
            self.assertNotEqual(first, second)
            data = json.loads(first.read_text(), parse_constant=lambda token: self.fail(token))
            self.assertEqual(data["control_message"], "WAIT: test")
            self.assertEqual(data["robots"]["ur5e"]["detected_px"][1], [None, None])
            np.testing.assert_allclose(data["robots"]["ur10e"]["q_rad"], self.joints[1])
            np.testing.assert_allclose(data["K"], self.K)
            for label, expected in (("raw", raw), ("overlay", overlay)):
                image = cv2.imread(str(Path(folder) / data["images"][label]))
                np.testing.assert_array_equal(image, expected)
            self.assertEqual(len(list(Path(folder).iterdir())), 6)


if __name__ == "__main__":
    unittest.main()
