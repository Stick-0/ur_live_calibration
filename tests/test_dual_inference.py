"""Dual-frame routing, image geometry and pose isolation without hardware/weights."""

import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
import urcal.inference.infer_live_dual as dual
from urcal.inference.reporting import InferenceReporter, inference_status_line


def cli_args(*extra):
    return dual.parse_args([
        "--ur5e_joints", "0", "-1", "1", "-1", "-1", "0",
        "--ur10e_joints", "0.2", "-1.1", "0.9", "-1.7", "0.5", "0.8",
        *extra,
    ])


def estimator(name, fk, args):
    pose_args = argparse.Namespace(**vars(args))
    pose_args.empnp_lambda = getattr(args, f"{name}_empnp_lambda")
    return dual.RobotEstimator(name, Mock(), [f"{name}_{i}" for i in range(7)], fk, pose_args)


class DualInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_regions_cover_odd_width_and_reject_empty_sides(self):
        left, right = dual.split_regions(641, 480, 0.4)
        self.assertEqual(left, (0, 0, 256, 480))
        self.assertEqual(right, (256, 0, 641, 480))
        for width, height, ratio in ((1, 480, 0.5), (640, 0, 0.5), (640, 480, 0),
                                     (640, 480, 1), (640, 480, float("nan")), (2, 480, 0.01)):
            with self.subTest(width=width, height=height, ratio=ratio), self.assertRaises(ValueError):
                dual.split_regions(width, height, ratio)

    def test_letterbox_keeps_each_robot_and_restores_covariance(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :640] = [10, 20, 30]
        frame[:, 640:] = [90, 80, 70]
        for region, color in zip(dual.split_regions(1280, 720), ([10, 20, 30], [90, 80, 70])):
            canvas, (left, top, right, bottom) = dual.letterbox_region(frame, region)
            self.assertEqual(canvas.shape, (480, 640, 3))
            self.assertEqual((right - left, bottom - top), (427, 480))
            np.testing.assert_array_equal(canvas[240, 320], color)
            self.assertFalse(canvas[:, :left].any())
            self.assertFalse(canvas[:, right:].any())
            local_points = np.array([[10.5, 20.25], [200.0, 300.0]])
            cov = np.array([[[4.0, 1.0], [1.0, 9.0]]] * 2)
            points, restored_cov = dual.restore_keypoints(local_points, cov, region, (480, 427))
            S = np.diag([640 / 427, 720 / 480])
            expected = (local_points + 0.5) @ S - 0.5 + [region[0], 0]
            np.testing.assert_allclose(points, expected)
            np.testing.assert_allclose(restored_cov, [S @ c @ S.T for c in cov])

    def test_real_voting_uses_content_only_and_maps_right_offset(self):
        region = (320, 0, 640, 480)
        # A deterministic network predicting exact rays towards seven points.
        targets = torch.tensor([[45., 55.], [80., 105.], [120., 155.], [155., 220.],
                                [195., 280.], [240., 350.], [280., 420.]])
        yy, xx = torch.meshgrid(torch.arange(480), torch.arange(640), indexing="ij")
        canvas_targets = targets + torch.tensor([160., 0.])
        vectors = canvas_targets[:, :, None, None] - torch.stack([xx, yy])[None]
        vectors = torch.nn.functional.normalize(vectors, dim=1)[None]
        model = Mock(return_value=(torch.full((1, 7, 480, 640), 10.), vectors))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :320] = 20
        image[:, 320:] = 180
        transform = dual.make_transform(480, 640, normalize=False)
        cfg = dual.PVNetVoteConfig(num_hypotheses=8)
        outputs = []
        for debug in (False, True):
            torch.manual_seed(32)
            outputs.append(dual.infer_region(model, image, region, transform, cfg, torch.device("cpu"), debug))
        points, covariances, valid, probs = outputs[0]
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, targets.numpy() + [320, 0], atol=0.01)
        self.assertTrue(np.isfinite(covariances).all())
        self.assertIsNone(probs)
        self.assertEqual(outputs[1][3].shape, (7, 480, 320))
        for a, b in zip(outputs[0][:3], outputs[1][:3]):
            np.testing.assert_array_equal(a, b)
        tensor = model.call_args.args[0]
        self.assertAlmostEqual(tensor[0, 0, 240, 320].item(), 180 / 255, places=6)
        self.assertEqual(tensor[0, 0, 240, 0].item(), 0.)

    def test_invalid_votes_are_excluded_and_one_failure_preserves_other_pose(self):
        args = cli_args()
        robots = [estimator("ur5e", dual.ur5e_keypoints_from_joints, args),
                  estimator("ur10e", dual.ur10e_keypoints_from_joints, args)]
        K = np.array([[615., 0, 320], [0, 610., 240], [0, 0, 1.]])
        dist = np.array([0.01, -0.005, 0.0002, -0.0001, 0.001])
        joints = [args.ur5e_joints, args.ur10e_joints]
        truths = [(np.array([0.15, -0.1, 0.2]), np.array([-0.5, -0.05, 3.])),
                  (np.array([-0.1, 0.2, -0.1]), np.array([0.5, -0.1, 3.5]))]
        votes = []
        for robot, q, (rvec, tvec) in zip(robots, joints, truths):
            points = cv2.projectPoints(robot.fk(q), rvec, tvec, K, dist)[0].reshape(-1, 2)
            valid = np.ones(7, dtype=bool)
            valid[3] = False
            points[3] = np.nan
            votes.append((points, np.tile(np.eye(2), (7, 1, 1)), valid, None))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(dual, "infer_region", side_effect=votes):
            _, results = dual.process_frame(image, joints, robots, K, dist, 0.5, None, None, torch.device("cpu"))
        for result, (rvec, tvec) in zip(results, truths):
            self.assertTrue(result.ok)
            np.testing.assert_allclose(result.rvec, rvec, atol=1e-5)
            np.testing.assert_allclose(result.tvec, tvec, atol=1e-5)
        previous = [(r.prev_rvec.copy(), r.prev_tvec.copy()) for r in robots]
        votes[0] = (votes[0][0], votes[0][1], np.zeros(7, dtype=bool), None)
        with patch.object(dual, "infer_region", side_effect=votes):
            _, results = dual.process_frame(image, joints, robots, K, dist, 0.5, None, None, torch.device("cpu"))
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)
        for robot, (rvec, tvec) in zip(robots, previous):
            np.testing.assert_allclose(robot.prev_rvec, rvec, atol=1e-5)
            np.testing.assert_allclose(robot.prev_tvec, tvec, atol=1e-5)

    def test_process_frame_routes_same_rgb_and_separate_joints(self):
        image = np.full((120, 241, 3), [10, 20, 30], dtype=np.uint8)
        robots = [Mock(), Mock()]
        joints = [list(range(6)), list(range(10, 16))]
        K, dist = np.eye(3), np.zeros(5)
        regions, results = dual.process_frame(image, joints, robots, K, dist, 0.6, None, None, torch.device("cpu"))
        left, right = [robot.infer.call_args.args for robot in robots]
        self.assertIs(left[0], right[0])
        np.testing.assert_array_equal(left[0][0, 0], [30, 20, 10])
        self.assertEqual(left[1], regions[0])
        self.assertEqual(right[1], regions[1])
        self.assertIs(left[2], joints[0])
        self.assertIs(right[2], joints[1])
        self.assertIs(left[3], K)
        self.assertIs(right[3], K)
        self.assertEqual(len(results), 2)

    def test_empty_results_leave_both_images_unobstructed_without_divider_or_text(self):
        args = cli_args()
        robots = [estimator("ur5e", dual.ur5e_keypoints_from_joints, args),
                  estimator("ur10e", dual.ur10e_keypoints_from_joints, args)]
        frame = np.full((480, 640, 3), [13, 25, 37], dtype=np.uint8)
        results = [dual.RobotResult(np.full((7, 2), np.nan), np.zeros(7, dtype=bool),
                                   np.zeros((7, 480, 320), dtype=np.float32), False,
                                   None, None, np.nan) for _ in robots]
        with patch.object(dual.cv2, "putText", wraps=cv2.putText) as text:
            vis, seg_vis = dual.draw_results(frame, dual.split_regions(640, 480), robots, results,
                                            np.eye(3), np.zeros(5), .06, 23.4, True, False)
        text.assert_not_called()
        np.testing.assert_array_equal(vis, frame)
        np.testing.assert_array_equal(seg_vis, frame)

    def test_clean_render_preserves_points_pose_axes_and_segmentation(self):
        args = cli_args()
        robots = [estimator("ur5e", dual.ur5e_keypoints_from_joints, args),
                  estimator("ur10e", dual.ur10e_keypoints_from_joints, args)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = dual.split_regions(640, 480)
        results = []
        for region, x in zip(regions, (-.5, .5)):
            probs = np.zeros((7, 480, 320), dtype=np.float32)
            probs[0] = 1.
            results.append(dual.RobotResult(np.tile([region[0] + 70., 100.], (7, 1)),
                                           np.ones(7, dtype=bool), probs, True,
                                           np.zeros(3), np.array([x, 0., 2.]), 1.))
        K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        with patch.object(dual.cv2, "putText", wraps=cv2.putText) as text, \
                patch.object(dual, "draw_pose_axes", wraps=dual.draw_pose_axes) as axes:
            vis, seg_vis = dual.draw_results(frame, regions, robots, results,
                                            K, np.zeros(5), .06, 23.4, True)
        text.assert_not_called()
        self.assertEqual(axes.call_count, 2)
        for x in (70, 390):
            self.assertTrue(vis[100, x].any())
        self.assertTrue(seg_vis.any())
        # A uniform single-class mask remains uniform: no legend blocks/text.
        self.assertTrue(np.all(seg_vis == seg_vis[0, 0]))

    def test_shared_single_robot_rendering_keeps_labels_by_default(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        with patch.object(dual.cv2, "putText", wraps=cv2.putText) as text:
            dual.draw_keypoints(frame, np.array([[25., 30.]]), ["joint"])
            self.assertEqual(text.call_count, 1)
            text.reset_mock()
            dual.draw_keypoints(frame, np.array([[25., 30.]]), ["joint"], show_labels=False)
            text.assert_not_called()
            probs = np.ones((1, 80, 120), dtype=np.float32)
            dual.make_segmentation_debug_bgr(frame, probs, ["joint"], .5)
            self.assertEqual(text.call_count, 2)
            text.reset_mock()
            dual.make_segmentation_debug_bgr(frame, probs, ["joint"], .5, show_legend=False)
            text.assert_not_called()

    def test_terminal_reports_are_throttled_and_keep_solver_and_vote_status(self):
        robots = [argparse.Namespace(name="ur5e"), argparse.Namespace(name="UR10E")]
        results = [dual.RobotResult(np.zeros((7, 2)), np.array([True] * 6 + [False]),
                                   None, True, None, None, .00001, base_plane={"applied": True}),
                   dual.RobotResult(np.zeros((7, 2)), np.zeros(7, dtype=bool),
                                    None, False, None, None, np.nan)]
        output = []
        reporter = InferenceReporter(writer=output.append)
        first = reporter.emit(10., 12.3, robots, results)
        self.assertIn("FPS 12.3", first)
        self.assertIn("UR5E PnP+plane OK keypoints=6/7", first)
        self.assertIn("UR10E EMPnP FAIL keypoints=0/7", first)
        self.assertNotIn("raw", first)
        for at in (10.01, 10.3, 10.999):
            self.assertIsNone(reporter.emit(at, 20., robots, results))
        self.assertIsNotNone(reporter.emit(11., 20., robots, results))
        self.assertEqual(len(output), 2)

    def test_control_summary_uses_original_pixel_residuals_and_packet_reason(self):
        robot = argparse.Namespace(name="ur5e")
        result = dual.RobotResult(np.zeros((7, 2)), np.ones(7, dtype=bool), None,
                                  True, None, None, .00001)
        packet = argparse.Namespace(raw_mean_px={"ur5e": 5.6}, raw_max_px={"ur5e": 8.9},
                                    valid_counts={"ur5e": 5}, captured_at=9.8,
                                    reason="joint/image\n timestamp mismatch")
        line = inference_status_line(10., [robot], [result], packet=packet, now=10.)
        self.assertIn("keypoints=5/7", line)
        self.assertIn("raw mean/max=5.6/8.9px", line)
        self.assertIn("capture age=200.0ms", line)
        self.assertIn("measurement: joint/image timestamp mismatch", line)
        self.assertNotIn("\n", line)

    def test_cli_requires_two_sources_and_preserves_robot_defaults(self):
        args = cli_args()
        self.assertEqual(args.split_ratio, 0.5)
        self.assertEqual(args.ur5e_empnp_lambda, 0.5)
        self.assertEqual(args.ur10e_empnp_lambda, 0.2)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                dual.parse_args(["--ur5e_ip", "192.0.2.1"])
            with self.assertRaises(SystemExit):
                cli_args("--split_ratio", "1")
            with self.assertRaises(SystemExit):
                cli_args("--ur5e_joints", "nan", "0", "0", "0", "0", "0")
        mixed = dual.parse_args(["--ur5e_ip", "192.0.2.1", "--ur10e_joints", *(["0"] * 6)])
        self.assertEqual(mixed.ur5e_ip, "192.0.2.1")

    def test_lfs_pointer_fails_before_hardware_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "best.pt"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
            with patch.object(dual, "parse_args", return_value=cli_args("--ur5e_checkpoint", str(pointer))), \
                    patch.object(dual, "D435ColorStream") as camera, \
                    patch.object(dual, "URJointReader") as reader, \
                    patch.object(dual, "load_model") as model:
                with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
                    dual.main()
                camera.assert_not_called()
                reader.assert_not_called()
                model.assert_not_called()

    def test_live_loop_reads_one_frame_and_both_joints_before_inference_and_cleans_up(self):
        scenarios = [(False, ()), (False, ("UR5E",)), (False, ("UR10E",)),
                     (False, ("UR5E", "UR10E")), (True, ())]
        for fail_inference, empty_robots in scenarios:
            with self.subTest(fail_inference=fail_inference, empty_robots=empty_robots):
                events = []
                args = cli_args("--show_seg", "--device", "cpu")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                camera = Mock(K=np.eye(3), dist=np.zeros(5))
                camera.read.side_effect = lambda: (events.append("frame"), frame)[1]
                readers = [Mock(), Mock()]
                readers[0].read.side_effect = lambda: (events.append("q5"), args.ur5e_joints)[1]
                readers[1].read.side_effect = lambda: (events.append("q10"), args.ur10e_joints)[1]
                seen_images = []

                def infer(robot, rgb, region, joints, *unused):
                    events.append(robot.name)
                    seen_images.append(rgb)
                    if fail_inference:
                        raise RuntimeError("synthetic inference failure")
                    points = np.tile([region[0] + 100., 200.], (7, 1))
                    detected = robot.name not in empty_robots
                    probs = np.full((7, 480, 320), float(detected), dtype=np.float32)
                    return dual.RobotResult(points, np.full(7, detected), probs, False, None, None, np.nan)

                with patch.object(dual, "parse_args", return_value=args), \
                        patch.object(dual, "require_weights"), \
                        patch.object(dual, "load_model", side_effect=[Mock(), Mock()]), \
                        patch.object(dual.torch.hub, "set_dir"), \
                        patch.object(dual, "D435ColorStream", return_value=camera) as open_camera, \
                        patch.object(dual, "URJointReader", side_effect=readers), \
                        patch.object(dual.RobotEstimator, "infer", new=infer), \
                        patch.object(dual.cv2, "imshow") as show, \
                        patch.object(dual.cv2, "waitKey", side_effect=[-1, ord("q")]), \
                        patch.object(dual.cv2, "destroyAllWindows") as close_windows:
                    if fail_inference:
                        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                            dual.main()
                    else:
                        dual.main()
                        self.assertEqual(events, ["frame", "q5", "q10", "UR5E", "UR10E"] * 2)
                        self.assertIs(seen_images[0], seen_images[1])
                        self.assertIs(seen_images[2], seen_images[3])
                        self.assertEqual(show.call_count, 4)
                        for call in show.call_args_list:
                            self.assertEqual(call.args[1].shape, frame.shape)
                    open_camera.assert_called_once()
                    self.assertEqual(camera.read.call_count, 1 if fail_inference else 2)
                    camera.stop.assert_called_once()
                    close_windows.assert_called_once()
                    for reader in readers:
                        reader.receiver.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
