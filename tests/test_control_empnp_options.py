"""Control CLI to vision worker to solver options, without robot/camera IO."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from urcal.control.runtime import parse_args
from urcal.control.vision import make_pose_args, vision_worker
from urcal.inference import infer_live, infer_live_dual as dual


CLI = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10", "--device", "cpu"]


def config(*options, mode="follow"):
    return parse_args(["--mode", mode, *CLI, *options])


class ControlEMPnPOptionsTests(unittest.TestCase):
    def test_defaults_preserve_existing_control_previous_pose_off(self):
        for mode in ("follow",):
            args = config(mode=mode)
            for robot, weight in (("ur5e", .5), ("ur10e", .2)):
                settings = make_pose_args(args, robot)
                self.assertFalse(settings.empnp_use_prev)
                self.assertEqual(settings.empnp_lambda, weight)
                self.assertEqual(settings.empnp_max_iter, 50)

    def test_weak_prior_weights_remain_independent_for_both_robots(self):
        for mode in ("follow",):
            args = config("--empnp_use_prev", "--ur5e_empnp_lambda", ".005",
                          "--ur10e_empnp_lambda", ".001", mode=mode)
            self.assertTrue(make_pose_args(args, "ur5e").empnp_use_prev)
            self.assertEqual(make_pose_args(args, "ur5e").empnp_lambda, .005)
            self.assertEqual(make_pose_args(args, "ur10e").empnp_lambda, .001)
            self.assertEqual(args.filter_alpha, .2)

    def test_weights_alone_do_not_silently_reenable_previous_pose(self):
        for options in ([], ["--no_empnp_use_prev"]):
            args = config(*options, "--ur5e_empnp_lambda", ".005")
            self.assertFalse(make_pose_args(args, "ur5e").empnp_use_prev)

    def test_invalid_weights_and_conflicting_switches_are_rejected(self):
        for robot in ("ur5e", "ur10e"):
            for value in ("nan", "inf", "-inf", "-.001", "1.001"):
                with self.subTest(robot=robot, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        config(f"--{robot}_empnp_lambda={value}")
            for value in ("0", "1"):
                self.assertEqual(make_pose_args(config(f"--{robot}_empnp_lambda={value}"), robot).empnp_lambda,
                                 float(value))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            config("--empnp_use_prev", "--no_empnp_use_prev")

    def test_legacy_namespace_preserves_control_vision_configuration(self):
        args = SimpleNamespace(camera_w=640, camera_h=480)
        self.assertFalse(make_pose_args(args, "ur5e").empnp_use_prev)
        self.assertEqual(make_pose_args(args, "ur5e").empnp_lambda, .5)
        self.assertEqual(make_pose_args(args, "ur10e").empnp_lambda, .2)

    def test_prior_points_and_weight_reach_real_estimator_solver_boundary(self):
        points3d = np.array([[-.1, -.1, 0], [.1, -.1, .04], [.1, .1, .02],
                             [-.1, .1, -.03], [.02, -.02, .12], [.08, .07, .07], [-.09, .04, .05]])
        K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
        rotation, translation = np.array([.1, -.2, .05]), np.array([.2, 0., 1.5])
        previous_points = cv2.projectPoints(points3d, rotation, translation, K, np.zeros(5))[0].reshape(-1, 2)
        observed = previous_points + [3., 0.]
        covariances = np.repeat(np.eye(2)[None], 7, axis=0)
        for enabled in (True, False):
            args = config("--empnp_use_prev" if enabled else "--no_empnp_use_prev",
                          "--ur5e_empnp_lambda", ".005", "--ur10e_empnp_lambda", ".001")
            for robot, weight in (("ur5e", .005), ("ur10e", .001)):
                with self.subTest(enabled=enabled, robot=robot):
                    estimator = dual.RobotEstimator(robot, Mock(), [str(i) for i in range(7)],
                                                     lambda joints: points3d, make_pose_args(args, robot))
                    estimator.prev_rvec, estimator.prev_tvec = rotation.copy(), translation.copy()
                    with patch.object(dual, "infer_region", return_value=(observed, covariances, np.ones(7, bool), None)), \
                            patch.object(infer_live, "EMPnP") as solver:
                        solver.return_value.fit.return_value = (rotation.copy(), translation.copy(), np.ones(7), np.zeros(7))
                        result = estimator.infer(None, None, np.zeros(6), K, np.zeros(5), None, None, None)
                    self.assertTrue(result.ok)
                    call = solver.return_value.fit.call_args
                    self.assertEqual(call.kwargs["lamda"], weight)
                    np.testing.assert_allclose(call.args[1], observed)
                    np.testing.assert_allclose(call.args[2], previous_points if enabled else observed)

    def test_worker_constructs_both_estimators_with_requested_options_before_io(self):
        args = config("--empnp_use_prev", "--ur5e_empnp_lambda", ".005", "--ur10e_empnp_lambda", ".001")
        stop, output = threading.Event(), io.StringIO()
        receiver = Mock(side_effect=RuntimeError("test ends before robot IO"))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"rtde_receive": SimpleNamespace(RTDEReceiveInterface=receiver)}))
            for name in ("require_weights", "load_model", "make_transform"):
                stack.enter_context(patch.object(dual, name))
            constructor = stack.enter_context(patch.object(dual, "RobotEstimator"))
            stack.enter_context(patch("torch.hub.set_dir"))
            stack.enter_context(patch("urcal.control.vision.cv2.destroyAllWindows"))
            stack.enter_context(redirect_stdout(output))
            vision_worker(args, Mock(), Mock(), Mock(), stop)
        self.assertTrue(stop.is_set())
        self.assertEqual(constructor.call_count, 2)
        for call, robot, weight in zip(constructor.call_args_list, ("ur5e", "ur10e"), (.005, .001)):
            self.assertEqual(call.args[0], robot)
            self.assertTrue(call.args[4].empnp_use_prev)
            self.assertEqual(call.args[4].empnp_lambda, weight)
            self.assertIn(f"{robot} EMPnP previous pose: ON; lambda={weight:g}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
