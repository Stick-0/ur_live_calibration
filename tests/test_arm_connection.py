"""Remote-mode connection rejection is recoverable without any motion packet."""

import unittest
from unittest.mock import Mock

import numpy as np

import urcal.control.runtime as runtime
from tests.test_dual_control_runtime import args, controller


REMOTE_ERROR = "ur_rtde: Please enable remote control on the robot!"


class ArmConnectionTests(unittest.TestCase):
    def test_remote_mode_factory_failure_identifies_robot_and_preserves_cause(self):
        config = args()
        for robot in runtime.ROBOTS:
            with self.subTest(robot=robot):
                cause = RuntimeError(REMOTE_ERROR)
                factory = Mock(side_effect=cause)
                with self.assertRaises(runtime.ArmConnectionError) as caught:
                    runtime.MotionSession(config, (robot,), {robot: np.eye(4)}, factory=factory,
                                          defer_watchdog=True)
                fault = caught.exception
                self.assertEqual(fault.category, "arm_connection")
                self.assertEqual(fault.details, {"robot": robot, "ip": getattr(config, robot + "_ip"),
                                                 "reason": "remote_control_required"})
                self.assertIs(fault.__cause__, cause)
                self.assertIn(robot, str(fault).lower())
                self.assertIn(getattr(config, robot + "_ip"), str(fault))
                self.assertIn("remote", str(fault).lower())
                factory.assert_called_once_with(getattr(config, robot + "_ip"))

    def test_second_arm_remote_rejection_closes_first_without_upload_or_motion(self):
        config = args()
        first = controller()
        factory = Mock(side_effect=[first, RuntimeError(REMOTE_ERROR)])
        with self.assertRaises(runtime.ArmConnectionError) as caught:
            runtime.MotionSession(config, runtime.ROBOTS,
                                  {robot: np.eye(4) for robot in runtime.ROBOTS}, factory=factory)
        self.assertIn("ur10e", str(caught.exception).lower())
        first.reuploadScript.assert_not_called()
        first.setWatchdog.assert_not_called()
        first.speedL.assert_not_called()
        first.speedJ.assert_not_called()
        first.speedStop.assert_not_called()
        first.speedJStop.assert_not_called()
        first.stopScript.assert_not_called()
        first.disconnect.assert_called_once()
        self.assertEqual(factory.call_count, 2)

    def test_unrecognized_factory_error_is_not_mislabeled_or_retried(self):
        config = args()
        for cause in (RuntimeError("No route to host"), RuntimeError("unrelated remote control transport failure"),
                      ValueError("invalid SDK argument")):
            with self.subTest(message=str(cause)):
                factory = Mock(side_effect=cause)
                with self.assertRaises(type(cause)) as caught:
                    runtime.MotionSession(config, ("ur5e",), {"ur5e": np.eye(4)}, factory=factory)
                self.assertIs(caught.exception, cause)
                self.assertNotIsInstance(caught.exception, runtime.ArmConnectionError)
                factory.assert_called_once()

    def test_remote_word_in_a_post_connection_failure_is_not_a_recoverable_connect_error(self):
        config = args()
        control = controller()
        cause = RuntimeError(REMOTE_ERROR)
        control.reuploadScript.side_effect = cause
        with self.assertRaises(RuntimeError) as caught:
            runtime.MotionSession(config, ("ur5e",), {"ur5e": np.eye(4)}, factory=Mock(return_value=control))
        self.assertIs(caught.exception, cause)
        self.assertNotIsInstance(caught.exception, runtime.ArmConnectionError)
        control.speedL.assert_not_called()
        control.speedJ.assert_not_called()
        control.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
