"""The standalone display demo reports invented TCP errors, with no robot IO."""

import builtins
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.geometry import flange_from_tcp
from urcal.tools.preview_gui import DEMO_NOTICE, DemoState


class PreviewErrorTelemetryTests(unittest.TestCase):
    def test_demo_labels_synthetic_error_and_keeps_requested_timestamp(self):
        demo = DemoState(now=100.)
        status = demo.status(now=102.2)
        error = status["position_error"]
        self.assertTrue(error["valid"], error["reason"])
        self.assertEqual(error["at"], 102.2)
        self.assertEqual(error["clock"], "monotonic")
        self.assertEqual(error["phase"], "demo")
        self.assertEqual(error["follower"], "ur5e")
        self.assertEqual(error["target_source"], "synthetic nominal target")
        self.assertIn(DEMO_NOTICE, status["message"])
        self.assertFalse(status["active"])
        self.assertFalse(status["ready"])

    def test_signed_error_uses_follower_axes_and_actual_nonzero_tool_offsets(self):
        demo = DemoState(now=100.)
        status = demo.status(now=102.2)
        scene = status["scene"]
        base, state = scene["bases"]["ur5e"], scene["robots"]["ur5e"]
        target_camera = scene["goals"]["ur5e"] @ state["tcp_offset"]
        actual_camera = base @ state["tcp"]
        camera_error = target_camera[:3, 3] - actual_camera[:3, 3]
        expected = base[:3, :3].T @ camera_error
        np.testing.assert_allclose(status["position_error"]["error_m"], expected, atol=1e-12)
        self.assertFalse(np.allclose(expected, camera_error, atol=1e-5))
        self.assertGreater(np.linalg.norm(state["tcp_offset"][:3, 3]), .03)
        # This is TCP error, not flange error: rotating an attached tool changes
        # its origin even when the target was specified at the flange.
        current_flange = base @ flange_from_tcp(state["tcp"], state["tcp_offset"])
        flange_error = base[:3, :3].T @ (scene["goals"]["ur5e"][:3, 3] - current_flange[:3, 3])
        self.assertGreater(np.linalg.norm(expected - flange_error), 1e-5)

    def test_display_target_edit_updates_error_without_moving_synthetic_robot(self):
        demo = DemoState(now=100.)
        before = demo.status(now=101.)
        command = deepcopy(demo.request)
        command["gap"] += .04
        demo.apply(command)
        after = demo.status(now=101.)
        leader_state = before["scene"]["robots"]["ur10e"]
        leader_camera = (before["scene"]["bases"]["ur10e"]
                         @ flange_from_tcp(leader_state["tcp"], leader_state["tcp_offset"]))
        expected_delta = .04 * before["scene"]["bases"]["ur5e"][:3, :3].T @ leader_camera[:3, 2]
        np.testing.assert_allclose(np.asarray(after["position_error"]["error_m"])
                                   - before["position_error"]["error_m"], expected_delta, atol=1e-12)
        for name in before["scene"]["robots"]:
            np.testing.assert_array_equal(before["scene"]["robots"][name]["tcp"],
                                          after["scene"]["robots"][name]["tcp"])
        self.assertFalse(after["active"])

    def test_status_is_detached_and_never_imports_hardware_sdks(self):
        original_import = builtins.__import__
        def offline_import(name, *args, **kwargs):
            if name.split(".")[0] in {"rtde_control", "rtde_receive", "pyrealsense2"}:
                raise AssertionError("demo imported hardware SDK: " + name)
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=offline_import):
            demo = DemoState(now=100.)
            first = demo.status(now=101.)
            expected = list(first["position_error"]["error_m"])
            first["position_error"]["error_m"][0] = 999.
            first["scene"]["robots"]["ur5e"]["tcp"][:] = 0.
            second = demo.status(now=101.)
        np.testing.assert_array_equal(second["position_error"]["error_m"], expected)
        self.assertTrue(second["position_error"]["valid"])
        self.assertEqual(second["position_error"]["at"], first["position_error"]["at"])


if __name__ == "__main__":
    unittest.main()
