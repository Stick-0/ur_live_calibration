"""RealSense selection checks without opening cameras or connecting to robots."""

from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from urcal.inference import infer_live as live


class FakeDevice:
    def __init__(self, serial, name="Intel RealSense D435"):
        self.info = {"serial_number": serial, "name": name}

    def supports(self, field):
        return field in self.info

    def get_info(self, field):
        return self.info[field]


def fake_realsense(devices):
    config = Mock()
    selected = {"serial": None}
    config.enable_device.side_effect = lambda serial: selected.update(serial=serial)
    intrinsics = SimpleNamespace(fx=601., fy=602., ppx=320., ppy=240., coeffs=[0.] * 5)
    video_profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
    color_profile = SimpleNamespace(as_video_stream_profile=lambda: video_profile)

    def get_device():
        if selected["serial"] is None:
            return devices[0]
        return next(device for device in devices
                    if device.get_info("serial_number") == selected["serial"])

    profile = SimpleNamespace(get_device=get_device, get_stream=lambda stream: color_profile)
    pipeline = Mock()
    pipeline.start.return_value = profile
    return SimpleNamespace(
        camera_info=SimpleNamespace(serial_number="serial_number", name="name"),
        stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8"),
        context=Mock(return_value=SimpleNamespace(query_devices=lambda: devices)),
        pipeline=Mock(return_value=pipeline), config=Mock(return_value=config),
    )


class CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        # Deliberately reverse discovery order to verify the displayed indices.
        self.rs = fake_realsense([FakeDevice("222222"), FakeDevice("111111")])
        self.module_patch = patch.dict(sys.modules, {"pyrealsense2": self.rs})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_listing_and_index_selection_use_the_same_serial_order(self):
        output = io.StringIO()
        with redirect_stdout(output):
            live.list_cameras()
        lines = output.getvalue().splitlines()
        self.assertTrue(any("[0]" in line and "111111" in line for line in lines))
        self.assertTrue(any("[1]" in line and "222222" in line for line in lines))
        self.rs.pipeline.assert_not_called()
        for index, serial in enumerate(("111111", "222222")):
            with self.subTest(index=index), redirect_stdout(io.StringIO()):
                stream = live.D435ColorStream(640, 480, 30, camera_index=index)
                self.rs.config.return_value.enable_device.assert_called_with(serial)
                stream.stop()

    def test_serial_binds_requested_camera_and_reports_selected_device(self):
        output = io.StringIO()
        with redirect_stdout(output):
            stream = live.D435ColorStream(640, 480, 30, camera_serial="111111")
        config = self.rs.config.return_value
        config.enable_device.assert_called_once_with("111111")
        config.enable_stream.assert_called_once_with("color", 640, 480, "bgr8", 30)
        self.rs.pipeline.return_value.start.assert_called_once_with(config)
        self.assertIn("111111", output.getvalue())
        np.testing.assert_array_equal(stream.K, [[601., 0., 320.], [0., 602., 240.], [0., 0., 1.]])
        stream.stop()
        self.rs.pipeline.return_value.stop.assert_called_once()

    def test_no_selector_preserves_sdk_automatic_selection(self):
        with redirect_stdout(io.StringIO()):
            stream = live.D435ColorStream(640, 480, 30)
        self.rs.context.assert_not_called()
        self.rs.config.return_value.enable_device.assert_not_called()
        self.rs.pipeline.return_value.start.assert_called_once()
        stream.stop()

    def test_invalid_selection_never_creates_a_pipeline(self):
        for selection in ({"camera_index": -1}, {"camera_index": 2},
                          {"camera_serial": "missing"},
                          {"camera_serial": "111111", "camera_index": 0}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                live.D435ColorStream(640, 480, 30, **selection)
        self.rs.pipeline.assert_not_called()

    def test_no_devices_lists_message_and_rejects_selection(self):
        self.rs.context.return_value.query_devices = lambda: []
        output = io.StringIO()
        with redirect_stdout(output):
            live.list_cameras()
        self.assertIn("No RealSense cameras connected", output.getvalue())
        for selection in ({"camera_index": 0}, {"camera_serial": "111111"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                live.D435ColorStream(640, 480, 30, **selection)
        self.rs.pipeline.assert_not_called()

    def test_list_only_needs_no_checkpoint_robot_or_model(self):
        with patch.object(sys, "argv", ["live", "--list_cameras"]), \
                patch.object(live, "D435ColorStream") as stream, \
                patch.object(live, "URJointReader") as reader, \
                patch.object(live, "RobotJointDetection") as model, \
                patch.object(live, "load_checkpoint") as checkpoint, \
                redirect_stdout(io.StringIO()) as output:
            live.main()
        self.assertIn("111111", output.getvalue())
        stream.assert_not_called()
        reader.assert_not_called()
        model.assert_not_called()
        checkpoint.assert_not_called()
        self.rs.pipeline.assert_not_called()

    def test_cli_requires_checkpoint_for_inference_and_exclusive_selectors(self):
        invalid = [[], ["--camera_index", "0"],
                   ["--checkpoint", "test.pt", "--camera_index", "0", "--camera_serial", "111111"]]
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                live.parse_args(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_main_forwards_selection_before_robot_connection(self):
        for option, value in (("--camera_index", "1"), ("--camera_serial", "111111")):
            with self.subTest(option=option), \
                    patch.object(sys, "argv", ["live", "--checkpoint", "test.pt", option, value]), \
                    patch.object(live, "D435ColorStream", side_effect=RuntimeError("camera boundary")) as stream, \
                    patch.object(live, "URJointReader") as reader:
                with self.assertRaisesRegex(RuntimeError, "camera boundary"):
                    live.main()
                self.assertEqual(stream.call_args.kwargs[option[2:]], int(value) if option == "--camera_index" else value)
                reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
