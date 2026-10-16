"""Target conversion and GUI callbacks with no Tk display or hardware."""

from pathlib import Path
from queue import Queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
import urcal.control.gui as gui
from urcal.control.geometry import pose_matrix


class Variable:
    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


def fake_panel(mode="hold", execute=True):
    panel = object.__new__(gui.TargetPanel)
    panel.args = SimpleNamespace(mode=mode, execute=execute, min_gap=.15)
    panel.mode = mode
    panel.send_command = Mock()
    panel.stop_event = threading.Event()
    panel._status = {"active": False, "ready": False}
    panel._position_vars = [Variable(value) for value in [.2, -.1, 1.3]]
    panel._angle_vars = [Variable(value) for value in [10., 20., 30.]]
    panel._gap = Variable(30.)
    panel._message = Variable()
    panel._error = Variable()
    panel._editable = [Mock(), Mock()]
    panel._apply_button = Mock()
    panel._arm_button = Mock()
    panel._closed = False
    panel._seeded = False
    panel._edited = False
    panel._programmatic = False
    panel.root = Mock()
    panel._tk = SimpleNamespace(TclError=type("FakeTkError", (Exception,), {}))
    return panel


def research_panel():
    panel = fake_panel(mode="follow")
    panel.args.method = "separated"
    panel._research = True
    panel._task_paused = False
    panel._path = Variable("hold")
    panel._delta_vars = [Variable(value) for value in [0, 0, -5]]
    panel._delta_rotation_vars = [Variable(value) for value in [0, 0, 0]]
    panel._radius = Variable(2)
    panel._duration = Variable(10)
    panel._path_selector = Mock()
    panel._pause_button = Mock()
    panel._progress_message = Variable()
    panel._editable.extend([Mock() for _ in range(8)])
    return panel


class TargetConversionTests(unittest.TestCase):
    def test_hold_converts_cm_and_degrees_with_explicit_rz_ry_rx_order(self):
        command = gui.build_target_command("hold", "30", .15, [".2", "-.1", "1.3"], [10, 20, 30])
        self.assertEqual(command["type"], "target")
        self.assertAlmostEqual(command["gap"], .3)
        pose = pose_matrix(command["center_pose"])
        rx = Rotation.from_euler("x", 10, degrees=True).as_matrix()
        ry = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        rz = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        np.testing.assert_allclose(pose[:3, :3], rz @ ry @ rx, atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], [.2, -.1, 1.3])

    def test_follow_only_sends_gap(self):
        self.assertEqual(gui.build_target_command("follow", 30, .15), {"type": "target", "gap": .3})

    def test_pose_to_fields_preserves_orientation_including_gimbal_lock(self):
        for angles in ([10, 20, 30], [-175, 89, 190], [0, 90, 40], [0, -90, -40]):
            original = gui.build_target_command("hold", 30, .15, [0, 0, 1], angles)["center_pose"]
            position, reconstructed_angles = gui.pose_to_fields(original)
            roundtrip = gui.build_target_command("hold", 30, .15, position, reconstructed_angles)["center_pose"]
            np.testing.assert_allclose(pose_matrix(original), pose_matrix(roundtrip), atol=1e-12)

    def test_invalid_gap_center_and_orientation_are_rejected(self):
        for gap in (14.9, 0, -2, "", "bad", np.nan, np.inf):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                gui.build_target_command("follow", gap, .15)
        for position in ([0, 0, 0], [0, 0, -1], [0, 0], [0, np.nan, 1], [0, "", 1]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                gui.build_target_command("hold", 30, .15, position, [0, 0, 0])
        for rotation in ([0, 0], [0, np.inf, 0], ["", 0, 0]):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                gui.build_target_command("hold", 30, .15, [0, 0, 1], rotation)
        with self.assertRaises(ValueError):
            gui.build_target_command("invalid", 30, .15)

    def test_relative_path_defaults_and_units_do_not_use_euler_conversion(self):
        defaults = gui.build_target_command("follow", 30, .15, path="hold")
        self.assertEqual(defaults["path_delta"], [0, 0, -.05, 0, 0, 0])
        self.assertEqual(defaults["path_radius"], .02)
        self.assertEqual(defaults["path_duration"], 10.)
        command = gui.build_target_command("follow", 30, .15, path="line",
                                           path_delta_cm=[2, -3, -5], path_rotation_degrees=[90, 45, -30],
                                           path_radius_cm=4, path_duration=20)
        np.testing.assert_allclose(command["path_delta"], [.02, -.03, -.05, np.pi / 2, np.pi / 4, -np.pi / 6])
        self.assertEqual(command["path_radius"], .04)
        self.assertEqual(command["path_duration"], 20.)

    def test_invalid_relative_path_parameters_are_rejected(self):
        for kwargs in ({"path": "bad"}, {"path_radius_cm": 0}, {"path_duration": -1},
                       {"path_duration": np.nan}, {"path_delta_cm": [0, np.inf, 0]},
                       {"path_rotation_degrees": [0, 0]}):
            parameters = {"path": "circle", **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gui.build_target_command("follow", 30, .15, **parameters)


class PanelCallbackTests(unittest.TestCase):
    def test_apply_sends_valid_pose_and_reports_local_errors_without_sending(self):
        panel = fake_panel()
        panel._apply()
        payload = panel.send_command.call_args.args[0]
        self.assertEqual(payload["type"], "target")
        self.assertAlmostEqual(payload["gap"], .3)
        panel.send_command.reset_mock()
        panel._gap.set("bad")
        panel._apply()
        panel.send_command.assert_not_called()
        self.assertIn("finite", panel._error.get())

    def test_active_disables_fields_apply_and_arm_and_callbacks_do_not_send(self):
        panel = fake_panel()
        panel._status = {"active": True, "ready": True}
        panel._refresh_enabled()
        for widget in panel._editable + [panel._apply_button, panel._arm_button]:
            widget.configure.assert_called_with(state="disabled")
        panel._apply()
        panel._arm()
        panel.send_command.assert_not_called()

    def test_arm_requires_execute_and_ready(self):
        for execute, ready in ((False, False), (False, True), (True, False)):
            panel = fake_panel(execute=execute)
            panel._status["ready"] = ready
            panel._arm()
            panel.send_command.assert_not_called()
        panel = fake_panel(execute=True)
        panel._status["ready"] = True
        panel._refresh_enabled()
        panel._arm_button.configure.assert_called_with(state="normal")
        panel._arm()
        panel.send_command.assert_called_once_with("arm")

    def test_seed_happens_once_and_does_not_overwrite_edited_center(self):
        panel = fake_panel()
        center = [.4, .2, 1.1, .1, .2, -.3]
        panel.poll({"center_pose": center, "message": "READY", "ready": True})
        self.assertTrue(panel._seeded)
        values = [variable.get() for variable in panel._position_vars + panel._angle_vars]
        panel.poll({"center_pose": [1, 2, 3, 0, 0, 0]})
        self.assertEqual(values, [variable.get() for variable in panel._position_vars + panel._angle_vars])
        self.assertEqual(panel._message.get(), "READY")
        edited = fake_panel()
        edited._on_center_edit()
        edited.poll({"center_pose": center})
        self.assertEqual(edited._position_vars[0].get(), "0.2")
        self.assertFalse(edited._seeded)

    def test_follow_does_not_seed_or_send_fixed_center(self):
        panel = fake_panel(mode="follow")
        panel.poll({"center_pose": [1, 2, 3, 0, 0, 0]})
        self.assertFalse(panel._seeded)
        panel._apply()
        panel.send_command.assert_called_once_with({"type": "target", "gap": .3})

    def test_queue_failure_is_visible_and_does_not_stop_gui(self):
        panel = fake_panel()
        panel.send_command.side_effect = RuntimeError("command queue full")
        panel._apply()
        self.assertIn("queue full", panel._error.get())
        self.assertFalse(panel.stop_event.is_set())
        panel._status["ready"] = True
        panel._arm()
        self.assertIn("queue full", panel._error.get())

    def test_stop_uses_shared_event_even_when_send_would_fail(self):
        panel = fake_panel()
        panel.send_command.side_effect = RuntimeError("full")
        panel._request_stop()
        self.assertTrue(panel.stop_event.is_set())
        self.assertTrue(panel._closed)
        panel.root.destroy.assert_called_once_with()
        panel.send_command.assert_not_called()
        panel.close()
        panel.root.destroy.assert_called_once_with()

    def test_external_stop_closes_panel_and_unexpected_tk_failure_sets_stop(self):
        panel = fake_panel()
        panel.stop_event.set()
        panel.poll()
        panel.root.destroy.assert_called_once_with()
        broken = fake_panel()
        broken.root.update.side_effect = broken._tk.TclError("gone")
        with self.assertRaisesRegex(RuntimeError, "closed unexpectedly"):
            broken.poll()
        self.assertTrue(broken.stop_event.is_set())

    def test_research_apply_sends_all_path_fields_and_active_locks_path(self):
        panel = research_panel()
        panel._path.set("line")
        panel._apply()
        payload = panel.send_command.call_args.args[0]
        self.assertEqual(payload, {"type": "target", "gap": .3, "path": "line",
                                  "path_delta": [0., 0., -.05, 0., 0., 0.],
                                  "path_radius": .02, "path_duration": 10.})
        panel._status["active"] = True
        panel._refresh_enabled()
        for widget in panel._editable:
            widget.configure.assert_called_with(state="disabled")
        panel._path_selector.configure.assert_called_with(state="disabled")
        panel._pause_button.configure.assert_called_with(state="normal", text="Pause task")
        panel._status["active"] = False
        panel._refresh_enabled()
        panel._path_selector.configure.assert_called_with(state="readonly")

    def test_pause_and_resume_do_not_stop_compensation_or_close_window(self):
        panel = research_panel()
        panel._toggle_task_pause()
        panel.send_command.assert_not_called()
        panel._status["active"] = True
        panel._toggle_task_pause()
        panel.send_command.assert_called_once_with("pause_task")
        self.assertTrue(panel._task_paused)
        self.assertFalse(panel.stop_event.is_set())
        panel.root.destroy.assert_not_called()
        panel._toggle_task_pause()
        self.assertEqual(panel.send_command.call_args.args[0], "resume_task")
        self.assertFalse(panel._task_paused)
        self.assertFalse(panel.stop_event.is_set())

    def test_pause_queue_failure_does_not_change_button_state(self):
        panel = research_panel()
        panel._status["active"] = True
        panel.send_command.side_effect = RuntimeError("queue full")
        panel._toggle_task_pause()
        self.assertFalse(panel._task_paused)
        self.assertIn("queue full", panel._error.get())
        self.assertFalse(panel.stop_event.is_set())

    def test_research_status_formats_progress_age_and_authoritative_pause(self):
        panel = research_panel()
        panel.poll({"active": True, "progress": .4, "task_scale": .25,
                    "measurement_age": {"ur5e": .03, "ur10e": .08},
                    "fault_category": "measurement", "task_paused": True})
        text = panel._progress_message.get()
        self.assertIn("40.0%", text)
        self.assertIn("25.0%", text)
        self.assertIn("80.0 ms", text)
        self.assertIn("measurement", text)
        self.assertTrue(panel._task_paused)
        panel._pause_button.configure.assert_called_with(state="normal", text="Resume task")


class WorkerTests(unittest.TestCase):
    def test_research_panel_constructs_default_controls_without_a_display(self):
        error_type = type("FakeTkError", (Exception,), {})
        ttk = Mock()
        fake_tk = SimpleNamespace(Tk=Mock(return_value=Mock()), TclError=error_type, ttk=ttk,
                                  StringVar=lambda _master, value="": Variable(value))
        args = SimpleNamespace(mode="follow", execute=False, min_gap=.15, gap=.3, method="separated")
        send = Mock()
        with patch.dict(sys.modules, {"tkinter": fake_tk, "tkinter.ttk": ttk}):
            panel = gui.TargetPanel(args, send, threading.Event())
        panel._apply()
        request = send.call_args.args[0]
        self.assertEqual(request["path"], "hold")
        self.assertEqual(request["path_delta"], [0., 0., -.05, 0., 0., 0.])
        self.assertEqual(request["path_radius"], .02)
        self.assertEqual(request["path_duration"], 10.)
        panel._path_selector.configure.assert_called_with(state="readonly")
        panel.close()

    def test_worker_uses_latest_status_and_closes_on_stop(self):
        commands, statuses = Queue(maxsize=4), Queue()
        statuses.put({"message": "old"})
        statuses.put({"message": "latest"})
        stop = threading.Event()
        panel = Mock()
        panel.poll.side_effect = lambda status: stop.set()
        with patch.object(gui, "TargetPanel", return_value=panel) as factory:
            gui.gui_worker(SimpleNamespace(), commands, statuses, stop)
        panel.poll.assert_called_once_with({"message": "latest"})
        panel.close.assert_called_once_with()
        callback = factory.call_args.args[1]
        callback("arm")
        self.assertEqual(commands.get_nowait(), "arm")

    def test_worker_full_queue_does_not_drop_existing_request(self):
        commands, statuses = Queue(maxsize=1), Queue()
        original = {"type": "target", "gap": .3}
        commands.put(original)
        stop = threading.Event()
        panel = Mock()
        panel.poll.side_effect = lambda status: stop.set()
        with patch.object(gui, "TargetPanel", return_value=panel) as factory:
            gui.gui_worker(SimpleNamespace(), commands, statuses, stop)
        callback = factory.call_args.args[1]
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            callback("arm")
        self.assertEqual(commands.get_nowait(), original)

    def test_missing_display_fails_explicitly_without_constructing_widgets(self):
        error_type = type("FakeTkError", (Exception,), {})
        fake_tk = SimpleNamespace(Tk=Mock(side_effect=error_type("no display")), TclError=error_type,
                                  ttk=SimpleNamespace())
        with patch.dict(sys.modules, {"tkinter": fake_tk, "tkinter.ttk": fake_tk.ttk}):
            with self.assertRaisesRegex(RuntimeError, "graphical display"):
                gui.TargetPanel(SimpleNamespace(), Mock(), threading.Event())

    def test_missing_tk_dependency_has_an_explicit_error(self):
        with patch.dict(sys.modules, {"tkinter": None}):
            with self.assertRaisesRegex(RuntimeError, "requires Tkinter"):
                gui.TargetPanel(SimpleNamespace(), Mock(), threading.Event())


if __name__ == "__main__":
    unittest.main()
