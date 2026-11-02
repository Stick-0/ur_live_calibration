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
from urcal.control.geometry import facing_follower, pose_matrix


class Variable:
    def __init__(self, value=""):
        self.value = str(value)
        self.traces = []

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)
        for callback in self.traces:
            callback("variable", "", "write")

    def trace_add(self, mode, callback):
        assert mode == "write"
        self.traces.append(callback)
        return str(len(self.traces))


def fake_panel(mode="follow", execute=True):
    panel = object.__new__(gui.TargetPanel)
    panel.args = SimpleNamespace(mode=mode, execute=execute, min_gap=.15)
    panel.mode = mode
    panel.send_command = Mock()
    panel.stop_event = threading.Event()
    panel._status = {"active": False, "ready": False}
    panel._gap = Variable(30.)
    panel._message = Variable()
    panel._error = Variable()
    panel._editable = [Mock(), Mock()]
    panel._apply_button = Mock()
    panel._arm_button = Mock()
    panel._closed = False
    panel._draft_dirty = False
    panel._view = None
    panel.root = Mock()
    panel._tk = SimpleNamespace(TclError=type("FakeTkError", (Exception,), {}))
    panel._gap.trace_add("write", panel._on_target_edit)
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
    for variable in [panel._path, *panel._delta_vars, *panel._delta_rotation_vars, panel._radius, panel._duration]:
        variable.trace_add("write", panel._on_target_edit)
    return panel


def preview_scene():
    return {"robots": {
        "ur5e": {"flange_C": pose_matrix([-.3, .1, 1.4, .1, .2, .3])},
        "ur10e": {"flange_C": pose_matrix([.3, -.1, 1.5, -.2, .3, -.4])},
    }}


def attach_view(panel):
    view = SimpleNamespace(scene=preview_scene(), _draft={}, _draft_path={},
                           set_snapshot=Mock(), refresh=Mock())

    def set_draft(goals=None, path=None):
        view._draft = {} if goals is None else goals
        view._draft_path = {} if path is None else path

    view.set_draft = Mock(side_effect=set_draft)
    panel._view = view
    return view


class TargetConversionTests(unittest.TestCase):

    def test_follow_only_sends_gap(self):
        self.assertEqual(gui.build_target_command("follow", 30, .15), {"type": "target", "gap": .3})

    def test_invalid_gap_and_removed_modes_are_rejected(self):
        for gap in (14.9, 0, -2, "", "bad", np.nan, np.inf):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                gui.build_target_command("follow", gap, .15)
        for mode in ("hold", "invalid"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "only follower"):
                gui.build_target_command(mode, 30, .15)

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

    def test_follow_does_not_seed_or_send_fixed_center(self):
        panel = fake_panel(mode="follow")
        panel.poll({"center_pose": [1, 2, 3, 0, 0, 0]})
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


class DraftPreviewTests(unittest.TestCase):

    def test_follow_hold_draft_uses_live_leader_and_explicit_role(self):
        scene = preview_scene()
        for leader, follower in (("ur10e", "ur5e"), ("ur5e", "ur10e")):
            goals, path = gui.draft_geometry(scene, {"gap": .4, "path": "hold"}, "follow", leader)
            self.assertEqual(set(goals), {follower})
            np.testing.assert_allclose(goals[follower], facing_follower(scene["robots"][leader]["flange_C"], .4))
            self.assertEqual(path, {})

    def test_research_circle_draft_has_correct_leader_frame_radius(self):
        scene = preview_scene()
        request = gui.build_target_command("follow", 30, .15, path="circle", path_radius_cm=4)
        goals, paths = gui.draft_geometry(scene, request, "follow")
        leader = scene["robots"]["ur10e"]["flange_C"]
        start = facing_follower(leader, .3)
        np.testing.assert_allclose(paths["ur5e"][0], start[:3, 3])
        np.testing.assert_allclose(paths["ur5e"][-1], start[:3, 3], atol=1e-15)
        np.testing.assert_allclose(paths["ur5e"][24], start[:3, 3] - .08 * leader[:3, 0], atol=1e-15)
        np.testing.assert_allclose(goals["ur5e"], start)

    def test_line_endpoint_draft_shows_translation_and_rotation_vector_edit(self):
        scene = preview_scene()
        request = gui.build_target_command("follow", 35, .15, path="line",
                                           path_delta_cm=[3, -4, -5], path_rotation_degrees=[20, -30, 40])
        goals, paths = gui.draft_geometry(scene, request, "follow")
        leader = scene["robots"]["ur10e"]["flange_C"]
        expected_position = leader[:3, 3] + leader[:3, :3] @ [.03, -.04, .30]
        expected_rotation = (leader[:3, :3]
                             @ Rotation.from_rotvec(np.deg2rad([20, -30, 40])).as_matrix()
                             @ np.diag([1., -1., -1.]))
        np.testing.assert_allclose(goals["ur5e"][:3, 3], expected_position)
        np.testing.assert_allclose(goals["ur5e"][:3, :3], expected_rotation, atol=1e-15)
        np.testing.assert_allclose(paths["ur5e"][-1], expected_position)

    def test_unapplied_edits_disable_arm_until_apply(self):
        panel = fake_panel()
        attach_view(panel)
        panel._status["ready"] = True
        panel._refresh_enabled()
        panel._arm_button.configure.assert_called_with(state="normal")
        panel._gap.set("34")
        panel._arm_button.configure.assert_called_with(state="disabled")
        panel._arm()
        panel.send_command.assert_not_called()
        panel.poll({"ready": True})
        panel._arm_button.configure.assert_called_with(state="disabled")
        panel._apply()
        panel.send_command.assert_called_once()
        self.assertEqual(panel.send_command.call_args.args[0]["type"], "target")
        self.assertFalse(panel._draft_dirty)

    def test_numeric_edits_only_update_local_draft_and_apply_submits_once(self):
        panel = fake_panel()
        view = attach_view(panel)
        panel._gap.set("40")
        panel.poll()
        self.assertTrue(panel._draft_dirty)
        self.assertEqual(set(view._draft), {"ur5e"})
        expected = facing_follower(view.scene["robots"]["ur10e"]["flange_C"], .4)
        np.testing.assert_allclose(view._draft["ur5e"], expected)
        panel.send_command.assert_not_called()
        panel._apply()
        panel.send_command.assert_called_once()
        self.assertEqual(panel.send_command.call_args.args[0]["type"], "target")
        self.assertFalse(panel._draft_dirty)
        self.assertEqual(view._draft, {})
        panel.poll()
        panel.send_command.assert_called_once()

    def test_incomplete_numeric_edit_clears_previous_draft_without_sending(self):
        panel = fake_panel()
        view = attach_view(panel)
        panel._gap.set("35")
        panel.poll()
        self.assertTrue(view._draft)
        panel._gap.set("")
        panel.poll()
        self.assertEqual(view._draft, {})
        self.assertEqual(view._draft_path, {})
        panel.send_command.assert_not_called()
        panel._apply()
        panel.send_command.assert_not_called()

    def test_active_status_removes_draft_and_never_transmits_edits(self):
        panel = fake_panel()
        view = attach_view(panel)
        panel._gap.set("35")
        panel.poll()
        self.assertTrue(view._draft)
        panel.poll({"active": True, "scene": {"new": "snapshot"}})
        self.assertEqual(view._draft, {})
        view.set_snapshot.assert_called_once_with({"new": "snapshot"})
        panel._apply()
        panel.send_command.assert_not_called()

    def test_status_does_not_overwrite_manual_fields_and_none_scene_is_forwarded(self):
        panel = fake_panel()
        view = attach_view(panel)
        panel._gap.set("35")
        panel.poll({"center_pose": [8, 9, 10, 0, 0, 0], "scene": None})
        self.assertEqual(panel._gap.get(), "35")
        view.set_snapshot.assert_called_once_with(None)
        panel.send_command.assert_not_called()

    def test_research_path_edit_draws_locally_and_queue_failure_keeps_draft(self):
        panel = research_panel()
        view = attach_view(panel)
        panel._path.set("line")
        panel._delta_vars[0].set("7")
        panel._delta_rotation_vars[2].set("90")
        panel.poll()
        self.assertEqual(view._draft_path["ur5e"].shape, (49, 3))
        leader = view.scene["robots"]["ur10e"]["flange_C"]
        expected_delta = leader[:3, :3] @ [.07, 0, -.05]
        np.testing.assert_allclose(view._draft_path["ur5e"][-1] - view._draft_path["ur5e"][0], expected_delta)
        np.testing.assert_allclose(view._draft["ur5e"][:3, :3],
                                   leader[:3, :3] @ Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix()
                                   @ np.diag([1., -1., -1.]), atol=1e-15)
        panel.send_command.assert_not_called()
        panel.send_command.side_effect = RuntimeError("queue full")
        panel._apply()
        self.assertTrue(panel._draft_dirty)
        self.assertTrue(view._draft)
        self.assertIn("queue full", panel._error.get())


class WorkerTests(unittest.TestCase):
    def test_research_panel_constructs_default_controls_without_a_display(self):
        error_type = type("FakeTkError", (Exception,), {})
        ttk = Mock()
        fake_tk = SimpleNamespace(Tk=Mock(return_value=Mock()), Canvas=Mock(), TclError=error_type, ttk=ttk,
                                  StringVar=lambda _master, value="": Variable(value))
        fake_tk.Tk.return_value.winfo_screenwidth.return_value = 1280
        fake_tk.Tk.return_value.winfo_screenheight.return_value = 800
        args = SimpleNamespace(mode="follow", execute=False, min_gap=.15, gap=.3, method="separated")
        send = Mock()
        with patch.dict(sys.modules, {"tkinter": fake_tk, "tkinter.ttk": ttk}), \
                patch.object(gui, "RobotSceneView"), patch.object(gui, "PositionErrorWindow") as plot_factory:
            panel = gui.TargetPanel(args, send, threading.Event())
        plot_factory.assert_called_once_with(panel.root, follower="ur5e", simulation=False)
        panel._show_error_plot()
        plot_factory.return_value.show.assert_called_once_with()
        panel._apply()
        request = send.call_args.args[0]
        self.assertEqual(request["path"], "hold")
        self.assertEqual(request["path_delta"], [0., 0., -.05, 0., 0., 0.])
        self.assertEqual(request["path_radius"], .02)
        self.assertEqual(request["path_duration"], 10.)
        panel._path_selector.configure.assert_called_with(state="readonly")
        panel.root.geometry.assert_called_once_with("1240x720")
        panel.close()

    def test_position_plot_only_receives_new_telemetry_and_refreshes_without_status(self):
        panel = fake_panel()
        panel._error_plot = Mock()
        sample = {"valid": True, "at": 3., "error_m": [.01, -.02, .03]}
        panel.poll({"position_error": sample})
        panel._error_plot.update.assert_called_once_with(sample)
        panel.poll()
        panel._error_plot.update.assert_called_once_with(sample)
        self.assertEqual(panel._error_plot.refresh.call_count, 2)
        # A new status lacking telemetry must not replay the merged old value.
        panel.poll({"message": "Waiting"})
        panel._error_plot.update.assert_called_with(None)
        panel.send_command.assert_not_called()
        self.assertFalse(panel.stop_event.is_set())

    def test_recording_status_updates_without_a_new_error_sample_or_robot_request(self):
        panel = fake_panel()
        panel._error_plot = Mock()
        status = {"state": "saving", "path": "/tmp/errors.csv", "request_id": "save"}
        panel.poll({"error_recording": status})
        panel._error_plot.set_recording_status.assert_called_once_with(status)
        panel._error_plot.update.assert_called_once_with(None)
        panel.poll()
        panel._error_plot.set_recording_status.assert_called_once_with(status)
        panel.send_command.assert_not_called()
        self.assertFalse(panel.stop_event.is_set())

    def test_graph_receives_only_the_separate_recording_callback(self):
        panel = fake_panel()
        panel._recording_command = Mock()
        with patch.object(gui, "PositionErrorWindow") as factory:
            panel._show_error_plot()
        self.assertIs(factory.call_args.kwargs["recording_command"], panel._recording_command)
        panel.send_command.assert_not_called()

    def test_position_plot_tk_failure_does_not_stop_or_send_robot_commands(self):
        panel = fake_panel()
        panel._error_plot = Mock()
        panel._error_plot.refresh.side_effect = panel._tk.TclError("display unavailable")
        panel.poll({"message": "Tracking"})
        self.assertIn("Position error graph unavailable", panel._error.get())
        self.assertFalse(panel.stop_event.is_set())
        panel.send_command.assert_not_called()
        panel.root.update.assert_called_once_with()

    def test_position_plot_open_failure_does_not_stop_or_send_robot_commands(self):
        panel = fake_panel()
        with patch.object(gui, "PositionErrorWindow", side_effect=panel._tk.TclError("display unavailable")):
            panel._show_error_plot()
        self.assertIn("Position error graph unavailable", panel._error.get())
        self.assertFalse(panel.stop_event.is_set())
        panel.send_command.assert_not_called()

    def test_worker_uses_latest_status_and_closes_on_stop(self):
        for simulation in (False, True):
            with self.subTest(simulation=simulation):
                commands, statuses = Queue(maxsize=4), Queue()
                statuses.put({"message": "old"})
                statuses.put({"message": "latest"})
                stop = threading.Event()
                panel = Mock()
                panel.poll.side_effect = lambda status: stop.set()
                with patch.object(gui, "TargetPanel", return_value=panel) as factory:
                    gui.gui_worker(SimpleNamespace(simulate=simulation), commands, statuses, stop)
                panel.poll.assert_called_once_with({"message": "latest"})
                panel.close.assert_called_once_with()
                callback = factory.call_args.args[1]
                with patch.object(gui.time, "monotonic", return_value=12.3):
                    callback("arm")
                self.assertEqual(commands.get_nowait(), "arm" if simulation else ("arm", 12.3))

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

    def test_worker_recording_requests_use_separate_queue_and_preserve_motion_queue(self):
        commands, statuses, recordings = Queue(maxsize=1), Queue(), Queue(maxsize=1)
        original = {"type": "target", "gap": .3}
        commands.put(original)
        stop = threading.Event()
        panel = Mock()
        panel.poll.side_effect = lambda status: stop.set()
        with patch.object(gui, "TargetPanel", return_value=panel) as factory:
            gui.gui_worker(SimpleNamespace(), commands, statuses, stop, recordings)
        callback = factory.call_args.kwargs["recording_command"]
        start = {"action": "start", "request_id": "one"}
        callback(start)
        with self.assertRaisesRegex(RuntimeError, "Recording request queue is full"):
            callback({"action": "stop", "request_id": "two"})
        self.assertEqual(recordings.get_nowait(), start)
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
