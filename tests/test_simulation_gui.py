"""Simulation UI and entry-point boundaries without Tk, devices, or sockets."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urcal.control.gui as gui
import urcal.control.runtime as runtime
import urcal.cooperative.runtime as cooperative_runtime


class Variable:
    def __init__(self, _master=None, value=""):
        self.value, self.callbacks = str(value), []

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)
        for callback in self.callbacks:
            callback("variable", "", "write")

    def trace_add(self, mode, callback):
        self.callbacks.append(callback)
        return str(len(self.callbacks))


class SimulationPanelTests(unittest.TestCase):
    def panel(self, mode="follow", simulate=True):
        extra = (["--simulate", "--initial_state", "saved.json"] if simulate else
                 ["--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2"])
        args = runtime.parse_args(["--mode", mode, *extra])
        root = Mock()
        root.winfo_screenwidth.return_value, root.winfo_screenheight.return_value = 1280, 800
        ttk = SimpleNamespace(**{name: Mock(side_effect=lambda *a, **kw: Mock()) for name in
                                  ("Frame", "Panedwindow", "Scrollbar", "Label", "Entry", "Button", "Combobox")})
        tk = SimpleNamespace(Tk=Mock(return_value=root), Canvas=Mock(), StringVar=Variable,
                             TclError=type("FakeTclError", (Exception,), {}), ttk=ttk)
        view = SimpleNamespace(frame=Mock(), scene=None, _draft={}, set_snapshot=Mock(),
                               set_draft=Mock(), refresh=Mock())
        send = Mock()
        with patch.dict(sys.modules, {"tkinter": tk, "tkinter.ttk": ttk}), \
                patch.object(gui, "RobotSceneView", return_value=view):
            panel = gui.TargetPanel(args, send, threading.Event())
        self.addCleanup(panel.close)
        return panel, ttk

    def test_simulation_constructor_labels_start_pause_reset_without_execute_notice(self):
        panel, ttk = self.panel()
        button_text = [call.kwargs.get("text", "") for call in ttk.Button.call_args_list]
        labels = [call.kwargs.get("text", "") for call in ttk.Label.call_args_list]
        for text in ("Start simulation", "Pause simulation", "Reset simulation"):
            self.assertIn(text, button_text)
        self.assertNotIn("Arm motion", button_text)
        self.assertTrue(any("No hardware commands" in text for text in labels))
        self.assertFalse(any("--execute is required" in text for text in labels))
        self.assertIn("SIMULATION", panel.root.title.call_args.args[0])
        self.assertFalse(panel.args.execute)

    def test_start_requires_ready_but_does_not_require_execute_in_both_modes(self):
        for mode in ("hold", "follow"):
            with self.subTest(mode=mode):
                panel, _ = self.panel(mode)
                panel._arm()
                panel.send_command.assert_not_called()
                panel.poll({"ready": True, "center_pose": [0, 0, 1, 0, 0, 0]})
                panel._arm_button.configure.assert_called_with(state="normal")
                panel._arm()
                panel.send_command.assert_called_once_with("arm")
                self.assertFalse(panel.args.execute)

    def test_unapplied_edits_block_simulation_start_until_apply(self):
        panel, _ = self.panel()
        panel.poll({"ready": True})
        panel._gap.set("32")
        panel._arm_button.configure.assert_called_with(state="disabled")
        panel._arm()
        panel.send_command.assert_not_called()
        panel._apply()
        request = panel.send_command.call_args.args[0]
        self.assertEqual(request["type"], "target")
        self.assertEqual(request["gap"], .32)
        panel.send_command.reset_mock()
        panel._arm()
        panel.send_command.assert_called_once_with("arm")

    def test_active_simulation_locks_target_and_start_while_enabling_pause(self):
        panel, _ = self.panel()
        panel.poll({"active": True, "ready": True})
        for widget in [*panel._editable, panel._apply_button, panel._arm_button]:
            widget.configure.assert_called_with(state="disabled")
        panel._sim_pause_button.configure.assert_called_with(state="normal", text="Pause simulation")
        panel._apply()
        panel._arm()
        panel.send_command.assert_not_called()

    def test_simulation_pause_and_resume_follow_authoritative_status(self):
        panel, _ = self.panel()
        panel._toggle_simulation_pause()
        panel.send_command.assert_not_called()
        panel.poll({"active": True, "sim_paused": False})
        panel._toggle_simulation_pause()
        panel.send_command.assert_called_once_with("sim_pause")
        self.assertFalse(panel._status["sim_paused"])
        panel.poll({"active": True, "sim_paused": True})
        panel._sim_pause_button.configure.assert_called_with(state="normal", text="Resume simulation")
        panel._toggle_simulation_pause()
        self.assertEqual(panel.send_command.call_count, 2)
        self.assertEqual(panel.send_command.call_args.args[0], "sim_pause")
        self.assertFalse(panel.stop_event.is_set())

    def test_paused_simulation_retains_target_lock_and_task_pause_is_separate(self):
        panel, _ = self.panel()
        panel.poll({"active": True, "sim_paused": True, "task_paused": False})
        panel._apply_button.configure.assert_called_with(state="disabled")
        panel._toggle_task_pause()
        panel.send_command.assert_called_once_with("pause_task")
        self.assertTrue(panel._task_paused)
        self.assertTrue(panel._status["sim_paused"])

    def test_reset_requests_restoration_without_closing_or_locally_advancing_state(self):
        panel, _ = self.panel()
        panel.poll({"active": True, "sim_paused": True})
        panel._reset_simulation()
        panel.send_command.assert_called_once_with("sim_reset")
        self.assertTrue(panel._status["active"])
        self.assertFalse(panel.stop_event.is_set())
        panel.root.destroy.assert_not_called()
        panel.poll({"active": False, "sim_paused": False, "ready": True})
        panel._apply_button.configure.assert_called_with(state="normal")
        panel._arm_button.configure.assert_called_with(state="normal")
        panel._sim_pause_button.configure.assert_called_with(state="disabled", text="Pause simulation")

    def test_failed_simulation_commands_show_error_and_do_not_change_run_state(self):
        for callback in ("_arm", "_toggle_simulation_pause", "_reset_simulation"):
            with self.subTest(callback=callback):
                panel, _ = self.panel()
                active = callback != "_arm"
                panel.poll({"active": active, "ready": True, "sim_paused": False})
                panel.send_command.side_effect = RuntimeError("simulation request failed")
                getattr(panel, callback)()
                self.assertIn("simulation request failed", panel._error.get())
                self.assertEqual(panel._status["active"], active)
                self.assertFalse(panel._status["sim_paused"])
                self.assertFalse(panel.stop_event.is_set())

    def test_receive_only_live_preview_cannot_start_or_send_simulation_commands(self):
        panel, ttk = self.panel(simulate=False)
        panel.poll({"ready": True})
        panel._arm_button.configure.assert_called_with(state="disabled")
        panel._arm()
        panel._toggle_simulation_pause()
        panel._reset_simulation()
        panel.send_command.assert_not_called()
        self.assertNotIn("Start simulation", [call.kwargs.get("text", "") for call in ttk.Button.call_args_list])


class SimulationEntryPointTests(unittest.TestCase):
    @staticmethod
    def saved(mode="follow", *extra):
        return ["--mode", mode, "--simulate", "--initial_state", "saved.json", *extra]

    def reject(self, arguments, reason):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            runtime.parse_args(arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(reason, stderr.getvalue())

    def test_saved_snapshot_allows_missing_robot_ips_and_never_sets_execute(self):
        for mode in ("hold", "follow"):
            args = runtime.parse_args(self.saved(mode))
            self.assertEqual(args.initial_state, Path("saved.json"))
            self.assertIsNone(args.ur5e_ip)
            self.assertIsNone(args.ur10e_ip)
            self.assertTrue(args.simulate)
            self.assertFalse(args.execute)

    def test_simulation_execute_and_non_simulation_saved_state_are_rejected(self):
        self.reject(self.saved("hold", "--execute"), "mutually exclusive")
        self.reject(["--mode", "hold", "--initial_state", "saved.json"], "requires --simulate")

    def test_live_capture_requires_both_robot_ips(self):
        for options in ([], ["--ur5e_ip", "192.0.2.1"], ["--ur10e_ip", "192.0.2.2"]):
            self.reject(["--mode", "hold", "--simulate", *options], "required for live capture")
        args = runtime.parse_args(["--mode", "hold", "--simulate", "--ur5e_ip", "192.0.2.1",
                                   "--ur10e_ip", "192.0.2.2"])
        self.assertIsNone(args.initial_state)
        self.assertFalse(args.execute)

    def test_virtual_external_motion_is_limited_to_compatible_modes(self):
        for motion in ("none", "leader", "bases", "both"):
            args = runtime.parse_args(self.saved("follow", "--sim_motion", motion))
            self.assertEqual(args.sim_motion, motion)
        for motion in ("none", "bases"):
            runtime.parse_args(self.saved("hold", "--sim_motion", motion))
        for motion in ("leader", "both"):
            self.reject(self.saved("hold", "--sim_motion", motion), "hold simulation commands both arms")
        self.reject(["--mode", "follow", "--sim_motion", "leader"], "requires --simulate")

    def test_initial_capture_timeout_rejects_nonpositive_and_nonfinite_values(self):
        for timeout in ("0", "-1", "nan", "inf"):
            self.reject(self.saved("hold", "--sim_initial_timeout", timeout), "finite and positive")

    def test_both_entry_points_dispatch_to_simulation_without_calling_live_runtimes(self):
        for entrypoint, argv, mode in (
            (runtime.main, self.saved("hold"), "hold"),
            (runtime.main, self.saved("follow"), "follow"),
            (cooperative_runtime.main, ["--simulate", "--initial_state", "saved.json"], "follow"),
        ):
            with self.subTest(entrypoint=entrypoint.__module__, mode=mode):
                simulation = SimpleNamespace(run=Mock())
                with patch.dict(sys.modules, {"urcal.simulation.runtime": simulation}), \
                        patch.object(runtime, "run") as live_hold, \
                        patch.object(cooperative_runtime, "run") as live_follow, \
                        patch.object(runtime, "MotionSession") as motion:
                    entrypoint(argv)
                simulation.run.assert_called_once()
                args = simulation.run.call_args.args[0]
                self.assertEqual(args.mode, mode)
                self.assertFalse(args.execute)
                live_hold.assert_not_called()
                live_follow.assert_not_called()
                motion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
