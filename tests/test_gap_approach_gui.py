"""Flange approach UI requests and status, without Tk or hardware."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import urcal.control.gui as gui
from tests.test_dual_control_gui import Variable, research_panel


def approach_panel(*, simulation=False, execute=True):
    panel = research_panel()
    panel.args.execute = execute
    panel._simulation = simulation
    if simulation:
        panel._sim_pause_button = Mock()
    panel._approach_button = Mock()
    panel._approach_message = Variable()
    return panel


def status(state="idle", **changes):
    approach = {"state": state, "enabled": True, "target_gap_m": .15,
                "commanded_gap_m": .3, "estimated_gap_m": .314, "progress": 0., "reason": ""}
    approach.update(changes)
    return {"active": True, "phase": "tracking", "gap_approach": approach}


class GapApproachGuiTests(unittest.TestCase):
    def test_button_actions_follow_authoritative_state_and_use_timestamps(self):
        panel = approach_panel()
        for state, action, label in (("idle", "start", "Approach to 15 cm"),
                                     ("running", "pause", "Pause approach"),
                                     ("paused", "resume", "Resume approach")):
            with self.subTest(state=state):
                panel.poll(status(state))
                panel._approach_button.configure.assert_called_with(state="normal", text=label)
                before = deepcopy(panel._status)
                with patch.object(gui.time, "monotonic", return_value=123.4):
                    panel._request_gap_approach()
                panel.send_command.assert_called_with(
                    {"type": "gap_approach", "action": action, "requested_at": 123.4})
                self.assertEqual(panel._status, before)
                self.assertFalse(panel._task_paused)
                self.assertFalse(panel.stop_event.is_set())

    def test_repeated_click_before_status_repeats_action_instead_of_toggling(self):
        panel = approach_panel()
        panel.poll(status("idle"))
        with patch.object(gui.time, "monotonic", side_effect=[1., 1.1]):
            panel._request_gap_approach()
            panel._request_gap_approach()
        commands = [call.args[0] for call in panel.send_command.call_args_list]
        self.assertEqual([command["action"] for command in commands], ["start", "start"])
        self.assertEqual([command["requested_at"] for command in commands], [1., 1.1])

    def test_complete_reports_target_set_without_claiming_measured_arrival(self):
        panel = approach_panel()
        panel.poll(status("complete", commanded_gap_m=.15, estimated_gap_m=.167, progress=1.))
        panel._approach_button.configure.assert_called_with(state="disabled", text="15 cm target set")
        panel._request_gap_approach()
        panel.send_command.assert_not_called()
        self.assertIn("Commanded gap: 15.0 cm", panel._approach_message.get())
        self.assertIn("Estimated gap: 16.7 cm", panel._approach_message.get())
        self.assertIn("100.0%", panel._approach_message.get())

    def test_preview_or_inactive_session_cannot_issue_approach_even_if_enabled(self):
        for execute, active in ((False, True), (False, False), (True, False)):
            with self.subTest(execute=execute, active=active):
                panel = approach_panel(execute=execute)
                panel.poll({**status(), "active": active})
                panel._request_gap_approach()
                panel.send_command.assert_not_called()
                panel._approach_button.configure.assert_called_with(state="disabled", text="Approach to 15 cm")

    def test_controller_disabled_states_prevent_requests_and_show_reason(self):
        for phase in ("alignment", "alignment_calibration_handoff", "calibration_observation",
                      "recovery", "tracking"):
            with self.subTest(phase=phase):
                panel = approach_panel()
                panel.poll({**status("paused", enabled=False, reason="Waiting for tracking"), "phase": phase})
                panel._request_gap_approach()
                panel.send_command.assert_not_called()
                self.assertIn("Waiting for tracking", panel._approach_message.get())
                panel._approach_button.configure.assert_called_with(state="disabled", text="Resume approach")

    def test_authorized_pause_remains_available_while_controller_handles_observation(self):
        panel = approach_panel()
        panel.poll({**status("running"), "phase": "calibration_observation"})
        panel._request_gap_approach()
        self.assertEqual(panel.send_command.call_args.args[0]["action"], "pause")

    def test_temporary_coordinate_hold_shows_automatic_continuation_and_allows_operator_pause(self):
        panel = approach_panel()
        panel.poll(status("running", temporary_hold=True))
        self.assertIn("waiting for coordinates (automatic continuation)", panel._approach_message.get())
        panel._approach_button.configure.assert_called_with(state="normal", text="Pause approach")
        panel._request_gap_approach()
        self.assertEqual(panel.send_command.call_args.args[0]["action"], "pause")
        panel.poll(status("paused", temporary_hold=False))
        self.assertIn("paused (press Resume)", panel._approach_message.get())

    def test_simulation_tracking_can_approach_but_global_pause_blocks_request(self):
        panel = approach_panel(simulation=True, execute=False)
        panel.poll({**status(), "sim_paused": False})
        panel._request_gap_approach()
        self.assertEqual(panel.send_command.call_args.args[0]["action"], "start")
        panel.send_command.reset_mock()
        panel.poll({**status("running"), "sim_paused": True})
        panel._request_gap_approach()
        panel.send_command.assert_not_called()
        panel._approach_button.configure.assert_called_with(state="disabled", text="Pause approach")

    def test_queue_error_preserves_controller_state_and_other_controls(self):
        panel = approach_panel()
        panel.poll(status("running"))
        panel.send_command.side_effect = RuntimeError("command queue full")
        before = deepcopy(panel._status)
        panel._request_gap_approach()
        self.assertIn("queue full", panel._error.get())
        self.assertEqual(panel._status, before)
        self.assertFalse(panel._task_paused)
        self.assertFalse(panel.stop_event.is_set())
        panel.root.destroy.assert_not_called()

    def test_missing_or_unknown_state_is_disabled_and_bad_numbers_show_placeholders(self):
        for approach in (None, {}, {"enabled": True, "state": "unknown"}):
            with self.subTest(approach=approach):
                panel = approach_panel()
                panel.poll({"active": True, "gap_approach": approach})
                panel._request_gap_approach()
                panel.send_command.assert_not_called()
        text = gui.gap_approach_status_text({"gap_approach": {
            "commanded_gap_m": float("nan"), "estimated_gap_m": None,
            "progress": "invalid", "error": "Approach request expired"}})
        self.assertIn("Commanded gap: --", text)
        self.assertIn("Estimated gap: --", text)
        self.assertIn("Approach progress: --", text)
        self.assertIn("Approach request expired", text)


if __name__ == "__main__":
    unittest.main()
