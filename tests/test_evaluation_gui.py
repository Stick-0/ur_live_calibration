"""Data-only evaluation requests and recorder acknowledgements without Tk IO."""

import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from urcal.control import gui
from urcal.control.evaluation_panel import EvaluationPanel


class Variable:
    def __init__(self, _parent=None, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)

    def trace_add(self, *_args):
        return "trace"


class Widget:
    def __init__(self, _parent=None, **options):
        self.options = dict(options)

    def grid(self, **_kwargs):
        pass

    def columnconfigure(self, *_args, **_kwargs):
        pass

    def configure(self, **options):
        self.options.update(options)

    def invoke(self):
        # Test callbacks themselves as well as the visual disabled state.
        self.options["command"]()


class EvaluationPanelTests(unittest.TestCase):
    def setUp(self):
        self.labels = []

        def label(parent, **options):
            self.labels.append(options.get("text", ""))
            return Widget(parent, **options)

        ttk = SimpleNamespace(Frame=Widget, Label=label, Entry=Widget, Button=Widget)
        tk = SimpleNamespace(StringVar=Variable, ttk=ttk)
        self.modules = patch.dict(sys.modules, {"tkinter": tk, "tkinter.ttk": ttk})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.send = Mock()
        self.panel = EvaluationPanel(object(), recording_command=self.send)

    def status(self, state="idle", **changes):
        packet = {"state": state, "active": state == "recording", "busy": state in ("recording", "saving"),
                  "can_start": state in ("idle", "saved"), "can_stop": state == "recording",
                  "table_moving": False}
        packet.update(changes)
        self.panel.set_status(packet)
        return packet

    def assert_disabled(self, *widgets):
        for widget in widgets:
            self.assertEqual(widget.options["state"], "disabled")

    def test_labels_explain_fixed_target_and_data_only_annotations(self):
        labels = "\n".join(self.labels)
        self.assertIn("Trial ID", labels)
        self.assertIn("frozen at evaluation start", labels)
        self.assertIn("nominal relative flange target", labels)
        self.assertIn("only record data and mark table movement", labels)
        self.assertEqual(self.panel._table_start_button.options["text"], "Table movement start")
        self.assertEqual(self.panel._table_stop_button.options["text"], "Table movement end")

    def test_no_producer_or_no_authoritative_status_cannot_start(self):
        self.panel._start_button.invoke()
        self.send.assert_not_called()
        self.assertIn("Waiting for", self.panel._status_text.get())
        unavailable = EvaluationPanel(object())
        unavailable.set_status({"state": "idle", "can_start": True})
        unavailable._request("start")
        self.assertIn("unavailable", unavailable._status_text.get())
        self.assert_disabled(unavailable._start_button, unavailable._stop_button,
                             unavailable._table_start_button, unavailable._table_stop_button)

    def test_start_uses_evaluation_type_uuid_wall_clock_and_trial_label(self):
        self.status()
        self.panel._trial_id.set("  moving-table-03  ")
        with patch("urcal.control.evaluation_panel.time.monotonic", return_value=42.5):
            self.panel._start_button.invoke()
        request = self.send.call_args.args[0]
        self.assertEqual(request["type"], "evaluation")
        self.assertEqual(request["action"], "start")
        self.assertEqual(request["trial_id"], "moving-table-03")
        self.assertEqual(request["requested_at"], 42.5)
        self.assertEqual(uuid.UUID(hex=request["request_id"]).hex, request["request_id"])
        self.assert_disabled(self.panel._start_button, self.panel._stop_button, self.panel._trial_entry)

    def test_blank_trial_id_is_forwarded_for_producer_assignment(self):
        self.status()
        self.panel._start_button.invoke()
        self.assertEqual(self.send.call_args.args[0]["trial_id"], "")

    def test_pending_request_needs_matching_ack_and_blocks_repeated_actions(self):
        self.status()
        self.panel._request("start")
        request = self.send.call_args.args[0]
        self.status("recording", request_id="different-request")
        for action in ("start", "stop", "table_start", "table_stop"):
            self.panel._request(action)
        self.send.assert_called_once_with(request)
        self.assert_disabled(self.panel._stop_button, self.panel._table_start_button)
        self.status("recording", request_id=request["request_id"], session_id="session-A", trial_id="auto-trial")
        self.assertEqual(self.panel._stop_button.options["state"], "normal")
        self.assertEqual(self.panel._table_start_button.options["state"], "normal")
        self.assertIn("session-A", self.panel._status_text.get())
        self.assertIn("auto-trial", self.panel._status_text.get())

    def test_table_markers_require_recording_and_alternate_only_after_ack(self):
        self.status()
        self.panel._request("table_start")
        self.send.assert_not_called()
        self.status("recording")
        self.panel._table_start_button.invoke()
        start = self.send.call_args.args[0]
        self.assertEqual(start["action"], "table_start")
        self.assertNotIn("trial_id", start)
        self.panel._table_stop_button.invoke()
        self.send.assert_called_once()
        self.status("recording", request_id=start["request_id"], table_moving=True)
        self.assert_disabled(self.panel._table_start_button)
        self.assertEqual(self.panel._table_stop_button.options["state"], "normal")
        self.assertIn("Table movement marked active", self.panel._status_text.get())
        self.panel._table_stop_button.invoke()
        end = self.send.call_args.args[0]
        self.assertEqual(end["action"], "table_stop")
        self.assertNotEqual(start["request_id"], end["request_id"])
        self.status("recording", request_id=end["request_id"], table_moving=False)
        self.assertEqual(self.panel._table_start_button.options["state"], "normal")
        self.assert_disabled(self.panel._table_stop_button)

    def test_stop_and_export_waits_for_save_completion_and_displays_paths(self):
        self.status("recording", raw_path="/tmp/evaluation/raw.jsonl")
        self.panel._stop_button.invoke()
        request = self.send.call_args.args[0]
        self.assertEqual(request["action"], "stop")
        self.assertNotIn("trial_id", request)
        self.status("saving", request_id=request["request_id"], raw_path="/tmp/evaluation/raw.jsonl")
        self.assertIsNone(self.panel._pending)
        self.assertIn("Exporting evaluation", self.panel._status_text.get())
        self.assert_disabled(self.panel._start_button, self.panel._stop_button,
                             self.panel._table_start_button, self.panel._table_stop_button)
        self.status("saved", request_id=request["request_id"], raw_path="/tmp/evaluation/raw.jsonl",
                    csv_path="/tmp/evaluation/metrics.csv")
        self.assertIn("Evaluation exported", self.panel._status_text.get())
        self.assertIn("Raw: /tmp/evaluation/raw.jsonl", self.panel._status_text.get())
        self.assertIn("CSV: /tmp/evaluation/metrics.csv", self.panel._status_text.get())
        self.assertEqual(self.panel._start_button.options["state"], "normal")

    def test_busy_and_producer_capabilities_override_state_defaults(self):
        self.status(can_start=False)
        self.panel._request("start")
        self.status("recording", can_stop=False, busy=True)
        self.panel._request("stop")
        self.send.assert_not_called()
        self.status("recording", can_stop=True, busy=True)
        self.assertEqual(self.panel._stop_button.options["state"], "normal")
        self.assertEqual(self.panel._table_start_button.options["state"], "normal")
        self.status("error", active=False, busy=False, can_start=False, can_stop=False,
                    incomplete=True, error="writer disk full", raw_path="/tmp/partial.jsonl")
        self.assertIn("INCOMPLETE", self.panel._status_text.get())
        self.assertIn("writer disk full", self.panel._status_text.get())
        self.assertIn("/tmp/partial.jsonl", self.panel._status_text.get())
        self.assert_disabled(self.panel._start_button, self.panel._stop_button)
        self.status("error", can_start=True)
        self.assertEqual(self.panel._start_button.options["state"], "normal")

    def test_queue_failure_can_retry_without_faking_authoritative_state(self):
        self.status("recording", table_moving=False)
        self.send.side_effect = RuntimeError("Recording request queue is full; retry shortly")
        self.panel._table_start_button.invoke()
        self.assertIsNone(self.panel._pending)
        self.assertFalse(self.panel._status["table_moving"])
        self.assertIn("queue is full", self.panel._status_text.get())
        self.assertIn("EVALUATION RECORDING", self.panel._status_text.get())
        self.assertEqual(self.panel._table_start_button.options["state"], "normal")
        self.send.side_effect = None
        self.panel._table_start_button.invoke()
        self.assertEqual(self.send.call_count, 2)

    def test_invalid_status_is_ignored_and_status_dict_is_not_mutated(self):
        packet = self.status("recording", trial_id="trial-A")
        previous = packet.copy()
        self.panel.set_status(None)
        self.panel._request("unknown")
        self.send.assert_not_called()
        self.assertEqual(packet, previous)
        packet["trial_id"] = "changed-outside"
        self.assertEqual(self.panel._status["trial_id"], "trial-A")


class TargetEvaluationIntegrationTests(unittest.TestCase):
    def panel(self, simulation=False):
        root = Mock()
        root.winfo_screenwidth.return_value, root.winfo_screenheight.return_value = 1280, 800
        ttk = Mock()
        tk = SimpleNamespace(Tk=Mock(return_value=root), Canvas=Mock(), StringVar=Variable,
                             TclError=type("FakeTkError", (Exception,), {}), ttk=ttk)
        args = SimpleNamespace(mode="follow", execute=False, simulate=simulation,
                               min_gap=.15, gap=.2, method="separated")
        motion, recording = Mock(), Mock()
        with patch.dict(sys.modules, {"tkinter": tk, "tkinter.ttk": ttk}), \
                patch.object(gui, "RobotSceneView"), patch.object(gui, "PositionErrorWindow"), \
                patch.object(gui, "EvaluationPanel") as factory:
            panel = gui.TargetPanel(args, motion, threading.Event(), recording_command=recording)
        self.addCleanup(panel.close)
        return panel, factory, motion, recording

    def test_live_and_simulation_wire_only_recording_callback_and_forward_new_status(self):
        for simulation in (False, True):
            with self.subTest(simulation=simulation):
                panel, factory, motion, recording = self.panel(simulation)
                self.assertIs(factory.call_args.kwargs["recording_command"], recording)
                self.assertIsNot(factory.call_args.kwargs["recording_command"], motion)
                status = {"state": "recording", "trial_id": "trial-A", "can_stop": True}
                panel.poll({"evaluation_recording": status})
                factory.return_value.set_status.assert_called_once_with(status)
                panel.poll({"message": "Another controller update"})
                panel.poll()
                factory.return_value.set_status.assert_called_once_with(status)
                motion.assert_not_called()
                recording.assert_not_called()
                self.assertFalse(panel.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
