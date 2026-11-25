"""Live evaluation lifecycle and failure evidence without hardware access."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import urcal.cooperative.runtime as runtime
from urcal.control.evaluation import EvaluationRecorder
from urcal.cooperative.logging import ExperimentLogger
import test_cooperative_runtime as fixtures


class RecorderSpy:
    def __init__(self, *_args, **kwargs):
        self.settings = kwargs["settings"]
        self.requests, self.records = [], []
        self.active, self.closed = False, False

    def request(self, command, **metadata):
        self.requests.append(deepcopy({"command": command, **metadata}))
        if command["action"] == "start":
            self.active = True
        elif command["action"] == "stop":
            self.active = False
        return True

    def record(self, **record):
        if self.active:
            self.records.append(deepcopy(record))
        return self.active

    def status(self):
        return {"state": "recording" if self.active else "idle", "active": self.active}

    def close(self, reason="shutdown", **_kwargs):
        self.close_reason, self.closed, self.active = reason, True, False
        return self.status()


START = {"type": "evaluation", "action": "start", "request_id": "start", "trial_id": "trial-01"}


class EvaluationRuntimeTests(unittest.TestCase):
    def run_fixture(self, **kwargs):
        self.recorder = None

        def create(*args, **settings):
            self.recorder = RecorderSpy(*args, **settings)
            return self.recorder

        with patch.object(runtime, "EvaluationRecorder", side_effect=create):
            result = fixtures.ResearchRuntimeTests().mocked_run(
                initial_recording_commands=(deepcopy(START),), **kwargs)
        self.assertTrue(self.recorder.closed)
        return result

    def test_preview_records_every_tick_without_arming_or_sending(self):
        result = self.run_fixture(initial_commands=(), stop_after=4)
        self.assertEqual(len(self.recorder.records), 4)
        result.motion_factory.assert_not_called()
        for row in self.recorder.records:
            self.assertIsNone(row["transport"])
            self.assertFalse(row["control_record"]["command_sent"])
            self.assertIn("robot_state_host_times", row["control_record"])
            self.assertEqual(set(row["control_record"]["robots"]), {"ur5e", "ur10e"})
            self.assertIn("tcp_speed", row["control_record"]["robots"]["ur5e"])
        result.recorder.request.assert_not_called()
        self.assertIn("evaluation_recording", result.gui_statuses.get_nowait())

    def test_start_freezes_nominal_goal_and_recording_actions_use_separate_channel(self):
        stop = {"type": "evaluation", "action": "stop", "request_id": "stop"}
        table = {"type": "evaluation", "action": "table_start", "request_id": "table"}
        old = {"action": "start", "request_id": "old"}
        result = self.run_fixture(execute=True, stop_after=4,
                                  later_recording_commands={1: (table, old), 3: (stop,)})
        self.assertEqual(len(self.recorder.records), 3)
        start = self.recorder.requests[0]
        np.testing.assert_array_equal(start["target"], result.engine.path.pose(0.))
        self.assertGreater(result.engine.progress, 0.)
        self.assertEqual([r["command"]["action"] for r in self.recorder.requests],
                         ["start", "table_start", "stop"])
        result.recorder.request.assert_called_once_with(old)

    def test_actual_transport_only_attached_to_cycle_that_attempted_send(self):
        transaction = {"operations": [{"robot": "ur5e", "stage": "speedL", "returned": True,
                                        "finished_at": 10.04, "command": {"velocity": [.001] * 6}}]}
        with patch.object(runtime, "transport_record", return_value=transaction):
            self.run_fixture(execute=True, stop_after=2)
        self.assertIsNone(self.recorder.records[0]["transport"])
        self.assertEqual(self.recorder.records[1]["transport"], transaction)

    def test_sdk_exception_preserves_current_proposal_transport_and_failure_row(self):
        transaction = {"operations": [{"robot": "ur5e", "stage": "speedL",
                                        "exception_type": "RuntimeError", "command": {"velocity": [.001] * 6}}]}

        def fail(_clock, _event):
            raise RuntimeError("injected SDK failure")

        with patch.object(runtime, "transport_record", return_value=transaction):
            self.run_fixture(execute=True, send_hook=fail, expected=RuntimeError)
        self.assertEqual(len(self.recorder.records), 2)
        failure = self.recorder.records[-1]
        self.assertEqual(failure["phase"], "fault")
        self.assertIn("injected SDK failure", failure["reason"])
        self.assertIsNone(failure["scene"])
        self.assertEqual(failure["transport"], transaction)
        self.assertFalse(failure["control_record"]["command_sent"])
        self.assertIn("robots", failure["control_record"])
        self.assertEqual(self.recorder.close_reason, "control_fault")

    def test_stop_command_does_not_erase_pending_cycle(self):
        self.run_fixture(initial_commands=(), later_commands={1: ("stop",)}, stop_after=4)
        self.assertEqual(len(self.recorder.records), 2)
        self.assertEqual(self.recorder.records[-1]["phase"], "stopped")
        self.assertIsNone(self.recorder.records[-1]["transport"])

    def test_emit_time_includes_sdk_latency_but_source_state_time_is_unchanged(self):
        def delayed(clock, _event):
            clock.now += .02

        self.run_fixture(execute=True, send_hook=delayed, stop_after=2)
        row = self.recorder.records[-1]
        self.assertAlmostEqual(row["at"] - row["control_record"]["at"], .02)
        self.assertAlmostEqual(row["control_record"]["robot_state_host_times"]["ur5e"], 10.04)

    def test_real_live_recorder_exports_terminal_sdk_failure_and_original_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "live.jsonl"
            created = []
            with ExperimentLogger(raw) as logger:
                def create(_logger, _raw, _directory, **kwargs):
                    recorder = EvaluationRecorder(logger, raw, Path(directory) / "exports", **kwargs)
                    created.append(recorder)
                    return recorder

                def fail(_clock, _event):
                    raise RuntimeError("SDK failed after evaluation start")

                transport = {"failed_at": 10.04, "operations": [
                    {"robot": "ur5e", "stage": "speedL", "exception_type": "RuntimeError",
                     "started_at": 10.04, "finished_at": 10.04,
                     "command": {"velocity": [.01] * 6, "acceleration": .1, "duration_s": .002}}]}
                with patch.object(runtime, "EvaluationRecorder", side_effect=create), \
                        patch.object(runtime, "transport_record", return_value=transport):
                    fixtures.ResearchRuntimeTests().mocked_run(
                        execute=True, initial_recording_commands=(START,), send_hook=fail, expected=RuntimeError)
            status = created[0].status()
            self.assertEqual(status["state"], "saved", status)
            with Path(status["csv_path"]).open(newline="") as source:
                rows = list(csv.DictReader(source))
            cycles = [row for row in rows if row["kind"] == "evaluation_cycle"]
            self.assertEqual(len(cycles), 2)
            self.assertEqual(cycles[-1]["phase"], "fault")
            self.assertEqual(cycles[-1]["command_sent"], "0")
            self.assertEqual(cycles[-1]["command_attempted"], "1")
            self.assertEqual(cycles[-1]["transport_failed"], "1")
            self.assertEqual(cycles[-1]["final_sent_velocity_0"], "")
            self.assertIn("SDK failed", cycles[-1]["reason"])
            records = [json.loads(line) for line in raw.read_text().splitlines()]
            self.assertEqual(records[0]["trial_id"], START["trial_id"])
            self.assertIn("path", records[0]["runtime_context"])
            self.assertEqual(records[-1]["kind"], "evaluation_stop")
            self.assertEqual(records[-1]["reason"], "control_fault")

    def test_recovery_snapshot_does_not_refresh_preserved_robot_sample_times(self):
        previous = {"robot_state_host_times": {"ur5e": 10., "ur10e": 10.}}
        readers = {name: type("Reader", (), {"last_advance": 11.})() for name in previous["robot_state_host_times"]}
        result = runtime.evaluation_control_record(previous, None, readers)
        self.assertEqual(result["robot_state_host_times"], previous["robot_state_host_times"])

    def test_shutdown_observation_keeps_stop_attempt_without_replaying_velocity(self):
        cleanup = {"mode": "cleanup", "operations": [
            {"stage": "speedStop", "robot": "ur5e", "returned": True,
             "command": {"acceleration": .3, "motion_mode": "linear"}}]}

        def attach_cleanup(_clock, _event):
            runtime.MotionSession.return_value.cleanup_transport_diagnostics.return_value = cleanup

        self.run_fixture(execute=True, send_hook=attach_cleanup, stop_after=2)
        self.assertEqual(len(self.recorder.records), 3)
        final = self.recorder.records[-1]
        self.assertEqual(final["phase"], "shutdown")
        self.assertEqual(final["control_record"]["stop_transport"], cleanup)
        self.assertIsNone(final["transport"])
        self.assertIsNone(final["scene"])

    def test_scene_keeps_calibration_age_epoch_separate_from_encoder_epoch(self):
        result = self.run_fixture(initial_commands=(), stop_after=1)
        readers = {name: fixtures.Reader() for name in runtime.ROBOTS}
        states = {name: reader.read(10.) for name, reader in readers.items()}
        scene = runtime.cooperative_gui_scene(10.1, result.engine, states, readers, {}, .15)
        self.assertEqual(scene["at"], 10.)
        self.assertEqual(scene["evaluated_at"], 10.1)


if __name__ == "__main__":
    unittest.main()
