"""Evaluation evidence around live alignment, using fake RTDE transports only."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests import test_live_alignment_runtime as fixture
from urcal.control.policy import ControlFault
import urcal.cooperative.runtime as runtime


class EvaluationAlignmentTests(unittest.TestCase):
    def run_recorded(self, *, transport=None, **kwargs):
        rows = []
        recorder = Mock()
        recorder.status.return_value = {"state": "recording", "active": True,
                                        "busy": True, "can_start": False, "can_stop": True}
        recorder.close.return_value = {"state": "saved", "csv_path": "evaluation.csv"}
        recorder.record.side_effect = lambda **row: rows.append(deepcopy(row))
        fake_run = fixture.LiveAlignmentRuntimeTests(methodName="runTest")
        with patch.object(runtime, "EvaluationRecorder", return_value=recorder), \
                patch.object(runtime, "transport_record", return_value=transport):
            result = fake_run.run_fake(**kwargs)
        return SimpleNamespace(result=result, recorder=recorder, rows=rows)

    def test_preflight_duration_does_not_make_evaluation_timestamps_reverse(self):
        observed = self.run_recorded(stop_after=1, preflight_seconds=.4)
        self.assertIsNone(observed.result.caught)
        self.assertEqual([row["phase"] for row in observed.rows[:2]],
                         ["alignment_preflight", "alignment_preflight"])
        times = [row["at"] for row in observed.rows]
        self.assertGreaterEqual(len(times), 3)
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(times[-1], 10.4)
        # Preflight reads newer encoders locally; the outer Arm snapshot must
        # retain its own older host time rather than borrowing those reads.
        final_control = observed.rows[-1]["control_record"]
        for name in runtime.ROBOTS:
            self.assertEqual(final_control["robots"][name]["robot_timestamp"], 60.)
            self.assertEqual(final_control["robot_state_host_times"][name], 10.)
        observed.result.session.send_joints.assert_not_called()
        observed.result.session.send.assert_not_called()
        observed.recorder.close.assert_called_once_with(reason="operator_stop")

    def test_speedj_alignment_retains_active_proposal_and_successful_transport(self):
        transport = {"operations": [{"robot": "ur5e", "stage": "speedJ", "returned": True,
                                     "finished_at": 10.04,
                                     "command": {"qd": [.001] * 6}}]}
        observed = self.run_recorded(stop_after=3, done_after=5, transport=transport)
        self.assertIsNone(observed.result.caught)
        sent = [row for row in observed.rows if row.get("control_record", {}).get("command_sent")]
        self.assertEqual(len(sent), 2)
        self.assertEqual(len(sent), observed.result.session.send_joints.call_count)
        for row in sent:
            control = row["control_record"]
            self.assertTrue(control["active"])
            self.assertEqual(row["phase"], "alignment")
            self.assertEqual(row["transport"], transport)
            self.assertGreaterEqual(row["at"], control["at"])
            self.assertEqual(set(control["robots"]), {"ur5e", "ur10e"})
            np.testing.assert_array_equal(control["joint_command"], [.001] * 6)
            np.testing.assert_array_equal(control["joint_commands"]["ur5e"], [.001] * 6)

    def test_failed_speedj_keeps_failed_transaction_in_final_fault_row(self):
        failure = ControlFault("speedJ rejected")
        transport = {"failed_at": 10.04, "error": str(failure), "operations": [
            {"robot": "ur5e", "stage": "speedJ", "returned": False,
             "finished_at": 10.04, "command": {"qd": [.001] * 6}}]}
        observed = self.run_recorded(send_error=failure, transport=transport)
        self.assertIs(observed.result.caught, failure)
        faults = [row for row in observed.rows if row["phase"] == "fault"]
        self.assertEqual(len(faults), 1)
        row = faults[0]
        self.assertIn("speedJ rejected", row["reason"])
        self.assertTrue(row["control_record"]["active"])
        self.assertFalse(row["control_record"]["command_sent"])
        self.assertEqual(row["transport"], transport)
        self.assertEqual(set(row["control_record"]["robots"]), {"ur5e", "ur10e"})
        observed.result.alignment.commit.assert_not_called()
        observed.result.session.send.assert_not_called()
        observed.recorder.close.assert_called_once()

    def test_queued_operator_stop_records_the_started_tick_before_closing(self):
        observed = self.run_recorded(commands=("stop",), stop_after=1)
        self.assertIsNone(observed.result.caught)
        self.assertEqual(len(observed.rows), 1)
        row = observed.rows[0]
        self.assertEqual(row["phase"], "stopped")
        self.assertEqual(row["reason"], "operator_stop")
        self.assertIsNone(row["scene"])
        self.assertIsNone(row["transport"])
        observed.result.motion_factory.assert_not_called()
        observed.recorder.close.assert_called_once_with(reason="operator_stop")


if __name__ == "__main__":
    unittest.main()
