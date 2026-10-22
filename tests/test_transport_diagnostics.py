"""Fault diagnostics must be passive, detached, and unable to hide the fault."""

from enum import IntEnum
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from urcal.control.transport_diagnostics import (
    capture_runtime_fault, control_snapshot, fault_transport_summary, receive_snapshot, transport_record,
)


class RobotMode(IntEnum):
    RUNNING = 7


class TransportDiagnosticsTests(unittest.TestCase):
    def test_control_reads_only_the_explicit_cached_getters(self):
        control = SimpleNamespace(
            isConnected=Mock(return_value=True),
            isProgramRunning=Mock(return_value=False),
            getRobotStatus=Mock(return_value=1),
            isProtectiveStopped=Mock(return_value=False),
            isEmergencyStopped=Mock(return_value=False),
            speedL=Mock(side_effect=AssertionError("motion")),
            stopScript=Mock(side_effect=AssertionError("script command")),
            reuploadScript=Mock(side_effect=AssertionError("restart")),
            kickWatchdog=Mock(side_effect=AssertionError("watchdog command")),
            getTCPOffset=Mock(side_effect=AssertionError("native command")),
            getInverseKinematics=Mock(side_effect=AssertionError("native query")),
        )
        result = control_snapshot(control, 15.)
        self.assertEqual(result, {
            "at": 15., "connected": True, "program_running": False,
            "robot_status": 1, "protective_stopped": False,
            "emergency_stopped": False, "getter_errors": {},
        })
        for name in ("isConnected", "isProgramRunning", "getRobotStatus",
                     "isProtectiveStopped", "isEmergencyStopped"):
            getattr(control, name).assert_called_once_with()
        for name in ("speedL", "stopScript", "reuploadScript", "kickWatchdog",
                     "getTCPOffset", "getInverseKinematics"):
            getattr(control, name).assert_not_called()

    def test_receive_preserves_each_status_without_cross_interface_inference(self):
        receiver = SimpleNamespace(
            isConnected=lambda: True, getTimestamp=lambda: 123.25,
            getRobotMode=lambda: RobotMode.RUNNING, getSafetyMode=lambda: 3,
            getRuntimeState=lambda: 0, getSafetyStatusBits=lambda: 4,
            getRobotStatus=lambda: 1,
        )
        result = receive_snapshot(receiver, 42.)
        self.assertEqual(result, {
            "at": 42., "connected": True, "robot_timestamp": 123.25,
            "robot_mode": 7, "safety_mode": 3, "runtime_state": 0,
            "safety_status_bits": 4, "robot_status": 1, "getter_errors": {},
        })
        self.assertIs(type(result["robot_mode"]), int)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_missing_and_failed_getters_preserve_other_values_and_original_fault(self):
        class BrokenControl:
            isEmergencyStopped = False

            def isConnected(self):
                return False

            def isProgramRunning(self):
                raise RuntimeError("receive socket closed")

            @property
            def getRobotStatus(self):
                raise ValueError("status attribute unavailable")

        original = RuntimeError("original control fault")
        try:
            raise original
        except RuntimeError as caught:
            result = control_snapshot(BrokenControl(), 12.)
            self.assertIs(caught, original)
            self.assertEqual(str(caught), "original control fault")
        self.assertFalse(result["connected"])
        self.assertIsNone(result["program_running"])
        self.assertIsNone(result["robot_status"])
        self.assertIsNone(result["protective_stopped"])
        self.assertIsNone(result["emergency_stopped"])
        self.assertEqual(set(result["getter_errors"]), {
            "program_running", "robot_status", "protective_stopped", "emergency_stopped",
        })
        self.assertIn("receive socket closed", result["getter_errors"]["program_running"])
        self.assertIn("unavailable", result["getter_errors"]["protective_stopped"])
        json.dumps(result, allow_nan=False)

    def test_nonfinite_and_nonscalar_values_are_null_with_field_errors(self):
        for invalid in (float("nan"), float("inf"), float("-inf"),
                        [1], {"status": 1}, "connected", object(),
                        np.array([1]), np.array(1.)):
            with self.subTest(value=repr(invalid)):
                receiver = SimpleNamespace(getRobotStatus=lambda: invalid)
                result = receive_snapshot(receiver, 1.)
                self.assertIsNone(result["robot_status"])
                self.assertIn("robot_status", result["getter_errors"])
                json.dumps(result, allow_nan=False)

    def test_numeric_scalar_values_are_copied_to_builtin_json_types(self):
        receiver = SimpleNamespace(
            isConnected=lambda: True, getTimestamp=lambda: np.float64(123.5),
            getRobotMode=lambda: np.int64(7), getSafetyMode=lambda: np.int32(1),
            getRuntimeState=lambda: None, getSafetyStatusBits=lambda: 1,
            getRobotStatus=lambda: 3,
        )
        result = receive_snapshot(receiver, np.float64(42.))
        self.assertEqual(result["getter_errors"], {})
        for field in ("at", "robot_timestamp"):
            self.assertIs(type(result[field]), float)
        for field in ("robot_mode", "safety_mode"):
            self.assertIs(type(result[field]), int)
        self.assertIsNone(result["runtime_state"])
        receiver.getRobotStatus = lambda: 0
        self.assertEqual(result["robot_status"], 3)
        json.dumps(result, allow_nan=False)

    def test_bad_error_string_and_invalid_time_do_not_hide_status(self):
        class BrokenError(RuntimeError):
            def __str__(self):
                raise RuntimeError("broken error string")

        control = SimpleNamespace(isConnected=Mock(side_effect=BrokenError()),
                                  getRobotStatus=lambda: 1)
        result = control_snapshot(control, float("nan"))
        self.assertIsNone(result["at"])
        self.assertEqual(result["robot_status"], 1)
        self.assertIn("error message unavailable", result["getter_errors"]["connected"])
        self.assertIn("at", result["getter_errors"])
        json.dumps(result, allow_nan=False)

    def test_runtime_capture_retains_original_details_and_earliest_status(self):
        error = RuntimeError("program stopped")
        error.details = {"detected_at": 10., "transport": {"failed_at": 10.}, "target": 3}
        receiver = SimpleNamespace(isConnected=lambda: True, getRuntimeState=lambda: 0)
        readers = {"ur5e": SimpleNamespace(receiver=receiver)}
        capture_runtime_fault(error, None, readers, 10.01)
        receiver.isConnected = lambda: False
        capture_runtime_fault(error, None, readers, 12.)
        self.assertEqual(error.details["detected_at"], 10.)
        self.assertEqual(error.details["transport"], {"failed_at": 10.})
        self.assertEqual(error.details["target"], 3)
        self.assertTrue(error.details["receive_status"]["ur5e"]["connected"])
        self.assertEqual(error.details["receive_status"]["ur5e"]["at"], 10.01)
        json.dumps(error.details, allow_nan=False)

    def test_runtime_capture_failure_cannot_replace_original_exception(self):
        original = RuntimeError("original fault")
        session = SimpleNamespace(capture_fault=Mock(side_effect=ValueError("bad status")))
        capture_runtime_fault(original, session, {"ur5e": SimpleNamespace()}, 10.)
        self.assertEqual(str(original), "original fault")
        self.assertEqual(original.details["detected_at"], 10.)
        self.assertIn("bad status", original.details["transport_snapshot_error"])
        self.assertIn("snapshot_error", original.details["receive_status"]["ur5e"])
        self.assertIsNone(transport_record(None))
        self.assertIsNone(transport_record(SimpleNamespace(transport_diagnostics=lambda: [])))
        self.assertIsNone(transport_record(SimpleNamespace(
            transport_diagnostics=Mock(side_effect=ValueError("bad snapshot")))))
        interrupted = RuntimeError("original program stop")
        receiver = SimpleNamespace(isConnected=Mock(side_effect=KeyboardInterrupt()))
        capture_runtime_fault(interrupted, None, {"ur5e": SimpleNamespace(receiver=receiver)}, 11.)
        self.assertEqual(str(interrupted), "original program stop")
        self.assertEqual(interrupted.details["detected_at"], 11.)

    def test_console_summary_uses_retained_status_without_inventing_a_cause(self):
        error = RuntimeError("stopped")
        error.details = {
            "transport": {"control_status": {"ur5e": {"connected": True, "program_running": False}},
                          "seconds_since_last_successful_send": {"ur5e": .091}},
            "receive_status": {"ur5e": {"safety_mode": 1, "runtime_state": 0}},
        }
        summary = fault_transport_summary(error)
        self.assertIn("connected=True, program_running=False", summary)
        self.assertIn("safety_mode=1, runtime_state=0", summary)
        self.assertIn("last_speed_send_age=91.0ms", summary)
        self.assertNotIn("watchdog", summary)
        self.assertEqual(fault_transport_summary(RuntimeError("no snapshot")), "")


if __name__ == "__main__":
    unittest.main()
