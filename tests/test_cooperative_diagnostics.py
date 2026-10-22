"""WAIT logging captures rejected tick states without flooding the log."""

import json
from types import SimpleNamespace
import unittest

import numpy as np

from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.diagnostics import WaitDiagnostics, fault_diagnostics


class WaitDiagnosticsTests(unittest.TestCase):
    def test_numeric_reason_changes_are_throttled_per_category(self):
        waits = WaitDiagnostics()
        fault = lambda value: ResearchFault("tracking_error", f"error={value:.3f}", {"position_error_m": value})
        records = [waits.record(index / 25., fault(index / 10.)) for index in range(76)]
        records = [record for record in records if record is not None]
        self.assertEqual([record["at"] for record in records], [0., 1., 2., 3.])
        self.assertEqual(records[-1]["details"]["position_error_m"], 7.5)
        self.assertEqual(records[-1]["reason"], "error=7.500")

    def test_category_change_recovery_and_clock_reset_emit_immediately(self):
        waits = WaitDiagnostics()
        tracking = ResearchFault("tracking_error", "pose too far")
        stale = ResearchFault("measurement_unavailable", "calibration stale")
        self.assertIsNotNone(waits.record(10., tracking))
        self.assertIsNone(waits.record(10.04, tracking))
        self.assertIsNotNone(waits.record(10.08, stale))
        self.assertIsNotNone(waits.record(10.12, tracking))
        self.assertIsNone(waits.record(10.16, tracking))
        waits.clear()
        self.assertIsNotNone(waits.record(10.20, tracking))
        self.assertIsNotNone(waits.record(9., tracking))

    def test_fault_states_are_current_independent_and_json_finite(self):
        tcp = np.eye(4)
        original_details = {"target_tcp": tcp.copy(), "position_error_m": .12}
        fault = ResearchFault("tracking_error", "pose too far", original_details)
        original_details["target_tcp"][0, 3] = 999.
        self.assertEqual(fault.details["target_tcp"][0, 3], 0.)
        states = {"ur5e": SimpleNamespace(joints=np.arange(6, dtype=float), joint_velocity=np.zeros(6),
                                         tcp=tcp, speed=np.array([np.nan, 0, 0, 0, 0, 0]), robot_timestamp=123.),
                  "ur10e": SimpleNamespace(joints=np.ones(6), joint_velocity=None, tcp=np.eye(4),
                                           speed=np.zeros(6), robot_timestamp=np.inf)}
        record = WaitDiagnostics().record(10., fault, states)
        self.assertEqual(record["robots"]["ur5e"]["robot_timestamp"], 123.)
        self.assertIsNone(record["robots"]["ur5e"]["tcp_speed"][0])
        self.assertIsNone(record["robots"]["ur10e"]["robot_timestamp"])
        self.assertIsNone(record["robots"]["ur10e"]["qd"])
        states["ur5e"].joints[0] = 888.
        states["ur5e"].tcp[0, 3] = 777.
        fault.details["target_tcp"][0, 3] = 666.
        self.assertEqual(record["robots"]["ur5e"]["q"][0], 0.)
        self.assertEqual(record["robots"]["ur5e"]["tcp"][0][3], 0.)
        self.assertEqual(record["details"]["target_tcp"][0][3], 0.)
        json.dumps(record, allow_nan=False)

    def test_missing_current_state_never_reuses_previous_tick(self):
        waits = WaitDiagnostics()
        fault = ResearchFault("tracking_error", "pose too far")
        self.assertEqual(waits.record(10., fault, {"ur5e": SimpleNamespace()})["robots"]["ur5e"]["q"], None)
        self.assertIsNone(waits.record(11., fault, states=None)["robots"])
        plain = fault_diagnostics(ValueError("bad input"))
        self.assertEqual(plain["details"], {})
        self.assertIsNone(plain["robots"])

    def test_invalid_throttle_parameters_are_rejected(self):
        for value in (0., -1., np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WaitDiagnostics(value)
        with self.assertRaises(ValueError):
            WaitDiagnostics().record(np.nan, ValueError("bad input"))


if __name__ == "__main__":
    unittest.main()
