"""Pre-cleanup motion transport timing with SDK mocks and a virtual clock."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from urcal.control.policy import ControlFault
from urcal.control.runtime import MotionSession


METHODS = (
    "isConnected", "isProgramRunning", "getRobotStatus", "isProtectiveStopped", "isEmergencyStopped",
    "reuploadScript", "getTCPOffset", "setWatchdog", "kickWatchdog", "isPoseWithinSafetyLimits",
    "isJointsWithinSafetyLimits", "speedL", "speedJ", "speedStop", "stopScript", "disconnect",
)


class MotionTransportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.clock = patch("urcal.control.runtime.time.monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.args = SimpleNamespace(ur5e_ip="test5", ur10e_ip="test10", watchdog_hz=5.,
                                    max_accel=.1, max_angular_accel=.2, max_joint_speed=.3,
                                    max_joint_accel=.8, stop_accel=.3)
        self.names = ("ur5e", "ur10e")
        self.poses = {name: np.eye(4) for name in self.names}
        self.velocities = {name: np.zeros(6) for name in self.names}
        self.controls = {}
        for name in self.names:
            control = Mock(spec_set=METHODS)
            for method in METHODS:
                getattr(control, method).return_value = True
            control.getTCPOffset.return_value = [0.] * 6
            control.getRobotStatus.return_value = 3
            control.isProtectiveStopped.return_value = False
            control.isEmergencyStopped.return_value = False
            control.stopScript.return_value = None
            self.controls[name] = control

    def session(self, pair=False):
        names = self.names if pair else self.names[:1]
        session = MotionSession(self.args, names, self.poses,
                                factory=Mock(side_effect=[self.controls[name] for name in names]))
        self.addCleanup(session.close)
        for control in self.controls.values():
            control.reset_mock()
        return session

    def delayed(self, duration, result=True, exception=None):
        def call(*_args):
            self.now += duration
            if exception is not None:
                raise exception
            return result
        return call

    def test_cartesian_native_checks_and_send_are_timed_without_extra_commands(self):
        session = self.session()
        control = self.controls["ur5e"]
        control.getTCPOffset.side_effect = self.delayed(.01, [0.] * 6)
        control.isPoseWithinSafetyLimits.side_effect = self.delayed(.02)
        control.speedL.side_effect = self.delayed(.03)
        fresh = Mock(side_effect=self.delayed(.005, None))
        session.send(self.poses, self.velocities, check_fresh=fresh, safety_targets=self.poses)
        record = session.transport_diagnostics()
        self.assertEqual(record["mode"], "speedL")
        self.assertEqual([op["stage"] for op in record["operations"]],
                         ["isConnected", "isProgramRunning", "getTCPOffset", "goal_safety",
                          "next_goal_safety", "check_fresh", "speedL"])
        for actual, expected in zip([op["duration"] for op in record["operations"]],
                                    [0., 0., .01, .02, .02, .005, .03]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(record["duration"], .085)
        self.assertAlmostEqual(record["last_successful_send"]["ur5e"]["at"], 10.085)
        control.getTCPOffset.assert_called_once_with()
        self.assertEqual(control.isPoseWithinSafetyLimits.call_count, 2)
        control.speedL.assert_called_once_with([0.] * 6, .1, .002)
        self.assertEqual(record["operations"][-1]["command"], {
            "velocity": [0.] * 6, "acceleration": .1, "duration_s": .002,
            "frame": "follower_base_tcp", "unit": "m/s,rad/s"})
        control.kickWatchdog.assert_not_called()
        control.reuploadScript.assert_not_called()
        control.getRobotStatus.assert_not_called()  # Extra cached state is read only on fault.
        control.speedStop.assert_not_called()
        json.dumps(record, allow_nan=False)

    def test_joint_native_check_and_speed_call_keep_original_arguments(self):
        session = self.session()
        control = self.controls["ur5e"]
        control.isJointsWithinSafetyLimits.side_effect = self.delayed(.025)
        control.speedJ.side_effect = self.delayed(.012)
        session.send_joints(self.velocities, self.velocities, accelerations={"ur5e": .4})
        record = session.transport_diagnostics()
        self.assertEqual(record["mode"], "speedJ")
        self.assertAlmostEqual(record["duration"], .037)
        self.assertEqual([op["stage"] for op in record["operations"]],
                         ["isConnected", "isProgramRunning", "getTCPOffset", "next_joint_safety", "speedJ"])
        self.assertEqual(record["last_successful_send"]["ur5e"]["method"], "speedJ")
        control.speedJ.assert_called_once_with([0.] * 6, .4, .002)
        self.assertEqual(record["operations"][-1]["command"], {
            "velocity": [0.] * 6, "acceleration": .4, "duration_s": .002,
            "frame": "joint", "unit": "rad/s"})
        control.kickWatchdog.assert_not_called()

    def test_disconnection_and_program_stop_have_distinct_causes_before_cleanup(self):
        for method, message in (("isConnected", "connection disconnected"),
                                ("isProgramRunning", "program stopped")):
            with self.subTest(method=method):
                session = self.session()
                control = self.controls["ur5e"]
                getattr(control, method).return_value = False
                with self.assertRaisesRegex(ControlFault, message) as caught:
                    session.send(self.poses, self.velocities)
                details = caught.exception.details
                self.assertEqual(details["detected_at"], 10.)
                self.assertEqual(details["transport"]["operations"][-1]["stage"], method)
                self.assertFalse(details["transport"]["operations"][-1]["returned"])
                self.assertEqual(details["transport"]["control_status"]["ur5e"]["connected"],
                                 method != "isConnected")
                control.speedL.assert_not_called()
                control.getTCPOffset.assert_not_called()
                control.speedStop.assert_not_called()
                control.disconnect.assert_not_called()
                getattr(control, method).return_value = True
                session.close()

    def test_delayed_sdk_failure_preserves_original_exception_and_send_age(self):
        session = self.session()
        control = self.controls["ur5e"]
        session.send(self.poses, self.velocities)
        self.now = 10.05
        failure = RuntimeError("injected SDK failure")
        failure.details = {"upstream": {"code": 17}}
        control.speedL.side_effect = self.delayed(.2, exception=failure)
        with self.assertRaises(RuntimeError) as caught:
            session.send(self.poses, self.velocities)
        self.assertIs(caught.exception, failure)
        self.assertEqual(failure.details["upstream"], {"code": 17})
        record = failure.details["transport"]
        self.assertAlmostEqual(failure.details["detected_at"], 10.25)
        self.assertAlmostEqual(record["seconds_since_last_successful_send"]["ur5e"], .25)
        self.assertAlmostEqual(record["operations"][-1]["duration"], .2)
        self.assertEqual(record["operations"][-1]["exception_type"], "RuntimeError")
        self.assertEqual(record["operations"][-1]["command"]["velocity"], [0.] * 6)
        self.assertNotIn("returned", record["operations"][-1])
        self.assertEqual(record["last_successful_send"]["ur5e"]["at"], 10.)
        json.dumps(failure.details, allow_nan=False)

    def test_fault_snapshot_is_not_replaced_after_slow_cleanup(self):
        session = self.session()
        control = self.controls["ur5e"]
        control.isProgramRunning.return_value = False
        with self.assertRaises(ControlFault) as caught:
            session.send(self.poses, self.velocities)
        before = deepcopy(caught.exception.details)
        control.speedStop.side_effect = self.delayed(2.)
        control.disconnect.side_effect = self.delayed(3.)
        session.close()
        session.capture_fault(caught.exception, at=self.now)
        self.assertEqual(caught.exception.details, before)
        self.assertEqual(before["detected_at"], 10.)
        self.assertEqual(self.now, 15.)

    def test_diagnostic_failure_never_masks_original_sdk_exception(self):
        session = self.session()
        failure = RuntimeError("original send error")
        failure.details = {"original": "retained"}
        self.controls["ur5e"].speedL.side_effect = failure
        with patch("urcal.control.transport_diagnostics.control_snapshot", side_effect=RuntimeError("bad getter")):
            with self.assertRaises(RuntimeError) as caught:
                session.send(self.poses, self.velocities)
        self.assertIs(caught.exception, failure)
        self.assertEqual(failure.details, {"original": "retained"})

    def test_native_check_failure_has_its_stage_and_never_sends_velocity(self):
        session = self.session()
        failure = RuntimeError("offset cache command failed")
        self.controls["ur5e"].getTCPOffset.side_effect = self.delayed(.08, exception=failure)
        with self.assertRaises(RuntimeError) as caught:
            session.send(self.poses, self.velocities)
        self.assertIs(caught.exception, failure)
        operation = failure.details["transport"]["operations"][-1]
        self.assertEqual(operation["stage"], "getTCPOffset")
        self.assertAlmostEqual(operation["duration"], .08)
        self.controls["ur5e"].speedL.assert_not_called()

    def test_partial_pair_send_tracks_only_successful_arm(self):
        session = self.session(pair=True)
        self.controls["ur10e"].speedL.return_value = False
        with self.assertRaisesRegex(ControlFault, "ur10e: speedL failed") as caught:
            session.send(self.poses, self.velocities)
        record = caught.exception.details["transport"]
        self.assertEqual(set(record["last_successful_send"]), {"ur5e"})
        self.assertFalse(record["operations"][-1]["returned"])
        self.assertEqual(record["operations"][-1]["robot"], "ur10e")
        self.assertEqual(record["operations"][-1]["command"]["acceleration"], .1)

    def test_runtime_fault_captures_last_transaction_and_cached_status(self):
        session = self.session()
        session.send_joints(self.velocities, self.velocities)
        self.now = 10.12
        failure = ControlFault("outside send")
        session.capture_fault(failure, at=self.now)
        record = failure.details["transport"]
        self.assertEqual(record["mode"], "speedJ")
        self.assertAlmostEqual(record["seconds_since_last_successful_send"]["ur5e"], .12)
        self.assertTrue(record["control_status"]["ur5e"]["program_running"])
        self.controls["ur5e"].speedStop.assert_not_called()

    def test_exported_transaction_is_independent_of_mutation_and_later_sends(self):
        session = self.session()
        session.send(self.poses, self.velocities)
        record = session.transport_diagnostics()
        preserved = deepcopy(record)
        record["operations"][0]["stage"] = "changed outside"
        record["last_successful_send"]["ur5e"]["at"] = -1.
        self.assertEqual(session.transport_diagnostics(), preserved)
        self.now += .04
        session.send_joints(self.velocities, self.velocities)
        self.assertEqual(preserved["mode"], "speedL")
        self.assertEqual(preserved["last_successful_send"]["ur5e"]["at"], 10.)
        self.assertEqual(session.transport_diagnostics()["mode"], "speedJ")

    def test_cleanup_records_exact_stop_arguments_order_and_void_sdk_returns(self):
        session = self.session(pair=True)
        self.assertEqual(session.cleanup_transport_diagnostics(), {})
        session.send_joints(self.velocities, self.velocities)
        # The first Cartesian attempt changes that arm's stop mode even when
        # rejected; the second arm remains in the last attempted joint mode.
        self.controls["ur5e"].speedL.return_value = False
        with self.assertRaises(ControlFault):
            session.send(self.poses, self.velocities)
        velocity_record = session.transport_diagnostics()
        for control in self.controls.values():
            control.reset_mock()
            control.speedStop.side_effect = self.delayed(.01)
            control.stopScript.side_effect = self.delayed(.02, None)
            control.disconnect.side_effect = self.delayed(.03, None)
        session.stop_for_recovery()
        record = session.cleanup_transport_diagnostics()
        self.assertEqual(record["mode"], "cleanup")
        self.assertEqual(record["purpose"], "recovery")
        self.assertTrue(record["require_confirmation"])
        self.assertEqual(record["failures"], [])
        self.assertAlmostEqual(record["duration"], .12)
        self.assertEqual([(op["robot"], op["stage"]) for op in record["operations"]], [
            ("ur5e", "speedStop"), ("ur10e", "speedStop"),
            ("ur5e", "stopScript"), ("ur5e", "disconnect"),
            ("ur10e", "stopScript"), ("ur10e", "disconnect")])
        for op, duration in zip(record["operations"], (.01, .01, .02, .03, .02, .03)):
            self.assertAlmostEqual(op["duration"], duration)
            self.assertAlmostEqual(op["finished_at"] - op["started_at"], duration)
            self.assertIs(op["returned"], True if op["stage"] == "speedStop" else None)
        self.assertEqual(record["operations"][0]["command"], {"acceleration": .3, "motion_mode": "linear"})
        self.assertEqual(record["operations"][1]["command"], {"acceleration": .8, "motion_mode": "joint"})
        for name, control in self.controls.items():
            control.speedStop.assert_called_once_with(.3 if name == "ur5e" else .8)
            self.assertEqual([call[0] for call in control.mock_calls], ["speedStop", "stopScript", "disconnect"])
        self.assertEqual(session.transport_diagnostics(), velocity_record)
        session.close()
        self.assertEqual(session.cleanup_transport_diagnostics(), record)
        self.assertEqual(session.transport_diagnostics(), velocity_record)
        for control in self.controls.values():
            self.assertEqual(len(control.mock_calls), 3)
        json.dumps(record, allow_nan=False)

    def test_failed_recovery_retains_every_attempt_without_changing_velocity_record(self):
        session = self.session(pair=True)
        session.send(self.poses, self.velocities)
        velocity_record = session.transport_diagnostics()
        self.controls["ur5e"].speedStop.side_effect = self.delayed(.01, False)
        self.controls["ur5e"].stopScript.return_value = False
        self.controls["ur10e"].speedStop.side_effect = self.delayed(.02, exception=RuntimeError("stop unavailable"))
        self.controls["ur10e"].disconnect.side_effect = self.delayed(.03, exception=RuntimeError("disconnect unavailable"))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ControlFault, "could not be confirmed"):
            session.stop_for_recovery()
        record = session.cleanup_transport_diagnostics()
        self.assertEqual(len(record["operations"]), 6)
        self.assertEqual(len(record["failures"]), 4)
        self.assertFalse(record["operations"][0]["returned"])
        failed = record["operations"][1]
        self.assertEqual(failed["exception_type"], "RuntimeError")
        self.assertEqual(failed["exception"], "stop unavailable")
        self.assertNotIn("returned", failed)
        self.assertAlmostEqual(failed["duration"], .02)
        self.assertEqual(record["operations"][-1]["exception"], "disconnect unavailable")
        self.assertEqual(session.transport_diagnostics(), velocity_record)
        self.assertFalse(session.controls)
        for control in self.controls.values():
            control.speedStop.assert_called_once()
            control.stopScript.assert_called_once()
            control.disconnect.assert_called_once()
        session.close()
        self.assertEqual(session.cleanup_transport_diagnostics(), record)
        with self.assertRaisesRegex(ControlFault, "session is closed"):
            session.stop_for_recovery()
        self.assertEqual(session.cleanup_transport_diagnostics(), record)

    def test_best_effort_shutdown_records_script_exception_and_preserves_policy(self):
        session = self.session()
        control = self.controls["ur5e"]
        control.speedStop.return_value = False
        control.stopScript.side_effect = self.delayed(.03, exception=RuntimeError("script unavailable"))
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(session.close())
        record = session.cleanup_transport_diagnostics()
        self.assertEqual(record["purpose"], "shutdown")
        self.assertFalse(record["require_confirmation"])
        self.assertEqual(record["operations"][0]["command"], {"acceleration": .3, "motion_mode": None})
        self.assertFalse(record["operations"][0]["returned"])
        self.assertEqual(record["operations"][1]["exception"], "script unavailable")
        self.assertEqual(len(record["failures"]), 1)
        control.disconnect.assert_called_once()
        snapshot = deepcopy(record)
        record["operations"][0]["command"]["acceleration"] = -1.
        record["failures"].append("external mutation")
        self.assertEqual(session.cleanup_transport_diagnostics(), snapshot)

    def test_interrupted_stop_keeps_exception_and_original_cleanup_interruption_policy(self):
        session = self.session()
        interrupt = KeyboardInterrupt()
        control = self.controls["ur5e"]
        control.speedStop.side_effect = self.delayed(.04, exception=interrupt)
        with self.assertRaises(KeyboardInterrupt) as caught:
            session.close()
        self.assertIs(caught.exception, interrupt)
        record = session.cleanup_transport_diagnostics()
        self.assertAlmostEqual(record["duration"], .04)
        self.assertEqual(record["operations"][0]["exception_type"], "KeyboardInterrupt")
        self.assertNotIn("returned", record["operations"][0])
        control.stopScript.assert_not_called()
        control.disconnect.assert_not_called()

    def test_unavailable_diagnostic_clock_cannot_prevent_stop_and_disconnect(self):
        session = self.session()
        with patch("urcal.control.runtime.time.monotonic", side_effect=RuntimeError("clock unavailable")):
            session.close()
        record = session.cleanup_transport_diagnostics()
        self.assertIsNone(record["started_at"])
        self.assertIsNone(record["finished_at"])
        self.assertIsNone(record["duration"])
        for operation in record["operations"]:
            self.assertIsNone(operation["started_at"])
            self.assertIsNone(operation["finished_at"])
            self.assertIsNone(operation["duration"])
        control = self.controls["ur5e"]
        control.speedStop.assert_called_once_with(.3)
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
