"""Pre-cleanup motion transport timing with SDK mocks and a virtual clock."""

from copy import deepcopy
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


if __name__ == "__main__":
    unittest.main()
