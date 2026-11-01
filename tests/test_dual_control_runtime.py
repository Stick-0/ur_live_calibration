"""Shared follower transport, state readers, and scene snapshots without hardware."""

from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
from queue import Queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urcal.control.runtime as runtime
from urcal.control.geometry import matrix_pose, pose_matrix
from urcal.control.policy import ControlFault
from urcal.cooperative.controller import CooperativeController


def args(mode="follow", *extra):
    return runtime.parse_args([
        "--mode", mode, "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2",
        "--target_tolerance_m", "0", "--target_tolerance_deg", "0", "--fault_response", "exit",
        "--max_robot_age", "0.15",
        "--ur5e_tcp_offset", "0", "0", "0", "0", "0", "0",
        "--ur10e_tcp_offset", "0", "0", "0", "0", "0", "0", *extra,
    ])


def receiver():
    result = Mock()
    result.isConnected.return_value = True
    result.getTimestamp.return_value = 1.0
    result.getSafetyMode.return_value = 1
    result.getRobotMode.return_value = 7
    result.getActualQ.return_value = [0.] * 6
    result.getActualTCPSpeed.return_value = [0.] * 6
    result.getActualTCPPose.return_value = [0.] * 6
    return result


def controller(offset=None):
    result = Mock()
    result.getTCPOffset.return_value = [0.] * 6 if offset is None else matrix_pose(offset).tolist()
    for name in ("reuploadScript", "setWatchdog", "isConnected", "isProgramRunning", "isPoseWithinSafetyLimits",
                 "kickWatchdog", "speedL", "speedStop", "stopScript"):
        getattr(result, name).return_value = True
    return result


class CommandInboxTests(unittest.TestCase):
    def test_abandoned_iteration_preserves_later_target_and_stop(self):
        for after_failure in (False, True):
            with self.subTest(after_failure=after_failure):
                queue = Queue()
                inbox = runtime.CommandInbox(queue)
                invalid_target = {"type": "target", "gap": -.1}
                valid_target = {"type": "target", "gap": .3}
                for command in (invalid_target, valid_target, "stop"):
                    queue.put_nowait(command)
                if after_failure:
                    inbox.cancel_arm_requests()
                # A target handler can raise, abandoning this iterator early.
                iterator = inbox.read()
                self.assertEqual(next(iterator), invalid_target)
                iterator.close()
                self.assertEqual(list(inbox.read()), [valid_target, "stop"])


class StateReaderTests(unittest.TestCase):
    def test_cached_receive_data_expires_without_advancing_robot_timestamp(self):
        source = receiver()
        reader = runtime.StateReader(source, max_age=0.15)
        reader.read(10.0)
        reader.read(10.1)
        with self.assertRaisesRegex(ControlFault, "stopped updating"):
            reader.read(10.16)
        source.getTimestamp.return_value = 1.1
        state = reader.read(10.17)
        np.testing.assert_array_equal(state.tcp, np.eye(4))

    def test_disconnect_bad_safety_and_not_running_are_rejected(self):
        cases = [("isConnected", False), ("getSafetyMode", 3),
                 ("getSafetyMode", 9), ("getRobotMode", 5)]
        for method, value in cases:
            with self.subTest(method=method, value=value):
                source = receiver()
                getattr(source, method).return_value = value
                with self.assertRaises(ControlFault):
                    runtime.StateReader(source).read(10.0)
        source = receiver()
        source.getSafetyMode.return_value = 2
        runtime.StateReader(source).read(10.0)

    def test_robot_clock_backwards_nonfinite_and_bad_state_are_rejected(self):
        source = receiver()
        reader = runtime.StateReader(source)
        reader.read(10.0)
        source.getTimestamp.return_value = 0.9
        with self.assertRaisesRegex(ControlFault, "backwards"):
            reader.read(10.01)
        cases = [("getTimestamp", float("nan")), ("getActualQ", [0.] * 5),
                 ("getActualTCPSpeed", [float("inf")] * 6),
                 ("getActualTCPPose", [float("nan")] * 6)]
        for method, value in cases:
            with self.subTest(method=method):
                source = receiver()
                getattr(source, method).return_value = value
                with self.assertRaises((ControlFault, ValueError)):
                    runtime.StateReader(source).read(10.0)


class MotionSessionTests(unittest.TestCase):
    def test_default_driver_disables_auto_upload_and_uploads_once_per_explicit_arm(self):
        config = args()
        controls = [controller(), controller()]
        driver = Mock(side_effect=controls, FLAG_NO_WAIT=2, FLAG_UPLOAD_SCRIPT=1, FLAG_CUSTOM_SCRIPT=4)
        with patch.dict(sys.modules, {"rtde_control": SimpleNamespace(RTDEControlInterface=driver)}):
            session = runtime.MotionSession(config, runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS})
        self.addCleanup(session.close)
        self.assertEqual(driver.call_count, 2)
        for invocation, name in zip(driver.call_args_list, runtime.ROBOTS):
            self.assertEqual(invocation.args, (getattr(config, name + "_ip"),))
            flags = invocation.kwargs["flags"]
            self.assertEqual(flags, driver.FLAG_NO_WAIT)
            self.assertFalse(flags & (driver.FLAG_UPLOAD_SCRIPT | driver.FLAG_CUSTOM_SCRIPT))
        for _ in range(2):
            session.send({name: np.eye(4) for name in runtime.ROBOTS},
                         {name: np.zeros(6) for name in runtime.ROBOTS})
        # A stopped script must raise, without trying to restart it on send.
        controls[1].isProgramRunning.return_value = False
        with self.assertRaisesRegex(ControlFault, "program stopped"):
            session.send({name: np.eye(4) for name in runtime.ROBOTS},
                         {name: np.zeros(6) for name in runtime.ROBOTS})
        for control in controls:
            control.reuploadScript.assert_called_once_with()
            self.assertEqual(control.speedL.call_count, 2)

    def test_watchdog_configured_only_after_both_uploaded_scripts_are_running(self):
        controls = [controller(), controller()]
        events = []
        for index, control in enumerate(controls):
            control.reuploadScript.side_effect = lambda i=index: events.append(("upload", i)) or True
            states = iter([False, True])

            def running(i=index, values=states):
                result = next(values)
                events.append(("running" if result else "waiting", i))
                return result

            control.isProgramRunning.side_effect = running
            control.getTCPOffset.side_effect = lambda i=index: events.append(("offset", i)) or [0.] * 6
            control.setWatchdog.side_effect = lambda _rate, i=index: events.append(("watchdog", i)) or True
        with patch.object(runtime.time, "monotonic", side_effect=itertools.count(10., 0.01)), \
                patch.object(runtime.time, "sleep") as sleep:
            session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                            factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        self.assertEqual(sleep.call_count, 2)
        last_ready = max(events.index(("running", index)) for index in range(2))
        first_watchdog = min(events.index(("watchdog", index)) for index in range(2))
        first_offset = min(events.index(("offset", index)) for index in range(2))
        self.assertLess(last_ready, first_offset)
        self.assertLess(last_ready, first_watchdog)
        for control in controls:
            control.reuploadScript.assert_called_once_with()
            control.speedL.assert_not_called()

    def test_failed_explicit_upload_stops_and_disconnects_every_connected_arm(self):
        controls = [controller(), controller()]
        controls[1].reuploadScript.return_value = False
        with self.assertRaisesRegex(ControlFault, "failed to start control script"):
            runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                  factory=Mock(side_effect=controls))
        for control in controls:
            control.reuploadScript.assert_called_once_with()
            control.setWatchdog.assert_not_called()
            control.speedL.assert_not_called()
            control.speedStop.assert_called_once()
            control.disconnect.assert_called_once()

    def test_script_startup_timeout_cleans_up_without_watchdog_or_motion(self):
        control = controller()
        control.isProgramRunning.return_value = False
        with patch.object(runtime.time, "monotonic", side_effect=[10., 15.]), \
                patch.object(runtime.time, "sleep"), \
                self.assertRaisesRegex(ControlFault, "startup timed out"):
            runtime.MotionSession(args(), ("ur5e",), {"ur5e": np.eye(4)}, factory=Mock(return_value=control))
        control.reuploadScript.assert_called_once_with()
        control.setWatchdog.assert_not_called()
        control.speedL.assert_not_called()
        control.speedStop.assert_called_once()
        control.disconnect.assert_called_once()

    def test_operator_stop_during_script_startup_cleans_up(self):
        control = controller()
        control.isProgramRunning.return_value = False
        stop = Mock()
        stop.is_set.side_effect = [False, False, True]  # connect, upload, startup wait
        with self.assertRaisesRegex(ControlFault, "operator stop during controller startup"):
            runtime.MotionSession(args(), ("ur5e",), {"ur5e": np.eye(4)},
                                  factory=Mock(return_value=control), stop_event=stop)
        control.reuploadScript.assert_called_once_with()
        control.setWatchdog.assert_not_called()
        control.speedL.assert_not_called()
        control.speedStop.assert_called_once()
        control.disconnect.assert_called_once()

    def test_preexisting_operator_stop_prevents_control_connection(self):
        factory = Mock()
        stop = Mock()
        stop.is_set.return_value = True
        with self.assertRaisesRegex(ControlFault, "operator stop during controller startup"):
            runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                  factory=factory, stop_event=stop)
        factory.assert_not_called()

    def test_follow_connects_only_commanded_follower(self):
        config = args("follow", "--leader", "ur5e")
        planner = CooperativeController(config)
        control = controller()
        factory = Mock(return_value=control)
        session = runtime.MotionSession(config, planner.commanded, planner.offsets, factory=factory)
        self.addCleanup(session.close)
        factory.assert_called_once_with(config.ur10e_ip)
        self.assertEqual(set(session.controls), {"ur10e"})
        control.speedL.assert_not_called()

    def test_no_motion_before_every_pose_safety_check_succeeds(self):
        config = args()
        controls = [controller(), controller()]
        session = runtime.MotionSession(config, runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        controls[1].isPoseWithinSafetyLimits.return_value = False
        with self.assertRaisesRegex(ControlFault, "safety limits"):
            session.send({name: np.eye(4) for name in runtime.ROBOTS}, {name: np.zeros(6) for name in runtime.ROBOTS})
        for control in controls:
            control.speedL.assert_not_called()

    def test_offset_mismatch_blocks_constructor_and_stops_both_connections(self):
        controls = [controller(), controller(pose_matrix([0, 0, 0.1, 0, 0, 0]))]
        with self.assertRaisesRegex(ControlFault, "active TCP differs"):
            runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                  factory=Mock(side_effect=controls))
        for control in controls:
            control.speedL.assert_not_called()
            control.speedStop.assert_called_once()
            control.disconnect.assert_called_once()

    def test_active_tcp_offset_change_blocks_both_motion_commands(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        controls[1].getTCPOffset.return_value = [0, 0, 0.001, 0, 0, 0]
        with self.assertRaisesRegex(ControlFault, "active TCP differs"):
            session.send({name: np.eye(4) for name in runtime.ROBOTS}, {name: np.zeros(6) for name in runtime.ROBOTS})
        for control in controls:
            control.speedL.assert_not_called()

    def test_second_connection_failure_cleans_up_first_connection(self):
        first = controller()
        with self.assertRaisesRegex(RuntimeError, "second connection"):
            runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                  factory=Mock(side_effect=[first, RuntimeError("second connection failed")]))
        first.speedL.assert_not_called()
        first.speedStop.assert_called_once()
        first.stopScript.assert_called_once()
        first.disconnect.assert_called_once()

    def test_command_failure_is_reported_and_cleanup_stops_all_controlled_arms(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        controls[1].speedL.return_value = False
        try:
            with self.assertRaisesRegex(ControlFault, "speedL failed"):
                session.send({name: np.eye(4) for name in runtime.ROBOTS}, {name: np.zeros(6) for name in runtime.ROBOTS})
        finally:
            session.close()
        for control in controls:
            control.speedL.assert_called_once()
            control.speedStop.assert_called_once()
            control.disconnect.assert_called_once()

    def test_freshness_checked_after_pair_preflight_before_each_motion_command(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        events = []
        for index, control in enumerate(controls):
            control.isPoseWithinSafetyLimits.side_effect = lambda _pose, i=index: events.append(("check", i)) or True
            control.speedL.side_effect = lambda *_args, i=index: events.append(("move", i)) or True
        session.send({name: np.eye(4) for name in runtime.ROBOTS}, {name: np.zeros(6) for name in runtime.ROBOTS},
                     check_fresh=lambda: events.append(("fresh", None)))
        self.assertEqual(events, [("check", 0), ("check", 1), ("fresh", None), ("move", 0),
                                  ("fresh", None), ("move", 1)])

    def test_estimate_expiring_during_preflight_blocks_motion(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ControlFault, "stale"):
            session.send({name: np.eye(4) for name in runtime.ROBOTS}, {name: np.zeros(6) for name in runtime.ROBOTS},
                         check_fresh=Mock(side_effect=ControlFault("calibration stale during preflight")))
        for control in controls:
            control.isPoseWithinSafetyLimits.assert_called_once()
            control.speedL.assert_not_called()

    def test_close_stops_both_arms_before_script_or_connection_cleanup(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        events = []
        for index, control in enumerate(controls):
            control.speedStop.side_effect = lambda *_args, i=index: events.append(("stop", i))
            control.stopScript.side_effect = lambda i=index: events.append(("script", i))
            control.disconnect.side_effect = lambda i=index: events.append(("disconnect", i))
        session.close()
        self.assertEqual(events[:2], [("stop", 0), ("stop", 1)])

    def test_close_continues_after_stop_and_disconnect_failures(self):
        controls = [controller(), controller()]
        session = runtime.MotionSession(args(), runtime.ROBOTS, {name: np.eye(4) for name in runtime.ROBOTS},
                                        factory=Mock(side_effect=controls))
        controls[0].speedStop.side_effect = RuntimeError("stop failure")
        controls[0].disconnect.side_effect = RuntimeError("disconnect failure")
        with redirect_stdout(io.StringIO()):
            session.close()
            session.close()  # idempotent
        for control in controls:
            control.speedStop.assert_called_once()
            control.stopScript.assert_called_once()
            control.disconnect.assert_called_once()
        self.assertEqual(session.controls, {})


class GuiSceneSnapshotTests(unittest.TestCase):
    def inputs(self):
        bases = {name: np.eye(4) for name in runtime.ROBOTS}
        states = {name: runtime.RobotState(np.eye(4), np.zeros(6), np.zeros(6), robot_timestamp=1.)
                  for name in runtime.ROBOTS}
        offsets = {name: np.eye(4) for name in runtime.ROBOTS}
        readers = {name: SimpleNamespace(last_advance=10.) for name in runtime.ROBOTS}
        return bases, states, offsets, readers

    def test_snapshot_copies_arrays_and_uses_original_joint_receipt_time(self):
        bases, states, offsets, readers = self.inputs()
        goals = {"ur5e": np.eye(4)}
        previous = {}
        scene = runtime.make_gui_scene(10.02, bases, states, offsets, goals, .12, readers, previous, .15)
        self.assertEqual(scene["at"], 10.)
        self.assertEqual(scene["measurement_age"], .12)
        bases["ur5e"][0, 3] = 9.
        states["ur5e"].joints[0] = 9.
        states["ur5e"].tcp[0, 3] = 9.
        offsets["ur5e"][0, 3] = 9.
        goals["ur5e"][0, 3] = 9.
        np.testing.assert_array_equal(scene["bases"]["ur5e"], np.eye(4))
        np.testing.assert_array_equal(scene["robots"]["ur5e"]["q"], np.zeros(6))
        for name in ("tcp", "tcp_offset"):
            np.testing.assert_array_equal(scene["robots"]["ur5e"][name], np.eye(4))
        np.testing.assert_array_equal(scene["goals"]["ur5e"], np.eye(4))

    def test_invalid_stale_or_duplicate_data_never_advance_display_timestamps(self):
        bases, states, offsets, readers = self.inputs()
        previous = {name: 1. for name in runtime.ROBOTS}
        states["ur5e"].robot_timestamp = 2.
        self.assertIsNone(runtime.make_gui_scene(10., bases, states, offsets, {}, .1, readers, previous, .15))
        self.assertEqual(previous, {name: 1. for name in runtime.ROBOTS})
        states["ur10e"].robot_timestamp = 2.
        self.assertIsNone(runtime.make_gui_scene(10.2, bases, states, offsets, {}, .3, readers, previous, .15))
        bases["ur5e"][0, 0] = 2.
        self.assertIsNone(runtime.make_gui_scene(10., bases, states, offsets, {}, .1, readers, previous, .15))
        self.assertEqual(previous, {name: 1. for name in runtime.ROBOTS})

if __name__ == "__main__":
    unittest.main()
