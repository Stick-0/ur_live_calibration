"""Dual control lifecycle and frame planning with mocked robot interfaces."""

from contextlib import ExitStack, redirect_stdout
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
from urcal.control.geometry import facing_follower, inverse, matrix_pose, pose_matrix
from urcal.control.policy import CalibrationFrame, ControlFault


def args(mode="hold", *extra):
    return runtime.parse_args([
        "--mode", mode, "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2",
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


class FacingPlannerTests(unittest.TestCase):
    def test_defaults_are_ur10e_leader_and_thirty_centimetre_gap(self):
        config = args("follow")
        self.assertEqual(config.leader, "ur10e")
        self.assertAlmostEqual(config.gap, 0.30)
        self.assertEqual(runtime.FacingPlanner(config).commanded, ("ur5e",))

    def setup_scene(self, mode="hold", leader="ur5e"):
        config = args(mode, "--leader", leader, "--gap", "0.3")
        config.ur5e_tcp_offset = [0.02, -0.01, 0.08, 0.1, 0, -0.05]
        config.ur10e_tcp_offset = [-0.03, 0.01, 0.12, 0, 0.15, 0.02]
        planner = runtime.FacingPlanner(config)
        bases = {"ur5e": pose_matrix([-0.2, 0, 1, 0, 0.1, 0.05]),
                 "ur10e": pose_matrix([0.3, 0, 1, 0.1, -0.05, 0.1])}
        calibration = {name: bases[name] @ inverse(planner.mappings[name]) for name in runtime.ROBOTS}
        flange5 = pose_matrix([0.1, 0.05, 0.4, 0, 0.2, 0])
        camera5 = bases["ur5e"] @ flange5
        flange10 = inverse(bases["ur10e"]) @ facing_follower(camera5, 0.3)
        flanges = {"ur5e": flange5, "ur10e": flange10}
        states = {name: runtime.RobotState(flange @ planner.offsets[name], np.zeros(6), np.zeros(6))
                  for name, flange in flanges.items()}

        def consistent_nominal(name, _joints):
            return planner.mappings[name] @ states[name].tcp @ inverse(planner.offsets[name])

        return planner, calibration, states, consistent_nominal

    def test_hold_targets_remain_fixed_in_camera_frame_after_calibration_update(self):
        planner, calibration, states, nominal = self.setup_scene()
        with patch.object(runtime, "model_flange_pose", side_effect=nominal):
            goals_before, tcp_before = planner.plan(calibration, states)
            camera_shift = pose_matrix([0.01, -0.005, 0, 0, 0, 0.01])
            moved_calibration = {name: camera_shift @ value for name, value in calibration.items()}
            goals_after, tcp_after = planner.plan(moved_calibration, states)
        self.assertEqual(set(goals_after), set(runtime.ROBOTS))
        for name in runtime.ROBOTS:
            np.testing.assert_allclose(goals_after[name], goals_before[name], atol=1e-12)
            self.assertGreater(np.linalg.norm(tcp_after[name] - tcp_before[name]), 1e-3)
            reconstructed = moved_calibration[name] @ planner.mappings[name] @ tcp_after[name] @ inverse(planner.offsets[name])
            np.testing.assert_allclose(reconstructed, goals_before[name], atol=1e-12)

    def test_follow_uses_live_leader_and_preserves_nonzero_active_tcp_offsets(self):
        for leader in runtime.ROBOTS:
            with self.subTest(leader=leader):
                planner, calibration, states, nominal = self.setup_scene("follow", leader)
                follower = next(name for name in runtime.ROBOTS if name != leader)
                self.assertEqual(planner.commanded, (follower,))
                with patch.object(runtime, "model_flange_pose", side_effect=nominal):
                    first_goals, _ = planner.plan(calibration, states)
                    states[leader].tcp = pose_matrix([0.01, 0, 0, 0, 0, 0.02]) @ states[leader].tcp
                    calibration[follower] = pose_matrix([0.005, 0, 0, 0, 0, 0]) @ calibration[follower]
                    goals, tcp_targets = planner.plan(calibration, states)
                self.assertEqual(set(tcp_targets), {follower})
                leader_flange = (calibration[leader] @ planner.mappings[leader] @ states[leader].tcp
                                 @ inverse(planner.offsets[leader]))
                expected_flange = facing_follower(leader_flange, 0.3)
                expected_tcp = inverse(calibration[follower] @ planner.mappings[follower]) @ expected_flange @ planner.offsets[follower]
                np.testing.assert_allclose(goals[follower], expected_flange, atol=1e-12)
                np.testing.assert_allclose(tcp_targets[follower], expected_tcp, atol=1e-12)
                self.assertGreater(np.linalg.norm(goals[follower] - first_goals[follower]), 1e-3)

    def test_fk_frame_mismatch_blocks_targets(self):
        planner, calibration, states, nominal = self.setup_scene()

        def wrong_nominal(name, joints):
            transform = nominal(name, joints)
            transform[0, 3] += 0.04
            return transform

        with patch.object(runtime, "model_flange_pose", side_effect=wrong_nominal):
            with self.assertRaisesRegex(ControlFault, "FK/base/TCP mismatch"):
                planner.plan(calibration, states)

    def test_gui_apply_recomputes_hold_center_gap_then_freezes_camera_goals(self):
        planner, calibration, states, nominal = self.setup_scene()
        self.assertFalse(planner.target_applied)
        with patch.object(runtime, "model_flange_pose", side_effect=nominal):
            provisional, _ = planner.plan(calibration, states)
            seeded = pose_matrix(planner.center_pose)
            np.testing.assert_allclose(seeded[:3, 3],
                                       (provisional["ur5e"][:3, 3] + provisional["ur10e"][:3, 3]) / 2)
            requested = np.asarray(planner.center_pose).copy()
            requested[:3] += [0.01, -0.02, 0.005]
            planner.apply_target({"type": "target", "center_pose": requested.tolist(), "gap": 0.34})
            self.assertTrue(planner.target_applied)
            self.assertIsNone(planner.hold_targets)
            applied, _ = planner.plan(calibration, states)
            center = pose_matrix(requested)
            expected5 = center.copy()
            expected5[:3, 3] -= center[:3, 2] * 0.17
            np.testing.assert_allclose(applied["ur5e"], expected5, atol=1e-12)
            np.testing.assert_allclose(applied["ur10e"], facing_follower(expected5, 0.34), atol=1e-12)
            np.testing.assert_allclose(pose_matrix(planner.center_pose), center, atol=1e-12)
            # Live calibration changes the base-space commands, not the chosen camera target.
            moved = {name: pose_matrix([0.004, 0, 0, 0, 0, 0]) @ transform
                     for name, transform in calibration.items()}
            repeated, _ = planner.plan(moved, states)
            for name in runtime.ROBOTS:
                np.testing.assert_allclose(repeated[name], applied[name], atol=1e-12)

    def test_explicit_hold_center_counts_as_applied_target(self):
        config = args("hold", "--center_pose", "0", "0", "1", "0", "0", "0")
        planner = runtime.FacingPlanner(config)
        self.assertTrue(planner.target_applied)
        np.testing.assert_allclose(planner.center_pose, [0, 0, 1, 0, 0, 0])

    def test_invalid_gui_target_does_not_mutate_accepted_target(self):
        planner, calibration, states, nominal = self.setup_scene()
        with patch.object(runtime, "model_flange_pose", side_effect=nominal):
            planner.plan(calibration, states)
            chosen = list(planner.center_pose)
            planner.apply_target({"type": "target", "center_pose": chosen, "gap": 0.3})
            accepted, _ = planner.plan(calibration, states)
        bad_commands = [
            {"type": "target", "center_pose": chosen, "gap": 0.0},
            {"type": "target", "center_pose": chosen, "gap": 0.1},
            {"type": "target", "center_pose": chosen, "gap": float("nan")},
            {"type": "target", "center_pose": chosen, "gap": float("inf")},
            {"type": "target", "center_pose": [0, 0, 1], "gap": 0.4},
            {"type": "target", "center_pose": [0, 0, 1, 0, 0, float("nan")], "gap": 0.4},
        ]
        for command in bad_commands:
            with self.subTest(command=command), self.assertRaises((ControlFault, ValueError)):
                planner.apply_target(command)
            self.assertAlmostEqual(planner.gap, 0.3)
            np.testing.assert_allclose(planner.center_pose, chosen, atol=1e-12)
            for name in runtime.ROBOTS:
                np.testing.assert_allclose(planner.hold_targets[name], accepted[name], atol=1e-12)

    def test_follow_apply_accepts_only_gap_and_updates_facing_target(self):
        planner, calibration, states, nominal = self.setup_scene("follow", "ur10e")
        self.assertTrue(planner.target_applied)
        with patch.object(runtime, "model_flange_pose", side_effect=nominal):
            first, _ = planner.plan(calibration, states)
            planner.apply_target({"type": "target", "gap": 0.4})
            second, _ = planner.plan(calibration, states)
        self.assertAlmostEqual(np.linalg.norm(second["ur5e"][:3, 3] - first["ur5e"][:3, 3]), 0.1)
        with self.assertRaises((ControlFault, ValueError)):
            planner.apply_target({"type": "target", "gap": 0.35, "center_pose": [0, 0, 1, 0, 0, 0]})
        self.assertAlmostEqual(planner.gap, 0.4)

    def test_envelope_and_measured_velocity_are_checked_only_for_commanded_arms(self):
        planner, calibration, states, nominal = self.setup_scene("follow")
        with patch.object(runtime, "model_flange_pose", side_effect=nominal):
            _, goals = planner.plan(calibration, states)
        planner.check_envelope(states, goals)
        states["ur5e"].speed[0] = 1.0  # external leader may move; never commanded here
        planner.check_envelope(states, goals)
        states["ur10e"].speed[0] = 1.0
        with self.assertRaisesRegex(ControlFault, "measured speed"):
            planner.check_envelope(states, goals)
        states["ur10e"].speed[:] = 0
        far = {"ur10e": goals["ur10e"].copy()}
        far["ur10e"][0, 3] += 1.0
        with self.assertRaisesRegex(ControlFault, "envelope"):
            planner.check_envelope(states, far)


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
        planner = runtime.FacingPlanner(config)
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


class _LocalQueue(Queue):
    def cancel_join_thread(self):
        pass

    def close(self):
        pass


class _Event:
    def __init__(self, stop_after_waits=None, on_wait=None):
        self.stopped = False
        self.waits = 0
        self.stop_after_waits = stop_after_waits
        self.on_wait = on_wait

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, _timeout):
        self.waits += 1
        if self.on_wait is not None:
            self.on_wait(self.waits)
        if self.stop_after_waits is not None and self.waits >= self.stop_after_waits:
            self.set()
        return self.stopped


class RuntimeModeTests(unittest.TestCase):
    def mocked_run(self, config, session=None, *, planner=None, initial_commands=None,
                   later_commands=None, expect_fault=None, clock=None, stop_after_waits=None,
                   lifecycle=None):
        frames, commands, statuses, gui_statuses = (_LocalQueue() for _ in range(4))
        frames.put(CalibrationFrame(1, 10.0, {name: np.eye(4) for name in runtime.ROBOTS},
                                    {name: 1. for name in runtime.ROBOTS},
                                    {name: 2. for name in runtime.ROBOTS},
                                    {name: 7 for name in runtime.ROBOTS}))
        for command in ["arm"] if initial_commands is None else initial_commands:
            commands.put(command)
        if expect_fault is None:
            expect_fault = session is not None
        if stop_after_waits is None and not expect_fault:
            stop_after_waits = 1 if session is None else 2

        def publish_later(wait_count):
            if wait_count == 1:
                for command in later_commands or []:
                    commands.put(command)

        event = _Event(stop_after_waits=stop_after_waits, on_wait=publish_later)
        worker = Mock(pid=123)
        worker.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = [frames, commands, statuses, gui_statuses]
        context.Event.return_value = event
        context.Process.return_value = worker
        if planner is None:
            planner = Mock(commanded=("ur10e",), gap=0.3, center_pose=None, target_applied=True,
                           offsets={name: np.eye(4) for name in runtime.ROBOTS})
            planner.plan.return_value = ({"ur10e": np.eye(4)}, {"ur10e": np.eye(4)})
        receivers = [receiver(), receiver()]
        for index, item in enumerate(receivers):
            item.getTimestamp.side_effect = itertools.count(1., 0.02)
            if lifecycle is not None:
                item.disconnect.side_effect = lambda i=index: lifecycle.append(("receiver", i))
        source_factory = Mock(side_effect=receivers)
        control_factory = Mock(return_value=session)
        with ExitStack() as cleanup:
            cleanup.enter_context(patch.dict(sys.modules, {
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=source_factory),
                "urcal.control.vision": SimpleNamespace(vision_worker=Mock()),
                "urcal.control.gui": SimpleNamespace(gui_worker=Mock()),
                "rtde_control": SimpleNamespace(RTDEControlInterface=Mock(side_effect=AssertionError("unexpected driver"))),
            }))
            cleanup.enter_context(patch.object(runtime.mp, "get_context", return_value=context))
            cleanup.enter_context(patch.object(runtime, "FacingPlanner", return_value=planner))
            cleanup.enter_context(patch.object(runtime, "MotionSession", control_factory))
            cleanup.enter_context(patch.object(runtime, "model_flange_pose", return_value=np.eye(4)))
            cleanup.enter_context(patch.object(runtime.time, "monotonic",
                                        side_effect=clock or itertools.count(10.01, 0.001)))
            cleanup.enter_context(redirect_stdout(io.StringIO()))
            if expect_fault:
                with self.assertRaises(ControlFault):
                    runtime.run(config)
            else:
                runtime.run(config)
        return control_factory, statuses, receivers

    def test_preview_arm_request_never_constructs_motion_session(self):
        config = args("follow", "--leader", "ur5e", "--stable_frames", "1")
        factory, statuses, receivers = self.mocked_run(config)
        factory.assert_not_called()
        self.assertIn("PREVIEW ONLY", runtime.take_latest(statuses)["message"])
        for source in receivers:
            source.disconnect.assert_called_once()

    def test_active_command_failure_closes_session_before_return(self):
        config = args("follow", "--leader", "ur5e", "--execute", "--stable_frames", "1")
        session = Mock()
        session.send.side_effect = ControlFault("injected command failure")
        factory, _statuses, receivers = self.mocked_run(config, session)
        factory.assert_called_once()
        session.send.assert_called_once()
        session.close.assert_called()
        for source in receivers:
            source.disconnect.assert_called_once()

    def test_gui_target_is_applied_only_during_preview(self):
        config = args("hold", "--stable_frames", "1")
        planner = Mock(commanded=runtime.ROBOTS, gap=0.3, target_applied=True,
                       center_pose=[0, 0, 1, 0, 0, 0], offsets={name: np.eye(4) for name in runtime.ROBOTS})
        planner.plan.return_value = ({name: np.eye(4) for name in runtime.ROBOTS},
                                     {name: np.eye(4) for name in runtime.ROBOTS})
        command = {"type": "target", "center_pose": [0.1, 0.2, 1.1, 0, 0, 0], "gap": 0.35}
        factory, _statuses, _sources = self.mocked_run(config, planner=planner, initial_commands=[command])
        planner.apply_target.assert_called_once_with(command)
        factory.assert_not_called()

    def test_gui_apply_cancels_arm_request_in_same_batch(self):
        config = args("follow", "--execute", "--stable_frames", "1")
        command = {"type": "target", "gap": 0.35}
        for batch in (["arm", command], [command, "arm"]):
            with self.subTest(batch=batch):
                factory, _statuses, _sources = self.mocked_run(config, initial_commands=batch)
                factory.assert_not_called()

    def test_hold_preview_requires_explicit_apply_before_arm(self):
        config = args("hold", "--execute", "--stable_frames", "1")
        planner = Mock(commanded=runtime.ROBOTS, gap=0.3, center_pose=[0, 0, 1, 0, 0, 0],
                       target_applied=False, offsets={name: np.eye(4) for name in runtime.ROBOTS})
        planner.plan.return_value = ({name: np.eye(4) for name in runtime.ROBOTS},
                                     {name: np.eye(4) for name in runtime.ROBOTS})
        factory, statuses, _sources = self.mocked_run(config, planner=planner)
        factory.assert_not_called()
        status = runtime.take_latest(statuses)
        self.assertFalse(status["ready"])
        self.assertFalse(status["active"])
        self.assertIn("Apply", status["message"])

    def test_gui_target_while_active_stops_without_mutating_target(self):
        config = args("follow", "--execute", "--stable_frames", "1")
        planner = Mock(commanded=("ur5e",), gap=0.3, center_pose=None, target_applied=True,
                       offsets={name: np.eye(4) for name in runtime.ROBOTS})
        planner.plan.return_value = ({"ur5e": np.eye(4)}, {"ur5e": np.eye(4)})
        session = Mock()
        factory, _statuses, _sources = self.mocked_run(
            config, session, planner=planner, later_commands=[{"type": "target", "gap": 0.4}],
        )
        factory.assert_called_once()
        planner.apply_target.assert_not_called()
        session.close.assert_called()

    def test_slow_preflight_expires_robot_state_before_calibration_deadline(self):
        config = args("follow", "--execute", "--stable_frames", "1")
        session = Mock()
        clock_value = [10.01]

        def clock():
            clock_value[0] += 0.001
            return clock_value[0]

        def delayed_send(_targets, _twists, check_fresh):
            clock_value[0] += 0.2  # state expires at .15s; calibration remains fresh at .5s
            with self.assertRaisesRegex(ControlFault, "expired during controller checks"):
                check_fresh()
            raise ControlFault("verified expired state")

        session.send.side_effect = delayed_send
        self.mocked_run(config, session, clock=clock)
        session.send.assert_called_once()
        session.close.assert_called()

    def test_failure_stops_motion_before_disconnect_of_receive_interfaces(self):
        config = args("follow", "--execute", "--stable_frames", "1")
        session = Mock()
        session.send.side_effect = ControlFault("command failure")
        lifecycle = []
        session.close.side_effect = lambda: lifecycle.append(("motion", None))
        self.mocked_run(config, session, lifecycle=lifecycle)
        self.assertEqual(lifecycle[0], ("motion", None))


if __name__ == "__main__":
    unittest.main()
