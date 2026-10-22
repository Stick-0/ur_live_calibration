"""Controller-native preflight and speedJ transport with SDK mocks only."""

import copy
import re
import unittest
from unittest.mock import Mock

import numpy as np

from urcal.control.geometry import matrix_pose, pose_matrix
from urcal.control.policy import ControlFault
from urcal.control.runtime import ArmIKUnavailable, MotionSession, ROBOTS
from test_dual_control_runtime import args


METHODS = ("isConnected", "isProgramRunning", "reuploadScript", "getTCPOffset", "setWatchdog", "kickWatchdog",
           "getInverseKinematics", "getInverseKinematicsHasSolution", "getForwardKinematics",
           "isJointsWithinSafetyLimits", "isPoseWithinSafetyLimits", "speedJ", "speedL", "speedStop", "stopScript", "disconnect")


def native_controller(offset, target, solution):
    result = Mock(spec_set=METHODS)
    for method in ("isConnected", "isProgramRunning", "reuploadScript", "setWatchdog", "kickWatchdog",
                   "getInverseKinematicsHasSolution", "isJointsWithinSafetyLimits", "isPoseWithinSafetyLimits",
                   "speedJ", "speedL", "speedStop", "stopScript"):
        getattr(result, method).return_value = True
    result.getTCPOffset.return_value = matrix_pose(offset).tolist()
    result.getInverseKinematics.return_value = solution.tolist()
    result.getForwardKinematics.return_value = matrix_pose(target).tolist()
    return result


class LiveAlignmentTransportTests(unittest.TestCase):
    def setUp(self):
        self.config = args()
        self.config.max_joint_accel = .8
        self.config.stop_accel = .3
        self.offsets = {name: pose_matrix([.01, -.02, .08 + i * .03, .1, -.2, .05]) for i, name in enumerate(ROBOTS)}
        self.targets = {name: pose_matrix([.2 + i * .1, -.1, .4, .2, -.3, .1]) for i, name in enumerate(ROBOTS)}
        self.solutions = {name: np.array([.1, -.8, .9, -.2, .5, .3 + i * .1]) for i, name in enumerate(ROBOTS)}
        self.seeds = {name: value + .02 for name, value in self.solutions.items()}
        self.controls = {name: native_controller(self.offsets[name], self.targets[name], self.solutions[name]) for name in ROBOTS}

    def session(self, deferred=False):
        session = MotionSession(self.config, ROBOTS, self.offsets, defer_watchdog=deferred,
                                factory=Mock(side_effect=[self.controls[name] for name in ROBOTS]))
        self.addCleanup(session.close)
        return session

    def assert_no_motion(self):
        for control in self.controls.values():
            control.speedJ.assert_not_called()
            control.speedL.assert_not_called()

    def recipe_controllers(self):
        """Model SDK 1.6.3's broken explicit all-zero TCP FK recipe selection."""
        self.targets = {name: pose_matrix(q) @ self.offsets[name] for name, q in self.solutions.items()}
        self.controls = {name: native_controller(self.offsets[name], self.targets[name], self.solutions[name])
                         for name in ROBOTS}

        def forward(joints, offset):
            if not np.any(np.asarray(offset)):
                raise RuntimeError("recipe has 6 fields but explicit q+TCP supplies 12")
            return matrix_pose(pose_matrix(joints) @ pose_matrix(offset)).tolist()

        for control in self.controls.values():
            control.getForwardKinematics.side_effect = forward

    def test_deferred_watchdog_permits_native_preflight_but_blocks_both_motion_modes(self):
        session = self.session(deferred=True)
        session.solve_joint_targets(self.targets, self.seeds)
        session.validate_joint_path([self.solutions])
        for control in self.controls.values():
            control.setWatchdog.assert_not_called()
            control.kickWatchdog.assert_not_called()
        with self.assertRaisesRegex(ControlFault, "watchdog"):
            session.send_joints({name: np.zeros(6) for name in ROBOTS}, self.solutions)
        with self.assertRaisesRegex(ControlFault, "watchdog"):
            session.send(self.targets, {name: np.zeros(6) for name in ROBOTS})
        self.assert_no_motion()
        session.enable_watchdog()
        session.enable_watchdog()  # Idempotent; never rewrites an active stream.
        for control in self.controls.values():
            control.setWatchdog.assert_called_once_with(self.config.watchdog_hz)
        session.send_joints({name: np.full(6, .01) for name in ROBOTS}, self.solutions)
        for control in self.controls.values():
            control.speedJ.assert_called_once()
            control.kickWatchdog.assert_not_called()

    def test_native_ik_uses_seed_and_explicit_tcp_fk_to_verify_both_results(self):
        session = self.session(deferred=True)
        original = copy.deepcopy(self.seeds)
        result = session.solve_joint_targets(self.targets, self.seeds)
        for name, control in self.controls.items():
            np.testing.assert_array_equal(result[name], self.solutions[name])
            np.testing.assert_array_equal(self.seeds[name], original[name])
            control.getInverseKinematics.assert_called_once_with(matrix_pose(self.targets[name]).tolist(),
                                                                 self.seeds[name].tolist(), .0005, .002)
            control.getInverseKinematicsHasSolution.assert_called_once_with(matrix_pose(self.targets[name]).tolist(),
                                                                            self.seeds[name].tolist(), .0005, .002)
            control.getForwardKinematics.assert_called_once_with(self.solutions[name].tolist(),
                                                                 matrix_pose(self.offsets[name]).tolist())
            control.isJointsWithinSafetyLimits.assert_called_once_with(self.solutions[name].tolist())
            control.isPoseWithinSafetyLimits.assert_called_once()
            result[name][:] = 500.
            np.testing.assert_array_equal(control.getInverseKinematics.return_value, self.solutions[name])
        self.assert_no_motion()

    def test_native_endpoint_cache_copies_actual_fk_and_clears_on_failed_retry(self):
        session = self.session(deferred=True)
        self.assertEqual(session.last_ik_tcp, {})
        native = copy.deepcopy(self.targets)
        native["ur5e"][0, 3] += .0002
        for name in ROBOTS:
            self.controls[name].getForwardKinematics.return_value = matrix_pose(native[name]).tolist()
        session.solve_joint_targets(self.targets, self.seeds)
        self.assertEqual(set(session.last_ik_tcp), set(ROBOTS))
        for name in ROBOTS:
            np.testing.assert_allclose(session.last_ik_tcp[name], native[name], atol=1e-12)
        cached = copy.deepcopy(session.last_ik_tcp)
        self.targets["ur5e"][0, 3] += .1
        self.controls["ur5e"].getForwardKinematics.return_value[0] += .1
        np.testing.assert_array_equal(session.last_ik_tcp["ur5e"], cached["ur5e"])
        # UR5 succeeds before UR10 refuses: neither an old pair nor a partial
        # new pair may be exposed as a completed native IK result.
        self.controls["ur10e"].getInverseKinematicsHasSolution.return_value = False
        with self.assertRaises(ArmIKUnavailable):
            session.solve_joint_targets(self.targets, self.seeds)
        self.assertEqual(session.last_ik_tcp, {})
        self.assert_no_motion()

    def test_zero_tcp_fk_workaround_preserves_targets_and_path_for_rotated_flanges(self):
        rotations = ([np.pi / 2., 0., 0.], [0., np.pi - .0001, 0.], [.9, -.3, .4])
        for index, rotation in enumerate(rotations):
            with self.subTest(rotation=rotation):
                self.setUp()
                self.offsets = {name: np.eye(4) for name in ROBOTS}
                self.solutions = {name: np.array([.2 + .01 * index, -.1 + .02 * i, .4, *rotation])
                                  for i, name in enumerate(ROBOTS)}
                self.seeds = {name: q + .02 for name, q in self.solutions.items()}
                self.recipe_controllers()
                session = self.session(deferred=True)
                solved = session.solve_joint_targets(self.targets, self.seeds)
                samples = [{name: q + delta for name, q in self.solutions.items()} for delta in (0., .01, .025)]
                poses = session.validate_joint_path(iter(samples))
                for name, control in self.controls.items():
                    np.testing.assert_allclose(solved[name], self.solutions[name], atol=1e-12)
                    for sample, actual in zip(samples, poses):
                        np.testing.assert_allclose(actual[name], pose_matrix(sample[name]), atol=1e-12)
                    for call in control.getForwardKinematics.call_args_list:
                        np.testing.assert_array_equal(call.args[1], [0., 0., .001, 0., 0., 0.])
                    # Safety receives the original TCP, never the temporary FK query frame.
                    safety = control.isPoseWithinSafetyLimits.call_args_list
                    expected = [self.targets[name]] + [pose_matrix(sample[name]) for sample in samples]
                    for call, target in zip(safety, expected):
                        np.testing.assert_allclose(pose_matrix(call.args[0]), target, atol=1e-12)
                    np.testing.assert_array_equal(control.getTCPOffset.return_value, np.zeros(6))
                    np.testing.assert_array_equal(session.offsets[name], np.eye(4))
                self.assert_no_motion()

    def test_mixed_tcp_offsets_change_only_the_zero_query_and_preserve_each_tcp(self):
        self.offsets["ur5e"] = np.eye(4)
        self.recipe_controllers()
        original_offsets = copy.deepcopy(self.offsets)
        session = self.session(deferred=True)
        session.solve_joint_targets(self.targets, self.seeds)
        actual = session.validate_joint_path([self.solutions])[0]
        for name, control in self.controls.items():
            expected_query = ([0., 0., .001, 0., 0., 0.] if name == "ur5e"
                              else matrix_pose(original_offsets[name]).tolist())
            for call in control.getForwardKinematics.call_args_list:
                np.testing.assert_array_equal(call.args[1], expected_query)
            np.testing.assert_allclose(actual[name], self.targets[name], atol=1e-12)
            np.testing.assert_array_equal(self.offsets[name], original_offsets[name])
        self.assert_no_motion()

    def test_zero_tcp_workaround_does_not_hide_target_residual_or_safety_rejection(self):
        for failure in ("position", "rotation", "joint safety", "pose safety"):
            with self.subTest(failure=failure):
                self.setUp()
                self.offsets = {name: np.eye(4) for name in ROBOTS}
                self.recipe_controllers()
                control = self.controls["ur10e"]
                forward = control.getForwardKinematics.side_effect

                def incorrect_fk(joints, offset):
                    value = pose_matrix(forward(joints, offset))
                    if failure == "position":
                        value[0, 3] += .0006
                    else:
                        value = value @ pose_matrix([0., 0., 0., .003, 0., 0.])
                    return matrix_pose(value).tolist()

                if failure in ("position", "rotation"):
                    control.getForwardKinematics.side_effect = incorrect_fk
                    message = "residual"
                elif failure == "joint safety":
                    control.isJointsWithinSafetyLimits.return_value = False
                    message = "joints outside"
                else:
                    control.isPoseWithinSafetyLimits.return_value = False
                    message = "TCP outside"
                session = self.session(deferred=True)
                with self.assertRaisesRegex(ControlFault, message):
                    session.solve_joint_targets(self.targets, self.seeds)
                self.assert_no_motion()

    def test_zero_tcp_invalid_native_pose_or_sdk_failure_never_returns_a_corrected_pose(self):
        for operation in ("target", "path"):
            for response in (None, [0.] * 5, [float("nan")] * 6, RuntimeError("native query failed")):
                with self.subTest(operation=operation, response=response):
                    self.setUp()
                    self.offsets = {name: np.eye(4) for name in ROBOTS}
                    self.recipe_controllers()
                    method = self.controls["ur10e"].getForwardKinematics
                    method.side_effect = response if isinstance(response, Exception) else None
                    method.return_value = response
                    session = self.session(deferred=True)
                    with self.assertRaisesRegex(ControlFault, "invalid native|native getForwardKinematics failed") as caught:
                        if operation == "target":
                            session.solve_joint_targets(self.targets, self.seeds)
                        else:
                            session.validate_joint_path([self.solutions])
                    if isinstance(response, Exception):
                        self.assertIs(caught.exception.__cause__, response)
                    self.assert_no_motion()

    def test_optional_has_solution_method_can_be_absent(self):
        for control in self.controls.values():
            del control.getInverseKinematicsHasSolution
        session = self.session(deferred=True)
        result = session.solve_joint_targets(self.targets, self.seeds)
        self.assertEqual(set(result), set(ROBOTS))
        self.assert_no_motion()

    def test_missing_native_capabilities_fail_clearly_without_motion(self):
        for method in ("getInverseKinematics", "getForwardKinematics", "isJointsWithinSafetyLimits"):
            with self.subTest(method=method):
                self.setUp()
                delattr(self.controls["ur10e"], method)
                session = self.session(deferred=True)
                with self.assertRaisesRegex(ControlFault, method):
                    session.solve_joint_targets(self.targets, self.seeds)
                self.assert_no_motion()

    def test_native_rejections_and_fk_residuals_are_not_accepted_as_a_solution(self):
        bad_position = matrix_pose(self.targets["ur10e"]).tolist()
        bad_position[0] += .0006
        bad_orientation = self.targets["ur10e"] @ pose_matrix([0, 0, 0, .003, 0, 0])
        cases = [
            ("getInverseKinematicsHasSolution", False, "no IK solution"),
            ("getInverseKinematics", [0.] * 5, "six finite"),
            ("getInverseKinematics", [float("nan")] * 6, "six finite"),
            ("getForwardKinematics", bad_position, "residual"),
            ("getForwardKinematics", matrix_pose(bad_orientation).tolist(), "residual"),
            ("getForwardKinematics", [0.] * 5, "invalid native"),
            ("isJointsWithinSafetyLimits", False, "joints outside"),
            ("isPoseWithinSafetyLimits", False, "TCP outside"),
        ]
        for method, value, reason in cases:
            with self.subTest(method=method, reason=reason):
                self.setUp()
                getattr(self.controls["ur10e"], method).return_value = value
                session = self.session(deferred=True)
                with self.assertRaisesRegex(ControlFault, reason):
                    session.solve_joint_targets(self.targets, self.seeds)
                self.assert_no_motion()

    def test_native_no_solution_preserves_exact_request_and_never_calls_ik_or_motion(self):
        self.controls["ur5e"].getInverseKinematicsHasSolution.return_value = False
        original = copy.deepcopy((self.targets["ur5e"], self.seeds["ur5e"], self.offsets["ur5e"]))
        session = self.session(deferred=True)
        with self.assertRaises(ArmIKUnavailable) as caught:
            session.solve_joint_targets(self.targets, self.seeds)
        fault = caught.exception
        self.assertEqual(fault.category, "arm_ik")
        self.assertIn("base XYZ=[0.2000, -0.1000, 0.4000]m", str(fault))
        self.assertIn("No motion started", str(fault))
        self.targets["ur5e"][:] = 0
        self.seeds["ur5e"][:] = 0
        self.offsets["ur5e"][:] = 0
        np.testing.assert_array_equal(fault.details["target_tcp"], original[0])
        np.testing.assert_array_equal(fault.details["target_pose_base"], matrix_pose(original[0]))
        np.testing.assert_array_equal(fault.details["seed_q_rad"], original[1])
        np.testing.assert_array_equal(fault.details["tcp_offset"], original[2])
        self.assertEqual(fault.details["position_tolerance_m"], .0005)
        self.assertEqual(fault.details["rotation_tolerance_rad"], .002)
        for control in self.controls.values():
            control.getInverseKinematics.assert_not_called()
            control.getForwardKinematics.assert_not_called()
            control.setWatchdog.assert_not_called()
        self.assert_no_motion()

    def test_path_samples_check_native_joint_and_tcp_safety_with_explicit_offset(self):
        session = self.session(deferred=True)
        samples = [{name: q + delta for name, q in self.solutions.items()} for delta in (0., .01, .02)]
        for control in self.controls.values():
            control.getForwardKinematics.side_effect = lambda q, offset: matrix_pose(pose_matrix(q) @ pose_matrix(offset)).tolist()
        result = session.validate_joint_path(iter(samples))
        self.assertEqual(len(result), len(samples))
        for index, sample in enumerate(samples):
            for name, q in sample.items():
                expected = pose_matrix(q) @ self.offsets[name]
                np.testing.assert_allclose(result[index][name], expected, atol=1e-12)
                np.testing.assert_allclose(self.controls[name].getForwardKinematics.call_args_list[index].args[1],
                                           matrix_pose(self.offsets[name]), atol=1e-12)
                np.testing.assert_allclose(self.controls[name].isPoseWithinSafetyLimits.call_args_list[index].args[0],
                                           matrix_pose(expected), atol=1e-12)
        self.assert_no_motion()

    def test_target_reports_identify_native_stage_before_query_with_fresh_checks_between_queries(self):
        session = self.session(deferred=True)
        events = []
        stages = (
            ("checking native IK availability", "getInverseKinematicsHasSolution"),
            ("solving native IK", "getInverseKinematics"),
            ("checking IK joint limits", "isJointsWithinSafetyLimits"),
            ("verifying native FK", "getForwardKinematics"),
            ("checking IK TCP limits", "isPoseWithinSafetyLimits"),
        )
        for name, control in self.controls.items():
            for _, method in stages:
                returned = getattr(control, method).return_value

                def queried(*values, robot=name, function=method, result=returned):
                    events.append(("query", robot, function))
                    return result

                getattr(control, method).side_effect = queried
        session.solve_joint_targets(self.targets, self.seeds,
                                    check_fresh=lambda: events.append(("fresh",)),
                                    report=lambda message: events.append(("report", message)))
        previous_query = -1
        for name in ROBOTS:
            for stage, method in stages:
                index = events.index(("query", name, method))
                reports = [event[1] for event in events[:index] if event[0] == "report"]
                self.assertIn(name.upper(), reports[-1])
                self.assertIn(stage, reports[-1])
                self.assertIn(("fresh",), events[previous_query + 1:index])
                previous_query = index
        self.assertIn(("fresh",), events[previous_query + 1:])
        self.assertIn("validated", events[-1][1])
        for control in self.controls.values():
            control.setWatchdog.assert_not_called()
        self.assert_no_motion()

    def test_path_progress_counts_only_fully_checked_samples_and_reports_final_partial_group(self):
        session = self.session(deferred=True)
        reports = []
        fresh = Mock()

        def report(message):
            match = re.search(r"joint path (\d+)/(\d+) checked", message)
            self.assertIsNotNone(match)
            queries = sum(control.getForwardKinematics.call_count for control in self.controls.values())
            reports.append((int(match[1]), int(match[2]), queries))

        samples = [{name: q + index * .001 for name, q in self.solutions.items()} for index in range(23)]
        actual = session.validate_joint_path(iter(samples), check_fresh=fresh, report=report)
        self.assertEqual(len(actual), 23)
        self.assertEqual(reports, [(0, 23, 0), (10, 23, 20), (20, 23, 40), (23, 23, 46)])
        self.assertGreaterEqual(fresh.call_count, 3 * 2 * len(samples))
        for control in self.controls.values():
            control.isJointsWithinSafetyLimits.assert_called()
            self.assertEqual(control.isPoseWithinSafetyLimits.call_count, 23)
        self.assert_no_motion()

    def test_data_expiry_after_native_fk_prevents_next_query_and_success_report(self):
        for operation in ("target", "path"):
            with self.subTest(operation=operation):
                self.setUp()
                self.offsets = {name: np.eye(4) for name in ROBOTS}
                self.recipe_controllers()
                session = self.session(deferred=True)
                expired, reports = [], []
                forward = self.controls["ur5e"].getForwardKinematics.side_effect

                def slow_query(joints, offset):
                    expired.append(True)
                    return forward(joints, offset)

                def fresh():
                    if expired:
                        raise ControlFault("capture expired during FK")

                self.controls["ur5e"].getForwardKinematics.side_effect = slow_query
                with self.assertRaisesRegex(ControlFault, "capture expired during FK"):
                    if operation == "target":
                        session.solve_joint_targets(self.targets, self.seeds, fresh, report=reports.append)
                    else:
                        session.validate_joint_path([self.solutions], fresh, report=reports.append)
                self.controls["ur5e"].getForwardKinematics.assert_called_once()
                self.controls["ur10e"].getForwardKinematics.assert_not_called()
                for control in self.controls.values():
                    control.isPoseWithinSafetyLimits.assert_not_called()
                self.assertFalse(any("validated" in message or "1/1 checked" in message for message in reports))
                self.assert_no_motion()

    def test_bad_path_sample_and_stale_preflight_never_send_motion(self):
        session = self.session(deferred=True)
        self.controls["ur10e"].isJointsWithinSafetyLimits.side_effect = [True, False]
        with self.assertRaisesRegex(ControlFault, "sample 1 joints"):
            session.validate_joint_path([self.solutions, self.solutions])
        self.assert_no_motion()
        with self.assertRaisesRegex(ControlFault, "at least one"):
            session.validate_joint_path([])
        with self.assertRaisesRegex(ControlFault, "stale capture"):
            session.solve_joint_targets(self.targets, self.seeds,
                                        check_fresh=Mock(side_effect=ControlFault("stale capture")))
        self.assert_no_motion()

    def test_idle_watchdog_kicks_stop_permanently_after_any_velocity_packet(self):
        for mode in ("joint", "linear"):
            with self.subTest(mode=mode):
                self.setUp()
                session = self.session()
                session.validate_joint_path([self.solutions])
                self.assertTrue(all(c.kickWatchdog.called for c in self.controls.values()))
                for control in self.controls.values():
                    control.kickWatchdog.reset_mock()
                    control.getForwardKinematics.reset_mock()
                zero = {name: np.zeros(6) for name in ROBOTS}
                if mode == "joint":
                    session.send_joints(zero, self.solutions)
                else:
                    session.send(self.targets, zero)
                with self.assertRaisesRegex(ControlFault, "after velocity commands"):
                    session.validate_joint_path([self.solutions])
                with self.assertRaisesRegex(ControlFault, "after velocity commands"):
                    session.solve_joint_targets(self.targets, self.seeds)
                for control in self.controls.values():
                    control.kickWatchdog.assert_not_called()
                    control.getForwardKinematics.assert_not_called()

    def test_joint_send_checks_whole_pair_before_each_fresh_command_without_watchdog_kicks(self):
        session = self.session()
        events = []
        for name, control in self.controls.items():
            control.isJointsWithinSafetyLimits.side_effect = lambda q, n=name: events.append(("safety", n)) or True
            control.speedJ.side_effect = lambda *values, n=name: events.append(("speed", n)) or True
        velocity = {name: np.full(6, .01) for name in ROBOTS}
        session.send_joints(velocity, self.solutions, check_fresh=lambda: events.append(("fresh", None)))
        self.assertEqual(events, [("safety", "ur5e"), ("safety", "ur10e"), ("fresh", None),
                                  ("speed", "ur5e"), ("fresh", None), ("speed", "ur10e")])
        for name, control in self.controls.items():
            control.speedJ.assert_called_once_with(velocity[name].tolist(), self.config.max_joint_accel, .002)
            control.kickWatchdog.assert_not_called()
            control.getInverseKinematics.assert_not_called()
            control.getForwardKinematics.assert_not_called()

    def test_second_arm_safety_or_offset_failure_blocks_both_joint_commands(self):
        for failure in ("joint safety", "offset", "disconnected", "excess velocity", "nan velocity"):
            with self.subTest(failure=failure):
                self.setUp()
                session = self.session()
                velocity = {name: np.full(6, .01) for name in ROBOTS}
                second = self.controls["ur10e"]
                if failure == "joint safety":
                    second.isJointsWithinSafetyLimits.return_value = False
                elif failure == "offset":
                    second.getTCPOffset.return_value = [0.] * 6
                elif failure == "disconnected":
                    second.isConnected.return_value = False
                elif failure == "excess velocity":
                    velocity["ur10e"][0] = self.config.max_joint_speed + .01
                else:
                    velocity["ur10e"][0] = float("nan")
                with self.assertRaises(ControlFault):
                    session.send_joints(velocity, self.solutions)
                self.assert_no_motion()

    def test_joint_send_preserves_distinct_per_arm_accelerations_including_tiny_values(self):
        session = self.session()
        velocity = {name: np.full(6, .01) for name in ROBOTS}
        accelerations = {"ur5e": 1e-10, "ur10e": .035}
        session.send_joints(velocity, self.solutions, accelerations=accelerations)
        for name, control in self.controls.items():
            control.speedJ.assert_called_once_with(velocity[name].tolist(), accelerations[name], .002)
            control.kickWatchdog.assert_not_called()

    def test_invalid_second_arm_acceleration_blocks_both_joint_commands(self):
        for invalid in (0., -1e-10, float("nan"), float("inf"), self.config.max_joint_accel + 1e-8,
                        [.01], None, "missing"):
            with self.subTest(acceleration=invalid):
                self.setUp()
                session = self.session()
                accelerations = {"ur5e": .025, "ur10e": invalid}
                if isinstance(invalid, str) and invalid == "missing":
                    del accelerations["ur10e"]
                with self.assertRaisesRegex(ControlFault, "ur10e:.*acceleration"):
                    session.send_joints({name: np.full(6, .01) for name in ROBOTS}, self.solutions,
                                        accelerations=accelerations)
                self.assert_no_motion()

    def test_joint_command_failure_closes_both_arms_using_joint_deceleration(self):
        session = self.session()
        self.controls["ur10e"].speedJ.return_value = False
        with self.assertRaisesRegex(ControlFault, "speedJ failed"):
            session.send_joints({name: np.zeros(6) for name in ROBOTS}, self.solutions)
        events = []
        for name, control in self.controls.items():
            control.speedStop.side_effect = lambda accel, n=name: events.append(("stop", n)) or True
            control.stopScript.side_effect = lambda n=name: events.append(("script", n)) or True
            control.disconnect.side_effect = lambda n=name: events.append(("disconnect", n))
        session.close()
        self.assertEqual(events[:2], [("stop", "ur5e"), ("stop", "ur10e")])
        for control in self.controls.values():
            control.speedStop.assert_called_once_with(self.config.max_joint_accel)
            control.disconnect.assert_called_once()

    def test_linear_handover_updates_stop_units_individually_even_if_second_arm_not_sent(self):
        session = self.session()
        zero = {name: np.zeros(6) for name in ROBOTS}
        session.send_joints(zero, self.solutions)
        self.controls["ur5e"].speedL.return_value = False
        with self.assertRaisesRegex(ControlFault, "speedL failed"):
            session.send(self.targets, zero)
        session.close()
        self.controls["ur5e"].speedStop.assert_called_once_with(self.config.stop_accel)
        self.controls["ur10e"].speedStop.assert_called_once_with(self.config.max_joint_accel)

    def test_failed_watchdog_enabling_stops_all_connections_without_motion(self):
        session = self.session(deferred=True)
        self.controls["ur10e"].setWatchdog.return_value = False
        with self.assertRaisesRegex(ControlFault, "watchdog"):
            session.enable_watchdog()
        self.assertEqual(session.controls, {})
        self.assert_no_motion()
        for control in self.controls.values():
            control.speedStop.assert_called_once()
            control.disconnect.assert_called_once()

    def test_freshness_failure_before_first_joint_send_blocks_the_pair(self):
        session = self.session()
        with self.assertRaisesRegex(ControlFault, "expired"):
            session.send_joints({name: np.zeros(6) for name in ROBOTS}, self.solutions,
                                check_fresh=Mock(side_effect=ControlFault("expired")))
        self.assert_no_motion()


if __name__ == "__main__":
    unittest.main()
