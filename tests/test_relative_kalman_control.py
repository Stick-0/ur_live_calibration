"""Relative-base estimator/control integration with synthetic encoders only.

The production controller, estimator, damping and motion governor run without
camera, RTDE, checkpoints or hardware. A Cartesian fixture isolates frame and
timing contracts from the separately tested UR native IK implementation.
"""

from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import io
import unittest

import numpy as np

from test_cooperative_controller import accept, packet, scene
from test_relative_kalman import precise
from urcal.control.geometry import flange_from_tcp, inverse, pose_matrix
from urcal.control.handoff import CalibrationHandoff
from urcal.control.runtime import parse_args
from urcal.cooperative.base_compensation import relative_base_compensation
from urcal.cooperative.calibration_braking import frozen_braking_snapshot, prepare_calibration_braking
from urcal.cooperative.controller import ResearchFault


CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]


def kalman_scene(**overrides):
    options = dict(base_motion_mode="relative_kalman", leader_response="bounded",
                   common_base_plane=True, follow_law="feedforward",
                   relative_kalman_parameters=asdict(precise()))
    options.update(overrides)
    return scene(**options)


def moving_scene(*, speed=.03, yaw_rate=0., steps=25, **overrides):
    controller, bases, states = kalman_scene(**overrides)
    reference = bases[controller.leader].copy()
    for index in range(steps):
        elapsed = index * .05
        bases[controller.leader] = reference @ pose_matrix([speed * elapsed, 0., 0., 0., 0., yaw_rate * elapsed])
        accept(controller, bases, at=10. + elapsed, sequence=index + 1)
    return controller, bases, states, 10. + (steps - 1) * .05


class RelativeKalmanControlTests(unittest.TestCase):
    def test_parser_restricts_new_mode_and_keeps_existing_physical_caps(self):
        baseline = parse_args(CLI)
        selected = parse_args(CLI + ["--base_motion_mode", "relative_kalman"])
        for name in ("max_speed", "follow_max_speed", "max_angular_speed", "max_accel",
                     "max_angular_accel", "max_joint_speed", "max_joint_accel",
                     "max_target_m", "min_gap", "max_travel"):
            self.assertEqual(getattr(selected, name), getattr(baseline, name), name)
        self.assertEqual(selected.relative_kalman_parameters["min_velocity_samples"], 4)
        self.assertEqual(selected.joint_history_hz, 100.)
        for invalid in (["--method", "direct"], ["--follow_law", "pose"],
                        ["--leader_response", "strict"], ["--no_common_base_plane"]):
            with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(CLI + ["--base_motion_mode", "relative_kalman"] + invalid)

    def test_constructor_rejects_incompatible_captured_settings(self):
        for invalid in (dict(method="direct"), dict(follow_law="pose"),
                        dict(leader_response="strict"), dict(common_base_plane=False)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                kalman_scene(**invalid)

    def test_same_control_time_relative_base_drives_target_and_current_pose(self):
        controller, bases, states, at = moving_scene(yaw_rate=.02,
            ur5e_tcp_offset=[.04, -.02, .03, .1, 0., .2],
            ur10e_tcp_offset=[-.02, .01, .06, 0., -.1, .1])
        proposal = controller.prepare(at + .07, states, .04, active=True)
        record = proposal.record
        G = record["relative_kalman"]["G_control"]
        measured_bases = record["bases"]
        np.testing.assert_allclose(G, inverse(measured_bases["ur5e"]) @ measured_bases["ur10e"], atol=1e-12)
        leader = flange_from_tcp(states["ur10e"].tcp, controller.offsets["ur10e"])
        follower = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
        np.testing.assert_allclose(record["target_tcp"],
                                   G @ leader @ record["desired_relative_pose"] @ controller.offsets["ur5e"], atol=1e-12)
        np.testing.assert_allclose(record["estimated_relative_pose"], inverse(G @ leader) @ follower, atol=1e-12)
        np.testing.assert_allclose(record["relative_base_target"], G @ leader @ record["tracking_relative_target"], atol=1e-12)
        self.assertAlmostEqual(record["measurement_age"]["ur5e"], .07)
        self.assertEqual(record["state_at"], at + .07)
        self.assertEqual(record["measurement_times"]["ur10e"], at)
        for twist in record["base_twists"].values():
            np.testing.assert_array_equal(twist, np.zeros(6))

    def test_current_flange_lever_and_nonzero_tcp_use_follower_axes(self):
        rotation = pose_matrix([0, 0, 0, .3, -.2, .4])
        G = rotation @ pose_matrix([.8, .2, 0., 0., 0., .1])
        flange = rotation @ pose_matrix([.4, -.1, .6, .2, .1, -.3])
        tcp = flange @ pose_matrix([.06, -.03, .08, 0., .2, 0.])
        velocity = np.r_[rotation[:3, :3] @ [.02, -.01, 0.], rotation[:3, :3] @ [0., 0., .08]]
        result = relative_base_compensation(G, flange, tcp, velocity, .7, np.zeros(6), 1.)
        expected_flange = velocity.copy()
        expected_flange[:3] += np.cross(velocity[3:], flange[:3, 3] - G[:3, 3])
        expected_tcp = .7 * expected_flange
        expected_tcp[:3] += np.cross(expected_tcp[3:], tcp[:3, 3] - flange[:3, 3])
        np.testing.assert_allclose(result["raw_flange"], expected_flange, atol=1e-12)
        np.testing.assert_allclose(result["effective_tcp"], expected_tcp, atol=1e-12)
        np.testing.assert_allclose(result["effective_origin"], .7 * velocity, atol=1e-12)

    def test_compensation_has_independent_norm_caps_and_bounded_decay(self):
        previous = np.array([.03, .04, 0., 0., 0., .06])
        raw = np.array([1., -2., .5, 0., 0., 1.])
        result = relative_base_compensation(np.eye(4), np.eye(4), np.eye(4), raw,
            1., previous, .04, linear_speed=.06, angular_speed=.08, linear_accel=.1, angular_accel=.2)
        self.assertLessEqual(np.linalg.norm(result["effective_flange"][:3]), .06 + 1e-12)
        self.assertLessEqual(np.linalg.norm(result["effective_flange"][3:]), .08 + 1e-12)
        self.assertAlmostEqual(np.linalg.norm(result["effective_flange"][:3] - previous[:3]), .004)
        decayed = relative_base_compensation(np.eye(4), np.eye(4), np.eye(4), raw,
            0., previous, .04, linear_speed=.06, angular_speed=.08, linear_accel=.1, angular_accel=.2)
        self.assertAlmostEqual(np.linalg.norm(decayed["effective_flange"][:3]), .046)
        self.assertAlmostEqual(np.linalg.norm(decayed["effective_flange"][3:]), .052)
        np.testing.assert_array_equal(decayed["requested_flange"], np.zeros(6))

    def test_leader_joint_motion_and_path_derivative_do_not_enter_base_channel(self):
        controller, _bases, states, at = moving_scene(path="line", path_delta=[0., 0., -.02, 0., 0., 0.])
        first = controller.prepare(at, states, .04, active=True)
        states["ur10e"].joint_velocity[0] = .02
        states["ur10e"].speed[0] = .02
        controller.progress = .4
        next_proposal = controller.prepare(at, states, .04, active=True)
        np.testing.assert_array_equal(first.record["base_compensation"], next_proposal.record["base_compensation"])
        np.testing.assert_array_equal(first.record["relative_kalman"]["state"], next_proposal.record["relative_kalman"]["state"])
        self.assertGreater(np.linalg.norm(next_proposal.record["leader_compensation"]), .019)
        self.assertGreater(np.linalg.norm(next_proposal.record["task_twist_at_command"]), 0.)
        record = next_proposal.record
        np.testing.assert_allclose(record["requested_twist"], record["base_compensation"] +
            record["leader_compensation"] + record["feedback"] + record["requested_task_scale"] * record["task_twist_at_command"])
        np.testing.assert_array_equal(record["applied_compensation"], np.zeros(6))

    def test_accepted_flange_offset_remains_in_target_without_entering_base_velocity(self):
        controller, _bases, states, at = moving_scene(target_tolerance_m=.05, target_tolerance_deg=10.,
            ur5e_tcp_offset=[.03, -.02, .04, 0., 0., 0.])
        proposal = controller.prepare(at, states, .04, active=True)
        controller.commit(proposal)
        self.assertGreater(np.linalg.norm(controller.tracking_offset[:3, 3]), .03)
        offset = controller.tracking_offset.copy()
        states["ur10e"].joints[1] += .005
        states["ur10e"].tcp = controller.models["ur10e"].pose_and_jacobian(states["ur10e"].joints)[0]
        record = controller.prepare(at + .01, states, .04, active=True).record
        expected_relative = controller.path.pose(controller.progress) @ offset
        leader_flange = flange_from_tcp(states["ur10e"].tcp, controller.offsets["ur10e"])
        np.testing.assert_allclose(record["tracking_relative_target"], expected_relative, atol=1e-12)
        np.testing.assert_allclose(record["relative_base_target"],
            record["relative_kalman"]["G_control"] @ leader_flange @ expected_relative, atol=1e-12)
        np.testing.assert_array_equal(controller.tracking_offset, offset)

    def test_approach_changes_relative_path_only_and_pause_keeps_base_following(self):
        controller, _bases, states, at = moving_scene(approach_speed=.01)
        first = controller.prepare(at, states, .04, active=True)
        controller.commit(first)
        # Start/pause actions must not consume a compensation slew step or
        # silently reset the already transmitted total command.
        history, command = controller.previous_relative_base_flange.copy(), controller.previous_twist.copy()
        controller.handle_gap_approach("start", enabled=True)
        np.testing.assert_array_equal(controller.previous_relative_base_flange, history)
        np.testing.assert_array_equal(controller.previous_twist, command)
        self.assertAlmostEqual(controller.path.pose(1.)[2, 3], .15)
        states["ur5e"].speed[:] = command
        states["ur5e"].joint_velocity[:] = controller.previous_qdot
        running = controller.prepare(at, states, .04, active=True)
        controller.handle_gap_approach("pause", enabled=True)
        paused = controller.prepare(at, states, .04, active=True)
        self.assertEqual(paused.next_progress, controller.progress)
        self.assertEqual(paused.record["requested_task_scale"], 0.)
        np.testing.assert_array_equal(paused.record["base_compensation"], running.record["base_compensation"])
        self.assertGreater(np.linalg.norm(paused.record["base_compensation"]), 0.)

    def test_damping_subtracts_exact_limited_tcp_compensation(self):
        controller, _bases, states, at = moving_scene(yaw_rate=.04, follow_damping_time=.12,
            path="line", path_delta=[0., 0., -.02, 0., 0., 0.],
            ur5e_tcp_offset=[.06, .03, .04, 0., 0., 0.])
        states["ur5e"].speed[:] = [.01, -.004, .001, 0., 0., .005]
        states["ur5e"].joint_velocity[:] = states["ur5e"].speed
        states["ur10e"].joint_velocity[1] = .008
        controller.previous_task_scale = .6
        proposal = controller.prepare(at, states, .04, active=True)
        record = proposal.record
        effective = record["relative_base_compensation"]["effective_tcp"]
        self.assertTrue(record["relative_base_compensation"]["limited"])
        self.assertGreater(np.linalg.norm(effective - record["base_compensation_nominal"]), .01)
        np.testing.assert_array_equal(record["base_compensation"], effective)
        np.testing.assert_allclose(record["feedback_relative_tcp_velocity"], states["ur5e"].speed - effective
            - record["leader_compensation"] - .6 * record["task_twist_at_command"], atol=1e-12)

    def test_preparing_and_rejecting_does_not_advance_filter_or_command_history(self):
        controller, _bases, states, at = moving_scene(path="line")
        original_state, original_covariance = controller.estimator._x.copy(), controller.estimator._P.copy()
        first = controller.prepare(at, states, .04)
        second = controller.prepare(at, states, .04, active=True)
        np.testing.assert_array_equal(first.record["base_compensation"], second.record["base_compensation"])
        self.assertEqual(controller.progress, 0.)
        np.testing.assert_array_equal(controller.previous_relative_base_flange, np.zeros(6))
        np.testing.assert_array_equal(controller.previous_twist, np.zeros(6))
        np.testing.assert_array_equal(controller.estimator._x, original_state)
        np.testing.assert_array_equal(controller.estimator._P, original_covariance)
        controller.guard.allowed = False
        with self.assertRaises(ResearchFault):
            controller.prepare(at, states, .04, active=True)
        np.testing.assert_array_equal(controller.previous_relative_base_flange, np.zeros(6))
        controller.guard.allowed = True
        controller.commit(second)
        np.testing.assert_array_equal(controller.previous_relative_base_flange,
                                      second.record["relative_base_compensation"]["effective_flange"])
        np.testing.assert_array_equal(controller.previous_twist, second.decision.twist)

    def test_common_camera_motion_is_not_an_individual_base_jump(self):
        controller, bases, states = kalman_scene()
        accept(controller, bases)
        before = controller.prepare(10., states, .04).record
        camera_motion = pose_matrix([.5, -.3, .2, .3, -.2, .4])
        moved = {name: camera_motion @ pose for name, pose in bases.items()}
        accept(controller, moved, 10.05, 2)
        after = controller.prepare(10.05, states, .04, active=True).record
        self.assertFalse(controller.gate.check_individual_jumps)
        np.testing.assert_allclose(after["relative_kalman"]["G_control"], before["relative_kalman"]["G_control"], atol=1e-12)
        np.testing.assert_allclose(after["target_tcp"], before["target_tcp"], atol=1e-12)
        np.testing.assert_allclose(after["base_compensation"], np.zeros(6), atol=1e-12)

    def test_rejected_observation_drops_confidence_without_freshening_and_decays(self):
        controller, bases, states, at = moving_scene()
        controller.commit(controller.prepare(at, states, .04, active=True))
        previous = controller.previous_relative_base_flange.copy()
        states["ur5e"].speed[:] = controller.previous_twist
        states["ur5e"].joint_velocity[:] = controller.previous_qdot
        self.assertFalse(controller.ingest(packet(controller, bases, at + .01, 26, reason="missing capture bracket"), at + .02))
        proposal = controller.prepare(at + .03, states, .01, active=True)
        data = proposal.record["relative_base_compensation"]
        self.assertEqual(proposal.record["relative_kalman"]["lambda"], 0.)
        self.assertEqual(proposal.record["measurement_times"]["ur5e"], at)
        self.assertLess(np.linalg.norm(data["effective_flange"]), np.linalg.norm(previous))
        self.assertLessEqual(np.linalg.norm(data["effective_flange"][:3] - previous[:3]),
                             controller.args.relative_base_accel * .01 + 1e-12)
        np.testing.assert_array_equal(data["requested_flange"], np.zeros(6))

    def test_capture_gap_reinitializes_velocity_before_new_compensation(self):
        controller, bases, states, at = moving_scene()
        accept(controller, bases, at + .4, 26)
        record = controller.prepare(at + .4, states, .04, active=True).record
        self.assertEqual(record["relative_kalman"]["support"], 1)
        self.assertEqual(record["relative_kalman"]["lambda"], 0.)
        np.testing.assert_array_equal(record["base_compensation"], np.zeros(6))
        np.testing.assert_array_equal(record["relative_kalman"]["state"][3:], np.zeros(3))

    def test_recovery_blend_rebuilds_G_and_suppresses_base_feedforward(self):
        controller, bases, states, at = moving_scene(live_calibration_recovery=True)
        before = controller.state(at, active=True)
        controller.live_calibration.start(before, at, "synthetic rejected observation")
        controller.live_calibration.complete(at + .01)
        bases["ur10e"][0, 3] += .0015
        accept(controller, bases, at + .05, 26)
        record = controller.prepare(at + .06, states, .04, active=True).record
        self.assertTrue(controller.live_calibration.blending)
        self.assertEqual(record["relative_kalman"]["lambda"], 0.)
        np.testing.assert_array_equal(record["base_compensation"], np.zeros(6))
        np.testing.assert_allclose(record["relative_kalman"]["G_control"],
            inverse(record["bases"]["ur5e"]) @ record["bases"]["ur10e"], atol=1e-12)
        self.assertEqual(record["measurement_times"]["ur5e"], at + .05)
        self.assertIn("blending", record["relative_kalman"]["reason"])

    def test_frozen_braking_preserves_capture_and_zeroes_new_velocity(self):
        controller, _bases, states, at = moving_scene()
        controller.commit(controller.prepare(at, states, .04, active=True))
        snapshot = controller.state(at, active=True)
        frozen = frozen_braking_snapshot(snapshot, at + .08, leader=controller.leader)
        self.assertEqual(frozen.measurement_times, snapshot.measurement_times)
        self.assertAlmostEqual(frozen.measurement_age["ur5e"], .08)
        self.assertEqual(frozen.relative_filter["lambda"], 0.)
        np.testing.assert_array_equal(frozen.relative_filter["raw_origin_velocity"], np.zeros(6))
        proposal = prepare_calibration_braking(controller, at + .08, states, .04, snapshot)
        self.assertTrue(proposal.record["relative_base_compensation"]["held_for_braking"])
        np.testing.assert_array_equal(proposal.record["base_compensation"], np.zeros(6))
        self.assertEqual(proposal.next_progress, controller.progress)
        controller.commit(proposal)
        np.testing.assert_array_equal(controller.previous_relative_base_flange, np.zeros(6))

    def test_handoff_preserves_mode_metadata_and_requires_post_endpoint_images(self):
        for consensus in (False, True):
            with self.subTest(consensus=consensus):
                controller, bases, _states = kalman_scene()
                accept(controller, bases)
                handoff = CalibrationHandoff(controller.gate, 10., estimator_factory=controller._new_estimator,
                    mappings=controller.mappings, leader=controller.leader, consensus=consensus)
                self.assertFalse(handoff.ingest(packet(controller, bases, 10., 2), 10.02))
                self.assertTrue(handoff.ingest(packet(controller, bases, 10.05, 3), 10.08))
                self.assertTrue(handoff.ready(10.08))
                self.assertFalse(handoff.tracker.check_individual_jumps)
                result = handoff.estimator.estimate(10.08)
                self.assertEqual(result.measurement_times["ur5e"], 10.05)
                self.assertEqual(result.relative_filter["lambda"], 0.)
                self.assertEqual(result.relative_filter["support"], 1)
                np.testing.assert_array_equal(result.relative_filter["raw_origin_velocity"], np.zeros(6))

    def test_rearm_clears_new_history_without_resetting_task_or_travel_origin(self):
        controller, bases, states, at = moving_scene(path="line")
        controller.commit(controller.prepare(at, states, .04, active=True))
        origin, progress = controller.initial_tcp.copy(), controller.progress
        controller.reset_after_stop()
        np.testing.assert_array_equal(controller.previous_relative_base_flange, np.zeros(6))
        np.testing.assert_array_equal(controller.initial_tcp, origin)
        self.assertEqual(controller.progress, progress)
        self.assertFalse(controller.gate.check_individual_jumps)
        with self.assertRaises(ResearchFault):
            controller.prepare(at + .1, states, .04, active=True)
        accept(controller, bases, at + .2, 26)
        record = controller.prepare(at + .2, states, .04, active=True).record
        self.assertEqual(record["relative_kalman"]["lambda"], 0.)
        np.testing.assert_array_equal(record["base_compensation"], np.zeros(6))

    def test_delayed_slow_base_plant_tracks_with_unchanged_cli_caps(self):
        baseline = parse_args(CLI)
        cap_names = ("max_speed", "follow_max_speed", "max_angular_speed", "max_accel",
                     "max_angular_accel", "max_joint_speed", "max_joint_accel", "follow_damping_time",
                     "kp_linear", "kp_angular", "feedback_speed", "feedback_angular_speed")
        controller, bases, states = kalman_scene(**{name: getattr(baseline, name) for name in cap_names})
        initial_base = bases["ur10e"].copy()
        dt, latency, speed = .04, .08, .02
        last_command = np.zeros(6)
        max_error, max_command = 0., 0.
        for index in range(180):
            elapsed = index * dt
            captured_elapsed = max(0., elapsed - latency)
            bases["ur10e"] = initial_base @ pose_matrix([speed * min(captured_elapsed, 3.), 0., 0., 0., 0., 0.])
            # Capture timestamps advance during the two initial stationary
            # images as well; the movement starts only at elapsed zero.
            captured = 10. + elapsed - latency
            accept(controller, bases, captured, index + 1, received=10. + elapsed)
            proposal = controller.prepare(10. + elapsed, states, dt, active=True)
            command = proposal.decision.twist
            self.assertLessEqual(np.linalg.norm(command[:3]), baseline.follow_max_speed + 1e-10)
            self.assertLessEqual(np.linalg.norm(command[3:]), baseline.max_angular_speed + 1e-10)
            self.assertLessEqual(np.max(np.abs(proposal.decision.qdot)), baseline.max_joint_speed + 1e-10)
            self.assertLessEqual(np.linalg.norm(command[:3] - last_command[:3]), baseline.max_accel * dt + 1e-9)
            self.assertLessEqual(np.max(np.abs(proposal.decision.qdot - controller.previous_qdot)), baseline.max_joint_accel * dt + 1e-9)
            controller.commit(proposal)
            # A measured first-order plant (80 ms response) is independent of
            # the estimator and command model. Only translational q is moved.
            states["ur5e"].speed += .5 * (command - states["ur5e"].speed)
            states["ur5e"].joint_velocity[:] = states["ur5e"].speed
            states["ur5e"].joints[:3] += states["ur5e"].speed[:3] * dt
            states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
            for robot in states.values():
                robot.robot_timestamp += dt
            truth_x = speed * min(elapsed + dt, 3.)
            max_error = max(max_error, abs(truth_x - states["ur5e"].tcp[0, 3]))
            max_command = max(max_command, abs(command[0]))
            last_command = command.copy()
        self.assertLess(max_error, .015)
        self.assertGreater(max_command, .015, "plant must actually follow rather than remain stopped")
        self.assertLess(abs(.06 - states["ur5e"].tcp[0, 3]), .001)


if __name__ == "__main__":
    unittest.main()
