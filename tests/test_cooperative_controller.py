"""Production cooperative engine with an ideal Cartesian plant, no hardware."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))

from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.task import MotionInfeasible
from urcal.control.geometry import flange_from_tcp, inverse, matrix_pose, pose_matrix
from urcal.control.policy import CalibrationFrame, ControlFault


ROBOTS = ("ur5e", "ur10e")


class CartesianModel:
    """Six virtual Cartesian joints; tests use translations or static rotations."""

    def __init__(self, offset):
        self.offset = offset
        self.jacobian = np.eye(6)

    def pose_and_jacobian(self, joints, flange=False):
        flange_pose = pose_matrix(joints)
        result = flange_pose if flange else flange_pose @ self.offset
        return result, self.jacobian.copy()

    def inverse(self, target, near, lower, upper):
        joints = matrix_pose(target @ inverse(self.offset))
        return joints if np.all(joints >= lower) and np.all(joints <= upper) else None


class PermissiveGuard:
    def __init__(self):
        self.lower = {name: np.full(6, -10.) for name in ROBOTS}
        self.upper = {name: np.full(6, 10.) for name in ROBOTS}
        self.allowed = True

    def check(self, bases, joints):
        return self.allowed, "synthetic guard rejected scene" if not self.allowed else "", 1.

    def pose_allowed(self, name, pose):
        return self.allowed


def settings(**overrides):
    args = parse_args(["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"])
    # Explicit research parameters keep this numerical fixture independent of
    # CLI safety defaults while still checking the production parser baseline.
    values = dict(
        leader="ur10e", mode="follow", method="separated", path="hold", gap=.3, leader_response="strict",
        path_delta=[0., 0., -.05, 0., 0., 0.], path_radius=.02, path_duration=10.,
        ur5e_tcp_offset=[0.] * 6, ur10e_tcp_offset=[0.] * 6,
        workspace_config=None, stable_frames=1, filter_alpha=1., velocity_alpha=1.,
        prediction_horizon=.3, max_calibration_age=.5, max_invalid_duration=.1,
        max_base_speed=.15, max_base_angular_speed=.5,
        max_calibration_jump_m=.015, max_calibration_jump_deg=2.,
        handoff_max_shift_m=0., handoff_max_shift_deg=0., handoff_timeout=5.,
        max_speed=.2, follow_max_speed=None, max_angular_speed=.5, max_accel=2., max_angular_accel=4.,
        max_joint_speed=1., max_joint_accel=10., min_singular_value=.01,
        max_travel=.3, max_target_m=.1, max_target_deg=25.,
        frame_check_m=.03, frame_check_deg=5.,
        feedback_speed=.05, feedback_angular_speed=.2,
        position_deadband=.0001, rotation_deadband_deg=.01,
        target_tolerance_m=0., target_tolerance_deg=0.,
        slow_error_m=.03, slow_error_deg=10.,
    )
    values.update(overrides)
    for name, value in values.items():
        setattr(args, name, value)
    return args


def scene(**overrides):
    args = settings(**overrides)
    offsets = {name: pose_matrix(getattr(args, name + "_tcp_offset")) for name in ROBOTS}
    models = {name: CartesianModel(offsets[name]) for name in ROBOTS}
    guard = PermissiveGuard()
    controller = CooperativeController(args, models=models, guard=guard)
    bases = {name: pose_matrix([.2, -.1, .5, 0., 0., 0.]) for name in ROBOTS}
    flanges = {"ur10e": pose_matrix([0., 0., .4, 0., 0., 0.]),
               "ur5e": pose_matrix([0., 0., .7, np.pi, 0., 0.])}
    states = {
        name: SimpleNamespace(tcp=flange @ offsets[name], joints=matrix_pose(flange),
                              speed=np.zeros(6), joint_velocity=np.zeros(6), robot_timestamp=50.)
        for name, flange in flanges.items()
    }
    return controller, bases, states


def packet(controller, bases, at=10., sequence=1, reason=""):
    return CalibrationFrame(
        sequence=sequence, captured_at=at,
        poses={name: pose @ inverse(controller.mappings[name]) for name, pose in bases.items()},
        raw_mean_px={name: 0. for name in ROBOTS}, raw_max_px={name: 0. for name in ROBOTS},
        valid_counts={name: 7 for name in ROBOTS}, reason=reason,
    )


def accept(controller, bases, at=10., sequence=1, received=None):
    accepted = controller.ingest(packet(controller, bases, at, sequence), at if received is None else received)
    if not accepted:
        raise AssertionError("fixture calibration rejected: " + controller.last_rejection)


class CooperativeControllerTests(unittest.TestCase):
    def assert_target_relation(self, controller, proposal, states):
        bases = proposal.record["bases"]
        leader_flange = flange_from_tcp(states[controller.leader].tcp, controller.offsets[controller.leader])
        follower_target = flange_from_tcp(proposal.tcp_goals[controller.follower], controller.offsets[controller.follower])
        actual_target = inverse(bases[controller.leader] @ leader_flange) @ bases[controller.follower] @ follower_target
        np.testing.assert_allclose(actual_target, controller.path.pose(controller.progress), atol=1e-12)

    def test_fixed_relative_target_survives_either_base_shift_and_leader_joint_motion(self):
        for moving in ROBOTS:
            with self.subTest(moving=moving):
                controller, bases, states = scene(method="direct")
                start = controller.path.start.copy()
                accept(controller, bases)
                first = controller.prepare(10., states, .1)
                self.assert_target_relation(controller, first, states)
                bases[moving][0, 3] += .005
                accept(controller, bases, 10.1, 2)
                moved = controller.prepare(10.1, states, .1, active=True)
                self.assert_target_relation(controller, moved, states)
                self.assertAlmostEqual(moved.tcp_goals["ur5e"][0, 3], -.005 if moving == "ur5e" else .005)
                states["ur10e"].joints[1] += .004
                states["ur10e"].tcp = controller.models["ur10e"].pose_and_jacobian(states["ur10e"].joints)[0]
                states["ur5e"].tcp[0, 3] += .002  # Actual following error must not redefine the goal.
                states["ur5e"].joints[0] += .002
                following = controller.prepare(10.11, states, .1, active=True)
                self.assert_target_relation(controller, following, states)
                self.assertAlmostEqual(following.tcp_goals["ur5e"][1, 3], .004)
                np.testing.assert_array_equal(controller.path.start, start)

    def test_tracking_fault_reports_actual_tcp_errors_limits_and_independent_poses(self):
        for displacement, rotation in ((.12, 0.), (0., .6), (.12, .6)):
            with self.subTest(displacement=displacement, rotation=rotation):
                controller, bases, states = scene(method="direct")
                accept(controller, bases)
                states["ur5e"].joints[0] += displacement
                states["ur5e"].joints[3] += rotation
                states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
                with self.assertRaises(ResearchFault) as caught:
                    controller.prepare(10., states, .04)
                fault = caught.exception
                self.assertEqual(fault.category, "tracking_error")
                detail = fault.details
                self.assertEqual(detail["frame"], "ur5e_controller_base")
                self.assertEqual((detail["leader"], detail["follower"]), ("ur10e", "ur5e"))
                self.assertAlmostEqual(detail["position_error_m"], displacement)
                self.assertAlmostEqual(detail["rotation_error_rad"], rotation)
                self.assertAlmostEqual(detail["position_limit_m"], .1)
                self.assertAlmostEqual(detail["rotation_limit_rad"], np.deg2rad(25.))
                self.assertEqual(detail["position_exceeded"], displacement > .1)
                self.assertEqual(detail["rotation_exceeded"], rotation > np.deg2rad(25.))
                np.testing.assert_allclose(detail["target_tcp"], pose_matrix([0, 0, .7, np.pi, 0, 0]), atol=1e-12)
                np.testing.assert_array_equal(detail["current_tcp"], states["ur5e"].tcp)
                self.assertIn(f"{displacement * 1000:.1f}/100.0 mm", str(fault))
                self.assertIn(f"{np.rad2deg(rotation):.1f}/25.0 deg", str(fault))
                self.assertEqual(controller.progress, 0.)
                self.assertFalse(np.any(controller.previous_twist))
                # A logger or later caller cannot mutate the actual state/goal.
                detail["current_tcp"][0, 3] = 900.
                detail["desired_relative_flange"][0, 3] = 900.
                self.assertAlmostEqual(states["ur5e"].tcp[0, 3], displacement)
                self.assertEqual(controller.path.start[0, 3], 0.)

    def test_stale_wait_keeps_last_rejected_pair_reason_and_capture_age(self):
        controller, bases, states = scene()
        accept(controller, bases)
        bases["ur10e"][0, 3] += .016
        self.assertFalse(controller.ingest(packet(controller, bases, 10.1, 2), 10.11))
        rejection = controller.last_rejection
        self.assertIn("calibration jump", rejection)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10.803, states, .04, active=False)
        fault = caught.exception
        self.assertEqual(fault.category, "measurement_unavailable")
        self.assertIn("stale (0.803s > 0.500s)", str(fault))
        self.assertIn("last rejected pair: " + rejection, str(fault))
        self.assertEqual(fault.details["last_accepted_capture"], 10.)
        self.assertAlmostEqual(fault.details["last_accepted_age"], .803)
        self.assertEqual(fault.details["last_rejection"], rejection)
        # Reporting does not accept rejected calibration or alter the limits.
        self.assertEqual(controller.gate.last_capture, 10.)
        self.assertEqual(controller.gate.max_jump_m, .015)
        self.assertEqual(controller.gate.max_age, .5)

    def test_target_uses_controller_actual_tcp_with_nonzero_tool_offsets(self):
        controller, bases, states = scene(
            method="direct", ur10e_tcp_offset=[.02, .01, .05, .1, .2, -.1],
            ur5e_tcp_offset=[-.01, .02, .03, -.1, .2, .1])
        accept(controller, bases)
        states["ur10e"].tcp[0, 3] += .003  # Within nominal-vs-calibrated FK tolerance.
        proposal = controller.prepare(10., states, .1)
        self.assert_target_relation(controller, proposal, states)
        self.assertAlmostEqual(proposal.goals["ur5e"][0, 3], bases["ur10e"][0, 3] + .003)
        recorded = proposal.record["robots"]["ur10e"]["tcp"]
        np.testing.assert_array_equal(recorded, states["ur10e"].tcp)

    def test_fixed_relative_target_survives_either_base_rotation(self):
        for moving in ROBOTS:
            with self.subTest(moving=moving):
                controller, bases, states = scene(method="direct")
                accept(controller, bases)
                controller.prepare(10., states, .1)
                bases[moving][:3, :3] = pose_matrix([0, 0, 0, 0, .01, 0])[:3, :3]
                accept(controller, bases, 10.1, 2)
                rotated = controller.prepare(10.1, states, .1, active=True)
                self.assert_target_relation(controller, rotated, states)

    def test_raw_capture_receipt_and_control_times_stay_distinct(self):
        controller, bases, states = scene()
        accept(controller, bases, at=10., received=10.08)
        proposal = controller.prepare(10.12, states, .04)
        self.assertEqual(proposal.record["at"], 10.12)
        self.assertEqual(proposal.record["state_at"], 10.12)
        self.assertEqual(proposal.record["measurement_times"], {name: 10. for name in ROBOTS})
        self.assertEqual(proposal.record["received_times"], {name: 10.08 for name in ROBOTS})
        for name in ROBOTS:
            self.assertAlmostEqual(proposal.record["measurement_age"][name], .12)
            self.assertAlmostEqual(proposal.record["prediction_dt"][name], .12)

    def test_preview_prepares_do_not_advance_until_successful_command_commit(self):
        controller, bases, states = scene(path="line")
        accept(controller, bases)
        for now in (10., 10.04, 10.08):
            proposal = controller.prepare(now, states, .04, active=False)
            self.assertGreater(proposal.next_progress, 0.)
            self.assertEqual(controller.progress, 0.)
            np.testing.assert_array_equal(controller.previous_twist, np.zeros(6))
        # Runtime deliberately discards a proposal when a send fails.
        discarded = controller.prepare(10.1, states, .04, active=True)
        self.assertEqual(controller.progress, 0.)
        committed = controller.prepare(10.12, states, .04, active=True)
        controller.commit(committed)
        self.assertEqual(controller.progress, committed.next_progress)
        self.assertEqual(controller.progress, discarded.next_progress)
        np.testing.assert_array_equal(controller.previous_twist, committed.decision.twist)
        np.testing.assert_array_equal(controller.previous_qdot, committed.decision.qdot)

    def test_short_invalid_measurement_freezes_task_but_keeps_base_compensation(self):
        controller, bases, states = scene(path="line")
        accept(controller, bases)
        bases["ur10e"][0, 3] += .002
        accept(controller, bases, 10.1, 2)
        good = controller.prepare(10.1, states, .1, active=True)
        controller.commit(good)
        progress = controller.progress
        self.assertFalse(controller.ingest(packet(controller, bases, 10.11, 3, reason="synthetic bad image"), 10.12))
        fallback = controller.prepare(10.15, states, .05, active=True)
        self.assertEqual(fallback.next_progress, progress)
        self.assertEqual(fallback.decision.task_scale, 0.)
        np.testing.assert_allclose(fallback.compensation, [.02, 0, 0, 0, 0, 0], atol=1e-12)
        self.assertGreater(np.linalg.norm(fallback.twists["ur5e"]), 0.)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10.23, states, .04, active=True)
        self.assertEqual(caught.exception.category, "measurement_invalid")

    def test_invalid_preview_cannot_use_active_prediction_grace(self):
        controller, bases, states = scene()
        accept(controller, bases)
        controller.ingest(packet(controller, bases, 10.01, 2, reason="invalid"), 10.02)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10.04, states, .04, active=False)
        self.assertEqual(caught.exception.category, "measurement_unavailable")

    def test_initial_quality_rejection_remains_visible_until_new_pair_is_accepted(self):
        controller, bases, states = scene(check_reprojection=True)
        bad = packet(controller, bases)
        bad.raw_mean_px["ur5e"] = 4.2
        bad.raw_max_px["ur5e"] = 12.5
        self.assertFalse(controller.ingest(bad, 10.01))
        reason = controller.last_rejection
        self.assertIn("mean=4.20px limit=5.00px", reason)
        self.assertIn("max=12.50px limit=8.00px", reason)
        for now in (10.02, 10.04, 10.08):
            with self.assertRaises(ResearchFault) as caught:
                controller.prepare(now, states, .02, active=False)
            self.assertEqual(caught.exception.category, "measurement_unavailable")
            self.assertEqual(str(caught.exception), reason)
        self.assertEqual(controller.gate.poses, {})
        self.assertIsNone(controller.gate.last_capture)
        accept(controller, bases, 10.1, 2)
        self.assertEqual(controller.last_rejection, "")
        self.assertEqual(controller.gate.last_rejection, "")
        controller.prepare(10.11, states, .02)

    def test_disabled_reprojection_accepts_large_errors_for_preview_and_active_planning(self):
        for active in (False, True):
            with self.subTest(active=active):
                controller, bases, states = scene()
                self.assertFalse(controller.gate.check_reprojection)
                sample = packet(controller, bases)
                sample.raw_mean_px.update({name: 100. for name in ROBOTS})
                sample.raw_max_px.update({name: 300. for name in ROBOTS})
                self.assertTrue(controller.ingest(sample, 10.01))
                proposal = controller.prepare(10.02, states, .02, active=active)
                self.assert_target_relation(controller, proposal, states)
                self.assertEqual(controller.last_rejection, "")

    def test_disabled_reprojection_rejects_nonfinite_statistics_before_estimation(self):
        controller, bases, states = scene()
        sample = packet(controller, bases)
        sample.raw_mean_px["ur5e"] = float("nan")
        self.assertFalse(controller.ingest(sample, 10.01))
        self.assertIsNone(controller.gate.last_capture)
        self.assertEqual(controller.gate.poses, {})
        with self.assertRaisesRegex(ResearchFault, "finite"):
            controller.prepare(10.02, states, .02, active=True)

    def test_velocity_rejection_does_not_poison_quality_gate_recovery(self):
        controller, bases, states = scene(max_calibration_jump_m=.015, max_base_speed=.15)
        accept(controller, bases)
        initial = bases["ur5e"].copy()
        bases["ur5e"][0, 3] += .014  # Allowed single-frame jump, impossible .7 m/s.
        self.assertFalse(controller.ingest(packet(controller, bases, 10.02, 2), 10.02))
        self.assertIn("speed", controller.last_rejection)
        bases["ur5e"] = initial.copy()
        bases["ur5e"][0, 3] -= .002  # Valid -.02 m/s since the last accepted sample.
        self.assertTrue(controller.ingest(packet(controller, bases, 10.1, 3), 10.1), controller.last_rejection)
        proposal = controller.prepare(10.1, states, .1)
        self.assertEqual(proposal.record["measurement_times"], {name: 10.1 for name in ROBOTS})

    def test_preview_reacquires_only_after_five_new_frames_and_replaces_both_states_atomically(self):
        controller, bases, states = scene(stable_frames=5)
        for sequence, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
            accept(controller, bases, at, sequence)
        old_gate, old_estimator = controller.gate, controller.estimator
        old_bases = deepcopy(bases)
        bases["ur5e"][0, 3] += .2
        for index in range(5):
            at = 10.6 + index * .04
            accepted = controller.ingest(packet(controller, bases, at, index + 6), at, allow_reacquire=True)
            self.assertEqual(accepted, index == 4)
            if index < 4:
                self.assertEqual(controller.preview_recovery.count, index + 1)
                self.assertIs(controller.gate, old_gate)
                self.assertIs(controller.estimator, old_estimator)
                self.assertEqual(controller.gate.last_capture, 10.)
                for name in ROBOTS:
                    np.testing.assert_allclose(controller.gate.poses[name] @ controller.mappings[name], old_bases[name])
                with self.assertRaises(ResearchFault):
                    controller.state(at, active=False)
        self.assertIsNot(controller.gate, old_gate)
        self.assertIsNot(controller.estimator, old_estimator)
        self.assertEqual(controller.reacquisition_count, 1)
        self.assertTrue(controller.gate.ready)
        self.assertIsNone(controller.invalid_since)
        self.assertEqual(controller.last_rejection, "")
        snapshot = controller.state(10.76, active=False)
        for name in ROBOTS:
            np.testing.assert_allclose(snapshot.base_poses[name], bases[name], atol=1e-12)
            np.testing.assert_allclose(snapshot.base_twists[name], np.zeros(6), atol=1e-12)
        self.assertEqual(controller.progress, 0.)

    def test_preview_recovery_keeps_cluster_velocity_without_interpreting_the_baseline_shift_as_motion(self):
        controller, bases, states = scene(stable_frames=5, max_base_speed=.15)
        for sequence, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
            accept(controller, bases, at, sequence)
        original_x = bases["ur5e"][0, 3]
        for index in range(5):
            bases["ur5e"][0, 3] = original_x + .2 + index * .001
            at = 10.6 + index * .05
            accepted = controller.ingest(packet(controller, bases, at, index + 6), at, allow_reacquire=True)
        self.assertTrue(accepted, controller.last_rejection)
        snapshot = controller.state(10.8, active=False)
        np.testing.assert_allclose(snapshot.base_twists["ur5e"], [.02, 0, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(snapshot.base_twists["ur10e"], np.zeros(6), atol=1e-12)
        self.assertEqual(controller.reacquisition_count, 1)

    def test_preview_candidate_still_rejects_excessive_motion_inside_the_new_cluster(self):
        for movement in ("translation", "rotation"):
            with self.subTest(movement=movement):
                controller, bases, states = scene(stable_frames=5)
                for sequence, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
                    accept(controller, bases, at, sequence)
                old_gate, old_estimator = controller.gate, controller.estimator
                bases["ur5e"][0, 3] += .2
                cluster_start = bases["ur5e"].copy()
                for index in range(5):
                    # All samples remain within the 15mm / 2deg cluster
                    # envelope, but their 5ms intervals imply excessive speed.
                    increment = ([index * .003, 0, 0, 0, 0, 0] if movement == "translation"
                                 else [0, 0, 0, 0, 0, index * .007])
                    bases["ur5e"] = cluster_start @ pose_matrix(increment)
                    at = 10.6 + index * .005
                    self.assertFalse(controller.ingest(packet(controller, bases, at, index + 6), at,
                                                       allow_reacquire=True))
                self.assertIn("speed", controller.last_rejection)
                self.assertIs(controller.gate, old_gate)
                self.assertIs(controller.estimator, old_estimator)
                self.assertEqual(controller.gate.last_capture, 10.)
                self.assertEqual(controller.reacquisition_count, 0)

    def test_default_strict_ingest_cannot_finish_a_partial_preview_reacquisition(self):
        controller, bases, states = scene(stable_frames=5)
        for sequence, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
            accept(controller, bases, at, sequence)
        old_gate, old_estimator = controller.gate, controller.estimator
        bases["ur5e"][0, 3] += .2
        for index in range(4):
            at = 10.6 + index * .04
            self.assertFalse(controller.ingest(packet(controller, bases, at, index + 6), at,
                                               allow_reacquire=True))
        self.assertEqual(controller.preview_recovery.count, 4)
        for index in range(5):
            at = 10.8 + index * .04
            self.assertFalse(controller.ingest(packet(controller, bases, at, index + 10), at))
        self.assertEqual(controller.preview_recovery.count, 0)
        self.assertEqual(controller.reacquisition_count, 0)
        self.assertIs(controller.gate, old_gate)
        self.assertIs(controller.estimator, old_estimator)
        self.assertEqual(controller.gate.last_capture, 10.)

    def test_preview_cannot_replace_a_still_fresh_baseline_with_a_jump_cluster(self):
        controller, bases, states = scene(stable_frames=5)
        for sequence, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
            accept(controller, bases, at, sequence)
        old_gate, old_estimator = controller.gate, controller.estimator
        bases["ur5e"][0, 3] += .2
        for index in range(5):
            at = 10.1 + index * .04
            self.assertFalse(controller.ingest(packet(controller, bases, at, index + 6), at,
                                               allow_reacquire=True))
        self.assertEqual(controller.preview_recovery.count, 0)
        self.assertEqual(controller.reacquisition_count, 0)
        self.assertIs(controller.gate, old_gate)
        self.assertIs(controller.estimator, old_estimator)

    def test_prediction_expiration_is_a_measurement_fault(self):
        controller, bases, states = scene()
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10.31, states, .04, active=True)
        self.assertEqual(caught.exception.category, "measurement_expired")

    def test_pause_preserves_compensation_and_freezes_committed_progress(self):
        controller, bases, states = scene(path="circle")
        accept(controller, bases)
        bases["ur5e"][0, 3] += .002
        accept(controller, bases, 10.1, 2)
        controller.task_paused = True
        paused = controller.prepare(10.1, states, .1, active=True)
        np.testing.assert_allclose(paused.compensation, [-.02, 0, 0, 0, 0, 0], atol=1e-12)
        self.assertEqual(paused.next_progress, controller.progress)
        self.assertEqual(paused.decision.task_scale, 0.)
        controller.commit(paused)
        self.assertEqual(controller.progress, 0.)
        self.assertTrue(paused.record["task_paused"])

    def test_target_configuration_validation_is_atomic(self):
        controller, _, _ = scene(path="line")
        controller.progress = .4
        controller.previous_twist[:] = .001
        old_path = controller.path
        for changes in ({"gap": .01}, {"path": "bad"}, {"path_duration": 0},
                        {"path_delta": [1, 2]}, {"path_radius": np.nan}):
            request = {"type": "target", "gap": .3, **changes}
            with self.subTest(request=request), self.assertRaises(ValueError):
                controller.apply_target(request)
            self.assertIs(controller.path, old_path)
            self.assertEqual(controller.progress, .4)
            np.testing.assert_array_equal(controller.previous_twist, np.full(6, .001))
        controller.apply_target({"type": "target", "gap": .35, "path": "circle", "path_radius": .01, "path_duration": 15})
        self.assertEqual(controller.path.kind, "circle")
        self.assertEqual(controller.progress, 0.)
        self.assertAlmostEqual(controller.path.start[2, 3], .35)
        np.testing.assert_array_equal(controller.previous_twist, np.zeros(6))

    def test_required_leader_motion_exceeding_follower_joint_speed_is_not_clipped(self):
        controller, bases, states = scene(max_joint_speed=.05)
        accept(controller, bases)
        states["ur10e"].joint_velocity[0] = .08
        with self.assertRaises(MotionInfeasible) as caught:
            controller.prepare(10., states, .1, active=True)
        self.assertEqual(caught.exception.category, "compensation_infeasible")
        self.assertEqual(controller.progress, 0.)

    def test_joint_singularity_and_fk_mismatch_have_distinct_fault_categories(self):
        controller, bases, states = scene()
        accept(controller, bases)
        controller.models["ur5e"].jacobian[0, 0] = 0.
        with self.assertRaises(ControlFault) as caught:
            controller.prepare(10., states, .1, active=True)
        self.assertEqual(caught.exception.category, "singularity")
        controller.models["ur5e"].jacobian[0, 0] = 1.
        states["ur10e"].tcp[0, 3] += .04
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .1, active=True)
        self.assertEqual(caught.exception.category, "frame_mismatch")

    def test_mandatory_compensation_without_geometric_room_stops(self):
        controller, bases, states = scene()
        accept(controller, bases)
        states["ur10e"].joint_velocity[0] = .02

        def limited_scene(bases, joints):
            allowed = joints["ur5e"][0] <= .0011
            return allowed, "synthetic joint position boundary", 1.

        controller.guard.check = limited_scene
        with self.assertRaises(ControlFault) as caught:
            controller.prepare(10., states, .1, active=True)
        self.assertEqual(caught.exception.category, "compensation_infeasible")
        self.assertEqual(controller.progress, 0.)


if __name__ == "__main__":
    unittest.main()
