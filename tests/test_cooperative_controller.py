"""Production cooperative engine with an ideal Cartesian plant, no hardware."""

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
        leader="ur10e", mode="follow", method="separated", path="hold", gap=.3,
        path_delta=[0., 0., -.05, 0., 0., 0.], path_radius=.02, path_duration=10.,
        ur5e_tcp_offset=[0.] * 6, ur10e_tcp_offset=[0.] * 6,
        workspace_config=None, stable_frames=1, filter_alpha=1., velocity_alpha=1.,
        prediction_horizon=.3, max_calibration_age=.5, max_invalid_duration=.1,
        max_base_speed=.15, max_base_angular_speed=.5,
        max_speed=.2, max_angular_speed=.5, max_accel=2., max_angular_accel=4.,
        max_joint_speed=1., max_joint_accel=10., min_singular_value=.01,
        max_travel=.3, max_target_m=.1, max_target_deg=25.,
        frame_check_m=.03, frame_check_deg=5.,
        feedback_speed=.05, feedback_angular_speed=.2,
        position_deadband=.0001, rotation_deadband_deg=.01,
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
