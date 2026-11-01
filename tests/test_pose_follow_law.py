"""Coordinate-target following exercises the real controller without robot IO.

The independent Cartesian fixture makes transform errors visible and a delayed
plant checks the unchanged motion governor, rather than mirroring its formulas.
"""

from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from test_approximate_tracking import OffsetCartesianModel, set_flange
from test_bounded_follow import bounded_scene, move_joint
from test_cooperative_controller import accept, packet, settings
from urcal.control.geometry import flange_from_tcp, inverse, pose_matrix
from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.feedback import damped_feedback_velocity


CLI = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]


def pose_scene(commit=True, **overrides):
    options = dict(follow_law="pose", base_motion_mode="pose", base_pose_window=0,
                   kp_linear=3., kp_angular=3., feedback_speed=.1,
                   feedback_angular_speed=.15, follow_damping_time=.12,
                   follow_max_speed=1., max_joint_accel=.75)
    options.update(overrides)
    controller, bases, states = bounded_scene(**options)
    accept(controller, bases)
    if commit:
        controller.commit(controller.prepare(10., states, .04, active=True))
    return controller, bases, states


class PoseFollowLawTests(unittest.TestCase):
    def assert_motion_limits(self, engine, proposal, states, dt):
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        limits = engine.limits
        self.assertLessEqual(np.linalg.norm(twist[:3]), limits.linear_speed + 1e-9)
        self.assertLessEqual(np.linalg.norm(twist[3:]), limits.angular_speed + 1e-9)
        self.assertTrue(np.all(abs(qdot) <= limits.joint_speed + 1e-9))
        follower = states[engine.follower]
        for old_twist, old_qdot in ((engine.previous_twist, engine.previous_qdot),
                                   (follower.speed, follower.joint_velocity)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]),
                                 limits.linear_accel * dt + 1e-9)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]),
                                 limits.angular_accel * dt + 1e-9)
            self.assertTrue(np.all(abs(qdot - old_qdot) <= limits.joint_accel * dt + 1e-9))

    def assert_single_term(self, proposal):
        self.assertEqual(proposal.record["follow_law"], "pose")
        for key in ("base_compensation", "leader_compensation", "requested_compensation",
                    "applied_compensation", "task_twist_at_command"):
            np.testing.assert_array_equal(proposal.record[key], np.zeros(6), err_msg=key)
        np.testing.assert_allclose(proposal.record["tracking_requested"], proposal.feedback, atol=1e-12)

    def test_parser_default_explicit_modes_and_incompatible_combinations(self):
        self.assertEqual(parse_args(CLI).follow_law, "feedforward")
        pose = parse_args([*CLI, "--follow_law", "pose"])
        self.assertEqual((pose.follow_law, pose.method, pose.base_motion_mode, pose.leader_response),
                         ("pose", "separated", "pose", "bounded"))
        for arguments in (("--method", "direct"), ("--base_motion_mode", "velocity"),
                          ("--leader_response", "strict"), ("--follow_law", "unknown")):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([*CLI, "--follow_law", "pose", *arguments])
        # The additional mode must not silently retune any physical limits.
        legacy = parse_args(CLI)
        for key in ("max_speed", "follow_max_speed", "max_angular_speed", "max_accel",
                    "max_angular_accel", "max_joint_speed", "max_joint_accel", "max_travel",
                    "feedback_speed", "feedback_angular_speed", "min_gap", "min_singular_value"):
            self.assertEqual(getattr(pose, key), getattr(legacy, key), key)

    def test_engine_rejects_invalid_saved_configurations_too(self):
        for invalid in ({"method": "direct"}, {"base_motion_mode": "velocity"},
                        {"leader_response": "strict"}, {"follow_law": "unknown"}):
            options = dict(follow_law="pose", base_motion_mode="pose", leader_response="bounded")
            options.update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CooperativeController(settings(**options))

    def test_missing_law_keeps_legacy_feedforward_behavior(self):
        engine, bases, states = pose_scene(commit=False, follow_law="feedforward")
        del engine.args.follow_law
        legacy = CooperativeController(engine.args, models=engine.models, guard=engine.guard)
        accept(legacy, bases)
        states["ur10e"].joint_velocity[0] = .02
        states["ur10e"].speed[0] = .02
        default = legacy.prepare(10., states, .04, active=True)
        explicit = engine.prepare(10., states, .04, active=True)
        self.assertEqual(default.record["follow_law"], "feedforward")
        self.assertGreater(default.record["leader_compensation"][0], .01)
        np.testing.assert_allclose(default.decision.twist, explicit.decision.twist, atol=1e-12)

    def test_exact_pose_has_no_leader_velocity_feedforward(self):
        engine, _, states = pose_scene()
        states["ur10e"].joint_velocity[0] = .03
        states["ur10e"].speed[0] = .03
        proposal = engine.prepare(10., states, .04, active=True)
        self.assert_single_term(proposal)
        np.testing.assert_allclose(proposal.record["tracking_requested"], 0., atol=1e-12)
        np.testing.assert_allclose(proposal.decision.twist, 0., atol=1e-12)

    def test_leader_qdot_does_not_change_pose_demand_at_the_same_poses(self):
        engine, _, states = pose_scene()
        move_joint(engine, states, "ur10e", 0, .012)
        stationary = engine.prepare(10., states, .04, active=True)
        self.assertGreater(stationary.feedback[0], .005)
        states["ur10e"].joint_velocity[:] = [.03, -.01, 0., 0., 0., .01]
        states["ur10e"].speed[:] = states["ur10e"].joint_velocity
        moving = engine.prepare(10., states, .04, active=True)
        # Leader velocity is still legitimate future-geometry input, so compare
        # the requested law, not a potentially governor-limited final command.
        np.testing.assert_allclose(moving.record["tracking_requested"],
                                   stationary.record["tracking_requested"], atol=1e-12)
        np.testing.assert_allclose(moving.record["pose_command_target_tcp"],
                                   stationary.record["pose_command_target_tcp"], atol=1e-12)
        self.assert_single_term(moving)

    def test_each_base_change_and_leader_pose_change_recompute_the_target(self):
        for changed in ("ur5e", "ur10e", "leader_joint"):
            engine, bases, states = pose_scene()
            if changed == "leader_joint":
                move_joint(engine, states, "ur10e", 0, .008)
            else:
                bases[changed][0, 3] += .008
                accept(engine, bases, at=10.04, sequence=2)
            proposal = engine.prepare(10.04, states, .04, active=True)
            expected_x = -.008 if changed == "ur5e" else .008
            with self.subTest(changed=changed):
                self.assertAlmostEqual(proposal.record["pose_command_target_tcp"][0, 3], expected_x)
                self.assertAlmostEqual(proposal.record["pose_command_error_m"], .008)
                self.assertGreater(proposal.feedback[0] * expected_x, 0.)
                self.assert_single_term(proposal)

    def test_rotated_frames_and_nonzero_tcp_offsets_define_one_absolute_tcp_goal(self):
        engine, bases, states = pose_scene(
            commit=False, ur5e_tcp_offset=[.12, -.02, .08, .2, -.1, .05],
            ur10e_tcp_offset=[-.03, .02, .04, -.1, .05, .2])
        engine.models = {name: OffsetCartesianModel(engine.offsets[name]) for name in states}
        bases["ur10e"] = pose_matrix([.18, -.12, .55, .1, -.1, .25])
        bases["ur5e"] = pose_matrix([.21, -.08, .52, -.1, .12, -.2])
        # This is a separate initial scene, not a calibration-jump test.
        engine = CooperativeController(engine.args, models=engine.models, guard=engine.guard)
        leader_flange = flange_from_tcp(states["ur10e"].tcp, engine.offsets["ur10e"])
        relative = engine.path.pose(0.)
        aligned = inverse(bases["ur5e"]) @ bases["ur10e"] @ leader_flange @ relative
        set_flange(engine, states, "ur5e", aligned)
        accept(engine, bases)
        engine.commit(engine.prepare(10., states, .04, active=True))
        leader_flange[:3, 3] += [.006, -.003, .002]
        leader_flange[:3, :3] = Rotation.from_rotvec([.01, -.015, .02]).as_matrix() @ leader_flange[:3, :3]
        set_flange(engine, states, "ur10e", leader_flange)
        proposal = engine.prepare(10., states, .04, active=True)
        expected = (inverse(bases["ur5e"]) @ bases["ur10e"] @ leader_flange
                    @ relative @ engine.tracking_offset @ engine.offsets["ur5e"])
        actual = states["ur5e"].tcp
        position_error = expected[:3, 3] - actual[:3, 3]
        rotation_error = Rotation.from_matrix(expected[:3, :3] @ actual[:3, :3].T).as_rotvec()
        np.testing.assert_allclose(proposal.record["pose_command_target_tcp"], expected, atol=1e-12)
        self.assertAlmostEqual(proposal.record["pose_command_error_m"], np.linalg.norm(position_error))
        self.assertAlmostEqual(proposal.record["pose_command_error_rad"], np.linalg.norm(rotation_error))
        self.assertGreater(proposal.feedback[:3] @ position_error, 0.)
        self.assertGreater(proposal.feedback[3:] @ rotation_error, 0.)
        self.assert_single_term(proposal)
        self.assert_motion_limits(engine, proposal, states, .04)

    def test_damping_uses_actual_tcp_velocity_without_subtracting_leader_or_tool_motion(self):
        engine, _, states = pose_scene(ur5e_tcp_offset=[.4, .07, .1, .1, -.2, .05])
        engine.models = {name: OffsetCartesianModel(engine.offsets[name]) for name in states}
        move_joint(engine, states, "ur5e", 0, -.01)
        follower = states["ur5e"]
        follower.joint_velocity[5] = .02
        _, jacobian = engine.models["ur5e"].pose_and_jacobian(follower.joints)
        follower.speed = jacobian @ follower.joint_velocity
        states["ur10e"].joint_velocity[0] = .005
        states["ur10e"].speed[0] = .005
        engine.previous_twist = follower.speed.copy()
        engine.previous_qdot = follower.joint_velocity.copy()
        self.assertGreater(np.linalg.norm(follower.speed[:3]), .004)
        with patch("urcal.cooperative.controller.damped_feedback_velocity",
                   wraps=damped_feedback_velocity) as damp:
            proposal = engine.prepare(10., states, .04, active=True)
        self.assertEqual(damp.call_count, 2)
        np.testing.assert_allclose(damp.call_args_list[0].args[1], follower.speed[:3], atol=1e-12)
        np.testing.assert_allclose(damp.call_args_list[1].args[1], follower.speed[3:], atol=1e-12)
        np.testing.assert_allclose(proposal.record["feedback_relative_tcp_velocity"], follower.speed, atol=1e-12)
        self.assert_motion_limits(engine, proposal, states, .04)

    def test_discarded_proposal_and_later_lag_never_recapture_committed_offset(self):
        engine, _, states = pose_scene(commit=False)
        move_joint(engine, states, "ur5e", 0, .02)
        discarded = engine.prepare(10., states, .04, active=True)
        self.assertIsNone(engine.tracking_offset)
        move_joint(engine, states, "ur5e", 0, .03)
        first = engine.prepare(10., states, .04, active=True)
        engine.commit(first)
        offset = engine.tracking_offset.copy()
        self.assertFalse(np.allclose(offset, discarded.tracking_offset))
        move_joint(engine, states, "ur5e", 0, .035)
        after = engine.prepare(10., states, .04, active=True)
        self.assertAlmostEqual(after.record["pose_command_target_tcp"][0, 3], .03)
        self.assertLess(after.feedback[0], 0.)
        engine.commit(after)
        np.testing.assert_array_equal(engine.tracking_offset, offset)

    def test_delayed_plant_obeys_speed_and_both_acceleration_histories(self):
        engine, bases, states = pose_scene(follow_max_speed=.04, max_target_m=.2)
        move_joint(engine, states, "ur10e", 0, .09)
        last_sent = np.zeros(6)
        commands, errors = [], []
        for tick in range(1, 161):
            follower = states["ur5e"]
            follower.joint_velocity = .9 * last_sent
            follower.joints += follower.joint_velocity * .04
            follower.tcp, jacobian = engine.models["ur5e"].pose_and_jacobian(follower.joints)
            follower.speed = jacobian @ follower.joint_velocity
            now = 10. + tick * .04
            accept(engine, bases, at=now, sequence=tick + 1)
            proposal = engine.prepare(now, states, .04, active=True)
            self.assert_motion_limits(engine, proposal, states, .04)
            commands.append(np.linalg.norm(proposal.decision.twist[:3]))
            errors.append(proposal.record["pose_command_error_m"])
            engine.commit(proposal)
            last_sent = proposal.decision.qdot.copy()
        self.assertGreater(max(commands), .035)
        self.assertLess(errors[-1], .002)
        self.assertGreaterEqual(min(.09 - states["ur5e"].tcp[0, 3], errors[-1]), -.001)

    def test_stale_measurements_actual_geometry_and_speed_faults_are_retained(self):
        for issue in ("stale", "clearance", "joint_speed", "tcp_speed"):
            engine, _, states = pose_scene()
            now = 10.
            if issue == "stale":
                now = 10.6
            elif issue == "clearance":
                engine.guard.check = lambda bases, joints: (False, "synthetic collision", -.01)
            elif issue == "joint_speed":
                states["ur5e"].joint_velocity[0] = engine.args.max_joint_speed * 1.3
            else:
                states["ur5e"].speed[0] = engine.limits.linear_speed * 1.2
            with self.subTest(issue=issue), self.assertRaises(ResearchFault) as caught:
                engine.prepare(now, states, .04, active=True)
            self.assertEqual(caught.exception.category,
                             "measurement_expired" if issue == "stale" else issue)

    def test_approach_changes_pose_target_with_no_path_velocity_term(self):
        engine, bases, states = pose_scene(approach_speed=.01)
        engine.handle_gap_approach("start", enabled=True)
        first = engine.prepare(10., states, .04, active=True)
        self.assertGreater(first.next_progress, 0.)
        self.assertEqual(engine.progress, 0.)
        self.assertEqual(engine.gap_approach.info()["progress"], 0.)
        engine.commit(first)
        last_sent = first.decision.qdot.copy()
        initial_target = first.record["pose_command_target_tcp"].copy()
        for tick in range(1, 251):
            follower = states["ur5e"]
            follower.joint_velocity = .9 * last_sent
            follower.joints += follower.joint_velocity * .04
            follower.tcp, jacobian = engine.models["ur5e"].pose_and_jacobian(follower.joints)
            follower.speed = jacobian @ follower.joint_velocity
            now = 10. + tick * .04
            accept(engine, bases, at=now, sequence=tick + 1)
            proposal = engine.prepare(now, states, .04, active=True)
            self.assert_single_term(proposal)
            self.assert_motion_limits(engine, proposal, states, .04)
            engine.commit(proposal)
            last_sent = proposal.decision.qdot.copy()
        self.assertGreater(engine.progress, .2)
        self.assertGreater(np.linalg.norm(engine.path.derivative(engine.progress)), .001)
        self.assertLess(proposal.record["pose_command_target_tcp"][2, 3], initial_target[2, 3] - .01)
        self.assertLess(states["ur5e"].tcp[2, 3], initial_target[2, 3] - .005)
        progress, path = engine.progress, engine.path
        engine.handle_gap_approach("pause", enabled=False)
        paused = engine.prepare(now, states, .04, active=True)
        self.assertEqual(paused.next_progress, progress)
        engine.commit(paused)
        engine.handle_gap_approach("resume", enabled=True)
        resumed = engine.prepare(now, states, .04, active=True)
        self.assertGreater(resumed.next_progress, progress)
        self.assertIs(engine.path, path)

    def test_short_invalid_frame_freezes_approach_then_recovers_without_new_arm(self):
        engine, bases, states = pose_scene(live_calibration_recovery=True)
        engine.handle_gap_approach("start", enabled=True)
        engine.commit(engine.prepare_live(10., states, .04))
        progress = engine.progress
        engine.ingest(packet(engine, bases, 10.04, 2, reason="occlusion"), 10.04, active_recovery=True)
        held = engine.prepare_live(10.04, states, .04)
        self.assertEqual(held.next_progress, progress)
        self.assertEqual(engine.gap_approach.state, "running")
        self.assertTrue(engine.gap_approach_status(enabled=True)["temporary_hold"])
        engine.commit(held)
        self.assertTrue(engine.ingest(packet(engine, bases, 10.08, 3), 10.08, active_recovery=True))
        resumed = engine.prepare_live(10.08, states, .04)
        self.assertGreater(resumed.next_progress, progress)
        self.assertFalse(engine.gap_approach_status(enabled=True)["temporary_hold"])
        self.assert_single_term(resumed)

    def test_limited_pose_catch_up_freezes_approach_even_without_task_velocity(self):
        engine, _, states = pose_scene(approach_speed=.01)
        engine.handle_gap_approach("start", enabled=True)
        first = engine.prepare(10., states, .04, active=True)
        self.assertGreater(first.next_progress, 0.)
        engine.commit(first)
        progress = engine.progress
        # Reaching this changed leader pose in one tick violates the existing
        # acceleration limit. Catch up first instead of advancing the approach
        # simply because its additive path-velocity channel is zero.
        move_joint(engine, states, "ur10e", 0, .04)
        limited = engine.prepare(10., states, .04, active=True)
        self.assertGreater(np.linalg.norm(limited.record["tracking_requested"]), .01)
        self.assertLess(np.linalg.norm(limited.decision.twist),
                        np.linalg.norm(limited.record["tracking_requested"]))
        self.assertEqual(limited.decision.task_scale, 0.)
        self.assertEqual(limited.next_progress, progress)
        self.assert_single_term(limited)
        self.assert_motion_limits(engine, limited, states, .04)


if __name__ == "__main__":
    unittest.main()
