"""Recorded leader-start regression and delayed offline following plant.

No RTDE interfaces are created. The recorded fixture is an encoder/calibration
snapshot, and the independent Cartesian plant applies only 60% of a command
one sample later to expose missing catch-up feedback.
"""

from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from test_cooperative_controller import accept, scene
from test_approximate_tracking import OffsetCartesianModel, set_flange
from urcal.control.alignment import CapturedKinematics
from urcal.control.geometry import flange_from_tcp, inverse, pose_delta
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.state import RelativeSnapshot
from urcal.cooperative.task import MotionInfeasible


FIXTURE = Path(__file__).parent / "fixtures" / "follow_leader_start.json"


def recorded_scene(response="bounded"):
    data = json.loads(FIXTURE.read_text())
    args = SimpleNamespace(**data["settings"], leader_response=response)
    controller = CooperativeController(args)
    snapshot_data = data["snapshot"]
    snapshot = RelativeSnapshot(**{
        key: ({name: np.asarray(value) for name, value in values.items()}
              if key in ("base_poses", "base_twists") else
              np.asarray(values) if key in ("relative_pose", "relative_twist") else values)
        for key, values in snapshot_data.items()
    })
    controller.state = lambda now, active: snapshot
    controller.initial_tcp = np.array(data["initial_tcp"])
    for key, value in data["history"].items():
        setattr(controller, key, np.asarray(value) if isinstance(value, list) else value)
    states = {
        name: SimpleNamespace(joints=np.array(value["q"]), tcp=np.array(value["tcp"]),
                              joint_velocity=np.array(value["qd"]), speed=np.array(value["tcp_speed"]),
                              robot_timestamp=value["robot_timestamp"])
        for name, value in data["robots"].items()
    }
    return controller, states, data


def bounded_scene(**overrides):
    values = dict(leader_response="bounded", target_tolerance_m=.05, target_tolerance_deg=10.,
                  max_speed=.02, max_angular_speed=.1, max_accel=.1,
                  max_angular_accel=.5, max_joint_speed=.2, max_joint_accel=.5,
                  feedback_speed=.01, feedback_angular_speed=.05,
                  position_deadband=.001, rotation_deadband_deg=.3)
    values.update(overrides)
    return scene(**values)


def move_joint(controller, states, name, index, position):
    states[name].joints[index] = position
    states[name].tcp = controller.models[name].pose_and_jacobian(states[name].joints)[0]


class BoundedFollowingTests(unittest.TestCase):
    def assert_command_limits(self, controller, proposal, states, previous, dt):
        args = controller.args
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        self.assertLessEqual(np.linalg.norm(twist[:3]), args.max_speed + 1e-10)
        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed + 1e-10)
        self.assertLessEqual(np.max(np.abs(qdot)), args.max_joint_speed + 1e-10)
        for previous_qdot, previous_twist in (
                (previous[0], previous[1]),
                (states[controller.follower].joint_velocity, states[controller.follower].speed)):
            self.assertLessEqual(np.max(np.abs(qdot - previous_qdot)) / dt,
                                 args.max_joint_accel + 1e-8)
            self.assertLessEqual(np.linalg.norm(twist[:3] - previous_twist[:3]) / dt,
                                 args.max_accel + 1e-8)
            self.assertLessEqual(np.linalg.norm(twist[3:] - previous_twist[3:]) / dt,
                                 args.max_angular_accel + 1e-8)

    def test_recorded_slow_leader_start_is_bounded_and_future_pose_is_executable(self):
        controller, states, data = recorded_scene()
        before = (controller.previous_qdot.copy(), controller.previous_twist.copy())
        proposal = controller.prepare(data["control_at"], states, data["dt"], active=True)
        self.assert_command_limits(controller, proposal, states, before, data["dt"])
        self.assertGreater(np.linalg.norm(proposal.decision.qdot), .001)
        self.assertGreater(np.linalg.norm(np.array(data["expected_original_failure"]["demanded_qdot"])
                                         - proposal.decision.qdot), .001)
        model = controller.models[controller.follower]
        measured = states[controller.follower]
        nominal = model.pose_and_jacobian(measured.joints)[0]
        predicted = model.pose_and_jacobian(measured.joints + proposal.decision.qdot * data["dt"])[0]
        expected = predicted @ inverse(nominal) @ measured.tcp
        np.testing.assert_allclose(proposal.future_tcp_goals[controller.follower], expected, atol=1e-12)
        np.testing.assert_array_equal(proposal.record["desired_relative_pose"], controller.path.pose(0.))
        self.assertGreater(proposal.record["relative_error_m"], .02)

    def test_strict_mode_preserves_recorded_acceleration_failure(self):
        controller, states, data = recorded_scene("strict")
        with self.assertRaises(MotionInfeasible) as caught:
            controller.prepare(data["control_at"], states, data["dt"], active=True)
        self.assertEqual(caught.exception.category, "compensation_infeasible")
        self.assertIn("joint acceleration", str(caught.exception))

    def test_preview_and_discarded_active_proposal_do_not_capture_reference(self):
        controller, bases, states = bounded_scene()
        move_joint(controller, states, "ur5e", 0, .02)
        accept(controller, bases)
        path = controller.path.start.copy()
        controller.prepare(10., states, .04, active=False)
        controller.prepare(10., states, .04, active=True)  # Failed send: no commit.
        move_joint(controller, states, "ur5e", 0, .03)
        success = controller.prepare(10., states, .04, active=True)
        np.testing.assert_allclose(success.twists["ur5e"], 0., atol=1e-10)
        controller.commit(success)
        move_joint(controller, states, "ur5e", 0, .035)
        after = controller.prepare(10.04, states, .04, active=True)
        self.assertLess(after.twists["ur5e"][0], -.0001)
        np.testing.assert_array_equal(controller.path.start, path)
        self.assertAlmostEqual(after.record["relative_error_m"], .035)

    def test_failed_prepare_does_not_capture_reference(self):
        controller, bases, states = bounded_scene()
        move_joint(controller, states, "ur5e", 0, .02)
        accept(controller, bases)
        controller.guard.allowed = False
        with self.assertRaises(ResearchFault):
            controller.prepare(10., states, .04, active=True)
        controller.guard.allowed = True
        move_joint(controller, states, "ur5e", 0, .03)
        success = controller.prepare(10., states, .04, active=True)
        np.testing.assert_allclose(success.twists["ur5e"], 0., atol=1e-10)
        controller.commit(success)
        move_joint(controller, states, "ur5e", 0, .035)
        after = controller.prepare(10.04, states, .04, active=True)
        self.assertLess(after.twists["ur5e"][0], -.0001)

    def test_delayed_sixty_percent_plant_catches_up_after_start_stop_and_reversal(self):
        controller, bases, states = bounded_scene()
        move_joint(controller, states, "ur5e", 0, .02)
        initial_offset = .02
        initial_nominal = controller.path.start.copy()
        dt, tick = .04, 0
        queue = deque([np.zeros(6)])
        previous = (np.zeros(6), np.zeros(6))
        # A 15 mm translation is smaller than the 50 mm allowance. It must
        # still be followed and caught up after the leader stops.
        stages = ((0., .2), (.03, .5), (0., 8.), (-.03, 1.), (0., 8.))
        stopped_errors = []
        limited_samples = 0
        for speed, duration in stages:
            for _ in range(round(duration / dt)):
                now = 10. + tick * dt
                states["ur10e"].joint_velocity[:] = 0.
                states["ur10e"].joint_velocity[0] = speed
                states["ur10e"].speed = states["ur10e"].joint_velocity.copy()
                move_joint(controller, states, "ur10e", 0, states["ur10e"].joints[0] + speed * dt)
                applied = .6 * queue.popleft()
                states["ur5e"].joint_velocity = applied
                states["ur5e"].speed = applied.copy()
                states["ur5e"].joints += applied * dt
                states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
                accept(controller, bases, at=now, sequence=tick + 1)
                proposal = controller.prepare(now, states, dt, active=True)
                self.assert_command_limits(controller, proposal, states, previous, dt)
                if speed and abs(proposal.twists["ur5e"][0]) < abs(speed) - .001:
                    limited_samples += 1
                self.assertLess(proposal.record["relative_error_m"], controller.args.max_target_m)
                controller.commit(proposal)
                queue.append(proposal.decision.qdot.copy())
                previous = (proposal.decision.qdot.copy(), proposal.decision.twist.copy())
                tick += 1
            if speed == 0. and duration > 1.:
                error = states["ur5e"].tcp[0, 3] - states["ur10e"].tcp[0, 3] - initial_offset
                stopped_errors.append(abs(error))
        self.assertGreater(limited_samples, 5)
        self.assertEqual(len(stopped_errors), 2)
        for error in stopped_errors:
            self.assertLess(error, .002, "leader stopped but follower retained avoidable lag")
        np.testing.assert_array_equal(controller.path.start, initial_nominal)
        self.assertGreater(proposal.record["relative_error_m"], .015)

    def test_recorded_ur_geometry_follows_transient_and_catches_up_with_servo_delay(self):
        controller, states, data = recorded_scene()
        initial_snapshot = controller.state(data["control_at"], active=True)
        # Hold the measured base transforms fixed and remove their tiny noisy
        # velocity estimates. These local FK corrections reproduce the initial
        # recorded controller TCPs without claiming factory-calibrated dynamics.
        fixed_snapshot = RelativeSnapshot(**{
            **vars(initial_snapshot),
            "base_twists": {name: np.zeros(6) for name in states}})
        controller.state = lambda now, active: fixed_snapshot
        plants = {name: CapturedKinematics(name, state.joints, state.tcp, controller.offsets[name])
                  for name, state in states.items()}
        initial_jacobians = {name: plants[name].pose_and_jacobian(state.joints)[1]
                            for name, state in states.items()}
        leader_velocity = states["ur10e"].joint_velocity.copy() * 2.
        dt = data["dt"]
        first = controller.prepare(data["control_at"], states, dt, active=True)
        controller.commit(first)
        reference = first.record["tracking_relative_target"].copy()
        queue = deque([first.decision.qdot.copy()])
        previous = (first.decision.qdot.copy(), first.decision.twist.copy())
        stopped_errors = []
        peak_lag, peak_jacobian_change, peak_position_change = 0., 0., 0.
        start_follower = states["ur5e"].tcp[:3, 3].copy()
        tick = 0
        # The plant applies 60% of the previous qdot, rather than emulating a
        # manufacturer's speedL servo. Native transport ramps have separate
        # tests; this test exposes real UR Jacobian coupling and capsule checks.
        for multiplier, duration in ((1., 1.), (0., 6.), (-1., 1.), (0., 6.)):
            for _ in range(round(duration / dt)):
                tick += 1
                velocities = {"ur10e": multiplier * leader_velocity,
                              "ur5e": .6 * queue.popleft()}
                for name, state in states.items():
                    state.joint_velocity = velocities[name]
                    state.joints = state.joints + state.joint_velocity * dt
                    state.tcp, jacobian = plants[name].pose_and_jacobian(state.joints)
                    state.speed = jacobian @ state.joint_velocity
                    peak_jacobian_change = max(peak_jacobian_change,
                                               np.linalg.norm(jacobian - initial_jacobians[name]))
                now = data["control_at"] + tick * dt
                proposal = controller.prepare(now, states, dt, active=True)
                self.assert_command_limits(controller, proposal, states, previous, dt)
                relative = proposal.record["estimated_relative_pose"]
                dp, dr = pose_delta(relative, reference)
                peak_lag = max(peak_lag, np.linalg.norm(dp))
                peak_position_change = max(peak_position_change,
                                            np.linalg.norm(states["ur5e"].tcp[:3, 3] - start_follower))
                self.assertLess(proposal.record["relative_error_m"], controller.args.max_target_m)
                self.assertGreater(proposal.record["capsule_clearance_m"], 0.)
                controller.commit(proposal)
                queue.append(proposal.decision.qdot.copy())
                previous = (proposal.decision.qdot.copy(), proposal.decision.twist.copy())
            if multiplier == 0.:
                stopped_errors.append((np.linalg.norm(dp), np.linalg.norm(dr)))
        self.assertGreater(peak_position_change, .001)
        self.assertGreater(peak_jacobian_change, .005)
        self.assertLess(peak_lag, .01)
        self.assertEqual(len(stopped_errors), 2)
        for position_error, rotation_error in stopped_errors:
            self.assertLess(position_error, .002)
            self.assertLess(rotation_error, np.deg2rad(.5))
        np.testing.assert_array_equal(controller.tracking_offset, first.tracking_offset)

    def test_base_compensation_infeasibility_is_not_disguised_as_leader_lag(self):
        controller, bases, states = bounded_scene()
        accept(controller, bases)
        first = controller.prepare(10., states, .04, active=True)
        controller.commit(first)
        snapshot = controller.state(10., active=True)
        twists = {name: np.zeros(6) for name in states}
        twists["ur10e"][0] = .03  # Above follower's .02 m/s speed cap.
        forced = RelativeSnapshot(**{**vars(snapshot), "base_twists": twists})
        with patch.object(controller, "state", return_value=forced):
            with self.assertRaises(MotionInfeasible) as caught:
                controller.prepare(10., states, .04, active=True)
        self.assertEqual(caught.exception.category, "compensation_infeasible")

    def test_initial_reference_clamps_adjustment_to_declared_task_tolerances(self):
        controller, bases, states = bounded_scene()
        controller.models = {name: OffsetCartesianModel(controller.offsets[name]) for name in states}
        flange = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
        flange[0, 3] += .07
        flange[:3, :3] = Rotation.from_rotvec([0., 0., .25]).as_matrix() @ flange[:3, :3]
        set_flange(controller, states, "ur5e", flange)
        accept(controller, bases)
        proposal = controller.prepare(10., states, .04, active=True)
        self.assertIsNone(controller.tracking_offset)
        nominal = controller.path.pose(0.)
        dp, dr = pose_delta(nominal, proposal.record["tracking_relative_target"])
        self.assertAlmostEqual(np.linalg.norm(dp), .05)
        self.assertAlmostEqual(np.linalg.norm(dr), np.deg2rad(10.))
        self.assertAlmostEqual(proposal.record["relative_error_m"], .07)
        self.assertAlmostEqual(proposal.record["relative_error_rad"], .25)
        self.assertLess(proposal.decision.twist[0], 0.)
        controller.commit(proposal)
        np.testing.assert_allclose(nominal @ controller.tracking_offset,
                                   proposal.record["tracking_relative_target"], atol=1e-12)

    def test_rotating_line_and_circle_keep_fixed_offset_and_correct_tool_velocity(self):
        for path in ("line", "circle"):
            with self.subTest(path=path):
                controller, bases, states = bounded_scene(
                    path=path, path_delta=[.01, 0., .005, 0., .1, .2], path_radius=.01,
                    ur5e_tcp_offset=[.08, .02, .03, .05, -.02, .01],
                    ur10e_tcp_offset=[.03, -.01, .04, .01, .04, -.03])
                controller.models = {
                    name: OffsetCartesianModel(controller.offsets[name]) for name in states}
                flange = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
                flange[0, 3] += .02
                flange[:3, :3] = Rotation.from_rotvec([0., 0., .06]).as_matrix() @ flange[:3, :3]
                set_flange(controller, states, "ur5e", flange)
                accept(controller, bases)
                first = controller.prepare(10., states, .04, active=True)
                controller.commit(first)
                offset = controller.tracking_offset.copy()
                controller.progress = .4
                proposal = controller.prepare(10.04, states, .04, active=True)
                np.testing.assert_allclose(proposal.record["tracking_relative_target"],
                                           controller.path.pose(.4) @ offset, atol=1e-12)
                np.testing.assert_allclose(proposal.record["desired_relative_pose"],
                                           controller.path.pose(.4), atol=1e-12)
                leader_flange = flange_from_tcp(states["ur10e"].tcp, controller.offsets["ur10e"])
                outer = inverse(bases["ur5e"]) @ bases["ur10e"] @ leader_flange
                epsilon = 1e-6
                def desired_tcp(progress):
                    return outer @ controller.path.pose(progress) @ offset @ controller.offsets["ur5e"]
                dp, dr = pose_delta(desired_tcp(.4 - epsilon), desired_tcp(.4 + epsilon))
                finite_velocity = np.r_[dp, dr] / (2. * epsilon * controller.path.duration)
                np.testing.assert_allclose(proposal.record["task_twist_at_command"],
                                           finite_velocity, atol=1e-9)
                controller.commit(proposal)
                np.testing.assert_array_equal(controller.tracking_offset, offset)

    def test_zero_tolerance_retains_strict_response_and_target_change_clears_offset(self):
        exact, bases, states = bounded_scene(target_tolerance_m=0., target_tolerance_deg=0.)
        self.assertFalse(exact.bounded_leader)
        accept(exact, bases)
        states["ur10e"].joint_velocity[0] = .03
        with self.assertRaises(MotionInfeasible):
            exact.prepare(10., states, .04, active=True)
        controller, bases, states = bounded_scene()
        move_joint(controller, states, "ur5e", 0, .02)
        accept(controller, bases)
        controller.commit(controller.prepare(10., states, .04, active=True))
        self.assertIsNotNone(controller.tracking_offset)
        controller.apply_target({"type": "target", "gap": .31})
        self.assertIsNone(controller.tracking_offset)
        np.testing.assert_array_equal(controller.previous_base_compensation, np.zeros(6))

    def test_tracking_and_collision_guards_remain_hard_limits(self):
        for failure in ("tracking", "geometry"):
            with self.subTest(failure=failure):
                controller, bases, states = bounded_scene()
                accept(controller, bases)
                controller.commit(controller.prepare(10., states, .04, active=True))
                if failure == "tracking":
                    move_joint(controller, states, "ur10e", 0, .11)
                else:
                    controller.guard.check = lambda *_: (False, "synthetic collision", -.001)
                with self.assertRaises(ResearchFault) as caught:
                    controller.prepare(10.04, states, .04, active=True)
                self.assertEqual(caught.exception.category, "tracking_error" if failure == "tracking" else "clearance")


if __name__ == "__main__":
    unittest.main()
