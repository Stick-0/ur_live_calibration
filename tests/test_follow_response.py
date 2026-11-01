"""Offline catch-up response with unchanged motion limits and a delayed plant.

The plant applies 90% of the previous joint command. It deliberately lags the
controller without pretending to reproduce the manufacturer's speed servo.
No camera, socket, or RTDE connection is opened.
"""

from collections import deque
import unittest

import numpy as np

from test_bounded_follow import bounded_scene, move_joint, recorded_scene
from test_cooperative_controller import accept
from urcal.control.alignment import CapturedKinematics
from urcal.control.geometry import pose_delta
from urcal.control.runtime import parse_args
from urcal.cooperative.state import RelativeSnapshot


CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]
LEGACY_RESPONSE = dict(feedback_speed=.01, feedback_angular_speed=.05,
                       kp_linear=1., kp_angular=1.)
UNCHANGED_CAPS = dict(max_speed=.02, follow_max_speed=1., max_angular_speed=.1,
                      max_joint_speed=.2, max_accel=.1, max_angular_accel=.5,
                      max_joint_accel=.5)


def response_settings(faster):
    current = parse_args(CLI)
    return ({name: getattr(current, name) for name in LEGACY_RESPONSE}
            if faster else LEGACY_RESPONSE.copy())


class FollowResponseTests(unittest.TestCase):
    def assert_bounded(self, controller, proposal, states, dt):
        args = controller.args
        twist, qdot = proposal.decision.twist, proposal.decision.qdot
        self.assertLessEqual(np.linalg.norm(twist[:3]), controller.limits.linear_speed + 1e-9)
        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed + 1e-9)
        self.assertLessEqual(np.max(np.abs(qdot)), args.max_joint_speed + 1e-9)
        follower = states[controller.follower]
        for old_twist, old_qdot in ((controller.previous_twist, controller.previous_qdot),
                                   (follower.speed, follower.joint_velocity)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]),
                                 args.max_accel * dt + 1e-9)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]),
                                 args.max_angular_accel * dt + 1e-9)
            self.assertLessEqual(np.max(np.abs(qdot - old_qdot)),
                                 args.max_joint_accel * dt + 1e-9)

    def step_response(self, faster, rotational=False):
        controller, bases, states = bounded_scene(
            **UNCHANGED_CAPS, **response_settings(faster), base_motion_mode="pose")
        dt = .04
        accept(controller, bases)
        first = controller.prepare(10., states, dt, active=True)
        controller.commit(first)
        reference = first.record["tracking_relative_target"].copy()
        # Add a position/orientation tracking error after capturing the same
        # aligned reference. This isolates catching up after the leader stops.
        axis, step = (3, .12) if rotational else (0, .05)
        move_joint(controller, states, "ur5e", axis, states["ur5e"].joints[axis] + step)
        queue = deque([first.decision.qdot.copy()])
        errors, speeds = [], []
        settled_at, consecutive = None, 0
        threshold = np.deg2rad(.5) if rotational else .002
        part = slice(3, 6) if rotational else slice(0, 3)
        for tick in range(1, 351):
            state = states["ur5e"]
            state.joint_velocity = .9 * queue.popleft()
            state.joints += state.joint_velocity * dt
            state.tcp, jacobian = controller.models["ur5e"].pose_and_jacobian(state.joints)
            state.speed = jacobian @ state.joint_velocity
            now = 10. + tick * dt
            accept(controller, bases, at=now, sequence=tick + 1)
            proposal = controller.prepare(now, states, dt, active=True)
            self.assert_bounded(controller, proposal, states, dt)
            dp, dr = pose_delta(proposal.record["estimated_relative_pose"], reference)
            error = np.linalg.norm(dr if rotational else dp)
            errors.append(error)
            speeds.append(np.linalg.norm(proposal.decision.twist[part]))
            consecutive = consecutive + 1 if error < threshold else 0
            if consecutive >= 5 and settled_at is None:
                settled_at = (tick - 4) * dt
            controller.commit(proposal)
            queue.append(proposal.decision.qdot.copy())
        self.assertIsNotNone(settled_at, "follower never settled after the target stopped")
        self.assertLess(max(errors[-25:]), threshold, "stationary target caused continuing oscillation")
        return dict(settled_at=settled_at, peak_speed=max(speeds),
                    integrated_error=sum(errors) * dt, final_error=errors[-1])

    def test_position_catch_up_is_materially_faster_with_same_physical_caps(self):
        old, new = self.step_response(False), self.step_response(True)
        self.assertLess(new["settled_at"], old["settled_at"] * .65)
        self.assertLess(new["integrated_error"], old["integrated_error"] * .65)
        self.assertGreater(new["peak_speed"], old["peak_speed"] * 2.)

    def test_orientation_catch_up_is_faster_without_relaxing_angular_limits(self):
        old, new = self.step_response(False, True), self.step_response(True, True)
        self.assertLess(new["settled_at"], old["settled_at"] * .75)
        self.assertLess(new["integrated_error"], old["integrated_error"] * .75)
        self.assertGreater(new["peak_speed"], old["peak_speed"] * 1.5)
        self.assertLessEqual(new["peak_speed"], UNCHANGED_CAPS["max_angular_speed"] + 1e-9)

    def recorded_response(self, faster):
        controller, states, data = recorded_scene()
        # Keep the recorded geometry and all physical limits; update only the
        # response gains/caps. This fixture predates the separate follow cap,
        # so its stricter 20 mm/s cap remains identical in both comparisons.
        for name, value in response_settings(faster).items():
            setattr(controller.args, name, value)
        controller.base_motion_mode = "pose"
        snapshot = controller.state(data["control_at"], active=True)
        fixed = RelativeSnapshot(**{**vars(snapshot),
                                   "base_twists": {name: np.zeros(6) for name in states}})
        controller.state = lambda now, active: fixed
        plants = {name: CapturedKinematics(name, state.joints, state.tcp, controller.offsets[name])
                  for name, state in states.items()}
        velocity = states["ur10e"].joint_velocity.copy() * 4.
        dt = data["dt"]
        first = controller.prepare(data["control_at"], states, dt, active=True)
        controller.commit(first)
        reference = first.record["tracking_relative_target"].copy()
        queue = deque([first.decision.qdot.copy()])
        errors, stopped_errors = [], []
        tick = 0
        for direction, duration in ((1., 1.), (0., 4.), (-1., 1.), (0., 4.)):
            for _ in range(round(duration / dt)):
                tick += 1
                velocities = {"ur10e": direction * velocity, "ur5e": .9 * queue.popleft()}
                for name, state in states.items():
                    state.joint_velocity = velocities[name]
                    state.joints += state.joint_velocity * dt
                    state.tcp, jacobian = plants[name].pose_and_jacobian(state.joints)
                    state.speed = jacobian @ state.joint_velocity
                proposal = controller.prepare(data["control_at"] + tick * dt, states, dt, active=True)
                self.assert_bounded(controller, proposal, states, dt)
                self.assertGreater(proposal.record["capsule_clearance_m"], 0.)
                dp, dr = pose_delta(proposal.record["estimated_relative_pose"], reference)
                errors.append(np.linalg.norm(dp))
                controller.commit(proposal)
                queue.append(proposal.decision.qdot.copy())
            if direction == 0.:
                stopped_errors.append((np.linalg.norm(dp), np.linalg.norm(dr)))
        for position, rotation in stopped_errors:
            self.assertLess(position, .002)
            self.assertLess(rotation, np.deg2rad(.5))
        return dict(integrated_error=sum(errors) * dt, peak_error=max(errors))

    def test_recorded_ur_geometry_tracks_start_stop_and_reversal_without_new_faults(self):
        old, new = self.recorded_response(False), self.recorded_response(True)
        self.assertLess(new["integrated_error"], old["integrated_error"] * .9)
        self.assertLessEqual(new["peak_error"], old["peak_error"] * 1.1)


if __name__ == "__main__":
    unittest.main()
