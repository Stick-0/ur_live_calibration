"""speedL's future joint demand must use the future inverse Jacobian."""

import unittest

import numpy as np

from test_bounded_follow import bounded_scene, recorded_scene
from test_cooperative_controller import CartesianModel, accept
from urcal.control.geometry import inverse


class ShrinkingTranslationModel(CartesianModel):
    """Smooth scalar kinematics whose translational gain shrinks with motion."""

    def pose_and_jacobian(self, joints, flange=False):
        pose, jacobian = super().pose_and_jacobian(joints, flange)
        q = float(joints[0])
        pose[0, 3] = q - 2000. * q * q
        jacobian[0, 0] = 1. - 4000. * q
        return pose, jacobian


class FollowingFutureJacobianTests(unittest.TestCase):
    def test_shrinking_gain_reduces_command_before_future_joint_limit(self):
        for limit in ("speed", "acceleration"):
            with self.subTest(limit=limit):
                controller, bases, states = bounded_scene(
                    max_joint_speed=.003 if limit == "speed" else .2,
                    max_joint_accel=.1 if limit == "acceleration" else 1.)
                model = ShrinkingTranslationModel(controller.offsets["ur5e"])
                controller.models["ur5e"] = model
                states["ur10e"].joint_velocity[0] = .003
                accept(controller, bases)
                proposal = controller.prepare(10., states, .04, active=True)
                self.assertGreater(proposal.decision.twist[0], 0.)
                self.assertLess(proposal.decision.twist[0], .003 - 1e-9)
                for fraction in (.5, 1.):
                    trial_q = states["ur5e"].joints + proposal.decision.qdot * .04 * fraction
                    future_jacobian = model.pose_and_jacobian(trial_q)[1]
                    actual_future_qdot = np.linalg.solve(future_jacobian, proposal.decision.twist)
                    self.assertLessEqual(np.max(np.abs(actual_future_qdot)), controller.args.max_joint_speed + 1e-9)
                    self.assertLessEqual(np.max(np.abs(actual_future_qdot)) / .04, controller.args.max_joint_accel + 1e-8)
                # The old constant-qdot projection is smaller than its current
                # Cartesian speed and would miss the actual speedL joint demand.
                raw_qdot = np.array([.003, 0., 0., 0., 0., 0.])
                raw_jacobian = model.pose_and_jacobian(states["ur5e"].joints + raw_qdot * .04)[1]
                self.assertLess(np.linalg.norm((raw_jacobian @ raw_qdot)[:3]), .003)
                raw_future = np.linalg.solve(raw_jacobian, raw_qdot)
                bound = controller.args.max_joint_speed if limit == "speed" else controller.args.max_joint_accel * .04
                self.assertGreater(raw_future[0], bound)

    def test_recorded_leader_start_remains_feasible_at_future_inverse_jacobians(self):
        controller, states, data = recorded_scene()
        follower = controller.follower
        state = states[follower]
        model = controller.models[follower]
        proposal = controller.prepare(data["control_at"], states, data["dt"], active=True)
        self.assertGreater(np.linalg.norm(proposal.decision.qdot), .001)
        nominal = model.pose_and_jacobian(state.joints)[0]
        correction = inverse(nominal) @ state.tcp
        for fraction in (.5, 1.):
            trial_q = state.joints + proposal.decision.qdot * data["dt"] * fraction
            predicted, future_jacobian = model.pose_and_jacobian(trial_q)
            anchored_tcp = predicted @ correction
            future_jacobian[:3] += np.cross(future_jacobian[3:].T, anchored_tcp[:3, 3] - predicted[:3, 3]).T
            future_qdot = np.linalg.solve(future_jacobian, proposal.decision.twist)
            self.assertLessEqual(np.max(np.abs(future_qdot)), controller.args.max_joint_speed + 1e-9)
            for previous in (controller.previous_qdot, state.joint_velocity):
                self.assertLessEqual(np.max(np.abs(future_qdot - previous)),
                                     controller.args.max_joint_accel * data["dt"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
