"""Tolerance-aware relative tracking against an offline Cartesian plant."""

import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from test_cooperative_controller import accept, scene, CartesianModel
from urcal.control.geometry import flange_from_tcp, inverse, matrix_pose, pose_delta
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.task import target_components


class OffsetCartesianModel(CartesianModel):
    """Finite-difference Jacobian includes the active tool's origin velocity."""

    def pose_and_jacobian(self, joints, flange=False):
        pose = super().pose_and_jacobian(joints, flange)[0]
        columns = []
        for index in range(6):
            trial = np.array(joints, dtype=float)
            trial[index] += 1e-6
            moved = super().pose_and_jacobian(trial, flange)[0]
            dp, dr = pose_delta(pose, moved)
            columns.append(np.r_[dp, dr] / 1e-6)
        return pose, np.stack(columns, axis=1)


def tolerant_scene(**overrides):
    values = dict(target_tolerance_m=.05, target_tolerance_deg=10.)
    values.update(overrides)
    return scene(**values)


def set_flange(controller, states, name, pose):
    states[name].joints = matrix_pose(pose)
    states[name].tcp = pose @ controller.offsets[name]


class ApproximateTrackingTests(unittest.TestCase):
    def test_inside_tube_does_not_require_exact_ik_or_erase_nominal_error(self):
        controller, bases, states = tolerant_scene(method="direct")
        initial = controller.path.start.copy()
        flange = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
        flange[0, 3] += .04
        flange[:3, :3] = Rotation.from_rotvec([0, 0, .12]).as_matrix() @ flange[:3, :3]
        set_flange(controller, states, "ur5e", flange)
        controller.models["ur5e"].inverse = Mock(side_effect=AssertionError("exact IK must not be queried"))
        accept(controller, bases)
        proposal = controller.prepare(10., states, .1)
        np.testing.assert_allclose(proposal.feedback, 0., atol=1e-12)
        np.testing.assert_allclose(proposal.twists["ur5e"], 0., atol=1e-12)
        np.testing.assert_allclose(proposal.tcp_goals["ur5e"], states["ur5e"].tcp)
        np.testing.assert_allclose(proposal.future_tcp_goals["ur5e"], states["ur5e"].tcp, atol=1e-12)
        self.assertAlmostEqual(proposal.record["relative_error_m"], .04)
        self.assertAlmostEqual(proposal.record["relative_error_rad"], .12)
        self.assertTrue(proposal.record["within_target_tolerance"])
        self.assertEqual(proposal.record["feedback_residual_m"], 0.)
        self.assertEqual(proposal.record["feedback_residual_rad"], 0.)
        np.testing.assert_array_equal(controller.path.start, initial)
        np.testing.assert_array_equal(proposal.record["desired_relative_pose"], initial)
        self.assertAlmostEqual(proposal.goals["ur5e"][0, 3], bases["ur5e"][0, 3])
        self.assertNotAlmostEqual(proposal.record["target_tcp"][0, 3], proposal.record["commanded_target_tcp"][0, 3])
        controller.models["ur5e"].inverse.assert_not_called()

    def test_feedback_is_continuous_at_flange_position_allowance(self):
        for displacement, residual in ((.04999, 0.), (.05, 0.), (.05001, .00001), (.06, .01)):
            with self.subTest(displacement=displacement):
                controller, bases, states = tolerant_scene(method="direct")
                states["ur5e"].joints[0] = displacement
                states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
                accept(controller, bases)
                proposal = controller.prepare(10., states, .1)
                self.assertAlmostEqual(proposal.feedback[0], -residual)
                np.testing.assert_allclose(proposal.feedback[1:], 0., atol=1e-12)
                self.assertAlmostEqual(proposal.record["feedback_residual_m"], residual)
                self.assertAlmostEqual(proposal.record["relative_error_m"], displacement)

    def test_nonzero_tool_rotation_correction_keeps_flange_origin_fixed(self):
        controller, bases, states = tolerant_scene(
            method="direct", ur5e_tcp_offset=[.15, .03, .08, .2, -.1, .05],
            feedback_speed=.5, feedback_angular_speed=.5)
        controller.models = {name: OffsetCartesianModel(controller.offsets[name]) for name in states}
        flange = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
        flange[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(15.)]).as_matrix() @ flange[:3, :3]
        set_flange(controller, states, "ur5e", flange)
        accept(controller, bases)
        proposal = controller.prepare(10., states, .1)
        expected_angular = np.array([0., 0., -np.deg2rad(5.)])
        lever = states["ur5e"].tcp[:3, 3] - flange[:3, 3]
        np.testing.assert_allclose(proposal.feedback[3:], expected_angular, atol=1e-10)
        np.testing.assert_allclose(proposal.feedback[:3], np.cross(expected_angular, lever), atol=1e-10)
        self.assertAlmostEqual(proposal.record["feedback_residual_m"], 0.)
        self.assertAlmostEqual(proposal.record["feedback_residual_rad"], np.deg2rad(5.))

    def test_tool_tcp_error_does_not_redefine_flange_task_allowance(self):
        controller, bases, states = tolerant_scene(
            method="direct", ur5e_tcp_offset=[.7, 0., 0., 0., 0., 0.])
        flange = flange_from_tcp(states["ur5e"].tcp, controller.offsets["ur5e"])
        flange[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(9.)]).as_matrix() @ flange[:3, :3]
        set_flange(controller, states, "ur5e", flange)
        accept(controller, bases)
        proposal = controller.prepare(10., states, .1)
        self.assertGreater(np.linalg.norm(proposal.record["target_tcp"][:3, 3] - states["ur5e"].tcp[:3, 3]), .1)
        self.assertTrue(proposal.record["within_target_tolerance"])
        np.testing.assert_allclose(proposal.feedback, 0., atol=1e-12)

    def test_near_term_tcp_uses_controller_anchor_not_uncalibrated_nominal_fk(self):
        controller, bases, states = tolerant_scene(method="direct")
        states["ur5e"].joints[0] = .06
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        states["ur5e"].tcp[1, 3] += .004
        accept(controller, bases)
        proposal = controller.prepare(10., states, .1)
        model = controller.models["ur5e"]
        nominal_current = model.pose_and_jacobian(states["ur5e"].joints)[0]
        next_q = states["ur5e"].joints + proposal.decision.qdot * .1
        expected = model.pose_and_jacobian(next_q)[0] @ inverse(nominal_current) @ states["ur5e"].tcp
        np.testing.assert_allclose(proposal.future_tcp_goals["ur5e"], expected, atol=1e-12)
        np.testing.assert_allclose(proposal.record["next_commanded_target_tcp"], expected, atol=1e-12)
        self.assertGreater(expected[1, 3], .0039)

    def test_leader_rotation_carries_accepted_offset_without_clipping_compensation(self):
        controller, bases, states = tolerant_scene()
        states["ur5e"].joints[0] = .04
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        states["ur10e"].joint_velocity[5] = .04
        accept(controller, bases)
        proposal = controller.prepare(10., states, .1)
        current_relative = proposal.record["estimated_relative_pose"]
        leader_flange = flange_from_tcp(states["ur10e"].tcp, controller.offsets["ur10e"])
        _, base, leader, _ = target_components(
            bases, {name: np.zeros(6) for name in states}, "ur10e", "ur5e", leader_flange,
            states["ur10e"].joint_velocity, current_relative, np.zeros(6), controller.offsets["ur5e"])
        np.testing.assert_allclose(proposal.compensation, base + leader, atol=1e-12)
        np.testing.assert_allclose(proposal.twists["ur5e"], proposal.compensation, atol=1e-12)
        self.assertAlmostEqual(proposal.compensation[1], .04 * .04)
        self.assertAlmostEqual(proposal.record["leader_compensation_nominal"][1], 0.)

    def test_actual_and_predicted_flange_gap_remain_hard_limits(self):
        controller, bases, states = tolerant_scene(method="direct", gap=.2)
        states["ur5e"].joints[2] = states["ur10e"].joints[2] + .151
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        states["ur10e"].joint_velocity[2] = .03
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .1)
        self.assertEqual(caught.exception.category, "clearance_infeasible")
        self.assertIn("flange gap", str(caught.exception))

    def test_tolerance_does_not_relax_tracking_envelope_or_geometry(self):
        controller, bases, states = tolerant_scene()
        states["ur5e"].joints[0] = .11
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .1)
        self.assertEqual(caught.exception.category, "tracking_error")
        self.assertEqual(caught.exception.details["error_origin"], "flange")
        controller, bases, states = tolerant_scene()
        controller.guard.allowed = False
        accept(controller, bases)
        with self.assertRaises(ResearchFault):
            controller.prepare(10., states, .1)

    def test_executable_future_tcp_still_must_fit_workspace(self):
        controller, bases, states = tolerant_scene()
        states["ur5e"].joints[0] = .04
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        states["ur10e"].joint_velocity[0] = .04
        controller.guard.pose_allowed = lambda name, pose: pose[0, 3] <= .041
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .1)
        self.assertEqual(caught.exception.category, "workspace_infeasible")
        self.assertIn("next command", str(caught.exception))

    def test_new_calibration_changes_nominal_goal_without_recapturing_path(self):
        controller, bases, states = tolerant_scene(method="direct", path="line")
        initial = controller.path.start.copy()
        controller.progress = .5
        accept(controller, bases)
        before = controller.prepare(10., states, .04)
        bases["ur5e"][0, 3] += .005
        accept(controller, bases, at=10.1, sequence=2)
        after = controller.prepare(10.1, states, .04)
        self.assertAlmostEqual(after.record["target_tcp"][0, 3], -.005)
        self.assertAlmostEqual(before.record["target_tcp"][0, 3], 0.)
        np.testing.assert_array_equal(controller.path.start, initial)
        self.assertEqual(controller.progress, .5)
        np.testing.assert_array_equal(after.record["desired_relative_pose"], controller.path.pose(.5))
        self.assertAlmostEqual(after.tcp_goals["ur5e"][0, 3], 0.)
        controller.commit(after)
        self.assertGreater(controller.progress, .5)

    def test_finish_dwell_uses_declared_task_allowance_and_keeps_measured_error(self):
        controller, bases, states = tolerant_scene(path="line", path_delta=[0.] * 6, finish_dwell=.2)
        controller.progress = 1.
        states["ur5e"].joints[0] = .04
        states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
        accept(controller, bases)
        for now in (10., 10.1):
            proposal = controller.prepare(now, states, .1)
            self.assertTrue(proposal.record["within_finish_tolerance"])
            self.assertFalse(proposal.record["within_nominal_finish_tolerance"])
            self.assertAlmostEqual(proposal.record["relative_error_m"], .04)
            controller.commit(proposal)
        self.assertTrue(controller.finished)
        self.assertEqual(proposal.record["finish_position_tolerance_m"], .05)

    def test_zero_allowance_retains_exact_target_ik_gate(self):
        controller, bases, states = tolerant_scene(target_tolerance_m=0., target_tolerance_deg=0.)
        controller.models["ur5e"].inverse = Mock(return_value=None)
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .1)
        self.assertEqual(caught.exception.category, "reachability_infeasible")
        controller.models["ur5e"].inverse.assert_called()


if __name__ == "__main__":
    unittest.main()
