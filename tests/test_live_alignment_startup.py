"""Initial alignment changes the start procedure without relaxing other bounds."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.geometry import pose_matrix, rotation_distance
from urcal.control.policy import ControlFault
from urcal.control.startup import check_alignment_request, check_start_snapshot
from tests.test_cooperative_controller import scene


def request(controller, bases, states, targets, origins=None):
    return check_alignment_request(controller.args, controller.offsets, bases, states, targets,
                                   controller.models, controller.guard, origins)


class LiveAlignmentStartupTests(unittest.TestCase):
    def test_large_orientation_can_enter_native_preflight_without_mutating_capture_or_target(self):
        controller, bases, states = scene()
        target = states["ur5e"].tcp.copy()
        target[:3, :3] = np.eye(3)
        targets = {"ur5e": target}
        self.assertGreater(rotation_distance(states["ur5e"].tcp, target),
                           np.deg2rad(controller.args.max_target_deg))
        before = deepcopy((bases, states, targets, controller.offsets))
        with patch.object(controller.models["ur5e"], "inverse", side_effect=AssertionError("native IK occurs after Arm")):
            request(controller, bases, states, targets)
        old_bases, old_states, old_targets, old_offsets = before
        for name in states:
            np.testing.assert_array_equal(bases[name], old_bases[name])
            np.testing.assert_array_equal(controller.offsets[name], old_offsets[name])
            for field in ("joints", "joint_velocity", "tcp", "speed"):
                np.testing.assert_array_equal(getattr(states[name], field), getattr(old_states[name], field))
        np.testing.assert_array_equal(targets["ur5e"], old_targets["ur5e"])
        self.assertEqual(controller.args.max_target_deg, 25.)

    def test_alignment_does_not_bypass_controller_fk_frame_mismatch(self):
        for mismatch in ("position", "rotation"):
            with self.subTest(mismatch=mismatch):
                controller, bases, states = scene()
                if mismatch == "position":
                    states["ur10e"].tcp[0, 3] += controller.args.frame_check_m + .001
                else:
                    states["ur10e"].tcp[:3, :3] = pose_matrix([0, 0, 0, .11, 0, 0])[:3, :3]
                with self.assertRaisesRegex(ControlFault, "nominal FK / controller TCP mismatch"):
                    request(controller, bases, states, {"ur5e": states["ur5e"].tcp.copy()})

    def test_original_travel_origin_bounds_both_current_and_target_positions(self):
        for outside in ("current", "target"):
            with self.subTest(outside=outside):
                controller, bases, states = scene()
                origins = {name: state.tcp.copy() for name, state in states.items()}
                target = origins["ur5e"].copy()
                if outside == "current":
                    states["ur5e"].joints[0] += controller.args.max_travel + .01
                    states["ur5e"].tcp = controller.models["ur5e"].pose_and_jacobian(states["ur5e"].joints)[0]
                else:
                    target[0, 3] += controller.args.max_travel + .01
                previous_origins = deepcopy(origins)
                with self.assertRaisesRegex(ControlFault, "start-position envelope") as caught:
                    request(controller, bases, states, {"ur5e": target}, origins)
                fault = caught.exception
                self.assertEqual(fault.details["checked_pose"], outside)
                self.assertEqual(fault.details["robot"], "ur5e")
                self.assertAlmostEqual(fault.details["distance_m"], controller.args.max_travel + .01)
                self.assertEqual(fault.details["limit_m"], controller.args.max_travel)
                self.assertIn("distance=310.0mm, limit=300.0mm", str(fault))
                for name in origins:
                    np.testing.assert_array_equal(origins[name], previous_origins[name])
                target[0, 3] += 1.
                origins["ur5e"][0, 3] += 1.
                self.assertNotEqual(fault.details["target_tcp"][0, 3], target[0, 3])
                np.testing.assert_array_equal(fault.details["origin_tcp"], previous_origins["ur5e"])

    def test_either_moving_arm_blocks_readiness_even_when_the_goal_is_already_reached(self):
        for name in ("ur5e", "ur10e"):
            for source, index, speed in (("joint_velocity", 2, .006), ("speed", 0, .003), ("speed", 4, .011)):
                with self.subTest(robot=name, source=source, component=index):
                    controller, bases, states = scene()
                    if name == controller.args.leader:
                        limit = (controller.args.align_leader_max_joint_speed if source == "joint_velocity"
                                 else controller.args.align_leader_max_speed if index < 3
                                 else controller.args.align_leader_max_angular_speed)
                        speed = limit + .001
                    getattr(states[name], source)[index] = speed
                    with self.assertRaisesRegex(ControlFault, "remain stationary"):
                        request(controller, bases, states, {"ur5e": states["ur5e"].tcp.copy()})

    def test_gap_workspace_and_scene_clearance_still_reject_alignment_readiness(self):
        controller, bases, states = scene()
        states["ur10e"].joints[2] = .6
        states["ur10e"].tcp = controller.models["ur10e"].pose_and_jacobian(states["ur10e"].joints)[0]
        with self.assertRaisesRegex(ControlFault, "flange separation below min_gap"):
            request(controller, bases, states, {"ur5e": states["ur5e"].tcp.copy()})
        controller, bases, states = scene()
        targets = {"ur5e": states["ur5e"].tcp.copy()}
        with patch.object(controller.guard, "pose_allowed", return_value=False):
            with self.assertRaisesRegex(ControlFault, "outside TCP workspace"):
                request(controller, bases, states, targets)
        with patch.object(controller.guard, "check", return_value=(False, "capsule clearance rejected", -.01)):
            with self.assertRaisesRegex(ControlFault, "capsule clearance rejected"):
                request(controller, bases, states, targets)

    def test_preflight_drift_is_compared_to_original_capture_not_last_good_sample(self):
        controller, original_bases, original_states = scene()
        for distance in (.004, .008):
            bases = deepcopy(original_bases)
            bases["ur10e"][0, 3] += distance
            check_start_snapshot(controller.args, original_bases, original_states, bases, deepcopy(original_states))
        bases = deepcopy(original_bases)
        bases["ur10e"][0, 3] += .012
        with self.assertRaisesRegex(ControlFault, "base moved"):
            check_start_snapshot(controller.args, original_bases, original_states, bases, deepcopy(original_states))
        rotated = deepcopy(original_bases)
        rotated["ur5e"][:3, :3] = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(1.2)])[:3, :3]
        with self.assertRaisesRegex(ControlFault, "base moved"):
            check_start_snapshot(controller.args, original_bases, original_states, rotated, deepcopy(original_states))
        # A stopped leader may already have changed position during native IK.
        # Fresh zero velocities alone must not authorize a recaptured start.
        for displacement in ("joints", "tcp"):
            states = deepcopy(original_states)
            if displacement == "joints":
                states["ur10e"].joints[5] += .011
            else:
                states["ur10e"].tcp[0, 3] += .004
            with self.subTest(displacement=displacement), self.assertRaisesRegex(ControlFault, "robot moved"):
                check_start_snapshot(controller.args, original_bases, original_states,
                                     deepcopy(original_bases), states)
        np.testing.assert_array_equal(original_bases["ur10e"][:3, 3], [.2, -.1, .5])
        self.assertEqual(original_states["ur10e"].joints[5], 0.)


if __name__ == "__main__":
    unittest.main()
