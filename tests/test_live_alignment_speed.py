"""Alignment speed changes exercised through measured virtual robot feedback."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest

import numpy as np

from test_live_alignment import fixture, make_plan
from urcal.control.alignment import AlignmentFault
from urcal.control.geometry import rotation_distance
from urcal.control.runtime import parse_args


CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2"]
MOTION_LIMITS = ("max_speed", "max_angular_speed", "max_joint_speed",
                 "max_accel", "max_angular_accel", "max_joint_accel")


class AlignmentSpeedTests(unittest.TestCase):
    def test_default_scale_and_overrides_leave_all_global_limits_unchanged(self):
        default = parse_args(CLI)
        self.assertEqual(default.align_speed_scale, 2.)
        for scale in (.25, 1., 2.):
            args = parse_args(CLI + ["--align_speed_scale", str(scale)])
            self.assertEqual(args.align_speed_scale, scale)
            self.assertEqual({key: getattr(args, key) for key in MOTION_LIMITS},
                             {key: getattr(default, key) for key in MOTION_LIMITS})

    def test_cli_rejects_nonfinite_nonpositive_and_above_full_limit_scales(self):
        for invalid in ("nan", "inf", "-inf", "0", "-1", "2.000001"):
            with self.subTest(scale=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(CLI + ["--align_speed_scale=" + invalid])

    def test_policy_rejects_invalid_scales_even_without_cli_validation(self):
        for invalid in (np.nan, np.inf, -np.inf, 0., -1., 2.000001):
            values = fixture("follow")
            values[0].align_speed_scale = invalid
            with self.subTest(scale=invalid), self.assertRaises(AlignmentFault):
                make_plan(values)

    def test_historical_namespace_uses_original_scale_without_changing_the_path(self):
        values = fixture("follow")
        legacy = make_plan(values)
        del values[0].align_speed_scale
        missing = make_plan(values)
        self.assertEqual(missing.speed_scale, 1.)
        self.assertEqual(missing.estimated_duration, legacy.estimated_duration)
        for old_sample, new_sample in zip(legacy.samples, missing.samples):
            for name in old_sample:
                np.testing.assert_array_equal(new_sample[name], old_sample[name])
        states = values[2].states(0.)
        old = legacy.propose(states, values[2].bases, .04)
        new = missing.propose(states, values[2].bases, .04)
        np.testing.assert_array_equal(new["ur5e"], old["ur5e"])

    def run_virtual_alignment(self, speed_scale, response):
        values = fixture("follow")
        args, _, plant, targets, goals = values
        defaults = parse_args(CLI)
        for key in MOTION_LIMITS:
            setattr(args, key, getattr(defaults, key))
        # JointPlant was built by the legacy fixture; use the same actual
        # joint bounds as the policy, not the fixture's relaxed test bounds.
        plant.max_joint_speed = args.max_joint_speed
        plant.max_joint_accel = args.max_joint_accel
        args.align_speed_scale = speed_scale
        plan = make_plan(values)
        original_limits = deepcopy(plan.limits)
        original_leader = plant.joints["ur10e"].copy()
        speed_fraction, dt, tolerance = min(.5 * speed_scale, .9), .04, 2e-9
        peak_angular_speed = 0.
        for index in range(1200):
            states = plant.states(1000. + index * dt)
            previous = plan.previous_commands["ur5e"].copy()
            previous_twist = plan.previous_twists["ur5e"].copy()
            commands = plan.propose(states, plant.bases, dt)
            self.assertEqual(set(commands), {"ur5e"})
            command, state = commands["ur5e"], states["ur5e"]
            jacobian = plan.models["ur5e"].pose_and_jacobian(state.joints)[1]
            current_twist = jacobian @ command
            predicted = state.joints + command * dt
            future_twist = plan.models["ur5e"].pose_and_jacobian(predicted)[1] @ command
            self.assertLessEqual(np.max(np.abs(command)), speed_fraction * args.max_joint_speed + tolerance)
            self.assertLessEqual(np.max(np.abs(command - previous)), .5 * args.max_joint_accel * dt + tolerance)
            self.assertLessEqual(np.max(np.abs(command - state.joint_velocity)), args.max_joint_accel * dt + tolerance)
            for part, speed, acceleration in ((slice(0, 3), args.max_speed, args.max_accel),
                                             (slice(3, 6), args.max_angular_speed, args.max_angular_accel)):
                self.assertLessEqual(np.linalg.norm(current_twist[part]), speed_fraction * speed + tolerance)
                self.assertLessEqual(np.linalg.norm(current_twist[part] - previous_twist[part]),
                                     .5 * acceleration * dt + tolerance)
                self.assertLessEqual(np.linalg.norm(current_twist[part] - state.speed[part]), acceleration * dt + tolerance)
                self.assertLessEqual(np.linalg.norm(future_twist[part]), speed_fraction * speed + tolerance)
                self.assertLessEqual(np.linalg.norm(future_twist[part] - state.speed[part]), acceleration * dt + tolerance)
            ramp = plan.joint_accelerations["ur5e"]
            self.assertGreater(ramp, 0.)
            self.assertLessEqual(ramp, args.max_joint_accel + tolerance)
            delta = command - state.joint_velocity
            if np.any(delta):
                peak_delta = np.max(np.abs(delta))
                self.assertLessEqual(ramp, peak_delta / dt + tolerance)
                ramp_twist = jacobian @ (delta / peak_delta) * ramp
                self.assertLessEqual(np.linalg.norm(ramp_twist[:3]), args.max_accel + tolerance)
                self.assertLessEqual(np.linalg.norm(ramp_twist[3:]), args.max_angular_accel + tolerance)
            peak_angular_speed = max(peak_angular_speed, np.linalg.norm(current_twist[3:]))
            applied = {"ur5e": response * command + (1. - response) * state.joint_velocity}
            plant.step(applied, dt)
            plan.commit()
            if plan.done:
                break
        self.assertTrue(plan.done, plan.message)
        self.assertLess(plan.elapsed, plan.timeout)
        self.assertEqual(plan.limits, original_limits)
        np.testing.assert_array_equal(plant.joints["ur10e"], original_leader)
        final = plant.states(2000.)["ur5e"]
        self.assertLessEqual(np.max(np.abs(final.joints - goals["ur5e"])), .01)
        self.assertLessEqual(np.linalg.norm(final.tcp[:3, 3] - targets["ur5e"][:3, 3]), .002)
        self.assertLessEqual(rotation_distance(final.tcp, targets["ur5e"]), np.deg2rad(1.))
        self.assertTrue(plan._at_rest(final))
        return {"elapsed": plan.elapsed, "estimated_duration": plan.estimated_duration,
                "peak_angular_speed": peak_angular_speed}

    def test_doubled_scale_finishes_sooner_at_default_limits_with_ideal_and_lagging_feedback(self):
        for response in (1., .6):
            with self.subTest(response=response):
                legacy = self.run_virtual_alignment(1., response)
                faster = self.run_virtual_alignment(2., response)
                self.assertLess(faster["elapsed"], .8 * legacy["elapsed"])
                self.assertLess(faster["estimated_duration"], legacy["estimated_duration"])
                self.assertGreater(faster["peak_angular_speed"], 1.7 * legacy["peak_angular_speed"])
                self.assertLessEqual(faster["peak_angular_speed"], .1 + 2e-9)


if __name__ == "__main__":
    unittest.main()
