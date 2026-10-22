"""Recorded Arm geometry with imperfect virtual velocity response.

The plant exercises controller lag, gain error and reported-speed noise. It is
an offline regression, not a UR firmware or continuous-time safety model.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from urcal.control.alignment import AlignmentFault, LiveAlignment
from urcal.control.geometry import rotation_distance


FIXTURE = Path(__file__).parent / "fixtures" / "alignment_speed_reserve.json"


def measured_states(values):
    return {name: SimpleNamespace(joints=np.array(state["q"]), tcp=np.array(state["tcp"]),
                                  joint_velocity=np.array(state["qd"]), speed=np.array(state["speed"]))
            for name, state in values.items()}


def recorded_alignment():
    data = json.loads(FIXTURE.read_text())
    arrays = lambda values: {name: np.array(value) for name, value in values.items()}
    bases = arrays(data["bases"])
    plan = LiveAlignment(SimpleNamespace(**data["settings"]),
                         {name: np.eye(4) for name in bases}, bases,
                         measured_states(data["initial_states"]),
                         arrays(data["targets"]), arrays(data["goal_joints"]))
    return plan, bases, data


class AlignmentSpeedReserveTests(unittest.TestCase):
    def test_recorded_measured_overspeed_still_stops_without_a_pending_command(self):
        plan, bases, data = recorded_alignment()
        states = measured_states(data["fault"]["states"])
        measured = np.linalg.norm(states["ur5e"].speed[3:])
        self.assertGreater(measured, plan.limits["max_angular_speed"])
        self.assertLess(measured, 1.01 * plan.limits["max_angular_speed"])
        limits = deepcopy(plan.limits)
        with self.assertRaisesRegex(AlignmentFault, "measured.*speed") as raised:
            plan.propose(states, bases, data["fault"]["dt"])
        self.assertEqual(raised.exception.details,
                         {"arm": "ur5e", "metric": "angular_speed", "value": measured,
                          "limit": plan.limits["max_angular_speed"], "unit": "rad/s"})
        self.assertIsNone(plan._pending)
        with self.assertRaises(AlignmentFault):
            plan.commit()
        self.assertEqual(plan.limits, limits)
        self.assertEqual(plan.elapsed, 0.)

    def test_joint_linear_and_angular_measured_limits_report_values_and_remain_hard(self):
        for metric, key, field, index, unit in (
                ("joint_speed", "max_joint_speed", "joint_velocity", 2, "rad/s"),
                ("linear_speed", "max_speed", "speed", 1, "m/s"),
                ("angular_speed", "max_angular_speed", "speed", 5, "rad/s")):
            with self.subTest(metric=metric):
                plan, bases, data = recorded_alignment()
                states = measured_states(data["initial_states"])
                limit = plan.limits[key]
                value = 1.001 * limit
                getattr(states["ur5e"], field)[index] = value
                with self.assertRaisesRegex(AlignmentFault, "measured.*speed") as raised:
                    plan.propose(states, bases, .04)
                self.assertEqual(raised.exception.details,
                                 {"arm": "ur5e", "metric": metric, "value": value,
                                  "limit": limit, "unit": unit})
                self.assertIsNone(plan._pending)
                self.assertEqual(plan.elapsed, 0.)

    def run_imperfect_plant(self, old_budget=False):
        plan, bases, data = recorded_alignment()
        # The old policy used a full command budget and a 98% desired target.
        # Reinstating that budget proves this plant reproduces the problem.
        if old_budget:
            plan.command_speed_fraction = 1.
        original_limits = deepcopy(plan.limits)
        states = measured_states(data["initial_states"])
        fixed = deepcopy(states["ur10e"])
        delayed_command = np.zeros(6)
        peak_angular = 0.
        periods = (.038, .042, .04, .041, .039)
        gain, response = 1.04, .8
        for index in range(1800):
            dt = periods[index % len(periods)]
            prior = states["ur5e"]
            previous_command = plan.previous_commands["ur5e"].copy()
            previous_twist = plan.previous_twists["ur5e"].copy()
            commands = plan.propose(states, bases, dt)
            self.assertEqual(set(commands), {"ur5e"})
            command = commands["ur5e"]
            jacobian = plan.models["ur5e"].pose_and_jacobian(prior.joints)[1]
            twist = jacobian @ command
            future_twist = plan.models["ur5e"].pose_and_jacobian(plan.predicted_joints["ur5e"])[1] @ command
            limits, tol = plan.limits, 2e-9
            self.assertLessEqual(np.max(np.abs(command)), limits["max_joint_speed"] + tol)
            self.assertLessEqual(np.max(np.abs(command - previous_command)), .5 * limits["max_joint_accel"] * dt + tol)
            self.assertLessEqual(np.max(np.abs(command - prior.joint_velocity)), limits["max_joint_accel"] * dt + tol)
            for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                       (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                self.assertLessEqual(np.linalg.norm(twist[part]), limits[speed] + tol)
                self.assertLessEqual(np.linalg.norm(future_twist[part]), limits[speed] + tol)
                self.assertLessEqual(np.linalg.norm(twist[part] - previous_twist[part]), .5 * limits[accel] * dt + tol)
                self.assertLessEqual(np.linalg.norm(twist[part] - prior.speed[part]), limits[accel] * dt + tol)
                self.assertLessEqual(np.linalg.norm(future_twist[part] - prior.speed[part]), limits[accel] * dt + tol)
            ramp = plan.joint_accelerations["ur5e"]
            self.assertGreater(ramp, 0.)
            self.assertLessEqual(ramp, limits["max_joint_accel"] + tol)
            delta = command - prior.joint_velocity
            peak_delta = np.max(np.abs(delta))
            if peak_delta:
                self.assertLessEqual(ramp, peak_delta / dt + tol)
                ramp_twist = jacobian @ (delta / peak_delta) * ramp
                self.assertLessEqual(np.linalg.norm(ramp_twist[:3]), limits["max_accel"] + tol)
                self.assertLessEqual(np.linalg.norm(ramp_twist[3:]), limits["max_angular_accel"] + tol)
            # One control-period transport delay, first-order servo lag and
            # 4% velocity gain. Integrate the resulting actual velocity, not
            # the requested command. Add small deterministic speed-read noise.
            actual_qd = (1. - response) * prior.joint_velocity + response * gain * delayed_command
            actual_q = prior.joints + actual_qd * dt
            actual_tcp, actual_jacobian = plan.models["ur5e"].pose_and_jacobian(actual_q)
            actual_speed = actual_jacobian @ actual_qd
            noise = np.zeros(6)
            noise[3:] = .00015 * np.sin(index * .7) * np.array([1., -.5, .25])
            reported_speed = actual_speed + noise
            delayed_command = command.copy()
            plan.commit()
            states["ur5e"] = SimpleNamespace(joints=actual_q, joint_velocity=actual_qd,
                                              tcp=actual_tcp, speed=reported_speed)
            peak_angular = max(peak_angular, np.linalg.norm(reported_speed[3:]))
            if old_budget and peak_angular > limits["max_angular_speed"]:
                # The next real policy tick must reject, not ignore, this
                # hard-limit violation. The revised policy avoids generating it.
                with self.assertRaisesRegex(AlignmentFault, "measured.*speed"):
                    plan.propose(states, bases, periods[(index + 1) % len(periods)])
                return {"failed_after": index + 1, "peak_angular": peak_angular}
            self.assertLessEqual(np.max(np.abs(actual_qd)), limits["max_joint_speed"] + tol)
            self.assertLessEqual(np.max(np.abs(actual_qd - prior.joint_velocity)) / dt, limits["max_joint_accel"] + tol)
            for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                       (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                self.assertLessEqual(np.linalg.norm(reported_speed[part]), limits[speed] + tol)
                self.assertLessEqual(np.linalg.norm(reported_speed[part] - prior.speed[part]) / dt, limits[accel] + tol)
            np.testing.assert_array_equal(states["ur10e"].joints, fixed.joints)
            np.testing.assert_array_equal(states["ur10e"].tcp, fixed.tcp)
            np.testing.assert_array_equal(plan.predicted_joints["ur10e"], fixed.joints)
            if plan.done:
                break
        self.assertFalse(old_budget, "legacy full budget should expose the overspeed")
        self.assertTrue(plan.done, plan.message)
        self.assertLess(plan.elapsed, plan.timeout)
        self.assertEqual(plan.limits, original_limits)
        final = states["ur5e"]
        self.assertLessEqual(np.max(np.abs(final.joints - plan.goal_joints["ur5e"])), .01)
        self.assertLessEqual(np.linalg.norm(final.tcp[:3, 3] - plan.targets["ur5e"][:3, 3]), .002)
        self.assertLessEqual(rotation_distance(final.tcp, plan.targets["ur5e"]), np.deg2rad(1.))
        self.assertTrue(plan._at_rest(final))
        return {"elapsed": plan.elapsed, "peak_angular": peak_angular}

    def test_recorded_alignment_settles_with_delayed_gain_error_and_noisy_speed_feedback(self):
        result = self.run_imperfect_plant()
        self.assertLess(result["peak_angular"], .1)
        self.assertGreater(result["peak_angular"], .08)

    def test_same_virtual_response_exposes_previous_full_budget_overspeed(self):
        result = self.run_imperfect_plant(old_budget=True)
        self.assertLess(result["failed_after"], 30)
        self.assertGreater(result["peak_angular"], .1)


if __name__ == "__main__":
    unittest.main()
