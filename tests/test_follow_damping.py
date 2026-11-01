"""Independent acceleration-limited delayed plant for follower overshoot.

The plant consumes a queued velocity request and ramps its physical velocity;
it does not import the motion governor or pretend to model a UR servo. Tests
exercise 40--80 ms sampling, one/two sample command delay, target stop/reversal,
and translation/rotation units without opening a device connection.
"""

from collections import deque
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.cooperative.feedback import damped_feedback_velocity


def correction(error, velocity, **overrides):
    options = dict(gain=3., speed_limit=.1, deceleration=.1, reaction_time=.12)
    options.update(overrides)
    return damped_feedback_velocity(error, velocity, **options)


def delayed_response(*, damped, dt=.04, delay=2, initial_error=.05, moving=False,
                     angular=False):
    """Scalar motion with a real acceleration ramp and independently queued commands."""
    position, velocity, command, feedback, target = -initial_error, 0., 0., 0., 0.
    queue = deque([0.] * delay)
    acceleration, cap, deadband = ((.5, .15, np.deg2rad(.3)) if angular
                                   else (.1, .1, .001))
    rows = []
    for tick in range(round(10. / dt)):
        now = tick * dt
        leader = (.05 if now < 1. else -.05 if 3. <= now < 4. else 0.) if moving else 0.
        target += leader * dt
        queued = queue.popleft()
        velocity += np.clip(queued - velocity, -acceleration * dt, acceleration * dt)
        position += velocity * dt
        error = target - position
        outside = np.sign(error) * max(abs(error) - deadband, 0.)
        if damped:
            feedback = correction([outside, 0., 0.], [velocity - leader, 0., 0.],
                                  speed_limit=cap, deceleration=acceleration)[0]
        else:
            # Previous control law: P correction independently slew-limited at
            # half the physical acceleration, then another total-command ramp.
            desired_feedback = np.clip(3. * outside, -cap, cap)
            feedback += np.clip(desired_feedback - feedback,
                                -.5 * acceleration * dt, .5 * acceleration * dt)
        desired_command = leader + feedback
        command += np.clip(desired_command - command, -acceleration * dt, acceleration * dt)
        queue.append(command)
        rows.append((now, error, velocity, command, feedback, leader))
    return np.asarray(rows)


def settling_time(samples, threshold):
    remaining_peak = np.maximum.accumulate(np.abs(samples[::-1, 1]))[::-1]
    settled = np.flatnonzero(remaining_peak < threshold)
    return float(samples[settled[0], 0]) if len(settled) else np.inf


class FollowDampingTests(unittest.TestCase):
    def test_delayed_translation_stops_before_target_without_slowing_catch_up(self):
        for dt in (.04, .08):
            for delay in (1, 2):
                with self.subTest(dt=dt, delay=delay):
                    old = delayed_response(damped=False, dt=dt, delay=delay)
                    new = delayed_response(damped=True, dt=dt, delay=delay)
                    self.assertGreater(-old[:, 1].min(), .01, "fixture must expose old overshoot")
                    self.assertGreaterEqual(new[:, 1].min(), -.001,
                                            "follower crossed the stationary target")
                    self.assertLess(settling_time(new, .002), settling_time(old, .002) * .8)
                    self.assertGreater(np.max(abs(new[:, 2])), .05,
                                       "overshoot must not be hidden by making tracking slow")
                    self.assertLess(np.sum(abs(new[:, 1])), np.sum(abs(old[:, 1])) * .8)
                    # Independent command and physical acceleration bounds.
                    for column in (2, 3):
                        self.assertLessEqual(np.max(abs(np.diff(np.r_[0., new[:, column]]))) / dt,
                                             .1 + 1e-10)

    def test_angular_error_settles_without_repeated_overshoot(self):
        for dt in (.04, .08):
            with self.subTest(dt=dt):
                samples = delayed_response(damped=True, dt=dt, delay=2,
                                           initial_error=np.deg2rad(10.), angular=True)
                self.assertGreaterEqual(samples[:, 1].min(), -np.deg2rad(.3))
                self.assertLess(settling_time(samples, np.deg2rad(.5)), 3.)
                self.assertLessEqual(np.max(abs(samples[:, 3])), .15 + 1e-12)

    def test_moving_leader_stop_and_reversal_converge_with_feedforward(self):
        samples = delayed_response(damped=True, initial_error=0., moving=True)
        previous = delayed_response(damped=False, initial_error=0., moving=True)
        self.assertLess(settling_time(samples, .002), 6.5)
        self.assertLess(np.max(abs(samples[-25:, 1])), .002)
        self.assertLess(-samples[:, 1].min(), -previous[:, 1].min() * .8)
        self.assertGreater(np.max(samples[:, 2]), .04)
        self.assertLess(np.min(samples[:, 2]), -.04)
        # Equal measured follower and target velocities are zero relative
        # motion: damping must not subtract the leader's feedforward speed.
        leader = np.array([.07, -.02, .01])
        np.testing.assert_array_equal(correction(np.zeros(3), leader - leader), np.zeros(3))

    def test_brakes_when_closing_and_preserves_response_to_distant_target(self):
        approaching = correction([.01, 0., 0.], [.05, 0., 0.])
        stationary = correction([.01, 0., 0.], [0., 0., 0.])
        self.assertGreater(approaching[0], 0.)
        self.assertLess(approaching[0], stationary[0] * .5)
        distant = correction([.2, 0., 0.], [.05, 0., 0.])
        self.assertAlmostEqual(distant[0], .1)
        # An already closing robot may need a braking demand before the
        # measured position crosses zero; the caller ramps this transition.
        self.assertLess(correction([.005, 0., 0.], [.07, 0., 0.])[0], 0.)

    def test_frame_rotation_does_not_change_control_behavior(self):
        rotation = Rotation.from_rotvec([.3, -.6, .7]).as_matrix()
        error, velocity = np.array([.02, -.01, .003]), np.array([.03, .02, -.01])
        np.testing.assert_allclose(correction(rotation @ error, rotation @ velocity),
                                   rotation @ correction(error, velocity), atol=1e-15)

    def test_zero_damping_preserves_legacy_proportional_request(self):
        error = np.array([.02, -.01, .003])
        np.testing.assert_allclose(correction(error, [99., -22., .5], reaction_time=0.), 3. * error)
        np.testing.assert_allclose(correction([1., 0., 0.], [-99., 0., 0.], reaction_time=0.),
                                   [.1, 0., 0.])

    def test_invalid_motion_inputs_are_rejected(self):
        for name, value in (("gain", 0.), ("speed_limit", -1.), ("deceleration", 0.),
                            ("reaction_time", -.1), ("reaction_time", np.nan)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                correction([.01, 0., 0.], [0., 0., 0.], **{name: value})
        for error, velocity in (([.01, 0.], [0., 0., 0.]),
                                ([.01, 0., 0.], [0., np.inf, 0.])):
            with self.subTest(error=error, velocity=velocity), self.assertRaises(ValueError):
                correction(error, velocity)


if __name__ == "__main__":
    unittest.main()
