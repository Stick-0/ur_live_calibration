"""Bounded observation timing and coordinate blending, independent of robot IO."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, pose_matrix
from urcal.control.policy import CalibrationUnavailable, ControlFault
from urcal.cooperative.calibration import LiveCalibrationWindow
from urcal.cooperative.state import RelativeSnapshot


ROBOTS = ("ur5e", "ur10e")


def snapshot(at=10., shift=0., angle=0.):
    bases = {"ur5e": pose_matrix([shift, 0., .5, 0., 0., angle]),
             "ur10e": pose_matrix([.8, 0., .5, 0., 0., 0.])}
    return RelativeSnapshot(
        at=at, base_poses=bases, base_twists={name: np.zeros(6) for name in ROBOTS},
        relative_pose=inverse(bases["ur10e"]) @ bases["ur5e"], relative_twist=np.zeros(6),
        measurement_age={name: .12 for name in ROBOTS},
        measurement_times={name: at - .12 for name in ROBOTS},
        received_times={name: at - .02 for name in ROBOTS}, prediction_dt={name: 0. for name in ROBOTS})


class LiveCalibrationWindowTests(unittest.TestCase):
    def completed(self):
        window = LiveCalibrationWindow()
        window.start(snapshot(), 10., "rejected pair")
        window.complete(10.2)
        return window

    def test_window_freezes_independent_last_good_state(self):
        window = LiveCalibrationWindow()
        state = snapshot()
        original = deepcopy(state)
        window.start(state, 10.1, "jump")
        state.base_poses["ur5e"][0, 3] = 8.
        state.measurement_times["ur5e"] = 99.
        np.testing.assert_array_equal(window.frozen.base_poses["ur5e"], original.base_poses["ur5e"])
        self.assertEqual(window.frozen.measurement_times, original.measurement_times)
        self.assertTrue(window.waiting)
        self.assertEqual(window.started_at, 10.1)

    def test_repeated_fault_never_restarts_deadline_or_cluster(self):
        window = LiveCalibrationWindow()
        window.start(snapshot(), 10., "first")
        original = deepcopy(window.frozen)
        window.candidate.reset = Mock()
        window.start(snapshot(11., .05), 11., "later")
        window.candidate.reset.assert_not_called()
        self.assertEqual(window.started_at, 10.)
        self.assertEqual(window.reason, "later")
        np.testing.assert_array_equal(window.frozen.base_poses["ur5e"], original.base_poses["ur5e"])

    def test_deadline_is_checked_before_late_success_and_cannot_restart(self):
        window = LiveCalibrationWindow()
        window.start(snapshot(), 10., "awaiting evidence")
        window.check_deadline(12.)
        for operation in (lambda: window.check_deadline(12.001), lambda: window.complete(12.002),
                          lambda: window.start(snapshot(12.003), 12.003, "fresh frame")):
            with self.assertRaisesRegex(CalibrationUnavailable, "exceeded 2s"):
                operation()
        self.assertTrue(window.waiting)
        self.assertEqual(window.reacquisitions, 0)

    def test_invalid_or_reversed_deadline_clock_is_a_control_fault(self):
        for value in (-1., np.nan, np.inf, "bad"):
            with self.subTest(value=value), self.assertRaises(ControlFault):
                LiveCalibrationWindow().check_deadline(value)
        window = LiveCalibrationWindow()
        window.start(snapshot(), 10., "bad pair")
        window.check_deadline(10.1)
        with self.assertRaisesRegex(ControlFault, "backwards"):
            window.check_deadline(10.09)

    def test_completion_requires_observation_and_starts_from_frozen_coordinates(self):
        window = LiveCalibrationWindow()
        with self.assertRaisesRegex(ControlFault, "requires an active"):
            window.complete(10.)
        window.start(snapshot(), 10., "bad pair")
        window.complete(10.2)
        self.assertFalse(window.waiting)
        self.assertTrue(window.blending)
        self.assertEqual(window.reacquisitions, 1)
        result = window.blend(snapshot(10.2, .05), 10.2, "ur10e")
        self.assertEqual(result.base_poses["ur5e"][0, 3], 0.)

    def test_blend_uses_at_least_declared_duration_and_recomputes_relative_pose(self):
        window = self.completed()
        target = snapshot(10.2, .025, .05)
        for step in range(1, 6):
            result = window.blend(target, 10.2 + step * .1, "ur10e")
            self.assertAlmostEqual(result.base_poses["ur5e"][0, 3], .025 * step / 5, places=10)
            np.testing.assert_allclose(result.relative_pose,
                                       inverse(result.base_poses["ur10e"]) @ result.base_poses["ur5e"])
            if step < 5:
                self.assertTrue(window.blending)
        self.assertFalse(window.blending)
        np.testing.assert_allclose(result.base_poses["ur5e"], target.base_poses["ur5e"], atol=1e-12)
        repeated = window.blend(snapshot(10.7, -.1), 10.7, "ur10e")
        np.testing.assert_array_equal(repeated.base_poses["ur5e"], result.base_poses["ur5e"])

    def test_large_replacement_respects_rate_caps_after_minimum_duration(self):
        window = self.completed()
        previous = snapshot().base_poses
        target = snapshot(10.2, .15, .5)
        for step in range(1, 17):
            result = window.blend(target, 10.2 + step * .1, "ur10e")
            for name in ROBOTS:
                dp = np.linalg.norm(result.base_poses[name][:3, 3] - previous[name][:3, 3])
                dr = Rotation.from_matrix(result.base_poses[name][:3, :3]
                                          @ previous[name][:3, :3].T).magnitude()
                self.assertLessEqual(dp, .10 * .1 + 1e-12)
                self.assertLessEqual(dr, .35 * .1 + 1e-12)
            previous = deepcopy(result.base_poses)
            if step == 5:
                self.assertTrue(window.blending)
        self.assertFalse(window.blending)
        np.testing.assert_allclose(result.base_poses["ur5e"], target.base_poses["ur5e"], atol=1e-12)

    def test_repeated_or_earlier_state_reads_cannot_advance_blend(self):
        window = self.completed()
        first = window.blend(snapshot(10.3, .1, .2), 10.3, "ur10e")
        changed_target = snapshot(10.3, -.1, -.2)
        for now in (10.3, 10.25, 10.3):
            following = window.blend(changed_target, now, "ur10e")
            for name in ROBOTS:
                np.testing.assert_array_equal(following.base_poses[name], first.base_poses[name])
        resumed = window.blend(changed_target, 10.4, "ur10e")
        self.assertLess(resumed.base_poses["ur5e"][0, 3], first.base_poses["ur5e"][0, 3])

    def test_blending_never_refreshes_camera_ages_or_adds_visual_velocity(self):
        window = self.completed()
        measured = snapshot(10.3, .05, .1)
        measured = replace(measured, base_twists={name: np.ones(6) for name in ROBOTS},
                           relative_twist=np.ones(6))
        result = window.blend(measured, 10.3, "ur5e")
        for name in ("at", "measurement_age", "measurement_times", "received_times", "prediction_dt"):
            self.assertEqual(getattr(result, name), getattr(measured, name))
        np.testing.assert_allclose(result.relative_pose,
                                   inverse(result.base_poses["ur5e"]) @ result.base_poses["ur10e"])
        for velocity in [*result.base_twists.values(), result.relative_twist]:
            np.testing.assert_array_equal(velocity, np.zeros(6))
        result.measurement_times["ur5e"] = 99.
        self.assertNotEqual(measured.measurement_times["ur5e"], 99.)

    def test_new_dropout_freezes_current_blended_state_and_has_own_deadline(self):
        window = self.completed()
        current = window.blend(snapshot(10.3, .1), 10.3, "ur10e")
        window.start(current, 10.3, "another dropout")
        self.assertTrue(window.waiting)
        self.assertFalse(window.blending)
        np.testing.assert_array_equal(window.frozen.base_poses["ur5e"], current.base_poses["ur5e"])
        self.assertEqual(window.started_at, 10.3)
        window.check_deadline(12.3)

    def test_reset_discards_observation_and_blend_without_fabricating_a_success(self):
        window = self.completed()
        window.start(snapshot(10.3), 10.3, "new failure")
        window.reset()
        self.assertFalse(window.waiting)
        self.assertFalse(window.blending)
        self.assertIsNone(window.frozen)
        self.assertEqual(window.reacquisitions, 1)
        self.assertEqual(window.candidate.count, 0)
        status = window.status(12.)
        self.assertEqual(status["elapsed"], 0.)
        self.assertEqual(status["timeout"], 2.)

    def test_constructor_rejects_invalid_windows_and_blend_durations(self):
        for key in ("timeout", "blend_time", "max_shift_m", "max_shift_deg"):
            for value in (0., -1., np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    LiveCalibrationWindow(**{key: value})


if __name__ == "__main__":
    unittest.main()
