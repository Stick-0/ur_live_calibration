"""Active-only calibration grace using synthetic measurements and no hardware."""

from copy import deepcopy
import unittest

import numpy as np

from urcal.control.geometry import pose_matrix
from urcal.control.policy import ActiveCalibrationGuard, CalibrationFrame, CalibrationTracker, ControlFault, ROBOTS


def packet(sequence, captured, delta=0., reason=""):
    poses = {"ur5e": pose_matrix([-.2 + delta, .1, 2., 0., 0., 0.]),
             "ur10e": pose_matrix([.4, .1, 2., 0., 0., 0.])}
    return CalibrationFrame(sequence, captured, poses, {name: 2. for name in ROBOTS},
                            {name: 4. for name in ROBOTS}, {name: 7 for name in ROBOTS}, reason)


class ActiveCalibrationTests(unittest.TestCase):
    def warmed(self, *, max_age=.5):
        tracker = CalibrationTracker(max_age=max_age, stable_frames=5, check_reprojection=False)
        for index in range(5):
            tracker.update(packet(index, 10. + index * .04), 10.02 + index * .04)
        guard = ActiveCalibrationGuard(.3)
        guard.arm(tracker, 10.18)
        return tracker, guard

    def reject(self, tracker, sequence=5, now=10.20, *, delta=.119, reason=""):
        frame = packet(sequence, now - .01, delta, reason)
        with self.assertRaises(ControlFault):
            tracker.update(frame, now)
        return frame

    def test_explicit_arm_requires_all_five_startup_frames(self):
        tracker, guard = CalibrationTracker(stable_frames=5), ActiveCalibrationGuard()
        with self.assertRaisesRegex(ControlFault, "explicit Arm"):
            guard.check(tracker, 10.)
        for index in range(4):
            tracker.update(packet(index, 10. + index * .04), 10.01 + index * .04)
            with self.assertRaisesRegex(ControlFault, "not stable"):
                guard.arm(tracker, 10.01 + index * .04)
            self.assertFalse(guard.armed)
        tracker.update(packet(4, 10.16), 10.17)
        guard.arm(tracker, 10.18)
        self.assertTrue(guard.check(tracker, 10.19))

    def test_119mm_jump_preserves_last_accepted_pose_capture_and_stability(self):
        tracker, guard = self.warmed()
        original = deepcopy((tracker.poses, tracker._raw_poses, tracker.last_capture))
        self.reject(tracker)
        self.assertIn("0.1190m", tracker.last_rejection)
        self.assertFalse(guard.check(tracker, 10.20))
        self.assertFalse(guard.check(tracker, 10.24))
        self.assertEqual(guard.invalid_since, 10.20)
        self.assertEqual(tracker.last_capture, original[2])
        self.assertEqual(tracker.stable_count, 0)
        for name in ROBOTS:
            np.testing.assert_array_equal(tracker.poses[name], original[0][name])
            np.testing.assert_array_equal(tracker._raw_poses[name], original[1][name])

    def test_next_accepted_pair_recovers_active_control_but_new_arm_still_needs_five(self):
        tracker, guard = self.warmed()
        self.reject(tracker)
        self.assertFalse(guard.check(tracker, 10.20))
        tracker.update(packet(6, 10.23, delta=.001), 10.24)
        self.assertEqual(tracker.stable_count, 1)
        self.assertTrue(guard.check(tracker, 10.24))
        self.assertIsNone(guard.invalid_since)
        with self.assertRaisesRegex(ControlFault, "not stable"):
            guard.arm(tracker, 10.25)
        self.assertFalse(guard.armed)
        for index in range(7, 11):
            tracker.update(packet(index, 10.23 + (index - 6) * .04, delta=.001),
                           10.24 + (index - 6) * .04)
        guard.arm(tracker, 10.41)
        self.assertTrue(guard.check(tracker, 10.42))

    def test_duplicate_and_repeated_rejections_never_extend_grace_or_capture_age(self):
        tracker, guard = self.warmed()
        duplicate = self.reject(tracker)
        self.assertFalse(guard.check(tracker, 10.20))
        self.assertFalse(tracker.update(duplicate, 10.26))
        self.assertFalse(guard.check(tracker, 10.26))
        self.reject(tracker, sequence=6, now=10.35, reason="PnP failed")
        self.assertFalse(guard.check(tracker, 10.35))
        self.assertEqual(guard.invalid_since, 10.20)
        self.assertEqual(tracker.last_capture, 10.16)
        with self.assertRaisesRegex(ControlFault, r"0\.310s.*0\.300s.*PnP failed"):
            guard.check(tracker, 10.51)
        self.assertFalse(guard.armed)
        tracker.update(packet(7, 10.52), 10.53)
        with self.assertRaisesRegex(ControlFault, "explicit Arm"):
            guard.check(tracker, 10.53)

    def test_late_good_pair_cannot_hide_an_expired_invalid_interval(self):
        tracker, guard = self.warmed()
        self.reject(tracker)
        guard.check(tracker, 10.20)
        tracker.update(packet(6, 10.51), 10.52)
        with self.assertRaisesRegex(ControlFault, r"0\.320s.*calibration jump"):
            guard.check(tracker, 10.52)
        self.assertFalse(guard.armed)

    def test_capture_age_can_expire_before_grace_without_mutating_tracker(self):
        tracker, guard = self.warmed(max_age=.25)
        self.reject(tracker, now=10.35)
        self.assertFalse(guard.check(tracker, 10.35))
        before = deepcopy((tracker.poses, tracker.last_capture, tracker.stable_count))
        with self.assertRaisesRegex(ControlFault, r"stale.*0\.260s.*calibration jump"):
            guard.check(tracker, 10.42)
        self.assertFalse(guard.armed)
        self.assertEqual((tracker.last_capture, tracker.stable_count), before[1:])
        for name in ROBOTS:
            np.testing.assert_array_equal(tracker.poses[name], before[0][name])

    def test_no_new_frame_staleness_faults_even_without_any_rejected_pair(self):
        tracker, guard = self.warmed()
        with self.assertRaisesRegex(ControlFault, "stale"):
            guard.check(tracker, 10.67)
        self.assertEqual(tracker.stable_count, 5)
        tracker.update(packet(5, 10.68), 10.69)
        with self.assertRaisesRegex(ControlFault, "explicit Arm"):
            guard.check(tracker, 10.69)

    def test_recovery_starts_a_new_grace_interval_only_for_a_later_rejection(self):
        tracker, guard = self.warmed()
        self.reject(tracker)
        guard.check(tracker, 10.20)
        tracker.update(packet(6, 10.30), 10.31)
        self.assertTrue(guard.check(tracker, 10.31))
        self.reject(tracker, sequence=7, now=10.45)
        self.assertFalse(guard.check(tracker, 10.45))
        self.assertEqual(guard.invalid_since, 10.45)
        self.assertFalse(guard.check(tracker, 10.70))

    def test_configuration_and_clock_values_are_validated(self):
        for value in (0., -1., float("nan"), float("inf"), None):
            with self.subTest(duration=value), self.assertRaises((ControlFault, ValueError)):
                ActiveCalibrationGuard(value)
        for now in (-1., float("nan"), float("inf"), None, 10.17):
            tracker, guard = self.warmed()
            with self.subTest(now=now), self.assertRaises(ControlFault):
                guard.check(tracker, now)
            self.assertFalse(guard.armed)


if __name__ == "__main__":
    unittest.main()
