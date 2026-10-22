"""Calibration acceptance and Cartesian command bounds without robot drivers."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.control.geometry import pose_matrix, rotation_distance
from urcal.control.policy import CalibrationFrame, CalibrationTracker, ControlFault, bounded_twist


def frame(sequence=1, captured_at=10.0, poses=None, **changes):
    if poses is None:
        poses = {"ur5e": pose_matrix([-0.3, 0, 1, 0, 0, 0]),
                 "ur10e": pose_matrix([0.3, 0, 1, 0, 0, 0])}
    result = CalibrationFrame(sequence, captured_at, poses,
                              {"ur5e": 1.0, "ur10e": 1.0},
                              {"ur5e": 2.0, "ur10e": 2.0},
                              {"ur5e": 7, "ur10e": 7})
    return replace(result, **changes)


class CalibrationTrackerTests(unittest.TestCase):
    def test_first_rejection_persists_between_frames_and_clears_on_acceptance(self):
        tracker = CalibrationTracker(stable_frames=2)
        bad = frame(raw_mean_px={"ur5e": 4.2, "ur10e": 1.},
                    raw_max_px={"ur5e": 12.5, "ur10e": 2.})
        with self.assertRaises(ControlFault) as caught:
            tracker.update(bad, 10.01)
        reason = str(caught.exception)
        for detail in ("ur5e", "mean=4.20px limit=5.00px", "max=12.50px limit=8.00px",
                       "points=7/7 min=5"):
            self.assertIn(detail, reason)
        for now in (10.02, 10.1, 10.7):
            with self.assertRaises(ControlFault) as waiting:
                tracker.check_fresh(now)
            self.assertEqual(str(waiting.exception), reason)
        # A duplicate cannot replace the rejected measurement with corrected data.
        with self.assertRaises(ControlFault) as duplicate:
            tracker.update(frame(), 10.02)
        self.assertEqual(str(duplicate.exception), reason)
        self.assertEqual(tracker.poses, {})
        self.assertIsNone(tracker.last_capture)
        self.assertFalse(tracker.ready)
        tracker.update(frame(2, 10.8), 10.81)
        self.assertEqual(tracker.last_rejection, "")
        with self.assertRaisesRegex(ControlFault, "not stable.*1/2"):
            tracker.check_fresh(10.82)

    def test_rejection_does_not_hide_age_of_previously_accepted_pair(self):
        tracker = CalibrationTracker(stable_frames=1)
        tracker.update(frame(), 10.01)
        with self.assertRaisesRegex(ControlFault, "PnP failed"):
            tracker.update(frame(2, 10.1, reason="ur5e PnP failed"), 10.11)
        with self.assertRaisesRegex(ControlFault, "PnP failed"):
            tracker.check_fresh(10.12)
        with self.assertRaisesRegex(ControlFault, "stale"):
            tracker.check_fresh(10.6)
        self.assertEqual(tracker.last_capture, 10.)
        self.assertEqual(tracker.last_rejection, "calibration rejected: ur5e PnP failed")

    def test_external_rejection_preserves_reason_without_accepting_any_pose(self):
        tracker = CalibrationTracker(stable_frames=1)
        tracker.reject(1, reason="base speed exceeds limit")
        with self.assertRaisesRegex(ControlFault, "base speed"):
            tracker.check_fresh(10.)
        self.assertFalse(tracker.ready)
        self.assertEqual(tracker.poses, {})
        tracker.update(frame(2), 10.01)
        self.assertEqual(tracker.last_rejection, "")
        self.assertTrue(tracker.ready)

    def test_stability_requires_distinct_samples(self):
        tracker = CalibrationTracker(stable_frames=3)
        first = frame()
        self.assertTrue(tracker.update(first, 10.01))
        for _ in range(20):
            self.assertFalse(tracker.update(first, 10.02))
        self.assertEqual(tracker.stable_count, 1)
        self.assertFalse(tracker.ready)
        with self.assertRaisesRegex(ControlFault, "not stable"):
            tracker.check_fresh(10.02)
        tracker.update(frame(2, 10.1), 10.11)
        tracker.update(frame(3, 10.2), 10.21)
        self.assertTrue(tracker.ready)
        tracker.check_fresh(10.21)

    def test_stale_capture_rejected_even_with_fresh_sequence(self):
        tracker = CalibrationTracker(stable_frames=1)
        tracker.update(frame(), 10.01)
        previous = tracker.poses
        with self.assertRaisesRegex(ControlFault, "stale"):
            tracker.update(frame(2, 10.1), 11.0)
        self.assertEqual(tracker.last_capture, 10.0)
        self.assertFalse(tracker.ready)
        for name in previous:
            np.testing.assert_array_equal(tracker.poses[name], previous[name])

    def test_stale_duplicate_does_not_refresh_acceptance(self):
        tracker = CalibrationTracker(stable_frames=1)
        tracker.update(frame(), 10.01)
        with self.assertRaisesRegex(ControlFault, "stale"):
            tracker.update(frame(1, 11.0), 11.01)
        self.assertEqual(tracker.last_capture, 10.0)
        self.assertFalse(tracker.ready)

    def test_capture_gap_requires_new_consecutive_stable_samples(self):
        for check_timeout in (False, True):
            with self.subTest(check_timeout=check_timeout):
                tracker = CalibrationTracker(stable_frames=2)
                tracker.update(frame(), 10.01)
                tracker.update(frame(2, 10.1), 10.11)
                self.assertTrue(tracker.ready)
                if check_timeout:
                    with self.assertRaisesRegex(ControlFault, "stale"):
                        tracker.check_fresh(10.7)
                    self.assertFalse(tracker.ready)
                tracker.update(frame(3, 10.8), 10.81)
                self.assertFalse(tracker.ready)
                tracker.update(frame(4, 10.9), 10.91)
                self.assertTrue(tracker.ready)

    def test_quality_failure_preserves_pair_and_requires_new_stable_samples(self):
        tracker = CalibrationTracker(stable_frames=2)
        tracker.update(frame(), 10.01)
        tracker.update(frame(2, 10.1), 10.11)
        previous = tracker.poses
        with self.assertRaisesRegex(ControlFault, "reprojection"):
            tracker.update(frame(3, 10.2, raw_max_px={"ur5e": 2.0, "ur10e": 50.0}), 10.21)
        self.assertFalse(tracker.ready)
        self.assertEqual(tracker.last_capture, 10.1)
        for name in previous:
            np.testing.assert_array_equal(tracker.poses[name], previous[name])
        # Changing the rejected frame's contents cannot turn it into a new sample.
        self.assertFalse(tracker.update(frame(3, 10.3), 10.31))
        self.assertEqual(tracker.stable_count, 0)
        tracker.update(frame(4, 10.4), 10.41)
        self.assertFalse(tracker.ready)
        tracker.update(frame(5, 10.5), 10.51)
        self.assertTrue(tracker.ready)

    def test_rejects_failure_reason_missing_pair_and_bad_quality(self):
        bad_frames = [
            frame(reason="PnP failed"),
            frame(poses={"ur5e": np.eye(4)}),
            frame(valid_counts={"ur5e": 7, "ur10e": 4}),
            frame(valid_counts={"ur5e": 7, "ur10e": 5.5}),
            frame(raw_mean_px={"ur5e": 1, "ur10e": float("nan")}),
            frame(raw_max_px={"ur5e": 1, "ur10e": -1}),
            frame(raw_mean_px={"ur5e": 1, "ur10e": 5.1}, raw_max_px={"ur5e": 2, "ur10e": 6}),
        ]
        for bad in bad_frames:
            with self.subTest(frame=bad):
                tracker = CalibrationTracker()
                with self.assertRaises(ControlFault):
                    tracker.update(bad, 10.01)
                self.assertIsNone(tracker.last_capture)
                self.assertEqual(tracker.poses, {})

    def test_default_mean_limit_accepts_five_but_retains_maximum_limit(self):
        for mean, maximum, accepted in ((4.2, 7., True), (5., 8., True),
                                        (5.0001, 8., False), (4.2, 8.0001, False)):
            with self.subTest(mean=mean, maximum=maximum):
                tracker = CalibrationTracker(stable_frames=1)
                sample = frame(raw_mean_px={"ur5e": mean, "ur10e": mean},
                               raw_max_px={"ur5e": maximum, "ur10e": maximum})
                if accepted:
                    self.assertTrue(tracker.update(sample, 10.01))
                    self.assertTrue(tracker.ready)
                else:
                    with self.assertRaisesRegex(ControlFault, "raw reprojection"):
                        tracker.update(sample, 10.01)
                    self.assertFalse(tracker.ready)
                    self.assertEqual(tracker.poses, {})

    def test_disabled_reprojection_accepts_large_finite_mean_and_maximum(self):
        tracker = CalibrationTracker(stable_frames=1, check_reprojection=False)
        sample = frame(raw_mean_px={name: 500. for name in ("ur5e", "ur10e")},
                       raw_max_px={name: 1000. for name in ("ur5e", "ur10e")})
        self.assertTrue(tracker.update(sample, 10.01))
        tracker.check_fresh(10.02)
        self.assertTrue(tracker.ready)
        self.assertEqual(tracker.last_capture, 10.)

    def test_disabled_reprojection_still_rejects_invalid_statistics(self):
        for mean, maximum in ((float("nan"), 10.), (1., float("nan")),
                              (float("inf"), float("inf")), (1., float("inf")),
                              (-1., 2.), (2., -1.), (2., 1.)):
            with self.subTest(mean=mean, maximum=maximum):
                tracker = CalibrationTracker(stable_frames=1, check_reprojection=False)
                sample = frame(raw_mean_px={"ur5e": mean, "ur10e": 1.},
                               raw_max_px={"ur5e": maximum, "ur10e": 2.})
                with self.assertRaisesRegex(ControlFault, "reprojection"):
                    tracker.update(sample, 10.01)
                self.assertFalse(tracker.ready)
                self.assertEqual(tracker.poses, {})

    def test_disabled_reprojection_preserves_other_calibration_checks(self):
        for bad in (frame(reason="PnP failed"), frame(captured_at=9.),
                    frame(poses={"ur5e": np.eye(4)}),
                    frame(valid_counts={"ur5e": 4, "ur10e": 7})):
            with self.subTest(frame=bad):
                tracker = CalibrationTracker(stable_frames=1, check_reprojection=False)
                with self.assertRaises(ControlFault):
                    tracker.update(bad, 10.01)
                self.assertFalse(tracker.ready)
        tracker = CalibrationTracker(stable_frames=2, check_reprojection=False)
        tracker.update(frame(), 10.01)
        self.assertFalse(tracker.ready)
        moved = frame(2, 10.1)
        moved.poses["ur5e"][0, 3] += 1.
        with self.assertRaisesRegex(ControlFault, "jump"):
            tracker.update(moved, 10.11)
        self.assertEqual(tracker.last_capture, 10.)

    def test_disabled_check_does_not_allow_infinite_configured_limits(self):
        for option in ("max_mean_px", "max_pixel_px"):
            with self.subTest(option=option), self.assertRaises(ControlFault):
                CalibrationTracker(check_reprojection=False, **{option: float("inf")})

    def test_low_keypoint_count_includes_observed_and_required_counts(self):
        tracker = CalibrationTracker(min_points=6)
        with self.assertRaisesRegex(ControlFault, r"ur5e.*points=4, required=6\.\.7"):
            tracker.update(frame(valid_counts={"ur5e": 4, "ur10e": 7}), 10.01)

    def test_rejects_future_nonmonotonic_time_and_sequence(self):
        for bad in (frame(2, 10.2), frame(2, 10.0), frame(2, 9.99),
                    frame(0, 10.05), frame(2, float("nan"))):
            with self.subTest(frame=bad):
                tracker = CalibrationTracker(stable_frames=1)
                tracker.update(frame(), 10.01)
                with self.assertRaises(ControlFault):
                    tracker.update(bad, 10.1)
                self.assertEqual(tracker.last_capture, 10.0)
                self.assertFalse(tracker.ready)

    def test_raw_consecutive_jump_check_is_independent_of_filter_lag(self):
        tracker = CalibrationTracker(alpha=0.01)
        for i in range(5):
            poses = frame().poses
            for pose in poses.values():
                pose[0, 3] += i * 0.01
            tracker.update(frame(i, 10 + i * 0.05, poses), 10.01 + i * 0.05)
        self.assertTrue(tracker.ready)
        self.assertLess(tracker.poses["ur5e"][0, 3], -0.295)
        poses = frame().poses
        for pose in poses.values():
            pose[0, 3] += 0.06
        with self.assertRaisesRegex(ControlFault, "jump"):
            tracker.update(frame(5, 10.3, poses), 10.31)

    def test_relative_translation_jump_rejected_when_individual_shifts_pass(self):
        tracker = CalibrationTracker()
        tracker.update(frame(), 10.01)
        poses = frame().poses
        poses["ur5e"][0, 3] -= 0.01
        poses["ur10e"][0, 3] += 0.01
        with self.assertRaisesRegex(ControlFault, "relative base"):
            tracker.update(frame(2, 10.1, poses), 10.11)

    def test_relative_rotation_jump_rejected_when_individual_rotations_pass(self):
        tracker = CalibrationTracker()
        tracker.update(frame(), 10.01)
        poses = {"ur5e": pose_matrix([-0.3, 0, 1, 0, 0, np.deg2rad(1.2)]),
                 "ur10e": pose_matrix([0.3, 0, 1, 0, 0, np.deg2rad(-1.2)])}
        with self.assertRaisesRegex(ControlFault, "relative base"):
            tracker.update(frame(2, 10.1, poses), 10.11)

    def test_rotation_wrap_filters_on_so3(self):
        tracker = CalibrationTracker(alpha=0.5)
        poses = {name: pose_matrix([0, 0, 1, 0, 0, np.deg2rad(179.8)]) for name in ("ur5e", "ur10e")}
        tracker.update(frame(1, 10.0, poses), 10.01)
        wrapped = {name: pose_matrix([0, 0, 1, 0, 0, np.deg2rad(-179.8)]) for name in poses}
        tracker.update(frame(2, 10.1, wrapped), 10.11)
        halfway = pose_matrix([0, 0, 1, 0, 0, np.pi])
        self.assertLess(rotation_distance(tracker.poses["ur5e"], halfway), 1e-8)

    def test_rejects_invalid_rotation_and_returns_defensive_copies(self):
        tracker = CalibrationTracker()
        first = frame()
        tracker.update(first, 10.01)
        first.poses["ur5e"][0, 3] = 99
        returned = tracker.poses
        returned["ur5e"][0, 3] = 99
        self.assertAlmostEqual(tracker.poses["ur5e"][0, 3], -0.3)
        invalid = frame(2, 10.1)
        invalid.poses["ur5e"][0, 0] = -1  # reflection, determinant -1
        with self.assertRaisesRegex(ControlFault, "proper rotation"):
            tracker.update(invalid, 10.11)


class BoundedTwistTests(unittest.TestCase):
    def test_velocity_and_acceleration_norm_limits(self):
        current = np.eye(4)
        target = pose_matrix([0.05, 0.05, 0, 0.15, 0.15, 0])
        previous = np.zeros(6)
        for _ in range(100):
            command = bounded_twist(current, target, previous, 0.01)
            self.assertLessEqual(np.linalg.norm(command[:3]), 0.02 + 1e-12)
            self.assertLessEqual(np.linalg.norm(command[3:]), 0.1 + 1e-12)
            self.assertLessEqual(np.linalg.norm(command[:3] - previous[:3]), 0.001 + 1e-12)
            self.assertLessEqual(np.linalg.norm(command[3:] - previous[3:]), 0.005 + 1e-12)
            previous = command
        self.assertAlmostEqual(np.linalg.norm(previous[:3]), 0.02)
        self.assertAlmostEqual(np.linalg.norm(previous[3:]), 0.1)

    def test_large_target_error_rejected_before_any_clipping(self):
        for target in (pose_matrix([0.101, 0, 0, 0, 0, 0]),
                       pose_matrix([0, 0, 0, 0, np.deg2rad(26), 0])):
            with self.subTest(target=target), self.assertRaisesRegex(ControlFault, "final TCP target"):
                bounded_twist(np.eye(4), target, np.zeros(6), 0.02)

    def test_twist_is_expressed_in_parent_base_frame(self):
        current = pose_matrix([0, 0, 0, 0, 0, np.pi / 2])
        relative = pose_matrix([0.01, 0, 0, 0.1, 0, 0])
        target = current @ relative
        command = bounded_twist(current, target, np.zeros(6), 1.0)
        np.testing.assert_allclose(command, [0, 0.01, 0, 0, 0.1, 0], atol=1e-10)

    def test_reversal_respects_acceleration_and_zero_target_decelerates(self):
        previous = np.array([0.02, 0, 0, 0.1, 0, 0])
        target = pose_matrix([-0.05, 0, 0, -0.2, 0, 0])
        command = bounded_twist(np.eye(4), target, previous, 0.02)
        np.testing.assert_allclose(command, [0.018, 0, 0, 0.09, 0, 0], atol=1e-12)
        stopped = bounded_twist(np.eye(4), np.eye(4), previous, 0.02)
        np.testing.assert_allclose(stopped, command, atol=1e-12)

    def test_rotation_representation_wrap_has_small_error(self):
        current = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(179.9)])
        target = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(-179.9)])
        command = bounded_twist(current, target, np.zeros(6), 1.0)
        np.testing.assert_allclose(command, [0, 0, 0, 0, 0, np.deg2rad(0.2)], atol=1e-10)

    def test_rejects_nonfinite_parameters_and_invalid_previous_twist(self):
        cases = [{"dt": 0}, {"dt": float("nan")}, {"max_linear": float("inf")},
                 {"max_accel": -1}, {"kp_angular": float("nan")},
                 {"previous_twist": np.full(6, float("nan"))},
                 {"previous_twist": np.zeros((6, 1))},
                 {"previous_twist": np.array([0.03, 0, 0, 0, 0, 0])}]
        for bad in cases:
            kwargs = {"current_tcp": np.eye(4), "target_tcp": np.eye(4),
                      "previous_twist": np.zeros(6), "dt": 0.02}
            kwargs.update(bad)
            with self.subTest(bad=bad), self.assertRaises(ControlFault):
                bounded_twist(**kwargs)


if __name__ == "__main__":
    unittest.main()
