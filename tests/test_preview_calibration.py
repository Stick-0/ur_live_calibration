"""Preview-only recovery from a stale synthetic baseline; no robot drivers."""

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from urcal.control.geometry import pose_matrix
from urcal.control.policy import (
    CalibrationFrame, CalibrationTracker, ControlFault, PreviewCalibrationRecovery, ROBOTS,
)


def packet(sequence, captured, dx=.119, other_dx=0., **changes):
    poses = {"ur5e": pose_matrix([-.2 + dx, .1, 2., 0., 0., 0.]),
             "ur10e": pose_matrix([.4 + other_dx, .1, 2., 0., 0., 0.])}
    result = CalibrationFrame(sequence, captured, poses, {name: 2. for name in ROBOTS},
                              {name: 4. for name in ROBOTS}, {name: 7 for name in ROBOTS})
    return replace(result, **changes)


class PreviewCalibrationTests(unittest.TestCase):
    def baseline(self, **options):
        tracker = CalibrationTracker(**options)
        tracker.update(packet(0, 9.96, dx=0.), 9.97)
        tracker.update(packet(1, 10., dx=0.), 10.01)
        return tracker

    def test_stale_two_frame_baseline_recovers_only_after_five_new_frames(self):
        tracker = self.baseline()
        recovery = PreviewCalibrationRecovery()
        before = deepcopy(tracker.__dict__)
        for index in range(5):
            frame = packet(index + 2, 10.60 + index * .04)
            ready = recovery.offer(frame, frame.captured_at + .01, tracker)
            self.assertEqual(recovery.count, index + 1)
            if index < 4:
                self.assertIsNone(ready)
        self.assertTrue(ready.ready)
        self.assertEqual(ready.last_capture, 10.76)
        self.assertEqual(ready._last_sequence, 6)
        ready.check_fresh(10.77)
        np.testing.assert_array_equal(ready.poses["ur5e"], frame.poses["ur5e"])
        for key, original in before.items():
            if key in ("_poses", "_raw_poses"):
                for name in ROBOTS:
                    np.testing.assert_array_equal(getattr(tracker, key)[name], original[name])
            else:
                self.assertEqual(getattr(tracker, key), original)
        self.assertEqual(tracker.stable_count, 2)
        self.assertFalse(tracker.ready)

    def test_no_baseline_or_fresh_baseline_cannot_be_replaced(self):
        recovery = PreviewCalibrationRecovery()
        self.assertIsNone(recovery.offer(packet(2, 10.6), 10.61, CalibrationTracker()))
        self.assertEqual(recovery.count, 0)
        tracker = self.baseline()
        for index in range(5):
            at = 10.1 + index * .04
            self.assertIsNone(recovery.offer(packet(index + 2, at), at + .01, tracker))
        self.assertEqual(recovery.count, 0)
        # Returning to the original branch clears a partial replacement too.
        recovery.reset()
        recovery.offer(packet(7, 10.6), 10.61, tracker)
        tracker.update(packet(8, 10.63, dx=0.), 10.64)
        self.assertIsNone(recovery.offer(packet(9, 10.65), 10.66, tracker))
        self.assertEqual(recovery.count, 0)

    def test_template_may_have_consumed_the_same_rejected_sequence(self):
        tracker = self.baseline()
        recovery = PreviewCalibrationRecovery()
        for index in range(5):
            frame = packet(index + 2, 10.6 + index * .04)
            with self.assertRaisesRegex(ControlFault, "calibration jump"):
                tracker.update(frame, frame.captured_at + .01)
            ready = recovery.offer(frame, frame.captured_at + .01, tracker)
        self.assertTrue(ready.ready)
        self.assertEqual(tracker.last_capture, 10.)
        self.assertIn("0.1190m", tracker.last_rejection)

    def test_anchor_prevents_small_steps_accumulating_into_a_large_drift(self):
        tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
        for index in range(4):
            at = 10.6 + index * .04
            self.assertIsNone(recovery.offer(packet(index + 2, at, dx=.119 + index * .006),
                                             at + .01, tracker))
        self.assertEqual(recovery.count, 1)
        self.assertIn("calibration jump", recovery.reason)
        self.assertEqual(recovery.samples[0][0].sequence, 5)

    def test_previous_frame_bounds_apply_even_inside_anchor_bounds(self):
        tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
        for index, dx in enumerate((.119, .129, .109)):
            at = 10.6 + index * .04
            recovery.offer(packet(index + 2, at, dx=dx), at + .01, tracker)
        self.assertEqual(recovery.count, 1)
        self.assertIn("0.0200m", recovery.reason)

    def test_relative_translation_and_rotation_bounds_also_apply(self):
        for rotation in (False, True):
            with self.subTest(rotation=rotation):
                tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
                first = packet(2, 10.6)
                recovery.offer(first, 10.61, tracker)
                second = packet(3, 10.64, dx=.129, other_dx=-.01)
                if rotation:
                    poses = deepcopy(first.poses)
                    for name, angle in zip(ROBOTS, (1.2, -1.2)):
                        poses[name] = poses[name] @ pose_matrix([0., 0., 0., 0., 0., np.deg2rad(angle)])
                    second = replace(second, poses=poses)
                self.assertIsNone(recovery.offer(second, 10.65, tracker))
                self.assertEqual(recovery.count, 1)
                self.assertIn("relative base calibration jump", recovery.reason)

    def test_quality_errors_reset_and_the_same_sequence_cannot_be_repaired(self):
        bad_fields = (
            {"reason": "PnP failed"},
            {"valid_counts": {"ur5e": 3, "ur10e": 7}},
            {"raw_mean_px": {"ur5e": np.nan, "ur10e": 2.}},
            {"raw_max_px": {"ur5e": 20., "ur10e": 4.}},
            {"poses": {"ur5e": np.eye(4)}},
        )
        for fields in bad_fields:
            with self.subTest(fields=fields):
                tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
                recovery.offer(packet(2, 10.6), 10.61, tracker)
                self.assertIsNone(recovery.offer(packet(3, 10.64, **fields), 10.65, tracker))
                self.assertEqual(recovery.count, 0)
                self.assertTrue(recovery.reason)
                recovery.offer(packet(3, 10.64), 10.66, tracker)
                self.assertEqual(recovery.count, 0)
                recovery.offer(packet(4, 10.68), 10.69, tracker)
                self.assertEqual(recovery.count, 1)

    def test_reprojection_disabled_still_requires_well_formed_quality_data(self):
        tracker, recovery = self.baseline(check_reprojection=False), PreviewCalibrationRecovery()
        for index in range(5):
            at = 10.6 + index * .04
            frame = packet(index + 2, at, raw_mean_px={name: 30. for name in ROBOTS},
                           raw_max_px={name: 50. for name in ROBOTS})
            ready = recovery.offer(frame, at + .01, tracker)
        self.assertTrue(ready.ready)
        self.assertFalse(ready.check_reprojection)
        self.assertIsNone(recovery.offer(packet(7, 10.8, raw_max_px={name: 1. for name in ROBOTS}),
                                         10.81, tracker))
        self.assertEqual(recovery.count, 0)

    def test_duplicates_neither_count_nor_refresh_age_and_stale_duplicate_resets(self):
        tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
        first = packet(2, 10.6)
        recovery.offer(first, 10.61, tracker)
        for at in (10.65, 10.8, 11.):
            self.assertIsNone(recovery.offer(replace(first, captured_at=at), at, tracker))
            self.assertEqual(recovery.count, 1)
            self.assertEqual(recovery.samples[0][0].captured_at, 10.6)
            self.assertEqual(recovery.samples[0][1], 10.61)
        self.assertIsNone(recovery.offer(first, 11.11, tracker))
        self.assertEqual(recovery.count, 0)
        self.assertIn("stale", recovery.reason)
        recovery.offer(replace(first, captured_at=11.12), 11.13, tracker)
        self.assertEqual(recovery.count, 0)

    def test_capture_gap_discards_previous_candidates_and_seeds_a_new_cluster(self):
        tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
        recovery.offer(packet(2, 10.6), 10.61, tracker)
        recovery.offer(packet(3, 10.64), 10.65, tracker)
        self.assertIsNone(recovery.offer(packet(4, 11.2), 11.21, tracker))
        self.assertEqual(recovery.count, 1)
        self.assertEqual(recovery.samples[0][0].sequence, 4)
        self.assertIn("capture gap", recovery.reason)

    def test_sequence_capture_and_receive_time_failures_are_reported_without_raising(self):
        cases = (
            (packet(1, 10.64), 10.65, "sequence"),
            (packet(3, 10.6), 10.65, "capture time did not increase"),
            (packet(3, 10.7), 10.65, "future"),
            (packet(3, 10.64), 11.2, "stale"),
            (packet(3, 10.59), 10.60, "time moved backwards"),
            (packet(3, 10.64), np.nan, "finite"),
            (packet(3, 10.64), -1., "nonnegative"),
            (replace(packet(3, 10.64), sequence=True), 10.65, "sequence"),
            (object(), 10.65, "CalibrationFrame"),
        )
        for frame, now, reason in cases:
            with self.subTest(reason=reason):
                tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
                recovery.offer(packet(2, 10.6), 10.61, tracker)
                self.assertIsNone(recovery.offer(frame, now, tracker))
                self.assertEqual(recovery.count, 0)
                self.assertIn(reason, recovery.reason)

    def test_samples_input_and_returned_tracker_are_defensive_copies(self):
        tracker, recovery = self.baseline(stable_frames=1), PreviewCalibrationRecovery()
        first = packet(2, 10.6)
        ready = recovery.offer(first, 10.61, tracker)
        first.poses["ur5e"][0, 3] = 999.
        samples = recovery.samples
        self.assertAlmostEqual(samples[0][0].poses["ur5e"][0, 3], -.081)
        samples[0][0].poses["ur5e"][0, 3] = 888.
        samples.clear()
        ready._raw_poses["ur5e"][0, 3] = 777.
        ready._poses["ur5e"][0, 3] = 666.
        self.assertEqual(recovery.count, 1)
        self.assertAlmostEqual(recovery.samples[0][0].poses["ur5e"][0, 3], -.081)
        ready_next = recovery.offer(packet(3, 10.64), 10.65, tracker)
        self.assertAlmostEqual(ready_next.poses["ur5e"][0, 3], -.081)
        recovery.reset()
        self.assertEqual(recovery.count, 0)
        self.assertEqual(recovery.samples, [])
        self.assertEqual(recovery.reason, "")

    def test_receive_clock_error_also_consumes_the_observed_sequence(self):
        tracker, recovery = self.baseline(), PreviewCalibrationRecovery()
        recovery.offer(packet(2, 10.6), 10.61, tracker)
        frame = packet(3, 10.64)
        self.assertIsNone(recovery.offer(frame, np.nan, tracker))
        self.assertEqual(recovery.count, 0)
        self.assertIsNone(recovery.offer(frame, 10.65, tracker))
        self.assertEqual(recovery.count, 0)
        recovery.offer(packet(4, 10.68), 10.69, tracker)
        self.assertEqual(recovery.count, 1)

    def test_all_configuration_limits_and_custom_stability_count_are_preserved(self):
        options = dict(max_age=.25, stable_frames=3, alpha=.7, max_jump_m=.02,
                       max_jump_rad=.07, max_mean_px=6., max_pixel_px=9.,
                       min_points=6, check_reprojection=False)
        tracker, recovery = self.baseline(**options), PreviewCalibrationRecovery()
        for index in range(3):
            at = 10.3 + index * .04
            ready = recovery.offer(packet(index + 2, at), at + .01, tracker)
            if index < 2:
                self.assertIsNone(ready)
        self.assertTrue(ready.ready)
        for name, value in options.items():
            self.assertEqual(getattr(ready, name), value)

    def test_stale_error_retains_the_reason_new_frames_were_rejected(self):
        tracker = self.baseline()
        with self.assertRaises(ControlFault):
            tracker.update(packet(2, 10.1), 10.11)
        reason = tracker.last_rejection
        with self.assertRaisesRegex(ControlFault, r"stale.*0\.600s.*last rejected pair.*0\.1190m"):
            tracker.check_fresh(10.6)
        self.assertEqual(tracker.last_rejection, reason)
        self.assertEqual(tracker.last_capture, 10.)


if __name__ == "__main__":
    unittest.main()
