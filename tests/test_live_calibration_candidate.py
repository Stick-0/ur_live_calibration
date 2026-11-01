"""Fresh live re-acquisition evidence without motion or fabricated timestamps."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.control.geometry import pose_matrix
from urcal.control.live_calibration import LiveCalibrationCandidate
from urcal.control.policy import CalibrationFrame, CalibrationTracker, ROBOTS


def packet(sequence, captured, shift=.0, *, other_shift=0., reason=""):
    return CalibrationFrame(
        sequence, captured,
        {"ur5e": pose_matrix([shift, 0., .8, 0., 0., 0.]),
         "ur10e": pose_matrix([1. + other_shift, 0., .8, 0., 0., 0.])},
        {name: 2. for name in ROBOTS}, {name: 3. for name in ROBOTS},
        {name: 7 for name in ROBOTS}, reason)


def baseline(stable_frames=3, check_reprojection=False):
    tracker = CalibrationTracker(max_age=.5, stable_frames=stable_frames,
                                 max_jump_m=.05, max_jump_rad=np.deg2rad(5.),
                                 check_reprojection=check_reprojection)
    tracker.update(packet(0, 10.), 10.12)
    tracker.stable_count = stable_frames
    return tracker


class LiveCalibrationCandidateTests(unittest.TestCase):
    def test_new_pose_is_collected_immediately_and_ready_with_true_capture_span(self):
        tracker, candidate = baseline(), LiveCalibrationCandidate()
        original = deepcopy(tracker.__dict__)
        self.assertIsNone(candidate.offer(packet(1, 10.1, .08), 10.22, tracker))
        self.assertEqual(candidate.count, 1)
        self.assertIsNone(candidate.offer(packet(2, 10.18, .082), 10.30, tracker))
        result = candidate.offer(packet(3, 10.26, .081), 10.38, tracker)
        self.assertIsNotNone(result)
        self.assertEqual(result.stable_count, 3)
        self.assertEqual(result.last_capture, 10.26)
        self.assertEqual(result._last_sequence, 3)
        self.assertAlmostEqual(result.poses["ur5e"][0, 3], .081)
        np.testing.assert_array_equal(result._raw_poses["ur5e"], packet(3, 10.26, .081).poses["ur5e"])
        result.check_fresh(10.38)
        self.assertEqual(tracker.last_capture, original["last_capture"])
        self.assertEqual(tracker._last_sequence, original["_last_sequence"])
        np.testing.assert_array_equal(tracker.poses["ur5e"], original["_poses"]["ur5e"])

    def test_count_and_span_are_independent_and_use_capture_not_receipt(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        for seq, capture, receipt in ((1, 10.1, 10.2), (2, 10.11, 10.35), (3, 10.12, 10.5)):
            self.assertIsNone(candidate.offer(packet(seq, capture, .08), receipt, tracker))
        self.assertEqual(candidate.count, 3)
        self.assertIn("span 0.020/0.150s", candidate.reason)
        result = candidate.offer(packet(4, 10.3, .08), 10.51, tracker)
        self.assertIsNotNone(result)
        self.assertEqual(result.last_capture, 10.3)

    def test_explicit_startup_stability_count_is_preserved(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline(stable_frames=5)
        for index in range(5):
            capture = 10.1 + .08 * index
            result = candidate.offer(packet(index + 1, capture, .08), capture + .1, tracker)
            self.assertEqual(result is not None, index == 4)
        self.assertEqual(result.stable_frames, 5)
        self.assertEqual(result.stable_count, 5)

    def test_bad_spike_cannot_refresh_a_candidate_and_new_cluster_can_recover(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        candidate.offer(packet(1, 10.1, .08), 10.2, tracker)
        candidate.offer(packet(2, 10.18, .081), 10.28, tracker)
        self.assertIsNone(candidate.offer(packet(3, 10.26, .4), 10.36, tracker))
        self.assertEqual(candidate.count, 0)
        self.assertIn("replacement shift", candidate.reason)
        for index in range(3):
            capture = 10.34 + .08 * index
            result = candidate.offer(packet(4 + index, capture, .08), capture + .1, tracker)
        self.assertIsNotNone(result)

    def test_oscillating_extremes_and_chained_drift_never_accumulate_readiness(self):
        for shifts in ((.09, -.09) * 5, tuple(.03 * index for index in range(1, 11))):
            candidate, tracker = LiveCalibrationCandidate(), baseline()
            for index, shift in enumerate(shifts):
                capture = 10.1 + .08 * index
                self.assertIsNone(candidate.offer(packet(index + 1, capture, shift), capture + .1, tracker))
        # Both extremes fit the first sample, but disagree with one another.
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        for index, shift in enumerate((.08, .129, .031)):
            capture = 10.1 + .08 * index
            self.assertIsNone(candidate.offer(packet(index + 1, capture, shift), capture + .1, tracker))
        self.assertEqual(candidate.count, 1)

    def test_duplicates_never_add_evidence_or_refresh_freshness(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        frame = packet(1, 10.1, .08)
        candidate.offer(frame, 10.2, tracker)
        self.assertIsNone(candidate.offer(frame, 10.3, tracker))
        self.assertEqual(candidate.count, 1)
        self.assertEqual(candidate.samples[-1][1], 10.2)
        self.assertIsNone(candidate.offer(frame, 10.7, tracker))
        self.assertEqual(candidate.count, 0)
        self.assertIn("stale", candidate.reason)

    def test_repaired_failed_sequence_or_capture_cannot_be_counted_again(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        self.assertIsNone(candidate.offer(packet(1, 10.1, .08, reason="PnP failed"), 10.2, tracker))
        self.assertIsNone(candidate.offer(packet(1, 10.1, .08), 10.21, tracker))
        self.assertEqual(candidate.count, 0)
        self.assertIsNone(candidate.offer(packet(2, 10.1, .08), 10.22, tracker))
        self.assertIn("capture time did not increase", candidate.reason)

    def test_stale_future_backward_and_malformed_inputs_cannot_seed_cluster(self):
        broken = [packet(1, 9.5, .08), packet(1, 10.4, .08), packet(-1, 10.1, .08), None]
        nan_pose = packet(1, 10.1, .08)
        nan_pose.poses["ur5e"][0, 0] = np.nan
        broken.append(nan_pose)
        missing = packet(1, 10.1, .08)
        del missing.poses["ur10e"]
        broken.append(missing)
        too_few = packet(1, 10.1, .08)
        too_few.valid_counts["ur5e"] = 3
        broken.append(too_few)
        invalid_statistics = packet(1, 10.1, .08)
        invalid_statistics.raw_mean_px["ur5e"] = np.nan
        broken.append(invalid_statistics)
        for frame in broken:
            with self.subTest(frame=frame):
                candidate, tracker = LiveCalibrationCandidate(), baseline()
                self.assertIsNone(candidate.offer(frame, 10.2, tracker))
                self.assertEqual(candidate.count, 0)
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        candidate.offer(packet(2, 10.2, .08), 10.3, tracker)
        self.assertIsNone(candidate.offer(packet(1, 10.3, .08), 10.4, tracker))
        self.assertEqual(candidate.count, 0)
        self.assertIsNone(candidate.offer(packet(3, 10.4, .08), 10.25, tracker))
        self.assertIn("time moved backwards", candidate.reason)

    def test_reprojection_configuration_is_honored_and_missing_baseline_is_rejected(self):
        frame = packet(1, 10.1, .08)
        frame.raw_mean_px["ur5e"] = 10.
        frame.raw_max_px["ur5e"] = 12.
        candidate = LiveCalibrationCandidate()
        self.assertIsNone(candidate.offer(frame, 10.2, baseline(check_reprojection=True)))
        self.assertEqual(candidate.count, 0)
        self.assertIn("reprojection", candidate.reason)
        candidate.offer(frame, 10.3, CalibrationTracker())
        self.assertEqual(candidate.count, 0)
        self.assertIn("accepted calibration baseline", candidate.reason)

    def test_replacement_bound_applies_to_relative_pose_as_well_as_each_arm(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        # Both arm shifts are only 10 cm, but their relative change is 20 cm.
        result = candidate.offer(packet(1, 10.1, .1, other_shift=-.1), 10.2, tracker)
        self.assertIsNone(result)
        self.assertEqual(candidate.count, 0)
        self.assertIn("relative base replacement shift", candidate.reason)
        frame = packet(2, 10.2, .08)
        frame.poses["ur5e"][:3, :3] = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(16)])[:3, :3]
        self.assertIsNone(candidate.offer(frame, 10.3, tracker))
        self.assertEqual(candidate.count, 0)
        self.assertIn("replacement shift", candidate.reason)

    def test_cluster_bound_also_checks_relative_transform_and_angular_spread(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        candidate.offer(packet(1, 10.1, .07), 10.2, tracker)
        # Each arm moved 3 cm; their relative difference moved 6 cm.
        self.assertIsNone(candidate.offer(packet(2, 10.18, .1, other_shift=-.03), 10.28, tracker))
        self.assertEqual(candidate.count, 1)
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        candidate.offer(packet(1, 10.1, .08), 10.2, tracker)
        frame = packet(2, 10.18, .08)
        frame.poses["ur5e"][:3, :3] = pose_matrix([0, 0, 0, np.deg2rad(6), 0, 0])[:3, :3]
        self.assertIsNone(candidate.offer(frame, 10.28, tracker))
        self.assertEqual(candidate.count, 1)

    def test_every_supporting_member_must_still_be_fresh(self):
        candidate, tracker = LiveCalibrationCandidate(min_span=.2), baseline()
        candidate.offer(packet(1, 10.1, .08), 10.2, tracker)
        candidate.offer(packet(2, 10.3, .08), 10.4, tracker)
        result = candidate.offer(packet(3, 10.6, .08), 10.7, tracker)
        self.assertIsNone(result)
        self.assertEqual(candidate.count, 2)
        result = candidate.offer(packet(4, 10.7, .08), 10.8, tracker)
        self.assertIsNotNone(result)
        self.assertTrue(all(10.8 - frame.captured_at <= .5 for frame, _ in candidate.samples))

    def test_baseline_change_reset_and_external_mutation_do_not_mix_clusters(self):
        candidate, tracker = LiveCalibrationCandidate(), baseline()
        candidate.offer(packet(1, 10.1, .08), 10.2, tracker)
        samples = candidate.samples
        samples[0][0].poses["ur5e"][0, 3] = 99.
        self.assertAlmostEqual(candidate.samples[0][0].poses["ur5e"][0, 3], .08)
        changed = baseline()
        changed.update(packet(2, 10.18, .02), 10.28)
        candidate.offer(packet(3, 10.26, .08), 10.36, changed)
        self.assertEqual(candidate.count, 1)
        candidate.reset()
        self.assertEqual(candidate.count, 0)

    def test_invalid_candidate_configuration_is_rejected(self):
        for options in ({"min_frames": 2}, {"min_frames": True}, {"min_span": 0},
                        {"max_shift_m": np.nan}, {"max_shift_deg": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LiveCalibrationCandidate(**options)

    def test_recorded_jumps_and_stop_gap_do_not_manufacture_a_replacement(self):
        data = json.loads((Path(__file__).parent / "fixtures" / "live_calibration_rejections.json").read_text())
        options = dict(data["template_options"])
        options["max_jump_rad"] = np.deg2rad(options.pop("max_jump_deg"))
        tracker = CalibrationTracker(**options)

        def observed(row):
            return CalibrationFrame(
                row["sequence"], row["captured_at"],
                {name: np.asarray(value) for name, value in row["poses_camera_model"].items()},
                row["raw_mean_px"], row["raw_max_px"], row["valid_counts"])

        tracker.update(observed(data["baseline"]), data["baseline"]["received_at"])
        tracker.stable_count = tracker.stable_frames
        candidate = LiveCalibrationCandidate()
        first_receipt = data["samples"][0]["received_at"]
        outcomes = []
        for row in data["samples"]:
            result = candidate.offer(observed(row), row["received_at"], tracker)
            outcomes.append((row, result, candidate.count, candidate.reason))
        # The first two outliers exceed the relative-base replacement bound,
        # while the next two provide too little independent capture span.
        for _row, result, count, reason in outcomes[:2]:
            self.assertIsNone(result)
            self.assertEqual(count, 0)
            self.assertIn("relative base replacement shift", reason)
        self.assertEqual([entry[2] for entry in outcomes[2:4]], [1, 2])
        self.assertTrue(all(entry[1] is None for entry in outcomes[:-1]))
        # The old runtime stopped; the gap cannot count as observations or
        # extend old support. Only the later fresh cluster becomes eligible.
        self.assertGreater(outcomes[4][0]["received_at"] - outcomes[3][0]["received_at"], 1.)
        self.assertEqual(outcomes[4][2], 1)
        latest, result, count, _reason = outcomes[-1]
        self.assertIsNotNone(result)
        self.assertEqual(count, 3)
        self.assertEqual(result.last_capture, latest["captured_at"])
        self.assertEqual(result._last_sequence, latest["sequence"])
        self.assertAlmostEqual(latest["received_at"] - first_receipt, 2.2461110139265656)
        self.assertGreater(candidate.samples[-1][0].captured_at - candidate.samples[0][0].captured_at, .15)


if __name__ == "__main__":
    unittest.main()
