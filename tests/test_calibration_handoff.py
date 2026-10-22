"""Post-alignment acquisition: real gates/estimator, synthetic data, no robot IO."""

from copy import deepcopy
import unittest

import numpy as np

from urcal.control.handoff import CalibrationHandoff
from urcal.control.policy import CalibrationTracker, ControlFault, ROBOTS
from urcal.cooperative.state import RelativeStateEstimator
from tests.test_active_calibration import packet


class CalibrationHandoffTests(unittest.TestCase):
    def build(self, *, estimate=False):
        template = CalibrationTracker(alpha=1., check_reprojection=False)
        for index in range(5):
            template.update(packet(index, 1. + .04 * index), 1. + .04 * index)
        extra = ({"estimator_factory": lambda: RelativeStateEstimator(mode="predict"),
                  "mappings": {name: np.eye(4) for name in ROBOTS}} if estimate else {})
        return CalibrationHandoff(template, 10., timeout=5., **extra), template

    def offer(self, handoff, sequence, at, delta=0., reason=""):
        return handoff.ingest(packet(sequence, at, delta, reason), at + .01)

    def test_does_not_reuse_old_capture_or_modify_template(self):
        handoff, template = self.build()
        original = deepcopy(template.poses)
        for index, at in enumerate((1.16, 9.99, 10.)):
            self.assertFalse(handoff.ingest(packet(index + 5, at), 10.1))
        self.assertFalse(handoff.ready(10.1))
        self.assertEqual(handoff.frames, [])
        self.assertEqual(template.stable_count, 5)
        for name in ROBOTS:
            np.testing.assert_array_equal(template.poses[name], original[name])

    def test_requires_five_distinct_fresh_post_endpoint_pairs(self):
        handoff, _ = self.build()
        for index in range(5):
            at = 10.04 + .04 * index
            self.assertTrue(self.offer(handoff, index + 5, at))
            self.assertFalse(self.offer(handoff, index + 5, at))
            self.assertEqual(handoff.ready(at + .01), index == 4)
        self.assertEqual(len(handoff.frames), 5)
        self.assertAlmostEqual(handoff.tracker.last_capture, 10.20)

    def test_rejected_frame_discards_cluster_and_is_not_new_baseline(self):
        handoff, _ = self.build()
        for index in range(4):
            self.offer(handoff, index, 10.04 + .04 * index)
        self.assertFalse(self.offer(handoff, 4, 10.20, delta=.119))
        self.assertFalse(handoff.ready(10.21))
        self.assertIn("calibration jump", handoff.reason)
        self.assertEqual(handoff.frames, [])
        self.assertEqual(handoff.tracker.poses, {})
        for index in range(5):
            self.offer(handoff, 5 + index, 10.24 + .04 * index)
            self.assertEqual(handoff.ready(10.25 + .04 * index), index == 4)

    def test_stale_cluster_cannot_resume_half_warmed(self):
        handoff, _ = self.build()
        for index in range(4):
            self.offer(handoff, index, 10.04 + .04 * index)
        self.assertFalse(handoff.ready(10.8))
        self.assertEqual(handoff.frames, [])
        self.offer(handoff, 4, 10.84)
        self.assertEqual(handoff.tracker.stable_count, 1)
        self.assertFalse(handoff.ready(10.85))

    def test_long_capture_gap_starts_new_cluster_even_without_ready_calls(self):
        handoff, _ = self.build()
        self.offer(handoff, 0, 10.04)
        self.offer(handoff, 1, 10.08)
        self.offer(handoff, 2, 10.84)
        self.assertEqual(len(handoff.frames), 1)
        self.assertEqual(handoff.tracker.stable_count, 1)

    def test_future_and_failed_pnp_frames_do_not_poison_next_cluster(self):
        handoff, _ = self.build()
        self.assertFalse(handoff.ingest(packet(0, 12.), 10.1))
        self.assertTrue(self.offer(handoff, 1, 10.2))
        self.assertFalse(self.offer(handoff, 2, 10.24, reason="PnP failed"))
        self.assertEqual(handoff.frames, [])
        self.assertTrue(self.offer(handoff, 3, 10.28))
        self.assertEqual(handoff.tracker.stable_count, 1)

    def test_replayed_or_reversed_sequence_cannot_count_after_rejection(self):
        handoff, _ = self.build()
        self.offer(handoff, 10, 10.04)
        self.offer(handoff, 11, 10.08, reason="PnP failed")
        self.assertFalse(self.offer(handoff, 11, 10.12))
        self.assertFalse(self.offer(handoff, 10, 10.16))
        self.assertEqual(handoff.frames, [])
        self.assertTrue(self.offer(handoff, 12, 10.20))

    def test_live_estimator_uses_only_new_cluster_no_old_to_new_velocity(self):
        handoff, _ = self.build(estimate=True)
        for index in range(5):
            # Helper intentionally has no old coordinate baseline. Runtime
            # independently checks this against the frozen Arm bases.
            self.assertTrue(self.offer(handoff, index + 5, 10.04 + .04 * index, delta=.25))
        snapshot = handoff.estimator.estimate(10.22)
        self.assertTrue(handoff.ready(10.22))
        for twist in snapshot.base_twists.values():
            np.testing.assert_allclose(twist, 0., atol=1e-12)

    def test_intra_cluster_velocity_limit_discards_both_gate_and_estimator(self):
        handoff, _ = self.build(estimate=True)
        self.assertTrue(self.offer(handoff, 5, 10.04))
        # 10 mm is inside jump tolerance; 0.25 m/s is outside base speed.
        self.assertFalse(self.offer(handoff, 6, 10.08, delta=.01))
        self.assertIn("base linear speed", handoff.reason)
        self.assertEqual(handoff.frames, [])
        self.assertFalse(handoff.ready(10.09))
        for index in range(5):
            self.assertTrue(self.offer(handoff, index + 7, 10.12 + .04 * index))
        self.assertTrue(handoff.ready(10.29))

    def test_total_deadline_is_not_extended_by_rejections_or_fresh_frames(self):
        handoff, _ = self.build()
        for index in range(4):
            self.offer(handoff, index, 10.5 + index, reason="PnP failed")
        with self.assertRaisesRegex(ControlFault, "wait exceeded 5s"):
            self.offer(handoff, 5, 15.01)
        with self.assertRaisesRegex(ControlFault, "new Arm"):
            handoff.ready(15.01)

    def test_reprojection_off_setting_is_preserved(self):
        handoff, _ = self.build()
        sample = packet(0, 10.04)
        for name in ROBOTS:
            sample.raw_mean_px[name], sample.raw_max_px[name] = 20., 50.
        self.assertTrue(handoff.ingest(sample, 10.05))
        self.assertFalse(handoff.tracker.check_reprojection)


if __name__ == "__main__":
    unittest.main()
