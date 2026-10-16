"""Synthetic time alignment and SE(3) motion, without inference or hardware."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))

from urcal.control.geometry import inverse, pose_matrix
from urcal.cooperative.state import RelativeStateError, RelativeStateEstimator


def pose(x=0., y=0., z=0., rz=0.):
    return pose_matrix([x, y, z, 0., 0., rz])


def pair(t=0., moving="ur5e"):
    poses = {"ur5e": pose(.4, .1, 1.), "ur10e": pose(-.3, -.2, 1.)}
    if moving is not None:
        poses[moving] = pose_matrix([*(poses[moving][:3, 3] + [.04 * t, -.02 * t, 0.]), 0., 0., .1 * t])
    return poses


class RelativeStateTests(unittest.TestCase):
    def estimator(self, **kwargs):
        return RelativeStateEstimator(max_measurement_age=2., max_prediction_age=.5, **kwargs)

    def test_same_capture_pair_ignores_different_processing_completion_times(self):
        estimator = self.estimator()
        sample = pair()
        estimator.update("ur5e", sample["ur5e"], 10., 10.12, 1)
        estimator.update("ur10e", sample["ur10e"], 10., 10.23, 1)
        result = estimator.estimate(10.3)
        self.assertEqual(result.at, 10.)
        self.assertEqual(result.measurement_times, {"ur5e": 10., "ur10e": 10.})
        self.assertEqual(result.received_times, {"ur5e": 10.12, "ur10e": 10.23})
        self.assertEqual(result.prediction_dt, {"ur5e": 0., "ur10e": 0.})
        np.testing.assert_allclose(result.relative_pose, inverse(sample["ur10e"]) @ sample["ur5e"], atol=1e-12)

    def test_pair_accepts_per_robot_receipts_and_copies_inputs_outputs(self):
        estimator = self.estimator()
        sample = pair()
        expected = sample["ur5e"].copy()
        estimator.update_pair(sample, 10., {"ur5e": 10.1, "ur10e": 10.2}, 1)
        sample["ur5e"][0, 3] = 100.
        first = estimator.estimate(10.3)
        np.testing.assert_array_equal(first.base_poses["ur5e"], expected)
        first.base_poses["ur5e"][1, 3] = 100.
        first.base_twists["ur5e"][:] = 100.
        second = estimator.estimate(10.3)
        np.testing.assert_array_equal(second.base_poses["ur5e"], expected)
        np.testing.assert_array_equal(second.base_twists["ur5e"], np.zeros(6))

    def test_direct_async_alignment_interpolates_at_common_historical_time(self):
        estimator = self.estimator()
        estimator.update("ur5e", pose(0., 0., 1.), 10., 10.01, 1)
        estimator.update("ur5e", pose(.02, 0., 1.), 10.2, 10.21, 2)
        estimator.update("ur10e", pose(-.4, 0., 1.), 10.1, 10.11, 1)
        result = estimator.estimate(10.25)
        self.assertEqual(result.at, 10.1)
        np.testing.assert_allclose(result.base_poses["ur5e"][:3, 3], [.01, 0, 1], atol=1e-12)
        np.testing.assert_allclose(result.relative_pose[:3, 3], [.41, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result.base_twists["ur5e"], [.1, 0, 0, 0, 0, 0], atol=1e-12)

    def test_direct_async_without_a_bracket_is_rejected(self):
        estimator = self.estimator()
        estimator.update("ur5e", pose(), 10.2, 10.21, 1)
        estimator.update("ur10e", pose(.4), 10.1, 10.11, 1)
        with self.assertRaisesRegex(RelativeStateError, "bracket"):
            estimator.estimate(10.3)

    def test_constant_velocity_prediction_removes_capture_to_control_delay(self):
        for moving in ("ur5e", "ur10e"):
            estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
            for sequence, t in enumerate([0., .1, .2]):
                estimator.update_pair(pair(t, moving), 10. + t, 10. + t + .08, sequence)
            result = estimator.estimate(10.3)
            # Translation follows the SE(3) screw inferred from two observations;
            # the small nonzero rotation makes independent linear XYZ prediction
            # differ only at second order, while orientation remains exact.
            expected = pair(.3, moving)
            np.testing.assert_allclose(result.base_poses[moving][:3, 3], expected[moving][:3, 3], atol=5e-5)
            np.testing.assert_allclose(result.base_poses[moving][:3, :3], expected[moving][:3, :3], atol=1e-12)
            self.assertAlmostEqual(result.prediction_dt[moving], .1)
            self.assertAlmostEqual(result.measurement_age[moving], .1)

    def test_async_prediction_extrapolates_each_robot_to_same_now(self):
        estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
        for name, initial, speed, samples in (("ur5e", .4, .03, [0., .1]),
                                              ("ur10e", -.4, -.02, [.05, .15])):
            for sequence, t in enumerate(samples):
                estimator.update(name, pose(initial + speed * t, 0, 1), 10 + t, 10 + t + .02, sequence)
        result = estimator.estimate(10.2)
        self.assertEqual(result.at, 10.2)
        self.assertAlmostEqual(result.relative_pose[0, 3], .81)
        np.testing.assert_allclose(result.relative_twist, [.05, 0, 0, 0, 0, 0], atol=1e-12)

    def test_common_rigid_motion_preserves_relative_pose_and_zero_relative_velocity(self):
        relative = pose_matrix([.45, -.2, .08, .1, -.2, .3])
        initial = pose(.2, .1, .7)
        for mode in ("direct", "predict"):
            estimator = self.estimator(mode=mode)
            for sequence in range(30):
                t = sequence * .04
                common = pose(0., 0., .01 * t, .12 * t)
                leader = common @ initial
                estimator.update_pair({"ur10e": leader, "ur5e": leader @ relative}, 10 + t, 10 + t + .02, sequence)
                result = estimator.estimate(10 + t + .03)
                np.testing.assert_allclose(result.relative_pose, relative, atol=3e-12)
                np.testing.assert_allclose(result.relative_twist, 0., atol=2e-12)

    def test_relative_velocity_includes_transport_from_rotating_leader_frame(self):
        estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
        for sequence, t in enumerate([0., .1]):
            estimator.update_pair({"ur10e": pose(0, 0, 1, .2 * t), "ur5e": pose(.5, 0, 1)},
                                  10 + t, 10 + t, sequence)
        result = estimator.estimate(10.1)
        R = result.base_poses["ur10e"][:3, :3]
        expected_linear = R.T @ [0, -.1, 0]
        np.testing.assert_allclose(result.relative_twist[:3], expected_linear, atol=1e-12)
        np.testing.assert_allclose(result.relative_twist[3:], [0, 0, -.2], atol=1e-12)

    def test_follower_rotation_and_both_independent_base_velocities(self):
        estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
        for sequence, t in enumerate([0., .1]):
            estimator.update_pair({"ur10e": pose(-.2 + .02 * t, 0, 1),
                                   "ur5e": pose(.2 - .03 * t, 0, 1, .15 * t)},
                                  10 + t, 10 + t, sequence)
        result = estimator.estimate(10.1)
        np.testing.assert_allclose(result.relative_twist[:3], [-.05, -.000225, 0], atol=1e-6)
        np.testing.assert_allclose(result.relative_twist[3:], [0, 0, .15], atol=1e-12)

    def test_camera_coordinate_change_preserves_relative_predictions(self):
        change = pose_matrix([.4, -.2, .3, .2, -.1, .7])
        estimators = [self.estimator(mode="predict"), self.estimator(mode="predict")]
        for sequence in range(12):
            t = sequence * .05
            sample = pair(t)
            for estimator, prefix in zip(estimators, (np.eye(4), change)):
                estimator.update_pair({name: prefix @ value for name, value in sample.items()}, 10 + t, 10 + t, sequence)
        outputs = [estimator.estimate(10.65) for estimator in estimators]
        np.testing.assert_allclose(outputs[0].relative_pose, outputs[1].relative_pose, atol=1e-12)
        np.testing.assert_allclose(outputs[0].relative_twist, outputs[1].relative_twist, atol=1e-12)

    def test_prediction_expires_without_refreshed_measurement(self):
        estimator = RelativeStateEstimator(mode="predict", max_prediction_age=.1, max_measurement_age=.5)
        estimator.update_pair(pair(), 10., 10.05, 1)
        estimator.estimate(10.09)
        with self.assertRaisesRegex(RelativeStateError, "prediction horizon"):
            estimator.estimate(10.11)
        direct = RelativeStateEstimator(max_measurement_age=.1)
        direct.update_pair(pair(), 10., 10.01, 1)
        with self.assertRaisesRegex(RelativeStateError, "expired"):
            direct.estimate(10.11)

    def test_fresh_measurement_after_long_gap_does_not_project_old_velocity(self):
        estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
        for sequence, captured, x in ((1, 10., 0.), (2, 10.1, .01), (3, 11.2, .012)):
            estimator.update_pair({"ur5e": pose(x, 0, 1), "ur10e": pose(-.4, 0, 1)},
                                  captured, captured, sequence)
        result = estimator.estimate(11.25)
        self.assertAlmostEqual(result.base_poses["ur5e"][0, 3], .012)
        np.testing.assert_array_equal(result.base_twists["ur5e"], np.zeros(6))

    def test_duplicate_does_not_refresh_capture_receipt_or_velocity(self):
        estimator = self.estimator(mode="predict")
        sample = pair()
        estimator.update_pair(sample, 10., 10.01, 1)
        self.assertFalse(estimator.update_pair(sample, 10., 10.3, 1))
        result = estimator.estimate(10.31)
        self.assertEqual(result.received_times, {"ur5e": 10.01, "ur10e": 10.01})
        self.assertAlmostEqual(result.measurement_age["ur5e"], .31)
        np.testing.assert_array_equal(result.relative_twist, np.zeros(6))
        changed = pair()
        changed["ur5e"][0, 3] += .001
        with self.assertRaisesRegex(RelativeStateError, "duplicate"):
            estimator.update_pair(changed, 10., 10.31, 1)

    def test_raw_speed_limits_reject_and_pair_commit_is_atomic(self):
        estimator = self.estimator()
        initial = pair(moving=None)
        estimator.update_pair(initial, 10., 10., 1)
        for moved in (pose(2., 0., 1.), pose(-.3, -.2, 1., .2)):
            sample = {"ur5e": initial["ur5e"].copy(), "ur10e": moved}
            sample["ur5e"][0, 3] += .001
            with self.assertRaisesRegex(RelativeStateError, "speed"):
                estimator.update_pair(sample, 10.1, 10.1, 2)
            result = estimator.estimate(10.1)
            self.assertEqual(result.measurement_times, {"ur5e": 10., "ur10e": 10.})
            np.testing.assert_array_equal(result.base_poses["ur5e"], initial["ur5e"])

    def test_rotation_wrap_uses_short_arc(self):
        estimator = self.estimator(mode="predict", alpha=1., velocity_alpha=1.)
        for sequence, angle in enumerate([179., -179.]):
            estimator.update_pair({"ur5e": pose(0, 0, 1, np.deg2rad(angle)), "ur10e": pose(-.3, 0, 1)},
                                  10 + sequence * .1, 10 + sequence * .1, sequence)
        result = estimator.estimate(10.2)
        np.testing.assert_allclose(result.base_poses["ur5e"][:3, :3], Rotation.from_euler("z", -177, degrees=True).as_matrix(), atol=1e-12)
        self.assertAlmostEqual(result.base_twists["ur5e"][5], np.deg2rad(20.))

    def test_stationary_inputs_produce_no_filter_jitter_or_velocity(self):
        estimator = self.estimator(mode="predict")
        sample = {"ur5e": pose_matrix([.2, .1, .8, .2, -.3, .4]), "ur10e": pose_matrix([-.3, .2, 1., -.1, .2, -.4])}
        expected = inverse(sample["ur10e"]) @ sample["ur5e"]
        for sequence in range(50):
            t = 10 + sequence * .03
            estimator.update_pair(sample, t, t + .01, sequence)
            output = estimator.estimate(t + .02)
            np.testing.assert_allclose(output.relative_pose, expected, atol=1e-12)
            np.testing.assert_allclose(output.relative_twist, 0., atol=1e-12)

    def test_filter_reduces_stationary_small_measurement_noise(self):
        estimator = self.estimator(mode="predict")
        rng = np.random.default_rng(85)
        raw_errors, filtered_errors = [], []
        for sequence in range(120):
            noise = rng.normal(0, .0003)
            sample = {"ur5e": pose(.4 + noise, .1, 1.), "ur10e": pose(-.3, -.2, 1.)}
            t = 10 + sequence * .05
            estimator.update_pair(sample, t, t, sequence)
            result = estimator.estimate(t)
            if sequence > 20:
                raw_errors.append(noise)
                filtered_errors.append(result.base_poses["ur5e"][0, 3] - .4)
        self.assertLess(np.std(filtered_errors), np.std(raw_errors))

    def test_malformed_future_delayed_and_out_of_order_data_are_rejected(self):
        estimator = RelativeStateEstimator()
        sample = pair()
        estimator.update_pair(sample, 10., 10.01, 1)
        for captured, received, sequence in ((10.2, 10.1, 2), (10.1, 10.7, 2),
                                               (9.9, 10.1, 2), (10.1, 10.2, 0),
                                               (10.1, np.nan, 2), (10.1, 10.2, True)):
            with self.subTest(captured=captured, received=received, sequence=sequence), self.assertRaises(RelativeStateError):
                estimator.update_pair(sample, captured, received, sequence)
        for invalid in (np.zeros((4, 4)), np.diag([-1., 1., 1., 1.]), np.eye(3), np.full((4, 4), np.nan)):
            with self.subTest(pose=invalid), self.assertRaises(RelativeStateError):
                estimator.update("ur5e", invalid, 10.1, 10.1, 2)
        with self.assertRaises(RelativeStateError):
            estimator.estimate(10.)
        with self.assertRaises(RelativeStateError):
            estimator.estimate(10.1, leader="bad")
        with self.assertRaises(RelativeStateError):
            estimator.update_pair({"ur5e": sample["ur5e"]}, 10.1, 10.1, 2)

    def test_configuration_missing_state_and_receipt_regression_are_rejected(self):
        for kwargs in ({"robots": None}, {"robots": [[], []]}, {"robots": ["one", "one"]},
                       {"mode": "bad"}, {"alpha": 0}, {"velocity_alpha": 1.1},
                       {"max_prediction_age": np.nan}, {"max_linear_speed": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RelativeStateError):
                RelativeStateEstimator(**kwargs)
        estimator = self.estimator()
        with self.assertRaisesRegex(RelativeStateError, "both robot"):
            estimator.estimate(10.)
        sample = pair()
        estimator.update_pair(sample, 10., 10.2, 1)
        with self.assertRaisesRegex(RelativeStateError, "receipt time"):
            estimator.update_pair(sample, 10., 10.1, 1)


if __name__ == "__main__":
    unittest.main()
