"""Independent synthetic paired-base truth; no images, weights, or robot IO."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import calibration_output_mapping, inverse, model_from_controller_base, pose_matrix
from urcal.cooperative.relative_kalman import (PlanarKalmanConfig, PlanarRelativeStateEstimator,
                                               load_planar_kalman_config)
from urcal.cooperative.state import RelativeStateError


ROBOTS = ("ur5e", "ur10e")


def mappings(corrected=True):
    return {name: calibration_output_mapping(name, ur10e_output_correction=corrected) for name in ROBOTS}


def pair(x=.9, y=.2, yaw=.4, *, camera=None, selected=None, leader="ur10e"):
    selected = mappings() if selected is None else selected
    camera = pose_matrix([.2, -.3, 1.7, .3, -.2, .1]) if camera is None else camera
    follower = next(name for name in ROBOTS if name != leader)
    physical = {follower: camera.copy(), leader: camera @ pose_matrix([x, y, 0., 0., 0., yaw])}
    return {name: physical[name] @ inverse(inverse(selected[name]) @ model_from_controller_base(name))
            for name in ROBOTS}


def precise(**kwargs):
    values = dict(measurement_std_m=.001, measurement_std_yaw=.001,
                  acceleration_std_m=.1, acceleration_std_yaw=.3,
                  prediction_horizon=.3)
    values.update(kwargs)
    return PlanarKalmanConfig(**values)


def moving_filter(*, leader="ur10e", selected=None, config=None, velocity=(.04, -.01, .06), steps=50):
    selected = mappings() if selected is None else selected
    estimator = PlanarRelativeStateEstimator(leader=leader, mappings=selected,
                                             config=precise() if config is None else config)
    dt = .05
    for index in range(steps):
        t = index * dt
        estimator.update_pair(pair(.9 + velocity[0]*t, .2 + velocity[1]*t, .4 + velocity[2]*t,
                                   selected=selected, leader=leader), t, t, index)
    return estimator, (steps - 1) * dt


class PlanarRelativeKalmanTests(unittest.TestCase):
    def test_initial_state_is_relative_and_warms_up_without_compensation(self):
        estimator = PlanarRelativeStateEstimator()
        poses = pair()
        estimator.update_pair(poses, 1., 1.02, 1)
        snapshot = estimator.estimate(1.03)
        data = snapshot.relative_filter
        np.testing.assert_allclose(data["G_control"], inverse(poses["ur5e"]) @ poses["ur10e"], atol=1e-12)
        np.testing.assert_array_equal(data["state"][3:], np.zeros(3))
        np.testing.assert_array_equal(data["origin_velocity"], np.zeros(6))
        self.assertEqual(data["lambda"], 0.)
        np.testing.assert_allclose(snapshot.base_poses["ur5e"], poses["ur5e"], atol=1e-12)
        self.assertEqual(snapshot.measurement_times, {name: 1. for name in ROBOTS})
        for twist in snapshot.base_twists.values():
            np.testing.assert_array_equal(twist, np.zeros(6))

    def test_constant_motion_converges_with_positive_confidence(self):
        estimator, at = moving_filter()
        data = estimator.estimate(at).relative_filter
        np.testing.assert_allclose(data["state"][3:], [.04, -.01, .06], atol=.003)
        self.assertGreater(data["lambda"], .4)
        np.testing.assert_allclose(data["origin_velocity"], data["lambda"] * data["raw_origin_velocity"], atol=1e-15)

    def test_common_rigid_camera_motion_has_zero_relative_velocity(self):
        estimator = PlanarRelativeStateEstimator(config=precise())
        start = pair()
        for index in range(30):
            camera_change = pose_matrix([.03*index, -.01*index, .005*index, .01*index, -.015*index, .02*index])
            poses = {name: camera_change @ pose for name, pose in start.items()}
            estimator.update_pair(poses, index*.05, index*.05, index)
        data = estimator.estimate(29*.05).relative_filter
        np.testing.assert_allclose(data["G_control"], inverse(start["ur5e"]) @ start["ur10e"], atol=1e-12)
        np.testing.assert_allclose(data["raw_origin_velocity"], np.zeros(6), atol=1e-12)
        self.assertEqual(data["lambda"], 0.)

    def test_yaw_crossing_pi_has_wrapped_innovation(self):
        estimator = PlanarRelativeStateEstimator(config=precise())
        for index in range(35):
            yaw = math.radians(177.) + .07 * index * .05
            estimator.update_pair(pair(yaw=yaw), index*.05, index*.05, index)
            self.assertLess(abs(estimator.estimate(index*.05).relative_filter["innovation"][2]), .02)
        data = estimator.estimate(34*.05).relative_filter
        self.assertAlmostEqual(data["state"][5], .07, delta=.003)
        self.assertLess(data["state"][2], -3.)

    def test_physical_offsets_output_correction_and_role_reversal(self):
        for corrected in (False, True):
            for leader in ROBOTS:
                selected = mappings(corrected)
                estimator = PlanarRelativeStateEstimator(leader=leader, mappings=selected)
                poses = pair(selected=selected, leader=leader)
                estimator.update_pair(poses, 1., 1., 1)
                snapshot = estimator.estimate(1., leader=leader)
                follower = next(name for name in ROBOTS if name != leader)
                np.testing.assert_allclose(snapshot.relative_filter["G_control"],
                                           inverse(poses[follower]) @ poses[leader], atol=1e-12)
                np.testing.assert_allclose(snapshot.relative_pose, inverse(poses[leader]) @ poses[follower], atol=1e-12)
                physical = {name: snapshot.base_poses[name] @ inverse(selected[name]) @ model_from_controller_base(name)
                            for name in ROBOTS}
                normal = physical["ur5e"][:3, 2]
                np.testing.assert_allclose(normal, physical["ur10e"][:3, 2], atol=1e-12)
                self.assertAlmostEqual(normal @ (physical["ur10e"][:3, 3] - physical["ur5e"][:3, 3]), 0.)

    def test_arbitrary_output_offset_velocity_matches_pose_finite_difference(self):
        offsets = {"ur5e": pose_matrix([.04, -.02, .03, .1, -.2, .3]),
                   "ur10e": pose_matrix([-.03, .05, .02, -.2, .1, -.4])}
        selected = {name: model_from_controller_base(name) @ inverse(offsets[name]) for name in ROBOTS}
        for leader in ROBOTS:
            estimator, at = moving_filter(leader=leader, selected=selected)
            for age in (.02, .21):
                epsilon = 1e-6
                current = estimator.estimate(at + age).relative_filter
                before = estimator.estimate(at + age - epsilon).relative_filter["G_control"]
                after = estimator.estimate(at + age + epsilon).relative_filter["G_control"]
                velocity = np.r_[(after[:3, 3] - before[:3, 3]) / (2*epsilon),
                    Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec() / (2*epsilon)]
                np.testing.assert_allclose(current["raw_origin_velocity"], velocity, atol=1e-8)

    def test_horizon_prediction_stops_but_age_and_covariance_keep_growing(self):
        estimator, at = moving_filter()
        first = estimator.estimate(at).relative_filter
        end = estimator.estimate(at + .3).relative_filter
        late = estimator.estimate(at + .49).relative_filter
        np.testing.assert_allclose(end["G_control"], late["G_control"], atol=1e-12)
        np.testing.assert_array_equal(late["raw_origin_velocity"], np.zeros(6))
        self.assertEqual(late["lambda"], 0.)
        self.assertGreater(np.trace(late["covariance"]), np.trace(first["covariance"]))
        with self.assertRaisesRegex(RelativeStateError, "expired"):
            estimator.estimate(at + .501)

    def test_estimate_does_not_mutate_filter_or_refresh_timestamp(self):
        estimator, at = moving_filter()
        state, covariance = estimator._x.copy(), estimator._P.copy()
        before = estimator.estimate(at + .1)
        for age in np.linspace(.01, .49, 40):
            snapshot = estimator.estimate(at + age)
            snapshot.relative_filter["state"][:] = 99.
            snapshot.base_poses["ur5e"][:] = 88.
        after = estimator.estimate(at + .1)
        np.testing.assert_array_equal(estimator._x, state)
        np.testing.assert_array_equal(estimator._P, covariance)
        np.testing.assert_array_equal(before.relative_filter["state"], after.relative_filter["state"])
        self.assertEqual(after.measurement_times, before.measurement_times)

    def test_outlier_rejection_is_atomic_and_reacquisition_warms_up(self):
        estimator, at = moving_filter()
        state, covariance, capture = estimator._x.copy(), estimator._P.copy(), estimator._capture
        with self.assertRaises(RelativeStateError):
            estimator.update_pair(pair(x=1.8), at+.05, at+.05, 50)
        np.testing.assert_array_equal(estimator._x, state)
        np.testing.assert_array_equal(estimator._P, covariance)
        self.assertEqual(estimator._capture, capture)
        self.assertEqual(estimator.estimate(at+.05).relative_filter["lambda"], 0.)
        estimator.update_pair(pair(.9+.04*(at+.1), .2-.01*(at+.1), .4+.06*(at+.1)), at+.1, at+.1, 51)
        data = estimator.estimate(at+.1).relative_filter
        self.assertEqual(data["support"], 1)
        np.testing.assert_array_equal(data["state"][3:], np.zeros(3))
        self.assertEqual(data["lambda"], 0.)

    def test_external_rejection_and_long_gap_zero_velocity_without_clock_refresh(self):
        estimator, at = moving_filter()
        estimator.note_rejected(at+.01, "camera rejected")
        snap = estimator.estimate(at+.02)
        np.testing.assert_array_equal(snap.relative_filter["origin_velocity"], np.zeros(6))
        np.testing.assert_array_equal(snap.relative_filter["raw_origin_velocity"], np.zeros(6))
        np.testing.assert_allclose(snap.relative_filter["state"][:3], estimator._x[:3], atol=1e-12)
        self.assertEqual(snap.measurement_times["ur5e"], at)
        estimator.update_pair(pair(.9+.04*(at+.4), .2-.01*(at+.4), .4+.06*(at+.4)), at+.4, at+.4, 51)
        np.testing.assert_array_equal(estimator.estimate(at+.4).relative_filter["state"][3:], np.zeros(3))

    def test_stationary_measurements_suppress_previous_motion(self):
        estimator, at = moving_filter()
        held = pair(.9+.04*at, .2-.01*at, .4+.06*at)
        for index in range(1, 15):
            estimator.update_pair(held, at+index*.05, at+index*.05, 49+index)
        data = estimator.estimate(at+.7).relative_filter
        np.testing.assert_array_equal(data["state"][3:], np.zeros(3))
        self.assertEqual(data["lambda"], 0.)

    def test_joseph_covariance_stays_symmetric_psd_under_noisy_motion(self):
        random = np.random.default_rng(812)
        estimator = PlanarRelativeStateEstimator(config=precise(innovation_gate=40.))
        for index in range(160):
            t = index*.04
            noise = random.normal(0., .0004, 3)
            estimator.update_pair(pair(.9+.04*t+noise[0], .2-.01*t+noise[1], .4+.03*t+noise[2]), t, t, index)
            covariance = estimator.estimate(t+.05).relative_filter["covariance"]
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-14)

    def test_identical_duplicate_is_ignored_but_reordered_changed_or_stale_data_rejected(self):
        poses = pair()
        estimator = PlanarRelativeStateEstimator()
        estimator.update_pair(poses, 1., 1., 10)
        self.assertFalse(estimator.update_pair(deepcopy(poses), 1., 1.1, 10))
        self.assertEqual(estimator.estimate(1.1).received_times["ur5e"], 1.)
        for args in ((pair(x=.91), 1., 1.1, 10), (poses, 1., 1.1, 11),
                     (poses, .9, 1.1, 9), (poses, 1.1, 1., 11), (poses, 1.1, 1.7, 11)):
            with self.subTest(args=args[1:]), self.assertRaises(RelativeStateError):
                estimator.update_pair(*args)
        self.assertEqual(estimator._capture, 1.)

    def test_stationary_consensus_uses_filtered_pose_but_keeps_raw_capture(self):
        estimator = PlanarRelativeStateEstimator()
        raw, consensus = pair(.904), pair(.9)
        estimator.initialize_stationary_pair(raw, consensus, 1., 1.1, 1)
        snapshot = estimator.estimate(1.1)
        self.assertAlmostEqual(snapshot.relative_filter["state"][0], .9)
        np.testing.assert_array_equal(snapshot.base_poses["ur5e"], raw["ur5e"])
        self.assertFalse(estimator.update_pair(raw, 1., 1.2, 1))
        with self.assertRaisesRegex(RelativeStateError, "empty"):
            estimator.initialize_stationary_pair(raw, consensus, 1.2, 1.2, 2)

    def test_configuration_roundtrip_rejects_unknown_and_invalid_settings(self):
        original = PlanarKalmanConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"relative.json"
            path.write_text(json.dumps(asdict(original)))
            self.assertEqual(load_planar_kalman_config(path), original)
        for invalid in ({"typo": 1}, {"measurement_std_m": -1}, {"measurement_std_m": "bad"},
                        {"min_velocity_samples": True}, {"velocity_snr_full": .1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PlanarKalmanConfig.from_dict(invalid)
        with self.assertRaises(ValueError):
            PlanarRelativeStateEstimator(max_measurement_age=.1)

    def test_noisy_stationary_prediction_reduces_position_and_yaw_jitter(self):
        # Independent held-out noise seeds, with the measured camera latency.
        # This checks the actual delayed prediction, not just the posterior.
        for seed in (812, 923):
            random = np.random.default_rng(seed)
            estimator = PlanarRelativeStateEstimator()
            raw, predicted, compensation = [], [], []
            for index in range(250):
                t = index*.08
                noise = random.normal(0., [.007, .007, math.radians(.3)])
                estimator.update_pair(pair(*([.9, .2, .4]+noise)), t, t+.18, index)
                data = estimator.estimate(t+.18).relative_filter
                if t >= 3.:
                    raw.append(noise)
                    predicted.append(data["state"][:3]-[.9, .2, .4])
                    compensation.append(data["origin_velocity"])
            raw_rms = np.sqrt(np.mean(np.square(raw), axis=0))
            predicted_rms = np.sqrt(np.mean(np.square(predicted), axis=0))
            ff_rms = np.sqrt(np.mean(np.square(compensation), axis=0))
            with self.subTest(seed=seed):
                self.assertTrue(np.all(predicted_rms < .8*raw_rms), (raw_rms, predicted_rms))
                self.assertLess(np.linalg.norm(ff_rms[:3]), .001)
                self.assertLess(np.linalg.norm(ff_rms[3:]), math.radians(.1))

    def test_noisy_delayed_motion_has_useful_compensation_and_stops_reverses(self):
        random = np.random.default_rng(812)
        estimator = PlanarRelativeStateEstimator()
        rows = []
        for index in range(250):
            t = index*.08
            travel = 0. if t < 2. else t-2. if t < 6. else 4. if t < 8. else 12.-t if t < 12. else 0.
            truth = np.array([.9, .2, .4])+np.array([.04, .01, .035])*travel
            noise = random.normal(0., [.007, .007, math.radians(.3)])
            estimator.update_pair(pair(*(truth+noise)), t, t+.18, index)
            data = estimator.estimate(t+.18).relative_filter
            rows.append([t, data["lambda"], *data["origin_velocity"][[0, 1, 5]]])
        rows = np.array(rows)
        forward = rows[(rows[:, 0] >= 3.) & (rows[:, 0] < 6.)]
        reverse = rows[(rows[:, 0] >= 9.) & (rows[:, 0] < 12.)]
        stopped = rows[(rows[:, 0] >= 7.) & (rows[:, 0] < 8.)]
        settled = rows[rows[:, 0] >= 14.]
        self.assertGreater(np.mean(forward[:, 1] > 0.), .8)
        self.assertGreater(forward[:, 2].mean(), .008)
        self.assertLess(reverse[:, 2].mean(), -.008)
        self.assertLess(np.mean(np.abs(stopped[:, 2])), .002)
        self.assertLess(np.mean(np.abs(settled[:, 2])), .001)
        self.assertLess(np.mean(np.abs(settled[:, 4])), math.radians(.1))

    def test_prediction_gain_is_directional_and_not_applied_twice(self):
        estimator, at = moving_filter()
        # Hold a high-confidence translational estimate with an unobservable
        # yaw rate. Translation must not authorize extrapolating noisy yaw.
        estimator._x[5] = .001
        estimator._P[5, 5] = .01
        data = estimator.estimate(at+.08).relative_filter
        self.assertGreater(data["prediction_gain"][0], .9)
        self.assertEqual(data["prediction_gain"][2], 0.)
        self.assertEqual(data["state"][5], 0.)
        self.assertEqual(data["state"][2], estimator._x[2])
        factors = data["confidence_factors"]
        self.assertAlmostEqual(data["lambda"], factors["covariance"]*factors["age"])
        np.testing.assert_allclose(data["origin_velocity"], data["lambda"]*data["raw_origin_velocity"])

    def test_partial_confidence_prediction_derivative_stays_coherent(self):
        estimator, at = moving_filter()
        # SNR=2 is a fractional predictor gain, not a yes/no motion switch.
        velocity = estimator._x[3:5]
        estimator._P[3:5, 3:5] = np.eye(2)*(np.linalg.norm(velocity)/2.)**2
        for age in (.03, .21):
            epsilon = 1e-6
            data = estimator.estimate(at+age).relative_filter
            self.assertGreater(data["prediction_gain"][0], 0.)
            self.assertLess(data["prediction_gain"][0], 1.)
            before = estimator.estimate(at+age-epsilon).relative_filter["G_control"]
            after = estimator.estimate(at+age+epsilon).relative_filter["G_control"]
            derivative = np.r_[(after[:3, 3]-before[:3, 3])/(2*epsilon),
                              Rotation.from_matrix(after[:3, :3]@before[:3, :3].T).as_rotvec()/(2*epsilon)]
            np.testing.assert_allclose(data["raw_origin_velocity"], derivative, atol=1e-8)

    def test_clear_reversal_requires_new_velocity_support(self):
        estimator, at = moving_filter(config=precise(innovation_gate=1e6))
        estimator.update_pair(pair(.9+.04*at-.01, .2-.01*at, .4+.06*at), at+.05, at+.05, 50)
        data = estimator.estimate(at+.05).relative_filter
        self.assertEqual(data["support"], 1)
        self.assertIn("direction change", estimator._reason)
        np.testing.assert_array_equal(data["state"][3:], np.zeros(3))
        self.assertEqual(data["lambda"], 0.)

    def test_relative_velocity_is_bounded_before_control_compensation(self):
        config = precise(max_speed=.02, max_yaw_rate=.02, innovation_gate=1e6)
        estimator, at = moving_filter(config=config)
        data = estimator.estimate(at).relative_filter
        self.assertLessEqual(np.linalg.norm(data["posterior_state"][3:5]), .02+1e-12)
        self.assertLessEqual(abs(data["posterior_state"][5]), .02+1e-12)
        self.assertLessEqual(np.linalg.norm(data["origin_velocity"][:3]), .02+1e-12)

    def test_same_pair_with_reversed_roles_inverts_relative_pose(self):
        estimators = {name: PlanarRelativeStateEstimator(leader=name, config=precise()) for name in ROBOTS}
        for index in range(50):
            t = index*.05
            poses = pair(.9+.04*t, .2-.01*t)
            for estimator in estimators.values():
                estimator.update_pair(poses, t, t, index)
        relations = {name: estimator.estimate(49*.05+.1).relative_filter["G_control"]
                     for name, estimator in estimators.items()}
        np.testing.assert_allclose(relations["ur5e"], inverse(relations["ur10e"]), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
