"""Offline calibration tuning must separate evidence, labels and truth."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base, pose_matrix
from urcal.cooperative.relative_kalman import PlanarKalmanConfig
from urcal.tools.tune_relative_kalman import (
    Series, arm_motion, audit, centered_residuals, fit_measurement_noise,
    heldout_validation, load_segments, main, planar_relative, provisional_config,
    read_series, replay, robust_covariance, synthetic_comparison, synthetic_validation, wrap_angle,
)


def poses(value, anchor=None):
    anchor = np.eye(4) if anchor is None else anchor
    g = pose_matrix([value[0], value[1], 0., 0., 0., value[2]])
    return {"ur5e": anchor @ inverse(model_from_controller_base("ur5e")),
            "ur10e": anchor @ g @ inverse(model_from_controller_base("ur10e"))}


def write_log(path, count=80, *, seed=31):
    rng = np.random.default_rng(seed)
    rows = [{"kind": "metadata", "at": 100., "settings": {"common_base_plane": True}}]
    for index in range(count):
        stamp = 100. + index * .08
        value = np.array([1.2, .2, np.pi - .001]) + rng.normal(size=3) * [.002, .004, .002]
        rows.append({"kind": "measurement", "at": stamp + .16, "received_at": stamp + .16,
                     "captured_at": stamp, "sequence": index, "accepted": index % 5 != 0,
                     "valid_counts": {"ur5e": 7, "ur10e": 7},
                     "poses_camera_model": {name: matrix.tolist() for name, matrix in poses(value).items()},
                     "pose_diagnostics": {"robots": {name: {"q_rad": [index * .002 if name == "ur5e" else 0.] * 6}
                                                       for name in ("ur5e", "ur10e")}}})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def manifest(path, log, entries):
    value = {"schema": "relative-kalman-segments-v1", "leader": "ur10e", "segments": [
        {"log": log.name, "start_s": start, "end_s": end, "condition": label,
         "partition": partition, "evidence": evidence}
        for start, end, label, partition, evidence in entries]}
    path.write_text(json.dumps(value))


class RelativeKalmanTuningTests(unittest.TestCase):
    def test_same_capture_physical_relative_pose_cancels_camera_and_model_mapping(self):
        value = np.array([1.2, -.2, np.pi - .01])
        camera = pose_matrix([.2, -.7, 2., .3, -.5, .9])
        np.testing.assert_allclose(planar_relative(poses(value)), value, atol=1e-12)
        np.testing.assert_allclose(planar_relative(poses(value, camera)), value, atol=1e-12)
        reversed_value = planar_relative(poses(value, camera), leader="ur5e")
        reversed_g = inverse(pose_matrix([*value[:2], 0., 0., 0., value[2]]))
        np.testing.assert_allclose(reversed_value[:2], reversed_g[:2, 3], atol=1e-12)
        self.assertAlmostEqual(reversed_value[2], -value[2])

    def test_nonplanar_measurement_is_not_silently_projected(self):
        pair = poses([1., .2, .1])
        pair["ur10e"][2, 3] += .01
        with self.assertRaisesRegex(ValueError, "not physically coplanar"):
            planar_relative(pair)

    def test_audit_includes_gate_rejections_but_never_calls_arm_motion_base_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            rows = write_log(path)
            with path.open("a") as stream:
                stream.write(json.dumps(rows[5]) + "\n")
            series = read_series(path)
            self.assertEqual(len(series.times), 80)
            self.assertEqual(series.excluded["duplicate_capture"], 1)
            self.assertEqual(int((~series.accepted).sum()), 16)
            moving, known = arm_motion(series)
            self.assertEqual(int(known.sum()), 79)
            self.assertTrue(moving[1:].all())
            result = audit(series)
            self.assertEqual(result["base_motion_label"], "unknown")
            self.assertFalse(result["ground_truth_available"])
            self.assertFalse(result["raw_spread_is_noise_estimate"])
            self.assertAlmostEqual(result["timing"]["receipt_age_median_p95_s"][0], .16)

    def test_mad_handles_yaw_wrap_correlated_noise_and_outliers(self):
        rng = np.random.default_rng(132)
        raw = rng.normal(size=(2000, 3)) * [.002, .004, .002]
        raw[:, 1] += .8 * raw[:, 0]
        raw += [1.2, .3, np.pi - .0005]
        raw[:, 2] = wrap_angle(raw[:, 2])
        raw[::40, :2] += .2
        residual, center = centered_residuals(raw)
        covariance, scale = robust_covariance(residual)
        self.assertAlmostEqual(abs(center[2]), np.pi, delta=.003)
        self.assertGreater(scale[0], .0017)
        self.assertLess(scale[0], .0024)
        self.assertLess(scale[2], .0024)
        self.assertGreater(covariance[0, 1], 0.)
        self.assertTrue((np.linalg.eigvalsh(covariance) > 0.).all())

    def test_unknown_and_active_motion_are_never_fitted_as_measurement_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_log(root / "run.jsonl")
            for label in ("unknown", "base_moving"):
                manifest(root / "segments.json", root / "run.jsonl",
                         [(0., 5., label, "train", "operator notes" if label != "unknown" else "")])
                with self.assertRaisesRegex(ValueError, "no explicitly labelled"):
                    fit_measurement_noise(load_segments(root / "segments.json"))

    def test_manifest_requires_evidence_and_disallows_train_validation_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_log(root / "run.jsonl")
            manifest(root / "segments.json", root / "run.jsonl",
                     [(0., 3., "base_stationary_arm_stationary", "train", "")])
            with self.assertRaisesRegex(ValueError, "evidence"):
                load_segments(root / "segments.json")
            manifest(root / "segments.json", root / "run.jsonl", [
                (0., 3., "base_stationary_arm_moving", "train", "operator confirms fixed bases"),
                (2., 5., "base_stationary_arm_moving", "validation", "operator confirms fixed bases")])
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_segments(root / "segments.json")

    def test_validation_does_not_tune_noise_and_motion_without_truth_is_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_log(root / "run.jsonl", count=120)
            manifest(root / "segments.json", root / "run.jsonl", [
                (0., 4., "base_stationary_arm_moving", "train", "operator fixed base"),
                (4., 8., "base_moving", "validation", "operator moved base")])
            segments = load_segments(root / "segments.json")
            before, report = fit_measurement_noise(segments)
            segments[1][1].values[:, :2] *= 10.
            after, _ = fit_measurement_noise(segments)
            self.assertEqual(before, after)
            self.assertEqual(report["training_samples"], 50)
            self.assertFalse(report["absolute_accuracy_validated"])
            validation = heldout_validation(segments, PlanarKalmanConfig(**before))
            self.assertIsNone(validation[0]["filtered"])
            self.assertIn("unscored", validation[0]["metric"])

    def test_provisional_uses_explicit_protocol_proxy_and_age_not_active_spread(self):
        audits = [{"phases": [{"phase": "preview", "raw_spread_m_m_rad": [.001, .002, .002]},
                              {"phase": "alignment", "raw_spread_m_m_rad": [.0036, .0063, np.deg2rad(.228)]},
                              {"phase": "tracking", "raw_spread_m_m_rad": [.2, .3, .4]}],
                   "timing": {"control_age_median_p95_s": [.19, .26]}}]
        config, report = provisional_config(audits)
        self.assertAlmostEqual(config.measurement_std_m, .007)
        self.assertAlmostEqual(config.measurement_std_yaw, np.deg2rad(.3))
        self.assertAlmostEqual(config.prediction_horizon, .35)
        self.assertFalse(report["measurement_noise_validated"])
        self.assertFalse(report["motion_tracking_improvement_validated"])
        self.assertIn("not independently verified", report["assumption"])

    def test_production_filter_replay_preserves_gaps_and_rejects_duplicate_times(self):
        config = PlanarKalmanConfig(prediction_horizon=.35)
        t = np.arange(20) * .08
        values = np.repeat([[1., .2, 0.]], len(t), axis=0)
        result = replay(t, values, config)
        np.testing.assert_allclose(result["estimates"], values, atol=1e-10)
        np.testing.assert_allclose(result["velocities"], 0., atol=1e-10)
        self.assertEqual(result["rejected"], 0)
        with self.assertRaisesRegex(ValueError, "distinct increasing"):
            replay(np.zeros(20), values, config)
        times = np.arange(28) * .08
        samples = np.repeat([[1., .2, 0.]], len(times), axis=0)
        samples[20:27, 0] += .5
        missing = replay(times, samples, config)
        self.assertEqual(missing["rejected"], 7)
        self.assertTrue(np.isnan(missing["estimates"][24:27]).all())
        self.assertTrue(np.isfinite(missing["estimates"][-1]).all())

    def test_synthetic_comparison_is_reproducible_and_does_not_claim_hardware_validation(self):
        config = PlanarKalmanConfig(measurement_std_m=.007, measurement_std_yaw=np.deg2rad(.3), prediction_horizon=.35)
        first, second = synthetic_comparison(config), synthetic_comparison(config)
        self.assertEqual(first, second)
        self.assertEqual(first["kind"], "synthetic_known_truth_not_hardware_validation")
        self.assertEqual(set(first["comparisons"]), {"q_half", "candidate", "yaw_q_half", "q_double", "short_horizon"})
        for comparison in first["comparisons"].values():
            self.assertGreater(comparison["errors"]["moving"]["valid_samples"], 30)
            self.assertGreaterEqual(comparison["errors"]["stopped"]["position_rmse_m"], 0.)

    def test_cli_cannot_overwrite_original_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            write_log(path)
            before = path.read_bytes()
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                main(["audit", str(path), "--output", str(path)])
            self.assertEqual(before, path.read_bytes())

    def test_heldout_seeds_exclude_exploration_and_do_not_change_selected_config(self):
        current = PlanarKalmanConfig()
        for seeds in ((2718,), (31, 31), (), (-1,)):
            with self.assertRaisesRegex(ValueError, "excluding exploratory"):
                synthetic_validation(current, seeds=seeds)
        result = synthetic_validation(current, seeds=(31415,))
        self.assertFalse(result["config_retuned_by_validation"])
        self.assertEqual(result["comparisons"]["candidate"]["config"], asdict(current))
        aggregate = result["comparisons"]["candidate"]["trajectory_errors"]
        sole_run = result["comparisons"]["candidate"]["runs"][0]["trajectory"]["errors"]
        for phase, metrics in aggregate.items():
            self.assertEqual(metrics["valid_samples"], sole_run[phase]["valid_samples"])
            for name in ("position_rmse_m", "yaw_rmse_rad"):
                self.assertAlmostEqual(metrics[name], sole_run[phase][name], places=14)


if __name__ == "__main__":
    unittest.main()
