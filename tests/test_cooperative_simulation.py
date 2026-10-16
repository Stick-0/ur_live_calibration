"""Offline synthetic-plant integration checks; never instantiate robot drivers."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.tools.analyze_log import summarize_log
from urcal.cooperative.controller import CooperativeController
from urcal.control.geometry import pose_matrix
from urcal.tools.simulate import CartesianModel, METHODS, run_comparison, sensor_stream


class CooperativeSimulationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def records(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_smoke_uses_production_prepare_commit_and_exports_analyzable_artifacts(self):
        prepare, commit = CooperativeController.prepare, CooperativeController.commit
        with patch.object(CooperativeController, "prepare", autospec=True, side_effect=prepare) as prepared, \
                patch.object(CooperativeController, "commit", autospec=True, side_effect=commit) as committed:
            report = run_comparison(self.root, duration=.2, scenario="static")
        self.assertGreater(prepared.call_count, 0)
        self.assertGreater(committed.call_count, 0)
        self.assertTrue(report["simulation"])
        self.assertIn("no UR dynamics", report["notice"])
        self.assertTrue((self.root / "comparison.json").is_file())
        for method in METHODS:
            log, reference = self.root / f"{method}.jsonl", self.root / f"{method}.reference.jsonl"
            actual = summarize_log(log, reference)
            self.assertEqual(actual["active_control_records"], 6)
            self.assertEqual(actual["independent_real_error"]["matched_control_records"], 6)
            self.assertAlmostEqual(actual["elapsed_control_time_s"], .2)
            self.assertIsNone(actual["completion_time_s"])
            self.assertEqual(actual["faults"], [])
            self.assertTrue(report["methods"][method]["simulation"])
            self.assertIn("synthetic", report["methods"][method]["reference_provenance"])
            self.assertTrue(all(record["synthetic"] for record in self.records(reference)))

    def test_identical_seed_produces_identical_trials_and_shared_sensor_samples(self):
        first, second = self.root / "first", self.root / "second"
        run_comparison(first, duration=.2, scenario="noise", seed=43)
        run_comparison(second, duration=.2, scenario="noise", seed=43)
        measurement_sets = []
        for method in METHODS:
            self.assertEqual((first / f"{method}.jsonl").read_text(), (second / f"{method}.jsonl").read_text())
            measurement_sets.append([(record["sequence"], record["captured_at"], record["poses"])
                                     for record in self.records(first / f"{method}.jsonl") if record["kind"] == "measurement"])
        self.assertEqual(measurement_sets[0], measurement_sets[1])
        self.assertEqual(measurement_sets[1], measurement_sets[2])

    def test_sensor_delay_and_noise_seed_are_explicit(self):
        first, second = sensor_stream(.2, 3, "noise"), sensor_stream(.2, 4, "noise")
        self.assertAlmostEqual(first[0][0] - first[0][1].captured_at, .08)
        self.assertFalse(np.array_equal(first[0][1].poses["ur5e"], second[0][1].poses["ur5e"]))

    def test_dropout_stops_all_methods_without_fabricating_completion(self):
        report = run_comparison(self.root, duration=1.4, scenario="dropout")
        for method in METHODS:
            result = report["methods"][method]
            self.assertFalse(result["task_completed"])
            self.assertTrue(result["faults"])
            self.assertTrue(result["faults"][0]["category"].startswith("measurement"))
            self.assertLess(result["elapsed_active_time_s"], 1.4)
            self.assertAlmostEqual(result["elapsed_control_time_s"], 1.4)
            records = self.records(self.root / f"{method}.jsonl")
            fault_index = next(i for i, record in enumerate(records) if record["kind"] == "fault")
            self.assertFalse(any(record.get("active") for record in records[fault_index:] if record["kind"] == "control"))

    def test_compensation_infeasibility_is_recorded_as_a_fault(self):
        report = run_comparison(self.root, duration=1., scenario="infeasible")
        result = report["methods"]["separated"]
        self.assertTrue(result["faults"])
        self.assertEqual(result["faults"][0]["category"], "compensation_infeasible")
        self.assertIsNone(result["completion_time_s"])

    def test_hold_has_no_completion_even_with_continuous_compensation(self):
        report = run_comparison(self.root, duration=.2, scenario="bases", path="hold")
        for result in report["methods"].values():
            self.assertFalse(result["task_completed"])
            self.assertIsNone(result["completion_time_s"])
            self.assertEqual(result["progress"]["coverage"], 0.)

    def test_path_phase_finishing_does_not_imply_endpoint_attainment(self):
        report = run_comparison(self.root, duration=.4, scenario="static", path_duration=.08)
        # The direct baseline advances the path clock quickly but its bounded
        # feedback plant needs longer to reach and settle at the actual target.
        direct = report["methods"]["direct"]
        self.assertEqual(direct["progress"]["last"], 1.)
        self.assertFalse(direct["task_completed"])
        self.assertIsNone(direct["completion_time_s"])
        records = self.records(self.root / "direct.jsonl")
        controls = [record for record in records if record["kind"] == "control"]
        self.assertTrue(any(record["path_finished"] for record in controls))
        self.assertFalse(any(record["finished"] for record in controls))
        self.assertIn("estimated endpoint attainment", records[0]["completion_definition"])

    def test_completion_requires_endpoint_tolerance_and_dwell_after_phase_end(self):
        report = run_comparison(self.root, duration=4., scenario="static", path_duration=.08)
        records = self.records(self.root / "direct.jsonl")
        controls = [record for record in records if record["kind"] == "control"]
        phase_finished = next(record for record in controls if record["path_finished"])
        completed = next(record for record in controls if record["finished"])
        self.assertLessEqual(completed["relative_error_m"], .005)
        self.assertLessEqual(completed["relative_error_rad"], np.deg2rad(2.))
        self.assertGreaterEqual(completed["at"] - phase_finished["at"], .5)
        self.assertTrue(report["methods"]["direct"]["task_completed"])

    def test_existing_artifacts_are_never_overwritten(self):
        run_comparison(self.root, duration=.08, scenario="static")
        original = (self.root / "direct.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            run_comparison(self.root, duration=.08, scenario="static")
        self.assertEqual((self.root / "direct.jsonl").read_bytes(), original)

    def test_cartesian_model_jacobian_matches_its_actual_coordinate_kinematics(self):
        q = np.array([.2, -.3, .4, .4, -.2, .3])
        model = CartesianModel(Rotation.from_rotvec([np.pi, 0, 0]).as_matrix())
        actual, jacobian = model.pose_and_jacobian(q)
        np.testing.assert_allclose(model.coordinates(actual), q, atol=1e-12)
        epsilon = 1e-6
        finite = np.empty((6, 6))
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = epsilon
            plus = model.pose_and_jacobian(q + delta)[0]
            minus = model.pose_and_jacobian(q - delta)[0]
            finite[:3, index] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
            finite[3:, index] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon)
        np.testing.assert_allclose(jacobian, finite, atol=1e-9)

    def test_import_does_not_import_robot_or_camera_drivers(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import urcal.tools.simulate; "
                                 "assert not {'rtde_control','rtde_receive','pyrealsense2'} & set(sys.modules)"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10.)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
