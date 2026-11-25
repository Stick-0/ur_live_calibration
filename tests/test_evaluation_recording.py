"""Evaluation truthfulness, immutable targets and reproducible raw-log export."""

import csv
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.evaluation import EvaluationRecorder, evaluation_flags
from urcal.control.geometry import inverse, pose_matrix
from urcal.cooperative.logging import ExperimentLogger, LoggingError
from urcal.tools.export_evaluation import export_session


def scene(at=10., relative=None, *, valid=True, age=.1):
    relative = pose_matrix([.1, -.2, .3, 0., 0., .2]) if relative is None else relative
    base = pose_matrix([.5, -.3, 2., .1, -.2, .3])
    bases = {"ur10e": base, "ur5e": base @ pose_matrix([.8, .1, 0., 0., 0., .4])}
    offsets = {"ur10e": pose_matrix([.02, 0., .1, 0., .2, 0.]), "ur5e": pose_matrix([0., .03, .2, .1, 0., 0.])}
    leader_flange = pose_matrix([.3, .1, .2, .1, .2, -.1])
    follower_flange = inverse(bases["ur5e"]) @ bases["ur10e"] @ leader_flange @ relative
    flanges = {"ur10e": leader_flange, "ur5e": follower_flange}
    return {"at": at, "bases": bases, "calibration_valid": valid, "measurement_age": {name: age for name in bases},
            "robots": {name: {"q": [0.] * 6, "qd": [.01] * 6, "tcp": flanges[name] @ offsets[name],
                              "tcp_offset": offsets[name], "robot_timestamp": 42., "at": at} for name in bases}}


def wait_until(predicate, timeout=3.):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("evaluation worker did not complete")


class EvaluationRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name) / "source.jsonl"
        self.logger = ExperimentLogger(self.raw)
        self.addCleanup(self.logger.close)
        self.settings = {"base_motion_mode": "pose", "method": "separated", "follow_law": "feedforward",
                         "max_robot_age": .5, "max_calibration_age": .5, "nested": {"gain": 3.}}
        self.recorder = EvaluationRecorder(self.logger, self.raw, Path(self.temp.name) / "exports",
            settings=self.settings, leader="ur10e", follower="ur5e")
        self.addCleanup(self.recorder.close)
        self.target = pose_matrix([.2, -.1, .4, 0., 0., .4])

    def start(self, target=None, request_id="start", trial_id="trial-1", now=10.):
        self.assertTrue(self.recorder.request({"type": "evaluation", "action": "start", "request_id": request_id,
                                              "trial_id": trial_id}, now=now, target=self.target if target is None else target))
        return self.recorder.status()["session_id"]

    def stop(self, now=11., request_id="stop"):
        self.assertTrue(self.recorder.request({"type": "evaluation", "action": "stop", "request_id": request_id}, now=now))
        wait_until(lambda: not self.recorder.status()["busy"])
        status = self.recorder.status()
        self.assertEqual(status["state"], "saved", status)
        with Path(status["csv_path"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows, json.loads(Path(status["metadata_path"]).read_text())

    def test_frozen_nominal_target_and_settings_ignore_mutation_and_control_target(self):
        frozen = self.target.copy()
        self.start()
        self.target[:3, 3] = 99.
        self.settings["nested"]["gain"] = 999.
        actual = pose_matrix([.1, -.2, .3, 0., 0., .2])
        control = {"at": 10., "estimated_relative_pose": actual, "desired_relative_pose": actual.copy(),
                   "tracking_relative_target": actual.copy(), "measurement_times": {"ur5e": 9.9, "ur10e": 9.9}}
        self.recorder.record(at=10., phase="ACTIVE", scene=scene(), control_record=control)
        actual[:3, 3] = 999.
        rows, metadata = self.stop()
        self.assertEqual(metadata["start"]["settings"]["nested"]["gain"], 3.)
        np.testing.assert_allclose(metadata["start"]["frozen_target"], frozen)
        cycle = rows[1]
        self.assertEqual(cycle["metric_valid"], "1")
        np.testing.assert_allclose([float(cycle[f"position_error_m_{i}"]) for i in range(3)], [.1, .1, .1])
        self.assertAlmostEqual(float(cycle["rotation_error_rad_2"]), .2)
        self.assertEqual(cycle["metric_source"], "control_record")

    def test_scene_relative_flange_accounts_for_camera_bases_and_both_tcp_offsets(self):
        self.start()
        supplied = scene()
        self.recorder.record(at=10., phase="PREVIEW", scene=supplied)
        rows, _ = self.stop()
        np.testing.assert_allclose([float(rows[1][f"position_error_m_{i}"]) for i in range(3)], [.1, .1, .1], atol=1e-14)
        reconstructed = np.array([[float(rows[1][f"G_control_r{i}c{j}"]) for j in range(4)] for i in range(4)])
        np.testing.assert_allclose(reconstructed, inverse(supplied["bases"]["ur5e"]) @ supplied["bases"]["ur10e"])
        self.assertEqual(rows[1]["preview"], "1")

    def test_every_failure_and_repeated_producer_cycle_is_retained_with_blank_invalid_errors(self):
        self.start()
        for _ in range(3):
            self.recorder.record(at=10., phase="FAULT", scene=None, reason="connection failure")
        self.recorder.record(at=11., phase="STALE", scene=scene(), control_record={"at": 10., "twist": [float("nan")] * 6})
        self.recorder.record(at=11., phase="FAULT", scene=scene(11.), reason="send failed but fresh measurement")
        rows, metadata = self.stop(now=11.1)
        cycles = [row for row in rows if row["kind"] == "evaluation_cycle"]
        self.assertEqual(len(cycles), 5)
        self.assertEqual([row["metric_valid"] for row in cycles], ["0", "0", "0", "0", "1"])
        self.assertTrue(all(row["position_error_norm_mm"] == "" for row in cycles[:4]))
        self.assertEqual(cycles[4]["reason"], "send failed but fresh measurement")
        self.assertIn("__nonfinite_float__", cycles[3]["control_record_json"])
        self.assertFalse(metadata["incomplete"])

    def test_no_sent_command_without_this_transaction_sdk_success(self):
        self.start()
        control = {"at": 10., "twist": [.04] * 6, "joint_command": [.02] * 6, "active": True}
        transports = [None, {"last_successful_send": {"ur5e": {"at": 9., "command": {"velocity": [1.] * 6}}}},
            {"operations": [{"stage": "speedL", "robot": "ur5e", "returned": False,
                             "command": {"velocity": [.1] * 6}, "finished_at": 10.}]},
            {"operations": [{"stage": "speedL", "robot": "ur5e", "returned": True,
                             "command": {"velocity": [.03] * 6, "acceleration": .5, "duration_s": .04}, "finished_at": 10.}]}]
        for transport in transports:
            self.recorder.record(at=10., phase="ACTIVE", scene=scene(), control_record=control, transport=transport)
        rows, _ = self.stop()
        cycles = rows[1:-1]
        self.assertEqual([row["command_sent"] for row in cycles], ["0", "0", "0", "1"])
        self.assertEqual([row["command_attempted"] for row in cycles], ["0", "0", "1", "1"])
        self.assertEqual(cycles[-1]["final_sent_velocity_0"], "0.03")
        self.assertEqual(cycles[-1]["proposed_velocity_0"], "0.04")
        self.assertTrue(all(row["final_sent_velocity_0"] == "" for row in cycles[:-1]))

    def test_simulated_applied_command_has_separate_columns_and_no_physical_send(self):
        self.start()
        transport = {"simulated": True, "simulated_command_applied": True,
                     "commands": {"ur5e": {"method": "speedJ", "command": [.1] * 6,
                                              "joint_velocity": [.1] * 6, "applied_method": "joint_velocity_step"}},
                     "operations": [{"stage": "speedL", "robot": "ur5e", "returned": True,
                                     "command": {"velocity": [.2] * 6}, "finished_at": 10.}]}
        self.recorder.record(at=10., phase="ALIGNMENT", scene=scene(), transport=transport)
        rows, _ = self.stop()
        self.assertEqual(rows[1]["command_sent"], "0")
        self.assertEqual(rows[1]["simulated_command_applied"], "1")
        self.assertEqual(rows[1]["simulated_applied_method"], "speedJ")
        self.assertEqual(rows[1]["simulated_applied_joint_velocity_0"], "0.1")
        self.assertEqual(rows[1]["final_sent_velocity_0"], "")

    def test_configured_flags_distinguish_state_estimation_prediction_and_compensation(self):
        cases = [("pose", "predict", "feedforward", (False, False, False)),
                 ("velocity", "direct", "feedforward", (False, False, False)),
                 ("velocity", "predict", "feedforward", (False, True, False)),
                 ("velocity", "separated", "feedforward", (False, True, True)),
                 ("relative_kalman", "separated", "feedforward", (True, True, True)),
                 ("pose", "separated", "pose", (False, False, False))]
        for mode, method, law, expected in cases:
            with self.subTest(mode=mode, method=method):
                result = evaluation_flags({"base_motion_mode": mode, "method": method, "follow_law": law})
                self.assertEqual(tuple(result.values()), expected)

    def test_markers_request_order_idempotency_and_repeated_trial_labels(self):
        first = self.start()
        self.assertTrue(self.recorder.request({"action": "start", "request_id": "start"}, now=20.))
        for action, stamp in (("table_start", 10.1), ("table_stop", 10.3)):
            command = {"action": action, "request_id": action}
            self.assertTrue(self.recorder.request(command, now=stamp))
            self.assertTrue(self.recorder.request(command, now=stamp + .1))
            self.recorder.record(at=stamp, phase="ACTIVE", scene=scene(stamp))
        rows, _ = self.stop()
        self.assertEqual([row["kind"] for row in rows], ["evaluation_start", "evaluation_marker", "evaluation_cycle",
                         "evaluation_marker", "evaluation_cycle", "evaluation_stop"])
        self.assertEqual([row["table_moving"] for row in rows if row["kind"] == "evaluation_cycle"], ["1", "0"])
        first_path = self.recorder.status()["csv_path"]
        second = self.start(request_id="start2", now=20.)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_path, self.recorder.status()["csv_path"])
        self.assertTrue(self.recorder.request({"action": "stop", "request_id": "stop"}, now=20.))
        self.assertEqual(self.recorder.status()["state"], "recording")
        self.stop(now=20.1, request_id="stop2")

    def test_manual_stop_preserves_request_time_and_automatic_stop_has_no_request(self):
        self.start()
        command = {"type": "evaluation", "action": "stop", "request_id": "manual-stop",
                   "requested_at": 10.75, "reason": "operator finished"}
        self.assertTrue(self.recorder.request(command, now=11.))
        command["requested_at"] = 999.
        wait_until(lambda: not self.recorder.status()["busy"])
        metadata = json.loads(Path(self.recorder.status()["metadata_path"]).read_text())
        stop = metadata["stops"][0]
        self.assertEqual(stop["at"], 11.)
        self.assertEqual(stop["request"]["requested_at"], 10.75)
        self.assertEqual(stop["request"]["request_id"], "manual-stop")
        self.start(request_id="next-start", now=20.)
        status = self.recorder.close()
        self.assertEqual(status["state"], "saved", status)
        metadata = json.loads(Path(status["metadata_path"]).read_text())
        self.assertIsNone(metadata["stops"][0]["request"])

    def test_all_producer_cycles_export_and_raw_prefix_is_unchanged(self):
        self.start()
        for index in range(160):
            at = 10. + index * .04
            self.recorder.record(at=at, phase="ACTIVE", scene=scene(at), reason="" if index % 3 else "event")
        rows, metadata = self.stop(now=17.)
        self.assertEqual(len(rows), 162)
        self.assertEqual(metadata["cycles"], 160)
        before = self.raw.read_bytes()
        destination = Path(self.temp.name) / "replay.csv"
        again = export_session(self.raw, destination, session_id=self.recorder.status()["session_id"])
        self.assertEqual(again["source_sha256"], metadata["source_sha256"])
        self.assertEqual(self.raw.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            export_session(self.raw, destination, session_id=self.recorder.status()["session_id"])

    def test_used_measurement_times_and_latest_rejected_frame_remain_distinct(self):
        self.start()
        control = {"at": 10., "state_at": 9.9, "measurement_times": {"ur5e": 9.9, "ur10e": 9.9},
                   "received_times": {"ur5e": 9.95, "ur10e": 9.95}, "robot_state_host_times": {"ur5e": 9.98, "ur10e": 9.99},
                   "robots": {"ur5e": {"robot_timestamp": 42., "qd": [.1] * 6}}}
        measurement = {"sequence": 15, "captured_at": 9.97, "received_at": 10., "accepted": False,
                       "pose_diagnostics": {"joint_alignment": {"robots": {"ur5e": {"q": [1.] * 6}}}}}
        self.recorder.record(at=10., phase="ACTIVE", scene=scene(), control_record=control, measurement=measurement)
        rows, _ = self.stop()
        self.assertEqual(rows[1]["ur5e_measurement_at"], "9.9")
        self.assertEqual(rows[1]["measurement_captured_at"], "9.97")
        self.assertEqual(rows[1]["ur5e_host_sample_at"], "9.98")
        self.assertIn('"accepted":false', rows[1]["measurement_json"])
        self.assertIn('"joint_alignment"', rows[1]["measurement_json"])

    def test_raw_logger_failure_propagates_and_latches_incomplete(self):
        self.start()
        with patch.object(self.logger, "log", side_effect=LoggingError("queue full")):
            with self.assertRaisesRegex(LoggingError, "queue full"):
                self.recorder.record(at=10., phase="ACTIVE", scene=scene())
        status = self.recorder.status()
        self.assertEqual(status["state"], "error")
        self.assertTrue(status["incomplete"])
        self.assertFalse(status["can_start"])

    def test_export_failure_visible_and_shutdown_timeout_cannot_later_claim_saved(self):
        self.start()
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(2.)
            return {"incomplete": False, "metadata_path": "ignored"}
        with patch("urcal.tools.export_evaluation.export_session", side_effect=blocked):
            self.recorder.request({"action": "stop", "request_id": "stop"}, now=11.)
            self.assertTrue(entered.wait(1.))
            status = self.recorder.close(timeout=.005)
            self.assertEqual(status["state"], "error")
            self.assertTrue(status["incomplete"])
            release.set()
            wait_until(lambda: not self.recorder.status()["busy"])
        self.assertEqual(self.recorder.status()["state"], "error")

    def test_export_io_failure_is_visible_and_does_not_claim_saved(self):
        self.start()
        with patch("urcal.tools.export_evaluation.export_session", side_effect=OSError("disk full")):
            self.recorder.request({"action": "stop", "request_id": "stop"}, now=11.)
            wait_until(lambda: not self.recorder.status()["busy"])
        status = self.recorder.status()
        self.assertEqual(status["state"], "error")
        self.assertTrue(status["incomplete"])
        self.assertIn("disk full", status["error"])
        self.assertTrue(status["can_start"])
        self.assertTrue(self.raw.exists())

    def test_stop_diagnostics_and_malformed_cycle_payload_are_retained(self):
        self.start()
        stop = {"operations": [{"stage": "speedStop", "returned": True, "command": {"acceleration": .4}}]}
        self.recorder.record(at=10., phase="shutdown", scene=None,
                             control_record={"stop_transport": stop}, transport={"operations": None})
        self.recorder.record(at=10., phase="FAULT", scene={"at": 10., "calibration_valid": True,
                             "measurement_age": .1, "robots": []}, control_record=["malformed"])
        rows, _ = self.stop()
        self.assertEqual(json.loads(rows[1]["stop_transport_json"]), stop)
        self.assertEqual(rows[1]["command_sent"], "0")
        self.assertEqual(rows[2]["control_record_json"], '["malformed"]')
        self.assertEqual(rows[2]["metric_valid"], "0")

    def test_scene_timing_metadata_is_retained_when_no_proposal_exists(self):
        self.start()
        supplied = scene()
        supplied.update(measurement_times={"ur5e": 9.9, "ur10e": 9.9},
                        received_times={"ur5e": 9.95, "ur10e": 9.95}, state_at=10.)
        self.recorder.record(at=10., phase="paused", scene=supplied)
        rows, _ = self.stop()
        self.assertEqual(rows[1]["ur5e_measurement_at"], "9.9")
        self.assertEqual(rows[1]["ur10e_received_at"], "9.95")
        self.assertEqual(rows[1]["state_at"], "10.0")

    def test_preflight_without_control_proposal_is_not_mislabelled_preview(self):
        self.start()
        self.recorder.record(at=10., phase="alignment_preflight", scene=None, reason="checking native IK")
        rows, _ = self.stop()
        self.assertEqual(rows[1]["preview"], "0")
        self.assertEqual(rows[1]["control_active"], "")

    def test_stale_per_robot_host_time_and_delayed_scene_ages_invalidate_metric(self):
        self.start()
        self.recorder.record(at=10., phase="ACTIVE", scene=scene(),
            control_record={"at": 10., "robot_state_host_times": {"ur5e": 9., "ur10e": 10.}})
        self.recorder.record(at=10.4, phase="ACTIVE", scene=scene(age=.2))
        rows, _ = self.stop()
        self.assertEqual([row["metric_valid"] for row in rows[1:-1]], ["0", "0"])
        self.assertIn("robot state is stale", rows[1]["metric_reason"])
        self.assertIn("calibration is stale", rows[2]["metric_reason"])


class PartialEvaluationExportTests(unittest.TestCase):
    def test_partial_and_corrupt_source_is_exported_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / "partial.jsonl", Path(directory) / "partial.csv"
            start = {"kind": "evaluation_start", "session_id": "s", "trial_id": "trial", "at": 1., "flags": {}}
            cycle = {"kind": "evaluation_cycle", "session_id": "s", "at": 2., "cycle": 0,
                     "phase": "FAULT", "metric_valid": False, "reason": "connection lost"}
            raw.write_text(json.dumps(start) + "\n" + json.dumps(cycle) + '\n{"kind":"evaluation_cycle",')
            original = raw.read_bytes()
            result = export_session(raw, output, session_id="s")
            self.assertTrue(result["incomplete"])
            self.assertEqual(len(result["parse_errors"]), 1)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_line"] for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[-1]["kind"], "evaluation_parse_error")
            self.assertTrue(all(row["export_incomplete"] == "1" for row in rows))
            self.assertEqual(raw.read_bytes(), original)

    def test_multiple_sessions_require_explicit_selection_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / "raw.jsonl", Path(directory) / "out.csv"
            raw.write_text("".join(json.dumps({"kind": kind, "session_id": session, "at": 1., "cycles": 0}) + "\n"
                for session in ("one", "two") for kind in ("evaluation_start", "evaluation_stop")))
            with self.assertRaisesRegex(ValueError, "session-id"):
                export_session(raw, output)
            self.assertFalse(export_session(raw, output, session_id="two")["incomplete"])
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                export_session(raw, output, session_id="one")
            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
