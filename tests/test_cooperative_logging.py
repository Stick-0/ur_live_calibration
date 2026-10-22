"""Experiment logging and reference analysis checks; no robot/network access."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.tools.analyze_log import summarize_log
from urcal.cooperative.logging import ExperimentLogger, LoggingError
from urcal.control.geometry import pose_matrix


class ExperimentLoggerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "trial.jsonl"

    def test_numpy_snapshot_roundtrip_and_timestamp_preservation(self):
        pose = np.eye(4)
        record = {"kind": "measurement", "captured_at": np.float64(10.),
                  "received_at": 10.2, "sequence": np.int64(9), "pose": pose,
                  "valid": np.bool_(True), "source": Path("weights.pt")}
        logger = ExperimentLogger(self.path)
        logger.log(record)
        pose[0, 3] = 123.
        logger.log({"kind": "stop", "at": 11., "reason": "operator"})
        logger.close()
        logger.close()
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["pose"], np.eye(4).tolist())
        self.assertEqual(records[0]["captured_at"], 10.)
        self.assertEqual(records[0]["received_at"], 10.2)
        self.assertNotIn("at", records[0])
        self.assertEqual(records[0]["sequence"], 9)
        self.assertIs(records[0]["valid"], True)
        self.assertEqual(records[0]["source"], "weights.pt")
        with self.assertRaises(LoggingError):
            logger.log({"kind": "after_close"})

    def test_existing_file_is_preserved(self):
        self.path.parent.mkdir()
        self.path.write_text("original\n")
        with self.assertRaises(LoggingError):
            ExperimentLogger(self.path)
        self.assertEqual(self.path.read_text(), "original\n")

    def test_nonfinite_or_unserializable_record_latches_failure(self):
        for value in (np.nan, np.inf, object()):
            with self.subTest(value=value):
                logger = ExperimentLogger(self.path.parent / f"bad-{id(value)}.jsonl")
                with self.assertRaises(LoggingError):
                    logger.log({"kind": "bad", "value": value})
                with self.assertRaises(LoggingError):
                    logger.log({"kind": "good"})
                with self.assertRaises(LoggingError):
                    logger.close()

    def test_writer_error_is_reported_on_log_and_close(self):
        logger = ExperimentLogger(self.path)
        logger._file.close()
        failed = threading.Event()

        def fail_write(_line):
            failed.set()
            raise OSError("disk full")

        logger._file = Mock(write=fail_write)
        # The asynchronous failure can be seen by this call or the next call.
        try:
            logger.log({"kind": "control", "at": 1.})
        except LoggingError:
            pass
        self.assertTrue(failed.wait(1.))
        logger._thread.join(1.)
        with self.assertRaisesRegex(LoggingError, "disk full"):
            logger.log({"kind": "control", "at": 2.})
        with self.assertRaisesRegex(LoggingError, "disk full"):
            logger.close()

    def _block_writer(self, logger):
        entered, release = threading.Event(), threading.Event()
        real_file = logger._file

        class BlockedFile:
            def write(self, line):
                entered.set()
                if not release.wait(2.):
                    raise RuntimeError("test writer was not released")
                return real_file.write(line)

            def flush(self):
                return real_file.flush()

            def close(self):
                return real_file.close()

        logger._file = BlockedFile()
        self.addCleanup(release.set)
        return entered, release

    def test_queue_overflow_fails_immediately_and_drains_already_accepted_records(self):
        logger = ExperimentLogger(self.path, max_pending=1)
        entered, release = self._block_writer(logger)
        try:
            logger.log({"kind": "control", "at": 1.})
            self.assertTrue(entered.wait(1.))
            logger.log({"kind": "control", "at": 2.})
            started = time.monotonic()
            with self.assertRaisesRegex(LoggingError, "queue is full"):
                logger.log({"kind": "control", "at": 3.})
            self.assertLess(time.monotonic() - started, .2)
        finally:
            release.set()
            with self.assertRaises(LoggingError):
                logger.close()
        self.assertEqual([json.loads(line)["at"] for line in self.path.read_text().splitlines()], [1., 2.])

    def test_close_deadline_does_not_wait_forever_for_stalled_io(self):
        logger = ExperimentLogger(self.path)
        entered, release = self._block_writer(logger)
        try:
            logger.log({"kind": "control"})
            self.assertTrue(entered.wait(1.))
            started = time.monotonic()
            with self.assertRaisesRegex(LoggingError, "deadline"):
                logger.close(timeout=.02)
            self.assertLess(time.monotonic() - started, .3)
        finally:
            release.set()
            with self.assertRaises(LoggingError):
                logger.close()

    def test_pending_records_are_all_flushed_on_close(self):
        with ExperimentLogger(self.path, max_pending=256) as logger:
            for index in range(100):
                logger.log({"kind": "measurement", "sequence": index})
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([record["sequence"] for record in records], list(range(100)))


class LogAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trial.jsonl"
        self.reference = Path(self.directory.name) / "reference.jsonl"

    def write(self, records, path=None):
        path = self.path if path is None else path
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def control(self, at, **extra):
        record = {"kind": "control", "at": at, "active": True,
                  "progress": 0., "task_scale": 1.,
                  "desired_relative_pose": [0, 0, .3, 0, 0, 0],
                  "estimated_relative_pose": [0, 0, .3, 0, 0, 0]}
        record.update(extra)
        return record

    def test_missing_reference_never_labels_estimated_error_as_actual(self):
        self.write([self.control(1., estimated_relative_pose=[.01, 0, .3, 0, 0, np.deg2rad(5.)]),
                    self.control(2.)])
        report = summarize_log(self.path)
        self.assertIsNone(report["independent_real_error"])
        errors = report["estimated_relative_error"]
        self.assertAlmostEqual(errors["position_m"]["mean"], .005)
        self.assertAlmostEqual(errors["position_m"]["rms"], .01 / np.sqrt(2))
        self.assertAlmostEqual(errors["angle_deg"]["max"], 5.)

    def test_reference_uses_desired_pose_and_nearest_timestamp_with_tolerance(self):
        self.write([self.control(10.), self.control(11.)])
        actual = pose_matrix([.03, 0, .3, np.deg2rad(10.), 0, 0]).tolist()
        self.write([{"at": 11.05, "relative_pose": actual},
                    {"at": 9.99, "relative_pose": actual}], self.reference)
        report = summarize_log(self.path, self.reference, .03)
        self.assertEqual(report["estimated_relative_error"]["position_m"]["max"], 0.)
        reference = report["independent_real_error"]
        self.assertEqual(reference["matched_control_records"], 1)
        self.assertEqual(reference["unmatched_control_records"], 1)
        self.assertAlmostEqual(reference["position_m"]["max"], .03)
        self.assertAlmostEqual(reference["angle_deg"]["max"], 10.)
        self.assertAlmostEqual(reference["matching_skew_s"]["max"], .01)

    def test_supplied_reference_with_no_matches_has_no_fabricated_error(self):
        self.write([self.control(10.)])
        self.write([{"at": 50., "relative_pose": [0, 0, .3, 0, 0, 0]}], self.reference)
        report = summarize_log(self.path, self.reference)
        self.assertEqual(report["independent_real_error"]["matched_control_records"], 0)
        self.assertIsNone(report["independent_real_error"]["position_m"]["rms"])

    def test_frozen_phase_and_hold_at_progress_one_are_not_task_completion(self):
        for progress in (.5, 1.):
            with self.subTest(progress=progress):
                self.write([self.control(1., progress=progress, task_scale=0., compensation_twist=[.01, 0, 0, 0, 0, 0]),
                            self.control(2., progress=progress, task_scale=0.)])
                report = summarize_log(self.path)
                self.assertFalse(report["task_completed"])
                self.assertIsNone(report["completion_time_s"])
                self.assertEqual(report["progress"]["coverage"], 0.)
                if progress < 1.:
                    self.assertEqual(report["slowdown_ratio"], 1.)
                    self.assertEqual(report["maintenance_time_s"], 0.)
                else:
                    self.assertIsNone(report["slowdown_ratio"])
                    self.assertEqual(report["maintenance_time_s"], 1.)
                self.assertEqual(report["elapsed_active_time_s"], 1.)

    def test_completion_time_and_slowdown_distinguish_progress_from_active_duration(self):
        self.write([self.control(100.),
                    self.control(101., progress=.25, task_scale=.5),
                    self.control(102., active=False, progress=.5),
                    self.control(103., progress=.5),
                    self.control(104., progress=1., finished=True),
                    self.control(105., active=False, progress=1.)])
        report = summarize_log(self.path)
        self.assertTrue(report["task_completed"])
        self.assertEqual(report["completion_time_s"], 4.)
        self.assertEqual(report["elapsed_active_time_s"], 4.)
        self.assertEqual(report["active_time_until_completion_s"], 3.)
        self.assertAlmostEqual(report["slowdown_ratio"], 1. / 3.)
        self.assertAlmostEqual(report["mean_task_scale"], 2.5 / 3.)
        self.assertEqual(report["path_active_time_s"], 3.)
        self.assertEqual(report["maintenance_time_s"], 1.)
        self.assertEqual(report["progress"]["coverage"], 1.)

    def test_phase_completion_endpoint_dwell_and_maintenance_have_separate_metrics(self):
        self.write([self.control(100., path="line", path_duration=.8),
                    self.control(101., path="line", progress=1., path_finished=True, task_scale=0., finished=False),
                    self.control(102., path="line", progress=1., path_finished=True, task_scale=0., finished=True,
                                 settle_elapsed=.5, within_finish_tolerance=True),
                    self.control(104., active=False, path="line", progress=1., task_scale=0.)])
        report = summarize_log(self.path)
        self.assertEqual(report["nominal_path_duration_s"], .8)
        self.assertEqual(report["phase_completion_time_s"], 1.)
        self.assertEqual(report["completion_time_s"], 2.)
        self.assertEqual(report["maintenance_time_s"], 3.)
        self.assertEqual(report["path_active_time_s"], 1.)
        self.assertEqual(report["slowdown_ratio"], 0.)
        self.assertEqual(report["mean_task_scale"], 1.)
        self.assertIn("estimated endpoint", report["completion_basis"])

    def test_explicit_hold_is_maintenance_not_slow_task_or_completed_work(self):
        self.write([self.control(1., path="hold", progress=1., finished=True, task_scale=0.),
                    self.control(3., path="hold", progress=1., finished=True, task_scale=0.)])
        report = summarize_log(self.path)
        self.assertFalse(report["phase_completed"])
        self.assertFalse(report["task_completed"])
        self.assertIsNone(report["phase_completion_time_s"])
        self.assertIsNone(report["completion_time_s"])
        self.assertEqual(report["maintenance_time_s"], 2.)
        self.assertEqual(report["path_active_time_s"], 0.)
        self.assertIsNone(report["slowdown_ratio"])

    def test_finished_flag_before_progress_one_does_not_claim_completion(self):
        self.write([self.control(1., progress=.9, finished=True), self.control(2., progress=.9)])
        self.assertIsNone(summarize_log(self.path)["completion_time_s"])

    def test_preview_errors_and_completion_are_excluded(self):
        self.write([self.control(1., active=False, progress=1., finished=True,
                                 estimated_relative_pose=[5, 0, 0, 0, 0, 0]), self.control(2.)])
        report = summarize_log(self.path)
        self.assertFalse(report["task_completed"])
        self.assertEqual(report["estimated_relative_error"]["position_m"]["count"], 1)
        self.assertEqual(report["estimated_relative_error"]["position_m"]["max"], 0.)

    def test_fault_categories_and_record_counts_are_retained(self):
        self.write([{"kind": "metadata", "mode": "cooperative"}, self.control(1.),
                    {"kind": "fault", "at": 2., "category": "measurement", "message": "stale"},
                    {"kind": "stop", "at": 3., "fault_category": "infeasible", "reason": "compensation limit"},
                    {"kind": "stop", "at": 4., "reason": "operator"}])
        report = summarize_log(self.path)
        self.assertEqual(report["records"], 5)
        self.assertEqual(report["record_counts"]["stop"], 2)
        self.assertEqual([fault["category"] for fault in report["faults"]], ["measurement", "infeasible"])

    def test_recoverable_motion_stop_survives_a_later_normal_operator_exit(self):
        self.write([self.control(1., path="hold"),
                    {"kind": "motion_stopped", "at": 2., "category": "measurement_expired",
                     "reason": "calibration stale", "new_arm_required": True},
                    {"kind": "wait", "at": 5., "reason": "collecting fresh calibration"},
                    {"kind": "end", "at": 10., "reason": "operator_stop"}])
        report = summarize_log(self.path)
        self.assertEqual(report["faults"], [])
        self.assertEqual(report["recoverable_motion_stop_count"], 1)
        self.assertEqual(report["recoverable_motion_stops"], [{
            "at": 2., "category": "measurement_expired", "message": "calibration stale"}])
        self.assertEqual(report["active_control_records"], 1)
        self.assertEqual(report["estimated_relative_error"]["position_m"]["count"], 1)
        self.assertEqual(report["elapsed_active_time_s"], 1.)
        self.assertEqual(report["maintenance_time_s"], 1.)

    def test_recovery_wait_is_excluded_from_active_duration_after_explicit_rearm(self):
        self.write([self.control(1., path="line", task_scale=.5),
                    {"kind": "motion_stopped", "at": 2., "category": "tracking_error", "reason": "target error"},
                    self.control(10., active=False, path="line", task_scale=0.,
                                 estimated_relative_pose=[5, 0, 0, 0, 0, 0]),
                    self.control(12., path="line", progress=.5, task_scale=1.),
                    self.control(14., path="line", progress=1., finished=True),
                    {"kind": "motion_stopped", "at": 15., "category": "measurement_expired", "reason": "stale"},
                    {"kind": "end", "at": 30., "reason": "operator_stop"}])
        report = summarize_log(self.path)
        self.assertEqual(report["recoverable_motion_stop_count"], 2)
        self.assertEqual(report["elapsed_active_time_s"], 4.)
        self.assertEqual(report["active_time_until_completion_s"], 3.)
        self.assertEqual(report["completion_time_s"], 13.)  # Explicitly wall time.
        self.assertEqual(report["path_active_time_s"], 3.)
        self.assertEqual(report["maintenance_time_s"], 1.)
        self.assertAlmostEqual(report["slowdown_ratio"], 1. / 3.)
        self.assertAlmostEqual(report["mean_task_scale"], 2.5 / 3.)
        self.assertEqual(report["estimated_relative_error"]["position_m"]["count"], 3)
        self.assertEqual(report["estimated_relative_error"]["position_m"]["max"], 0.)

    def test_recovery_stop_timestamp_must_be_finite_and_ordered(self):
        for at in (None, float("nan"), .5):
            with self.subTest(at=at):
                self.write([self.control(1.), {"kind": "motion_stopped", "at": at}])
                with self.assertRaises(ValueError):
                    summarize_log(self.path)

    def test_backwards_control_clock_invalid_transform_and_partial_json_fail(self):
        cases = ([self.control(2.), self.control(1.)],
                 [self.control(1., desired_relative_pose=np.zeros((4, 4)).tolist())],
                 [self.control(1., task_scale=2.)])
        for records in cases:
            with self.subTest(records=records):
                self.write(records)
                with self.assertRaises(ValueError):
                    summarize_log(self.path)
        self.path.write_text('{"kind": "control"')
        with self.assertRaisesRegex(ValueError, "incomplete log"):
            summarize_log(self.path)

    def test_reference_cannot_be_the_same_file(self):
        self.write([self.control(1.)])
        with self.assertRaisesRegex(ValueError, "separate"):
            summarize_log(self.path, self.path)

    def test_duplicate_reference_timestamps_are_rejected(self):
        self.write([self.control(1.)])
        self.write([{"at": 1., "relative_pose": [0, 0, .3, 0, 0, 0]}] * 2, self.reference)
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_log(self.path, self.reference)


if __name__ == "__main__":
    unittest.main()
