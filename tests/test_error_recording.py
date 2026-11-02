"""CSV interval semantics, producer completeness, and isolated writer faults."""

import csv
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from urcal.control.error_recording import PositionErrorRecorder


def sample(at=10., *, valid=True, **extra):
    return {"at": at, "valid": valid, "error_m": [.003, -.004, .012],
            "clock": "monotonic", "episode": 0, "follower": "ur5e", "phase": "tracking",
            "target_source": "tracking_accepted_relative", "reason": "", **extra}


def wait_until(predicate, timeout=2.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError("writer condition did not complete")


class BlockedFile(io.StringIO):
    """Block the first write to deterministically fill the producer queue."""

    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()
        self.saved = None

    def write(self, value):
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(3.)
        return super().write(value)

    def close(self):
        if not self.closed:
            self.saved = self.getvalue()
        super().close()


class PositionErrorRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "error_segments"
        self.recorder = PositionErrorRecorder(self.directory)
        self.addCleanup(self.recorder.close)

    def start(self, recorder=None, request_id="start-1"):
        recorder = recorder or self.recorder
        self.assertTrue(recorder.request({"action": "start", "request_id": request_id}))
        return Path(recorder.status()["path"])

    def finish(self, recorder=None, request_id="stop-1"):
        recorder = recorder or self.recorder
        self.assertTrue(recorder.request({"action": "stop", "request_id": request_id}))
        wait_until(lambda: recorder.status()["state"] in ("saved", "error"))
        self.assertEqual(recorder.status()["state"], "saved", recorder.status())
        with Path(recorder.status()["path"]).open(newline="") as source:
            return list(csv.DictReader(source))

    def test_records_full_interval_beyond_visible_history_and_preserves_large_errors(self):
        path = self.start()
        for index in range(1000):
            self.assertTrue(self.recorder.record(sample(10. + .04 * index, error_m=[.06, .08, 0.])))
        rows = self.finish()
        points = [row for row in rows if row["record_type"] == "sample"]
        self.assertEqual(len(points), 1000)
        self.assertEqual(float(points[-1]["sample_elapsed_s"]), 39.96)
        self.assertEqual(float(points[0]["position_error_mm"]), 100.)
        self.assertEqual(float(points[0]["error_x_mm"]), 60.)
        self.assertEqual([rows[0]["record_type"], rows[-1]["record_type"]], ["start", "stop"])
        self.assertEqual(self.recorder.status()["valid_samples"], 1000)
        self.assertIn("Z-error-", path.name)
        self.assertFalse(self.recorder.status()["incomplete"])

    def test_no_io_or_samples_before_start_and_packet_is_snapshotted(self):
        self.assertFalse(self.directory.exists())
        self.assertFalse(self.recorder.record(sample()))
        self.start()
        value = sample()
        self.recorder.record(value)
        value["error_m"][0] = 1000.
        self.recorder.record(sample(10.1))
        rows = self.finish()
        self.assertEqual(float(rows[1]["position_error_mm"]), 13.)
        self.assertFalse(self.recorder.record(sample(10.2)))

    def test_duplicate_sample_is_not_recounted_but_changed_metadata_is_explicit(self):
        self.start()
        self.assertTrue(self.recorder.record(sample(0., clock="simulation")))
        self.assertFalse(self.recorder.record(sample(0., clock="simulation", error_m=[.1, 0., 0.])))
        self.assertTrue(self.recorder.record(sample(0., clock="simulation", phase="paused")))
        self.assertFalse(self.recorder.record(sample(0., clock="simulation", phase="paused")))
        rows = self.finish()
        self.assertEqual(sum(row["valid"] == "1" for row in rows), 1)
        event = next(row for row in rows if row["record_type"] == "event")
        self.assertEqual(event["position_error_mm"], "")
        self.assertEqual(event["phase"], "paused")
        self.assertEqual(float(event["sample_elapsed_s"]), 0.)

    def test_invalid_gap_and_repeated_invalid_time_have_no_fabricated_error(self):
        self.start()
        self.recorder.record(sample(10.))
        self.assertTrue(self.recorder.record(sample(10.1, valid=False, reason="calibration unavailable")))
        self.assertFalse(self.recorder.record(sample(10.1, valid=False, reason="calibration unavailable")))
        self.recorder.record(sample(10.2))
        rows = self.finish()
        gap = next(row for row in rows if row["record_type"] == "gap")
        self.assertEqual(gap["position_error_mm"], "")
        self.assertEqual(gap["error_x_mm"], "")
        self.assertEqual(gap["reason"], "calibration unavailable")
        self.assertEqual(self.recorder.status()["valid_samples"], 2)
        self.assertNotEqual(rows[1]["segment"], rows[-2]["segment"])

    def test_simulation_reset_and_clock_change_have_separate_elapsed_origins(self):
        self.start()
        self.recorder.record(sample(5., clock="simulation"))
        self.recorder.record(sample(6., clock="simulation"))
        self.recorder.record(sample(0., clock="simulation", episode=1))
        self.recorder.record(sample(.1, clock="simulation", episode=1))
        self.recorder.record(sample(20., clock="monotonic", episode=1))
        rows = self.finish()
        events = [row for row in rows if row["record_type"] == "event"]
        self.assertEqual(len(events), 2)
        points = [row for row in rows if row["record_type"] == "sample"]
        self.assertEqual([float(row["sample_elapsed_s"]) for row in points], [0., 1., 0., .1, 0.])
        self.assertEqual(len({row["segment"] for row in points}), 3)

    def test_live_clock_reversal_is_gap_not_a_new_measurement(self):
        self.start()
        self.recorder.record(sample(10.))
        self.assertTrue(self.recorder.record(sample(9.)))
        self.assertFalse(self.recorder.record(sample(9.)))
        self.recorder.record(sample(10.1))
        rows = self.finish()
        self.assertEqual(sum(row["valid"] == "1" for row in rows), 2)
        self.assertIn("out-of-order", rows[2]["reason"])

    def test_multiple_intervals_unique_and_duplicate_requests_idempotent(self):
        first_path = self.start()
        first_id = self.recorder.status()["recording_id"]
        self.assertTrue(self.recorder.request({"action": "start", "request_id": "start-1"}))
        self.assertEqual(self.recorder.status()["recording_id"], first_id)
        self.recorder.record(sample())
        self.finish()
        first_bytes = first_path.read_bytes()
        self.assertTrue(self.recorder.request({"action": "start", "request_id": "start-1"}))
        self.assertEqual(self.recorder.status()["state"], "saved")
        next_path = self.start(request_id="start-2")
        self.assertNotEqual(next_path, first_path)
        self.recorder.record(sample(20.))
        self.finish(request_id="stop-2")
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(self.recorder.status()["valid_samples"], 1)

    def test_shutdown_finalizes_active_interval_without_manual_stop(self):
        path = self.start()
        self.recorder.record(sample())
        result = self.recorder.close()
        self.assertEqual(result["state"], "saved")
        self.assertFalse(result["can_start"])
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[-1]["record_type"], "stop")
        self.assertFalse(self.recorder.request({"action": "start", "request_id": "after-close"}))

    def test_open_failure_is_visible_and_retry_does_not_overwrite_failed_path(self):
        with patch.object(Path, "open", side_effect=OSError("disk unavailable")):
            failed_path = self.start()
            self.recorder.record(sample())
            wait_until(lambda: self.recorder.status()["state"] == "error" and self.recorder.status()["can_start"])
        self.assertTrue(self.recorder.status()["incomplete"])
        self.assertIn("disk unavailable", self.recorder.status()["error"])
        self.assertFalse(self.recorder.record(sample(10.2)))
        path = self.start(request_id="retry")
        self.recorder.record(sample(20.))
        rows = self.finish()
        self.assertNotEqual(path, failed_path)
        self.assertEqual(sum(row["valid"] == "1" for row in rows), 1)
        self.assertFalse(failed_path.exists())

    def test_queue_overflow_is_explicit_incomplete_and_next_interval_is_clean(self):
        blocked = BlockedFile()
        recorder = PositionErrorRecorder(self.directory, max_pending=2)
        self.addCleanup(recorder.close)
        real_open = Path.open
        opens = []
        def open_once(path, *args, **kwargs):
            if not args or args[0] != "x":
                return real_open(path, *args, **kwargs)
            opens.append(path)
            return blocked if len(opens) == 1 else real_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_once):
            failed_path = self.start(recorder)
            self.assertTrue(blocked.entered.wait(1.))
            self.assertTrue(recorder.record(sample(10.)))
            self.assertTrue(recorder.record(sample(10.1)))
            self.assertFalse(recorder.record(sample(10.2)))
            self.assertEqual(recorder.status()["state"], "error")
            self.assertTrue(recorder.status()["incomplete"])
            blocked.release.set()
            wait_until(lambda: recorder.status()["can_start"])
            next_path = self.start(recorder, "retry-overflow")
            self.assertNotEqual(next_path, failed_path)
            wait_until(lambda: len(opens) == 2)
            recorder.record(sample(20.))
            rows = self.finish(recorder)
        self.assertIn("incomplete", blocked.saved)
        self.assertNotIn("\r\nstop,", blocked.saved)
        self.assertEqual([float(row["sample_at_s"]) for row in rows if row["valid"] == "1"], [20.])
        self.assertEqual(opens, [failed_path, next_path])

    def test_shutdown_timeout_cannot_later_turn_incomplete_into_saved(self):
        blocked = BlockedFile()
        recorder = PositionErrorRecorder(self.directory)
        self.addCleanup(recorder.close)
        with patch.object(Path, "open", return_value=blocked):
            self.start(recorder)
            self.assertTrue(blocked.entered.wait(1.))
            recorder.record(sample())
            result = recorder.close(timeout=.005)
            self.assertEqual(result["state"], "error")
            self.assertTrue(result["incomplete"])
            blocked.release.set()
            wait_until(lambda: not recorder._thread.is_alive())
        self.assertEqual(recorder.status()["state"], "error")
        self.assertIn("incomplete", blocked.saved)
        self.assertNotIn("\r\nstop,", blocked.saved)

    def test_bad_packet_is_explicit_gap_and_does_not_raise(self):
        self.start()
        self.assertTrue(self.recorder.record({"at": 10., "valid": True, "error_m": [float("nan"), 0., 0.]}))
        self.assertFalse(self.recorder.record({"at": 10., "valid": True, "error_m": [float("nan"), 0., 0.]}))
        self.recorder.record(sample(10.1))
        rows = self.finish()
        self.assertEqual(rows[1]["record_type"], "gap")
        self.assertIn("Invalid error sample", rows[1]["reason"])
        self.assertEqual(rows[1]["position_error_mm"], "")


if __name__ == "__main__":
    unittest.main()
