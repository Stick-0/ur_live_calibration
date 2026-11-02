"""Button-selected, full-producer-rate position-error CSV intervals.

This module has no robot or motion-command dependencies. The producer calls
request() in command order and record() for every error packet, before any GUI
throttling. Start/stop boundaries are when the producer processes the request,
not when a possibly delayed UI click occurred. Only a background writer opens,
writes and closes files. A recording failure is visible and never stops motion.

CSV values retain the measured error, including values above the graph's 50 mm
display ceiling. Invalid samples and state/clock changes are explicit rows with
blank numeric errors. Repeated valid timestamps do not fabricate new samples.
Flush is not fsync: process/power loss can still leave an incomplete file.
"""

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import uuid

from urcal.control.error_plot import PositionErrorHistory


FIELDS = (
    "record_type", "recording_id", "request_id", "created_at_utc",
    "emitted_at_monotonic_s", "recording_elapsed_wall_s", "sample_at_s",
    "sample_elapsed_s", "clock", "episode", "segment", "follower", "phase",
    "target_source", "valid", "position_error_mm", "error_x_mm", "error_y_mm",
    "error_z_mm", "reason",
)


class PositionErrorRecorder:
    """A bounded, nonblocking recorder with status instead of raised failures.

    Public producer methods are request(dict), record(sample, emitted_at=None),
    status(), and close(timeout=5). ``request_id`` acknowledgements are immediate
    when processed by the producer. ``saved`` means the writer flushed/closed
    the interval; ``error`` and ``incomplete`` never claim a complete recording.
    A new interval can start only when ``can_start`` is true.
    """

    def __init__(self, directory, *, max_pending=4096):
        self.directory = Path(directory)
        self._pending = Queue(maxsize=max_pending)
        self._lock = threading.RLock()
        self._closing = threading.Event()
        self._active = False
        self._busy = False
        self._failed_id = None
        self._seen_requests = []
        self._started_at = self._stopped_at = None
        self._identity = self._last_at = self._last_signature = self._sample_origin = None
        self._last_packet_key = None
        self._segment = 0
        self._status = {
            "state": "idle", "request_id": None, "recording_id": None,
            "path": None, "rows": 0, "valid_samples": 0, "invalid_samples": 0,
            "elapsed_s": 0., "error": "", "message": "Ready to record an error interval",
            "incomplete": False, "created_at_utc": None,
        }
        self._thread = threading.Thread(target=self._write, name="position-error-csv", daemon=True)
        try:
            if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
                raise ValueError("recording queue capacity must be a positive integer")
            self._thread.start()
        except Exception as exc:
            self._fail("Cannot start error recording writer: " + str(exc))

    def _fail(self, message, recording_id=None):
        with self._lock:
            if recording_id is not None and recording_id != self._status["recording_id"]:
                return
            self._failed_id = self._status["recording_id"]
            self._active = False
            if self._stopped_at is None:
                self._stopped_at = time.monotonic()
            self._status.update(state="error", error=str(message), message=str(message), incomplete=True)

    def _put(self, item):
        try:
            self._pending.put_nowait(item)
            return True
        except Full:
            self._fail("Error recording queue is full; CSV interval is incomplete")
            return False

    def _row(self, record_type, emitted_at, **values):
        row = dict.fromkeys(FIELDS, "")
        row.update(record_type=record_type, recording_id=self._status["recording_id"],
                   request_id=self._status["request_id"], created_at_utc=self._status["created_at_utc"],
                   emitted_at_monotonic_s=emitted_at,
                   recording_elapsed_wall_s=max(0., emitted_at - self._started_at),
                   segment=self._segment, **values)
        return row

    def request(self, command):
        """Process a start/stop request without filesystem work or motion IO."""
        try:
            if (not isinstance(command, dict) or command.get("action") not in ("start", "stop")
                    or not isinstance(command.get("request_id"), str) or not command["request_id"]):
                return False
            with self._lock:
                request_id = command["request_id"]
                if request_id in self._seen_requests:
                    return True
                self._seen_requests.append(request_id)
                del self._seen_requests[:-64]
                self._status["request_id"] = request_id
                action = command["action"]
                if action == "stop":
                    if not self._active:
                        return True
                    self._active = False
                    self._stopped_at = time.monotonic()
                    self._status.update(state="saving", message="Finishing CSV interval")
                    return self._put(("stop", self._status["recording_id"],
                                      self._row("stop", self._stopped_at, reason="user_stop")))
                if self._active:
                    return True
                if self._busy or self._closing.is_set() or not self._thread.is_alive():
                    self._status["message"] = "Wait for the previous CSV to finish before recording again"
                    return False
                started = time.monotonic()
                created = datetime.now(timezone.utc)
                recording_id = uuid.uuid4().hex
                path = self.directory / (created.strftime("%Y%m%dT%H%M%S.%fZ")
                                         + "-error-" + recording_id[:8] + ".csv")
                self._started_at, self._stopped_at = started, None
                self._identity = self._last_at = self._last_signature = self._sample_origin = None
                self._last_packet_key = None
                self._segment, self._failed_id = 0, None
                self._active, self._busy = True, True
                self._status.update(state="recording", recording_id=recording_id, path=str(path),
                    rows=0, valid_samples=0, invalid_samples=0, elapsed_s=0., error="", incomplete=False,
                    message="Recording error samples", created_at_utc=created.isoformat())
                opening = self._row("start", started)
                opening["_path"] = str(path)  # Immutable job metadata, removed before CSV serialization.
                return self._put(("start", recording_id, opening))
        except Exception as exc:
            self._fail("Error recording request failed: " + str(exc))
            return False

    def record(self, sample, emitted_at=None):
        """Snapshot one producer packet, preserving unique sample times/gaps.

        ``emitted_at`` is the producer host-monotonic timestamp. Sample times
        retain their own clock; ``sample_elapsed_s`` resets with clock/episode
        changes and simulation resets. It never advances while simulation time
        is paused. ``elapsed_s`` in status is host recording duration instead.
        """
        try:
            with self._lock:
                if not self._active:
                    return False
                emitted = time.monotonic() if emitted_at is None else float(emitted_at)
                if not math.isfinite(emitted):
                    raise ValueError("invalid error emission time")
                try:
                    data = PositionErrorHistory._sample(sample)
                except (ValueError, TypeError, OverflowError) as exc:
                    signature = ("malformed", str(exc))
                    if self._last_signature == signature:
                        return False
                    self._last_signature = signature
                    self._segment += 1
                    row = self._row("gap", emitted, valid=0, reason="Invalid error sample: " + str(exc))
                    return self._put(("row", self._status["recording_id"], row))
                identity = (data["clock"], data["episode"], data["follower"])
                signature = (data["valid"], data["phase"], data["target_source"], data["reason"])
                changed_clock = self._identity is not None and identity != self._identity
                rewound = self._last_at is not None and data["at"] < self._last_at
                if identity == self._identity and rewound and data["clock"] != "simulation":
                    data.update(valid=False, reason="out-of-order live sample timestamp")
                    signature = (False, data["phase"], data["target_source"], data["reason"])
                packet_key = (identity, data["at"], signature)
                if packet_key == self._last_packet_key:
                    return False
                self._last_packet_key = packet_key
                if self._identity != identity or (rewound and data["clock"] == "simulation"):
                    if self._identity is not None:
                        self._segment += 1
                        event = self._row("event", emitted, clock=data["clock"], episode=data["episode"],
                                          sample_at_s=data["at"], valid=0,
                                          reason="clock/episode/follower changed" if changed_clock else "simulation time reset")
                        if not self._put(("row", self._status["recording_id"], event)):
                            return False
                    self._identity, self._sample_origin = identity, data["at"]
                    self._last_at = self._last_signature = None
                same_time = data["at"] == self._last_at
                if same_time and signature == self._last_signature:
                    return False
                if self._last_signature is not None and signature != self._last_signature:
                    self._segment += 1
                kind = "sample" if data["valid"] else "gap"
                # Metadata may change on a paused sample. Record the event,
                # but do not count or repeat the old numerical error.
                if same_time and data["valid"]:
                    data = {**data, "valid": False, "reason": "metadata changed without a new sample"}
                    kind = "event"
                row = self._row(kind, emitted, sample_at_s=data["at"],
                    sample_elapsed_s=data["at"] - self._sample_origin,
                    clock=data["clock"], episode=data["episode"], follower=data["follower"],
                    phase=data["phase"], target_source=data["target_source"], valid=int(data["valid"]),
                    reason=data["reason"])
                if data["valid"]:
                    row.update(position_error_mm=data["norm_mm"], error_x_mm=data["error_mm"][0],
                               error_y_mm=data["error_mm"][1], error_z_mm=data["error_mm"][2])
                if not rewound or data["clock"] == "simulation" or changed_clock:
                    self._last_at = data["at"]
                self._last_signature = signature
                return self._put(("row", self._status["recording_id"], row))
        except Exception as exc:
            self._fail("Cannot record position error: " + str(exc))
            return False

    def status(self):
        with self._lock:
            result = self._status.copy()
            if self._started_at is not None:
                result["elapsed_s"] = max(0., (self._stopped_at if self._stopped_at is not None
                                                else time.monotonic()) - self._started_at)
            result.update(active=self._active, busy=self._busy,
                          can_start=not self._busy and not self._closing.is_set() and self._thread.is_alive(),
                          can_stop=self._active)
            return result

    def _write(self):
        handle = writer = None
        recording_id = None
        opening = None
        try:
            while not self._closing.is_set() or not self._pending.empty():
                try:
                    action, current_id, row = self._pending.get(timeout=.02)
                except Empty:
                    action = current_id = row = None
                try:
                    with self._lock:
                        failed_id = (self._failed_id if self._failed_id == self._status["recording_id"] else None)
                        failed_message = self._status["error"]
                    if failed_id is not None:
                        try:
                            if handle is not None and recording_id == failed_id:
                                failed_at = time.monotonic()
                                writer.writerow({**opening, "record_type": "incomplete", "valid": 0,
                                                 "emitted_at_monotonic_s": failed_at,
                                                 "recording_elapsed_wall_s": max(0., failed_at - opening["emitted_at_monotonic_s"]),
                                                 "reason": failed_message})
                                handle.flush()
                        finally:
                            if handle is not None and recording_id == failed_id:
                                handle.close()
                                handle = writer = None
                            with self._lock:
                                if self._status["recording_id"] == failed_id:
                                    self._busy = False
                    with self._lock:
                        if (action is None or current_id == self._failed_id
                                or current_id != self._status["recording_id"]):
                            continue
                    if action == "start":
                        recording_id = current_id
                        path = Path(row.pop("_path"))
                        opening = row.copy()
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handle = path.open("x", encoding="utf-8", newline="")
                        writer = csv.DictWriter(handle, fieldnames=FIELDS)
                        writer.writeheader()
                    if current_id != recording_id or writer is None:
                        continue
                    writer.writerow(row)
                    handle.flush()
                    if action == "row":
                        with self._lock:
                            if current_id == self._status["recording_id"]:
                                self._status["rows"] += 1
                                self._status["valid_samples"] += int(row["valid"] == 1)
                                self._status["invalid_samples"] += int(row["valid"] == 0)
                    if action == "stop":
                        handle.close()
                        handle = writer = None
                        with self._lock:
                            if current_id == self._status["recording_id"]:
                                self._busy = False
                                if current_id != self._failed_id and not self._status["incomplete"]:
                                    self._status.update(state="saved", message="CSV interval saved")
                except Exception as exc:
                    self._fail("Cannot write error CSV: " + str(exc), current_id or recording_id)
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception:
                            pass
                    handle = writer = None
                    with self._lock:
                        if current_id == self._status["recording_id"]:
                            self._busy = False
                finally:
                    if action is not None:
                        self._pending.task_done()
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    self._fail("Cannot close error CSV: " + str(exc), recording_id)
            with self._lock:
                self._busy = False

    def close(self, timeout=5.):
        """Auto-stop and drain on program shutdown; return final status."""
        try:
            with self._lock:
                if self._active:
                    self.request({"action": "stop", "request_id": "shutdown-" + uuid.uuid4().hex})
                self._closing.set()
            if self._thread.is_alive():
                self._thread.join(timeout=max(0., float(timeout)))
            if self._thread.is_alive():
                self._fail("Error CSV writer did not finish before shutdown; recording is incomplete")
        except Exception as exc:
            self._fail("Cannot finish error recording: " + str(exc))
        return self.status()
