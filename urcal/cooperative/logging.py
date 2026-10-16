"""Nonblocking, bounded JSONL recording for cooperative-control experiments.

The caller supplies every timestamp: ``at`` is host monotonic control/receipt
time; measurements additionally retain ``captured_at`` and ``received_at``.
This class never substitutes write time for measurement time. NumPy values are
snapshotted when log() is called, before the writer thread can observe changes.
Files are created exclusively, never appended to or overwritten. A queue
overflow or write failure latches LoggingError; the controller must stop the
experiment instead of silently losing records. close() drains pending records
within its deadline. Flush is not fsync and does not promise crash durability.
"""

import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading

import numpy as np

from urcal.control.policy import ControlFault


class LoggingError(ControlFault):
    """The experiment record is incomplete or cannot be written."""


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported log value: {type(value).__name__}")


class ExperimentLogger:
    """Write JSON records in order without performing disk I/O in log().

    ``log(record)`` accepts a dictionary with a nonempty ``kind`` string. No
    timestamp is added automatically. Queue capacity counts pending records,
    excluding the record currently being written. A latched failure is raised
    by subsequent log() calls and close(). The writer is a daemon so an OS-level
    stalled write cannot keep the application alive beyond close's deadline.
    """

    def __init__(self, path, max_pending=4096):
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        self.path = Path(path)
        self._pending = Queue(maxsize=max_pending)
        self._closing = threading.Event()
        self._closed = False
        self._failure = None
        self._failure_lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise LoggingError(f"cannot create experiment log {self.path}: {exc}") from exc
        self._thread = threading.Thread(target=self._write, name="experiment-log-writer", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._file.close()
            raise

    def _fail(self, message, cause=None):
        with self._failure_lock:
            if self._failure is None:
                self._failure = LoggingError(message)
                self._failure.__cause__ = cause
        return self._failure

    def _check_failure(self):
        if self._failure is not None:
            raise self._failure

    def log(self, record):
        self._check_failure()
        if self._closing.is_set():
            raise LoggingError("experiment logger is closing or closed")
        try:
            if not isinstance(record, dict) or not isinstance(record.get("kind"), str) or not record["kind"]:
                raise ValueError("log record needs a nonempty kind string")
            line = json.dumps(record, default=_json_default, allow_nan=False, separators=(",", ":")) + "\n"
        except (TypeError, ValueError, OverflowError) as exc:
            raise self._fail(f"cannot serialize experiment record: {exc}", exc)
        try:
            self._pending.put_nowait(line)
        except Full as exc:
            raise self._fail("experiment log queue is full; a record could not be recorded", exc)
        self._check_failure()

    def _write(self):
        try:
            while True:
                try:
                    line = self._pending.get(timeout=.05)
                except Empty:
                    if self._closing.is_set():
                        break
                    continue
                try:
                    self._file.write(line)
                    self._file.flush()
                finally:
                    self._pending.task_done()
        except Exception as exc:
            self._fail(f"experiment log writer failed: {exc}", exc)
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self._fail(f"experiment log close failed: {exc}", exc)

    def close(self, timeout=5.):
        """Drain the queue, close the file, and report any error within timeout."""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("close timeout must be finite and positive")
        self._closing.set()
        if not self._closed:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise self._fail("experiment log writer did not finish before the close deadline")
            self._closed = True
        self._check_failure()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
