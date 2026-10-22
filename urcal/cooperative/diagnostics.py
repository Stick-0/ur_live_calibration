"""Bounded WAIT diagnostics without robot IO or control-state changes."""

import math

import numpy as np


def _finite_copy(value):
    """Copy numerical diagnostic data to JSON values; invalid numbers are null."""
    if isinstance(value, np.ndarray):
        return _finite_copy(value.tolist())
    if isinstance(value, dict):
        return {key: _finite_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_copy(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _finite_copy(value.item())
    return value


def fault_diagnostics(exception, states=None):
    """Copy the failed tick's details and robot state, never cached old values."""
    result = {"details": _finite_copy(getattr(exception, "details", {}))}
    result["robots"] = None
    if isinstance(states, dict):
        result["robots"] = {
            name: _finite_copy({
                "q": getattr(state, "joints", None),
                "qd": getattr(state, "joint_velocity", None),
                "tcp": getattr(state, "tcp", None),
                "tcp_speed": getattr(state, "speed", None),
                "robot_timestamp": getattr(state, "robot_timestamp", None),
            }) for name, state in states.items() if name in ("ur5e", "ur10e")
        }
    return result


class WaitDiagnostics:
    """At most one WAIT per interval/category; category changes are immediate.

    Numeric errors and ages may change every control tick. That must not create
    a full-rate stream of identical fault categories. Call clear() after a
    successful tick so a new fault after recovery is recorded immediately.
    """

    def __init__(self, interval=1.):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("diagnostic interval must be finite and positive")
        self.clear()

    def clear(self):
        self._category = None
        self._at = None

    def record(self, now, exception, states=None):
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("diagnostic time must be finite")
        category = getattr(exception, "category", "control_fault")
        if (self._category == category and self._at is not None
                and 0 <= now - self._at < self.interval):
            return None
        record = {"kind": "wait", "at": now, "category": category,
                  "reason": str(exception), **fault_diagnostics(exception, states)}
        self._category, self._at = category, now
        return record
