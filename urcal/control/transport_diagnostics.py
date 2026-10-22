"""Best-effort snapshots of locally cached RTDE transport/status getters.

These helpers accept already-created interfaces. They never connect, restart a
program, call the Dashboard server, or submit an RTDE command. Control and
receive maintain separate caches, so their snapshots need not agree during a
state transition and cannot by themselves establish why a program stopped.
"""

import math
from numbers import Integral, Real


_CONTROL_GETTERS = (
    ("connected", "isConnected"),
    ("program_running", "isProgramRunning"),
    ("robot_status", "getRobotStatus"),
    ("protective_stopped", "isProtectiveStopped"),
    ("emergency_stopped", "isEmergencyStopped"),
)

_RECEIVE_GETTERS = (
    ("connected", "isConnected"),
    ("robot_timestamp", "getTimestamp"),
    ("robot_mode", "getRobotMode"),
    ("safety_mode", "getSafetyMode"),
    ("runtime_state", "getRuntimeState"),
    ("safety_status_bits", "getSafetyStatusBits"),
    ("robot_status", "getRobotStatus"),
)


def _scalar(value):
    """Copy a numeric scalar into a strict-JSON builtin, retaining no SDK data."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if math.isfinite(result):
            return result
    raise ValueError("expected a boolean, integer, finite float or null")


def _error_text(exc):
    # A broken SDK error's __str__ must not replace the original motion fault.
    try:
        message = str(exc)
    except Exception:
        message = "error message unavailable"
    return f"{type(exc).__name__}: {message}"


def _snapshot(interface, at, getters):
    errors = {}
    result = {"at": None}
    try:
        result["at"] = _scalar(at)
    except Exception as exc:
        errors["at"] = _error_text(exc)
    for field, method in getters:
        result[field] = None
        try:
            getter = getattr(interface, method, None)
            if not callable(getter):
                errors[field] = f"unavailable: {method}"
                continue
            result[field] = _scalar(getter())
        except Exception as exc:
            errors[field] = _error_text(exc)
    result["getter_errors"] = errors
    return result


def control_snapshot(control, at):
    """Read only control-interface cache/connection flags, without commands.

    Optional getters absent in an installed SDK return null and a field-labelled
    ``getter_errors`` entry. No SDK is imported or interface constructed here.
    """
    return _snapshot(control, at, _CONTROL_GETTERS)


def receive_snapshot(receiver, at):
    """Read only receive-interface cache/connection flags, without commands."""
    return _snapshot(receiver, at, _RECEIVE_GETTERS)


def capture_runtime_fault(exception, session, readers, at):
    """Retain pre-cleanup status without delaying shutdown for network queries.

    Call before disconnect/stop callbacks run. Existing send-level failure data
    takes precedence; the control and receive interfaces have separate caches.
    Diagnostic failures must never replace the exception that caused shutdown.
    """
    try:
        details = getattr(exception, "details", {})
        details = dict(details) if isinstance(details, dict) else {}
        details.setdefault("detected_at", _scalar(at))
        exception.details = details
        capture = getattr(session, "capture_fault", None)
        if callable(capture):
            try:
                capture(exception, at=at)
            except Exception as exc:
                details["transport_snapshot_error"] = _error_text(exc)
        # A session may replace details while copying its transport snapshot.
        details = exception.details
        if "receive_status" not in details:
            snapshots = {}
            for name, reader in readers.items():
                try:
                    snapshots[name] = receive_snapshot(reader.receiver, at)
                except Exception as exc:
                    snapshots[name] = {"at": _scalar(at), "snapshot_error": _error_text(exc)}
            details["receive_status"] = snapshots
    except BaseException:
        # Already unwinding a fault: even an interruption while collecting
        # advisory status must not replace it or bypass the pending cleanup.
        pass


def transport_record(session):
    """Return optional session timing data without changing a send result."""
    try:
        snapshot = getattr(session, "transport_diagnostics", None)
        result = snapshot() if callable(snapshot) else None
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def fault_transport_summary(exception):
    """Format retained status after shutdown; this function never reads SDKs."""
    try:
        details = getattr(exception, "details", {})
        transport = details.get("transport", {})
        receive = details.get("receive_status", {})
        parts = []
        for name, control in transport.get("control_status", {}).items():
            state = receive.get(name, {})
            age = transport.get("seconds_since_last_successful_send", {}).get(name)
            age_text = "unknown" if age is None else f"{float(age) * 1000:.1f}ms"
            parts.append(
                f"{name}: connected={control.get('connected')}, "
                f"program_running={control.get('program_running')}, "
                f"safety_mode={state.get('safety_mode')}, runtime_state={state.get('runtime_state')}, "
                f"last_speed_send_age={age_text}")
        return "; ".join(parts)
    except Exception:
        return ""
