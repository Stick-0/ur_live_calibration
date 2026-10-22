"""Capture and persist a pair of robot states without a control connection.

Only ``capture_initial_state`` opens hardware, and its RTDE interfaces are
receive-only. The camera/receivers are closed before it returns. Camera poses
are the existing tracker's filtered estimates. Encoder/TCP samples are fresh
at capture but are not hardware-synchronized to the camera exposure.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
import json
import multiprocessing as mp
from numbers import Integral
from pathlib import Path
import time

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base, pose_matrix
from urcal.control.policy import CalibrationTracker, ControlFault


SCHEMA = "urcal.simulation.initial.v1"
ROBOTS = ("ur5e", "ur10e")


def _number(value, label, minimum=0.):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum:g}")
    return value


def _transform(value, label):
    try:
        inverse(value)
        return np.asarray(value, dtype=float).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite rigid 4x4 transform") from exc


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain six finite numbers") from exc
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain six finite numbers")
    return result.copy()


def _pair(value, label):
    if not isinstance(value, dict) or set(value) != set(ROBOTS):
        raise ValueError(f"{label} requires exactly ur5e and ur10e")
    return value


def validate_initial_state(snapshot):
    """Normalize schema-v1 data and return independent arrays/dictionaries."""
    try:
        if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA:
            raise ValueError(f"initial state must use schema {SCHEMA}")
        captured_at = snapshot["captured_at_utc"]
        if not isinstance(captured_at, str):
            raise ValueError("captured_at_utc must be an ISO UTC timestamp")
        stamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("captured_at_utc must have a UTC timezone")
        calibration = snapshot["calibration"]
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be an object")
        sequence = calibration["sequence"]
        if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("calibration sequence must be a nonnegative integer")
        result = {
            "schema": SCHEMA,
            "captured_at_utc": stamp.isoformat().replace("+00:00", "Z"),
            "calibration": {
                "sequence": int(sequence),
                "captured_at_monotonic": _number(calibration["captured_at_monotonic"], "capture time"),
                "poses": {}, "raw_mean_px": {}, "raw_max_px": {}, "valid_counts": {},
            },
            "bases": {}, "robots": {},
        }
        for field in ("poses", "raw_mean_px", "raw_max_px", "valid_counts"):
            _pair(calibration[field], "calibration " + field)
        _pair(snapshot["bases"], "bases")
        _pair(snapshot["robots"], "robots")
        for name in ROBOTS:
            pose = _transform(calibration["poses"][name], name + " calibration")
            base = _transform(snapshot["bases"][name], name + " base")
            if not np.allclose(base, pose @ model_from_controller_base(name), atol=1e-8, rtol=0):
                raise ValueError(name + " base disagrees with model-to-controller mapping")
            mean = _number(calibration["raw_mean_px"][name], name + " raw mean")
            maximum = _number(calibration["raw_max_px"][name], name + " raw maximum")
            if maximum < mean:
                raise ValueError(name + " raw maximum must be >= mean")
            count = calibration["valid_counts"][name]
            if not isinstance(count, Integral) or isinstance(count, bool) or not 4 <= count <= 7:
                raise ValueError(name + " valid keypoint count must be an integer from 4 to 7")
            state = snapshot["robots"][name]
            if not isinstance(state, dict):
                raise ValueError(name + " state must be an object")
            source = state["tcp_offset_source"]
            if source not in ("explicit", "zero_assumed"):
                raise ValueError(name + " TCP offset source must be explicit or zero_assumed")
            offset = _transform(state["tcp_offset"], name + " TCP offset")
            if source == "zero_assumed" and not np.allclose(offset, np.eye(4), atol=1e-12, rtol=0):
                raise ValueError(name + " assumed-zero TCP offset must be identity")
            result["calibration"]["poses"][name] = pose
            result["calibration"]["raw_mean_px"][name] = mean
            result["calibration"]["raw_max_px"][name] = maximum
            result["calibration"]["valid_counts"][name] = int(count)
            result["bases"][name] = base
            result["robots"][name] = {
                "q": _vector(state["q"], name + " q"),
                "qd": _vector(state["qd"], name + " qd"),
                "speed": _vector(state["speed"], name + " TCP speed"),
                "tcp": _transform(state["tcp"], name + " TCP"),
                "tcp_offset": offset,
                "robot_timestamp": _number(state["robot_timestamp"], name + " robot timestamp"),
                "tcp_offset_source": source,
            }
        return result
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"invalid initial-state structure: {exc}") from exc


def save_initial_state(path, snapshot):
    """Write a validated JSON snapshot, refusing to overwrite an existing file."""
    value = validate_initial_state(snapshot)
    encoded = json.dumps(value, indent=2, allow_nan=False,
                         default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    return path


def load_initial_state(path):
    """Load a local snapshot; no camera, robot driver or network is accessed."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_initial_state(value)


def capture_initial_state(args, on_status=None, stop_event=None):
    """Wait for a stable camera pair and fresh encoder/TCP states, then close IO.

    ``stop_event`` is an optional caller-owned threading event. It is observed
    but never set here. The camera worker receives a separate multiprocessing
    event, so normal capture completion cannot stop a subsequent simulation.
    """
    if getattr(args, "execute", False):
        raise ValueError("initial-state capture is receive-only; --execute is not allowed")
    timeout = _number(getattr(args, "sim_initial_timeout", 60.), "initial capture timeout")
    if timeout <= 0:
        raise ValueError("initial capture timeout must be positive")
    if on_status is not None and not callable(on_status):
        raise TypeError("on_status must be callable")
    if stop_event is not None and stop_event.is_set():
        raise ControlFault("initial-state capture cancelled")
    offsets, offset_sources = {}, {}
    for name in ROBOTS:
        offset = getattr(args, name + "_tcp_offset", None)
        offsets[name] = pose_matrix(np.zeros(6) if offset is None else offset)
        offset_sources[name] = "zero_assumed" if offset is None else "explicit"
    tracker = CalibrationTracker(
        max_age=args.max_calibration_age, stable_frames=args.stable_frames, alpha=args.filter_alpha,
        max_jump_m=args.max_calibration_jump_m, max_jump_rad=np.deg2rad(args.max_calibration_jump_deg),
        max_mean_px=args.max_reproj_px, max_pixel_px=args.max_point_reproj_px,
        min_points=args.min_points, check_reprojection=args.check_reprojection,
    )
    # No control SDK is imported. Loading a saved snapshot never reaches here.
    from rtde_receive import RTDEReceiveInterface
    from urcal.control.runtime import StateReader, put_latest, take_latest
    from urcal.control.vision import vision_worker

    context = mp.get_context("spawn")
    queues = []
    worker = None
    worker_stop = context.Event()
    deadline = time.monotonic() + timeout
    last_report = float("-inf")
    accepted = None
    initial_timestamps = {}

    def report(message, now, *, force=False):
        nonlocal last_report
        if force or now - last_report >= 1.:
            print("[sim capture] " + message, flush=True)
            if on_status is not None:
                on_status(message)
            last_report = now

    try:
        for size in (1, 8, 1):
            queues.append(context.Queue(maxsize=size))
        frames, commands, statuses = queues
        worker = context.Process(target=vision_worker,
                                 args=(args, frames, commands, statuses, worker_stop), daemon=True)
        with ExitStack() as cleanup:
            readers = {}
            report("receive-only acquisition; no robot motion commands", time.monotonic(), force=True)
            for name in ROBOTS:
                if stop_event is not None and stop_event.is_set():
                    raise ControlFault("initial-state capture cancelled")
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                cleanup.callback(receiver.disconnect)
                readers[name] = StateReader(receiver, args.max_robot_age, require_joint_velocity=True)
            if "zero_assumed" in offset_sources.values():
                names = ", ".join(name for name in ROBOTS if offset_sources[name] == "zero_assumed")
                report(f"{names}: TCP offset omitted; recording zero assumption (not read from robot)",
                       time.monotonic(), force=True)
            worker.start()
            while True:
                now = time.monotonic()
                if stop_event is not None and stop_event.is_set():
                    raise ControlFault("initial-state capture cancelled")
                if worker_stop.is_set() or not worker.is_alive():
                    raise ControlFault("camera capture stopped before initial state was ready")
                if now >= deadline:
                    raise ControlFault(f"initial-state capture timed out after {timeout:g}s: "
                                       + (tracker.last_rejection or "no fresh stable calibration/robot pair"))
                try:
                    # Sample throughout camera/model startup, proving both robot
                    # clocks advance instead of accepting an initial cached read.
                    states = {name: reader.read(now) for name, reader in readers.items()}
                    for name in ROBOTS:
                        initial_timestamps.setdefault(name, states[name].robot_timestamp)
                    packet = take_latest(frames)
                    if packet is not None and tracker.update(packet, now):
                        accepted = packet
                    tracker.check_fresh(now)
                    if any(states[name].robot_timestamp <= initial_timestamps[name] for name in ROBOTS):
                        raise ControlFault("waiting for both robot timestamps to advance")
                    checked = time.monotonic()
                    tracker.check_fresh(checked)
                    if any(checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                        raise ControlFault("robot state expired during capture")
                    poses = tracker.poses
                    snapshot = validate_initial_state({
                        "schema": SCHEMA,
                        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "calibration": {
                            "sequence": accepted.sequence,
                            "captured_at_monotonic": accepted.captured_at,
                            "poses": poses,
                            "raw_mean_px": accepted.raw_mean_px,
                            "raw_max_px": accepted.raw_max_px,
                            "valid_counts": accepted.valid_counts,
                        },
                        "bases": {name: poses[name] @ model_from_controller_base(name) for name in ROBOTS},
                        "robots": {name: {
                            "q": states[name].joints, "qd": states[name].joint_velocity,
                            "speed": states[name].speed, "tcp": states[name].tcp,
                            "tcp_offset": offsets[name], "robot_timestamp": states[name].robot_timestamp,
                            "tcp_offset_source": offset_sources[name],
                        } for name in ROBOTS},
                    })
                    report("initial state captured; closing camera and robot receive connections", checked, force=True)
                    break
                except (ControlFault, ValueError) as exc:
                    message = "WAIT: " + str(exc)
                    put_latest(statuses, {"message": "SIM CAPTURE " + message, "goals": {}})
                    report(message, now)
                worker_stop.wait(.02)
    finally:
        worker_stop.set()
        # Queue teardown must run even if process shutdown raises. Never hand
        # a snapshot to a simulation while its camera worker is still alive.
        with ExitStack() as queue_cleanup:
            for queue in queues:
                queue_cleanup.callback(queue.close)
                queue_cleanup.callback(queue.cancel_join_thread)
            if worker is not None and worker.pid is not None:
                worker.join(timeout=3.)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=1.)
                if worker.is_alive():
                    raise ControlFault("camera worker did not exit; simulation will not start")
    if stop_event is not None and stop_event.is_set():
        raise ControlFault("initial-state capture cancelled")
    return snapshot
