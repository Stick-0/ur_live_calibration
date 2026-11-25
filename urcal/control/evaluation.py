"""Full producer-rate evaluation sessions in the append-only experiment JSONL.

Evaluation observes supplied snapshots only: it owns no robot connection or
motion command. A session freezes its nominal relative FLANGE target and full
settings. Metrics, proposal commands and successful SDK sends remain separate.
Raw log failures propagate LoggingError; CSV extraction failures are visible in
status and never masquerade as saved. Exporting never modifies the raw log.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose
from urcal.cooperative.logging import LoggingError


SCHEMA = "urcal-evaluation-v1"
ROBOTS = ("ur5e", "ur10e")


def _snapshot(value):
    """Preserve nonfinite raw evidence explicitly, without invalid JSON numbers."""
    if isinstance(value, np.ndarray):
        return _snapshot(value.tolist())
    if isinstance(value, np.generic):
        return _snapshot(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__nonfinite_float__": repr(value)}
    raise TypeError("unsupported evaluation value: " + type(value).__name__)


def _pose(value):
    result = np.asarray(value, dtype=float)
    matrix_pose(result)
    return result.copy()


def evaluation_flags(settings):
    """Configured mechanisms, not a claim that confidence or motion is nonzero."""
    mode = settings.get("base_motion_mode", "pose")
    method, law = settings.get("method", "separated"), settings.get("follow_law", "feedforward")
    kalman = mode == "relative_kalman"
    return {"kalman_state_estimation_enabled": kalman,
            "prediction_enabled": kalman or (mode == "velocity" and method in ("predict", "separated")),
            "table_compensation_enabled": mode in ("velocity", "relative_kalman")
                and method == "separated" and law == "feedforward"}


def _commands(control, transport, follower, simulated):
    """Only this transaction's successful SDK operations establish a send."""
    operations = transport.get("operations", []) if isinstance(transport, Mapping) else []
    operations = operations if isinstance(operations, (list, tuple)) else []
    attempts, sent = [], {}
    for operation in operations:
        if not isinstance(operation, Mapping) or operation.get("stage") not in ("speedL", "speedJ"):
            continue
        item = deepcopy(dict(operation))
        attempts.append(item)
        if (not simulated and operation.get("returned") is True
                and isinstance(operation.get("command"), Mapping)
                and operation.get("finished_at") is not None):
            sent[str(operation.get("robot", follower))] = item
    proposed = None
    if isinstance(control, Mapping) and ("twist" in control or "joint_command" in control):
        proposed = {"velocity": control.get("twist"), "joint_velocity": control.get("joint_command"),
                    "at": control.get("at"), "source": "controller_proposal"}
    return {"command_attempted": bool(attempts) and not simulated,
            "command_sent": bool(sent), "sent_commands": sent,
            "command_attempts": attempts, "proposed_command": proposed,
            "transport_failed": bool(transport and (transport.get("failed_at") is not None
                                                     or transport.get("error"))),
            "simulated_command_applied": bool(simulated and transport
                                                and transport.get("simulated_command_applied", False)),
            "simulated_applied_commands": deepcopy(transport.get("commands", {}))
                if simulated and transport and transport.get("simulated_command_applied", False) else {}}


def _metric(at, scene, control, target, leader, follower, settings):
    result = {"metric_valid": False, "robot_state_valid": False, "calibration_valid": False,
              "metric_reason": "", "metric_source": "unavailable", "metric_at": None,
              "actual_relative_pose": None, "position_error_m": None,
              "position_error_norm_m": None, "rotation_error_rad": None,
              "rotation_error_norm_rad": None, "G_control": None}
    try:
        now = float(at)
        if not math.isfinite(now):
            raise ValueError("invalid producer timestamp")
        source = control if isinstance(control, Mapping) and control.get("estimated_relative_pose") is not None else scene
        if not isinstance(source, Mapping):
            raise ValueError("paired robot state and calibration unavailable")
        source_at = float(source["at"])
        result.update(metric_source="control_record" if source is control else "scene", metric_at=source_at)
        max_robot_age = float(settings.get("max_robot_age", .5))
        state_stamps = [source_at]
        if isinstance(control, Mapping) and isinstance(control.get("robot_state_host_times"), Mapping):
            state_stamps.extend(float(stamp) for stamp in control["robot_state_host_times"].values())
        if isinstance(scene, Mapping):
            state_stamps.append(float(scene["at"]))
            for robot in scene.get("robots", {}).values():
                if isinstance(robot, Mapping):
                    for key in ("at", "host_sample_at", "sample_at", "sampled_at"):
                        if robot.get(key) is not None:
                            state_stamps.append(float(robot[key]))
                            break
        if not all(math.isfinite(stamp) and 0 <= now - stamp <= max_robot_age for stamp in state_stamps):
            raise ValueError("robot state is stale or has an invalid timestamp")
        result["robot_state_valid"] = True
        if isinstance(scene, Mapping) and not scene.get("calibration_valid", False):
            raise ValueError(str(scene.get("reason") or "calibration is invalid"))
        times = control.get("measurement_times") if isinstance(control, Mapping) else None
        if not isinstance(times, Mapping) and isinstance(scene, Mapping):
            times = scene.get("measurement_times")
        if isinstance(times, Mapping) and all(name in times for name in (leader, follower)):
            ages = [now - float(times[name]) for name in (leader, follower)]
        else:
            age_source = scene if isinstance(scene, Mapping) else control
            ages = age_source.get("measurement_age")
            if isinstance(ages, Mapping) and not all(name in ages for name in (leader, follower)):
                raise ValueError("calibration ages do not contain both robots")
            ages = [ages[name] for name in (leader, follower)] if isinstance(ages, Mapping) else [ages]
            ages = [float(value) for value in ages]
            age_at = float(age_source.get("evaluated_at", age_source.get("at", source_at)))
            ages = [value + max(0., now - age_at) for value in ages]
        if (not ages or not np.isfinite(ages).all() or min(ages) < 0
                or max(ages) > float(settings.get("max_calibration_age", .5))):
            raise ValueError("calibration is stale or has an invalid age")
        result["calibration_valid"] = True
        bases = control.get("bases") if isinstance(control, Mapping) else None
        if not isinstance(bases, Mapping) and isinstance(scene, Mapping):
            bases = scene.get("bases")
        if isinstance(bases, Mapping):
            result["G_control"] = inverse(_pose(bases[follower])) @ _pose(bases[leader])
        if source is control:
            actual = _pose(control["estimated_relative_pose"])
        else:
            robots = scene["robots"]
            flanges = {name: _pose(bases[name]) @ _pose(robots[name]["tcp"])
                       @ inverse(_pose(robots[name]["tcp_offset"])) for name in (leader, follower)}
            actual = inverse(flanges[leader]) @ flanges[follower]
        target = _pose(target)
        dp = target[:3, 3] - actual[:3, 3]
        dr = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        result.update(metric_valid=True, actual_relative_pose=actual, position_error_m=dp,
                      position_error_norm_m=float(np.linalg.norm(dp)), rotation_error_rad=dr,
                      rotation_error_norm_rad=float(np.linalg.norm(dr)))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, np.linalg.LinAlgError) as exc:
        result["metric_reason"] = str(exc)
    return result


class EvaluationRecorder:
    """One producer owns request/record; background work only extracts the CSV.

    Every record() call during an active session produces a row, even repeated
    times, pauses and failures. request_id is idempotent for this recorder's
    lifetime. close() closes the recorder, not the caller-owned raw logger.
    """

    def __init__(self, logger, raw_path, directory, *, settings, leader, follower, clock="monotonic"):
        if set((leader, follower)) != set(ROBOTS):
            raise ValueError("evaluation requires distinct ur5e/ur10e leader and follower")
        settings = vars(settings) if not isinstance(settings, Mapping) else settings
        self.logger, self.raw_path, self.directory = logger, Path(raw_path), Path(directory)
        self.settings = _snapshot(dict(settings))
        self.leader, self.follower, self.clock = leader, follower, str(clock)
        self._lock = threading.RLock()
        self._seen_requests = {}
        self._closed = self._active = self._fatal = False
        self._thread = None
        self._target = self._start = self._last_at = self._episode = None
        self._sequence = 0
        self._status = {"state": "idle", "session_id": None, "trial_id": None, "request_id": None,
                        "raw_path": str(self.raw_path), "csv_path": None, "metadata_path": None,
                        "path": None, "incomplete": False, "error": "", "message": "Ready for evaluation",
                        "table_moving": False, "cycles": 0, "valid_samples": 0, "invalid_samples": 0,
                        "elapsed_s": 0.}

    def _fail(self, error, *, fatal=False):
        self._active = False
        self._fatal = self._fatal or fatal
        self._status.update(state="error", incomplete=True, error=str(error), message=str(error))

    def _emit(self, record):
        try:
            self.logger.log(_snapshot(record))
        except Exception as exc:
            failure = exc if isinstance(exc, LoggingError) else LoggingError("evaluation raw record failed: " + str(exc))
            self._fail(failure, fatal=True)
            raise failure

    def _event(self, kind, at, **extra):
        return {"kind": kind, "schema": SCHEMA, "session_id": self._status["session_id"],
                "trial_id": self._status["trial_id"], "at": at, "clock": self.clock,
                "episode": self._episode, "leader": self.leader, "follower": self.follower, **extra}

    def request(self, command, *, now, target=None, episode=None):
        with self._lock:
            if not isinstance(command, Mapping):
                return False
            request_id, action = command.get("request_id"), command.get("action")
            if not isinstance(request_id, str) or not request_id:
                return False
            if request_id in self._seen_requests:
                return self._seen_requests[request_id]
            self._status["request_id"] = request_id
            self._seen_requests[request_id] = False
            try:
                now = float(now)
                if not math.isfinite(now):
                    raise ValueError("evaluation request time must be finite")
                if command.get("type", "evaluation") != "evaluation" or action not in ("start", "stop", "table_start", "table_stop"):
                    raise ValueError("unknown evaluation action")
                if self._closed or self._fatal:
                    raise ValueError("evaluation recorder is closed or raw logging failed")
                if action == "start":
                    if self._active or (self._thread is not None and self._thread.is_alive()):
                        raise ValueError("finish the previous evaluation before starting another")
                    frozen = _pose(target)
                    trial = command.get("trial_id", "")
                    if not isinstance(trial, str) or len(trial) > 128 or any(ord(c) < 32 for c in trial):
                        raise ValueError("trial_id must be a label of at most 128 characters without controls")
                    created = datetime.now(timezone.utc)
                    session_id = uuid.uuid4().hex
                    trial = trial.strip() or "trial-" + created.strftime("%Y%m%dT%H%M%S")
                    path = self.directory / (created.strftime("%Y%m%dT%H%M%S.%fZ") + "-evaluation-" + session_id + ".csv")
                    self._target, self._start, self._last_at, self._episode = frozen, now, now, episode
                    self._sequence, self._active = 0, True
                    self._status.update(state="recording", session_id=session_id, trial_id=trial,
                        csv_path=str(path), path=str(path), metadata_path=str(path.with_suffix(".metadata.json")),
                        table_moving=False, cycles=0, valid_samples=0, invalid_samples=0, elapsed_s=0.,
                        incomplete=False, error="", message="Recording evaluation", created_at_utc=created.isoformat())
                    self._emit(self._event("evaluation_start", now, request_id=request_id,
                        created_at_utc=created.isoformat(), settings=self.settings,
                        runtime_context=command.get("runtime_context"), frozen_target=self._target, target_frame="leader_flange", target_kind="nominal_relative_flange",
                        raw_path=str(self.raw_path), flags=evaluation_flags(self.settings), request=dict(command)))
                elif action == "stop":
                    if self._active:
                        self._stop(now, str(command.get("reason", "user_stop")), request_id, episode,
                                   request=dict(command))
                else:
                    if not self._active:
                        raise ValueError("table markers require an active evaluation")
                    previous = self._status["table_moving"]
                    self._status["table_moving"] = action == "table_start"
                    self._emit(self._event("evaluation_marker", now, action=action, request_id=request_id,
                        table_moving=self._status["table_moving"], previous_table_moving=previous,
                        episode=self._episode if episode is None else episode, request=dict(command), reason=str(command.get("reason", ""))))
                self._seen_requests[request_id] = True
                return True
            except LoggingError:
                raise
            except (ValueError, TypeError, OverflowError) as exc:
                self._status.update(error=str(exc), message=str(exc))
                if not self._active and not (self._thread and self._thread.is_alive()):
                    self._status["state"] = "error"
                return False

    def record(self, *, at, phase, scene, control_record=None, transport=None, measurement=None, reason="", episode=None):
        with self._lock:
            if not self._active:
                return False
            control = control_record if isinstance(control_record, Mapping) else {}
            trans = transport if isinstance(transport, Mapping) else {}
            simulated = self.clock == "simulation" or trans.get("simulated") is True
            metrics = _metric(at, scene, control, self._target, self.leader, self.follower, self.settings)
            try:
                when = float(at)
            except (TypeError, ValueError, OverflowError):
                when = None
            if when is not None and math.isfinite(when):
                if self._last_at is not None and when < self._last_at:
                    metrics.update(metric_valid=False, metric_reason="producer time moved backwards")
                self._last_at = when
                self._status["elapsed_s"] = max(0., when - self._start)
            if episode is not None and self._episode is not None and episode != self._episode:
                metrics.update(metric_valid=False, metric_reason="episode changed during evaluation")
            if not metrics["metric_valid"]:
                for key in ("position_error_m", "position_error_norm_m", "rotation_error_rad", "rotation_error_norm_rad"):
                    metrics[key] = None
            # Preserve the actually used geometry even when the metric is stale.
            try:
                bases = control.get("bases")
                if not isinstance(bases, Mapping) and isinstance(scene, Mapping):
                    bases = scene.get("bases")
                if isinstance(bases, Mapping):
                    metrics["G_control"] = inverse(_pose(bases[self.follower])) @ _pose(bases[self.leader])
                elif isinstance(control.get("relative_kalman"), Mapping):
                    metrics["G_control"] = _pose(control["relative_kalman"]["G_control"])
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
                pass
            commands = _commands(control, trans, self.follower, simulated)
            source_scene = scene if isinstance(scene, Mapping) else {}
            timings = {key: control.get(key) if control.get(key) is not None else source_scene.get(key)
                       for key in ("state_at", "measurement_times", "received_times", "measurement_age")}
            record = self._event("evaluation_cycle", at, episode=self._episode if episode is None else episode, cycle=self._sequence,
                phase=str(phase), reason=str(reason), table_moving=self._status["table_moving"],
                simulated=simulated,
                preview=not simulated and not commands["command_sent"] and (
                    control.get("active") is False if "active" in control else
                    str(phase).lower() in ("preview", "ready", "waiting", "wait", "recovery_preview")),
                control_active=bool(control["active"]) if "active" in control else None, flags=evaluation_flags(self.settings),
                frozen_target=self._target, metric_frame="leader_flange", **metrics, **commands,
                **timings,
                source_links={"control_at": control.get("at"), "scene_at": scene.get("at") if isinstance(scene, Mapping) else None,
                              "latest_received_measurement": measurement,
                              "used_measurement_times": timings["measurement_times"],
                              "robot_state_host_times": control.get("robot_state_host_times")},
                control_record=control_record, transport=transport, measurement=measurement, scene=scene)
            self._emit(record)
            self._sequence += 1
            self._status["cycles"] += 1
            self._status["valid_samples" if metrics["metric_valid"] else "invalid_samples"] += 1
            return True

    def _stop(self, at, reason, request_id=None, episode=None, *, request=None):
        self._emit(self._event("evaluation_stop", at, request_id=request_id, reason=reason,
                   episode=self._episode if episode is None else episode, cycles=self._sequence,
                   table_moving=self._status["table_moving"], incomplete=False, request=request))
        self._active = False
        self._status.update(state="saving", elapsed_s=max(0., float(at) - self._start),
                            message="Extracting evaluation CSV from raw JSONL")
        session_id, path = self._status["session_id"], self._status["csv_path"]
        self._thread = threading.Thread(target=self._export, args=(session_id, path),
                                        name="evaluation-csv-export", daemon=True)
        try:
            self._thread.start()
        except RuntimeError as exc:
            self._thread = None
            self._fail("Cannot start evaluation CSV export: " + str(exc))

    def _export(self, session_id, path):
        try:
            from urcal.tools.export_evaluation import export_session
            result = export_session(self.raw_path, path, session_id=session_id, wait_for_stop=4.)
            with self._lock:
                if self._status["session_id"] != session_id or self._status["state"] == "error":
                    return
                if result["incomplete"]:
                    self._fail("Evaluation export is incomplete: " + "; ".join(result["issues"]))
                else:
                    self._status.update(state="saved", incomplete=False, error="", message="Evaluation CSV saved",
                                        metadata_path=result["metadata_path"])
        except Exception as exc:
            with self._lock:
                if self._status["session_id"] == session_id:
                    self._fail("Evaluation CSV export failed: " + str(exc))

    def status(self):
        with self._lock:
            busy = self._active or bool(self._thread and self._thread.is_alive())
            return {**deepcopy(self._status), "active": self._active, "busy": busy,
                    "can_start": not busy and not self._closed and not self._fatal,
                    "can_stop": self._active}

    def close(self, reason="shutdown", timeout=5.):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("evaluation close timeout must be finite and positive")
        with self._lock:
            if self._active:
                at = time.monotonic() if self.clock == "monotonic" else self._last_at
                self._stop(at, str(reason))
            self._closed = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                with self._lock:
                    self._fail("Evaluation export did not finish before shutdown deadline")
        return self.status()
