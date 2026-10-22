"""Live-calibrated flange facing: fixed camera pose or read-only leader/follower.

Preview never constructs RTDEControlInterface. Motion needs --execute AND Arm
in the GUI (or 'a' in the preview). No hardware connection occurs on import. The controller
is a bounded Cartesian proportional regulator, not a collision-aware planner.
UR API: https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html
"""

import argparse
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time

import numpy as np

from urcal.control.geometry import (
    facing_follower, fixed_facing_targets, flange_from_tcp, inverse, matrix_pose,
    model_flange_pose, model_from_controller_base, pose_matrix, rotation_distance,
)
from urcal.control.policy import ActiveCalibrationGuard, CalibrationTracker, ControlFault, PlanningUnavailable, PreviewCalibrationRecovery, bounded_twist
from urcal.control.transport_diagnostics import capture_runtime_fault, fault_transport_summary


ROBOTS = ("ur5e", "ur10e")
from urcal.paths import BUNDLE


def put_latest(queue, item):
    """A full queue may drop an update; it must never block the motion loop."""
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass


def take_latest(queue):
    latest = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            return latest


class CommandInbox:
    """Preserve edits/Stop while invalidating Arm clicks from a failed attempt."""

    def __init__(self, queue):
        self.queue = queue
        self.pending = []
        self.arm_after = -float("inf")

    def read(self):
        while True:
            if self.pending:
                command = self.pending.pop(0)
            else:
                try:
                    command = self.queue.get_nowait()
                except Empty:
                    return
            # Producers timestamp clicks before IPC: an old Arm can arrive
            # after queue draining because multiprocessing uses feeder threads.
            if isinstance(command, tuple) and len(command) == 2 and command[0] == "arm":
                if np.isfinite(command[1]) and command[1] > self.arm_after:
                    yield "arm"
            else:
                yield command

    def cancel_arm_requests(self):
        self.arm_after = time.monotonic()
        self.pending = [command for command in self.read() if command != "arm"]


def make_gui_scene(now, bases, states, offsets, goals, measurement_age, readers,
                   previous_timestamps, max_robot_age, *, calibration_valid=True, reason="",
                   calibration_mode="live", calibration_captured_at=None,
                   nominal_goals=None, command_goals=None):
    """Copy display data without changing readiness or refreshing cached joints.

    ``at`` is the older of the two host times at which RTDE timestamps last
    advanced. Calibration ages retain their capture-time basis. None means no
    new complete scene; the GUI may retain its previous picture as stale.
    """
    try:
        timestamps = {name: float(states[name].robot_timestamp) for name in ROBOTS}
        advances = {name: float(readers[name].last_advance) for name in ROBOTS}
        if not np.isfinite([now, *timestamps.values(), *advances.values()]).all():
            return None
        if any(now < advances[name] or now - advances[name] > max_robot_age
               or (name in previous_timestamps and timestamps[name] <= previous_timestamps[name])
               for name in ROBOTS):
            return None

        def transform(value):
            inverse(value)  # Strict finite SE(3) validation, without changing it.
            return np.asarray(value, dtype=float).copy()

        age = ({name: float(measurement_age[name]) for name in ROBOTS}
               if isinstance(measurement_age, dict) else float(measurement_age))
        age_values = list(age.values()) if isinstance(age, dict) else [age]
        if not np.isfinite(age_values).all() or min(age_values) < 0:
            return None
        robots = {}
        for name in ROBOTS:
            q = np.asarray(states[name].joints, dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                return None
            robots[name] = {"q": q.copy(), "tcp": transform(states[name].tcp),
                            "tcp_offset": transform(offsets[name])}
        scene = {"at": min(advances.values()),
                 "bases": {name: transform(bases[name]) for name in ROBOTS},
                 "robots": robots, "goals": {name: transform(pose) for name, pose in goals.items()},
                 "measurement_age": age, "calibration_valid": bool(calibration_valid),
                 "reason": str(reason), "calibration_mode": str(calibration_mode),
                 "calibration_captured_at": calibration_captured_at}
        if nominal_goals is not None:
            scene["nominal_goals"] = {name: transform(pose) for name, pose in nominal_goals.items()}
        if command_goals is not None:
            scene["command_goals"] = {name: transform(pose) for name, pose in command_goals.items()}
        previous_timestamps.update(timestamps)
        return scene
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None


def task_tolerance_options(args, offset):
    """Keep saved configurations without the new settings in exact-pose mode."""
    return {"position_tolerance_m": getattr(args, "target_tolerance_m", 0.),
            "rotation_tolerance_rad": np.deg2rad(getattr(args, "target_tolerance_deg", 0.)),
            "tcp_offset": offset}


def near_term_tcp_targets(states, twists, dt):
    """Targets for native checks of the actual bounded Cartesian command.

    The full nominal task pose may be unreachable while a pose inside the
    accepted task tolerance is reachable. Check the measured TCP and its next
    commanded increment; the nominal goal remains unchanged for task errors.
    """
    result = {}
    for name, twist in twists.items():
        current = states[name].tcp
        step = pose_matrix(np.concatenate((np.zeros(3), np.asarray(twist)[3:] * dt)))
        target = current.copy()
        target[:3, :3] = step[:3, :3] @ current[:3, :3]
        target[:3, 3] += np.asarray(twist)[:3] * dt
        result[name] = target
    return result


def facing_gui_scene(now, tracker, planner, states, readers, previous_timestamps, max_robot_age, *,
                     active=False, command_tcp_goals=None):
    """Render accepted calibration and intended goals even before motion is feasible."""
    try:
        age = now - tracker.last_capture
        if age < 0 or age > tracker.max_age:
            return None
        calibration = tracker.poses
        bases = {name: calibration[name] @ planner.mappings[name] for name in ROBOTS}
        flanges = {name: bases[name] @ flange_from_tcp(states[name].tcp, planner.offsets[name])
                   for name in ROBOTS}
        calibration_valid = not tracker.last_rejection if active else tracker.ready
        reason = "" if calibration_valid else (tracker.last_rejection or "last accepted calibration; not ready")
        try:
            if planner.args.mode == "hold":
                goals = planner.hold_targets
                if goals is None:
                    goals = fixed_facing_targets(flanges["ur5e"], flanges["ur10e"],
                                                 planner.gap, planner.args.center_pose)
            else:
                goals = {planner.commanded[0]: facing_follower(flanges[planner.args.leader], planner.gap)}
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            goals = {}
            reason = (reason + "; " if reason else "") + "target preview unavailable: " + str(exc)
        return make_gui_scene(now, bases, states, planner.offsets, goals, age, readers,
                              previous_timestamps, max_robot_age, calibration_valid=calibration_valid,
                              reason=reason, command_goals=None if command_tcp_goals is None else {
                                  name: bases[name] @ flange_from_tcp(pose, planner.offsets[name])
                                  for name, pose in command_tcp_goals.items()})
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


@dataclass
class RobotState:
    tcp: np.ndarray
    joints: np.ndarray
    speed: np.ndarray
    joint_velocity: np.ndarray = None
    robot_timestamp: float = None


class StateReader:
    """Track receive freshness using robot timestamps, not host read frequency."""

    def __init__(self, receiver, max_age=0.15, require_joint_velocity=False):
        self.receiver = receiver
        self.max_age = max_age
        self.last_timestamp = None
        self.last_advance = None
        self.require_joint_velocity = require_joint_velocity

    def read(self, now):
        r = self.receiver
        if not r.isConnected():
            raise ControlFault("RTDE receive disconnected")
        timestamp = float(r.getTimestamp())
        if not np.isfinite(timestamp):
            raise ControlFault("nonfinite RTDE timestamp")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ControlFault("robot timestamp moved backwards")
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp, self.last_advance = timestamp, now
        if now - self.last_advance > self.max_age:
            raise ControlFault("RTDE state stopped updating")
        if r.getSafetyMode() not in (1, 2) or r.getRobotMode() != 7:
            raise ControlFault("robot is not running in normal/reduced safety mode")
        q = np.asarray(r.getActualQ(), dtype=float)
        speed = np.asarray(r.getActualTCPSpeed(), dtype=float)
        if q.shape != (6,) or speed.shape != (6,) or not np.isfinite([q, speed]).all():
            raise ControlFault("invalid RTDE joints/speed")
        qd = None
        if self.require_joint_velocity:
            qd = np.asarray(r.getActualQd(), dtype=float)
            if qd.shape != (6,) or not np.isfinite(qd).all():
                raise ControlFault("invalid RTDE joint velocity")
        return RobotState(pose_matrix(r.getActualTCPPose()), q, speed, qd, timestamp)


class ArmConnectionError(ControlFault):
    """Remote mode rejected a constructor before any script was uploaded."""

    category = "arm_connection"

    def __init__(self, robot, ip):
        self.details = {"robot": robot, "ip": ip, "reason": "remote_control_required"}
        super().__init__(f"{robot.upper()} ({ip}): Remote Control required. "
                         "Enable Remote Control in pendant Settings > System, then switch the "
                         "Local/Remote profile to Remote. Press Arm again after switching.")


class ArmIKUnavailable(ControlFault):
    """An explicit native IK rejection during stationary, pre-motion checks."""

    category = "arm_ik"

    def __init__(self, robot, target, seed, tcp_offset):
        target = np.asarray(target, dtype=float).copy()
        pose = matrix_pose(target)
        self.details = {"robot": robot, "reason": "native_ik_unavailable",
                        "target_tcp": target, "target_pose_base": pose.copy(),
                        "seed_q_rad": np.asarray(seed, dtype=float).copy(),
                        "tcp_offset": np.asarray(tcp_offset, dtype=float).copy(),
                        "position_tolerance_m": .0005, "rotation_tolerance_rad": .002}
        xyz = ", ".join(f"{value:.4f}" for value in pose[:3])
        rotation = ", ".join(f"{value:.4f}" for value in pose[3:])
        super().__init__(f"{robot}: controller reports no IK solution for alignment target "
                         f"(TCP position AND orientation: base XYZ=[{xyz}]m, rotvec=[{rotation}]rad). "
                         "This does not establish position-only unreachability. No motion started; "
                         "review calibration and target pose, then press Arm again.")


class ArmConditioningUnavailable(ControlFault):
    """Bounded stationary preflight exhausted usable native IK candidates.

    This exception is recoverable only while preparing an explicitly armed
    alignment, before watchdog activation or any motion command. Retain all
    candidate diagnostics so a preview refusal can be inspected afterwards.
    """

    category = "arm_conditioning"

    def __init__(self, details):
        self.details = deepcopy(details)
        distance = float(self.details.get("position_tolerance_m", 0.))
        angle = float(self.details.get("rotation_tolerance_deg", 0.))
        count = len(self.details.get("attempts", ()))
        super().__init__(
            f"No suitable controller IK within {distance * 1000:g}mm / {angle:g}deg "
            f"flange allowance after {count} candidate checks. No motion started; "
            "review target, then press Arm again.")


class MotionSession:
    """Own only commanded arms; all RTDEControl calls are serialized here."""

    def __init__(self, args, names, offsets, factory=None, stop_event=None, defer_watchdog=False):
        self.args, self.offsets = args, offsets
        self.controls = {}
        self._watchdog_enabled = False
        self._velocity_sent = False
        self._motion_modes = {}
        self.last_ik_tcp = {}
        self._transport_current = None
        self._last_transport = None
        self._last_successful_send = {}
        if factory is None:
            # Importing/constructing this interface is confined to explicit arm.
            from rtde_control import RTDEControlInterface
            # Default FLAG_UPLOAD_SCRIPT would re-upload on SDK reconnect and
            # could restart a watchdog-stopped program with an old speed packet.
            # Upload once explicitly below; reconnect must never start a script.
            factory = lambda ip: RTDEControlInterface(ip, flags=RTDEControlInterface.FLAG_NO_WAIT)
        try:
            for name in names:
                if stop_event is not None and stop_event.is_set():
                    raise ControlFault("operator stop during controller startup")
                ip = getattr(args, name + "_ip")
                try:
                    self.controls[name] = factory(ip)
                except RuntimeError as exc:
                    if "please enable remote control on the robot" not in str(exc).lower():
                        raise
                    # All constructors precede all script uploads. With
                    # FLAG_NO_WAIT alone these connections own no script yet;
                    # disconnect without stopping another controller program.
                    controls, self.controls = self.controls, {}
                    for connected_name, control in controls.items():
                        try:
                            control.disconnect()
                        except Exception as cleanup_exc:
                            print(f"[warn] {connected_name} disconnect: {cleanup_exc}", flush=True)
                    raise ArmConnectionError(name, ip) from exc
            for name, control in self.controls.items():
                if stop_event is not None and stop_event.is_set():
                    raise ControlFault("operator stop during controller startup")
                if not control.reuploadScript():
                    raise ControlFault(f"{name}: failed to start control script")
                deadline = time.monotonic() + 5.
                while not control.isProgramRunning():
                    if stop_event is not None and stop_event.is_set():
                        raise ControlFault("operator stop during controller startup")
                    if time.monotonic() >= deadline:
                        raise ControlFault(f"{name}: control script startup timed out")
                    time.sleep(.02)
            # No movement is issued until all controllers/offsets are checked.
            for name, control in self.controls.items():
                self._check_offset(name, control)
            if not defer_watchdog:
                self.enable_watchdog()
        except BaseException:
            self.close()
            raise

    def _check_offset(self, name, control):
        active = pose_matrix(self._transport_call(name, "getTCPOffset", control.getTCPOffset))
        if (np.linalg.norm(active[:3, 3] - self.offsets[name][:3, 3]) > 1e-5
                or rotation_distance(active, self.offsets[name]) > 1e-5):
            raise ControlFault(f"{name}: active TCP differs from --{name}_tcp_offset")

    def transport_diagnostics(self):
        """Return an independent snapshot of the latest velocity transaction.

        Times describe SDK call entry/return on the host, not robot receipt.
        Native safety/offset checks remain in their original order and the
        diagnostic path never transmits a controller command.
        """
        record = deepcopy(self._transport_current or self._last_transport or {})
        record["last_successful_send"] = deepcopy(self._last_successful_send)
        return record

    def capture_fault(self, exc, at=None):
        """Annotate the original failure before cleanup, at most once.

        Status access uses cached SDK state only. Diagnostic failures must
        never replace the original exception or prevent the stop path.
        """
        try:
            existing = getattr(exc, "details", None)
            if isinstance(existing, dict) and "transport" in existing:
                return
            from urcal.control.transport_diagnostics import control_snapshot

            at = time.monotonic() if at is None else at
            record = self.transport_diagnostics()
            record["captured_at"] = at
            record["seconds_since_last_successful_send"] = {
                name: max(0., at - sent["at"])
                for name, sent in self._last_successful_send.items()
            }
            record["control_status"] = {
                name: control_snapshot(control, at) for name, control in self.controls.items()
            }
            details = dict(existing) if isinstance(existing, dict) else {}
            details.setdefault("detected_at", at)
            details["transport"] = record
            exc.details = details
        except BaseException:
            # Even a broken optional status getter must not mask a motion
            # fault (or an operator interruption) already being handled.
            pass

    def _transport_call(self, name, stage, function, *values):
        if self._transport_current is None:
            return function(*values)
        operation = {"robot": name, "stage": stage, "started_at": time.monotonic()}
        self._transport_current["operations"].append(operation)
        try:
            result = function(*values)
            if isinstance(result, (bool, np.bool_)):
                operation["returned"] = bool(result)
            return result
        except BaseException as exc:
            operation["exception_type"] = type(exc).__name__
            raise
        finally:
            operation["finished_at"] = time.monotonic()
            operation["duration"] = operation["finished_at"] - operation["started_at"]

    def _run_transport(self, mode, function, *values, **options):
        record = {"mode": mode, "started_at": time.monotonic(), "operations": []}
        self._transport_current = record
        try:
            result = function(*values, **options)
        except BaseException as exc:
            record["failed_at"] = time.monotonic()
            record["finished_at"] = record["failed_at"]
            record["duration"] = record["finished_at"] - record["started_at"]
            self.capture_fault(exc, record["failed_at"])
            raise
        else:
            record["finished_at"] = time.monotonic()
            record["duration"] = record["finished_at"] - record["started_at"]
            return result
        finally:
            self._last_transport = record
            self._transport_current = None

    def _check_live_control(self, name, control):
        if not self._transport_call(name, "isConnected", control.isConnected):
            raise ControlFault(f"{name}: control connection disconnected")
        if not self._transport_call(name, "isProgramRunning", control.isProgramRunning):
            raise ControlFault(f"{name}: control program stopped")

    def enable_watchdog(self, check_fresh=None):
        """Arm communication supervision after a stationary native preflight.

        With ``defer_watchdog=True`` no velocity command can be transmitted
        until this succeeds for every commanded controller. Repeated calls
        after success do not write watchdog or command registers again.
        """
        if self._watchdog_enabled:
            return
        if not self.controls:
            raise ControlFault("motion session is closed")
        if self._velocity_sent:
            raise ControlFault("cannot enable watchdog after velocity commands")
        try:
            for name, control in self.controls.items():
                if check_fresh is not None:
                    check_fresh()
                if not control.isConnected():
                    raise ControlFault(f"{name}: control connection stopped")
                self._check_offset(name, control)
                if not control.setWatchdog(self.args.watchdog_hz):
                    raise ControlFault(f"{name}: failed to configure controller watchdog")
            if check_fresh is not None:
                check_fresh()
            self._watchdog_enabled = True
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _joint_vector(value, label):
        try:
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ControlFault(label + " must contain six finite values") from exc
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ControlFault(label + " must contain six finite values")
        return result.copy()

    @staticmethod
    def _native_method(name, control, method, optional=False):
        function = getattr(control, method, None)
        if callable(function):
            return function
        if optional:
            return None
        raise ControlFault(f"{name}: controller SDK does not provide {method}; native alignment preflight unavailable")

    def _native_call(self, name, method, *values):
        function = self._native_method(name, self.controls[name], method)
        try:
            return function(*values)
        except Exception as exc:
            raise ControlFault(f"{name}: native {method} failed: {exc}") from exc

    def _idle_preflight(self, check_fresh=None):
        # Even an all-zero speed packet enters velocity mode. Read-only native
        # queries and NO_CMD watchdog kicks must never interrupt that stream.
        if self._velocity_sent:
            raise ControlFault("native alignment preflight is unavailable after velocity commands")
        if not self.controls:
            raise ControlFault("motion session is closed")
        if check_fresh is not None:
            check_fresh()
        for name, control in self.controls.items():
            if not control.isConnected() or not control.isProgramRunning():
                raise ControlFault(f"{name}: control connection/program stopped")
            self._check_offset(name, control)
            if self._watchdog_enabled and not self._native_call(name, "kickWatchdog"):
                raise ControlFault(f"{name}: idle preflight watchdog kick failed")

    def _native_tcp(self, name, joints):
        offset = matrix_pose(self.offsets[name])
        query_offset = offset.copy()
        zero_offset = bool(np.all(offset == 0.))
        if zero_offset:
            # ur_rtde 1.6.3 selects its six-double recipe for a zero TCP,
            # but serializes q + TCP (twelve doubles), then waits up to 300s.
            # A nonzero, query-only TCP selects the correct twelve-double
            # recipe. Do not omit the TCP: that leaves stale input registers.
            # No setTcp or motion is involved; undo this transform below.
            query_offset[2] = .001
        value = self._native_call(name, "getForwardKinematics", joints.tolist(), query_offset.tolist())
        try:
            tcp = pose_matrix(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ControlFault(f"{name}: invalid native forward-kinematics TCP pose") from exc
        if zero_offset:
            tcp = tcp @ inverse(pose_matrix(query_offset)) @ self.offsets[name]
        return tcp

    def solve_joint_targets(self, targets, seeds, check_fresh=None, *, report=None):
        """Use calibrated controller IK/FK while stationary; return checked q.

        The active TCP offset is checked before querying IK. FK receives an
        explicit query TCP, then restores the configured TCP before verifying
        a 0.5 mm / 0.002 rad residual against the requested controller-base pose.
        """
        # A failed retry must never leave a prior candidate's native endpoint
        # available for the caller's conditioning assessment.
        self.last_ik_tcp = {}
        self._idle_preflight(check_fresh)
        prepared = {}
        for name in self.controls:
            try:
                target = np.asarray(targets[name], dtype=float)
                inverse(target)
                seed = self._joint_vector(seeds[name], name + " IK seed")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ControlFault(f"{name}: invalid native IK target/seed: {exc}") from exc
            prepared[name] = target.copy(), seed
        results, native_tcp = {}, {}
        for name, control in self.controls.items():
            def progress(stage):
                if report is not None:
                    report(f"PREFLIGHT: {name.upper()} {stage}; no motion yet")

            progress("checking native IK availability")
            self._idle_preflight(check_fresh)
            target, seed = prepared[name]
            pose, near = matrix_pose(target).tolist(), seed.tolist()
            available = self._native_method(name, control, "getInverseKinematicsHasSolution", optional=True)
            if available is not None:
                if not self._native_call(name, "getInverseKinematicsHasSolution", pose, near, .0005, .002):
                    raise ArmIKUnavailable(name, target, seed, self.offsets[name])
                self._idle_preflight(check_fresh)
            progress("solving native IK")
            result = self._joint_vector(self._native_call(name, "getInverseKinematics", pose, near, .0005, .002),
                                        name + " native IK result")
            self._idle_preflight(check_fresh)
            progress("checking IK joint limits")
            if not self._native_call(name, "isJointsWithinSafetyLimits", result.tolist()):
                raise ControlFault(f"{name}: native IK joints outside controller safety limits")
            self._idle_preflight(check_fresh)
            progress("verifying native FK at IK solution")
            actual = self._native_tcp(name, result)
            if (np.linalg.norm(actual[:3, 3] - target[:3, 3]) > .0005 + 1e-12
                    or rotation_distance(actual, target) > .002 + 1e-12):
                raise ControlFault(f"{name}: native IK/FK target residual exceeds 0.5 mm / 0.002 rad")
            self._idle_preflight(check_fresh)
            progress("checking IK TCP limits")
            if not self._native_call(name, "isPoseWithinSafetyLimits", matrix_pose(actual).tolist()):
                raise ControlFault(f"{name}: native IK TCP outside controller safety limits")
            self._idle_preflight(check_fresh)
            results[name] = result.copy()
            native_tcp[name] = actual.copy()
            progress("native IK/FK validated")
        self.last_ik_tcp = native_tcp
        return results

    def validate_joint_path(self, samples, check_fresh=None, *, report=None):
        """Check native joint/TCP safety of stationary preflight path samples.

        Return one dictionary of calibrated T_base_TCP poses per input sample.
        Sampling cannot certify swept-volume collision avoidance; geometric
        path checks and live freshness checks remain the caller's responsibility.
        Prefer a deferred watchdog for long preflights. If already enabled,
        watchdog kicks are confined to this pre-motion idle state.
        """
        self._idle_preflight(check_fresh)
        samples = list(samples)
        if not samples:
            raise ControlFault("native joint path preflight requires at least one sample")
        result = []
        for index, sample in enumerate(samples):
            if report is not None and index % 10 == 0:
                report(f"PREFLIGHT: joint path {index}/{len(samples)} checked; "
                       f"checking native safety/FK from sample {index + 1}; no motion yet")
            joints = {}
            for name in self.controls:
                try:
                    joints[name] = self._joint_vector(sample[name], f"{name} path sample {index}")
                except (KeyError, TypeError) as exc:
                    raise ControlFault(f"{name}: missing joints in path sample {index}") from exc
            poses = {}
            for name in self.controls:
                self._idle_preflight(check_fresh)
                if not self._native_call(name, "isJointsWithinSafetyLimits", joints[name].tolist()):
                    raise ControlFault(f"{name}: path sample {index} joints outside controller safety limits")
                self._idle_preflight(check_fresh)
                poses[name] = self._native_tcp(name, joints[name])
                self._idle_preflight(check_fresh)
                if not self._native_call(name, "isPoseWithinSafetyLimits", matrix_pose(poses[name]).tolist()):
                    raise ControlFault(f"{name}: path sample {index} TCP outside controller safety limits")
                self._idle_preflight(check_fresh)
            result.append(poses)
        if report is not None:
            report(f"PREFLIGHT: joint path {len(result)}/{len(samples)} checked; no motion yet")
        return result

    def send(self, targets, twists, check_fresh=None, safety_targets=None, *, accelerations=None):
        """Validate the complete pair before issuing Cartesian speed requests.

        Optional per-arm scalars come from the caller's local ramp prediction.
        They are capped by both Cartesian acceleration limits because speedL
        uses its scalar for translation and, by default, rotation. Existing
        callers without a ramp map retain the configured linear acceleration.
        """
        return self._run_transport("speedL", self._send, targets, twists,
                                   check_fresh=check_fresh, safety_targets=safety_targets,
                                   accelerations=accelerations)

    def _send(self, targets, twists, check_fresh=None, safety_targets=None, *, accelerations=None):
        if not self._watchdog_enabled:
            raise ControlFault("enable the controller watchdog before sending motion")
        prepared = {}
        # Validate the complete pair before sending either arm a motion command.
        for name, control in self.controls.items():
            self._check_live_control(name, control)
            self._check_offset(name, control)
            if not self._transport_call(name, "goal_safety", control.isPoseWithinSafetyLimits,
                                        matrix_pose(targets[name]).tolist()):
                raise ControlFault(f"{name}: goal is unreachable or outside controller safety limits")
            if safety_targets is not None and not self._transport_call(
                    name, "next_goal_safety", control.isPoseWithinSafetyLimits,
                    matrix_pose(safety_targets[name]).tolist()):
                raise ControlFault(f"{name}: next goal is unreachable or outside controller safety limits")
            try:
                raw_acceleration = np.asarray(
                    self.args.max_accel if accelerations is None else accelerations[name], dtype=float)
                if raw_acceleration.shape != ():
                    raise ValueError("acceleration must be a scalar")
                acceleration = float(raw_acceleration)
                linear_cap = float(self.args.max_accel)
                angular_cap = linear_cap if accelerations is None else float(self.args.max_angular_accel)
                if (not np.isfinite(linear_cap) or not np.isfinite(angular_cap)
                        or linear_cap <= 0. or angular_cap <= 0.):
                    raise ValueError("invalid acceleration limits")
                cap = min(linear_cap, angular_cap)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ControlFault(f"{name}: missing or invalid Cartesian acceleration") from exc
            if (not np.isfinite(acceleration) or not np.isfinite(cap)
                    or not 0. < acceleration <= cap):
                raise ControlFault(f"{name}: Cartesian acceleration must be positive and within configured limits")
            try:
                velocity = self._joint_vector(twists[name], name + " Cartesian command")
            except (KeyError, TypeError) as exc:
                raise ControlFault(f"{name}: missing Cartesian velocity") from exc
            prepared[name] = velocity, acceleration
        for name, control in self.controls.items():
            if check_fresh is not None:
                self._transport_call(name, "check_fresh", check_fresh)
            # Positive time bounds the robot-side speed thread's update period,
            # NOT the lifetime of a velocity. Each speedL packet feeds watchdog;
            # an extra kick would write NO_CMD into the same command register.
            self._velocity_sent = True
            self._motion_modes[name] = "linear"
            velocity, acceleration = prepared[name]
            if not self._transport_call(name, "speedL", control.speedL,
                                        velocity.tolist(), acceleration, 0.002):
                raise ControlFault(f"{name}: speedL failed")
            self._last_successful_send[name] = {
                "method": "speedL", "at": self._transport_current["operations"][-1]["finished_at"]}

    def send_joints(self, qdots, next_joints, check_fresh=None, *, accelerations=None):
        """Validate the whole commanded pair, then transmit bounded speedJ.

        No watchdog kick or native IK/FK query is inserted into the live speed
        stream. The caller validates acceleration against fresh measured qd and
        supplies the next joint configuration for controller safety checks.
        Per-arm acceleration values are passed unchanged to speedJ; callers
        that omit them retain the configured joint acceleration limit.
        """
        return self._run_transport("speedJ", self._send_joints, qdots, next_joints,
                                   check_fresh=check_fresh, accelerations=accelerations)

    def _send_joints(self, qdots, next_joints, check_fresh=None, *, accelerations=None):
        if not self._watchdog_enabled:
            raise ControlFault("enable the controller watchdog before sending motion")
        prepared = {}
        for name, control in self.controls.items():
            self._check_live_control(name, control)
            self._check_offset(name, control)
            try:
                velocity = self._joint_vector(qdots[name], name + " joint command")
                following = self._joint_vector(next_joints[name], name + " next joints")
            except (KeyError, TypeError) as exc:
                raise ControlFault(f"{name}: missing joint velocity/next configuration") from exc
            try:
                raw_acceleration = np.asarray(
                    self.args.max_joint_accel if accelerations is None else accelerations[name], dtype=float)
                if raw_acceleration.shape != ():
                    raise ValueError("acceleration must be a scalar")
                acceleration = float(raw_acceleration)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ControlFault(f"{name}: missing or invalid joint acceleration") from exc
            if not np.isfinite(acceleration) or not 0. < acceleration <= self.args.max_joint_accel:
                raise ControlFault(f"{name}: joint acceleration must be positive and within configured limit")
            if np.max(np.abs(velocity)) > self.args.max_joint_speed + 1e-12:
                raise ControlFault(f"{name}: joint velocity exceeds configured speed limit")
            if not self._transport_call(name, "next_joint_safety", self._native_call,
                                        name, "isJointsWithinSafetyLimits", following.tolist()):
                raise ControlFault(f"{name}: next joints outside controller safety limits")
            self._native_method(name, control, "speedJ")
            prepared[name] = velocity, acceleration
        for name, control in self.controls.items():
            if check_fresh is not None:
                self._transport_call(name, "check_fresh", check_fresh)
            self._velocity_sent = True
            self._motion_modes[name] = "joint"
            velocity, acceleration = prepared[name]
            if not self._transport_call(name, "speedJ", control.speedJ,
                                        velocity.tolist(), acceleration, .002):
                raise ControlFault(f"{name}: speedJ failed")
            self._last_successful_send[name] = {
                "method": "speedJ", "at": self._transport_current["operations"][-1]["finished_at"]}

    def stop_for_recovery(self):
        """Stop and release every commanded arm, or fail closed.

        Unlike best-effort shutdown, recovery requires an affirmative speed
        stop and successful script shutdown (the SDK's stopScript is void).
        Fresh encoder observations must subsequently
        confirm stationarity before the application may offer another Arm.
        """
        if not self.controls:
            raise ControlFault("cannot confirm recovery stop: motion session is closed")
        self._close_controls(require_confirmation=True)

    def close(self):
        self._close_controls(require_confirmation=False)

    def _close_controls(self, *, require_confirmation):
        controls, self.controls = self.controls, {}
        self._watchdog_enabled = False
        failures = []
        # Request both stops before any script shutdown/disconnect can block.
        for name, control in controls.items():
            try:
                acceleration = self.args.max_joint_accel if self._motion_modes.get(name) == "joint" else self.args.stop_accel
                result = control.speedStop(acceleration)
                if require_confirmation and not (isinstance(result, (bool, np.bool_)) and result):
                    failures.append(f"{name}: speedStop did not confirm success")
            except Exception as exc:
                failures.append(f"{name}: speedStop failed: {exc}")
                print(f"[warn] {name} speedStop: {exc}", flush=True)
        for name, control in controls.items():
            for label in ("stopScript", "disconnect"):
                try:
                    result = getattr(control, label)()
                    if (require_confirmation and label == "stopScript" and result is not None
                            and not (isinstance(result, (bool, np.bool_)) and result)):
                        failures.append(f"{name}: stopScript did not confirm success")
                except Exception as exc:
                    failures.append(f"{name}: {label} failed: {exc}")
                    print(f"[warn] {name} cleanup: {exc}", flush=True)
        if require_confirmation and failures:
            raise ControlFault("recovery stop could not be confirmed: " + "; ".join(failures))


class FacingPlanner:
    """Freeze hold goals in C; recompute follower goals from fresh leader pose."""

    def __init__(self, args):
        self.args = args
        self.offsets = {name: pose_matrix(getattr(args, name + "_tcp_offset") or [0.] * 6) for name in ROBOTS}
        self.mappings = {name: model_from_controller_base(name) for name in ROBOTS}
        self.gap = args.gap
        self.hold_targets = None
        self.initial_tcp = None
        self.target_applied = args.mode == "follow" or args.center_pose is not None

    @property
    def commanded(self):
        return ROBOTS if self.args.mode == "hold" else tuple(name for name in ROBOTS if name != self.args.leader)

    @property
    def center_pose(self):
        if self.args.mode != "hold":
            return None
        if self.hold_targets is None:
            return None if self.args.center_pose is None else list(self.args.center_pose)
        center = self.hold_targets["ur5e"].copy()
        center[:3, 3] = (self.hold_targets["ur5e"][:3, 3] + self.hold_targets["ur10e"][:3, 3]) / 2.
        return matrix_pose(center).tolist()

    def apply_target(self, command):
        """Atomically replace a preview target; the runtime forbids active edits."""
        if command.get("type") != "target":
            raise ValueError("unknown GUI command")
        gap = float(command["gap"])
        if not np.isfinite(gap) or gap < self.args.min_gap:
            raise ValueError("gap must be finite and at least min_gap")
        if self.args.mode == "hold":
            center = matrix_pose(pose_matrix(command["center_pose"])).tolist()
        else:
            if "center_pose" in command:
                raise ValueError("center_pose is only for hold mode")
            center = None
        # Validate everything above before altering the previously applied goal.
        self.gap, self.args.center_pose = gap, center
        self.hold_targets = None
        self.target_applied = True

    def plan(self, calibration, states):
        bases, flanges = {}, {}
        for name in ROBOTS:
            bases[name] = calibration[name] @ self.mappings[name]
            flange = flange_from_tcp(states[name].tcp, self.offsets[name])
            nominal = model_flange_pose(name, states[name].joints)
            observed = self.mappings[name] @ flange
            position_error = np.linalg.norm(nominal[:3, 3] - observed[:3, 3])
            angle_error = rotation_distance(nominal, observed)
            if position_error > self.args.frame_check_m or angle_error > np.deg2rad(self.args.frame_check_deg):
                raise ControlFault(f"{name}: FK/base/TCP mismatch {position_error:.3f} m, {np.rad2deg(angle_error):.1f} deg")
            flanges[name] = bases[name] @ flange
        separation = np.linalg.norm(flanges[ROBOTS[0]][:3, 3] - flanges[ROBOTS[1]][:3, 3])
        if separation < self.args.min_gap:
            raise PlanningUnavailable(f"flange separation {separation:.3f} m is below min_gap")
        if self.gap is None:
            self.gap = float(separation)
        if self.gap < self.args.min_gap:
            raise PlanningUnavailable("target gap is below min_gap")
        if self.args.mode == "hold":
            if self.hold_targets is None:
                self.hold_targets = fixed_facing_targets(
                    flanges["ur5e"], flanges["ur10e"], self.gap, self.args.center_pose,
                )
            goals = {name: pose.copy() for name, pose in self.hold_targets.items()}
        else:
            goals = {self.commanded[0]: facing_follower(flanges[self.args.leader], self.gap)}
        tcp_goals = {name: inverse(bases[name]) @ goals[name] @ self.offsets[name] for name in self.commanded}
        return goals, tcp_goals

    def check_envelope(self, states, targets):
        if self.initial_tcp is None:
            self.initial_tcp = {name: state.tcp.copy() for name, state in states.items()}
        for name in self.commanded:
            origin = self.initial_tcp[name]
            for pose in (states[name].tcp, targets[name]):
                if np.linalg.norm(pose[:3, 3] - origin[:3, 3]) > self.args.max_travel:
                    raise PlanningUnavailable(f"{name}: outside {self.args.max_travel:.3f} m start-position envelope")
            if (np.linalg.norm(states[name].speed[:3]) > self.args.max_speed * 1.5 + 0.005
                    or np.linalg.norm(states[name].speed[3:]) > self.args.max_angular_speed * 1.5 + 0.02):
                raise ControlFault(f"{name}: measured speed exceeded configured limit")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("hold", "follow"), required=True)
    p.add_argument("--leader", choices=ROBOTS, default="ur10e")
    p.add_argument("--gap", type=float, default=.30, help="Flange origin separation in metres (default: 0.30); editable in GUI")
    p.add_argument("--center_pose", nargs=6, type=float, help="Hold pair center in camera frame: x y z rx ry rz (m, rotation vector rad)")
    p.add_argument("--execute", action="store_true", help="Allow motion after GUI Arm or a in preview; otherwise receive-only")
    p.add_argument("--initial_align", action="store_true",
                   help="After explicit Arm, validate and slowly align from rest before hold/follow; keep leader/bases stationary")
    p.add_argument("--align_speed_scale", type=float, default=2.,
                   help="Live initial alignment speed multiplier: 1 keeps the previous half-speed cap; "
                        "2 (default) uses up to 90%% of configured speed limits, reserving servo headroom. "
                        "Range (0, 2]; measured speed and acceleration limits unchanged")
    p.add_argument("--align_leader_max_speed", type=float, default=.01,
                   help="Leader TCP rest threshold during follow alignment and calibration handoff (default: 0.01 m/s)")
    p.add_argument("--align_leader_max_angular_speed", type=float, default=.05,
                   help="Leader angular rest threshold during follow alignment/handoff (default: 0.05 rad/s)")
    p.add_argument("--align_leader_max_joint_speed", type=float, default=.025,
                   help="Leader per-joint rest threshold during follow alignment/handoff (default: 0.025 rad/s)")
    p.add_argument("--simulate", action="store_true",
                   help="Capture current joints/calibration once, disconnect hardware, then run a virtual joint simulation")
    p.add_argument("--initial_state", type=Path,
                   help="Saved simulation initial-state JSON; with --simulate, skips all camera/RTDE connections")
    p.add_argument("--sim_initial_timeout", type=float, default=60., help="Initial live snapshot timeout in seconds")
    p.add_argument("--sim_motion", choices=("none", "leader", "bases", "both"), default="none",
                   help="Virtual external motion after Start simulation; hold supports none/bases")
    for name, folder, default_lambda in (("ur5e", "v3m2a3", .5), ("ur10e", "ur10e_2", .2)):
        p.add_argument("--" + name + "_ip")
        p.add_argument("--" + name + "_tcp_offset", nargs=6, type=float,
                       help="Actual flange-to-active-TCP offset x y z rx ry rz; required with --execute")
        p.add_argument("--" + name + "_checkpoint", type=Path, default=BUNDLE / "trained_models" / folder / "best.pt")
        p.add_argument("--" + name + "_empnp_lambda", type=float, default=default_lambda,
                       help="EMPnP correction weight in [0, 1]; with --empnp_use_prev, smaller values weaken the prior-pose pull")
    previous_pose = p.add_mutually_exclusive_group()
    previous_pose.add_argument("--empnp_use_prev", action="store_true", dest="empnp_use_prev",
                               help="Enable previous-pose EMPnP correction (control default: off)")
    previous_pose.add_argument("--no_empnp_use_prev", action="store_false", dest="empnp_use_prev",
                               help="Disable previous-pose EMPnP correction")
    p.set_defaults(empnp_use_prev=False)
    p.add_argument("--split_ratio", type=float, default=.5)
    p.add_argument("--camera_w", type=int, default=640)
    p.add_argument("--camera_h", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--show_seg", action="store_true")
    p.add_argument("--hide_keypoints", action="store_true")
    p.add_argument("--control_hz", type=float, default=25.)
    p.add_argument("--watchdog_hz", type=float, default=5.)
    p.add_argument("--max_calibration_age", type=float, default=.5)
    p.add_argument("--max_joint_skew", type=float, default=.08)
    p.add_argument("--max_robot_age", type=float, default=.15)
    p.add_argument("--stable_frames", type=int, default=5)
    p.add_argument("--calibration_acquisition", choices=("consensus", "consecutive"), default="consensus",
                   help="Stationary initial-alignment acquisition: recent 1s consensus or consecutive raw frames")
    p.add_argument("--filter_alpha", type=float, default=.2)
    p.add_argument("--check_reprojection", action="store_true",
                   help="Enable mean/max reprojection rejection in preview and execution (default: off)")
    p.add_argument("--max_reproj_px", type=float, default=5.,
                   help="Mean original-detection reprojection limit with --check_reprojection (default: 5 pixels)")
    p.add_argument("--max_point_reproj_px", type=float, default=8.,
                   help="Maximum point reprojection limit with --check_reprojection (default: 8 pixels)")
    p.add_argument("--min_points", type=int, default=5)
    p.add_argument("--max_calibration_jump_m", type=float, default=.05,
                   help="Maximum change from the last accepted raw base pose in metres (default: 0.05)")
    p.add_argument("--max_calibration_jump_deg", type=float, default=5.,
                   help="Maximum change from the last accepted raw base orientation in degrees (default: 5)")
    p.add_argument("--handoff_max_shift_m", type=float, default=.15,
                   help="Allowed nominal flange target shift when switching from alignment to live calibration (default: 0.15 m)")
    p.add_argument("--handoff_max_shift_deg", type=float, default=15.,
                   help="Allowed nominal flange orientation shift at calibration handoff (default: 15 degrees)")
    p.add_argument("--handoff_timeout", type=float, default=15.,
                   help="Post-alignment acquisition wait notice / stop deadline in seconds (default: 15)")
    p.add_argument("--handoff_wait", choices=("hold", "stop"), default="hold",
                   help="After the handoff deadline keep checked zero commands at the aligned pose, or stop (default: hold)")
    p.add_argument("--frame_check_m", type=float, default=.03)
    p.add_argument("--frame_check_deg", type=float, default=5.)
    p.add_argument("--min_gap", type=float, default=.15)
    p.add_argument("--max_travel", type=float, default=.15)
    p.add_argument("--max_target_m", type=float, default=.20,
                   help="Tracking position error stop threshold in metres (default: 0.20); does not change travel/speed limits")
    p.add_argument("--max_target_deg", type=float, default=25.)
    p.add_argument("--target_tolerance_m", type=float, default=.05,
                   help="Allowed estimated flange target adjustment/error in metres (default: 0.05); not calibration accuracy")
    p.add_argument("--target_tolerance_deg", type=float, default=10.,
                   help="Allowed estimated flange orientation adjustment/error in degrees (default: 10); zero with zero position tolerance restores exact goals")
    p.add_argument("--max_speed", type=float,
                   help="TCP linear limit in m/s for hold and initial alignment (default: 0.02); "
                        "an explicit value also sets following unless --follow_max_speed is supplied")
    p.add_argument("--max_angular_speed", type=float, default=.1)
    p.add_argument("--max_accel", type=float, default=.1)
    p.add_argument("--max_angular_accel", type=float, default=.5)
    p.add_argument("--stop_accel", type=float, default=.5)
    p.add_argument("--fault_response", choices=("preview", "exit"), default="preview",
                   help="Recoverable calibration/planning faults: stop and keep preview for a fresh explicit Arm (default), "
                        "or exit as in earlier versions; hardware/operator faults still stop and exit")
    p.add_argument("--kp_linear", type=float, default=1.)
    p.add_argument("--kp_angular", type=float, default=1.)
    research = p.add_argument_group("moving-base relative collaboration (--mode follow)")
    research.add_argument("--method", choices=("direct", "predict", "separated"), default="separated")
    research.add_argument("--follow_max_speed", type=float,
                          help="Follower tracking TCP linear limit in m/s (default: 1.0, or explicit --max_speed); "
                               "does not change initial alignment, hold or acceleration limits")
    research.add_argument("--leader_response", choices=("bounded", "strict"), default="bounded",
                          help="Separated following with target tolerance: bounded ramps leader tracking within motion limits; "
                               "strict requires immediate full leader compensation for research comparisons")
    research.add_argument("--path", choices=("hold", "line", "circle"), default="hold")
    research.add_argument("--path_delta", nargs=6, type=float, default=[0, 0, -.05, 0, 0, 0],
                          help="Line delta in leader flange axes: XYZ metres, rotation vector radians")
    research.add_argument("--path_radius", type=float, default=.02)
    research.add_argument("--path_duration", type=float, default=10.)
    research.add_argument("--velocity_alpha", type=float, default=.3)
    research.add_argument("--prediction_horizon", type=float, default=.5)
    research.add_argument("--max_base_speed", type=float, default=.15)
    research.add_argument("--max_base_angular_speed", type=float, default=.5)
    research.add_argument("--max_invalid_duration", type=float, default=.3,
                          help="maximum active calibration rejection grace in seconds (alignment/hold/follow)")
    research.add_argument("--max_joint_speed", type=float, default=.2)
    research.add_argument("--max_joint_accel", type=float, default=.5)
    research.add_argument("--min_singular_value", type=float, default=.01)
    research.add_argument("--feedback_speed", type=float, default=.01)
    research.add_argument("--feedback_angular_speed", type=float, default=.05)
    research.add_argument("--position_deadband", type=float, default=.001)
    research.add_argument("--rotation_deadband_deg", type=float, default=.3)
    research.add_argument("--slow_error_m", type=float, default=.01)
    research.add_argument("--slow_error_deg", type=float, default=5.)
    research.add_argument("--finish_position_m", type=float, default=.005)
    research.add_argument("--finish_rotation_deg", type=float, default=2.)
    research.add_argument("--finish_dwell", type=float, default=.5)
    research.add_argument("--workspace_config", type=Path, help="Joint limits, TCP boxes, collision capsule radii and camera keep-outs JSON")
    research.add_argument("--log", type=Path, help="Exclusive JSONL output; default: runs/<timestamp>-<method>-<uuid>.jsonl")
    args = p.parse_args(argv)
    if args.follow_max_speed is None:
        args.follow_max_speed = 1. if args.max_speed is None else args.max_speed
    if args.max_speed is None:
        args.max_speed = .02
    if args.simulate and args.initial_align:
        p.error("--initial_align is for live control; simulation already provides its own initial alignment")
    if args.simulate and args.execute:
        p.error("--simulate and --execute are mutually exclusive; simulation never commands hardware")
    if args.initial_state is not None and not args.simulate:
        p.error("--initial_state requires --simulate")
    if not args.simulate and args.sim_motion != "none":
        p.error("--sim_motion requires --simulate")
    if args.simulate and args.mode == "hold" and args.sim_motion in ("leader", "both"):
        p.error("hold simulation commands both arms; use --sim_motion none or bases")
    if args.initial_state is None:
        for name in ROBOTS:
            if not getattr(args, name + "_ip"):
                p.error(f"--{name}_ip is required for live capture")
    if not np.isfinite(args.sim_initial_timeout) or args.sim_initial_timeout <= 0:
        p.error("sim_initial_timeout must be finite and positive")
    if not np.isfinite(args.align_speed_scale) or not 0 < args.align_speed_scale <= 2:
        p.error("align_speed_scale must be finite and in (0, 2]")
    positive = ("control_hz", "watchdog_hz", "max_calibration_age", "max_joint_skew", "max_robot_age",
                "stable_frames", "max_reproj_px", "max_point_reproj_px", "max_calibration_jump_m",
                "max_calibration_jump_deg", "handoff_timeout", "frame_check_m", "frame_check_deg", "min_gap", "max_travel",
                "max_target_m", "max_target_deg", "max_speed", "follow_max_speed", "max_angular_speed", "max_accel",
                "max_angular_accel", "stop_accel", "kp_linear", "kp_angular", "camera_w", "camera_h", "fps",
                "align_leader_max_speed", "align_leader_max_angular_speed", "align_leader_max_joint_speed",
                "path_radius", "path_duration", "prediction_horizon", "max_base_speed", "max_base_angular_speed",
                "max_invalid_duration", "max_joint_speed", "max_joint_accel", "min_singular_value",
                "feedback_speed", "feedback_angular_speed", "slow_error_m", "slow_error_deg",
                "finish_position_m", "finish_rotation_deg", "finish_dwell")
    if any(not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        p.error("rates, limits, camera dimensions and gains must be finite and positive")
    if (not np.isfinite([args.handoff_max_shift_m, args.handoff_max_shift_deg]).all()
            or min(args.handoff_max_shift_m, args.handoff_max_shift_deg) < 0):
        p.error("calibration handoff shift limits must be finite and nonnegative")
    if (not np.isfinite([args.target_tolerance_m, args.target_tolerance_deg]).all()
            or min(args.target_tolerance_m, args.target_tolerance_deg) < 0):
        p.error("target tolerances must be finite and nonnegative")
    if args.target_tolerance_m >= args.max_target_m or args.target_tolerance_deg >= args.max_target_deg:
        p.error("target tolerances must be below max_target_m/max_target_deg tracking-error limits")
    if not 0 < args.split_ratio < 1 or not 0 < args.filter_alpha <= 1 or not 4 <= args.min_points <= 7:
        p.error("invalid split_ratio, filter_alpha or min_points")
    if args.camera_w < 2 or not 0 < round(args.camera_w * args.split_ratio) < args.camera_w:
        p.error("split_ratio must leave both image regions nonempty")
    if args.watchdog_hz >= args.control_hz / 2:
        p.error("watchdog_hz must be less than half of control_hz")
    if args.gap is not None and (not np.isfinite(args.gap) or args.gap < args.min_gap):
        p.error("gap must be finite and at least min_gap")
    if args.mode != "hold" and args.center_pose is not None:
        p.error("center_pose is only for hold mode")
    if args.ur5e_ip is not None and args.ur5e_ip == args.ur10e_ip:
        p.error("UR5e and UR10e must have different controller IPs")
    for name in ROBOTS:
        weight = getattr(args, name + "_empnp_lambda")
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            p.error(f"{name}_empnp_lambda must be finite and in [0, 1]")
        offset = getattr(args, name + "_tcp_offset")
        if args.execute and offset is None:
            p.error(f"--execute requires explicit --{name}_tcp_offset from the robot installation")
        if offset is not None and not np.isfinite(offset).all():
            p.error("TCP offsets must be finite")
    if args.center_pose is not None and not np.isfinite(args.center_pose).all():
        p.error("center_pose must be finite")
    if not np.isfinite(args.path_delta).all() or not 0 < args.velocity_alpha <= 1:
        p.error("invalid path_delta or velocity_alpha")
    if (not np.isfinite([args.position_deadband, args.rotation_deadband_deg]).all()
            or min(args.position_deadband, args.rotation_deadband_deg) < 0):
        p.error("deadbands must be finite and nonnegative")
    if args.slow_error_m >= args.max_target_m or args.slow_error_deg >= args.max_target_deg:
        p.error("slow_error thresholds must be below final target error limits")
    if args.prediction_horizon > args.max_calibration_age:
        p.error("prediction_horizon must not exceed max_calibration_age")
    return args


def run(args):
    from rtde_receive import RTDEReceiveInterface
    from urcal.control.gui import gui_worker
    from urcal.control.vision import vision_worker
    from urcal.control.startup import (
        AlignmentCalibrationTimeout,
        check_alignment_request, check_handoff_snapshot, prepare_alignment,
    )
    from urcal.control.handoff import CalibrationHandoff
    from urcal.control.acquisition import StationaryCalibrationAcquisition
    from urcal.control.recovery import RecoveryStopGate, fresh_calibration_tracker, is_recoverable_fault
    from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard

    ctx = mp.get_context("spawn")
    frames, commands, statuses, gui_statuses = (ctx.Queue(maxsize=size) for size in (1, 8, 1, 1))
    inbox = CommandInbox(commands)
    stop = ctx.Event()
    worker = ctx.Process(target=vision_worker, args=(args, frames, commands, statuses, stop), daemon=True)
    gui = ctx.Process(target=gui_worker, args=(args, commands, gui_statuses, stop), daemon=True)
    tracker = CalibrationTracker(
        max_age=args.max_calibration_age, stable_frames=args.stable_frames, alpha=args.filter_alpha,
        max_jump_m=args.max_calibration_jump_m, max_jump_rad=np.deg2rad(args.max_calibration_jump_deg),
        max_mean_px=args.max_reproj_px, max_pixel_px=args.max_point_reproj_px, min_points=args.min_points,
        check_reprojection=args.check_reprojection,
    )
    planner = FacingPlanner(args)
    calibration_guard = ActiveCalibrationGuard(args.max_invalid_duration)
    preview_recovery = PreviewCalibrationRecovery()
    stationary_acquisition = StationaryCalibrationAcquisition(tracker)
    calibration_degraded = False
    session = None
    arm_error = None
    alignment, alignment_complete = None, False
    alignment_bases, alignment_goals, alignment_tcp_goals = None, None, None
    alignment_nominal_goals = None
    alignment_capture = None
    handoff, endpoint_states = None, None
    models = {name: NominalKinematics(name, planner.offsets[name]) for name in ROBOTS} if args.initial_align else None
    guard = WorkspaceGuard(models, args.workspace_config) if args.initial_align else None
    previous = {name: np.zeros(6) for name in planner.commanded}
    last_tick = time.monotonic()
    period = 1.0 / args.control_hz
    last_report = 0.
    scene_timestamps = {}
    recovery_stop = None
    recovery_reason = None
    recovery_capture_after = None
    recovery_ready = False
    try:
        with ExitStack() as cleanup:
            readers = {}
            for name in ROBOTS:
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                cleanup.callback(receiver.disconnect)
                readers[name] = StateReader(receiver, args.max_robot_age, require_joint_velocity=args.initial_align)
            cleanup.callback(lambda: session.close() if session is not None else None)

            def observe_fault(_kind, exception, _traceback):
                if exception is not None:
                    capture_runtime_fault(exception, session, readers, time.monotonic())
                return False

            cleanup.push(observe_fault)
            # Import nominal FK outside the timed motion loop (no robot writes).
            for name in ROBOTS:
                model_flange_pose(name, np.zeros(6))
            gui.start()
            worker.start()
            print("[info] PREVIEW. GUI: Apply target, then Arm (--execute only). SPACE/q/ESC: stop+exit.", flush=True)
            print("[info] Recoverable faults: " + (
                "stop motion, keep preview; fresh calibration and a new Arm are required."
                if getattr(args, "fault_response", "exit") == "preview" else "stop and exit."), flush=True)
            print("[info] Mean/max reprojection rejection: " +
                  (f"ON (mean <= {args.max_reproj_px:g}px, max <= {args.max_point_reproj_px:g}px)."
                   if args.check_reprojection else "OFF (preview and execution)."), flush=True)
            print(f"[info] Estimated flange task tolerance: {getattr(args, 'target_tolerance_m', 0.) * 100:g}cm / "
                  f"{getattr(args, 'target_tolerance_deg', 0.):g}deg; nominal targets retained.", flush=True)
            print(f"[info] Calibration limits: jump {args.max_calibration_jump_m * 100:g}cm / "
                  f"{args.max_calibration_jump_deg:g}deg; handoff target shift "
                  f"{args.handoff_max_shift_m * 100:g}cm / {args.handoff_max_shift_deg:g}deg; "
                  f"handoff wait {args.handoff_timeout:g}s; tracking error stop {args.max_target_m * 100:g}cm.", flush=True)
            if args.initial_align:
                print(f"[info] Stationary calibration: {args.calibration_acquisition}; "
                      f"after {args.handoff_timeout:g}s handoff wait: {args.handoff_wait}.", flush=True)
            if any(getattr(args, name + "_tcp_offset") is None for name in ROBOTS):
                print("[info] preview assumes zero TCP offsets where omitted; specify actual offsets before motion.", flush=True)
            def read_alignment_inputs():
                checked = time.monotonic()
                if stop.is_set() or not worker.is_alive() or not gui.is_alive():
                    raise ControlFault("operator/vision stop during initial alignment")
                measured = {name: reader.read(checked) for name, reader in readers.items()}
                # Arm selected a complete, stable calibration. Native path
                # validation uses exactly that snapshot; vision may keep
                # running, but cannot change the path under validation.
                return {name: pose.copy() for name, pose in alignment_bases.items()}, measured

            def alignment_report(message):
                print("[control] " + message, flush=True)
                put_latest(gui_statuses, {"message": message, "active": True, "ready": False,
                                          "phase": "alignment_preflight"})

            while not stop.is_set():
                now = time.monotonic()
                dt, last_tick = now - last_tick, now
                arm_requested = False
                target_updated = False
                if not worker.is_alive() or not gui.is_alive():
                    raise ControlFault("vision or GUI process exited")
                if session is not None and dt > min(args.max_robot_age, 1. / args.watchdog_hz):
                    raise ControlFault("control loop missed its deadline")
                goals = {}
                display_command_targets = None
                ready = False
                states = None
                try:
                    for command in inbox.read():
                        if isinstance(command, dict):
                            if session is not None:
                                raise ControlFault("stop before changing targets")
                            planner.apply_target(command)
                            target_updated = True
                        elif command in ("quit", "stop"):
                            stop.set()
                        elif command == "arm":
                            arm_requested = True
                    if target_updated:
                        # Show/check the new goal before a later explicit Arm.
                        arm_requested = False
                    if stop.is_set():
                        break
                    packet = take_latest(frames)
                    if recovery_stop is not None:
                        # Robot stop acknowledgement alone does not prove rest.
                        # Only fresh encoder observations unlock calibration
                        # collection, and only a later Arm can restart motion.
                        states = {name: reader.read(now) for name, reader in readers.items()}
                        if not recovery_stop.observe(states, now):
                            recovery_capture_after = None
                            inbox.cancel_arm_requests()
                            arm_requested = False
                            raise PlanningUnavailable("RECOVERY WAIT: " + recovery_stop.reason)
                        if recovery_capture_after is None:
                            recovery_capture_after = recovery_stop.ready_at
                            tracker = fresh_calibration_tracker(tracker)
                            preview_recovery.reset()
                            stationary_acquisition.reset()
                        if packet is not None and packet.captured_at <= recovery_capture_after:
                            packet = None
                    frozen_alignment = session is not None and args.initial_align and not alignment_complete
                    if session is not None:
                        preview_recovery.reset()
                    if packet is not None and frozen_alignment:
                        if handoff is not None:
                            handoff.ingest(packet, now)
                    elif (packet is not None and session is None and args.initial_align
                          and args.calibration_acquisition == "consensus"):
                        candidate = stationary_acquisition.offer(packet, now)
                        if candidate is None:
                            tracker.reject(packet.sequence, stationary_acquisition.reason)
                        else:
                            if tracker.poses and (now - tracker.last_capture > tracker.max_age or
                                    stationary_acquisition._distance(tracker.poses, candidate.poses) > 1.):
                                arm_requested = False
                            tracker = candidate
                    elif packet is not None:
                        stationary_acquisition.reset()
                        try:
                            accepted = tracker.update(packet, now)
                        except ControlFault as exc:
                            # Only classified vision rejections use bounded
                            # grace. Malformed input propagates immediately.
                            if session is not None and not is_recoverable_fault(exc):
                                raise
                            if session is None:
                                candidate = preview_recovery.offer(packet, now, tracker)
                                if candidate is None:
                                    raise
                                tracker = candidate
                                preview_recovery.reset()
                                arm_requested = False
                                print("[control] PREVIEW: calibration reacquired from stable frames; review pose, then Arm", flush=True)
                        else:
                            if accepted:
                                preview_recovery.reset()
                    calibration_current = True
                    if session is None:
                        tracker.check_fresh(now)
                    elif not frozen_alignment:
                        calibration_current = calibration_guard.check(tracker, now)
                    if states is None:
                        states = {name: reader.read(now) for name, reader in readers.items()}
                    if frozen_alignment:
                        goals = {name: pose.copy() for name, pose in alignment_goals.items()}
                        tcp_goals = {name: pose.copy() for name, pose in alignment_tcp_goals.items()}
                    else:
                        goals, tcp_goals = planner.plan(tracker.poses, states)
                    initializing = args.initial_align and not alignment_complete
                    tolerant = bool(getattr(args, "target_tolerance_m", 0.) or
                                    getattr(args, "target_tolerance_deg", 0.))
                    # The nominal task may lie beyond the travel envelope
                    # while the accepted current/next command remains inside
                    # it. Initial path planning keeps its conservative full
                    # nominal-target check before any selected endpoint.
                    envelope_targets = ({name: states[name].tcp for name in planner.commanded}
                                        if tolerant and not initializing else tcp_goals)
                    planner.check_envelope(states, envelope_targets)
                    if recovery_stop is not None and not recovery_ready:
                        # This tick first establishes readiness. Invalidate
                        # clicks made while stopping or rebuilding calibration.
                        arm_requested = False
                        inbox.cancel_arm_requests()
                    if initializing:
                        bases = (alignment_bases if frozen_alignment else
                                 {name: tracker.poses[name] @ planner.mappings[name] for name in ROBOTS})
                        if session is None:
                            check_alignment_request(args, planner.offsets, bases, states, tcp_goals,
                                                    models, guard, planner.initial_tcp)
                            ready = planner.target_applied
                            message = ("READY: Arm to validate joint path and align; keep bases stationary"
                                       if ready else "PREVIEW: Apply target in GUI")
                            if arm_requested:
                                if not ready:
                                    message = "WAIT: Apply the hold target in GUI before arming"
                                elif args.execute:
                                    alignment_bases = {name: pose.copy() for name, pose in bases.items()}
                                    alignment_goals = {name: pose.copy() for name, pose in goals.items()}
                                    alignment_nominal_goals = {name: pose.copy() for name, pose in goals.items()}
                                    alignment_tcp_goals = {name: pose.copy() for name, pose in tcp_goals.items()}
                                    alignment_capture = tracker.last_capture
                                    handoff, endpoint_states = None, None
                                    alignment_report("PREFLIGHT: connecting commanded arms; no motion yet")
                                    arm_error = None
                                    preview_recovery.reset()
                                    calibration_guard.arm(tracker, now)
                                    session = MotionSession(args, planner.commanded, planner.offsets,
                                                            stop_event=stop, defer_watchdog=True)
                                    recovery_stop, recovery_reason = None, None
                                    recovery_capture_after, recovery_ready = None, False
                                    try:
                                        alignment = prepare_alignment(session, args, planner.offsets, alignment_bases, states,
                                                                      alignment_tcp_goals, read_alignment_inputs, alignment_report,
                                                                      wait=stop.wait)
                                    except (AlignmentCalibrationTimeout, ArmIKUnavailable, ArmConditioningUnavailable):
                                        session.close()
                                        session = None
                                        alignment = None
                                        now, states = time.monotonic(), None
                                        raise
                                    chosen = getattr(alignment, "targets", None)
                                    if isinstance(chosen, dict):
                                        alignment_tcp_goals = {name: pose.copy() for name, pose in chosen.items()}
                                        alignment_goals = {name: alignment_bases[name] @ flange_from_tcp(pose, planner.offsets[name])
                                                           for name, pose in chosen.items()}
                                    ready = False
                                    last_tick = time.monotonic()
                                    message = "ARMED: initial alignment starts on the next fresh tick"
                                else:
                                    message = "PREVIEW ONLY: restart with --execute to allow motion"
                        else:
                            def alignment_fresh():
                                checked = time.monotonic()
                                if stop.is_set() or not worker.is_alive() or not gui.is_alive():
                                    raise ControlFault("operator/vision stop requested")
                                if checked - now > args.max_robot_age or any(
                                        checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                    raise ControlFault("alignment state/command expired")
                            if handoff is None:
                                qdots = alignment.propose(states, alignment_bases, max(dt, 1e-6))
                                session.send_joints(qdots, alignment.predicted_joints, check_fresh=alignment_fresh,
                                                    accelerations=alignment.joint_accelerations)
                                alignment.commit()
                                message = "ALIGNING: %.1f%%; Arm calibration frozen; keep bases stationary" % (100*alignment.progress)
                                if alignment.done:
                                    endpoint_states = deepcopy(states)
                                    handoff = CalibrationHandoff(tracker, time.monotonic(),
                                                                 timeout=args.handoff_timeout,
                                                                 consensus=args.calibration_acquisition == "consensus",
                                                                 wait_at_rest=args.handoff_wait == "hold")
                                    message = "ALIGNED: waiting for fresh stable calibration; holding position"
                            else:
                                handoff.check_timeout(now)
                                qdots = alignment.propose_hold(states, alignment_bases, max(dt, 1e-6))
                                session.send_joints(qdots, alignment.predicted_joints, check_fresh=alignment_fresh,
                                                    accelerations=alignment.joint_accelerations)
                                message = ("ALIGNED: " + handoff.status
                                           + f" ({len(handoff.frames)}/{args.stable_frames} frames); holding position")
                                if handoff.ready(now):
                                    live_bases = {name: handoff.tracker.poses[name] @ planner.mappings[name]
                                                  for name in ROBOTS}
                                    live_goals, live_tcp_goals = planner.plan(handoff.tracker.poses, states)
                                    def validate_live_calibration():
                                        nominal = getattr(alignment, "nominal_targets", None)
                                        check_handoff_snapshot(
                                            args, alignment_bases, endpoint_states, live_bases, states,
                                            offsets=planner.offsets,
                                            nominal_targets=nominal if isinstance(nominal, dict) else alignment_tcp_goals,
                                            live_targets=live_tcp_goals,
                                        )
                                        planner.check_envelope(
                                            states, {name: states[name].tcp for name in planner.commanded}
                                            if tolerant else live_tcp_goals,
                                        )
                                        for name in planner.commanded:
                                            bounded_twist(
                                                states[name].tcp, live_tcp_goals[name], np.zeros(6), min(dt, 2 * period),
                                                kp_linear=args.kp_linear, kp_angular=args.kp_angular,
                                                max_linear=args.max_speed, max_angular=args.max_angular_speed,
                                                max_accel=args.max_accel, max_angular_accel=args.max_angular_accel,
                                                max_target_m=args.max_target_m,
                                                max_target_rad=np.deg2rad(args.max_target_deg),
                                                **task_tolerance_options(args, planner.offsets[name]),
                                            )
                                    if handoff.try_validate(now, validate_live_calibration):
                                        tracker = handoff.tracker
                                        calibration_guard.arm(tracker, now)
                                        calibration_degraded = False
                                        alignment_complete = True
                                        previous = {name: np.zeros(6) for name in planner.commanded}
                                        goals, tcp_goals = live_goals, live_tcp_goals
                                        message = "ALIGNED: fresh calibration validated; starting live hold on next tick"
                                    else:
                                        message = "ALIGNED: " + handoff.status + "; holding position"
                    else:
                        twists = {
                            name: bounded_twist(
                                states[name].tcp, tcp_goals[name], previous[name], min(dt, 2 * period),
                                kp_linear=args.kp_linear, kp_angular=args.kp_angular,
                                max_linear=args.max_speed, max_angular=args.max_angular_speed,
                                max_accel=args.max_accel, max_angular_accel=args.max_angular_accel,
                                max_target_m=args.max_target_m, max_target_rad=np.deg2rad(args.max_target_deg),
                                **task_tolerance_options(args, planner.offsets[name]),
                            ) for name in planner.commanded
                        }
                        if getattr(args, "target_tolerance_m", 0.) or getattr(args, "target_tolerance_deg", 0.):
                            display_command_targets = near_term_tcp_targets(states, twists, min(dt, 2 * period))
                        ready = planner.target_applied and session is None
                        label = "ACTIVE" if session else "READY (preview)" if ready else "PREVIEW: Apply target in GUI"
                        message = f"{label} {args.mode}; gap={planner.gap:.3f}m"
                        if arm_requested and session is None:
                            if not ready:
                                message = "WAIT: Apply the hold target in GUI before arming"
                            elif args.execute:
                                arm_error = None
                                preview_recovery.reset()
                                calibration_guard.arm(tracker, now)
                                session = MotionSession(args, planner.commanded, planner.offsets, stop_event=stop)
                                recovery_stop, recovery_reason = None, None
                                recovery_capture_after, recovery_ready = None, False
                                ready = False
                                # Do not move on the connection tick: require fresh vision/state again.
                                previous = {name: np.zeros(6) for name in planner.commanded}
                                last_tick = time.monotonic()
                                message = "ARMED; waiting for next validated control tick"
                            else:
                                message = "PREVIEW ONLY: restart with --execute to allow motion"
                        elif session is not None:
                            def check_fresh():
                                checked = time.monotonic()
                                if stop.is_set() or not worker.is_alive() or not gui.is_alive():
                                    raise ControlFault("operator/vision stop requested")
                                calibration_guard.check(tracker, checked)
                                if checked - now > args.max_robot_age:
                                    raise ControlFault("state/command expired during controller checks")
                                if any(checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                    raise ControlFault("robot state expired during controller checks")
                            if getattr(args, "target_tolerance_m", 0.) or getattr(args, "target_tolerance_deg", 0.):
                                next_targets = display_command_targets
                                planner.check_envelope(states, next_targets)
                                for fraction in (.5, 1.):
                                    intermediate = (next_targets if fraction == 1. else
                                                    near_term_tcp_targets(states, twists, fraction * min(dt, 2 * period)))
                                    future_flanges = {
                                        name: tracker.poses[name] @ planner.mappings[name]
                                        @ flange_from_tcp(intermediate.get(name, states[name].tcp), planner.offsets[name])
                                        for name in ROBOTS
                                    }
                                    if np.linalg.norm(future_flanges[ROBOTS[0]][:3, 3]
                                                      - future_flanges[ROBOTS[1]][:3, 3]) < args.min_gap:
                                        raise PlanningUnavailable("next commanded flange separation is below min_gap")
                                # Native feasibility applies to what speedL
                                # actually commands, while goals retain the
                                # immutable nominal task for error display.
                                session.send({name: states[name].tcp for name in planner.commanded}, twists,
                                             check_fresh=check_fresh, safety_targets=next_targets)
                            else:
                                session.send(tcp_goals, twists, check_fresh=check_fresh)
                            previous = twists
                    if session is not None:
                        degraded = not calibration_current
                        if degraded != calibration_degraded:
                            event = ("CALIBRATION HOLD: rejected pair; using last accepted calibration"
                                     if degraded else "CALIBRATION RECOVERED: accepted new frame")
                            print("[control] " + event, flush=True)
                            calibration_degraded = degraded
                        if degraded:
                            message += "; CALIBRATION HOLD: waiting for next valid frame"
                    if session is None and arm_error is not None:
                        message = "ARM BLOCKED: " + arm_error
                    if recovery_stop is not None:
                        recovery_ready = ready
                        if ready:
                            message = "READY: stopped; fresh calibration validated; review pose, then press Arm again"
                except (ControlFault, ValueError, KeyError, TypeError) as exc:
                    if session is not None:
                        if (getattr(args, "fault_response", "exit") != "preview" or not is_recoverable_fault(exc)
                                or stop.is_set() or not worker.is_alive() or not gui.is_alive()):
                            raise ControlFault(str(exc)) from exc
                        # A failed stop/transport confirmation is fatal. Do not
                        # enter recoverable preview until every arm confirms.
                        session.stop_for_recovery()
                        session = None
                        now = time.monotonic()
                        if stop.is_set():
                            break
                        baseline = {name: reader.read(now) for name, reader in readers.items()}
                        recovery_stop = RecoveryStopGate(planner.commanded, baseline, now,
                                                        max_state_age=args.max_robot_age)
                        recovery_reason = str(exc)
                        recovery_capture_after, recovery_ready = None, False
                        inbox.cancel_arm_requests()
                        take_latest(frames)
                        calibration_guard = ActiveCalibrationGuard(args.max_invalid_duration)
                        preview_recovery.reset()
                        calibration_degraded = False
                        alignment, alignment_complete = None, False
                        alignment_bases, alignment_goals, alignment_tcp_goals = None, None, None
                        alignment_nominal_goals, alignment_capture = None, None
                        handoff, endpoint_states = None, None
                        previous = {name: np.zeros(6) for name in planner.commanded}
                        arm_error = None
                        last_tick = now
                        print("[control] RECOVERY STOP: " + recovery_reason +
                              "; motion stopped; preview remains open; a new Arm is required", flush=True)
                    elif (recovery_stop is not None and isinstance(exc, ControlFault) and not is_recoverable_fault(exc)
                          and not isinstance(exc, (ArmConnectionError, AlignmentCalibrationTimeout,
                                                   ArmIKUnavailable, ArmConditioningUnavailable))):
                        raise
                    if isinstance(exc, (ArmConnectionError, AlignmentCalibrationTimeout,
                                        ArmIKUnavailable, ArmConditioningUnavailable)):
                        inbox.cancel_arm_requests()
                        arm_error = str(exc)
                        print("[control] ARM BLOCKED: " + arm_error, flush=True)
                    ready = False
                    message = "WAIT: " + str(exc)
                    if session is None and preview_recovery.count:
                        message = (f"WAIT: PREVIEW REACQUIRING ({preview_recovery.count}/{args.stable_frames} frames); "
                                   + str(exc))
                    if recovery_stop is not None:
                        recovery_ready = False
                        inbox.cancel_arm_requests()
                        message = "RECOVERY WAIT: " + str(exc) + "; new Arm required after readiness"
                status = {"message": message, "goals": goals, "ready": ready, "arm_error": arm_error,
                          "active": session is not None, "center_pose": planner.center_pose,
                          "calibration_degraded": calibration_degraded,
                          "preview_reacquiring_frames": preview_recovery.count,
                          "recovery_required": recovery_stop is not None,
                          "recovery_reason": recovery_reason,
                          "recovery_stopped": recovery_stop.ready if recovery_stop is not None else False,
                          "phase": ("recovery" if recovery_stop is not None else
                                    "alignment_calibration_handoff" if handoff is not None and not alignment_complete
                                    else "alignment" if args.initial_align and not alignment_complete else "tracking"),
                          "alignment_progress": alignment.progress if alignment is not None else None}
                # A rejected preview may still have useful accepted calibration
                # and live encoder poses. These reads never authorize motion.
                if states is None:
                    try:
                        states = {name: reader.read(now) for name, reader in readers.items()}
                    except (ControlFault, ValueError, KeyError, TypeError):
                        states = None
                if session is not None and args.initial_align and not alignment_complete:
                    scene = make_gui_scene(
                        now, alignment_bases, states, planner.offsets, alignment_goals,
                        max(0., now - alignment_capture), readers, scene_timestamps, args.max_robot_age,
                        calibration_valid=True, reason="Arm calibration frozen; live encoder positions",
                        calibration_mode="frozen", calibration_captured_at=alignment_capture,
                        nominal_goals=alignment_nominal_goals, command_goals=alignment_goals,
                    )
                else:
                    scene = facing_gui_scene(now, tracker, planner, states, readers,
                                             scene_timestamps, args.max_robot_age, active=session is not None,
                                             command_tcp_goals=display_command_targets)
                put_latest(statuses, status)
                put_latest(gui_statuses, {**status, "scene": scene})
                if now - last_report >= 1.:
                    print("[control] " + message, flush=True)
                    last_report = now
                stop.wait(max(0., period - (time.monotonic() - last_tick)))
    finally:
        # Stop motion before waiting for inference or camera resources to close.
        if session is not None:
            session.close()
        stop.set()
        for child in (worker, gui):
            if child.pid is not None:
                child.join(timeout=3.)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=1.)
        for queue in (frames, commands, statuses, gui_statuses):
            queue.cancel_join_thread()
            queue.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.simulate:
            from urcal.simulation.runtime import run as run_simulation
            run_simulation(args)
        elif args.mode == "follow":
            from urcal.cooperative.runtime import run as run_research
            run_research(args)
        else:
            run(args)
    except KeyboardInterrupt:
        print("[info] stopped by operator", flush=True)
    except Exception as exc:
        diagnostic = fault_transport_summary(exc)
        if diagnostic:
            print("[RTDE at fault] " + diagnostic, flush=True)
        raise SystemExit(f"[STOP] {exc}. Motion is not automatically resumed.") from exc


if __name__ == "__main__":
    main()
