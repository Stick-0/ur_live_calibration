"""Live-calibrated flange facing with a read-only leader and controlled follower.

Preview never constructs RTDEControlInterface. Motion needs --execute AND Arm
in the GUI (or 'a' in the preview). No hardware connection occurs on import.
The follower runtime checks bounded velocity commands and nominal link capsules.
UR API: https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
import time

import numpy as np

from urcal.control.geometry import inverse, matrix_pose, pose_matrix, rotation_distance
from urcal.control.policy import ControlFault
from urcal.control.transport_diagnostics import fault_transport_summary


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("follow",), default="follow", help="Follower control (default; retained for existing commands)")
    p.add_argument("--leader", choices=ROBOTS, default="ur10e")
    p.add_argument("--gap", type=float, default=.20, help="Flange origin separation in metres (default: 0.20); editable in GUI")
    p.add_argument("--execute", action="store_true", help="Allow motion after GUI Arm or a in preview; otherwise receive-only")
    p.add_argument("--initial_align", action="store_true",
                   help="After explicit Arm, validate and slowly align from rest before following; keep leader/bases stationary")
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
                   help="Capture current joints/calibration once, disconnect hardware, then run a virtual follower simulation")
    p.add_argument("--initial_state", type=Path,
                   help="Saved simulation initial-state JSON; with --simulate, skips all camera/RTDE connections")
    p.add_argument("--sim_initial_timeout", type=float, default=60., help="Initial live snapshot timeout in seconds")
    p.add_argument("--sim_motion", choices=("none", "leader", "bases", "both"), default="none",
                   help="Virtual external motion after Start simulation")
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
    p.add_argument("--max_robot_age", type=float, default=.5)
    p.add_argument("--stable_frames", type=int, default=3)
    p.add_argument("--calibration_acquisition", choices=("consensus", "consecutive"), default="consensus",
                   help="Stationary initial-alignment acquisition: recent 1s consensus or consecutive raw frames")
    p.add_argument("--filter_alpha", type=float, default=.2)
    output_correction = p.add_mutually_exclusive_group()
    output_correction.add_argument("--ur10e_output_correction", action="store_true",
                                   help="Apply Z 180deg and local Z +5mm to UR10e PnP output before control (default: ON)")
    output_correction.add_argument("--no_ur10e_output_correction", action="store_false",
                                   dest="ur10e_output_correction",
                                   help="Use raw UR10e PnP output as the control base without the output transform")
    p.set_defaults(ur10e_output_correction=True)
    plane = p.add_mutually_exclusive_group()
    plane.add_argument("--common_base_plane", action="store_true",
                       help="Estimate both physical bases on a shared mounting plane with matching +Z normals (default)")
    plane.add_argument("--no_common_base_plane", dest="common_base_plane", action="store_false",
                       help="Disable the shared base-plane constraint for independently mounted bases")
    p.set_defaults(common_base_plane=True)
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
    p.add_argument("--max_travel", type=float,
                   help="TCP radius from the original start position in metres (default: 0.30)")
    p.add_argument("--max_target_m", type=float, default=.20,
                   help="Tracking position error stop threshold in metres (default: 0.20); does not change travel/speed limits")
    p.add_argument("--max_target_deg", type=float, default=25.)
    p.add_argument("--target_tolerance_m", type=float, default=.05,
                   help="Allowed estimated flange target adjustment/error in metres (default: 0.05); not calibration accuracy")
    p.add_argument("--target_tolerance_deg", type=float, default=10.,
                   help="Allowed estimated flange orientation adjustment/error in degrees (default: 10); zero with zero position tolerance restores exact goals")
    p.add_argument("--max_speed", type=float,
                   help="TCP linear limit in m/s for initial alignment (default: 0.02); "
                        "an explicit value also sets following unless --follow_max_speed is supplied")
    p.add_argument("--max_angular_speed", type=float, default=.1)
    p.add_argument("--max_accel", type=float, default=.1)
    p.add_argument("--max_angular_accel", type=float, default=.5)
    p.add_argument("--stop_accel", type=float, default=.5)
    p.add_argument("--fault_response", choices=("preview", "exit"), default="preview",
                   help="Recoverable calibration/planning faults: stop and keep preview for a fresh explicit Arm (default), "
                        "or exit as in earlier versions; hardware/operator faults still stop and exit")
    p.add_argument("--kp_linear", type=float,
                   help="Position error feedback gain in 1/s (default: 3)")
    p.add_argument("--kp_angular", type=float,
                   help="Orientation error feedback gain in 1/s (default: 3)")
    research = p.add_argument_group("moving-base relative collaboration (--mode follow)")
    research.add_argument("--method", choices=("direct", "predict", "separated"), default="separated")
    research.add_argument("--follow_law", choices=("feedforward", "pose"), default="feedforward",
                          help="Tracking law: feedforward (default) combines leader tracking and pose feedback; "
                               "pose recomputes the follower TCP target each cycle and uses only pose-error velocity "
                               "with measured-motion damping. Requires separated + pose base updates + bounded response; "
                               "all existing feedback and motion limits remain unchanged")
    research.add_argument("--base_motion_mode", choices=("pose", "velocity"), default="pose",
                          help="Visual base updates: pose (default) updates coordinates without inferred velocity; "
                               "velocity restores research prediction and mandatory base-velocity compensation. "
                               "Robot motion limits always use encoder/RTDE measurements")
    research.add_argument("--base_pose_window", type=int, default=5,
                          help="ACTIVE pose-mode relative-base medoid window: odd 3..31 images, 0 disables")
    research.add_argument("--base_pose_smoothing_time", type=float, default=.35,
                          help="Relative-base coordinate smoothing time constant in seconds; not leader motion filtering")
    research.add_argument("--base_pose_deadband_m", type=float, default=.005,
                          help="Ignore relative-base coordinate jitter within this translation radius (default: 5mm)")
    research.add_argument("--base_pose_deadband_deg", type=float, default=.5,
                          help="Ignore relative-base coordinate jitter within this angle (default: 0.5deg)")
    research.add_argument("--base_pose_update_speed", type=float, default=.02,
                          help="Relative-base coordinate application rate in m/s (default: 0.02); not a robot speed limit")
    research.add_argument("--base_pose_update_angular_speed", type=float, default=.05,
                          help="Relative-base coordinate application rate in rad/s (default: 0.05)")
    research.add_argument("--follow_max_speed", type=float,
                          help="Follower tracking TCP linear limit in m/s (default: 1.0, or explicit --max_speed); "
                               "does not change initial alignment or acceleration limits")
    research.add_argument("--leader_response", choices=("bounded", "strict"), default="bounded",
                          help="Separated following with target tolerance: bounded ramps leader tracking within motion limits; "
                               "strict requires immediate full leader compensation for research comparisons")
    research.add_argument("--path", choices=("hold", "line", "circle"), default="hold")
    research.add_argument("--path_delta", nargs=6, type=float, default=[0, 0, -.05, 0, 0, 0],
                          help="Line delta in leader flange axes: XYZ metres, rotation vector radians")
    research.add_argument("--path_radius", type=float, default=.02)
    research.add_argument("--path_duration", type=float, default=10.)
    research.add_argument("--approach_speed", type=float, default=.005,
                          help="peak axial target speed in m/s for the GUI approach to 15 cm")
    research.add_argument("--velocity_alpha", type=float, default=.3)
    research.add_argument("--prediction_horizon", type=float, default=.5,
                          help="Maximum base-velocity extrapolation age in seconds (velocity mode only)")
    research.add_argument("--max_base_speed", type=float, default=.25,
                          help="Visual base speed rejection in m/s (velocity mode only; not robot TCP speed)")
    research.add_argument("--max_base_angular_speed", type=float, default=.15,
                          help="Visual base angular speed rejection in rad/s (velocity mode only)")
    research.add_argument("--max_invalid_duration", type=float, default=.3,
                          help="Active rejected-image grace while the last accepted capture remains fresh, seconds")
    recovery = research.add_mutually_exclusive_group()
    recovery.add_argument("--live_calibration_recovery", dest="live_calibration_recovery", action="store_true",
                          help="Bounded pose following: brake and observe consistent new calibration before resuming (default)")
    recovery.add_argument("--no_live_calibration_recovery", dest="live_calibration_recovery", action="store_false",
                          help="Use legacy immediate calibration expiry/rejection faults")
    p.set_defaults(live_calibration_recovery=True)
    research.add_argument("--calibration_recovery_timeout", type=float, default=2.,
                          help="Maximum observation/braking time from detection, seconds (default: 2)")
    research.add_argument("--calibration_recovery_max_shift_m", type=float, default=.15,
                          help="Maximum new calibration displacement from the observation baseline (default: 0.15m)")
    research.add_argument("--calibration_recovery_max_shift_deg", type=float, default=15.,
                          help="Maximum new calibration rotation from the observation baseline (default: 15deg)")
    research.add_argument("--calibration_blend_time", type=float, default=.5,
                          help="Minimum time to blend recovered calibration; rate limits may extend it (default: 0.5s)")
    research.add_argument("--max_joint_speed", type=float, default=.2)
    research.add_argument("--max_joint_accel", type=float,
                          help="Per-joint acceleration limit in rad/s^2 (default: 0.75)")
    research.add_argument("--min_singular_value", type=float, default=.01)
    research.add_argument("--feedback_speed", type=float,
                          help="Follow catch-up linear feedback cap in m/s (default: 0.10); "
                               "total commands still obey TCP/joint speed and acceleration limits")
    research.add_argument("--feedback_angular_speed", type=float,
                          help="Follow catch-up angular feedback cap in rad/s (default: 0.15); "
                               "total commands still obey TCP/joint speed and acceleration limits")
    research.add_argument("--follow_damping_time", type=float, default=.12,
                          help="Measured relative-motion anticipation in seconds (default: 0.12); "
                               "bounded following brakes near the target; 0 selects legacy feedback")
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
    if args.follow_law == "pose" and (args.method != "separated"
                                    or args.base_motion_mode != "pose"
                                    or args.leader_response != "bounded"):
        p.error("--follow_law pose requires --method separated --base_motion_mode pose --leader_response bounded")
    if args.follow_max_speed is None:
        args.follow_max_speed = 1. if args.max_speed is None else args.max_speed
    if args.max_speed is None:
        args.max_speed = .02
    for name, default in {"kp_linear": 3., "kp_angular": 3., "feedback_speed": .10,
                          "feedback_angular_speed": .15, "max_joint_accel": .75, "max_travel": .30}.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.simulate and args.initial_align:
        p.error("--initial_align is for live control; simulation already provides its own initial alignment")
    if args.simulate and args.execute:
        p.error("--simulate and --execute are mutually exclusive; simulation never commands hardware")
    if args.initial_state is not None and not args.simulate:
        p.error("--initial_state requires --simulate")
    if not args.simulate and args.sim_motion != "none":
        p.error("--sim_motion requires --simulate")
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
                "path_radius", "path_duration", "approach_speed", "prediction_horizon", "max_base_speed", "max_base_angular_speed",
                "max_invalid_duration", "max_joint_speed", "max_joint_accel", "min_singular_value",
                "feedback_speed", "feedback_angular_speed", "slow_error_m", "slow_error_deg",
                "finish_position_m", "finish_rotation_deg", "finish_dwell", "calibration_recovery_timeout",
                "calibration_recovery_max_shift_m", "calibration_recovery_max_shift_deg", "calibration_blend_time",
                "base_pose_smoothing_time", "base_pose_update_speed", "base_pose_update_angular_speed")
    if any(not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        p.error("rates, limits, camera dimensions and gains must be finite and positive")
    if not np.isfinite(args.follow_damping_time) or args.follow_damping_time < 0:
        p.error("follow_damping_time must be finite and nonnegative")
    if args.base_pose_window != 0 and (args.base_pose_window < 3 or args.base_pose_window > 31
                                      or args.base_pose_window % 2 != 1):
        p.error("base_pose_window must be 0 or an odd integer from 3 to 31")
    if (not np.isfinite([args.base_pose_deadband_m, args.base_pose_deadband_deg]).all()
            or min(args.base_pose_deadband_m, args.base_pose_deadband_deg) < 0):
        p.error("base pose deadbands must be finite and nonnegative")
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
    if not np.isfinite(args.path_delta).all() or not 0 < args.velocity_alpha <= 1:
        p.error("invalid path_delta or velocity_alpha")
    if (not np.isfinite([args.position_deadband, args.rotation_deadband_deg]).all()
            or min(args.position_deadband, args.rotation_deadband_deg) < 0):
        p.error("deadbands must be finite and nonnegative")
    if args.slow_error_m >= args.max_target_m or args.slow_error_deg >= args.max_target_deg:
        p.error("slow_error thresholds must be below final target error limits")
    if args.base_motion_mode == "velocity" and args.prediction_horizon > args.max_calibration_age:
        p.error("prediction_horizon must not exceed max_calibration_age")
    return args


def run(args):
    """Delegate every live entry point to the single follower controller."""
    if args.mode != "follow":
        raise ValueError("Only follower mode is supported")
    from urcal.cooperative.runtime import run as run_follower
    return run_follower(args)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.simulate:
            from urcal.simulation.runtime import run as run_simulation
            run_simulation(args)
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
