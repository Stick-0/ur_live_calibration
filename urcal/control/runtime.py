"""Live-calibrated flange facing: fixed camera pose or read-only leader/follower.

Preview never constructs RTDEControlInterface. Motion needs --execute AND Arm
in the GUI (or 'a' in the preview). No hardware connection occurs on import. The controller
is a bounded Cartesian proportional regulator, not a collision-aware planner.
UR API: https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html
"""

import argparse
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
from urcal.control.policy import CalibrationTracker, ControlFault, bounded_twist


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


class MotionSession:
    """Own only commanded arms; all RTDEControl calls are serialized here."""

    def __init__(self, args, names, offsets, factory=None, stop_event=None):
        self.args, self.offsets = args, offsets
        self.controls = {}
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
                self.controls[name] = factory(getattr(args, name + "_ip"))
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
                if not control.setWatchdog(args.watchdog_hz):
                    raise ControlFault(f"{name}: failed to configure controller watchdog")
        except BaseException:
            self.close()
            raise

    def _check_offset(self, name, control):
        active = pose_matrix(control.getTCPOffset())
        if (np.linalg.norm(active[:3, 3] - self.offsets[name][:3, 3]) > 1e-5
                or rotation_distance(active, self.offsets[name]) > 1e-5):
            raise ControlFault(f"{name}: active TCP differs from --{name}_tcp_offset")

    def send(self, targets, twists, check_fresh=None, safety_targets=None):
        # Validate the complete pair before sending either arm a motion command.
        for name, control in self.controls.items():
            if not control.isConnected() or not control.isProgramRunning():
                raise ControlFault(f"{name}: control connection/program stopped")
            self._check_offset(name, control)
            if not control.isPoseWithinSafetyLimits(matrix_pose(targets[name]).tolist()):
                raise ControlFault(f"{name}: goal is unreachable or outside controller safety limits")
            if safety_targets is not None and not control.isPoseWithinSafetyLimits(matrix_pose(safety_targets[name]).tolist()):
                raise ControlFault(f"{name}: next goal is unreachable or outside controller safety limits")
        for name, control in self.controls.items():
            if check_fresh is not None:
                check_fresh()
            # Positive time bounds the robot-side speed thread's update period,
            # NOT the lifetime of a velocity. Each speedL packet feeds watchdog;
            # an extra kick would write NO_CMD into the same command register.
            if not control.speedL(twists[name].tolist(), self.args.max_accel, 0.002):
                raise ControlFault(f"{name}: speedL failed")

    def close(self):
        controls, self.controls = self.controls, {}
        # Request both stops before any script shutdown/disconnect can block.
        for name, control in controls.items():
            try:
                control.speedStop(self.args.stop_accel)
            except Exception as exc:
                print(f"[warn] {name} speedStop: {exc}", flush=True)
        for name, control in controls.items():
            for operation in (
                lambda c=control: c.stopScript(),
                lambda c=control: c.disconnect(),
            ):
                try:
                    operation()
                except Exception as exc:
                    print(f"[warn] {name} cleanup: {exc}", flush=True)


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
            raise ControlFault(f"flange separation {separation:.3f} m is below min_gap")
        if self.gap is None:
            self.gap = float(separation)
        if self.gap < self.args.min_gap:
            raise ControlFault("target gap is below min_gap")
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
                    raise ControlFault(f"{name}: outside {self.args.max_travel:.3f} m start-position envelope")
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
    for name, folder in (("ur5e", "v3m2a3"), ("ur10e", "ur10e_2")):
        p.add_argument("--" + name + "_ip", required=True)
        p.add_argument("--" + name + "_tcp_offset", nargs=6, type=float,
                       help="Actual flange-to-active-TCP offset x y z rx ry rz; required with --execute")
        p.add_argument("--" + name + "_checkpoint", type=Path, default=BUNDLE / "trained_models" / folder / "best.pt")
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
    p.add_argument("--filter_alpha", type=float, default=.2)
    p.add_argument("--max_reproj_px", type=float, default=3.)
    p.add_argument("--max_point_reproj_px", type=float, default=8.)
    p.add_argument("--min_points", type=int, default=5)
    p.add_argument("--max_calibration_jump_m", type=float, default=.015)
    p.add_argument("--max_calibration_jump_deg", type=float, default=2.)
    p.add_argument("--frame_check_m", type=float, default=.03)
    p.add_argument("--frame_check_deg", type=float, default=5.)
    p.add_argument("--min_gap", type=float, default=.15)
    p.add_argument("--max_travel", type=float, default=.15)
    p.add_argument("--max_target_m", type=float, default=.10)
    p.add_argument("--max_target_deg", type=float, default=25.)
    p.add_argument("--max_speed", type=float, default=.02)
    p.add_argument("--max_angular_speed", type=float, default=.1)
    p.add_argument("--max_accel", type=float, default=.1)
    p.add_argument("--max_angular_accel", type=float, default=.5)
    p.add_argument("--stop_accel", type=float, default=.5)
    p.add_argument("--kp_linear", type=float, default=1.)
    p.add_argument("--kp_angular", type=float, default=1.)
    research = p.add_argument_group("moving-base relative collaboration (--mode follow)")
    research.add_argument("--method", choices=("direct", "predict", "separated"), default="separated")
    research.add_argument("--path", choices=("hold", "line", "circle"), default="hold")
    research.add_argument("--path_delta", nargs=6, type=float, default=[0, 0, -.05, 0, 0, 0],
                          help="Line delta in leader flange axes: XYZ metres, rotation vector radians")
    research.add_argument("--path_radius", type=float, default=.02)
    research.add_argument("--path_duration", type=float, default=10.)
    research.add_argument("--velocity_alpha", type=float, default=.3)
    research.add_argument("--prediction_horizon", type=float, default=.5)
    research.add_argument("--max_base_speed", type=float, default=.15)
    research.add_argument("--max_base_angular_speed", type=float, default=.5)
    research.add_argument("--max_invalid_duration", type=float, default=.3)
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
    positive = ("control_hz", "watchdog_hz", "max_calibration_age", "max_joint_skew", "max_robot_age",
                "stable_frames", "max_reproj_px", "max_point_reproj_px", "max_calibration_jump_m",
                "max_calibration_jump_deg", "frame_check_m", "frame_check_deg", "min_gap", "max_travel",
                "max_target_m", "max_target_deg", "max_speed", "max_angular_speed", "max_accel",
                "max_angular_accel", "stop_accel", "kp_linear", "kp_angular", "camera_w", "camera_h", "fps",
                "path_radius", "path_duration", "prediction_horizon", "max_base_speed", "max_base_angular_speed",
                "max_invalid_duration", "max_joint_speed", "max_joint_accel", "min_singular_value",
                "feedback_speed", "feedback_angular_speed", "slow_error_m", "slow_error_deg",
                "finish_position_m", "finish_rotation_deg", "finish_dwell")
    if any(not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        p.error("rates, limits, camera dimensions and gains must be finite and positive")
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
    if args.ur5e_ip == args.ur10e_ip:
        p.error("UR5e and UR10e must have different controller IPs")
    for name in ROBOTS:
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

    ctx = mp.get_context("spawn")
    frames, commands, statuses, gui_statuses = (ctx.Queue(maxsize=size) for size in (1, 8, 1, 1))
    stop = ctx.Event()
    worker = ctx.Process(target=vision_worker, args=(args, frames, commands, statuses, stop), daemon=True)
    gui = ctx.Process(target=gui_worker, args=(args, commands, gui_statuses, stop), daemon=True)
    tracker = CalibrationTracker(
        max_age=args.max_calibration_age, stable_frames=args.stable_frames, alpha=args.filter_alpha,
        max_jump_m=args.max_calibration_jump_m, max_jump_rad=np.deg2rad(args.max_calibration_jump_deg),
        max_mean_px=args.max_reproj_px, max_pixel_px=args.max_point_reproj_px, min_points=args.min_points,
    )
    planner = FacingPlanner(args)
    session = None
    previous = {name: np.zeros(6) for name in planner.commanded}
    last_tick = time.monotonic()
    period = 1.0 / args.control_hz
    last_report = 0.
    try:
        with ExitStack() as cleanup:
            readers = {}
            for name in ROBOTS:
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                cleanup.callback(receiver.disconnect)
                readers[name] = StateReader(receiver, args.max_robot_age)
            cleanup.callback(lambda: session.close() if session is not None else None)
            # Import nominal FK outside the timed motion loop (no robot writes).
            for name in ROBOTS:
                model_flange_pose(name, np.zeros(6))
            gui.start()
            worker.start()
            print("[info] PREVIEW. GUI: Apply target, then Arm (--execute only). SPACE/q/ESC: stop+exit.", flush=True)
            if any(getattr(args, name + "_tcp_offset") is None for name in ROBOTS):
                print("[info] preview assumes zero TCP offsets where omitted; specify actual offsets before motion.", flush=True)
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
                ready = False
                try:
                    while True:
                        try:
                            command = commands.get_nowait()
                        except Empty:
                            break
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
                    if packet is not None:
                        tracker.update(packet, now)
                    tracker.check_fresh(now)
                    states = {name: reader.read(now) for name, reader in readers.items()}
                    goals, tcp_goals = planner.plan(tracker.poses, states)
                    planner.check_envelope(states, tcp_goals)
                    twists = {
                        name: bounded_twist(
                            states[name].tcp, tcp_goals[name], previous[name], min(dt, 2 * period),
                            kp_linear=args.kp_linear, kp_angular=args.kp_angular,
                            max_linear=args.max_speed, max_angular=args.max_angular_speed,
                            max_accel=args.max_accel, max_angular_accel=args.max_angular_accel,
                            max_target_m=args.max_target_m, max_target_rad=np.deg2rad(args.max_target_deg),
                        ) for name in planner.commanded
                    }
                    ready = planner.target_applied and session is None
                    label = "ACTIVE" if session else "READY (preview)" if ready else "PREVIEW: Apply target in GUI"
                    message = f"{label} {args.mode}; gap={planner.gap:.3f}m"
                    if arm_requested and session is None:
                        if not ready:
                            message = "WAIT: Apply the hold target in GUI before arming"
                        elif args.execute:
                            session = MotionSession(args, planner.commanded, planner.offsets, stop_event=stop)
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
                            if stop.is_set():
                                raise ControlFault("operator/vision stop requested")
                            tracker.check_fresh(checked)
                            if checked - now > args.max_robot_age:
                                raise ControlFault("state/command expired during controller checks")
                            if any(checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                raise ControlFault("robot state expired during controller checks")
                        session.send(tcp_goals, twists, check_fresh=check_fresh)
                        previous = twists
                except (ControlFault, ValueError, KeyError, TypeError) as exc:
                    if session is not None:
                        raise ControlFault(str(exc)) from exc
                    ready = False
                    message = "WAIT: " + str(exc)
                status = {"message": message, "goals": goals, "ready": ready,
                          "active": session is not None, "center_pose": planner.center_pose}
                put_latest(statuses, status)
                put_latest(gui_statuses, status)
                if now - last_report >= 1.:
                    print("[control] " + message, flush=True)
                    last_report = now
                stop.wait(max(0., period - (time.monotonic() - now)))
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
        if args.mode == "follow":
            from urcal.cooperative.runtime import run as run_research
            run_research(args)
        else:
            run(args)
    except KeyboardInterrupt:
        print("[info] stopped by operator", flush=True)
    except Exception as exc:
        raise SystemExit(f"[STOP] {exc}. Motion is not automatically resumed.") from exc


if __name__ == "__main__":
    main()
