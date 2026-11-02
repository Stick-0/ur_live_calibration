"""Moving-base relative collaboration from Research_Implementation_Plan.txt.

Uses the unchanged pose sensor and the same speedL low-level RTDE interface.
Optional --initial_align performs a checked speedJ approach before tracking.
The default is receive-only preview; --execute plus operator Arm is required.
"""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import time
import uuid

import numpy as np

from urcal.control.runtime import BUNDLE, ROBOTS, ArmConditioningUnavailable, ArmConnectionError, ArmIKUnavailable, CommandInbox, MotionSession, StateReader, make_gui_scene, parse_args, put_latest, take_latest
from urcal.control.geometry import flange_from_tcp, inverse
from urcal.cooperative.controller import CalibrationBrakingRequired, CooperativeController, ResearchFault
from urcal.cooperative.logging import ExperimentLogger
from urcal.control.policy import ActiveCalibrationGuard, CalibrationUnavailable, ControlFault
from urcal.control.recovery import RecoveryStopGate, is_recoverable_fault
from urcal.control.handoff import CalibrationHandoff
from urcal.control.startup import (
    AlignmentCalibrationTimeout,
    check_alignment_request, check_handoff_snapshot, prepare_alignment,
)
from urcal.cooperative.diagnostics import WaitDiagnostics, fault_diagnostics
from urcal.control.transport_diagnostics import capture_runtime_fault, fault_transport_summary, transport_record
from urcal.control.tracking_error import position_error_sample
from urcal.control.error_recording import PositionErrorRecorder


def finite_record(value):
    """Preserve rejected measurement structure, marking nonfinite values null."""
    if isinstance(value, np.ndarray):
        return finite_record(value.tolist())
    if isinstance(value, dict):
        return {key: finite_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_record(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def cooperative_gui_scene(now, engine, states, readers, previous_timestamps, max_robot_age,
                          *, proposal=None, active=False, goals_override=None, bases_override=None,
                          calibration_captured_at=None, nominal_goals_override=None):
    """Read-only target preview independent of IK, envelope and velocity feasibility."""
    if states is None:
        return None
    try:
        calibration_valid = engine.invalid_since is None
        reason = engine.last_rejection
        if bases_override is not None:
            bases = bases_override
            capture = engine.gate.last_capture if calibration_captured_at is None else calibration_captured_at
            age = now - capture
            calibration_valid, reason = True, "Arm calibration frozen; camera and bases assumed fixed"
            goals = goals_override or {}
        elif proposal is None:
            try:
                snapshot = engine.state(now, active=active)
            except ResearchFault as exc:
                # A newly accepted pair can be useful for the display while
                # the control gate is still accumulating stable frames. Never
                # use this path after a rejected sample or beyond estimator age
                # limits, and never change the gate or its stability counter.
                if exc.category != "measurement_unavailable" or engine.invalid_since is not None:
                    raise
                snapshot = engine.estimator.estimate(now, leader=engine.leader)
                if getattr(engine, "common_base_plane", False):
                    from urcal.cooperative.base_plane import constrain_base_snapshot
                    snapshot = constrain_base_snapshot(snapshot, engine.leader, engine.mappings)
                calibration_valid, reason = False, str(exc)
            bases, age = snapshot.base_poses, snapshot.measurement_age
            leader_flange = bases[engine.leader] @ flange_from_tcp(
                states[engine.leader].tcp, engine.offsets[engine.leader])
            goals = {engine.follower: leader_flange @ engine.path.pose(engine.progress)}
        else:
            bases, age = proposal.record["bases"], proposal.record["measurement_age"]
            goals = proposal.goals
        if goals_override is not None:
            goals = goals_override
        command_goals = goals_override if bases_override is not None else None
        if proposal is not None and "next_commanded_target_tcp" in proposal.record:
            command_goals = {engine.follower: bases[engine.follower]
                             @ flange_from_tcp(proposal.record["next_commanded_target_tcp"],
                                               engine.offsets[engine.follower])}
        task = {"leader": engine.leader, "kind": engine.path.kind, "delta": engine.path.delta.copy(),
                "radius": engine.path.radius, "duration": engine.path.duration,
                "gap": engine.gap, "progress": engine.progress, "start_pose": engine.path.start.copy()}
        scene = make_gui_scene(now, bases, states, engine.offsets, goals, age, readers,
                               previous_timestamps, max_robot_age,
                               calibration_valid=calibration_valid, reason=reason,
                               calibration_mode="frozen" if bases_override is not None else "live",
                               calibration_captured_at=calibration_captured_at,
                               nominal_goals=nominal_goals_override, command_goals=command_goals)
        if scene is not None:
            scene["task"] = task
        return scene
    except (ControlFault, ValueError, TypeError, KeyError, AttributeError):
        return None


def run(args):
    from rtde_receive import RTDEReceiveInterface
    from urcal.control.gui import gui_worker
    from urcal.control.vision import vision_worker

    if args.mode != "follow":
        raise ValueError("Only follower mode is supported")
    log_path = args.log or BUNDLE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                            + "-" + args.method + "-" + uuid.uuid4().hex[:8] + ".jsonl")
    logger = ExperimentLogger(log_path)
    error_recorder = PositionErrorRecorder(BUNDLE / "runs" / "error_segments")
    recording_notice = None
    session = None
    arm_error = None
    arm_error_category = ""
    alignment, alignment_complete = None, False
    alignment_goals = None
    alignment_nominal_goals = None
    alignment_bases = None
    alignment_capture = None
    handoff, endpoint_states = None, None
    calibration_guard = ActiveCalibrationGuard(args.max_invalid_duration)
    calibration_degraded = False
    recovery = None
    recovery_reason = ""
    recovery_capture_after = None
    recovery_ready = False
    gap_request_after, last_gap_request_at = float("inf"), float("-inf")
    gap_error = ""
    waits = WaitDiagnostics()
    ctx = mp.get_context("spawn")
    frames, commands, statuses, gui_statuses, recording_commands = (
        ctx.Queue(maxsize=size) for size in (1, 8, 1, 1, 8))
    inbox = CommandInbox(commands)
    stop = ctx.Event()
    vision = ctx.Process(target=vision_worker, args=(args, frames, commands, statuses, stop), daemon=True)
    gui = ctx.Process(target=gui_worker, args=(args, commands, gui_statuses, stop, recording_commands), daemon=True)
    engine = None
    states = None
    control_at = control_dt = None
    final_reason = "operator_stop"
    try:
        plan = BUNDLE / "docs" / "Research_Implementation_Plan.txt"
        logger.log({"kind": "metadata", "at": time.monotonic(), "schema": 1,
                    "source_plan": str(plan), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                    "settings": vars(args), "method": args.method,
                    "clock": "host monotonic seconds; robot timestamps are retained separately",
                    "error_basis": "estimated camera bases plus encoder-based controller FK; independent reference required for real error",
                    "collision_model": "nominal link/tool capsules, optional camera keep-out spheres; no CAD/self-collision certification"})
        engine = CooperativeController(args)
        with ExitStack() as cleanup:
            readers = {}
            for name in ROBOTS:
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                cleanup.callback(receiver.disconnect)
                readers[name] = StateReader(receiver, args.max_robot_age, require_joint_velocity=True)
            cleanup.callback(lambda: session.close() if session is not None else None)

            def observe_fault(_kind, exception, _traceback):
                if exception is not None:
                    capture_runtime_fault(exception, session, readers, time.monotonic())
                return False

            # ExitStack unwinds in reverse order: capture cached status before
            # stop/disconnect alters it. The outer logger runs after cleanup.
            cleanup.push(observe_fault)
            gui.start()
            vision.start()
            follow_law = getattr(args, "follow_law", "feedforward")
            print(f"[research] PREVIEW method={args.method}; law={follow_law}; "
                  f"leader={engine.leader}, follower={engine.follower}; log={log_path}", flush=True)
            if follow_law == "pose":
                print("[info] Follow law: POSE; transformed follower TCP target each cycle, "
                      "one pose-error velocity with measured follower-motion damping. "
                      "Leader/base/path velocity feedforward is not added; existing limits are unchanged.", flush=True)
            else:
                print("[info] Follow law: FEEDFORWARD; leader tracking and relative pose feedback "
                      "with bounded relative-path motion.", flush=True)
            follow_max_speed = getattr(args, "follow_max_speed", None)
            if follow_max_speed is None:
                follow_max_speed = args.max_speed
            print(f"[info] Follower tracking TCP linear limit: {follow_max_speed * 1000:g}mm/s; "
                  f"linear acceleration limit: {args.max_accel:g}m/s^2; "
                  f"hold/initial-alignment linear limit: {args.max_speed * 1000:g}mm/s.", flush=True)
            print(f"[info] Follower catch-up feedback caps: {args.feedback_speed * 1000:g}mm/s / "
                  f"{args.feedback_angular_speed:g}rad/s; position/orientation gains: "
                  f"{args.kp_linear:g}/{args.kp_angular:g} per second.", flush=True)
            print(f"[info] Motion bounds: joint acceleration {args.max_joint_accel:g}rad/s^2; "
                  f"travel radius {args.max_travel * 100:g}cm from the original start position.", flush=True)
            print("[info] Mean/max reprojection rejection: " +
                  (f"ON (mean <= {args.max_reproj_px:g}px, max <= {args.max_point_reproj_px:g}px)."
                   if args.check_reprojection else "OFF (preview and execution)."), flush=True)
            print(f"[info] Estimated flange task tolerance: {getattr(args, 'target_tolerance_m', 0.) * 100:g}cm / "
                  f"{getattr(args, 'target_tolerance_deg', 0.):g}deg; nominal targets retained.", flush=True)
            print("[info] UR10e PnP output correction: " + (
                "ON (local Z 180deg, +5mm)." if getattr(args, "ur10e_output_correction", True)
                else "OFF (raw PnP rotation/translation used as the control base)."), flush=True)
            print("[info] Common base plane: " + (
                "ON (shared physical mounting normal and height)." if getattr(args, "common_base_plane", False)
                else "OFF (independent base estimates)."), flush=True)
            if engine.base_motion_mode == "pose":
                print("[info] Base updates: POSE; live calibration updates coordinates without visual velocity "
                      "feedforward or base-speed rejection. Following uses measured joint/RTDE velocities.", flush=True)
                if getattr(engine, "pose_smoother", None) is not None:
                    smoother = engine.pose_smoother
                    print(f"[info] Base coordinates: median of {smoother.window} images, "
                          f"deadband {smoother.deadband_m * 1000:g}mm / "
                          f"{np.rad2deg(smoother.deadband_rad):g}deg, "
                          f"update cap {smoother.linear_rate * 1000:g}mm/s / "
                          f"{smoother.angular_rate:g}rad/s. Robot motion limits are unchanged.", flush=True)
            elif engine.base_motion_mode == "relative_kalman":
                config = engine.estimator.config
                print("[info] Base updates: RELATIVE KALMAN; same-exposure planar B^-1 A, "
                      "capture-time joint interpolation and confidence-weighted table feedforward.", flush=True)
                print(f"[info] Relative Kalman noise: {config.measurement_std_m * 1000:g}mm / "
                      f"{np.rad2deg(config.measurement_std_yaw):g}deg; prediction horizon "
                      f"{config.prediction_horizon:g}s; config="
                      f"{getattr(args, 'relative_kalman_config', None) or 'built-in provisional values'}.", flush=True)
                print(f"[info] Table compensation limits: {args.relative_base_speed * 1000:g}mm/s / "
                      f"{args.relative_base_angular_speed:g}rad/s; rate limits "
                      f"{args.relative_base_accel:g}m/s^2 / {args.relative_base_angular_accel:g}rad/s^2. "
                      "Complete robot motion limits still apply.", flush=True)
            else:
                print("[info] Base updates: VELOCITY; visual speed checks and mandatory base compensation enabled.",
                      flush=True)
            print(f"[info] Calibration limits: jump {args.max_calibration_jump_m * 100:g}cm / "
                  f"{args.max_calibration_jump_deg:g}deg; handoff target shift "
                  f"{args.handoff_max_shift_m * 100:g}cm / {args.handoff_max_shift_deg:g}deg; "
                  f"handoff wait {args.handoff_timeout:g}s; tracking error stop {args.max_target_m * 100:g}cm.", flush=True)
            if engine.live_calibration is not None:
                print(f"[info] Live calibration: skip transient rejects for up to {args.max_invalid_duration:g}s "
                      "while the last accepted capture is fresh; then brake/observe. "
                      f"Total loss budget {engine.live_calibration.timeout:g}s; "
                      "fresh consistent frames restart tracking with gradual coordinate blending.", flush=True)
            if args.initial_align:
                print(f"[info] Stationary calibration: {args.calibration_acquisition}; "
                      f"after {args.handoff_timeout:g}s handoff wait: {args.handoff_wait}.", flush=True)
                print(f"[info] Alignment leader rest limits: TCP {args.align_leader_max_speed * 1000:g}mm/s; "
                      f"angular {args.align_leader_max_angular_speed:g}rad/s; "
                      f"joint {args.align_leader_max_joint_speed:g}rad/s.", flush=True)
            print("[info] Recoverable faults: " + (
                "stop motion, keep preview, then require fresh calibration and a new Arm."
                if getattr(args, "fault_response", "exit") == "preview" else "stop and exit."), flush=True)
            if any(getattr(args, name + "_tcp_offset") is None for name in ROBOTS):
                print("[research] preview assumes zero unspecified TCP offsets; motion requires both actual offsets.", flush=True)
            period = 1. / args.control_hz
            last_tick, last_report = time.monotonic(), 0.
            scene_timestamps = {}

            def ingest_latest(when, *, allow_reacquire=False):
                nonlocal gap_request_after
                packet = take_latest(frames)
                if packet is not None:
                    if recovery is not None and (recovery_capture_after is None
                                                  or packet.captured_at <= recovery_capture_after):
                        logger.log({"kind": "recovery_measurement_discarded", "at": when,
                                    "sequence": packet.sequence, "captured_at": packet.captured_at,
                                    "reason": "waiting for images captured after confirmed standstill"})
                        return False
                    if session is not None and args.initial_align and not alignment_complete:
                        accepted = handoff.ingest(packet, when) if handoff is not None else False
                        logger.log(finite_record({
                            "kind": "handoff_measurement" if handoff is not None else "alignment_vision",
                            "at": when, "sequence": packet.sequence, "captured_at": packet.captured_at,
                            "poses_camera_model": packet.poses, "accepted": accepted,
                            "raw_mean_px": packet.raw_mean_px, "raw_max_px": packet.raw_max_px,
                            "valid_counts": packet.valid_counts, "pose_diagnostics": packet.pose_diagnostics,
                            "reason": handoff.reason if handoff is not None else "captured Arm calibration in use"}))
                        return False
                    previous_reacquisitions = engine.reacquisition_count
                    live = engine.live_calibration
                    previous_live = live.reacquisitions if live is not None else 0
                    accepted = engine.ingest(packet, when, allow_reacquire=allow_reacquire,
                                             active_recovery=session is not None and live is not None)
                    if session is not None and (
                            (not accepted and engine.invalid_since is not None)
                            or (live is not None and live.reacquisitions != previous_live)):
                        # Keep an already requested approach on temporary
                        # hold during grace, but reject earlier queued Start /
                        # Resume clicks even if the next good frame arrives
                        # before that click reaches this process.
                        gap_request_after = when
                    reacquired = engine.reacquisition_count != previous_reacquisitions
                    logger.log(finite_record({
                        "kind": "measurement", "at": when, "sequence": packet.sequence,
                        "captured_at": packet.captured_at, "received_at": when,
                        "poses_camera_model": packet.poses, "raw_mean_px": packet.raw_mean_px,
                        "raw_max_px": packet.raw_max_px, "valid_counts": packet.valid_counts,
                        "pose_diagnostics": packet.pose_diagnostics,
                        "accepted": accepted, "rejection": engine.last_rejection,
                        "live_calibration": live.status(when) if live is not None else None,
                        "preview_reacquiring_frames": engine.preview_recovery.count,
                        "preview_reacquired": reacquired}))
                    if live is not None and live.reacquisitions != previous_live:
                        print("[research] CALIBRATION REACQUIRED: blending fresh coordinates; continuing tracking", flush=True)
                        logger.log({"kind": "calibration_reacquired_live", "at": when,
                                    "sequence": packet.sequence, "captured_at": packet.captured_at,
                                    "stable_frames": engine.gate.stable_count, "motion_active": True,
                                    "live_calibration": live.status(when)})
                    if reacquired:
                        print("[research] PREVIEW: calibration reacquired from stable frames; review pose, then Arm", flush=True)
                        logger.log({"kind": "calibration_reacquired", "at": when,
                                    "sequence": packet.sequence, "captured_at": packet.captured_at,
                                    "stable_frames": engine.gate.stable_count,
                                    "motion_active": False})
                    return reacquired
                return False

            def read_alignment_inputs():
                checked = time.monotonic()
                if stop.is_set() or not vision.is_alive() or not gui.is_alive():
                    raise ResearchFault("operator_stop", "operator/vision stop during initial alignment")
                measured = {name: reader.read(checked) for name, reader in readers.items()}
                # The Arm snapshot defines this entire stationary-base phase.
                # Native preflight still checks live encoders against that pose.
                return {name: pose.copy() for name, pose in alignment_bases.items()}, measured

            def alignment_report(message):
                print("[research] " + message, flush=True)
                logger.log({"kind": "alignment_preflight", "at": time.monotonic(), "message": message})
                error_sample = position_error_sample(
                    None, now=time.monotonic(), leader=engine.leader, follower=engine.follower,
                    phase="alignment_preflight", reason="Preflight: no current TCP error sample")
                error_recorder.record(error_sample, emitted_at=time.monotonic())
                put_latest(gui_statuses, {"message": message, "active": True, "ready": False,
                                          "phase": "alignment_preflight", "progress": 0.,
                                          "position_error": error_sample,
                                          "error_recording": error_recorder.status()})

            def require_new_recovery_arm(requested):
                nonlocal recovery_ready
                if recovery is not None and not recovery_ready:
                    # Invalidate clicks issued during stop/acquisition, even
                    # if their IPC delivery happens after READY is displayed.
                    inbox.cancel_arm_requests()
                    recovery_ready = True
                    return False
                return requested

            def approach_enabled():
                return (session is not None and recovery is None
                        and (not args.initial_align or alignment_complete))

            while not stop.is_set():
                now = time.monotonic()
                # Recording has its own non-actuating command channel. It
                # never enters target/Arm handling or the latest-only GUI queue.
                for _ in range(8):
                    try:
                        error_recorder.request(recording_commands.get_nowait())
                    except Empty:
                        break
                dt, last_tick = now - last_tick, now
                control_at, control_dt = now, dt
                ready, goals, proposal = False, {}, None
                states = None
                category = ""
                try:
                    if not vision.is_alive() or not gui.is_alive():
                        raise ResearchFault("process_exit", "vision or GUI process exited")
                    if session is not None and dt > min(args.max_robot_age, 1. / args.watchdog_hz):
                        raise ResearchFault("control_deadline", "control loop missed its deadline")
                    arm, target_changed = False, False
                    gap_requests = []
                    for command in inbox.read():
                        if isinstance(command, dict) and command.get("type") == "gap_approach":
                            gap_requests.append(command)
                        elif isinstance(command, dict):
                            if session is not None:
                                raise ResearchFault("target_change", "stop before changing the relative task")
                            engine.apply_target(command)
                            gap_error = ""
                            target_changed = True
                            logger.log({"kind": "target", "at": now, "command": command})
                        elif command == "arm":
                            arm = True
                        elif command in ("stop", "quit"):
                            stop.set()
                        elif command in ("pause_task", "resume_task"):
                            if engine.gap_approach.state != "idle":
                                gap_requests.append({"type": "gap_approach", "requested_at": now,
                                                     "action": "pause" if command == "pause_task" else "resume"})
                            else:
                                engine.task_paused = command == "pause_task"
                            logger.log({"kind": "task_pause", "at": now, "paused": engine.task_paused})
                    if stop.is_set():
                        break
                    if target_changed:
                        arm = False
                    if recovery is not None:
                        states = {name: reader.read(now) for name, reader in readers.items()}
                        if not recovery.observe(states, now):
                            recovery_capture_after = None
                            recovery_ready = False
                            inbox.cancel_arm_requests()
                            ingest_latest(now)
                            raise CalibrationUnavailable("RECOVERY: " + recovery.reason)
                        if recovery_capture_after is None:
                            engine.reset_after_stop()
                            recovery_capture_after = now
                            recovery_ready = False
                            arm = False
                            inbox.cancel_arm_requests()
                            logger.log({"kind": "recovery_stationary", "at": now,
                                        "calibration_not_before": now, "progress": engine.progress})
                    if ingest_latest(now, allow_reacquire=session is None):
                        # Any Arm queued for the old coordinate baseline must
                        # not authorize motion using the newly acquired one.
                        arm = False
                    calibration_current = True
                    if session is not None and not (args.initial_align and not alignment_complete):
                        calibration_current = (engine.check_live_calibration(now) if engine.live_calibration is not None
                                               else calibration_guard.check(engine.gate, now))
                    states = {name: reader.read(now) for name, reader in readers.items()}
                    initializing = args.initial_align and not alignment_complete
                    live = engine.live_calibration
                    if not calibration_current or (live is not None and (live.waiting or live.blending)):
                        gap_request_after = now
                    if not calibration_current or (live is not None and live.waiting):
                        engine.pause_gap_approach("calibration recovery; resume approach when tracking is ready")
                    for request in gap_requests:
                        try:
                            requested_at = float(request.get("requested_at", float("nan")))
                            if (not np.isfinite(requested_at) or requested_at > now
                                    or now - requested_at > 1. or requested_at <= last_gap_request_at):
                                raise ValueError("expired, future or duplicate approach request")
                            last_gap_request_at = requested_at
                            action = request.get("action")
                            if action != "pause" and requested_at <= gap_request_after:
                                raise ValueError("approach request predates the current tracking phase")
                            engine.handle_gap_approach(action, enabled=approach_enabled() and calibration_current)
                            gap_error = ""
                            logger.log({"kind": "gap_approach", "at": now, "action": action,
                                        **engine.gap_approach_status(enabled=approach_enabled(), states=states)})
                            print(f"[research] APPROACH: {engine.gap_approach.state}; "
                                  "15 cm flange target; leader following continues", flush=True)
                        except (ValueError, TypeError) as exc:
                            gap_error = str(exc)
                            logger.log({"kind": "gap_approach_rejected", "at": now, "reason": gap_error})
                    if initializing:
                        if session is None:
                            engine.gate.check_fresh(now)
                            snapshot = engine.state(now, active=False)
                            alignment_bases = {name: pose.copy() for name, pose in snapshot.base_poses.items()}
                        # After Arm neither camera data nor prediction changes
                        # the copied bases or alignment target.
                        bases = alignment_bases
                        leader_flange = bases[engine.leader] @ flange_from_tcp(
                            states[engine.leader].tcp, engine.offsets[engine.leader])
                        goals = {engine.follower: leader_flange @ engine.path.pose(engine.progress)}
                        targets = {name: inverse(bases[name]) @ goals[name] @ engine.offsets[name]
                                   for name in engine.commanded}
                        if session is None:
                            origins = None if engine.initial_tcp is None else {engine.follower: engine.initial_tcp}
                            check_alignment_request(args, engine.offsets, bases, states, targets,
                                                    engine.models, engine.guard, origins)
                            ready = True
                            message = "READY: Arm to validate joint path and align; keep leader/bases stationary"
                            arm = require_new_recovery_arm(arm)
                            if arm:
                                if args.execute:
                                    alignment_capture = engine.gate.last_capture
                                    alignment_goals = {name: goal.copy() for name, goal in goals.items()}
                                    alignment_nominal_goals = {name: goal.copy() for name, goal in goals.items()}
                                    handoff, endpoint_states = None, None
                                    logger.log({"kind": "alignment_request", "at": now,
                                                "initial_state_at": now, "tcp_goals": targets,
                                                "initial_bases": bases,
                                                "initial_robots": {name: {"q": state.joints, "qd": state.joint_velocity,
                                                                          "tcp": state.tcp, "speed": state.speed}
                                                                   for name, state in states.items()}})
                                    alignment_report("PREFLIGHT: connecting commanded arm; no motion yet")
                                    arm_error = None
                                    engine.preview_recovery.reset()
                                    session = MotionSession(args, engine.commanded, engine.offsets,
                                                            stop_event=stop, defer_watchdog=True)
                                    try:
                                        alignment = prepare_alignment(session, args, engine.offsets, bases, states,
                                                                      targets, read_alignment_inputs, alignment_report,
                                                                      wait=stop.wait,
                                                                      diagnostic=lambda record: logger.log({
                                                                          "kind": "alignment_candidate",
                                                                          "at": time.monotonic(), **record}))
                                    except (AlignmentCalibrationTimeout, ArmIKUnavailable, ArmConditioningUnavailable):
                                        # These exceptions are confined to preflight
                                        # before watchdog setup or any velocity.
                                        session.close()
                                        session = None
                                        alignment = None
                                        alignment_goals = None
                                        now, states = time.monotonic(), None
                                        raise
                                    chosen = getattr(alignment, "targets", None)
                                    if isinstance(chosen, dict):
                                        alignment_goals = {name: bases[name] @ flange_from_tcp(pose, engine.offsets[name])
                                                           for name, pose in chosen.items()}
                                    if engine.initial_tcp is None:
                                        engine.initial_tcp = states[engine.follower].tcp.copy()
                                    ready = False
                                    recovery = None
                                    recovery_reason = ""
                                    last_tick = time.monotonic()
                                    gap_request_after = last_tick
                                    logger.log({"kind": "armed", "at": last_tick, "path": engine.path.kind,
                                                "phase": "initial_alignment", "goal_joints": alignment.goal_joints,
                                                "tcp_goals": chosen if isinstance(chosen, dict) else targets,
                                                "nominal_tcp_goals": targets,
                                                "target_adjustments": getattr(alignment, "target_adjustments", None)
                                                if isinstance(getattr(alignment, "target_adjustments", None), dict) else None,
                                                "initial_state_at": now,
                                                "initial_bases": bases,
                                                "initial_robots": {name: {"q": state.joints, "qd": state.joint_velocity,
                                                                          "tcp": state.tcp, "speed": state.speed}
                                                                   for name, state in states.items()}})
                                    message = "ARMED: captured calibration; initial alignment starts next tick"
                                else:
                                    message = "PREVIEW ONLY: --execute and actual TCP offsets are required"
                        else:
                            goals = alignment_goals
                            def alignment_fresh():
                                checked = time.monotonic()
                                if stop.is_set() or not vision.is_alive() or not gui.is_alive():
                                    raise ResearchFault("operator_stop", "operator/vision stop requested")
                                if checked - now > args.max_robot_age or any(
                                        checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                    raise ResearchFault("robot_state_expired", "alignment state/command expired")
                            if handoff is not None:
                                handoff.check_timeout(now)
                                qdots = alignment.propose_hold(states, bases, max(dt, 1e-6))
                            else:
                                qdots = alignment.propose(states, bases, max(dt, 1e-6))
                            logger.log({"kind": "alignment_proposal", "at": now,
                                        "prepared_at": time.monotonic(), "qdots": qdots,
                                        "joint_accelerations": alignment.joint_accelerations,
                                        "dt": max(dt, 1e-6), "bases": bases,
                                        "predicted_joints": alignment.predicted_joints,
                                        "progress": alignment.progress})
                            session.send_joints(qdots, alignment.predicted_joints, check_fresh=alignment_fresh,
                                                accelerations=alignment.joint_accelerations)
                            if handoff is None:
                                alignment.commit()
                            logger.log({"kind": "alignment", "at": now, "progress": alignment.progress,
                                        "dt": max(dt, 1e-6),
                                        "transport": transport_record(session),
                                        "command_sent": True, "done": alignment.done,
                                        "handoff_reason": None if handoff is None else handoff.reason,
                                        "robots": {name: {"q": state.joints, "qd": state.joint_velocity,
                                                          "tcp": state.tcp, "speed": state.speed}
                                                   for name, state in states.items()}})
                            message = "ALIGNING: %.1f%%; captured calibration; keep leader/bases stationary" % (100*alignment.progress)
                            if handoff is not None:
                                message = "ALIGNED; WAITING FOR LIVE CALIBRATION: " + handoff.status
                                if handoff.ready(now):
                                    live_snapshot = handoff.estimator.estimate(now, leader=engine.leader)
                                    if engine.common_base_plane:
                                        from urcal.cooperative.base_plane import constrain_base_snapshot
                                        live_snapshot = constrain_base_snapshot(live_snapshot, engine.leader, engine.mappings)
                                    live_bases = live_snapshot.base_poses
                                    live_leader = live_bases[engine.leader] @ flange_from_tcp(
                                        states[engine.leader].tcp, engine.offsets[engine.leader])
                                    live_targets = {engine.follower: inverse(live_bases[engine.follower])
                                                    @ live_leader @ engine.path.pose(engine.progress) @ engine.offsets[engine.follower]}
                                    nominal = getattr(alignment, "nominal_targets", None)
                                    def validate_live_calibration():
                                        check_handoff_snapshot(
                                            args, bases, endpoint_states, live_bases, states, offsets=engine.offsets,
                                            nominal_targets=nominal if isinstance(nominal, dict) else targets,
                                            live_targets=live_targets,
                                        )
                                        previous = (engine.gate, engine.estimator, engine.invalid_since,
                                                    engine.last_rejection, engine.ever_ready)
                                        try:
                                            engine.gate, engine.estimator = handoff.tracker, handoff.estimator
                                            engine.invalid_since, engine.last_rejection, engine.ever_ready = None, "", True
                                            engine.prepare(now, states, min(max(dt, 1e-6), 2*period), active=False)
                                        finally:
                                            (engine.gate, engine.estimator, engine.invalid_since,
                                             engine.last_rejection, engine.ever_ready) = previous

                                    if handoff.try_validate(now, validate_live_calibration):
                                        engine.gate, engine.estimator = handoff.tracker, handoff.estimator
                                        engine.invalid_since, engine.last_rejection, engine.ever_ready = None, "", True
                                        engine.preview_recovery.reset()
                                        calibration_guard.arm(engine.gate, now)
                                        alignment_complete = True
                                        gap_request_after = now
                                        message = "LIVE CALIBRATION READY: following starts next fresh tick"
                                        logger.log({"kind": "alignment_complete", "at": now,
                                                    "progress": engine.progress,
                                                    "live_capture": engine.gate.last_capture,
                                                    "handoff_started_at": handoff.started_at})
                                    else:
                                        message = "ALIGNED; WAITING FOR LIVE CALIBRATION: " + handoff.status
                            elif alignment.done:
                                for attribute in ("previous_twist", "previous_qdot", "previous_compensation",
                                                  "previous_compensation_qdot", "previous_feedback",
                                                  "previous_base_compensation", "previous_base_compensation_qdot"):
                                    getattr(engine, attribute)[:] = 0.
                                engine.settle_elapsed, engine.finished = 0., False
                                engine.tracking_offset = (np.eye(4) if engine.gap_approach.state != "idle" else None)
                                # A fresh estimator prevents the entire frozen
                                # interval from becoming an artificial velocity.
                                completed_at = time.monotonic()
                                handoff = CalibrationHandoff(
                                    engine.gate, completed_at, timeout=args.handoff_timeout,
                                    estimator_factory=engine._new_estimator,
                                    mappings=engine.mappings, leader=engine.leader,
                                    consensus=args.calibration_acquisition == "consensus",
                                    wait_at_rest=args.handoff_wait == "hold")
                                endpoint_states = deepcopy(states)
                                message = "ALIGNED and stationary; acquiring NEW live calibration; keep leader/bases stationary"
                                logger.log({"kind": "alignment_endpoint", "at": completed_at,
                                            "progress": engine.progress, "frozen_capture": alignment_capture})
                        waits.clear()
                    else:
                        planning_dt = min(max(dt, 1e-6), 2*period)
                        proposal = (engine.prepare_live(now, states, planning_dt)
                                    if session is not None and engine.live_calibration is not None else
                                    engine.prepare(now, states, planning_dt, active=session is not None))
                        proposal.record["command_sent"] = False
                        goals, ready = proposal.goals, session is None
                        if engine.path.kind == "hold":
                            following = ("decelerating tracking; checking next frame" if proposal.record.get("tracking_braking")
                                         else "following with bounded acceleration" if proposal.record.get("tracking_limited")
                                         else "relative pose following")
                            message = f"{'ACTIVE' if session else 'READY (preview)'} {args.method}; law={follow_law}; {following}"
                            if "relative_error_m" in proposal.record and "relative_error_rad" in proposal.record:
                                message += (f"; error={proposal.record['relative_error_m'] * 1000:.1f}mm / "
                                            f"{np.rad2deg(proposal.record['relative_error_rad']):.2f}deg")
                        else:
                            message = (f"{'ACTIVE' if session else 'READY (preview)'} {args.method}; law={follow_law}; "
                                       f"progress={engine.progress:.1%}, rate={proposal.decision.task_scale:.2f}")
                        if session is None:
                            arm = require_new_recovery_arm(arm)
                        if arm and session is None:
                            if args.execute:
                                arm_error = None
                                engine.preview_recovery.reset()
                                calibration_guard.arm(engine.gate, now)
                                session = MotionSession(args, engine.commanded, engine.offsets, stop_event=stop)
                                recovery = None
                                recovery_reason = ""
                                ready = False
                                last_tick = time.monotonic()
                                gap_request_after = last_tick
                                logger.log({"kind": "armed", "at": last_tick, "path": engine.path.kind})
                                message = "ARMED; waiting for fresh state before movement"
                            else:
                                message = "PREVIEW ONLY: --execute and actual TCP offsets are required"
                        elif session is not None:
                            def check_fresh():
                                checked = time.monotonic()
                                if stop.is_set() or not vision.is_alive() or not gui.is_alive():
                                    raise ResearchFault("operator_stop", "operator/vision stop requested")
                                needs_braking = False
                                if engine.live_calibration is not None:
                                    current = engine.check_live_calibration(checked)
                                    needs_braking = not current and not proposal.record.get("calibration_braking")
                                else:
                                    calibration_guard.check(engine.gate, checked)
                                    engine.state(checked, active=True)
                                if checked - now > args.max_robot_age or any(
                                        checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                    raise ResearchFault("robot_state_expired", "state/command expired during controller checks")
                                if needs_braking:
                                    raise CalibrationBrakingRequired(
                                        "calibration expired during command checks; discard tracking and brake next tick")
                            logger.log({"kind": "command_proposal", "at": now,
                                        "prepared_at": time.monotonic(), "tcp_goals": proposal.tcp_goals,
                                        "twists": proposal.twists, "next_progress": proposal.next_progress})
                            send_options = {}
                            if "speedl_acceleration" in proposal.record:
                                send_options["accelerations"] = {engine.follower: proposal.record["speedl_acceleration"]}
                            try:
                                session.send(proposal.tcp_goals, proposal.twists, check_fresh=check_fresh,
                                             safety_targets=proposal.future_tcp_goals, **send_options)
                            except CalibrationBrakingRequired as exc:
                                # Native checks ran, but check_fresh precedes
                                # speedL. No stale tracking command is sent or
                                # committed; the next bounded tick brakes.
                                logger.log({"kind": "calibration_command_discarded",
                                            "at": time.monotonic(), "reason": str(exc)})
                            else:
                                engine.commit(proposal)
                                proposal.record["command_sent"] = True
                                proposal.record["sent_at"] = time.monotonic()
                                proposal.record["transport"] = transport_record(session)
                                proposal.record["committed_progress"] = engine.progress
                                proposal.record["committed_finished"] = engine.finished
                        else:
                            proposal.record["command_sent"] = False
                        logger.log(proposal.record)
                        waits.clear()
                    if session is not None and not (args.initial_align and not alignment_complete):
                        live = engine.live_calibration
                        degraded = live.waiting if live is not None else not calibration_current
                        if degraded != calibration_degraded:
                            event = (("CALIBRATION OBSERVE: braking; collecting fresh consistent frames"
                                      if degraded else "CALIBRATION BLEND: applying fresh coordinates gradually")
                                     if live is not None else
                                     ("CALIBRATION HOLD: rejected pair; using last accepted calibration"
                                      if degraded else "CALIBRATION RECOVERED: accepted new frame"))
                            print("[research] " + event, flush=True)
                            logger.log({"kind": ("calibration_observation" if degraded else "calibration_blending")
                                        if live is not None else ("calibration_hold" if degraded else "calibration_recovered"),
                                        "at": now, "reason": engine.gate.last_rejection,
                                        "last_accepted_capture": engine.gate.last_capture,
                                        "live_calibration": live.status(now) if live is not None else None,
                                        "max_invalid_duration": args.max_invalid_duration})
                            calibration_degraded = degraded
                        if degraded:
                            if live is not None:
                                info = live.status(now)
                                message = (f"CALIBRATION OBSERVE: braking; {info['elapsed']:.2f}/{info['timeout']:g}s; "
                                           + info["candidate_reason"])
                            else:
                                message += "; CALIBRATION HOLD: waiting for next valid frame"
                        elif live is not None and live.blending:
                            message += "; CALIBRATION BLEND: applying fresh coordinates gradually"
                    if session is None and arm_error is not None:
                        message = "ARM BLOCKED: " + arm_error
                        category = arm_error_category
                except (ControlFault, ValueError, KeyError, TypeError) as exc:
                    category = getattr(exc, "category", "control_fault")
                    engine.pause_gap_approach("motion unavailable: " + str(exc))
                    if session is not None:
                        if (getattr(args, "fault_response", "exit") != "preview"
                                or not is_recoverable_fault(exc) or stop.is_set()
                                or not vision.is_alive() or not gui.is_alive()):
                            raise
                        # Stop first. Failed stop/script acknowledgements are
                        # fatal and cannot unlock preview or another Arm.
                        session.stop_for_recovery()
                        session = None
                        if engine.live_calibration is not None:
                            engine.live_calibration.reset()
                        stopped_at = time.monotonic()
                        gap_request_after = stopped_at
                        if stop.is_set():
                            break
                        stop_states = {name: reader.read(stopped_at) for name, reader in readers.items()}
                        recovery = RecoveryStopGate(engine.commanded, stop_states, stopped_at,
                                                    max_state_age=args.max_robot_age)
                        recovery_reason = str(exc)
                        recovery_capture_after, recovery_ready = None, False
                        calibration_guard = ActiveCalibrationGuard(args.max_invalid_duration)
                        calibration_degraded = False
                        alignment, alignment_complete = None, False
                        alignment_goals = alignment_nominal_goals = alignment_bases = alignment_capture = None
                        handoff = endpoint_states = None
                        arm_error, arm_error_category = None, ""
                        inbox.cancel_arm_requests()
                        ready, goals, proposal, states = False, {}, None, None
                        last_tick = stopped_at
                        logger.log({"kind": "motion_stopped", "at": stopped_at, "category": category,
                                    "reason": recovery_reason, "control_at": control_at, "control_dt": control_dt,
                                    "progress": engine.progress, "new_arm_required": True,
                                    "external_base_stop_required": True, **fault_diagnostics(exc, stop_states)})
                        print("[research] MOTION STOPPED; preview kept open: " + recovery_reason, flush=True)
                        print("[research] RECOVERY: confirming standstill, then collecting fresh calibration; "
                              "review pose and press Arm again. Stop external base motion separately.", flush=True)
                    elif (recovery is not None and isinstance(exc, ControlFault)
                          and not is_recoverable_fault(exc)
                          and not isinstance(exc, (ArmConnectionError, AlignmentCalibrationTimeout,
                                                   ArmIKUnavailable, ArmConditioningUnavailable))):
                        raise
                    if recovery is not None:
                        recovery_ready = False
                        inbox.cancel_arm_requests()
                    if isinstance(exc, (ArmConnectionError, AlignmentCalibrationTimeout,
                                        ArmIKUnavailable, ArmConditioningUnavailable)):
                        inbox.cancel_arm_requests()
                        arm_error = str(exc)
                        arm_error_category = exc.category
                        print("[research] ARM BLOCKED: " + arm_error, flush=True)
                    ready = False
                    message = "WAIT: " + str(exc)
                    if session is None and engine.preview_recovery.count:
                        message = (f"WAIT: PREVIEW REACQUIRING ({engine.preview_recovery.count}/{args.stable_frames} frames); "
                                   + str(exc))
                    record = waits.record(now, exc, states)
                    if record is not None:
                        logger.log(record)
                if recovery is not None:
                    if ready:
                        message = "RECOVERY READY: stopped; review current/target pose and press Arm again"
                    else:
                        message = "RECOVERY WAIT: " + message.removeprefix("WAIT: ").removeprefix("RECOVERY: ")
                status = {"message": message, "ready": ready, "active": session is not None, "arm_error": arm_error,
                          "recovery_required": recovery is not None, "recovery_reason": recovery_reason,
                          "recovery_stopped": recovery is not None and recovery_capture_after is not None,
                          "goals": goals, "progress": engine.progress, "task_paused": engine.task_paused,
                          "task_scale": proposal.decision.task_scale if proposal is not None else 0.,
                          "measurement_age": proposal.record["measurement_age"] if proposal is not None else None,
                          "fault_category": category,
                          "calibration_degraded": calibration_degraded,
                          "live_calibration": engine.live_calibration.status(now) if engine.live_calibration is not None else None,
                          "preview_reacquiring_frames": engine.preview_recovery.count,
                          "phase": "recovery" if recovery is not None else (
                              ("alignment_calibration_handoff" if handoff is not None else "alignment")
                              if args.initial_align and not alignment_complete else (
                                  "calibration_observation" if engine.live_calibration is not None and engine.live_calibration.waiting
                                  else "calibration_blending" if engine.live_calibration is not None and engine.live_calibration.blending
                                  else "tracking")),
                          "alignment_progress": alignment.progress if alignment is not None else None}
                status["gap_approach"] = engine.gap_approach_status(
                    enabled=approach_enabled() and not category, states=states)
                status["gap_approach"]["error"] = gap_error
                approach_status = status["gap_approach"]
                if approach_status["state"] != "idle":
                    approach_phase = ("waiting for coordinates (auto-continue)" if approach_status["temporary_hold"]
                                      else "paused (press Resume)" if approach_status["state"] == "paused"
                                      else approach_status["state"])
                    message += (f"; APPROACH {approach_phase}, "
                                f"gap={approach_status['commanded_gap_m'] * 100:.2f}->15.00cm")
                    status["message"] = message
                put_latest(statuses, status)
                scene = cooperative_gui_scene(now, engine, states, readers, scene_timestamps,
                                              args.max_robot_age, proposal=proposal,
                                              active=session is not None,
                                              goals_override=alignment_goals if session is not None
                                              and not alignment_complete else None,
                                              bases_override=alignment_bases if session is not None and args.initial_align
                                              and not alignment_complete else None,
                                              calibration_captured_at=alignment_capture if session is not None
                                              and args.initial_align and not alignment_complete else None,
                                              nominal_goals_override=alignment_nominal_goals if session is not None
                                              and args.initial_align and not alignment_complete else None)
                error_target, error_relative = None, None
                error_source = "preview_nominal_relative"
                error_phase = status["phase"] if session is not None or recovery is not None else "preview"
                if session is not None and alignment is not None and not alignment_complete:
                    chosen = getattr(alignment, "targets", None)
                    if isinstance(chosen, dict):
                        error_target = chosen.get(engine.follower)
                    error_source = "initial_alignment_endpoint"
                elif proposal is not None and "tracking_relative_target" in proposal.record:
                    error_relative = proposal.record["tracking_relative_target"]
                    error_source = ("tracking_accepted_relative" if proposal.record.get("command_sent")
                                    else "tracking_candidate_relative" if session is not None
                                    else "preview_candidate_relative")
                    if proposal.record.get("calibration_braking"):
                        error_source += "_during_calibration_braking"
                else:
                    error_relative = engine.path.pose(engine.progress)
                    offset = getattr(engine, "tracking_offset", None)
                    if isinstance(offset, np.ndarray):
                        error_relative = error_relative @ offset
                        error_source = "tracking_accepted_relative"
                error_sample = position_error_sample(
                    scene, now=now, leader=engine.leader, follower=engine.follower,
                    phase=error_phase, target_tcp=error_target, relative_target=error_relative,
                    target_source=error_source, max_robot_age=args.max_robot_age,
                    max_calibration_age=args.max_calibration_age)
                error_recorder.record(error_sample, emitted_at=now)
                recording_status = error_recorder.status()
                notice = (recording_status.get("state"), recording_status.get("path"), recording_status.get("error"))
                if notice != recording_notice and notice[0] != "idle":
                    print(f"[error CSV] {notice[0]}: {notice[2] or notice[1] or ''}", flush=True)
                recording_notice = notice
                put_latest(gui_statuses, {**status, "scene": scene, "position_error": error_sample,
                                          "error_recording": recording_status})
                if now - last_report >= 1.:
                    print("[research] " + message, flush=True)
                    last_report = now
                stop.wait(max(0., period - (time.monotonic() - last_tick)))
    except BaseException as exc:
        final_reason = getattr(exc, "category", "operator_stop" if isinstance(exc, KeyboardInterrupt) else "control_fault")
        try:
            logger.log({"kind": "fault", "at": time.monotonic(), "category": final_reason,
                        "control_at": control_at, "control_dt": control_dt,
                        "reason": str(exc), "external_base_stop_required": session is not None,
                        **fault_diagnostics(exc, states)})
        except Exception as logging_exc:
            print(f"[log STOP] {logging_exc}", flush=True)
        raise
    finally:
        if engine is not None:
            engine.pause_gap_approach("controller stopped")
        if session is not None:
            session.close()
        stop.set()
        error_recorder.close()
        recording_status = error_recorder.status()
        if recording_status.get("state") not in (None, "idle"):
            print(f"[error CSV] {recording_status.get('state')}: "
                  f"{recording_status.get('error') or recording_status.get('path') or ''}", flush=True)
        for child in (vision, gui):
            if child.pid is not None:
                child.join(timeout=3.)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=1.)
        for queue in (frames, commands, statuses, gui_statuses, recording_commands):
            queue.cancel_join_thread()
            queue.close()
        try:
            logger.log({"kind": "end", "at": time.monotonic(), "reason": final_reason,
                        "progress": engine.progress if engine is not None else None,
                        "external_base_stop_required": session is not None})
        finally:
            logger.close()
        if session is not None:
            print("[STOP] Robot control ended. Stop external table/base motion separately.", flush=True)


def main(argv=None):
    args = parse_args(["--mode", "follow", *(sys.argv[1:] if argv is None else argv)])
    try:
        if args.simulate:
            from urcal.simulation.runtime import run as run_simulation
            run_simulation(args)
        else:
            run(args)
    except KeyboardInterrupt:
        print("[research] stopped by operator", flush=True)
    except Exception as exc:
        diagnostic = fault_transport_summary(exc)
        if diagnostic:
            print("[RTDE at fault] " + diagnostic, flush=True)
        raise SystemExit(f"[STOP] {exc}. Stop external table motion; no automatic restart.") from exc


if __name__ == "__main__":
    main()
