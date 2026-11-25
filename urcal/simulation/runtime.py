"""Capture live initial conditions once, then run an isolated joint simulation.

The virtual clock advances by one fixed control period per accepted step. GUI
delays slow playback; they never cause a large integration step or catch-up.
Saved snapshots require neither camera, inference checkpoints nor robot SDKs.
"""

import copy
from contextlib import ExitStack, closing
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

from urcal.control.geometry import (calibration_output_mapping, flange_from_tcp, inverse,
                                    matrix_pose, model_from_controller_base, pose_delta, pose_matrix)
from urcal.control.policy import CalibrationFrame, ControlFault
from urcal.control.runtime import parse_args
from urcal.control.tracking_error import position_error_sample
from urcal.control.error_recording import PositionErrorRecorder
from urcal.control.evaluation import EvaluationRecorder
from urcal.cooperative.controller import CooperativeController
from urcal.cooperative.logging import ExperimentLogger
from urcal.paths import BUNDLE
from urcal.simulation.plant import JointPlant, SimulationFault
from urcal.simulation.snapshot import capture_initial_state, load_initial_state, save_initial_state, validate_initial_state


ROBOTS = ("ur5e", "ur10e")
NOTICE = "SIMULATION — captured initial state; hardware disconnected; no hardware commands."


class SimulationSession:
    """Controller/plant coupling without any hardware transport or wall clock IO."""

    def __init__(self, args, initial, logger=None, evaluation_recorder=None):
        if args.execute or not args.simulate:
            raise ValueError("simulation requires --simulate without --execute")
        if args.mode != "follow":
            raise ValueError("only follower mode is supported")
        self.args = copy.deepcopy(args)
        self.initial = validate_initial_state(initial)
        # A saved scene retains its capture convention regardless of CLI defaults.
        self.args.ur10e_output_correction = self.initial["calibration"]["ur10e_output_correction"]
        self.args.common_base_plane = self.initial["calibration"]["common_base_plane"]
        for name in ROBOTS:
            offset = self.initial["robots"][name]["tcp_offset"]
            specified = getattr(args, name + "_tcp_offset")
            if specified is not None and not np.allclose(pose_matrix(specified), offset, atol=1e-9, rtol=0):
                raise ValueError(name + ": TCP offset differs from the saved initial state; recapture it")
            setattr(self.args, name + "_tcp_offset", matrix_pose(offset).tolist())
        self.logger = logger
        self.evaluation_recorder = evaluation_recorder
        self.plant = JointPlant(self.initial, self.args)
        self.dt = 1. / self.args.control_hz
        self._mappings = {name: calibration_output_mapping(
            name, ur10e_output_correction=self.args.ur10e_output_correction) for name in ROBOTS}
        self._target_command = None
        self.episode = -1
        self.reset()

    @property
    def now(self):
        return 1000. + self.steps * self.dt

    @property
    def elapsed(self):
        return self.steps * self.dt

    def _log(self, kind, **fields):
        if self.logger is not None:
            self.logger.log({**fields, "kind": kind, "at": self.now, "simulation": True,
                             "hardware_commands": False, "simulation_time": self.elapsed,
                             "episode": self.episode})

    def _calibration(self):
        return {name: self.plant.bases[name] @ inverse(self._mappings[name]) for name in ROBOTS}

    def _ingest(self, at):
        self.sequence += 1
        # This is an ideal virtual base sensor, NOT another camera measurement.
        # Original camera residuals/time remain in the immutable snapshot.
        packet = CalibrationFrame(self.sequence, at, self._calibration(),
                                  {name: 0. for name in ROBOTS}, {name: 0. for name in ROBOTS},
                                  {name: 7 for name in ROBOTS})
        self._evaluation_measurement = {
            "source": "ideal_virtual_base_sensor", "sequence": self.sequence,
            "captured_at": at, "received_at": at, "poses": copy.deepcopy(packet.poses),
            "raw_mean_px": packet.raw_mean_px.copy(), "raw_max_px": packet.raw_max_px.copy(),
            "valid_counts": packet.valid_counts.copy(), "accepted": False,
        }
        accepted = self.engine.ingest(packet, at)
        self._evaluation_measurement["accepted"] = accepted
        if not accepted:
            self._evaluation_measurement["reason"] = self.engine.last_rejection
            raise SimulationFault("virtual base sensor rejected: " + self.engine.last_rejection)

    def reset(self):
        if self.episode >= 0 and self.evaluation_recorder is not None:
            # A clock rewind starts a new experiment episode. Keep its final
            # observation and end the old fixed-target session before reset.
            self._record_evaluation(self.now, "reset", reason="simulation reset; evaluation ended")
            self.evaluation_request({"type": "evaluation", "action": "stop",
                                     "request_id": uuid.uuid4().hex, "reason": "simulation_reset"})
        self.plant.reset()
        self.episode += 1
        self.active, self.paused, self.fault = False, False, ""
        self.gap_approach_error = ""
        self.steps, self.sequence = 0, 0
        self.approach = None
        self.approach_nominal_targets, self.approach_target_adjustments = None, None
        self._alignment_planned = False
        self._control_origin = 0.
        self.engine = CooperativeController(self.args, models=self.plant.models, guard=self.plant.guard)
        if self._target_command is not None:
            self.engine.apply_target(copy.deepcopy(self._target_command))
        for i in range(self.args.stable_frames):
            self._ingest(self.now - (self.args.stable_frames - i - 1) * self.dt)
        self.last_record, self.task_scale = {}, 0.
        self._preview()
        self._log("simulation_reset")

    def evaluation_request(self, command):
        """Record experiment actions without routing them through motion IO."""
        if self.evaluation_recorder is None:
            return False
        if isinstance(command, dict) and command.get("action") == "start":
            command = copy.deepcopy(command)
            path = self.engine.path
            command["runtime_context"] = {
                "gap": self.engine.gap, "progress": self.engine.progress,
                "task_paused": self.engine.task_paused,
                "path": {key: copy.deepcopy(getattr(path, key))
                         for key in ("kind", "start", "delta", "radius", "duration")},
                "tcp_offsets": copy.deepcopy(self.engine.offsets),
            }
        return self.evaluation_recorder.request(
            command, now=self.now, target=self.engine.path.pose(self.engine.progress).copy(),
            episode=self.episode)

    def _evaluation_scene(self, at, *, states=None, control_record=None):
        """Copy this cycle's measured state and already evaluated coordinates.

        Reading telemetry must not call prepare/state or advance an estimator.
        The virtual plant is retained separately from the controller estimate.
        """
        states = self.plant.states(at) if states is None else states
        record = {} if control_record is None else control_record
        bases = record.get("bases")
        captures = record.get("measurement_times")
        if bases is None:
            # Alignment/paused cycles have no current control proposal. Use
            # the actual accepted virtual observation, not a cached display
            # snapshot whose freshness would depend on GUI polling.
            accepted = self.engine.gate.poses
            bases = {name: pose @ self._mappings[name] for name, pose in accepted.items()}
            captured = self.engine.gate.last_capture
            captures = None if captured is None else {name: captured for name in ROBOTS}
        return {
            "at": float(at), "clock": "simulation", "episode": self.episode,
            "bases": {} if bases is None else copy.deepcopy(bases),
            "truth_bases": copy.deepcopy(self.plant.bases),
            "robots": {name: {"at": float(at), "q": state.joints.copy(), "qd": state.joint_velocity.copy(),
                               "tcp": state.tcp.copy(), "tcp_speed": state.speed.copy(),
                               "tcp_offset": self.plant.offsets[name].copy(),
                               "robot_timestamp": state.robot_timestamp}
                       for name, state in states.items()},
            "calibration_valid": bases is not None and captures is not None,
            "measurement_age": ({name: float(at - stamp) for name, stamp in captures.items()}
                                if captures is not None else None),
            "measurement_times": copy.deepcopy(captures),
            "pose_source": "controller_proposal" if control_record is not None else "accepted_virtual_calibration",
            "reason": "ideal virtual base sensor; no hardware measurements",
        }

    def _record_evaluation(self, at, phase, *, scene=None, control_record=None,
                           transport=None, measurement=None, reason=""):
        if self.evaluation_recorder is None:
            return
        control = copy.deepcopy(control_record)
        if control is not None:
            control.update(command_sent=False, hardware_commands=False,
                           simulated_command_applied=bool(transport and transport.get("simulated_command_applied")))
        self.evaluation_recorder.record(
            at=at, phase=phase, scene=self._evaluation_scene(at) if scene is None else scene,
            control_record=control, transport=transport,
            measurement=copy.deepcopy(self._evaluation_measurement) if measurement is None else measurement,
            reason=reason, episode=self.episode)

    def _prepare(self):
        states = self.plant.states(self.now)
        return self.engine.prepare(self.now, states, self.dt, active=self.active)

    def _plan_alignment(self):
        """Plan a virtual start-up move when the final pose is outside tracking limits.

        Tracking assumes an already aligned pair. Initial alignment instead
        follows a checked joint path at the configured speeds, accelerations
        and travel bounds. Failed exact IK may choose a separate precise goal
        inside the task allowance; the nominal target remains unchanged.
        Physical controllers never enter this path; the capture is not teleported.
        """
        states = self.plant.states(self.now)
        estimate = self.engine.state(self.now, active=False)
        leader, follower = self.engine.leader, self.engine.follower
        leader_flange = estimate.base_poses[leader] @ flange_from_tcp(states[leader].tcp, self.plant.offsets[leader])
        goal = leader_flange @ self.engine.path.pose(0.)
        targets = {follower: inverse(estimate.base_poses[follower]) @ goal @ self.plant.offsets[follower]}
        # Preserve the capture as the travel origin after alignment too.
        self.engine.initial_tcp = states[follower].tcp.copy()
        approximate = bool(getattr(self.args, "target_tolerance_m", 0.)
                           or getattr(self.args, "target_tolerance_deg", 0.))
        errors = [pose_delta(flange_from_tcp(states[name].tcp, self.plant.offsets[name]),
                             flange_from_tcp(target, self.plant.offsets[name])) if approximate
                  else pose_delta(states[name].tcp, target) for name, target in targets.items()]
        outside_tracking = any(np.linalg.norm(dp) > self.args.max_target_m
                               or np.linalg.norm(dr) > np.deg2rad(self.args.max_target_deg) for dp, dr in errors)
        if outside_tracking:
            from urcal.simulation.approach import InitialApproach, InitialApproachIKUnavailable
            self.approach_nominal_targets = {name: value.copy() for name, value in targets.items()}
            try:
                self.approach = InitialApproach(self.plant.models, self.plant.guard, self.args,
                                                self.plant.bases, states, targets)
            except InitialApproachIKUnavailable:
                if not (getattr(self.args, "target_tolerance_m", 0.)
                        or getattr(self.args, "target_tolerance_deg", 0.)):
                    raise
                from urcal.control.approximation import select_alignment_candidate
                selected, _, adjustment = select_alignment_candidate(
                    self.args, self.plant.offsets, self.plant.bases, states, targets)
                # The actual simulation model independently resolves IK and
                # checks the full joint path. Selector seeds are not executed.
                self.approach = InitialApproach(self.plant.models, self.plant.guard, self.args,
                                                self.plant.bases, states, selected)
                self.approach_target_adjustments = adjustment
            self._log("simulation_alignment_plan", nominal_tcp_goals=self.approach_nominal_targets,
                      command_tcp_goals=self.approach.targets,
                      target_adjustments=self.approach_target_adjustments)
        self._alignment_planned = True

    def _preview(self):
        self.ready = False
        if self.fault:
            self.message = "STOP: " + self.fault + "; Reset simulation to retry"
            return
        try:
            if self.steps == 0 and not self._alignment_planned:
                self._plan_alignment()
            if self.approach is not None and not self.approach.done:
                self.ready = True
                self.message = f"Ready — Start simulation; initial alignment ~{self.approach.duration:.1f} s, then follow control"
                return
            self._prepare()
            self.ready = True
            self.message = "Ready — Start simulation"
        except (ControlFault, ValueError, np.linalg.LinAlgError) as exc:
            self.message = "WAIT: " + str(exc)

    def _gap_approach_enabled(self):
        return bool(self.active and not self.paused and not self.fault
                    and not (self.approach is not None and not self.approach.done))

    def command(self, command):
        if isinstance(command, dict) and command.get("type") == "gap_approach":
            try:
                # GUI callbacks are synchronous in this process. Their host
                # timestamp is not comparable with the frozen virtual clock.
                if "requested_at" in command and not np.isfinite(float(command["requested_at"])):
                    raise ValueError("gap approach request timestamp must be finite")
                self.engine.handle_gap_approach(command.get("action"), enabled=self._gap_approach_enabled())
            except (TypeError, ValueError, OverflowError) as exc:
                self.gap_approach_error = str(exc)
                self._log("simulation_gap_approach", action=command.get("action"), accepted=False,
                          reason=self.gap_approach_error)
                raise ValueError(self.gap_approach_error) from exc
            self.gap_approach_error = ""
            self._log("simulation_gap_approach", action=command.get("action"), accepted=True,
                      gap_approach=self.engine.gap_approach_status(enabled=self._gap_approach_enabled()))
        elif isinstance(command, dict):
            if self.active:
                raise ValueError("Reset simulation before editing the target")
            if "center_pose" in command:
                raise ValueError("fixed camera-center targets are no longer supported")
            self.engine.apply_target(command)
            self.gap_approach_error = ""
            self._target_command = copy.deepcopy(command)
            self.approach = None
            self.approach_nominal_targets, self.approach_target_adjustments = None, None
            self._alignment_planned = False
            self._preview()
            self._log("simulation_target", target=command)
        elif command == "arm":
            if self.active:
                return
            self._preview()
            if not self.ready:
                raise ValueError(self.message)
            self.active = True
            self.message = ("Initial alignment — leader/base motion held until alignment completes"
                            if self.approach is not None and not self.approach.done else "Running virtual joint control")
            self._log("simulation_start")
        elif command == "sim_pause":
            if self.active:
                # Preserve velocities/controller history while freezing time;
                # resume continues the same virtual trajectory without a jump.
                self.paused = not self.paused
                self.message = ("Paused — virtual time frozen" if self.paused else
                                "Initial alignment" if self.approach is not None and not self.approach.done
                                else "Running virtual joint control")
                self._log("simulation_pause", paused=self.paused)
        elif command == "sim_reset":
            self.reset()
        elif command in ("pause_task", "resume_task"):
            if not self.active:
                raise ValueError("task pause requires an active follow simulation")
            if self.engine.gap_approach.state != "idle":
                action = "pause" if command == "pause_task" else "resume"
                self.engine.handle_gap_approach(action, enabled=self._gap_approach_enabled())
            else:
                self.engine.task_paused = command == "pause_task"
            self._log("simulation_task_pause", paused=self.engine.task_paused)
        else:
            raise ValueError("unknown simulation command")

    def _external_motion(self):
        """Small deterministic motions starting at zero displacement/velocity."""
        next_time, omega = self.elapsed - self._control_origin + self.dt, 2 * np.pi / 20.
        phase = .5 * (1 - np.cos(omega * next_time))
        bases = {name: value.copy() for name, value in self.initial["bases"].items()}
        if self.args.sim_motion in ("bases", "both"):
            direction = np.array([0., 1., 0.])
            if self.args.common_base_plane:
                # Move the virtual truth along the captured mounting plane,
                # which is generally tilted relative to camera XYZ. Preserve
                # the original camera-Y motion for unconstrained old captures.
                physical = (bases["ur5e"] @ inverse(self._mappings["ur5e"])
                            @ model_from_controller_base("ur5e"))
                direction = physical[:3, 0]
            displacement = .01 * phase * direction
            bases["ur5e"][:3, 3] += displacement
            bases["ur10e"][:3, 3] -= displacement
        leader_velocity = np.zeros(6)
        if self.args.sim_motion in ("leader", "both"):
            leader = self.args.leader
            delta = np.array([0., .025, 0., .015, 0., 0.]) * phase
            desired = (self.initial["robots"][leader]["q"] + delta - self.plant.joints[leader]) / self.dt
            limit = self.args.max_joint_accel * self.dt
            previous = self.plant.joint_velocity[leader]
            leader_velocity = np.clip(np.clip(desired, -self.args.max_joint_speed, self.args.max_joint_speed),
                                      previous - limit, previous + limit)
        return bases, leader_velocity

    def tick(self):
        """Advance at most one step. A rejected plant step never commits a task."""
        cycle_at = self.now
        if not self.active or self.paused:
            phase = "fault" if self.fault else "paused" if self.paused else "preview"
            self._record_evaluation(cycle_at, phase, reason=self.fault or (
                "simulation paused; virtual time is unchanged" if self.paused else ""))
            return False
        evaluation_scene = self._evaluation_scene(cycle_at) if self.evaluation_recorder is not None else None
        measurement = copy.deepcopy(self._evaluation_measurement) if self.evaluation_recorder is not None else None
        control_record, reason = None, ""
        phase = "alignment" if self.approach is not None and not self.approach.done else "tracking"
        transport = {"simulated": True, "hardware_commands": False,
                     "simulated_command_applied": False, "commands": {}}
        try:
            if self.approach is not None and not self.approach.done:
                control_record = {"kind": "simulation_alignment", "at": cycle_at, "active": True,
                                  "control_dt": self.dt, "progress": self.engine.progress,
                                  "alignment_progress": self.approach.progress}
                self._tick_alignment(transport=transport)
                control_record["next_alignment_progress"] = self.approach.progress
                return True
            proposal = self._prepare()
            control_record = proposal.record
            if evaluation_scene is not None:
                evaluation_scene = self._evaluation_scene(cycle_at, control_record=control_record)
            bases, leader_velocity = self._external_motion()
            velocities = {self.engine.follower: proposal.decision.qdot,
                          self.engine.leader: leader_velocity}
            transport["commands"] = {
                self.engine.follower: {"method": "speedL", "command": proposal.decision.twist.copy(),
                                       "joint_velocity": proposal.decision.qdot.copy(),
                                       "applied_method": "joint_velocity_step"}}
            self.plant.step(velocities, self.dt, bases=bases)
            transport["simulated_command_applied"] = True
            self.engine.commit(proposal)
            self.task_scale = proposal.decision.task_scale
            self.last_record = proposal.record
            self.steps += 1
            self._ingest(self.now)
            self._log("simulation_step", controller=self.last_record, bases=self.plant.bases,
                      joints=self.plant.joints, joint_velocity=self.plant.joint_velocity)
            return True
        except (ControlFault, ValueError, np.linalg.LinAlgError) as exc:
            phase, reason = "fault", str(exc)
            self.active, self.paused, self.ready = False, False, False
            self.fault = str(exc)
            self.engine.pause_gap_approach("simulation stopped: " + self.fault)
            self.plant.stop()
            self._preview()
            self._log("simulation_fault", reason=self.fault)
            return False
        finally:
            self._record_evaluation(cycle_at, phase, scene=evaluation_scene,
                                    control_record=control_record, transport=transport,
                                    measurement=measurement, reason=reason)

    def _tick_alignment(self, *, transport=None):
        velocities = self.approach.propose(self.plant.states(self.now))
        if transport is not None:
            transport["commands"] = {
                name: {"method": "speedJ", "command": command.copy(),
                       "joint_velocity": command.copy(), "applied_method": "joint_velocity_step"}
                for name, command in velocities.items() if name == self.engine.follower}
        self.plant.step(velocities, self.dt)  # Both external bases remain fixed.
        if transport is not None:
            transport["simulated_command_applied"] = True
        self.approach.commit()
        self.steps += 1
        self._ingest(self.now)
        self._log("simulation_alignment_step", progress=self.approach.progress,
                  duration=self.approach.duration, joints=self.plant.joints,
                  joint_velocity=self.plant.joint_velocity, bases=self.plant.bases)
        if self.approach.done:
            # The alignment ends with a stationary step. Start external motion
            # at phase zero, not at the wall/virtual time spent aligning.
            self._control_origin = self.elapsed
            self._prepare()
            self.message = "Alignment complete — running virtual joint control"
        else:
            self.message = f"Initial alignment {self.approach.progress * 100:.1f}% — leader/base motion held"

    def status(self, wall_now=None):
        """Fresh display data; original capture time is never relabeled as live."""
        states = self.plant.states(self.now)
        flanges = {name: self.plant.bases[name] @ flange_from_tcp(states[name].tcp, self.plant.offsets[name])
                   for name in ROBOTS}
        engine = self.engine
        goals = {engine.follower: flanges[engine.leader] @ engine.path.pose(engine.progress)}
        task = {"leader": engine.leader, "kind": engine.path.kind, "gap": engine.gap,
                "delta": engine.path.delta.copy(), "radius": engine.path.radius,
                "duration": engine.path.duration, "progress": engine.progress,
                "start_pose": engine.path.start}
        scene = {"at": time.monotonic() if wall_now is None else wall_now,
                 "bases": {name: value.copy() for name, value in self.plant.bases.items()},
                 "robots": {name: {"q": states[name].joints, "tcp": states[name].tcp,
                                   "tcp_offset": self.plant.offsets[name].copy()} for name in ROBOTS},
                 "goals": {name: value.copy() for name, value in goals.items()}, "task": task,
                 "measurement_age": 0., "calibration_valid": True,
                 "reason": "Ideal virtual base sensor; initial calibration captured " + self.initial["captured_at_utc"]}
        if self.approach is not None and not self.approach.done:
            scene["nominal_goals"] = {name: value.copy() for name, value in goals.items()}
            scene["command_goals"] = {
                name: self.plant.bases[name] @ flange_from_tcp(target, self.plant.offsets[name])
                for name, target in self.approach.targets.items()}
        elif self.last_record.get("next_commanded_target_tcp") is not None:
            name = self.engine.follower
            scene["nominal_goals"] = {key: value.copy() for key, value in goals.items()}
            scene["command_goals"] = {
                name: self.plant.bases[name] @ flange_from_tcp(
                    self.last_record["next_commanded_target_tcp"], self.plant.offsets[name])}
        try:
            snapshot = engine.state(self.now, active=False)
        except (ControlFault, ValueError, np.linalg.LinAlgError):
            snapshot = None
        gap_approach = engine.gap_approach_status(enabled=self._gap_approach_enabled(),
                                                 states=states, snapshot=snapshot)
        if self.gap_approach_error:
            gap_approach["error"] = self.gap_approach_error
        # Use this post-step robot state and committed path progress together;
        # last_record and next_commanded_target_tcp belong to the previous step.
        error_scene = {**scene, "at": self.elapsed}
        error_target, error_relative = None, None
        if self.approach is not None and not self.approach.done:
            error_target = self.approach.targets.get(engine.follower)
            error_source, error_phase = "initial_alignment_endpoint", "alignment"
        else:
            error_relative = engine.path.pose(engine.progress)
            if engine.tracking_offset is not None:
                error_relative = error_relative @ engine.tracking_offset
                error_source = "tracking_accepted_relative"
            else:
                error_source = "preview_nominal_relative" if not self.active else "tracking_nominal_relative"
            error_phase = "tracking" if self.active else "preview"
        error_sample = position_error_sample(
            error_scene, now=self.elapsed, leader=engine.leader, follower=engine.follower,
            phase="paused" if self.paused else error_phase,
            target_tcp=error_target, relative_target=error_relative, target_source=error_source,
            clock="simulation", episode=self.episode,
            reason=self.fault)
        return {"scene": scene, "active": self.active, "ready": self.ready, "sim_paused": self.paused,
                "message": f"SIM {self.elapsed:.2f} s / {self.args.sim_motion}: {self.message}",
                "progress": self.engine.progress, "task_scale": self.task_scale,
                "task_paused": self.engine.task_paused, "measurement_age": 0.,
                "gap_approach": gap_approach,
                "position_error": error_sample,
                "evaluation_recording": (self.evaluation_recorder.status()
                                         if self.evaluation_recorder is not None else {"state": "idle"}),
                "alignment_progress": self.approach.progress if self.approach else None,
                "fault_category": "simulation" if self.fault else ""}


def run(args):
    if args.execute or not args.simulate:
        raise ValueError("simulation requires --simulate without --execute")
    print("[sim] Initial capture is receive-only; subsequent control runs in virtual joints.", flush=True)
    initial = (load_initial_state(args.initial_state) if args.initial_state is not None
               else capture_initial_state(args))
    # capture_initial_state guarantees all camera/RTDE resources are closed here.
    output = BUNDLE / "runs" / "simulation" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    snapshot_path = save_initial_state(output / "initial_state.json", initial)
    print(f"[sim] Initial state: {snapshot_path}", flush=True)
    print("[sim] " + NOTICE, flush=True)
    assumed = [name for name in ROBOTS if initial["robots"][name]["tcp_offset_source"] == "zero_assumed"]
    if assumed:
        print("[sim] Assumed zero TCP offsets: " + ", ".join(assumed) + "; recapture with actual offsets if tools are installed.", flush=True)
    print("[sim] Initial velocities set to zero. Virtual base sensor has no camera noise or latency.", flush=True)
    log_path = args.log if args.log is not None else output / "simulation.jsonl"
    with ExperimentLogger(log_path) as logger, closing(
            PositionErrorRecorder(BUNDLE / "runs" / "error_segments")) as error_recorder, ExitStack() as recordings:
        session = SimulationSession(args, initial, logger)
        evaluation_settings = dict(vars(session.args))
        kalman_config = getattr(session.engine.estimator, "config", None)
        if is_dataclass(kalman_config):
            evaluation_settings["relative_kalman_parameters"] = asdict(kalman_config)
        evaluation_recorder = recordings.enter_context(closing(EvaluationRecorder(
            logger, log_path, output / "evaluation_sessions", settings=evaluation_settings,
            leader=session.engine.leader, follower=session.engine.follower, clock="simulation")))
        session.evaluation_recorder = evaluation_recorder
        logger.log({"kind": "simulation_settings", "simulation": True, "hardware_commands": False,
                    "initial_state": str(snapshot_path), "captured_at_utc": initial["captured_at_utc"],
                    "settings": vars(session.args), "plant": "seeded nominal FK; joint velocity integration",
                    "base_sensor": "ideal virtual poses", "initial_velocity": "zero"})
        from tkinter import ttk
        from urcal.control.gui import TargetPanel
        stop = threading.Event()

        def recording_command(command):
            if isinstance(command, dict) and command.get("type") == "evaluation":
                return session.evaluation_request(command)
            return error_recorder.request(command)

        panel = TargetPanel(session.args, session.command, stop, recording_command=recording_command)
        panel._view._models = session.plant.models
        ttk.Label(panel.root, text=NOTICE, foreground="#8a4100", padding=8,
                  font=("TkDefaultFont", 11, "bold"), wraplength=950).grid(row=1, column=0, sticky="ew")
        print(f"[sim] Log: {Path(log_path)}", flush=True)
        previous_message = None
        recording_notice = None
        try:
            while not stop.is_set():
                started = time.monotonic()
                status = session.status(started)
                error_recorder.record(status["position_error"], emitted_at=started)
                recording_status = error_recorder.status()
                status["error_recording"] = recording_status
                notice = (recording_status.get("state"), recording_status.get("path"), recording_status.get("error"))
                if notice != recording_notice and notice[0] != "idle":
                    print(f"[error CSV] {notice[0]}: {notice[2] or notice[1] or ''}", flush=True)
                recording_notice = notice
                if session.message != previous_message:
                    print("[sim] " + status["message"], flush=True)
                    previous_message = session.message
                panel.poll(status)
                if stop.is_set():
                    break
                session.tick()
                stop.wait(max(0., session.dt - (time.monotonic() - started)))
        finally:
            try:
                session._record_evaluation(session.now, "shutdown", reason="simulation shutdown")
                evaluation_recorder.close(reason="shutdown")
            finally:
                session.plant.stop()
                error_recorder.close()
                recording_status = error_recorder.status()
                if recording_status.get("state") not in (None, "idle"):
                    print(f"[error CSV] {recording_status.get('state')}: "
                          f"{recording_status.get('error') or recording_status.get('path') or ''}", flush=True)
                panel.close()


def main(argv=None):
    args = parse_args(["--simulate", *(sys.argv[1:] if argv is None else argv)])
    try:
        run(args)
    except KeyboardInterrupt:
        print("[sim] Closed", flush=True)
    except Exception as exc:
        raise SystemExit("[SIM STOP] " + str(exc)) from exc


if __name__ == "__main__":
    main()
