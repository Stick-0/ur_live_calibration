"""Capture live initial conditions once, then run an isolated joint simulation.

The virtual clock advances by one fixed control period per accepted step. GUI
delays slow playback; they never cause a large integration step or catch-up.
Saved snapshots require neither camera, inference checkpoints nor robot SDKs.
"""

import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

from urcal.control.geometry import (fixed_facing_targets, flange_from_tcp, inverse,
                                    matrix_pose, model_from_controller_base, pose_delta, pose_matrix)
from urcal.control.policy import CalibrationFrame, ControlFault, bounded_twist
from urcal.control.runtime import FacingPlanner, parse_args, task_tolerance_options
from urcal.cooperative.controller import CooperativeController
from urcal.cooperative.logging import ExperimentLogger
from urcal.paths import BUNDLE
from urcal.simulation.plant import JointPlant, SimulationFault
from urcal.simulation.snapshot import capture_initial_state, load_initial_state, save_initial_state, validate_initial_state


ROBOTS = ("ur5e", "ur10e")
NOTICE = "SIMULATION — captured initial state; hardware disconnected; no hardware commands."


class SimulationSession:
    """Controller/plant coupling without any hardware transport or wall clock IO."""

    def __init__(self, args, initial, logger=None):
        if args.execute or not args.simulate:
            raise ValueError("simulation requires --simulate without --execute")
        self.args = copy.deepcopy(args)
        self.initial = validate_initial_state(initial)
        for name in ROBOTS:
            offset = self.initial["robots"][name]["tcp_offset"]
            specified = getattr(args, name + "_tcp_offset")
            if specified is not None and not np.allclose(pose_matrix(specified), offset, atol=1e-9, rtol=0):
                raise ValueError(name + ": TCP offset differs from the saved initial state; recapture it")
            setattr(self.args, name + "_tcp_offset", matrix_pose(offset).tolist())
        if args.mode == "hold" and args.sim_motion in ("leader", "both"):
            raise ValueError("hold commands both arms; virtual leader motion requires follow mode")
        self.logger = logger
        self.plant = JointPlant(self.initial, self.args)
        self.dt = 1. / self.args.control_hz
        self._mappings = {name: model_from_controller_base(name) for name in ROBOTS}
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
        if self.engine is None:
            return
        self.sequence += 1
        # This is an ideal virtual base sensor, NOT another camera measurement.
        # Original camera residuals/time remain in the immutable snapshot.
        packet = CalibrationFrame(self.sequence, at, self._calibration(),
                                  {name: 0. for name in ROBOTS}, {name: 0. for name in ROBOTS},
                                  {name: 7 for name in ROBOTS})
        if not self.engine.ingest(packet, at):
            raise SimulationFault("virtual base sensor rejected: " + self.engine.last_rejection)

    def reset(self):
        self.plant.reset()
        self.episode += 1
        self.active, self.paused, self.fault = False, False, ""
        self.steps, self.sequence = 0, 0
        self.approach = None
        self.approach_nominal_targets, self.approach_target_adjustments = None, None
        self._alignment_planned = False
        self._control_origin = 0.
        self.planner = FacingPlanner(self.args) if self.args.mode == "hold" else None
        self.engine = (CooperativeController(self.args, models=self.plant.models, guard=self.plant.guard)
                       if self.args.mode == "follow" else None)
        if self._target_command is not None:
            (self.engine or self.planner).apply_target(copy.deepcopy(self._target_command))
        if self.engine is not None:
            for i in range(self.args.stable_frames):
                self._ingest(self.now - (self.args.stable_frames - i - 1) * self.dt)
        self.last_record, self.task_scale = {}, 0.
        self._preview()
        self._log("simulation_reset")

    def _hold_proposal(self, states):
        goals, targets = self.planner.plan(self._calibration(), states)
        approximate = bool(getattr(self.args, "target_tolerance_m", 0.)
                           or getattr(self.args, "target_tolerance_deg", 0.))
        self.planner.check_envelope(states, {name: state.tcp for name, state in states.items()}
                                    if approximate else targets)
        allowed, reason, _ = self.plant.guard.check(self.plant.bases, self.plant.joints)
        if not allowed:
            raise SimulationFault(reason)
        twists = {}
        for name in ROBOTS:
            # J changes after a joint integration step, so the measured TCP
            # velocity can lie just outside the previous command's speed ball.
            # Project the seed and reserve that correction from the slew budget.
            # Final checks below still use the unmodified measured velocity.
            seed = states[name].speed.copy()
            accelerations = []
            for part, speed, accel in ((slice(0, 3), self.args.max_speed, self.args.max_accel),
                                        (slice(3, 6), self.args.max_angular_speed, self.args.max_angular_accel)):
                magnitude = np.linalg.norm(seed[part])
                if magnitude > speed:
                    seed[part] *= speed / magnitude
                remaining = accel - np.linalg.norm(seed[part] - states[name].speed[part]) / self.dt
                if remaining <= 0:
                    raise SimulationFault(name + ": cannot return to TCP speed limit within acceleration bound")
                accelerations.append(remaining)
            twists[name] = bounded_twist(
                states[name].tcp, targets[name], seed, self.dt,
                kp_linear=self.args.kp_linear, kp_angular=self.args.kp_angular,
                max_linear=self.args.max_speed, max_angular=self.args.max_angular_speed,
                max_accel=accelerations[0], max_angular_accel=accelerations[1],
                max_target_m=self.args.max_target_m, max_target_rad=np.deg2rad(self.args.max_target_deg),
                **task_tolerance_options(self.args, self.plant.offsets[name]))
            model, guard = self.plant.models[name], self.plant.guard
            if not approximate:
                if not guard.pose_allowed(name, targets[name]):
                    raise SimulationFault(name + ": target outside TCP workspace")
                if model.inverse(targets[name], states[name].joints, guard.lower[name], guard.upper[name]) is None:
                    raise SimulationFault(name + ": target IK failed")
        qdots = self.plant.twists_to_qdots(twists, self.dt)
        # Joint clipping can change the achieved Cartesian twist. Verify it too.
        for name in ROBOTS:
            _, jacobian = self.plant.models[name].pose_and_jacobian(states[name].joints)
            actual = jacobian @ qdots[name]
            for part, speed, accel in ((slice(0, 3), self.args.max_speed, self.args.max_accel),
                                        (slice(3, 6), self.args.max_angular_speed, self.args.max_angular_accel)):
                if (np.linalg.norm(actual[part]) > speed + 1e-9
                        or np.linalg.norm(actual[part] - states[name].speed[part]) > accel * self.dt + 1e-9):
                    raise SimulationFault(name + ": virtual Cartesian speed/acceleration limit")
        if approximate:
            # These virtual commands integrate qdot, so their FK poses are
            # reachable by construction. Validate that actual joint sweep,
            # not an exact IK solution of the unchanged noisy nominal target.
            for fraction in (.5, 1.):
                joints = {name: states[name].joints + fraction * self.dt * qdots[name] for name in ROBOTS}
                allowed, reason, _ = self.plant.guard.check(self.plant.bases, joints)
                if not allowed:
                    raise SimulationFault("next virtual hold command: " + reason)
                flanges = {}
                for name in ROBOTS:
                    tcp, _ = self.plant.models[name].pose_and_jacobian(joints[name])
                    offset = self.plant.offsets[name]
                    flange = flange_from_tcp(tcp, offset)
                    for pose, origin in ((tcp, self.planner.initial_tcp[name]),
                                         (flange, flange_from_tcp(self.planner.initial_tcp[name], offset))):
                        if np.linalg.norm(pose[:3, 3] - origin[:3, 3]) > self.args.max_travel:
                            raise SimulationFault(name + ": next virtual hold command exceeds travel envelope")
                    if not self.plant.guard.pose_allowed(name, tcp):
                        raise SimulationFault(name + ": next virtual hold command outside TCP workspace")
                    dp, dr = pose_delta(flange, flange_from_tcp(targets[name], offset))
                    if np.linalg.norm(dp) > self.args.max_target_m or np.linalg.norm(dr) > np.deg2rad(self.args.max_target_deg):
                        raise SimulationFault(name + ": next virtual hold command exceeds tracking error")
                    flanges[name] = self.plant.bases[name] @ flange
                if np.linalg.norm(flanges["ur5e"][:3, 3] - flanges["ur10e"][:3, 3]) < self.args.min_gap:
                    raise SimulationFault("next virtual hold command flange separation below min_gap")
        return goals, targets, twists, qdots

    def _prepare(self):
        states = self.plant.states(self.now)
        if self.engine is not None:
            return self.engine.prepare(self.now, states, self.dt, active=self.active)
        return self._hold_proposal(states)

    def _plan_alignment(self):
        """Plan a virtual start-up move when the final pose is outside tracking limits.

        Tracking assumes an already aligned pair. Initial alignment instead
        follows a checked joint path at the configured speeds, accelerations
        and travel bounds. Failed exact IK may choose a separate precise goal
        inside the task allowance; the nominal target remains unchanged.
        Physical controllers never enter this path; the capture is not teleported.
        """
        states = self.plant.states(self.now)
        if self.planner is not None:
            _, targets = self.planner.plan(self._calibration(), states)
            self.planner.check_envelope(states, targets)
        else:
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
                self.ready = self.planner is None or self.planner.target_applied
                prefix = "Ready — Start simulation" if self.ready else "Set the target and click Apply target"
                self.message = f"{prefix}; initial alignment ~{self.approach.duration:.1f} s, then {self.args.mode} control"
                return
            self._prepare()
            if self.planner is not None and not self.planner.target_applied:
                self.message = "Set the target and click Apply target"
            else:
                self.ready = True
                self.message = "Ready — Start simulation"
        except (ControlFault, ValueError, np.linalg.LinAlgError) as exc:
            self.message = "WAIT: " + str(exc)

    def command(self, command):
        if isinstance(command, dict):
            if self.active:
                raise ValueError("Reset simulation before editing the target")
            (self.engine or self.planner).apply_target(command)
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
            if self.engine is None or not self.active:
                raise ValueError("task pause requires an active follow simulation")
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
            bases["ur5e"][:3, 3] += np.array([0., .01, 0.]) * phase
            bases["ur10e"][:3, 3] += np.array([0., -.01, 0.]) * phase
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
        if not self.active or self.paused:
            return False
        try:
            if self.approach is not None and not self.approach.done:
                self._tick_alignment()
                return True
            proposal = self._prepare()
            bases, leader_velocity = self._external_motion()
            if self.engine is not None:
                velocities = {self.engine.follower: proposal.decision.qdot,
                              self.engine.leader: leader_velocity}
                self.plant.step(velocities, self.dt, bases=bases)
                self.engine.commit(proposal)
                self.task_scale = proposal.decision.task_scale
                self.last_record = proposal.record
            else:
                goals, targets, twists, velocities = proposal
                if getattr(self.args, "target_tolerance_m", 0.) or getattr(self.args, "target_tolerance_deg", 0.):
                    self._check_hold_base_sweep(velocities, bases)
                self.plant.step(velocities, self.dt, bases=bases)
                self.last_record = {"goals": goals, "target_tcp": targets, "joint_command": velocities,
                                    "command_tcp": {name: self.plant.models[name].pose_and_jacobian(
                                        self.plant.joints[name])[0] for name in ROBOTS}}
            self.steps += 1
            self._ingest(self.now)
            self._log("simulation_step", controller=self.last_record, bases=self.plant.bases,
                      joints=self.plant.joints, joint_velocity=self.plant.joint_velocity)
            return True
        except (ControlFault, ValueError, np.linalg.LinAlgError) as exc:
            self.active, self.paused, self.ready = False, False, False
            self.fault = str(exc)
            self.plant.stop()
            self._preview()
            self._log("simulation_fault", reason=self.fault)
            return False

    def _check_hold_base_sweep(self, velocities, next_bases):
        """The ideal simulation knows its externally scheduled base motion."""
        for fraction in (.5, 1.):
            flanges = {}
            for name in ROBOTS:
                before, after = self.plant.bases[name], next_bases[name]
                dp, dr = pose_delta(before, after)
                base = before.copy()
                base[:3, 3] += fraction * dp
                base[:3, :3] = pose_matrix(np.r_[np.zeros(3), fraction * dr])[:3, :3] @ before[:3, :3]
                q = self.plant.joints[name] + fraction * self.dt * velocities[name]
                tcp, _ = self.plant.models[name].pose_and_jacobian(q)
                flanges[name] = base @ flange_from_tcp(tcp, self.plant.offsets[name])
            if np.linalg.norm(flanges["ur5e"][:3, 3] - flanges["ur10e"][:3, 3]) < self.args.min_gap:
                raise SimulationFault("virtual hold base motion brings flange separation below min_gap")

    def _tick_alignment(self):
        velocities = self.approach.propose(self.plant.states(self.now))
        self.plant.step(velocities, self.dt)  # Both external bases remain fixed.
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
        center, task = None, None
        if self.engine is not None:
            engine = self.engine
            goals = {engine.follower: flanges[engine.leader] @ engine.path.pose(engine.progress)}
            task = {"leader": engine.leader, "kind": engine.path.kind, "gap": engine.gap,
                    "delta": engine.path.delta.copy(), "radius": engine.path.radius,
                    "duration": engine.path.duration, "progress": engine.progress}
        else:
            goals = self.planner.hold_targets
            if goals is None:
                try:
                    goals = fixed_facing_targets(flanges["ur5e"], flanges["ur10e"], self.planner.gap, self.args.center_pose)
                except ValueError:
                    # Invalid initial geometry remains visible with its WAIT
                    # reason, even when no facing target can be constructed.
                    goals = {}
            if goals:
                center_matrix = goals["ur5e"].copy()
                center_matrix[:3, 3] = (goals["ur5e"][:3, 3] + goals["ur10e"][:3, 3]) / 2
                center = matrix_pose(center_matrix).tolist()
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
        elif self.engine is not None and self.last_record.get("next_commanded_target_tcp") is not None:
            name = self.engine.follower
            scene["nominal_goals"] = {key: value.copy() for key, value in goals.items()}
            scene["command_goals"] = {
                name: self.plant.bases[name] @ flange_from_tcp(
                    self.last_record["next_commanded_target_tcp"], self.plant.offsets[name])}
        elif self.planner is not None and self.last_record.get("command_tcp") is not None:
            scene["nominal_goals"] = {name: value.copy() for name, value in goals.items()}
            scene["command_goals"] = {
                name: self.plant.bases[name] @ flange_from_tcp(target, self.plant.offsets[name])
                for name, target in self.last_record["command_tcp"].items()}
        return {"scene": scene, "active": self.active, "ready": self.ready, "sim_paused": self.paused,
                "center_pose": center, "message": f"SIM {self.elapsed:.2f} s / {self.args.sim_motion}: {self.message}",
                "progress": self.engine.progress if self.engine else 0., "task_scale": self.task_scale,
                "task_paused": self.engine.task_paused if self.engine else False, "measurement_age": 0.,
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
    with ExperimentLogger(log_path) as logger:
        session = SimulationSession(args, initial, logger)
        logger.log({"kind": "simulation_settings", "simulation": True, "hardware_commands": False,
                    "initial_state": str(snapshot_path), "captured_at_utc": initial["captured_at_utc"],
                    "settings": vars(session.args), "plant": "seeded nominal FK; joint velocity integration",
                    "base_sensor": "ideal virtual poses", "initial_velocity": "zero"})
        from tkinter import ttk
        from urcal.control.gui import TargetPanel
        stop = threading.Event()
        panel = TargetPanel(session.args, session.command, stop)
        panel._view._models = session.plant.models
        ttk.Label(panel.root, text=NOTICE, foreground="#8a4100", padding=8,
                  font=("TkDefaultFont", 11, "bold"), wraplength=950).grid(row=1, column=0, sticky="ew")
        print(f"[sim] Log: {Path(log_path)}", flush=True)
        previous_message = None
        try:
            while not stop.is_set():
                started = time.monotonic()
                status = session.status(started)
                if session.message != previous_message:
                    print("[sim] " + status["message"], flush=True)
                    previous_message = session.message
                panel.poll(status)
                if stop.is_set():
                    break
                session.tick()
                stop.wait(max(0., session.dt - (time.monotonic() - started)))
        finally:
            session.plant.stop()
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
