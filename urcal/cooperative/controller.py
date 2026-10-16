"""State processing and relative-task planning between pose sensor and RTDE.

No hardware connections are made here. prepare() proposes a command; commit()
advances task time ONLY after its successful transmission. Preview never commits.
"""

from dataclasses import dataclass
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard
from urcal.cooperative.task import MotionInfeasible, MotionLimits, RelativePath, govern_task, target_components
from urcal.control.geometry import facing_follower, flange_from_tcp, inverse, matrix_pose, model_from_controller_base, pose_delta, pose_matrix, rotation_distance
from urcal.control.policy import CalibrationTracker, ControlFault
from urcal.cooperative.state import RelativeStateEstimator, RelativeStateError


ROBOTS = ("ur5e", "ur10e")


class ResearchFault(ControlFault):
    def __init__(self, category, message):
        self.category = category
        super().__init__(message)


def advance_origin_pose(pose, twist, dt):
    result = pose.copy()
    result[:3, 3] += np.asarray(twist[:3]) * dt
    result[:3, :3] = Rotation.from_rotvec(np.asarray(twist[3:]) * dt).as_matrix() @ pose[:3, :3]
    return result


def norm_limit(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * min(1., limit / norm) if norm else vector.copy()


@dataclass
class Proposal:
    goals: dict
    tcp_goals: dict
    twists: dict
    record: dict
    next_progress: float
    decision: object
    feedback: np.ndarray
    compensation: np.ndarray
    future_tcp_goals: dict


class CooperativeController:
    def __init__(self, args, models=None, guard=None):
        self.args = args
        self.leader = args.leader
        self.follower = next(name for name in ROBOTS if name != self.leader)
        self.commanded = (self.follower,)
        self.offsets = {name: pose_matrix(getattr(args, name + "_tcp_offset") or [0.]*6) for name in ROBOTS}
        self.mappings = {name: model_from_controller_base(name) for name in ROBOTS}
        self.models = models or {name: NominalKinematics(name, self.offsets[name]) for name in ROBOTS}
        self.guard = guard or WorkspaceGuard(self.models, args.workspace_config)
        self.gate = CalibrationTracker(
            max_age=args.max_calibration_age, stable_frames=args.stable_frames, alpha=1.,
            max_jump_m=args.max_calibration_jump_m, max_jump_rad=np.deg2rad(args.max_calibration_jump_deg),
            max_mean_px=args.max_reproj_px, max_pixel_px=args.max_point_reproj_px, min_points=args.min_points)
        self.estimator = RelativeStateEstimator(
            mode="direct" if args.method == "direct" else "predict", alpha=args.filter_alpha,
            velocity_alpha=args.velocity_alpha, max_prediction_age=args.prediction_horizon,
            max_measurement_age=args.max_calibration_age,
            max_linear_speed=args.max_base_speed, max_angular_speed=args.max_base_angular_speed)
        self.limits = MotionLimits(
            linear_speed=args.max_speed, angular_speed=args.max_angular_speed,
            linear_accel=args.max_accel, angular_accel=args.max_angular_accel,
            joint_speed=np.full(6, args.max_joint_speed), joint_accel=np.full(6, args.max_joint_accel),
            min_singular_value=args.min_singular_value)
        self.progress = 0.
        self.previous_twist = np.zeros(6)
        self.previous_qdot = np.zeros(6)
        self.previous_compensation = np.zeros(6)
        self.previous_compensation_qdot = np.zeros(6)
        self.previous_feedback = np.zeros(6)
        self.initial_tcp = None
        self.ever_ready = False
        self.invalid_since = None
        self.last_rejection = ""
        self.task_paused = False
        self.apply_target({"type": "target", "gap": args.gap, "path": args.path,
                           "path_delta": args.path_delta, "path_radius": args.path_radius,
                           "path_duration": args.path_duration})

    def apply_target(self, command):
        if command.get("type") != "target" or "center_pose" in command:
            raise ValueError("relative tasks require a follow target")
        gap = float(command["gap"])
        if not np.isfinite(gap) or gap < self.args.min_gap:
            raise ValueError("gap must be finite and at least min_gap")
        path = RelativePath(facing_follower(np.eye(4), gap),
                            kind=command.get("path", self.args.path),
                            delta=command.get("path_delta", self.args.path_delta),
                            radius=command.get("path_radius", self.args.path_radius),
                            duration=command.get("path_duration", self.args.path_duration))
        # The relative pose is configured, never re-captured from measured error.
        self.path, self.gap, self.progress = path, gap, 0.
        self.settle_elapsed, self.finished = 0., False
        self.previous_twist[:] = 0
        self.previous_qdot[:] = 0
        self.previous_compensation[:] = 0
        self.previous_compensation_qdot[:] = 0
        self.previous_feedback[:] = 0

    def ingest(self, packet, received_at):
        try:
            candidate_gate = copy.deepcopy(self.gate)
            accepted = candidate_gate.update(packet, received_at)
            if accepted:
                bases = {name: packet.poses[name] @ self.mappings[name] for name in ROBOTS}
                self.estimator.update_pair(bases, packet.captured_at, received_at, packet.sequence)
                self.gate = candidate_gate
                self.invalid_since, self.last_rejection = None, ""
                self.ever_ready = self.ever_ready or self.gate.ready
            return accepted
        except (ControlFault, RelativeStateError, ValueError) as exc:
            self.gate.reject(packet.sequence)
            self.last_rejection = str(exc)
            if self.invalid_since is None:
                self.invalid_since = received_at
            return False

    def state(self, now, active):
        if not active or not self.ever_ready:
            try:
                self.gate.check_fresh(now)
            except ControlFault as exc:
                raise ResearchFault("measurement_unavailable", str(exc)) from exc
        if self.invalid_since is not None and now - self.invalid_since > self.args.max_invalid_duration:
            raise ResearchFault("measurement_invalid", self.last_rejection)
        try:
            return self.estimator.estimate(now, leader=self.leader)
        except RelativeStateError as exc:
            raise ResearchFault("measurement_expired", str(exc)) from exc

    def prepare(self, now, states, dt, active=False):
        if not np.isfinite(dt) or dt <= 0:
            raise ResearchFault("timing", "invalid control period")
        snapshot = self.state(now, active)
        bases = snapshot.base_poses
        flanges, nominal_tcp, jacobians, qdots = {}, {}, {}, {}
        for name in ROBOTS:
            nominal_tcp[name], jacobians[name] = self.models[name].pose_and_jacobian(states[name].joints)
            actual = states[name].tcp  # UR computes this FK from measured joint encoders.
            dp, dr = pose_delta(nominal_tcp[name], actual)
            if np.linalg.norm(dp) > self.args.frame_check_m or np.linalg.norm(dr) > np.deg2rad(self.args.frame_check_deg):
                raise ResearchFault("frame_mismatch", f"{name}: nominal FK / controller TCP mismatch")
            flanges[name] = flange_from_tcp(actual, self.offsets[name])
            qdots[name] = np.asarray(states[name].joint_velocity, dtype=float)
            if qdots[name].shape != (6,) or not np.isfinite(qdots[name]).all():
                raise ResearchFault("robot_state", f"{name}: invalid joint velocity")
        follower, leader = self.follower, self.leader
        if np.any(np.abs(qdots[follower]) > self.args.max_joint_speed * 1.2):
            raise ResearchFault("joint_speed", "measured follower joint speed exceeded limit")
        measured_twist = np.asarray(states[follower].speed, dtype=float)
        if measured_twist.shape != (6,) or not np.isfinite(measured_twist).all():
            raise ResearchFault("robot_state", "invalid follower TCP velocity")
        if (np.linalg.norm(measured_twist[:3]) > self.args.max_speed * 1.1 + .001
                or np.linalg.norm(measured_twist[3:]) > self.args.max_angular_speed * 1.1 + .005):
            raise ResearchFault("tcp_speed", "measured follower TCP speed exceeded limit")
        if self.initial_tcp is None:
            self.initial_tcp = states[follower].tcp.copy()
        current_relative = inverse(bases[leader] @ flanges[leader]) @ bases[follower] @ flanges[follower]
        relative_goal = self.path.pose(self.progress)
        if min(np.linalg.norm(current_relative[:3, 3]), np.linalg.norm(relative_goal[:3, 3])) < self.args.min_gap:
            raise ResearchFault("clearance", "flange-origin separation below min_gap")
        dp_rel, dr_rel = pose_delta(current_relative, relative_goal)
        error_m, error_rad = np.linalg.norm(dp_rel), np.linalg.norm(dr_rel)
        _, leader_jacobian = self.models[leader].pose_and_jacobian(states[leader].joints, flange=True)
        target, base_comp, leader_comp, task_twist = target_components(
            bases, snapshot.base_twists, leader, follower, flanges[leader],
            leader_jacobian @ qdots[leader], relative_goal,
            self.path.derivative(self.progress), self.offsets[follower])
        current = states[follower].tcp
        dp, dr = pose_delta(current, target)
        if np.linalg.norm(dp) > self.args.max_target_m or np.linalg.norm(dr) > np.deg2rad(self.args.max_target_deg):
            raise ResearchFault("tracking_error", "relative target exceeds allowed tracking error")
        for pose in (current, target):
            if np.linalg.norm(pose[:3, 3] - self.initial_tcp[:3, 3]) > self.args.max_travel:
                raise ResearchFault("workspace", "TCP outside start-position travel envelope")
        valid, reason, clearance = self.guard.check(bases, {name: states[name].joints for name in ROBOTS})
        if not valid:
            raise ResearchFault("clearance", reason)
        # A deadband reduces stationary sensor jitter. Saturate/ramp FEEDBACK,
        # then assess the complete command; compensation itself is never clipped.
        desired_feedback = np.concatenate((
            norm_limit(dp * self.args.kp_linear, self.args.feedback_speed) if np.linalg.norm(dp) > self.args.position_deadband else np.zeros(3),
            norm_limit(dr * self.args.kp_angular, self.args.feedback_angular_speed) if np.linalg.norm(dr) > np.deg2rad(self.args.rotation_deadband_deg) else np.zeros(3)))
        feedback = self.previous_feedback.copy()
        feedback[:3] += norm_limit(desired_feedback[:3] - feedback[:3], self.args.max_accel * dt * .5)
        feedback[3:] += norm_limit(desired_feedback[3:] - feedback[3:], self.args.max_angular_accel * dt * .5)
        separated = self.args.method == "separated"
        compensation = base_comp + leader_comp if separated else np.zeros(6)
        work_twist = task_twist if separated else np.zeros(6)
        rho_cap = 1.
        if separated:
            rho_cap = min(1., max(0., (self.args.max_target_m - error_m) / (self.args.max_target_m - self.args.slow_error_m)),
                          max(0., (np.deg2rad(self.args.max_target_deg) - error_rad) /
                              np.deg2rad(self.args.max_target_deg - self.args.slow_error_deg)))
        if self.task_paused or self.invalid_since is not None or self.progress >= 1.:
            rho_cap = 0.
        next_targets = {}
        geometry_failure = None

        # Even with task rate zero, externally moving bases require motion.
        # Check that mandatory compensation itself has geometric room.
        try:
            compensation_qdot = np.linalg.solve(jacobians[follower], compensation)
        except np.linalg.LinAlgError as exc:
            raise ResearchFault("singularity", "cannot solve mandatory compensation joints") from exc
        for fraction in (.5, 1.):
            trial_bases = {name: advance_origin_pose(bases[name], snapshot.base_twists[name], dt * fraction) for name in ROBOTS}
            trial_q = {leader: states[leader].joints + qdots[leader] * dt * fraction,
                       follower: states[follower].joints + compensation_qdot * dt * fraction}
            allowed, why, _ = self.guard.check(trial_bases, trial_q)
            if not allowed:
                raise ResearchFault("compensation_infeasible", "mandatory compensation: " + why)

        def candidate_allowed(rho, twist, qdot):
            nonlocal geometry_failure
            # The governor constrains transitions from measured motion; also
            # preserve the slew limit between successive transmitted commands.
            if (np.linalg.norm(twist[:3] - self.previous_twist[:3]) > self.args.max_accel * dt + 1e-12
                    or np.linalg.norm(twist[3:] - self.previous_twist[3:]) > self.args.max_angular_accel * dt + 1e-12
                    or np.any(np.abs(qdot - self.previous_qdot) > self.args.max_joint_accel * dt + 1e-12)):
                geometry_failure = ("transition_infeasible", "successive command acceleration limit")
                return False
            next_progress = min(1., self.progress + rho * dt / self.path.duration) if self.path.kind != "hold" else self.progress
            desired = self.path.pose(next_progress)
            if np.linalg.norm(desired[:3, 3]) < self.args.min_gap:
                geometry_failure = ("clearance_infeasible", "next relative flange gap below min_gap")
                return False
            future_bases = {name: advance_origin_pose(bases[name], snapshot.base_twists[name], dt) for name in ROBOTS}
            future_leader = advance_origin_pose(flanges[leader], leader_jacobian @ qdots[leader], dt)
            next_target = inverse(future_bases[follower]) @ future_bases[leader] @ future_leader @ desired @ self.offsets[follower]
            if (not self.guard.pose_allowed(follower, next_target)
                    or np.linalg.norm(next_target[:3, 3] - self.initial_tcp[:3, 3]) > self.args.max_travel):
                geometry_failure = ("workspace_infeasible", "next target outside TCP workspace/travel envelope")
                return False
            goal_q = self.models[follower].inverse(next_target, states[follower].joints,
                                                   self.guard.lower[follower], self.guard.upper[follower])
            if goal_q is None:
                geometry_failure = ("reachability_infeasible", "next target has no nominal local IK solution within joint limits")
                return False
            # Check the nominal goal and sampled near-term swept configuration.
            if not self.guard.check(future_bases, {leader: states[leader].joints + qdots[leader] * dt, follower: goal_q})[0]:
                geometry_failure = ("clearance_infeasible", "next target violates geometric clearance")
                return False
            for fraction in (.5, 1.):
                trial_bases = {name: advance_origin_pose(bases[name], snapshot.base_twists[name], dt * fraction) for name in ROBOTS}
                trial_q = {leader: states[leader].joints + qdots[leader] * dt * fraction,
                           follower: states[follower].joints + qdot * dt * fraction}
                if not self.guard.check(trial_bases, trial_q)[0]:
                    geometry_failure = ("clearance_infeasible", "next command sweep violates geometric clearance")
                    return False
            next_targets[float(rho)] = next_target
            return True

        failure = None
        decision = None
        for feedback_scale in (1., .75, .5, .25, 0.):
            try:
                decision = govern_task(
                    compensation, feedback * feedback_scale, work_twist, jacobians[follower],
                    measured_twist, qdots[follower], dt, self.limits,
                    rho_cap=rho_cap, extra_feasible=candidate_allowed, grid_samples=9,
                    previous_compensation=self.previous_compensation,
                    previous_compensation_qdot=self.previous_compensation_qdot,
                    allow_deceleration=separated)
                feedback = feedback * feedback_scale
                break
            except MotionInfeasible as exc:
                failure = exc
                if exc.category in ("compensation_infeasible", "singularity"):
                    raise
        if decision is None:
            if geometry_failure is not None:
                raise ResearchFault(*geometry_failure)
            raise failure
        rho = decision.task_scale
        next_progress = min(1., self.progress + rho * dt / self.path.duration) if self.path.kind != "hold" else self.progress
        record = {
            "kind": "control", "at": now, "active": active, "method": self.args.method,
            "leader": leader, "follower": follower, "path": self.path.kind, "path_duration": self.path.duration,
            "progress": self.progress, "next_progress": next_progress,
            "task_scale": rho, "requested_task_scale": rho_cap, "task_paused": self.task_paused,
            "path_finished": self.path.kind != "hold" and self.progress >= 1.,
            "finished": self.finished,
            "within_finish_tolerance": error_m <= self.args.finish_position_m and error_rad <= np.deg2rad(self.args.finish_rotation_deg),
            "control_dt": dt, "settle_elapsed": self.settle_elapsed,
            "state_at": snapshot.at, "measurement_age": snapshot.measurement_age,
            "measurement_times": snapshot.measurement_times, "received_times": snapshot.received_times,
            "prediction_dt": snapshot.prediction_dt, "relative_base_pose": snapshot.relative_pose,
            "relative_base_twist": snapshot.relative_twist, "bases": bases,
            "base_twists": snapshot.base_twists, "estimated_relative_pose": current_relative,
            "desired_relative_pose": relative_goal, "relative_error_m": error_m,
            "relative_error_rad": error_rad, "target_tcp": target,
            "next_target_tcp": next_targets[float(rho)], "twist": decision.twist,
            "joint_command": decision.qdot, "base_compensation": base_comp,
            "leader_compensation": leader_comp, "applied_compensation": compensation,
            "task_twist_nominal": task_twist, "feedback": feedback,
            "feedback_scale": feedback_scale, "decision": decision.reason,
            "capsule_clearance_m": clearance, "last_rejection": self.last_rejection,
            "robots": {name: {"q": states[name].joints, "qd": qdots[name], "tcp": states[name].tcp,
                              "tcp_speed": states[name].speed, "robot_timestamp": states[name].robot_timestamp}
                       for name in ROBOTS}}
        return Proposal({follower: bases[follower] @ target @ inverse(self.offsets[follower])},
                        {follower: target}, {follower: decision.twist}, record, next_progress,
                        decision, feedback, compensation, {follower: next_targets[float(rho)]})

    def commit(self, proposal):
        self.progress = proposal.next_progress
        self.previous_twist = proposal.decision.twist.copy()
        self.previous_qdot = proposal.decision.qdot.copy()
        self.previous_compensation = proposal.compensation.copy()
        self.previous_compensation_qdot = np.asarray(proposal.decision.components["compensation_qdot"]).copy()
        self.previous_feedback = proposal.feedback.copy()
        if proposal.record["path_finished"] and proposal.record["within_finish_tolerance"]:
            self.settle_elapsed += proposal.record["control_dt"]
        else:
            self.settle_elapsed = 0.
        self.finished = self.finished or self.settle_elapsed >= self.args.finish_dwell
