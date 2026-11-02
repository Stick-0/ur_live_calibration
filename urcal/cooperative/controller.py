"""State processing and relative-task planning between pose sensor and RTDE.

No hardware connections are made here. prepare() proposes a command; commit()
advances task time ONLY after its successful transmission. Preview never commits.
"""

from dataclasses import dataclass, replace
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard
from urcal.cooperative.task import MotionInfeasible, MotionLimits, RelativePath, govern_task, target_components
from urcal.cooperative.follow import govern_follow
from urcal.cooperative.feedback import damped_feedback_velocity
from urcal.cooperative.approach import GapApproach
from urcal.cooperative.calibration import LiveCalibrationWindow
from urcal.cooperative.calibration_braking import prepare_calibration_braking
from urcal.control.cartesian_ramp import bounded_speedl_acceleration
from urcal.control.geometry import calibration_output_mapping, facing_follower, flange_from_tcp, inverse, pose_delta, pose_matrix
from urcal.control.policy import CalibrationTracker, ControlFault, PreviewCalibrationRecovery
from urcal.control.acquisition import StationaryCalibrationAcquisition, stationary_estimator
from urcal.cooperative.state import RelativeStateEstimator, RelativeStateError
from urcal.cooperative.pose_smoothing import RelativePoseSmoother
from urcal.cooperative.base_plane import constrain_base_snapshot
from urcal.cooperative.base_compensation import relative_base_compensation


ROBOTS = ("ur5e", "ur10e")


class ResearchFault(ControlFault):
    def __init__(self, category, message, details=None):
        self.category = category
        # Failed prepare() calls do not produce a normal control record. Keep
        # independent diagnostics on the exception for the runtime WAIT log.
        self.details = copy.deepcopy(details) if details is not None else {}
        super().__init__(message)


class CalibrationBrakingRequired(ControlFault):
    """Discard an unsent tracking proposal; the next tick must brake instead."""


def advance_origin_pose(pose, twist, dt):
    result = pose.copy()
    result[:3, 3] += np.asarray(twist[:3]) * dt
    result[:3, :3] = Rotation.from_rotvec(np.asarray(twist[3:]) * dt).as_matrix() @ pose[:3, :3]
    return result


def norm_limit(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * min(1., limit / norm) if norm else vector.copy()


def outside_tolerance(vector, allowance):
    """Continuous residual outside a Euclidean/rotation-vector task allowance."""
    norm = np.linalg.norm(vector)
    return vector * max(0., 1. - allowance / norm) if norm else vector.copy()


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
    tracking_offset: object = None


class CooperativeController:
    def __init__(self, args, models=None, guard=None):
        self.args = args
        self.leader = args.leader
        self.follower = next(name for name in ROBOTS if name != self.leader)
        self.commanded = (self.follower,)
        # Saved research configurations without this option keep their original
        # velocity model. The live CLI defaults to pose-only visual updates.
        self.base_motion_mode = getattr(args, "base_motion_mode", "velocity")
        if self.base_motion_mode not in ("pose", "velocity", "relative_kalman"):
            raise ValueError("base_motion_mode must be pose, velocity or relative_kalman")
        self.relative_kalman = self.base_motion_mode == "relative_kalman"
        self.target_tolerance_m = float(getattr(args, "target_tolerance_m", 0.))
        self.target_tolerance_rad = np.deg2rad(float(getattr(args, "target_tolerance_deg", 0.)))
        if (not np.isfinite([self.target_tolerance_m, self.target_tolerance_rad]).all()
                or min(self.target_tolerance_m, self.target_tolerance_rad) < 0):
            raise ValueError("target tolerances must be finite and nonnegative")
        self.approximate_targets = bool(self.target_tolerance_m or self.target_tolerance_rad)
        response = getattr(args, "leader_response", "strict")
        if response not in ("bounded", "strict"):
            raise ValueError("leader_response must be bounded or strict")
        self.follow_law = getattr(args, "follow_law", "feedforward")
        if self.follow_law not in ("feedforward", "pose"):
            raise ValueError("follow_law must be feedforward or pose")
        if self.follow_law == "pose" and (args.method != "separated"
                or self.base_motion_mode != "pose" or response != "bounded"):
            raise ValueError("pose follow_law requires method separated, pose base updates and bounded response")
        self.bounded_leader = (response == "bounded" and args.method == "separated"
                               and (self.approximate_targets or self.base_motion_mode in ("pose", "relative_kalman")))
        if self.relative_kalman and (args.method != "separated" or self.follow_law != "feedforward"
                                    or response != "bounded"):
            raise ValueError("relative_kalman requires separated, feedforward and bounded following")
        self.live_calibration = None
        if (getattr(args, "live_calibration_recovery", False)
                and self.base_motion_mode in ("pose", "relative_kalman") and self.bounded_leader):
            self.live_calibration = LiveCalibrationWindow(
                timeout=getattr(args, "calibration_recovery_timeout", 2.),
                blend_time=getattr(args, "calibration_blend_time", .5),
                max_shift_m=getattr(args, "calibration_recovery_max_shift_m", .15),
                max_shift_deg=getattr(args, "calibration_recovery_max_shift_deg", 15.))
        self._last_live_snapshot = None
        self.pose_smoother = None
        if self.base_motion_mode == "pose" and getattr(args, "base_pose_window", 0):
            self.pose_smoother = RelativePoseSmoother(
                window=args.base_pose_window,
                time_constant=getattr(args, "base_pose_smoothing_time", .35),
                deadband_m=getattr(args, "base_pose_deadband_m", .005),
                deadband_rad=np.deg2rad(getattr(args, "base_pose_deadband_deg", .5)),
                linear_rate=getattr(args, "base_pose_update_speed", .02),
                angular_rate=getattr(args, "base_pose_update_angular_speed", .05))
        # Historical research snapshots without this option retain their law.
        self.follow_damping_time = float(getattr(args, "follow_damping_time", 0.))
        if not np.isfinite(self.follow_damping_time) or self.follow_damping_time < 0.:
            raise ValueError("follow_damping_time must be finite and nonnegative")
        self.tracking_offset = None
        self.previous_base_compensation = np.zeros(6)
        self.previous_base_compensation_qdot = np.zeros(6)
        self.previous_relative_base_flange = np.zeros(6)
        self.offsets = {name: pose_matrix(getattr(args, name + "_tcp_offset") or [0.]*6) for name in ROBOTS}
        # Fixed dataset-frame conversion is enabled by default, including
        # old saved research argument sets that lacked this output option.
        self.ur10e_output_correction = getattr(args, "ur10e_output_correction", True)
        # Old saved configurations predate the mounting-plane assumption.
        self.common_base_plane = getattr(args, "common_base_plane", False)
        if not isinstance(self.common_base_plane, bool):
            raise ValueError("common_base_plane must be a boolean")
        if self.relative_kalman and not self.common_base_plane:
            raise ValueError("relative_kalman requires the common physical mounting plane")
        self.mappings = {name: calibration_output_mapping(
            name, ur10e_output_correction=self.ur10e_output_correction) for name in ROBOTS}
        self.models = models or {name: NominalKinematics(name, self.offsets[name]) for name in ROBOTS}
        self.guard = guard or WorkspaceGuard(self.models, args.workspace_config)
        self.gate = CalibrationTracker(
            max_age=args.max_calibration_age, stable_frames=args.stable_frames, alpha=1.,
            max_jump_m=args.max_calibration_jump_m, max_jump_rad=np.deg2rad(args.max_calibration_jump_deg),
            max_mean_px=args.max_reproj_px, max_pixel_px=args.max_point_reproj_px, min_points=args.min_points,
            check_reprojection=args.check_reprojection, check_individual_jumps=not self.relative_kalman)
        self.estimator = self._new_estimator()
        self.stationary_acquisition = StationaryCalibrationAcquisition(self.gate)
        self.preview_recovery = PreviewCalibrationRecovery()
        self.reacquisition_count = 0
        # Keep the original phase limits in args: initial alignment uses
        # max_speed, while following has its own TCP translation cap.
        follow_max_speed = getattr(args, "follow_max_speed", None)
        self.limits = MotionLimits(
            linear_speed=args.max_speed if follow_max_speed is None else follow_max_speed,
            angular_speed=args.max_angular_speed,
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
        self.gap_approach = GapApproach()
        self.apply_target({"type": "target", "gap": args.gap, "path": args.path,
                           "path_delta": args.path_delta, "path_radius": args.path_radius,
                           "path_duration": args.path_duration})

    def _new_estimator(self):
        args = self.args
        if self.relative_kalman:
            from urcal.cooperative.relative_kalman import PlanarRelativeStateEstimator, PlanarKalmanConfig, load_planar_kalman_config
            parameters = getattr(args, "relative_kalman_parameters", None)
            config = (PlanarKalmanConfig.from_dict(parameters) if parameters is not None else
                      load_planar_kalman_config(getattr(args, "relative_kalman_config", None)))
            return PlanarRelativeStateEstimator(leader=self.leader, mappings=self.mappings,
                                               max_measurement_age=args.max_calibration_age, config=config)
        mode = ("pose" if self.base_motion_mode == "pose" else
                "direct" if args.method == "direct" else "predict")
        return RelativeStateEstimator(
            mode=mode, alpha=1. if args.method == "direct" else args.filter_alpha,
            velocity_alpha=args.velocity_alpha, max_prediction_age=args.prediction_horizon,
            max_measurement_age=args.max_calibration_age,
            max_linear_speed=args.max_base_speed, max_angular_speed=args.max_base_angular_speed)

    def reset_after_stop(self):
        """Discard motion/sensor history only after measured standstill.

        The applied path, progress, pause choice and original travel origin
        remain unchanged. New post-stop frames and a new Arm must precede any
        new motion session; this method itself cannot authorize a command.
        """
        from urcal.control.recovery import fresh_calibration_tracker
        self.gate = fresh_calibration_tracker(self.gate)
        self.estimator = self._new_estimator()
        self.preview_recovery.reset()
        self.stationary_acquisition.reset()
        if self.live_calibration is not None:
            self.live_calibration.reset()
        self._last_live_snapshot = None
        self.ever_ready = False
        if self.pose_smoother is not None:
            self.pose_smoother.reset()
        self.invalid_since, self.last_rejection = None, ""
        for attribute in ("previous_twist", "previous_qdot", "previous_compensation",
                          "previous_compensation_qdot", "previous_feedback",
                          "previous_base_compensation", "previous_base_compensation_qdot"):
            getattr(self, attribute)[:] = 0.
        self.previous_relative_base_flange[:] = 0.
        self.pause_gap_approach("motion stopped; resume explicitly after Arm")
        self.tracking_offset = np.eye(4) if self.gap_approach.state != "idle" else None
        self.previous_task_scale = 0.
        self.settle_elapsed, self.finished = 0., False

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
        if self.gap_approach.state != "idle":
            self.task_paused = False
        self.gap_approach.reset()
        self.path, self.gap, self.progress = path, gap, 0.
        self.settle_elapsed, self.finished = 0., False
        self.previous_twist[:] = 0
        self.previous_qdot[:] = 0
        self.previous_compensation[:] = 0
        self.previous_compensation_qdot[:] = 0
        self.previous_feedback[:] = 0
        self.previous_task_scale = 0.
        self.tracking_offset = None
        self.previous_base_compensation[:] = 0
        self.previous_base_compensation_qdot[:] = 0
        self.previous_relative_base_flange[:] = 0
        if self.live_calibration is not None:
            self.live_calibration.reset()
        self._last_live_snapshot = None

        if self.pose_smoother is not None:
            self.pose_smoother.reset()

    def _gap_approach_block(self, enabled):
        if not enabled:
            return "available during active following"
        if not self.bounded_leader or self.base_motion_mode not in ("pose", "relative_kalman"):
            return "requires separated, bounded following with pose base updates"
        if self.invalid_since is not None or (self.live_calibration is not None and
                (self.live_calibration.waiting or self.live_calibration.blending)):
            return "waiting for live calibration recovery"
        if self.tracking_offset is None:
            return "waiting for the first committed following command"
        if self.gap_approach.state == "idle":
            if self.path.kind != "hold":
                return "approach requires stationary relative following"
            if self.args.min_gap > .15:
                return "configured minimum flange gap exceeds 15 cm"
            reference = self.path.pose(self.progress) @ self.tracking_offset
            if reference[2, 3] <= .15 + 1e-12:
                return "commanded axial gap is already at or below 15 cm"
        return ""

    def handle_gap_approach(self, action, *, enabled):
        """Change only the relative path; never reset an active motion history."""
        if action not in ("start", "pause", "resume"):
            raise ValueError("unknown gap approach action")
        if action == "pause":
            self.pause_gap_approach("operator paused approach; leader following continues")
            return
        # Explicit actions are idempotent, including delayed duplicate starts.
        if ((action == "start" and self.gap_approach.state != "idle") or
                (action == "resume" and self.gap_approach.state != "paused")):
            return
        reason = self._gap_approach_block(enabled)
        if reason:
            raise ValueError(reason)
        if action == "start":
            path = self.gap_approach.start(
                self.path, self.progress, self.tracking_offset,
                speed=getattr(self.args, "approach_speed", .005), min_gap=self.args.min_gap)
            # The former offset is already in path.start. Identity preserves
            # the command exactly and makes its final signed axial gap 15 cm.
            self.path, self.gap, self.progress = path, float(path.start[2, 3]), 0.
            self.tracking_offset = np.eye(4)
            self.settle_elapsed, self.finished = 0., False
        else:
            self.gap_approach.resume()
        self.task_paused = False

    def pause_gap_approach(self, reason):
        if self.gap_approach.pause(reason):
            self.task_paused = True

    def _gap_approach_temporary_hold(self):
        return bool(self.gap_approach.state == "running" and
                    (self.invalid_since is not None or
                     (self.live_calibration is not None and self.live_calibration.blending)))

    def gap_approach_status(self, *, enabled, states=None, snapshot=None):
        info = self.gap_approach.info(self.progress if self.gap_approach.state != "idle" else 0.)
        blocked = self._gap_approach_block(enabled)
        info["enabled"] = bool(enabled and self.gap_approach.state != "complete" and
                               (self.gap_approach.state == "running" or not blocked))
        info["reason"] = blocked or info["reason"]
        info["temporary_hold"] = self._gap_approach_temporary_hold()
        if info["temporary_hold"]:
            info["reason"] = "Approach waiting for fresh coordinates; continues automatically when ready"
        info["target_gap_m"] = .15
        offset = np.eye(4) if self.tracking_offset is None else self.tracking_offset
        info["commanded_gap_m"] = float((self.path.pose(self.progress) @ offset)[2, 3])
        info["estimated_gap_m"] = None
        snapshot = self._last_live_snapshot if snapshot is None else snapshot
        if states is not None and snapshot is not None:
            bases = snapshot.base_poses
            leader = bases[self.leader] @ flange_from_tcp(states[self.leader].tcp, self.offsets[self.leader])
            follower = bases[self.follower] @ flange_from_tcp(states[self.follower].tcp, self.offsets[self.follower])
            info["estimated_gap_m"] = float((inverse(leader) @ follower)[2, 3])
        return info

    def _following_reference(self, nominal, current_relative):
        """Keep one accepted task offset, without absorbing later following lag.

        Only commit() captures this offset, after a successfully sent command.
        Project an initial outside-tolerance state onto the permitted tube;
        neither preview nor a failed proposal changes the stored reference.
        """
        offset = self.tracking_offset
        if offset is None:
            dp, dr = pose_delta(nominal, current_relative)
            accepted = nominal.copy()
            accepted[:3, 3] += norm_limit(dp, self.target_tolerance_m)
            accepted[:3, :3] = Rotation.from_rotvec(norm_limit(dr, self.target_tolerance_rad)).as_matrix() @ nominal[:3, :3]
            offset = inverse(nominal) @ accepted
        return nominal @ offset, offset.copy()

    def _start_live_observation(self, now, reason, *, pause_approach=True):
        if self.relative_kalman:
            self.estimator.note_rejected(now, reason)
        if pause_approach:
            self.pause_gap_approach("calibration observation; resume approach after recovery")
        if self._last_live_snapshot is None:
            raise self._measurement_fault("measurement_unavailable", reason, now)
        # Collect independent replacement evidence during the short outlier
        # grace too. Entering braking must not discard those real captures or
        # assign them new receipt times. start() resets its collector, so keep
        # the existing one when this is the first transition to observation.
        candidate = copy.deepcopy(self.live_calibration.candidate)
        entering = not self.live_calibration.waiting
        self.live_calibration.start(self._last_live_snapshot, now, reason)
        if entering:
            self.live_calibration.candidate = candidate
            # The grace consumes part of the existing recovery budget; it
            # must not postpone the absolute loss deadline. In particular a
            # late good packet cannot restart a full observation interval.
            if self.invalid_since is not None:
                self.live_calibration.started_at = min(float(now), self.invalid_since)
                self.live_calibration.check_deadline(now)
        if self.invalid_since is None:
            self.invalid_since = now
        self.last_rejection = reason

    def _offer_live_candidate(self, packet, received_at):
        window = self.live_calibration
        window.check_deadline(received_at)
        candidate = window.candidate.offer(packet, received_at, self.gate)
        if candidate is None:
            if window.waiting:
                self.last_rejection = window.candidate.reason
                self.gate.reject(packet.sequence, reason=self.last_rejection)
            return False
        # Pose-only estimation has no visual base velocity to replay. This
        # installs only real, fresh candidate captures, without changing the
        # frozen coordinates used by the bounded braking proposals.
        estimator = stationary_estimator(self._new_estimator, candidate, self.mappings, received_at)
        estimator.estimate(received_at, leader=self.leader)
        if not window.waiting:
            # Three coherent fresh captures can establish a new baseline
            # while the old one is still usable. Blend from the held geometry
            # without inserting a gratuitous brake/restart between them.
            # This replacement has already passed the fresh-capture consensus
            # checks while normal following remains authorized. Freeze the
            # approach clock during blending, without latching operator pause.
            self._start_live_observation(received_at, self.last_rejection, pause_approach=False)
        window.complete(received_at)
        self.gate, self.estimator = candidate, estimator
        self.invalid_since, self.last_rejection, self.ever_ready = None, "", True
        return True

    def ingest(self, packet, received_at, *, allow_reacquire=False, active_recovery=False):
        active_recovery = active_recovery and self.live_calibration is not None
        if active_recovery:
            # A fresh packet arriving after the observation budget expires
            # cannot silently authorize continuing the old motion session.
            self.live_calibration.check_deadline(received_at)
            if (not self.live_calibration.waiting and self.invalid_since is not None
                    and not self._live_grace_usable(received_at)):
                self._start_live_observation(received_at, self.last_rejection)
            if self.live_calibration.waiting:
                return self._offer_live_candidate(packet, received_at)
        if (allow_reacquire and getattr(self.args, "initial_align", False)
                and getattr(self.args, "calibration_acquisition", "consecutive") == "consensus"):
            previous_sequence = self.stationary_acquisition.sequence
            candidate = self.stationary_acquisition.offer(packet, received_at)
            self.last_rejection = self.stationary_acquisition.reason if candidate is None else ""
            if candidate is None:
                self.gate.reject(packet.sequence, self.last_rejection)
                return False
            replaced = bool(self.gate.poses) and (
                self.gate.last_capture is None or received_at - self.gate.last_capture > self.gate.max_age
                or self.stationary_acquisition._distance(self.gate.poses, candidate.poses) > 1.)
            self.estimator = stationary_estimator(self._new_estimator, candidate, self.mappings, received_at)
            self.gate = candidate
            self.invalid_since, self.last_rejection, self.ever_ready = None, "", True
            if replaced:
                self.reacquisition_count += 1
            return previous_sequence != packet.sequence and candidate.last_capture == packet.captured_at
        self.stationary_acquisition.reset()
        # Live runtimes opt in only while no control session exists. Native
        # preflight, active control and other callers keep the strict baseline.
        if not allow_reacquire:
            self.preview_recovery.reset()
        try:
            candidate_gate = copy.deepcopy(self.gate)
            accepted = candidate_gate.update(packet, received_at)
            if accepted:
                bases = {name: packet.poses[name] @ self.mappings[name] for name in ROBOTS}
                self.estimator.update_pair(bases, packet.captured_at, received_at, packet.sequence)
                self.gate = candidate_gate
                self.invalid_since, self.last_rejection = None, ""
                self.ever_ready = self.ever_ready or self.gate.ready
                self.preview_recovery.reset()
                if active_recovery:
                    self.live_calibration.candidate.reset()
            return accepted
        except (ControlFault, RelativeStateError, ValueError) as exc:
            if self.relative_kalman:
                self.estimator.note_rejected(received_at, str(exc))
            if not active_recovery:
                self.pause_gap_approach("calibration rejected; resume approach after recovery")
            stable_count = self.gate.stable_count
            self.gate.reject(packet.sequence, reason=str(exc))
            self.last_rejection = str(exc)
            if self.invalid_since is None:
                self.invalid_since = received_at
            if active_recovery:
                # reject() consumes the sequence but normally resets startup
                # stability. An ACTIVE outlier does not erase the evidence for
                # the last accepted pose: hold it only while its original
                # capture is fresh and the short invalid-input budget remains.
                self.gate.stable_count = stable_count
                if not self._live_grace_usable(received_at):
                    self.gate.stable_count = 0
                    self._start_live_observation(received_at, str(exc))
                return self._offer_live_candidate(packet, received_at)
            if allow_reacquire:
                candidate = self.preview_recovery.offer(packet, received_at, self.gate)
                if candidate is not None:
                    # Replay only the new stable cluster. A new coordinate
                    # baseline must not become an old-to-new velocity spike,
                    # while speeds within the candidate cluster remain checked.
                    estimator = self._new_estimator()
                    try:
                        for sample, receipt in self.preview_recovery.samples:
                            bases = {name: sample.poses[name] @ self.mappings[name] for name in ROBOTS}
                            estimator.update_pair(bases, sample.captured_at, receipt, sample.sequence)
                        estimator.estimate(received_at, leader=self.leader)
                    except (RelativeStateError, ValueError) as recovery_exc:
                        self.last_rejection = "preview calibration recovery: " + str(recovery_exc)
                        self.gate.reject(packet.sequence, reason=self.last_rejection)
                        self.preview_recovery.reset()
                    else:
                        self.gate, self.estimator = candidate, estimator
                        self.invalid_since, self.last_rejection = None, ""
                        self.ever_ready = True
                        self.reacquisition_count += 1
                        self.preview_recovery.reset()
                        return True
            return False

    def _measurement_fault(self, category, message, now):
        capture = self.gate.last_capture
        details = {"last_accepted_capture": capture,
                   "last_accepted_age": None if capture is None else now - capture,
                   "max_measurement_age": self.args.max_calibration_age,
                   "last_rejection": self.last_rejection}
        if (("stale" in message or "expired" in message) and self.last_rejection
                and self.last_rejection not in message):
            message += "; last rejected pair: " + self.last_rejection
        return ResearchFault(category, message, details)

    def state(self, now, active):
        if active and self.live_calibration is not None and self.live_calibration.waiting:
            raise CalibrationBrakingRequired("calibration observation requires a braking proposal")
        if not active or not self.ever_ready:
            try:
                self.gate.check_fresh(now)
            except ControlFault as exc:
                raise self._measurement_fault("measurement_unavailable", str(exc), now) from exc
        if self.invalid_since is not None and now - self.invalid_since > self.args.max_invalid_duration:
            raise self._measurement_fault("measurement_invalid", self.last_rejection, now)
        try:
            snapshot = self.estimator.estimate(now, leader=self.leader)
        except RelativeStateError as exc:
            raise self._measurement_fault("measurement_expired", str(exc), now) from exc
        if self.common_base_plane and not self.relative_kalman:
            snapshot = constrain_base_snapshot(snapshot, self.leader, self.mappings)
        if active and self.pose_smoother is not None:
            # Retained across live estimator replacement: a new PnP branch
            # must not bypass gradual coordinate application. Only a new
            # stopped/target session resets this filter; no timestamps change.
            snapshot = self.pose_smoother.apply(snapshot, now, self.leader)
        relative_blending = False
        if active and self.live_calibration is not None:
            relative_blending = self.relative_kalman and self.live_calibration.blending
            snapshot = self.live_calibration.blend(snapshot, now, self.leader)
        # Independent temporal interpolation can leave the common-plane
        # manifold even when all input image pairs satisfy it exactly.
        if self.common_base_plane and active and (not self.relative_kalman or relative_blending):
            snapshot = constrain_base_snapshot(snapshot, self.leader, self.mappings)
        if self.relative_kalman:
            # Recovery blending must not reuse pre-blend velocities or expose
            # two different G values to task geometry and feedforward.
            information = copy.deepcopy(snapshot.relative_filter)
            information["G_control"] = inverse(snapshot.base_poses[self.follower]) @ snapshot.base_poses[self.leader]
            if self.invalid_since is not None or relative_blending:
                information.update(reason="rejected observation or recovery blending")
                information["lambda"] = 0.
                information["origin_velocity"] = np.zeros(6)
            snapshot = replace(snapshot, relative_filter=information)
        self._last_live_snapshot = copy.deepcopy(snapshot)
        return snapshot

    def _live_grace_usable(self, now):
        if (self._last_live_snapshot is None or not self.ever_ready
                or self.invalid_since is None
                or not 0. <= now - self.invalid_since <= self.args.max_invalid_duration):
            return False
        try:
            self.gate.check_fresh(now)
            self.estimator.estimate(now, leader=self.leader)
        except (ControlFault, RelativeStateError):
            return False
        return True

    def check_live_calibration(self, now):
        """Check the shared observation deadline without advancing pose blending."""
        window = self.live_calibration
        if window is None:
            raise ControlFault("live calibration observation is not enabled")
        window.check_deadline(now)
        if window.waiting:
            return False
        if self._live_grace_usable(now):
            return True
        reason = self.last_rejection if self.invalid_since is not None else ""
        if not reason:
            try:
                self.gate.check_fresh(now)
                self.estimator.estimate(now, leader=self.leader)
            except (ControlFault, RelativeStateError) as exc:
                reason = str(exc)
        if reason:
            self._start_live_observation(now, reason)
            return False
        return True

    def prepare_live(self, now, states, dt):
        if self.live_calibration is None:
            return self.prepare(now, states, dt, active=True)
        if self.check_live_calibration(now):
            proposal = self.prepare(now, states, dt, active=True)
        else:
            proposal = prepare_calibration_braking(self, now, states, dt, self.live_calibration.frozen)
        proposal.record["live_calibration"] = self.live_calibration.status(now)
        proposal.record["live_calibration"].update({
            "grace": self.invalid_since is not None and not self.live_calibration.waiting,
            "grace_elapsed": 0. if self.invalid_since is None else max(0., now - self.invalid_since),
            "grace_timeout": self.args.max_invalid_duration,
            "grace_reason": self.last_rejection,
            "last_accepted_capture": self.gate.last_capture,
        })
        return proposal

    def prepare(self, now, states, dt, active=False, *, snapshot_override=None, calibration_braking=False):
        if not np.isfinite(dt) or dt <= 0:
            raise ResearchFault("timing", "invalid control period")
        if calibration_braking and snapshot_override is None:
            raise ResearchFault("robot_state", "calibration braking needs an explicit frozen snapshot")
        snapshot = self.state(now, active) if snapshot_override is None else snapshot_override
        bounded_command = self.bounded_leader or calibration_braking
        incremental_target = self.approximate_targets or calibration_braking
        bases = snapshot.base_poses
        flanges, nominal_tcp, jacobians, qdots, fk_corrections = {}, {}, {}, {}, {}
        for name in ROBOTS:
            nominal_tcp[name], jacobians[name] = self.models[name].pose_and_jacobian(states[name].joints)
            actual = states[name].tcp  # UR computes this FK from measured joint encoders.
            dp, dr = pose_delta(nominal_tcp[name], actual)
            if np.linalg.norm(dp) > self.args.frame_check_m or np.linalg.norm(dr) > np.deg2rad(self.args.frame_check_deg):
                raise ResearchFault("frame_mismatch", f"{name}: nominal FK / controller TCP mismatch")
            flanges[name] = flange_from_tcp(actual, self.offsets[name])
            # Anchor short-horizon nominal FK at the measured controller TCP.
            # This is a local motion model, never a replacement calibration.
            fk_corrections[name] = inverse(nominal_tcp[name]) @ actual
            if incremental_target:
                jacobians[name] = jacobians[name].copy()
                jacobians[name][:3] += np.cross(
                    jacobians[name][3:].T, actual[:3, 3] - nominal_tcp[name][:3, 3]).T
            qdots[name] = np.asarray(states[name].joint_velocity, dtype=float)
            if qdots[name].shape != (6,) or not np.isfinite(qdots[name]).all():
                raise ResearchFault("robot_state", f"{name}: invalid joint velocity")
        follower, leader = self.follower, self.leader
        if np.any(np.abs(qdots[follower]) > self.args.max_joint_speed * 1.2):
            raise ResearchFault("joint_speed", "measured follower joint speed exceeded limit")
        measured_twist = np.asarray(states[follower].speed, dtype=float)
        if measured_twist.shape != (6,) or not np.isfinite(measured_twist).all():
            raise ResearchFault("robot_state", "invalid follower TCP velocity")
        if (np.linalg.norm(measured_twist[:3]) > self.limits.linear_speed * 1.1 + .001
                or np.linalg.norm(measured_twist[3:]) > self.args.max_angular_speed * 1.1 + .005):
            raise ResearchFault("tcp_speed", "measured follower TCP speed exceeded limit")
        if self.initial_tcp is None:
            self.initial_tcp = states[follower].tcp.copy()
        current_relative = inverse(bases[leader] @ flanges[leader]) @ bases[follower] @ flanges[follower]
        relative_goal = self.path.pose(self.progress)
        tracking_relative, tracking_offset = relative_goal, None
        if self.bounded_leader:
            tracking_relative, tracking_offset = self._following_reference(relative_goal, current_relative)
        if (np.linalg.norm(current_relative[:3, 3]) < self.args.min_gap
                or (not calibration_braking and np.linalg.norm(relative_goal[:3, 3]) < self.args.min_gap)):
            raise ResearchFault("clearance", "flange-origin separation below min_gap")
        dp_rel, dr_rel = pose_delta(current_relative, relative_goal)
        error_m, error_rad = np.linalg.norm(dp_rel), np.linalg.norm(dr_rel)
        nominal_leader_flange, leader_jacobian = self.models[leader].pose_and_jacobian(states[leader].joints, flange=True)
        if self.approximate_targets:
            leader_jacobian = leader_jacobian.copy()
            leader_jacobian[:3] += np.cross(
                leader_jacobian[3:].T, flanges[leader][:3, 3] - nominal_leader_flange[:3, 3]).T
        pose_command_target = None
        if self.follow_law == "pose":
            # Recompute the complete TCP target from this tick's coordinates.
            # The nominal goal remains available for the unchanged task gates;
            # feedback follows the one accepted, fixed relative offset.
            follower_from_leader_flange = inverse(bases[follower]) @ bases[leader] @ flanges[leader]
            target = follower_from_leader_flange @ relative_goal @ self.offsets[follower]
            pose_command_target = follower_from_leader_flange @ tracking_relative @ self.offsets[follower]
            base_comp, leader_comp, task_twist = (np.zeros(6) for _ in range(3))
            nominal_base_comp, nominal_leader_comp, nominal_task_twist = (np.zeros(6) for _ in range(3))
        else:
            target, base_comp, leader_comp, task_twist = target_components(
                bases, snapshot.base_twists, leader, follower, flanges[leader],
                leader_jacobian @ qdots[leader], relative_goal,
                self.path.derivative(self.progress), self.offsets[follower])
            nominal_base_comp, nominal_leader_comp, nominal_task_twist = (
                base_comp.copy(), leader_comp.copy(), task_twist.copy())
        if self.approximate_targets and self.follow_law != "pose":
            # Carry the currently achieved relative pose with the moving bases
            # and leader. A feasible offset inside the task allowance must not
            # become an artificial lever-arm error under leader rotation.
            _, base_comp, leader_comp, task_twist = target_components(
                bases, snapshot.base_twists, leader, follower, flanges[leader],
                leader_jacobian @ qdots[leader], current_relative,
                self.path.derivative(self.progress), self.offsets[follower])
        if self.bounded_leader and self.follow_law != "pose":
            # The fixed allowed offset follows the nominal relative path.
            # Include the lever arm introduced by its translated origin.
            tracking_derivative = self.path.derivative(self.progress).copy()
            tracking_derivative[:3] += np.cross(
                tracking_derivative[3:], relative_goal[:3, :3] @ tracking_offset[:3, 3])
            _, _, _, task_twist = target_components(
                bases, snapshot.base_twists, leader, follower, flanges[leader],
                leader_jacobian @ qdots[leader], tracking_relative,
                tracking_derivative, self.offsets[follower])
        relative_base = None
        if self.relative_kalman:
            information = snapshot.relative_filter
            confidence = 0. if calibration_braking else information["lambda"]
            relative_base = relative_base_compensation(
                inverse(bases[follower]) @ bases[leader], flanges[follower], states[follower].tcp,
                information["raw_origin_velocity"], confidence,
                self.previous_relative_base_flange, dt,
                linear_speed=getattr(self.args, "relative_base_speed", .10),
                angular_speed=getattr(self.args, "relative_base_angular_speed", .15),
                linear_accel=getattr(self.args, "relative_base_accel", .20),
                angular_accel=getattr(self.args, "relative_base_angular_accel", .30))
            base_comp = relative_base["effective_tcp"].copy()
            nominal_base_comp = relative_base["raw_flange"].copy()
            nominal_base_comp[:3] += np.cross(nominal_base_comp[3:],
                                             states[follower].tcp[:3, 3] - flanges[follower][:3, 3])
        current = states[follower].tcp
        dp, dr = pose_delta(current, target)
        tcp_error_m, tcp_error_rad = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        tracking_error_m, tracking_error_rad = ((error_m, error_rad) if self.approximate_targets
                                                else (tcp_error_m, tcp_error_rad))
        rotation_limit = np.deg2rad(self.args.max_target_deg)
        if not calibration_braking and (tracking_error_m > self.args.max_target_m or tracking_error_rad > rotation_limit):
            raise ResearchFault(
                "tracking_error",
                f"{follower} relative target error: {tracking_error_m * 1000:.1f}/{self.args.max_target_m * 1000:.1f} mm, "
                f"{np.rad2deg(tracking_error_rad):.1f}/{self.args.max_target_deg:.1f} deg (actual/limit)",
                {"leader": leader, "follower": follower, "frame": follower + "_controller_base",
                 "error_origin": "flange" if self.approximate_targets else "TCP",
                 "position_error_m": tracking_error_m, "rotation_error_rad": tracking_error_rad,
                 "position_limit_m": self.args.max_target_m, "rotation_limit_rad": float(rotation_limit),
                 "position_exceeded": tracking_error_m > self.args.max_target_m,
                 "rotation_exceeded": bool(tracking_error_rad > rotation_limit),
                 "current_tcp": current, "target_tcp": target,
                 "current_relative_flange": current_relative, "desired_relative_flange": relative_goal})
        for pose in ((current,) if incremental_target else (current, target)):
            travel = float(np.linalg.norm(pose[:3, 3] - self.initial_tcp[:3, 3]))
            if travel > self.args.max_travel:
                raise ResearchFault("workspace",
                                    f"TCP outside start-position travel envelope ({travel * 1000:.1f}/"
                                    f"{self.args.max_travel * 1000:.1f}mm)",
                                    {"travel_m": travel, "limit_m": self.args.max_travel,
                                     "origin_tcp": self.initial_tcp, "checked_tcp": pose,
                                     "frame": follower + "_controller_base"})
            if incremental_target and not self.guard.pose_allowed(follower, pose):
                raise ResearchFault("workspace", "controller TCP outside workspace")
        valid, reason, clearance = self.guard.check(bases, {name: states[name].joints for name in ROBOTS})
        if not valid:
            raise ResearchFault("clearance", reason)
        # A deadband reduces stationary sensor jitter. Bound the requested
        # feedback before assessing the whole command. Bounded following may
        # ramp leader+feedback, while the base component remains mandatory.
        damped = self.bounded_leader and self.follow_damping_time > 0. and not calibration_braking
        relative_tcp_velocity = measured_twist - base_comp - leader_comp - task_twist * self.previous_task_scale
        braking_linear = braking_angular = None
        if damped:
            # speedL uses one scalar acceleration for both Cartesian groups.
            # Bound it for arbitrary simultaneous unit directions through J^-1;
            # reserve half for the other correction/leader response. This is a
            # local braking budget, with final feasibility still governed below.
            try:
                inverse_jacobian = np.linalg.solve(jacobians[follower], np.eye(6))
            except np.linalg.LinAlgError as exc:
                raise MotionInfeasible("cannot solve feedback braking Jacobian", "singularity") from exc
            joint_response = (np.linalg.norm(inverse_jacobian[:, :3], axis=1)
                              + np.linalg.norm(inverse_jacobian[:, 3:], axis=1))
            ramp_budget = min(self.args.max_accel, self.args.max_angular_accel,
                              float(np.min(self.limits.joint_accel / np.maximum(joint_response, 1e-12))))
            lever_length = np.linalg.norm(current[:3, 3] - flanges[follower][:3, 3])
            braking_linear = .5 * ramp_budget / (1. + lever_length)
            braking_angular = .5 * ramp_budget / max(1., lever_length)
        if self.follow_law == "pose":
            # One Cartesian pose-error demand, expressed at the measured TCP
            # in follower-base axes. No leader, visual-base or path velocity
            # is subtracted from damping or added to the resulting command.
            pose_dp, pose_dr = pose_delta(current, pose_command_target)
            effective_dp = outside_tolerance(pose_dp, self.args.position_deadband)
            effective_dr = outside_tolerance(pose_dr, np.deg2rad(self.args.rotation_deadband_deg))
            desired_feedback = np.r_[
                norm_limit(effective_dp * self.args.kp_linear, self.args.feedback_speed),
                norm_limit(effective_dr * self.args.kp_angular, self.args.feedback_angular_speed)]
            if damped:
                desired_feedback = np.r_[
                    damped_feedback_velocity(
                        effective_dp, measured_twist[:3], gain=self.args.kp_linear,
                        speed_limit=self.args.feedback_speed, deceleration=braking_linear,
                        reaction_time=self.follow_damping_time),
                    damped_feedback_velocity(
                        effective_dr, measured_twist[3:], gain=self.args.kp_angular,
                        speed_limit=self.args.feedback_angular_speed, deceleration=braking_angular,
                        reaction_time=self.follow_damping_time)]
        elif self.approximate_targets:
            if self.bounded_leader:
                follow_dp, follow_dr = pose_delta(current_relative, tracking_relative)
                effective_dp = outside_tolerance(follow_dp, self.args.position_deadband)
                effective_dr = outside_tolerance(follow_dr, np.deg2rad(self.args.rotation_deadband_deg))
            else:
                effective_dp = outside_tolerance(dp_rel, self.target_tolerance_m)
                effective_dr = outside_tolerance(dr_rel, self.target_tolerance_rad)
            relative_to_base = bases[follower][:3, :3].T @ (bases[leader] @ flanges[leader])[:3, :3]
            flange_linear = norm_limit(relative_to_base @ effective_dp * self.args.kp_linear,
                                       self.args.feedback_speed)
            angular = norm_limit(relative_to_base @ effective_dr * self.args.kp_angular,
                                 self.args.feedback_angular_speed)
            if damped:
                lever = current[:3, 3] - flanges[follower][:3, 3]
                relative_flange_velocity = (relative_tcp_velocity[:3]
                                            - np.cross(relative_tcp_velocity[3:], lever))
                flange_linear = damped_feedback_velocity(
                    relative_to_base @ effective_dp, relative_flange_velocity,
                    gain=self.args.kp_linear, speed_limit=self.args.feedback_speed,
                    deceleration=braking_linear, reaction_time=self.follow_damping_time)
                angular = damped_feedback_velocity(
                    relative_to_base @ effective_dr, relative_tcp_velocity[3:],
                    gain=self.args.kp_angular, speed_limit=self.args.feedback_angular_speed,
                    deceleration=braking_angular, reaction_time=self.follow_damping_time)
            tcp_linear = flange_linear + np.cross(angular, current[:3, 3] - flanges[follower][:3, 3])
            desired_feedback = np.r_[tcp_linear, angular]
            # Preserve the flange/TCP lever-arm relation when a long tool makes
            # the rotational correction exceed the TCP feedback speed budget.
            linear_norm = np.linalg.norm(tcp_linear)
            if linear_norm > self.args.feedback_speed:
                desired_feedback *= self.args.feedback_speed / linear_norm
        else:
            effective_dp, effective_dr = dp_rel.copy(), dr_rel.copy()
            desired_feedback = np.concatenate((
                norm_limit(dp * self.args.kp_linear, self.args.feedback_speed) if np.linalg.norm(dp) > self.args.position_deadband else np.zeros(3),
                norm_limit(dr * self.args.kp_angular, self.args.feedback_angular_speed) if np.linalg.norm(dr) > np.deg2rad(self.args.rotation_deadband_deg) else np.zeros(3)))
            if damped:
                desired_feedback = np.r_[
                    damped_feedback_velocity(
                        outside_tolerance(dp, self.args.position_deadband), relative_tcp_velocity[:3],
                        gain=self.args.kp_linear, speed_limit=self.args.feedback_speed,
                        deceleration=braking_linear, reaction_time=self.follow_damping_time),
                    damped_feedback_velocity(
                        outside_tolerance(dr, np.deg2rad(self.args.rotation_deadband_deg)), relative_tcp_velocity[3:],
                        gain=self.args.kp_angular, speed_limit=self.args.feedback_angular_speed,
                        deceleration=braking_angular, reaction_time=self.follow_damping_time)]
        if damped:
            # The governor bounds the complete command against measured and
            # transmitted motion. A second feedback-only slew delayed braking.
            feedback = desired_feedback
        else:
            feedback = self.previous_feedback.copy()
            feedback[:3] += norm_limit(desired_feedback[:3] - feedback[:3], self.args.max_accel * dt * .5)
            feedback[3:] += norm_limit(desired_feedback[3:] - feedback[3:], self.args.max_angular_accel * dt * .5)
        separated = self.args.method == "separated"
        compensation = base_comp + leader_comp if separated else np.zeros(6)
        mandatory_compensation = base_comp if self.bounded_leader else compensation
        if self.relative_kalman:
            # Estimated table motion is an attenuable requested component, not
            # a mandatory external-motion obligation. The total still passes
            # exactly the same measured/command TCP and joint motion checks.
            mandatory_compensation = np.zeros(6)
        work_twist = task_twist if separated else np.zeros(6)
        rho_cap = 1.
        if separated:
            rho_cap = min(1., max(0., (self.args.max_target_m - error_m) / (self.args.max_target_m - self.args.slow_error_m)),
                          max(0., (np.deg2rad(self.args.max_target_deg) - error_rad) /
                              np.deg2rad(self.args.max_target_deg - self.args.slow_error_deg)))
        if self.task_paused or self.invalid_since is not None or self.progress >= 1.:
            rho_cap = 0.
        if self._gap_approach_temporary_hold():
            # A transient visual rejection already freezes the clock above.
            # Keep it frozen while a fresh consensus is blended in as well;
            # only a successfully sent command can advance it afterwards.
            rho_cap = 0.
        if calibration_braking:
            # Decelerate the physical robot, independent of the unavailable
            # visual target, leader feedforward or previously requested base
            # compensation. Complete command/encoder bounds still apply.
            feedback = np.zeros(6)
            base_comp, leader_comp, task_twist = (np.zeros(6) for _ in range(3))
            compensation, mandatory_compensation, work_twist = (np.zeros(6) for _ in range(3))
            rho_cap = 0.
        next_targets, next_nominal_targets, next_relative_commands = {}, {}, {}
        geometry_failure = None
        geometry_details = {}

        def future_base_poses(duration):
            if self.relative_kalman:
                G = inverse(bases[follower]) @ bases[leader]
                velocity = np.zeros(6) if calibration_braking else relative_base["effective_origin"]
                information = snapshot.relative_filter
                remaining = max(0., information["config"]["prediction_horizon"] - information["capture_age"])
                return {follower: bases[follower].copy(),
                        leader: bases[follower] @ advance_origin_pose(G, velocity, min(duration, remaining))}
            return {name: advance_origin_pose(bases[name], snapshot.base_twists[name], duration) for name in ROBOTS}

        # Velocity research mode requires independent feasibility of external
        # base compensation. Pose/bounded following has no such mandatory
        # motion: test the actual candidate's sweep below, rather than a
        # hypothetical stationary follower with the leader still moving.
        if not calibration_braking and not (self.base_motion_mode in ("pose", "relative_kalman") and self.bounded_leader):
            try:
                compensation_qdot = np.linalg.solve(jacobians[follower], mandatory_compensation)
            except np.linalg.LinAlgError as exc:
                raise ResearchFault("singularity", "cannot solve mandatory compensation joints") from exc
            for fraction in (.5, 1.):
                trial_bases = future_base_poses(dt * fraction)
                trial_q = {leader: states[leader].joints + qdots[leader] * dt * fraction,
                           follower: states[follower].joints + compensation_qdot * dt * fraction}
                allowed, why, _ = self.guard.check(trial_bases, trial_q)
                if not allowed:
                    raise ResearchFault("compensation_infeasible", "mandatory compensation: " + why)

        def candidate_allowed(rho, twist, qdot):
            nonlocal geometry_failure, geometry_details
            geometry_details = {}
            # The governor constrains transitions from measured motion; also
            # preserve the slew limit between successive transmitted commands.
            if (np.linalg.norm(twist[:3] - self.previous_twist[:3]) > self.args.max_accel * dt + 1e-12
                    or np.linalg.norm(twist[3:] - self.previous_twist[3:]) > self.args.max_angular_accel * dt + 1e-12
                    or np.any(np.abs(qdot - self.previous_qdot) > self.args.max_joint_accel * dt + 1e-12)):
                geometry_failure = ("transition_infeasible", "successive command acceleration limit")
                return False
            next_progress = min(1., self.progress + rho * dt / self.path.duration) if self.path.kind != "hold" else self.progress
            desired = self.path.pose(next_progress)
            if not calibration_braking and np.linalg.norm(desired[:3, 3]) < self.args.min_gap:
                geometry_failure = ("clearance_infeasible", "next relative flange gap below min_gap")
                return False
            future_bases = future_base_poses(dt)
            future_leader = advance_origin_pose(flanges[leader], leader_jacobian @ qdots[leader], dt)
            next_target = inverse(future_bases[follower]) @ future_bases[leader] @ future_leader @ desired @ self.offsets[follower]
            nominal_next_target = next_target.copy()
            if not incremental_target and (not self.guard.pose_allowed(follower, next_target)
                    or np.linalg.norm(next_target[:3, 3] - self.initial_tcp[:3, 3]) > self.args.max_travel):
                geometry_failure = ("workspace_infeasible", "next target outside TCP workspace/travel envelope")
                return False
            if not incremental_target:
                goal_q = self.models[follower].inverse(next_target, states[follower].joints,
                                                       self.guard.lower[follower], self.guard.upper[follower])
                if goal_q is None:
                    geometry_failure = ("reachability_infeasible", "next target has no nominal local IK solution within joint limits")
                    return False
                if not self.guard.check(future_bases, {leader: states[leader].joints + qdots[leader] * dt, follower: goal_q})[0]:
                    geometry_failure = ("clearance_infeasible", "next target violates geometric clearance")
                    return False
            # With task tolerance, feasibility concerns the command actually
            # being issued, not an exact IK solution of the noisy nominal goal.
            # The joint-space sweep remains bounded by all motion/geometry
            # limits and the original (larger) tracking-error envelope.
            for fraction in (.5, 1.):
                trial_bases = future_base_poses(dt * fraction)
                trial_q = {leader: states[leader].joints + qdots[leader] * dt * fraction,
                           follower: states[follower].joints + qdot * dt * fraction}
                if not self.guard.check(trial_bases, trial_q)[0]:
                    geometry_failure = ("clearance_infeasible", "next command sweep violates geometric clearance")
                    return False
                if incremental_target or bounded_command:
                    trial_tcp, trial_flange = {}, {}
                    for name in ROBOTS:
                        predicted, predicted_jacobian = self.models[name].pose_and_jacobian(trial_q[name])
                        trial_tcp[name] = predicted @ fk_corrections[name]
                        trial_flange[name] = flange_from_tcp(trial_tcp[name], self.offsets[name])
                        if name == follower:
                            predicted_jacobian = predicted_jacobian.copy()
                            predicted_jacobian[:3] += np.cross(
                                predicted_jacobian[3:].T, trial_tcp[name][:3, 3] - predicted[:3, 3]).T
                            if np.linalg.svd(predicted_jacobian, compute_uv=False)[-1] < self.args.min_singular_value:
                                geometry_failure = ("singularity", "next command approaches a joint singularity")
                                return False
                            if bounded_command:
                                # Follow uses speedL: the requested TCP twist
                                # stays constant while its required joint
                                # velocity changes with the local Jacobian.
                                # Check that future inverse mapping as well as
                                # the constant-qdot sweep approximation below.
                                try:
                                    future_qdot = np.linalg.solve(predicted_jacobian, twist)
                                except np.linalg.LinAlgError:
                                    geometry_failure = ("singularity", "cannot solve predicted following joint velocity")
                                    return False
                                if (not np.isfinite(future_qdot).all()
                                        or np.any(np.abs(future_qdot) > self.args.max_joint_speed + 1e-9)
                                        or np.any(np.abs(future_qdot - qdots[follower]) > self.args.max_joint_accel * dt + 1e-9)
                                        or np.any(np.abs(future_qdot - self.previous_qdot) > self.args.max_joint_accel * dt + 1e-9)):
                                    geometry_failure = ("transition_infeasible", "predicted following joint speed/acceleration limit")
                                    geometry_details = {
                                        "predicted_qdot": future_qdot.copy(), "candidate_qdot": qdot.copy(),
                                        "prediction_fraction": fraction,
                                        "speed_exceeded": bool(np.any(np.abs(future_qdot) > self.args.max_joint_speed + 1e-9)),
                                        "measured_acceleration_exceeded": bool(np.any(np.abs(future_qdot - qdots[follower]) > self.args.max_joint_accel * dt + 1e-9)),
                                        "command_acceleration_exceeded": bool(np.any(np.abs(future_qdot - self.previous_qdot) > self.args.max_joint_accel * dt + 1e-9)),
                                    }
                                    return False
                                predicted_twist = predicted_jacobian @ qdot
                                for part, speed_limit, acceleration_limit in (
                                        (slice(0, 3), self.limits.linear_speed, self.args.max_accel),
                                        (slice(3, 6), self.args.max_angular_speed, self.args.max_angular_accel)):
                                    if (np.linalg.norm(predicted_twist[part]) > speed_limit + 1e-9
                                            or np.linalg.norm(predicted_twist[part] - measured_twist[part]) > acceleration_limit * dt + 1e-9
                                            or np.linalg.norm(predicted_twist[part] - self.previous_twist[part]) > acceleration_limit * dt + 1e-9):
                                        geometry_failure = ("transition_infeasible", "predicted following TCP speed/acceleration limit")
                                        return False
                    executed = trial_tcp[follower]
                    if (not self.guard.pose_allowed(follower, executed)
                            or np.linalg.norm(executed[:3, 3] - self.initial_tcp[:3, 3]) > self.args.max_travel):
                        geometry_failure = ("workspace_infeasible", "next command outside TCP workspace/travel envelope")
                        return False
                    trial_relative = inverse(trial_bases[leader] @ trial_flange[leader]) @ trial_bases[follower] @ trial_flange[follower]
                    if np.linalg.norm(trial_relative[:3, 3]) < self.args.min_gap:
                        geometry_failure = ("clearance_infeasible", "next command flange gap below min_gap")
                        return False
                    trial_progress = self.progress + fraction * (next_progress - self.progress)
                    trial_dp, trial_dr = pose_delta(trial_relative, self.path.pose(trial_progress))
                    if not calibration_braking and (np.linalg.norm(trial_dp) > self.args.max_target_m or np.linalg.norm(trial_dr) > rotation_limit):
                        geometry_failure = ("tracking_error", "next command exceeds allowed relative tracking error")
                        return False
                    if fraction == 1. and incremental_target:
                        next_target = executed
                        next_relative_commands[float(rho)] = trial_relative
            next_targets[float(rho)] = next_target
            next_nominal_targets[float(rho)] = nominal_next_target
            return True

        failure = None
        decision = None
        for feedback_scale in (1., .75, .5, .25, 0.):
            geometry_failure, geometry_details = None, {}
            try:
                if bounded_command:
                    soft_base = base_comp if self.relative_kalman else np.zeros(6)
                    decision = govern_follow(
                        np.zeros(6) if self.relative_kalman else base_comp,
                        leader_comp + soft_base + feedback * feedback_scale, work_twist, jacobians[follower],
                        measured_twist, qdots[follower], self.previous_twist, self.previous_qdot, dt, self.limits,
                        previous_base=np.zeros(6) if calibration_braking else self.previous_base_compensation,
                        previous_base_qdot=np.zeros(6) if calibration_braking else self.previous_base_compensation_qdot,
                        rho_cap=rho_cap, extra_feasible=candidate_allowed,
                        moving_reference=(self.follow_law == "pose" and self.path.kind != "hold"
                                          and not calibration_braking),
                        extra_feasible_depends_on_rho=self.path.kind != "hold")
                else:
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
            details = {
                "dt": dt, "measured_qdot": qdots[follower], "previous_command_qdot": self.previous_qdot,
                "measured_tcp_speed": measured_twist, "previous_command_twist": self.previous_twist,
                "joint_speed_limit": self.args.max_joint_speed, "joint_acceleration_limit": self.args.max_joint_accel,
                **geometry_details,
            }
            if geometry_failure is not None:
                raise ResearchFault(*geometry_failure, details=details)
            if bounded_command and failure is not None:
                raise ResearchFault(failure.category, str(failure), details)
            raise failure
        rho = decision.task_scale
        next_progress = min(1., self.progress + rho * dt / self.path.duration) if self.path.kind != "hold" else self.progress
        finish_m = max(self.args.finish_position_m, self.target_tolerance_m)
        finish_rad = max(np.deg2rad(self.args.finish_rotation_deg), self.target_tolerance_rad)
        command_target = current.copy() if incremental_target else target
        record = {
            "kind": "control", "at": now, "active": active, "method": self.args.method,
            "follow_law": self.follow_law,
            "calibration_braking": bool(calibration_braking),
            "base_motion_mode": self.base_motion_mode,
            "base_velocity_used": self.base_motion_mode in ("velocity", "relative_kalman"),
            "leader_velocity_source": "joint_encoders",
            "follower_velocity_source": "rtde",
            "follow_damping_time": self.follow_damping_time if damped else 0.,
            "feedback_relative_tcp_velocity": relative_tcp_velocity,
            "feedback_braking_linear_accel": braking_linear,
            "feedback_braking_angular_accel": braking_angular,
            "leader": leader, "follower": follower, "path": self.path.kind, "path_duration": self.path.duration,
            "progress": self.progress, "next_progress": next_progress,
            "task_scale": rho, "requested_task_scale": rho_cap, "task_paused": self.task_paused,
            "path_finished": self.path.kind != "hold" and self.progress >= 1.,
            "finished": self.finished,
            "within_finish_tolerance": error_m <= finish_m and error_rad <= finish_rad,
            "within_nominal_finish_tolerance": error_m <= self.args.finish_position_m and error_rad <= np.deg2rad(self.args.finish_rotation_deg),
            "finish_position_tolerance_m": finish_m, "finish_rotation_tolerance_rad": finish_rad,
            "approximate_targets": self.approximate_targets,
            "target_tolerance_m": self.target_tolerance_m, "target_tolerance_rad": self.target_tolerance_rad,
            "within_target_tolerance": error_m <= self.target_tolerance_m and error_rad <= self.target_tolerance_rad,
            "feedback_residual_m": float(np.linalg.norm(effective_dp)),
            "feedback_residual_rad": float(np.linalg.norm(effective_dr)),
            "control_dt": dt, "settle_elapsed": self.settle_elapsed,
            "state_at": snapshot.at, "measurement_age": snapshot.measurement_age,
            "measurement_times": snapshot.measurement_times, "received_times": snapshot.received_times,
            "prediction_dt": snapshot.prediction_dt, "relative_base_pose": snapshot.relative_pose,
            "relative_base_twist": snapshot.relative_twist, "bases": bases,
            "base_twists": snapshot.base_twists, "estimated_relative_pose": current_relative,
            "desired_relative_pose": relative_goal, "relative_error_m": error_m,
            "relative_error_rad": error_rad, "target_tcp": target,
            "next_target_tcp": next_nominal_targets[float(rho)],
            "commanded_target_tcp": command_target, "next_commanded_target_tcp": next_targets[float(rho)],
            "native_safety_target_mode": "current_and_predicted_motion" if incremental_target else "nominal_task_goal",
            "commanded_relative_pose": current_relative.copy() if incremental_target else relative_goal.copy(),
            "next_commanded_relative_pose": next_relative_commands.get(float(rho), self.path.pose(next_progress)),
            "twist": decision.twist,
            "joint_command": decision.qdot, "base_compensation": base_comp,
            "leader_compensation": leader_comp, "requested_compensation": compensation,
            "applied_compensation": mandatory_compensation,
            "base_compensation_nominal": nominal_base_comp, "leader_compensation_nominal": nominal_leader_comp,
            "task_twist_nominal": nominal_task_twist, "task_twist_at_command": task_twist, "feedback": feedback,
            "feedback_scale": feedback_scale, "decision": decision.reason,
            "leader_response": "bounded" if self.bounded_leader else "strict",
            "follow_max_speed": self.limits.linear_speed,
            "tracking_relative_target": tracking_relative,
            "tracking_offset": tracking_offset,
            "tracking_limited": bool(decision.components.get("tracking_limited", False)),
            "tracking_braking": bool(decision.components.get("tracking_braking", False)),
            "tracking_scale": decision.components.get("tracking_scale", 1.),
            "tracking_speed_scale": decision.components.get("tracking_speed_scale", 1.),
            "tracking_endpoint": decision.components.get("tracking_endpoint", leader_comp + feedback),
            "tracking_applied": decision.components.get("tracking_applied", leader_comp + feedback),
            "tracking_requested": leader_comp + feedback,
            "feedback_is_requested": self.bounded_leader,
            "capsule_clearance_m": clearance, "last_rejection": self.last_rejection,
            "robots": {name: {"q": states[name].joints, "qd": qdots[name], "tcp": states[name].tcp,
                              "tcp_speed": states[name].speed, "robot_timestamp": states[name].robot_timestamp}
                       for name in ROBOTS}}
        if self.follow_law == "pose":
            record.update(pose_command_target_tcp=pose_command_target.copy(),
                          pose_command_error_m=float(np.linalg.norm(pose_dp)),
                          pose_command_error_rad=float(np.linalg.norm(pose_dr)),
                          leader_velocity_command_used=False,
                          feedback_error_origin="tcp", feedback_error_frame="follower_base",
                          feedback_velocity_source="rtde_actual_tcp")
        if self.relative_kalman:
            record["relative_kalman"] = copy.deepcopy(snapshot.relative_filter)
            record["relative_base_compensation"] = relative_base
            record["relative_base_compensation"]["held_for_braking"] = bool(calibration_braking)
            record["requested_twist"] = base_comp + leader_comp + feedback + rho_cap * work_twist
            record["tracking_requested"] = base_comp + leader_comp + feedback
            record["relative_base_target"] = snapshot.relative_filter["G_control"] @ flanges[leader] @ tracking_relative
            record["tracking_target_tcp"] = record["relative_base_target"] @ self.offsets[follower]
        if active and self.pose_smoother is not None:
            record["base_pose_smoothing"] = self.pose_smoother.diagnostics()
        if bounded_command:
            record["speedl_acceleration"] = bounded_speedl_acceleration(
                jacobians[follower], measured_twist, decision.twist, dt, self.limits)
        return Proposal({follower: bases[follower] @ target @ inverse(self.offsets[follower])},
                        {follower: command_target}, {follower: decision.twist}, record, next_progress,
                        decision, feedback, mandatory_compensation, {follower: next_targets[float(rho)]},
                        tracking_offset if active and not calibration_braking else None)

    def commit(self, proposal):
        self.progress = proposal.next_progress
        self.gap_approach.commit(self.progress)
        proposal.record["gap_approach"] = self.gap_approach.info(self.progress)
        proposal.record["gap_approach"]["temporary_hold"] = self._gap_approach_temporary_hold()
        self.previous_twist = proposal.decision.twist.copy()
        self.previous_qdot = proposal.decision.qdot.copy()
        self.previous_compensation = proposal.compensation.copy()
        self.previous_compensation_qdot = np.asarray(proposal.decision.components["compensation_qdot"]).copy()
        self.previous_feedback = proposal.feedback.copy()
        self.previous_task_scale = proposal.decision.task_scale
        if self.relative_kalman:
            self.previous_relative_base_flange = (
                np.zeros(6) if proposal.record.get("calibration_braking") else
                proposal.record["relative_base_compensation"]["effective_flange"].copy())
        if self.bounded_leader:
            self.previous_base_compensation = proposal.compensation.copy()
            self.previous_base_compensation_qdot = np.asarray(proposal.decision.components["compensation_qdot"]).copy()
            if self.tracking_offset is None and proposal.tracking_offset is not None:
                self.tracking_offset = proposal.tracking_offset.copy()
        if proposal.record.get("calibration_braking"):
            return  # Observation does not complete or reset a relative task.
        if proposal.record["path_finished"] and proposal.record["within_finish_tolerance"]:
            self.settle_elapsed += proposal.record["control_dt"]
        else:
            self.settle_elapsed = 0.
        self.finished = self.finished or self.settle_elapsed >= self.args.finish_dwell
