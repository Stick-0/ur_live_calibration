"""Pure calibration acceptance and Cartesian velocity limits for dual-arm control.

This module neither imports robot drivers nor opens any hardware connection.
All timestamps must use the same monotonic host clock; a frame's timestamp is
its capture time, never the time at which inference finished.
"""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, pose_delta, rotation_distance


ROBOTS = ("ur5e", "ur10e")


def stationary_velocity_limits(args=None, *, leader=False):
    """Joint, TCP linear and angular rest limits; only a fixed leader may relax.

    Saved configurations without the new options retain the original limits.
    Commanded arms must still settle to the original thresholds.
    """
    names = ("align_leader_max_joint_speed", "align_leader_max_speed", "align_leader_max_angular_speed")
    defaults = (.005, .002, .01)
    values = tuple(_finite_scalar(getattr(args, key, default) if leader else default, key)
                   for key, default in zip(names, defaults))
    if min(values) <= 0:
        raise ControlFault("stationary velocity limits must be positive")
    return values


class ControlFault(RuntimeError):
    """A violated control precondition; the caller must stop motion."""


class CalibrationUnavailable(ControlFault):
    """Valid sensor input is temporarily insufficient for motion planning."""


class PlanningUnavailable(ControlFault):
    """A valid model/target has no presently acceptable motion plan."""


@dataclass(frozen=True)
class CalibrationFrame:
    sequence: int
    captured_at: float
    poses: Dict[str, np.ndarray]
    raw_mean_px: Dict[str, float]
    raw_max_px: Dict[str, float]
    valid_counts: Dict[str, int]
    reason: str = ""
    # Read-only sensor input snapshots for logging; never used for acceptance.
    pose_diagnostics: Optional[dict] = None


def _finite_scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ControlFault(f"{name} must be a finite number")
    return result


def _transform(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault(f"{name} must be a finite rigid 4x4 transform") from exc
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ControlFault(f"{name} must be a finite rigid 4x4 transform")
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ControlFault(f"{name} must be a rigid transform with a proper rotation")
    return result.copy()


class CalibrationTracker:
    """Accept complete pairs and smooth only individually validated samples.

    Invalid input clears readiness but preserves the last accepted transforms
    and capture time for diagnostics. The caller should latch any fault during
    motion; this class deliberately has no robot restart behavior.
    """

    def __init__(self, max_age=0.5, stable_frames=5, alpha=0.2,
                 max_jump_m=0.015, max_jump_rad=np.deg2rad(2.0),
                 max_mean_px=5.0, max_pixel_px=8.0, min_points=5,
                 check_reprojection=True, check_individual_jumps=True):
        for name, value in (("max_age", max_age), ("alpha", alpha),
                            ("max_jump_m", max_jump_m), ("max_jump_rad", max_jump_rad),
                            ("max_mean_px", max_mean_px), ("max_pixel_px", max_pixel_px)):
            number = _finite_scalar(value, name)
            if number <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, number)
        if self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if not isinstance(stable_frames, Integral) or isinstance(stable_frames, bool) or stable_frames < 1:
            raise ValueError("stable_frames must be a positive integer")
        if not isinstance(min_points, Integral) or isinstance(min_points, bool) or not 4 <= min_points <= 7:
            raise ValueError("min_points must be an integer from 4 to 7")
        if not isinstance(check_reprojection, bool):
            raise ValueError("check_reprojection must be a boolean")
        self.check_reprojection = check_reprojection
        if not isinstance(check_individual_jumps, bool):
            raise ValueError("check_individual_jumps must be a boolean")
        self.check_individual_jumps = check_individual_jumps
        self.stable_frames = int(stable_frames)
        self.min_points = int(min_points)
        self.stable_count = 0
        self.last_capture: Optional[float] = None
        self._last_sequence: Optional[int] = None
        self._raw_poses: Dict[str, np.ndarray] = {}
        self._poses: Dict[str, np.ndarray] = {}
        self.last_rejection = ""

    @property
    def poses(self):
        """Return copies, so callers cannot corrupt the accepted state."""
        return {name: pose.copy() for name, pose in self._poses.items()}

    @property
    def ready(self):
        return self.stable_count >= self.stable_frames

    def reject(self, sequence, reason="calibration pair rejected"):
        """Consume an externally rejected pair without changing accepted poses."""
        self.stable_count = 0
        self.last_rejection = str(reason)
        if (isinstance(sequence, Integral) and not isinstance(sequence, bool) and sequence >= 0
                and (self._last_sequence is None or sequence > self._last_sequence)):
            self._last_sequence = int(sequence)

    def _check_age(self, captured_at, now):
        now = _finite_scalar(now, "current time")
        captured_at = _finite_scalar(captured_at, "capture time")
        age = now - captured_at
        if age < 0:
            raise ControlFault("calibration capture time is in the future")
        if age > self.max_age:
            raise CalibrationUnavailable(f"calibration is stale ({age:.3f}s > {self.max_age:.3f}s)")

    def check_fresh(self, now):
        """Require a stable pair whose original captured image is still fresh."""
        if self.last_capture is None:
            raise CalibrationUnavailable(self.last_rejection or "no accepted calibration pair")
        try:
            self._check_age(self.last_capture, now)
        except ControlFault as exc:
            self.stable_count = 0
            if self.last_rejection:
                raise type(exc)(str(exc) + "; last rejected pair: " + self.last_rejection) from exc
            raise
        if not self.ready:
            raise CalibrationUnavailable(self.last_rejection or
                               f"calibration is not stable ({self.stable_count}/{self.stable_frames} frames)")

    def _check_jump(self, previous, current, label):
        translation = np.linalg.norm(current[:3, 3] - previous[:3, 3])
        angle = rotation_distance(previous, current)
        if translation > self.max_jump_m or angle > self.max_jump_rad:
            raise CalibrationUnavailable(f"{label} calibration jump: {translation:.4f}m, {np.rad2deg(angle):.2f}deg")

    def update(self, frame, now):
        """Return True for a new accepted pair, False for a fresh duplicate.

        Sequence numbers must increase even across rejected observations. A
        repeated sample cannot increase stability or refresh its capture time.
        """
        try:
            if not isinstance(frame, CalibrationFrame):
                raise ControlFault("expected a CalibrationFrame")
            sequence = frame.sequence
            if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
                raise ControlFault("calibration sequence must be a nonnegative integer")
            if self._last_sequence is not None:
                if sequence < self._last_sequence:
                    raise ControlFault("calibration sequence moved backwards")
                if sequence == self._last_sequence:
                    if self.last_capture is None:
                        raise CalibrationUnavailable(self.last_rejection or "no accepted calibration pair")
                    self._check_age(self.last_capture, now)
                    return False
            self._last_sequence = int(sequence)
            captured_at = _finite_scalar(frame.captured_at, "capture time")
            self._check_age(captured_at, now)
            if self.last_capture is not None and captured_at <= self.last_capture:
                raise ControlFault("calibration capture time did not increase")
            if self.last_capture is not None and captured_at - self.last_capture > self.max_age:
                self.stable_count = 0
            if frame.reason:
                raise CalibrationUnavailable(f"calibration rejected: {frame.reason}")
            for field in (frame.poses, frame.raw_mean_px, frame.raw_max_px, frame.valid_counts):
                if not isinstance(field, dict) or set(field) != set(ROBOTS):
                    raise ControlFault("calibration requires both ur5e and ur10e")
            candidate = {}
            for name in ROBOTS:
                count = frame.valid_counts[name]
                if (not isinstance(count, Integral) or isinstance(count, bool)
                        or not self.min_points <= count <= 7):
                    fault = (CalibrationUnavailable if isinstance(count, Integral)
                             and not isinstance(count, bool) and 0 <= count < self.min_points else ControlFault)
                    raise fault(f"{name} has too few valid keypoints or an invalid count "
                                       f"(points={count!r}, required={self.min_points}..7)")
                mean = _finite_scalar(frame.raw_mean_px[name], f"{name} mean reprojection error")
                maximum = _finite_scalar(frame.raw_max_px[name], f"{name} maximum reprojection error")
                if mean < 0 or maximum < mean:
                    raise ControlFault(f"{name} raw reprojection statistics are invalid "
                                       f"(mean={mean:.2f}px; max={maximum:.2f}px; expected 0 <= mean <= max)")
                if self.check_reprojection and (mean > self.max_mean_px or maximum > self.max_pixel_px):
                    raise CalibrationUnavailable(f"{name} raw reprojection error exceeds limits "
                                       f"(mean={mean:.2f}px limit={self.max_mean_px:.2f}px; "
                                       f"max={maximum:.2f}px limit={self.max_pixel_px:.2f}px; "
                                       f"points={count}/7 min={self.min_points})")
                candidate[name] = _transform(frame.poses[name], f"{name} calibration")
                if self._raw_poses and self.check_individual_jumps:
                    self._check_jump(self._raw_poses[name], candidate[name], name)
            if self._raw_poses:
                old_relative = inverse(self._raw_poses[ROBOTS[0]]) @ self._raw_poses[ROBOTS[1]]
                new_relative = inverse(candidate[ROBOTS[0]]) @ candidate[ROBOTS[1]]
                self._check_jump(old_relative, new_relative, "relative base")
            filtered = {}
            for name, pose in candidate.items():
                result = pose.copy()
                if self._poses:
                    previous = self._poses[name]
                    result[:3, 3] = previous[:3, 3] + self.alpha * (pose[:3, 3] - previous[:3, 3])
                    relative_rotation = previous[:3, :3].T @ pose[:3, :3]
                    increment = Rotation.from_matrix(relative_rotation).as_rotvec() * self.alpha
                    result[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(increment).as_matrix()
                filtered[name] = result
            self._raw_poses = candidate
            self._poses = filtered
            self.last_capture = captured_at
            self.stable_count += 1
            self.last_rejection = ""
            return True
        except ControlFault as exc:
            self.stable_count = 0
            self.last_rejection = str(exc)
            raise


class PreviewCalibrationRecovery:
    """Build an independent replacement for a stale preview calibration.

    Callers must restrict this helper to unarmed preview. It never changes the
    supplied tracker or authorizes motion. The replacement requires a complete
    new cluster, bounded against both consecutive samples and its first sample.
    A caller with an additional estimator must validate/replay ``samples`` and
    commit both states together, then call ``reset``.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._candidate = None
        self._anchor = None
        self._samples = []
        self._last_sequence = None
        self._last_capture = None
        self._last_now = None
        self._configuration = None
        self.reason = ""

    @property
    def count(self):
        return len(self._samples)

    @property
    def samples(self):
        """Independent (frame, received_at) copies for atomic estimator replay."""
        return deepcopy(self._samples)

    @staticmethod
    def _options(template):
        # Copy configuration explicitly; never copy the old rejected baseline.
        return {name: getattr(template, name) for name in (
            "max_age", "stable_frames", "alpha", "max_jump_m", "max_jump_rad",
            "max_mean_px", "max_pixel_px", "min_points", "check_reprojection", "check_individual_jumps")}

    def _clear_cluster(self, reason):
        # Keep observation watermarks: a duplicate of a rejected observation
        # cannot be repaired or recounted by changing its payload.
        self._candidate = None
        self._anchor = None
        self._samples = []
        self.reason = str(reason)

    def offer(self, frame, now, template_tracker):
        """Return an independent ready tracker, or None while awaiting a cluster.

        Quality/time errors discard the cluster. A valid new sample outside
        its spatial envelope, or following a capture gap, starts a new cluster
        at one. Duplicate observations never count or refresh capture time.
        """
        try:
            now = _finite_scalar(now, "current time")
            if now < 0:
                raise ControlFault("current time must be nonnegative")
            if self._last_now is not None and now < self._last_now:
                raise ControlFault("preview calibration time moved backwards")
            self._last_now = now
            if template_tracker.last_capture is None:
                self._clear_cluster("preview recovery requires an accepted baseline")
                return None
            baseline_capture = _finite_scalar(template_tracker.last_capture, "accepted capture time")
            if now - baseline_capture <= template_tracker.max_age:
                self._clear_cluster("accepted calibration is not stale")
                return None
            options = self._options(template_tracker)
            if self._configuration is not None and options != self._configuration:
                self._clear_cluster("preview calibration configuration changed")
            self._configuration = options
            if not isinstance(frame, CalibrationFrame):
                raise ControlFault("expected a CalibrationFrame")
            sequence = frame.sequence
            if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
                raise ControlFault("calibration sequence must be a nonnegative integer")
            if (template_tracker._last_sequence is not None
                    and sequence < template_tracker._last_sequence):
                raise ControlFault("calibration sequence moved backwards")
            if self._last_sequence is not None:
                if sequence < self._last_sequence:
                    raise ControlFault("calibration sequence moved backwards")
                if sequence == self._last_sequence:
                    if self._candidate is not None:
                        self._candidate._check_age(self._candidate.last_capture, now)
                    return None
            self._last_sequence = int(sequence)
            captured_at = _finite_scalar(frame.captured_at, "capture time")
            template_tracker._check_age(captured_at, now)
            if captured_at <= baseline_capture or (self._last_capture is not None
                                                   and captured_at <= self._last_capture):
                raise ControlFault("calibration capture time did not increase")
            previous_capture = self._last_capture
            self._last_capture = captured_at
            # Validate all quality fields against the unchanged configured
            # limits before deciding whether this sample starts a new cluster.
            first = CalibrationTracker(**options)
            first.update(frame, now)
            restart_reason = ""
            if previous_capture is not None and captured_at - previous_capture > first.max_age:
                restart_reason = "preview calibration capture gap; collecting a new cluster"
            elif self._candidate is not None:
                try:
                    for name in ROBOTS:
                        self._candidate._check_jump(self._anchor[name], first._raw_poses[name], name)
                    anchor_relative = inverse(self._anchor[ROBOTS[0]]) @ self._anchor[ROBOTS[1]]
                    relative = inverse(first._raw_poses[ROBOTS[0]]) @ first._raw_poses[ROBOTS[1]]
                    self._candidate._check_jump(anchor_relative, relative, "relative base")
                    self._candidate.update(frame, now)
                except ControlFault as exc:
                    restart_reason = str(exc) + "; collecting a new preview cluster"
            if self._candidate is None or restart_reason:
                self._candidate = first
                self._anchor = deepcopy(first._raw_poses)
                self._samples = []
            self._samples.append((deepcopy(frame), now))
            self.reason = restart_reason or (
                f"preview calibration candidate {self.count}/{self._candidate.stable_frames}")
            if self._candidate.ready:
                self._candidate.check_fresh(now)
                return deepcopy(self._candidate)
            return None
        except ControlFault as exc:
            sequence = getattr(frame, "sequence", None)
            if (isinstance(sequence, Integral) and not isinstance(sequence, bool) and sequence >= 0
                    and (self._last_sequence is None or sequence > self._last_sequence)):
                self._last_sequence = int(sequence)
            self._clear_cluster(exc)
            return None


class ActiveCalibrationGuard:
    """Permit brief rejected measurements only after a strict, explicit Arm.

    Rejected measurements never become usable poses or refresh capture age.
    A fresh accepted pair can recover an armed session without repeating its
    startup stability count. A terminal fault disarms this guard; recovering
    later measurements alone cannot resume it.
    """

    def __init__(self, max_invalid_duration=.3):
        self.max_invalid_duration = _finite_scalar(max_invalid_duration, "max_invalid_duration")
        if self.max_invalid_duration <= 0:
            raise ValueError("max_invalid_duration must be positive")
        self.armed = False
        self.invalid_since = None
        self._last_check = None
        self._rejection_reason = ""
        self._fault_reason = ""

    @staticmethod
    def _now(value):
        now = _finite_scalar(value, "current time")
        if now < 0:
            raise ControlFault("current time must be nonnegative")
        return now

    def arm(self, tracker, now):
        """Require the unchanged startup freshness/stability checks again."""
        self.armed = False
        self.invalid_since = None
        self._rejection_reason = ""
        self._fault_reason = ""
        self._last_check = None
        now = self._now(now)
        tracker.check_fresh(now)
        self._last_check = now
        self.armed = True

    def check(self, tracker, now):
        """Return True for accepted vision, False while using its last good pair.

        Raises ControlFault after the grace or capture-age limit. This method
        does not mutate the tracker's poses, capture time or stability count.
        Callers must separately enforce robot state and motion constraints.
        """
        if not self.armed:
            suffix = ": " + self._fault_reason if self._fault_reason else ""
            raise ControlFault("active calibration guard requires explicit Arm" + suffix)
        try:
            now = self._now(now)
            if now < self._last_check:
                raise ControlFault("active calibration time moved backwards")
            self._last_check = now
            if tracker.last_capture is None:
                raise CalibrationUnavailable("no accepted calibration pair")
            try:
                tracker._check_age(tracker.last_capture, now)
            except ControlFault as exc:
                reason = tracker.last_rejection or self._rejection_reason
                raise type(exc)(str(exc) + ("; last rejected pair: " + reason if reason else "")) from exc
            # Check the old deadline before accepting a late recovery frame.
            if self.invalid_since is not None:
                elapsed = now - self.invalid_since
                if elapsed > self.max_invalid_duration:
                    raise CalibrationUnavailable(
                        f"calibration remained invalid for {elapsed:.3f}s "
                        f"(limit {self.max_invalid_duration:.3f}s): "
                        + (tracker.last_rejection or self._rejection_reason))
            if tracker.last_rejection:
                if self.invalid_since is None:
                    self.invalid_since = now
                self._rejection_reason = tracker.last_rejection
                return False
            self.invalid_since = None
            self._rejection_reason = ""
            return True
        except ControlFault as exc:
            self.armed = False
            self._fault_reason = str(exc)
            raise


def _norm_clamp(vector, limit):
    magnitude = np.linalg.norm(vector)
    return vector * (limit / magnitude) if magnitude > limit else vector


def bounded_twist(current_tcp, target_tcp, previous_twist, dt, kp_linear=1.0,
                  kp_angular=1.0, max_linear=0.02, max_angular=0.1,
                  max_accel=0.1, max_angular_accel=0.5,
                  max_target_m=0.1, max_target_rad=np.deg2rad(25.0), *,
                  position_tolerance_m=0., rotation_tolerance_rad=0., tcp_offset=None):
    """Return [vx, vy, vz, wx, wy, wz] in the robot controller base frame.

    The desired final pose is checked before speed/acceleration limiting. Both
    TCP transforms must be expressed in that same controller base frame.
    Task tolerances bound the estimated FLANGE error, not calibration accuracy.
    Feedback decreases continuously to zero at that boundary. The flange
    velocity is shifted to the actual TCP before existing speed/acceleration
    limits. Zero tolerances retain exact-pose feedback. Limits bound the
    commanded twist, not physical stopping distance.
    """
    current = _transform(current_tcp, "current TCP")
    target = _transform(target_tcp, "target TCP")
    try:
        previous = np.asarray(previous_twist, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault("previous twist must contain six finite values") from exc
    if previous.shape != (6,) or not np.isfinite(previous).all():
        raise ControlFault("previous twist must contain six finite values")
    values = {}
    for name, value in (("dt", dt), ("kp_linear", kp_linear), ("kp_angular", kp_angular),
                        ("max_linear", max_linear), ("max_angular", max_angular),
                        ("max_accel", max_accel), ("max_angular_accel", max_angular_accel),
                        ("max_target_m", max_target_m), ("max_target_rad", max_target_rad)):
        values[name] = _finite_scalar(value, name)
        if values[name] < 0 or (values[name] == 0 and not name.startswith("kp_")):
            raise ControlFault(f"{name} must be positive (gains may be zero)")
    translation_error, rotation_error = pose_delta(current, target)
    tolerances = [_finite_scalar(position_tolerance_m, "position_tolerance_m"),
                  _finite_scalar(rotation_tolerance_rad, "rotation_tolerance_rad")]
    if min(tolerances) < 0:
        raise ControlFault("task tolerances must be nonnegative")
    if any(tolerances):
        offset = np.eye(4) if tcp_offset is None else _transform(tcp_offset, "TCP offset")
        flange = current @ inverse(offset)
        goal_flange = target @ inverse(offset)
        translation_error, rotation_error = pose_delta(flange, goal_flange)
    if (np.linalg.norm(translation_error) > values["max_target_m"]
            or np.linalg.norm(rotation_error) > values["max_target_rad"]):
        frame = "flange" if any(tolerances) else "TCP"
        raise PlanningUnavailable(f"final {frame} target exceeds the allowed position/orientation error")
    if any(tolerances):
        for error, tolerance in zip((translation_error, rotation_error), tolerances):
            norm = float(np.linalg.norm(error))
            error *= max(0., 1. - tolerance / norm) if norm > 0 else 0.
        angular_velocity = rotation_error * values["kp_angular"]
        linear_velocity = (translation_error * values["kp_linear"]
                           + np.cross(angular_velocity, current[:3, 3] - flange[:3, 3]))
    else:
        linear_velocity = translation_error * values["kp_linear"]
        angular_velocity = rotation_error * values["kp_angular"]
    if (np.linalg.norm(previous[:3]) > values["max_linear"] + 1e-12
            or np.linalg.norm(previous[3:]) > values["max_angular"] + 1e-12):
        raise ControlFault("previous twist exceeds configured speed limits")
    output = np.empty(6)
    for indices, desired, speed, acceleration in (
            (slice(0, 3), linear_velocity, values["max_linear"], values["max_accel"]),
            (slice(3, 6), angular_velocity, values["max_angular"], values["max_angular_accel"])):
        desired = _norm_clamp(desired, speed)
        delta = _norm_clamp(desired - previous[indices], acceleration * values["dt"])
        # A segment between two vectors in the speed ball stays in that ball.
        output[indices] = previous[indices] + delta
    if not np.isfinite(output).all():
        raise ControlFault("computed twist is not finite")
    return output
