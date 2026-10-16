"""Pure calibration acceptance and Cartesian velocity limits for dual-arm control.

This module neither imports robot drivers nor opens any hardware connection.
All timestamps must use the same monotonic host clock; a frame's timestamp is
its capture time, never the time at which inference finished.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, pose_delta, rotation_distance


ROBOTS = ("ur5e", "ur10e")


class ControlFault(RuntimeError):
    """A violated control precondition; the caller must stop motion."""


@dataclass(frozen=True)
class CalibrationFrame:
    sequence: int
    captured_at: float
    poses: Dict[str, np.ndarray]
    raw_mean_px: Dict[str, float]
    raw_max_px: Dict[str, float]
    valid_counts: Dict[str, int]
    reason: str = ""


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
                 max_mean_px=3.0, max_pixel_px=8.0, min_points=5):
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
        self.stable_frames = int(stable_frames)
        self.min_points = int(min_points)
        self.stable_count = 0
        self.last_capture: Optional[float] = None
        self._last_sequence: Optional[int] = None
        self._raw_poses: Dict[str, np.ndarray] = {}
        self._poses: Dict[str, np.ndarray] = {}

    @property
    def poses(self):
        """Return copies, so callers cannot corrupt the accepted state."""
        return {name: pose.copy() for name, pose in self._poses.items()}

    @property
    def ready(self):
        return self.stable_count >= self.stable_frames

    def reject(self, sequence):
        """Consume an externally rejected pair without changing accepted poses."""
        self.stable_count = 0
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
            raise ControlFault(f"calibration is stale ({age:.3f}s > {self.max_age:.3f}s)")

    def check_fresh(self, now):
        """Require a stable pair whose original captured image is still fresh."""
        if self.last_capture is None:
            raise ControlFault("no accepted calibration pair")
        try:
            self._check_age(self.last_capture, now)
        except ControlFault:
            self.stable_count = 0
            raise
        if not self.ready:
            raise ControlFault(f"calibration is not stable ({self.stable_count}/{self.stable_frames} frames)")

    def _check_jump(self, previous, current, label):
        translation = np.linalg.norm(current[:3, 3] - previous[:3, 3])
        angle = rotation_distance(previous, current)
        if translation > self.max_jump_m or angle > self.max_jump_rad:
            raise ControlFault(f"{label} calibration jump: {translation:.4f}m, {np.rad2deg(angle):.2f}deg")

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
                        raise ControlFault("no accepted calibration pair")
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
                raise ControlFault(f"calibration rejected: {frame.reason}")
            for field in (frame.poses, frame.raw_mean_px, frame.raw_max_px, frame.valid_counts):
                if not isinstance(field, dict) or set(field) != set(ROBOTS):
                    raise ControlFault("calibration requires both ur5e and ur10e")
            candidate = {}
            for name in ROBOTS:
                count = frame.valid_counts[name]
                if (not isinstance(count, Integral) or isinstance(count, bool)
                        or not self.min_points <= count <= 7):
                    raise ControlFault(f"{name} has too few valid keypoints or an invalid count")
                mean = _finite_scalar(frame.raw_mean_px[name], f"{name} mean reprojection error")
                maximum = _finite_scalar(frame.raw_max_px[name], f"{name} maximum reprojection error")
                if mean < 0 or maximum < mean or mean > self.max_mean_px or maximum > self.max_pixel_px:
                    raise ControlFault(f"{name} raw reprojection error exceeds limits")
                candidate[name] = _transform(frame.poses[name], f"{name} calibration")
                if self._raw_poses:
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
            return True
        except ControlFault:
            self.stable_count = 0
            raise


def _norm_clamp(vector, limit):
    magnitude = np.linalg.norm(vector)
    return vector * (limit / magnitude) if magnitude > limit else vector


def bounded_twist(current_tcp, target_tcp, previous_twist, dt, kp_linear=1.0,
                  kp_angular=1.0, max_linear=0.02, max_angular=0.1,
                  max_accel=0.1, max_angular_accel=0.5,
                  max_target_m=0.1, max_target_rad=np.deg2rad(25.0)):
    """Return [vx, vy, vz, wx, wy, wz] in the robot controller base frame.

    The desired final pose is checked before speed/acceleration limiting. Both
    TCP transforms must be expressed in that same controller base frame.
    Limits bound the commanded twist, not physical stopping distance.
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
    if (np.linalg.norm(translation_error) > values["max_target_m"]
            or np.linalg.norm(rotation_error) > values["max_target_rad"]):
        raise ControlFault("final TCP target exceeds the allowed position/orientation error")
    if (np.linalg.norm(previous[:3]) > values["max_linear"] + 1e-12
            or np.linalg.norm(previous[3:]) > values["max_angular"] + 1e-12):
        raise ControlFault("previous twist exceeds configured speed limits")
    output = np.empty(6)
    for indices, error, gain, speed, acceleration in (
            (slice(0, 3), translation_error, values["kp_linear"], values["max_linear"], values["max_accel"]),
            (slice(3, 6), rotation_error, values["kp_angular"], values["max_angular"], values["max_angular_accel"])):
        desired = _norm_clamp(error * gain, speed)
        delta = _norm_clamp(desired - previous[indices], acceleration * values["dt"])
        # A segment between two vectors in the speed ball stays in that ball.
        output[indices] = previous[indices] + delta
    if not np.isfinite(output).all():
        raise ControlFault("computed twist is not finite")
    return output
