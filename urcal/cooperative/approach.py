"""An explicitly requested approach along the leader flange's positive Z axis.

Only the commanded relative reference is used. There is no robot IO, measured
pose capture, motion-history reset, task-clock advancement, or safety-limit
override here; those remain the controller/runtime's responsibility.
"""

import math

import numpy as np

from urcal.control.geometry import matrix_pose
from urcal.cooperative.task import RelativePath


def _positive(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array) or float(array) <= 0.:
        raise ValueError(name + " must be a finite positive scalar")
    return float(array)


def _committed_progress(value):
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array) or float(array) < 0.:
        raise ValueError("approach progress must be finite and nonnegative")
    return min(1., float(array))


class _AxialApproachPath(RelativePath):
    """Keep the exact endpoint representable at the existing clearance bound."""

    def __init__(self, start, target_gap, duration):
        self._target_gap = target_gap
        super().__init__(start, kind="line", delta=[0., 0., target_gap - start[2, 3], 0., 0., 0.],
                         duration=duration)

    def pose(self, progress):
        result = super().pose(progress)
        # Quintic evaluation/subtraction may round a few ulps below the exact
        # endpoint. This is a numeric clamp, not an added clearance margin.
        result[2, 3] = self._target_gap if progress == 1. else max(self._target_gap, result[2, 3])
        return result


def build_gap_approach(path, progress, tracking_offset, *, speed=.005, min_gap=.15,
                       target_gap=.15, acceleration=.01):
    """Build a line from the current commanded reference to the axial gap.

    A stationary relative path is required; an unrelated running line/circle
    must not be overwritten. The start is ``path.pose(progress) @ offset``.
    Its X/Y and orientation remain fixed while leader-frame Z decreases to
    ``target_gap``. The controller may fold the old offset into this path and
    install an identity offset without discontinuity in the command reference.

    ``speed`` is peak target speed in m/s, not average speed or robot speed.
    Quintic duration respects that peak and the target acceleration budget;
    normal command, joint, geometry and measured-state checks still apply.
    """
    if not isinstance(path, RelativePath) or path.kind != "hold":
        raise ValueError("approach requires stationary relative following")
    speed = _positive(speed, "approach speed")
    acceleration = _positive(acceleration, "approach target acceleration")
    min_gap = _positive(min_gap, "minimum flange gap")
    target_gap = _positive(target_gap, "approach target gap")
    if min_gap > target_gap:
        raise ValueError("configured minimum flange gap exceeds the approach target gap")
    offset = np.eye(4) if tracking_offset is None else np.asarray(tracking_offset, dtype=float)
    matrix_pose(offset)
    start = path.pose(progress) @ offset
    matrix_pose(start)
    distance = float(start[2, 3]) - target_gap
    if distance < 0.:
        raise ValueError("commanded axial gap is already below the approach target")
    if distance <= 1e-12:
        raise ValueError("commanded axial gap is already at the approach target")
    duration = max(1.875 * distance / speed,
                   math.sqrt((10. * math.sqrt(3.) / 3.) * distance / acceleration))
    if not np.isfinite(duration) or duration <= 0.:
        raise ValueError("approach duration must be finite and positive")
    return _AxialApproachPath(start, target_gap, duration)


class GapApproach:
    """Status for an explicit start/pause/resume action, separate from motion."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state, self.reason = "idle", ""
        self._path, self._speed = None, None
        self._target_gap, self._progress = None, 0.

    def start(self, path, progress, tracking_offset, *, speed=.005, min_gap=.15,
              target_gap=.15, acceleration=.01):
        if self.state != "idle":
            raise ValueError("an approach is already configured; pause or resume the existing approach")
        approach = build_gap_approach(path, progress, tracking_offset, speed=speed,
                                      min_gap=min_gap, target_gap=target_gap, acceleration=acceleration)
        # Validation completes before any lifecycle state changes.
        self._path, self._speed = approach, float(speed)
        self._target_gap, self._progress = float(target_gap), 0.
        self.state, self.reason = "running", ""
        return approach

    def pause(self, reason="operator"):
        if self.state != "running":
            return False
        self.state, self.reason = "paused", str(reason)
        return True

    def resume(self):
        if self.state != "paused":
            return False
        self.state, self.reason = "running", ""
        return True

    def commit(self, progress):
        """Record successfully sent reference progress, never measured arrival."""
        if self.state == "idle":
            return
        self._progress = _committed_progress(progress)
        if self._progress >= 1.:
            self.state, self.reason = "complete", "approach reference reached the target gap"

    def info(self, progress=None):
        current = self._progress if progress is None else _committed_progress(progress)
        return {
            "state": self.state, "reason": self.reason,
            "running": self.state == "running", "paused": self.state == "paused",
            "progress": current,
            "start_gap_m": None if self._path is None else float(self._path.start[2, 3]),
            "target_gap_m": self._target_gap,
            "reference_gap_m": None if self._path is None else float(self._path.pose(current)[2, 3]),
            "speed_m_s": self._speed,
            "duration_s": None if self._path is None else self._path.duration,
        }
