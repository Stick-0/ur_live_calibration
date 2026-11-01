"""Explicit stop, observation and re-arm policy; this module never sends motion."""

from collections import deque
import numpy as np

from urcal.control.geometry import pose_delta
from urcal.control.policy import (
    CalibrationTracker, CalibrationUnavailable, ControlFault, PlanningUnavailable,
    _transform,
)


_PLANNING_CATEGORIES = frozenset({
    "measurement_unavailable", "measurement_expired", "measurement_invalid",
    "tracking_error", "tracking_infeasible", "transition_infeasible",
    "compensation_infeasible", "reachability_infeasible", "singularity",
    "workspace", "clearance", "workspace_infeasible", "clearance_infeasible",
})


def is_recoverable_fault(exc):
    """Recognize planning/sensor faults, never infer recovery from message text.

    Recovery means a confirmed stop followed by observation and a NEW Arm. It
    never permits carrying on with an invalid command, controller failure,
    measured overspeed, missed deadline, operator stop or unknown exception.
    """
    if isinstance(exc, (CalibrationUnavailable, PlanningUnavailable)):
        return True
    # Local import avoids the runtime/controller import cycle and requires a
    # known fault type, not an arbitrary object bearing a category attribute.
    from urcal.cooperative.controller import ResearchFault
    from urcal.cooperative.task import MotionInfeasible
    return (isinstance(exc, (ResearchFault, MotionInfeasible))
            and exc.category in _PLANNING_CATEGORIES)


def fresh_calibration_tracker(template):
    """Empty acceptance state with unchanged configured calibration limits."""
    return CalibrationTracker(**{key: getattr(template, key) for key in (
        "max_age", "stable_frames", "alpha", "max_jump_m", "max_jump_rad",
        "max_mean_px", "max_pixel_px", "min_points", "check_reprojection",
    )})


def _positive(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + " must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(label + " must be finite and positive")
    return value


def _state_values(state):
    """Copy finite encoder state; missing qdot is allowed for hold readers."""
    try:
        timestamp = float(state.robot_timestamp)
        joints = np.asarray(state.joints, dtype=float)
        speed = np.asarray(state.speed, dtype=float)
        qdot = getattr(state, "joint_velocity", None)
        qdot = None if qdot is None else np.asarray(qdot, dtype=float)
        if (not np.isfinite(timestamp) or timestamp < 0
                or joints.shape != (6,) or speed.shape != (6,)
                or not np.isfinite([joints, speed]).all()
                or (qdot is not None and (qdot.shape != (6,) or not np.isfinite(qdot).all()))):
            raise ValueError("invalid encoder state")
        tcp = _transform(state.tcp, "recovery TCP")
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ControlFault("invalid robot state during stop verification") from exc
    return timestamp, joints.copy(), speed.copy(), None if qdot is None else qdot.copy(), tcp


class RecoveryStopGate:
    """Require fresh stationary encoder observations before offering a new Arm.

    The stop command's successful return is necessary but not sufficient. All
    commanded arms must advance beyond their pre-stop timestamp and remain
    stationary for a short dwell. Reported velocities AND changes of joint/TCP
    positions are checked. TCP translation uses a rolling dwell-length
    observation window: differentiating tens of micrometres of position noise
    at every control tick would repeatedly revoke a valid stop. Reported
    velocity, joint and angular checks still apply immediately.
    This gate neither accepts calibration nor resumes motion; the caller
    starts a new calibration cluster after ``ready_at``.
    """

    def __init__(self, names, states, stopped_at, *, max_state_age=.15,
                 stationary_duration=.2, linear_tolerance=.001,
                 angular_tolerance=.005, joint_tolerance=.01,
                 confirmation_timeout=5.):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("recovery requires distinct commanded arms")
        self.max_state_age = _positive(max_state_age, "max_state_age")
        self.stationary_duration = _positive(stationary_duration, "stationary_duration")
        self.linear_tolerance = _positive(linear_tolerance, "linear_tolerance")
        self.angular_tolerance = _positive(angular_tolerance, "angular_tolerance")
        self.joint_tolerance = _positive(joint_tolerance, "joint_tolerance")
        self.confirmation_timeout = _positive(confirmation_timeout, "confirmation_timeout")
        if not np.isfinite(stopped_at) or stopped_at < 0:
            raise ControlFault("invalid stop verification clock")
        self.stopped_at = self._last_now = float(stopped_at)
        try:
            self._previous = {name: _state_values(states[name]) for name in self.names}
        except KeyError as exc:
            raise ControlFault("missing commanded arm state during stop verification") from exc
        self._advanced_at = {name: self.stopped_at for name in self.names}
        self._position_history = {
            name: deque([(sample[0], sample[4][:3, 3].copy())])
            for name, sample in self._previous.items()
        }
        self._post_stop_seen = set()
        self._stationary_since = None
        self.ready = False
        self.ready_at = None
        self._ever_ready = False
        self.reason = "waiting for fresh stationary robot state after stop"

    def _wait(self, reason):
        self.ready = False
        self.ready_at = None
        self._stationary_since = None
        self.reason = reason
        return False

    def observe(self, states, now):
        if not np.isfinite(now) or now < self._last_now:
            raise ControlFault("invalid stop verification clock")
        self._last_now = float(now)
        if not self._ever_ready and now - self.stopped_at > self.confirmation_timeout:
            raise ControlFault(f"robot stop stationarity not confirmed within {self.confirmation_timeout:g}s")
        try:
            samples = {name: _state_values(states[name]) for name in self.names}
        except KeyError as exc:
            raise ControlFault("missing commanded arm state during stop verification") from exc
        stationary = True
        all_advanced = True
        for name, sample in samples.items():
            stamp, joints, speed, qdot, tcp = sample
            previous = self._previous[name]
            if stamp < previous[0]:
                raise ControlFault(name + ": robot timestamp moved backwards during stop verification")
            advanced = stamp > previous[0]
            all_advanced &= advanced
            if advanced:
                elapsed = stamp - previous[0]
                _, dr = pose_delta(previous[4], tcp)
                history = self._position_history[name]
                # Retain one sample at/before the rolling window boundary.
                # Using only consecutive samples amplifies stationary pose
                # noise; using only the newest pair could miss slow drift.
                while len(history) > 1 and history[1][0] <= stamp - self.stationary_duration:
                    history.popleft()
                position = tcp[:3, 3]
                translation_stable = all(
                    np.linalg.norm(position - old_position)
                    <= self.linear_tolerance * max(stamp - old_stamp, self.stationary_duration) + 1e-12
                    for old_stamp, old_position in history
                )
                history.append((stamp, position.copy()))
                stationary &= (np.max(np.abs(joints - previous[1])) / elapsed <= self.joint_tolerance
                               and translation_stable
                               and np.linalg.norm(dr) / elapsed <= self.angular_tolerance)
                self._previous[name] = sample
                self._advanced_at[name] = now
                self._post_stop_seen.add(name)
            elif now - self._advanced_at[name] > self.max_state_age:
                raise ControlFault(name + ": robot state stopped updating during stop verification")
            stationary &= (np.linalg.norm(speed[:3]) <= self.linear_tolerance
                           and np.linalg.norm(speed[3:]) <= self.angular_tolerance
                           and (qdot is None or np.max(np.abs(qdot)) <= self.joint_tolerance))
        if not stationary:
            return self._wait("waiting for commanded arms to become stationary")
        if not all_advanced or len(self._post_stop_seen) != len(self.names):
            self.reason = "waiting for advancing robot timestamps after stop"
            # Cached packets cannot establish readiness. Once established, a
            # still-fresh duplicate can retain it, subject to the age limit.
            return self.ready
        if self._stationary_since is None:
            self._stationary_since = now
        elapsed = now - self._stationary_since
        if elapsed + 1e-12 < self.stationary_duration:
            self.reason = f"confirming stationary robot state ({elapsed:.2f}/{self.stationary_duration:.2f}s)"
            return False
        if not self.ready:
            self.ready_at = float(now)
        self.ready = True
        self._ever_ready = True
        self.reason = "robot stop confirmed; acquire fresh calibration, then press Arm again"
        return True
