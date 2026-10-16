"""Time-aligned relative robot-base state without hardware or pose-sensor changes.

Inputs are camera-from-controller-base SE(3) matrices. Capture and receipt
times share a monotonic clock but serve different purposes: only capture time
advances the physical state. Direct mode interpolates raw histories to their
latest common historical time. Predict mode filters on SE(3), then applies a
bounded-duration constant spatial-twist extrapolation to the requested time.

Spatial Lie-algebra translation is not an origin velocity. Internally a twist
is [u, w], with p_dot = u + w cross p. Public base_twists instead contain
[p_dot, w], expressed in camera axes. relative_twist is the derivative of the
follower origin and rotation in the MOVING leader frame, including transport.

Spatial increments and left-multiplied corrections preserve a constant right-hand
relative transform under matched common rigid camera motion. This is a short
motion model, not a classifier of genuine sudden motion versus pose errors.
Samples exceeding configured physical speed limits raise an error; they are
never clipped into apparently valid measurements.
"""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose


class RelativeStateError(ValueError):
    """The measurement or requested state cannot satisfy the time/motion limits."""


@dataclass(frozen=True)
class RelativeSnapshot:
    at: float
    base_poses: dict
    base_twists: dict
    relative_pose: np.ndarray
    relative_twist: np.ndarray
    measurement_age: dict
    measurement_times: dict
    received_times: dict
    prediction_dt: dict


@dataclass(frozen=True)
class _Sample:
    captured: float
    received: float
    sequence: int
    raw: np.ndarray
    filtered: np.ndarray
    raw_velocity: np.ndarray
    velocity: np.ndarray


def _scalar(value, label, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RelativeStateError(f"{label} must be finite") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise RelativeStateError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _pose(value):
    try:
        pose = np.asarray(value, dtype=np.float64)
        matrix_pose(pose)  # Existing strict SE(3) validator; no projection of bad inputs.
    except (TypeError, ValueError, OverflowError) as exc:
        raise RelativeStateError("base pose must be a finite proper SE(3) transform") from exc
    return pose.copy()


def _hat(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _left_jacobian(phi):
    theta = float(np.linalg.norm(phi))
    phi_hat = _hat(phi)
    if theta < 1e-5:
        theta2 = theta * theta
        a = .5 - theta2 / 24. + theta2 * theta2 / 720.
        b = 1. / 6. - theta2 / 120. + theta2 * theta2 / 5040.
    else:
        a = (1. - np.cos(theta)) / theta ** 2
        b = (theta - np.sin(theta)) / theta ** 3
    return np.eye(3) + a * phi_hat + b * (phi_hat @ phi_hat)


def _exp(twist):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(twist[3:]).as_matrix()
    result[:3, 3] = _left_jacobian(twist[3:]) @ twist[:3]
    return result


def _log(pose):
    phi = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate((np.linalg.solve(_left_jacobian(phi), pose[:3, 3]), phi))


def _origin_twist(pose, spatial):
    return np.concatenate((spatial[:3] + np.cross(spatial[3:], pose[:3, 3]), spatial[3:]))


class RelativeStateEstimator:
    """Accept validated base measurements and estimate a time-consistent pair.

    update/update_pair return False for an identical duplicate without changing
    any timestamp. update_pair commits both samples atomically. Receipt times
    may differ per robot (pass a dict); capture time and sequence describe one
    shared image. Async sources can use update separately. At most 256 samples
    per robot are retained; unavailable direct-mode interpolation is rejected.

    Snapshot measurement_age and prediction_dt are per-robot dictionaries.
    measurement_times/received_times refer to the latest accepted observations.
    In direct mode snapshot.at is min(latest capture times), and prediction_dt
    is zero even when the newer stream was interpolated backwards to that time.
    In predict mode snapshot.at is now and prediction_dt is now-capture_time.
    A newly measured state after a gap longer than max_prediction_age starts a
    fresh zero-velocity filter; the old velocity is not projected across the gap.
    """

    def __init__(self, robots=("ur5e", "ur10e"), mode="direct", alpha=.2,
                 velocity_alpha=.3, max_prediction_age=.5, max_measurement_age=.5,
                 max_linear_speed=.15, max_angular_speed=.5):
        try:
            self.robots = tuple(robots)
        except TypeError as exc:
            raise RelativeStateError("robots must contain two distinct nonempty names") from exc
        if (len(self.robots) != 2 or not all(isinstance(r, str) and r for r in self.robots)
                or len(set(self.robots)) != 2):
            raise RelativeStateError("robots must contain two distinct nonempty names")
        if mode not in ("direct", "predict"):
            raise RelativeStateError("mode must be direct or predict")
        self.mode = mode
        for name, value in (("alpha", alpha), ("velocity_alpha", velocity_alpha),
                            ("max_prediction_age", max_prediction_age),
                            ("max_measurement_age", max_measurement_age),
                            ("max_linear_speed", max_linear_speed), ("max_angular_speed", max_angular_speed)):
            setattr(self, name, _scalar(value, name, positive=True))
        if self.alpha > 1 or self.velocity_alpha > 1:
            raise RelativeStateError("alpha and velocity_alpha must be in (0, 1]")
        self._history = {name: deque(maxlen=256) for name in self.robots}

    def _check_speed(self, spatial, poses, label):
        if not np.isfinite(spatial).all():
            raise RelativeStateError(f"{label}: nonfinite base velocity")
        if np.linalg.norm(spatial[3:]) > self.max_angular_speed + 1e-10:
            raise RelativeStateError(f"{label}: base angular speed exceeds limit")
        for pose in poses:
            if np.linalg.norm(_origin_twist(pose, spatial)[:3]) > self.max_linear_speed + 1e-10:
                raise RelativeStateError(f"{label}: base linear speed exceeds limit")

    def _candidate(self, robot, pose, captured_at, received_at, sequence):
        if robot not in self._history:
            raise RelativeStateError(f"unknown robot: {robot}")
        pose = _pose(pose)
        captured = _scalar(captured_at, "capture time")
        received = _scalar(received_at, "receipt time")
        if captured > received:
            raise RelativeStateError("capture time is in the future relative to receipt")
        if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
            raise RelativeStateError("sequence must be a nonnegative integer")
        history = self._history[robot]
        previous = history[-1] if history else None
        if previous is not None and received < previous.received:
            raise RelativeStateError("receipt time moved backwards")
        if previous is not None and sequence == previous.sequence:
            if captured != previous.captured or not np.array_equal(pose, previous.raw):
                raise RelativeStateError("duplicate sequence changed its measurement")
            return None
        if received - captured > self.max_measurement_age:
            raise RelativeStateError("measurement arrived too late")
        if previous is not None:
            if sequence < previous.sequence or captured <= previous.captured:
                raise RelativeStateError("out-of-order measurement sequence/capture time")
            dt = captured - previous.captured
            raw_velocity = _log(pose @ inverse(previous.raw)) / dt
            self._check_speed(raw_velocity, (previous.raw, pose), robot + " observed")
            if self.mode == "direct":
                filtered, velocity = pose.copy(), raw_velocity.copy()
            elif dt > self.max_prediction_age:
                filtered, velocity = pose.copy(), np.zeros(6)
            else:
                predicted = _exp(previous.velocity * dt) @ previous.filtered
                innovation = _log(pose @ inverse(predicted))
                filtered = _exp(self.alpha * innovation) @ predicted
                velocity = (1. - self.velocity_alpha) * previous.velocity + self.velocity_alpha * raw_velocity
                self._check_speed(velocity, (filtered,), robot + " filtered")
        else:
            raw_velocity = np.zeros(6)
            velocity = np.zeros(6)
            filtered = pose.copy()
        return _Sample(captured, received, int(sequence), pose, _pose(filtered), raw_velocity.copy(), velocity.copy())

    def update(self, robot, pose, captured_at, received_at, sequence):
        sample = self._candidate(robot, pose, captured_at, received_at, sequence)
        if sample is None:
            return False
        self._history[robot].append(sample)
        return True

    def update_pair(self, poses, captured_at, received_at, sequence):
        if not isinstance(poses, dict) or set(poses) != set(self.robots):
            raise RelativeStateError("a pair must contain exactly both robot poses")
        if isinstance(received_at, dict):
            if set(received_at) != set(self.robots):
                raise RelativeStateError("receipt times must contain exactly both robots")
            receipts = received_at
        else:
            receipts = {name: received_at for name in self.robots}
        candidates = {
            name: self._candidate(name, poses[name], captured_at, receipts[name], sequence)
            for name in self.robots
        }
        for name, candidate in candidates.items():
            if candidate is not None:
                self._history[name].append(candidate)
        return any(value is not None for value in candidates.values())

    def _historical(self, robot, at):
        history = self._history[robot]
        for sample in history:
            if sample.captured == at:
                return sample.raw.copy(), sample.raw_velocity.copy()
        for before, after in zip(history, list(history)[1:]):
            if before.captured < at < after.captured:
                velocity = _log(after.raw @ inverse(before.raw)) / (after.captured - before.captured)
                pose = _exp(velocity * (at - before.captured)) @ before.raw
                return pose, velocity
        raise RelativeStateError(f"{robot}: no observations bracket the common capture time")

    def estimate(self, now, leader="ur10e"):
        now = _scalar(now, "estimate time")
        if leader not in self.robots:
            raise RelativeStateError("leader must be one of the configured robots")
        if any(not self._history[name] for name in self.robots):
            raise RelativeStateError("both robot measurements are required")
        follower = next(name for name in self.robots if name != leader)
        latest = {name: self._history[name][-1] for name in self.robots}
        ages = {name: now - sample.captured for name, sample in latest.items()}
        if any(now < sample.received for sample in latest.values()):
            raise RelativeStateError("estimate time precedes a measurement receipt")
        if any(age < 0 or age > self.max_measurement_age for age in ages.values()):
            raise RelativeStateError("last valid base measurement has expired")
        poses, velocities, prediction_dt = {}, {}, {}
        at = min(sample.captured for sample in latest.values()) if self.mode == "direct" else now
        for name, sample in latest.items():
            if self.mode == "direct":
                poses[name], spatial = self._historical(name, at)
                prediction_dt[name] = 0.
            else:
                duration = ages[name]
                if duration > self.max_prediction_age:
                    raise RelativeStateError(f"{name}: short prediction horizon expired")
                poses[name] = _exp(sample.velocity * duration) @ sample.filtered
                spatial = sample.velocity
                self._check_speed(spatial, (poses[name],), name + " prediction")
                prediction_dt[name] = duration
            poses[name] = _pose(poses[name])
            velocities[name] = _origin_twist(poses[name], spatial)
        leader_pose, follower_pose = poses[leader], poses[follower]
        leader_velocity, follower_velocity = velocities[leader], velocities[follower]
        parent_to_leader = leader_pose[:3, :3].T
        displacement = follower_pose[:3, 3] - leader_pose[:3, 3]
        relative_linear = parent_to_leader @ (
            follower_velocity[:3] - leader_velocity[:3] - np.cross(leader_velocity[3:], displacement))
        relative_angular = parent_to_leader @ (follower_velocity[3:] - leader_velocity[3:])
        relative_velocity = np.concatenate((relative_linear, relative_angular))
        if not np.isfinite(relative_velocity).all():
            raise RelativeStateError("relative velocity is nonfinite")
        relative_transform = _pose(inverse(leader_pose) @ follower_pose)
        return RelativeSnapshot(
            at=at, base_poses={name: pose.copy() for name, pose in poses.items()},
            base_twists={name: twist.copy() for name, twist in velocities.items()},
            relative_pose=relative_transform,
            relative_twist=relative_velocity,
            measurement_age=ages,
            measurement_times={name: sample.captured for name, sample in latest.items()},
            received_times={name: sample.received for name, sample in latest.items()},
            prediction_dt=prediction_dt,
        )
