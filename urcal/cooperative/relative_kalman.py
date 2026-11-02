"""Same-image relative-base planar constant-velocity Kalman estimation.

The state is [x, y, yaw, vx, vy, yaw_rate] of the physical leader base in
the physical follower base. Camera poses are never separately differentiated.
Only accepted capture times advance the filter. estimate() is a read-only
prediction, not a measurement or a source of refreshed sensor timestamps.
"""

from collections import deque
from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base
from urcal.cooperative.base_plane import _physical_offsets, project_output_bases
from urcal.cooperative.state import RelativeSnapshot, RelativeStateError, _pose, _scalar


ROBOTS = ("ur5e", "ur10e")


@dataclass(frozen=True)
class PlanarKalmanConfig:
    measurement_std_m: float = .007
    measurement_std_yaw: float = math.radians(.3)
    acceleration_std_m: float = .10
    acceleration_std_yaw: float = .15
    initial_velocity_std_m: float = .05
    initial_velocity_std_yaw: float = .15
    prediction_horizon: float = .35
    gap_reset: float = .35
    min_velocity_samples: int = 4
    innovation_gate: float = 16.27
    max_step_m: float = .15
    max_step_yaw: float = math.radians(15.)
    max_speed: float = .15
    max_yaw_rate: float = .5
    stationary_window: float = .25
    stationary_sigma: float = .75
    reversal_sigma: float = 3.
    velocity_snr_min: float = 1.5
    velocity_snr_full: float = 3.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "min_velocity_samples":
                if not isinstance(value, Integral) or isinstance(value, bool) or value < 2:
                    raise ValueError("min_velocity_samples must be an integer >= 2")
                continue
            try:
                valid = not isinstance(value, bool) and np.isscalar(value) and np.isfinite(value) and value > 0
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(field.name + " must be positive and finite")
        if self.velocity_snr_full <= self.velocity_snr_min:
            raise ValueError("velocity_snr_full must exceed velocity_snr_min")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("relative Kalman configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown relative Kalman settings: " + ", ".join(sorted(unknown)))
        return cls(**values)


def load_planar_kalman_config(path=None):
    """Load optional SI-unit tuning values; reject misspelled/unused settings."""
    return PlanarKalmanConfig() if path is None else PlanarKalmanConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))


def _wrap(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _psd(covariance):
    symmetric = .5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, 0.)) @ vectors.T


class PlanarRelativeStateEstimator:
    """Atomically filter a same-capture pair, preserving the newest Bf anchor.

    Snapshot.relative_filter['raw_origin_velocity'] is the predicted geometric
    velocity at G_control's origin, in follower output-base axes. The separate
    'origin_velocity' already includes the residual confidence 'lambda'. Motion
    SNR/support have already scaled G prediction and its derivative through
    'prediction_gain'; lambda adds covariance/age validity once. Zero legacy
    base_twists/relative_twist prevent a second, independent velocity channel.
    """

    def __init__(self, leader="ur10e", mappings=None, max_measurement_age=.5, config=None):
        if leader not in ROBOTS:
            raise RelativeStateError("leader must be ur5e or ur10e")
        self.leader = leader
        self.follower = next(name for name in ROBOTS if name != leader)
        self.robots = ROBOTS
        self.mode = "relative_kalman"
        self.config = PlanarKalmanConfig() if config is None else config
        if not isinstance(self.config, PlanarKalmanConfig):
            raise ValueError("config must be PlanarKalmanConfig")
        self.max_measurement_age = _scalar(max_measurement_age, "maximum measurement age", positive=True)
        if self.config.prediction_horizon > self.max_measurement_age:
            raise ValueError("prediction_horizon must not exceed max_measurement_age")
        mappings = {name: model_from_controller_base(name) for name in ROBOTS} if mappings is None else mappings
        if not isinstance(mappings, dict) or set(mappings) != set(ROBOTS):
            raise RelativeStateError("mappings must contain exactly both robots")
        self.mappings = {name: _pose(value) for name, value in mappings.items()}
        self.offsets = _physical_offsets(self.mappings)
        c = self.config
        self.R = np.diag([c.measurement_std_m**2, c.measurement_std_m**2, c.measurement_std_yaw**2])
        self._x = self._P = self._raw_poses = self._receipts = self._raw_measurement = None
        self._capture = self._sequence = None
        self._support = 0
        self._rejected = False
        self._reason = "waiting for paired measurements"
        self._innovation = np.zeros(3)
        self._innovation_distance = 0.
        self._recent = deque(maxlen=128)

    def _noise(self, dt):
        diagonal = np.diag([self.config.acceleration_std_m**2] * 2
                           + [self.config.acceleration_std_yaw**2])
        return np.block([[diagonal * dt**4 / 4., diagonal * dt**3 / 2.],
                         [diagonal * dt**3 / 2., diagonal * dt**2]])

    def _initial_covariance(self):
        c = self.config
        return np.diag([c.measurement_std_m**2] * 2 + [c.measurement_std_yaw**2]
                       + [c.initial_velocity_std_m**2] * 2 + [c.initial_velocity_std_yaw**2])

    def _measurement(self, poses):
        projected = project_output_bases(poses, self.mappings)
        physical = {name: projected[name] @ self.offsets[name] for name in ROBOTS}
        relation = inverse(physical[self.follower]) @ physical[self.leader]
        return np.array([relation[0, 3], relation[1, 3], math.atan2(relation[1, 0], relation[0, 0])])

    def _validate(self, poses, captured_at, received_at, sequence):
        if not isinstance(poses, dict) or set(poses) != set(ROBOTS):
            raise RelativeStateError("a relative pair must contain exactly both robot poses")
        values = {name: _pose(poses[name]) for name in ROBOTS}
        captured = _scalar(captured_at, "capture time")
        if isinstance(received_at, dict):
            if set(received_at) != set(ROBOTS):
                raise RelativeStateError("receipt times must contain exactly both robots")
            receipts = {name: _scalar(received_at[name], "receipt time") for name in ROBOTS}
        else:
            receipts = {name: _scalar(received_at, "receipt time") for name in ROBOTS}
        if any(captured > receipt for receipt in receipts.values()):
            raise RelativeStateError("capture time is in the future relative to receipt")
        if any(receipt - captured > self.max_measurement_age for receipt in receipts.values()):
            raise RelativeStateError("relative measurement arrived too late")
        if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
            raise RelativeStateError("sequence must be a nonnegative integer")
        if self._capture is not None:
            if any(receipts[name] < self._receipts[name] for name in ROBOTS):
                raise RelativeStateError("relative receipt time moved backwards")
            if sequence == self._sequence:
                if captured != self._capture or any(not np.array_equal(values[name], self._raw_poses[name]) for name in ROBOTS):
                    raise RelativeStateError("duplicate relative sequence changed its measurement")
                return None
            if sequence < self._sequence or captured <= self._capture:
                raise RelativeStateError("out-of-order relative measurement sequence/capture time")
        return values, captured, receipts, int(sequence)

    def note_rejected(self, now, reason):
        """Suppress compensation without refreshing or replacing accepted data."""
        _scalar(now, "rejection time")
        self._rejected = True
        self._reason = "measurement rejected: " + str(reason)
        self._support = 0

    def update_pair(self, poses, captured_at, received_at, sequence):
        try:
            candidate = self._validate(poses, captured_at, received_at, sequence)
            if candidate is None:
                return False
            values, captured, receipts, sequence = candidate
            measurement = self._measurement(values)
            c = self.config
            reset = self._x is None
            dt = 0. if reset else captured - self._capture
            innovation, distance = np.zeros(3), 0.
            if not reset:
                step = measurement - self._raw_measurement
                step[2] = _wrap(step[2])
                if np.linalg.norm(step[:2]) > c.max_step_m or abs(step[2]) > c.max_step_yaw:
                    raise RelativeStateError("relative planar observation exceeds step limits")
                transition = np.eye(6)
                transition[:3, 3:] = np.eye(3) * dt
                predicted = transition @ self._x
                covariance = _psd(transition @ self._P @ transition.T + self._noise(dt))
                innovation = measurement - predicted[:3]
                innovation[2] = _wrap(innovation[2])
                residual_covariance = covariance[:3, :3] + self.R
                distance = float(innovation @ np.linalg.solve(residual_covariance, innovation))
                if distance > c.innovation_gate:
                    raise RelativeStateError(f"relative planar innovation rejected ({distance:.3f} > {c.innovation_gate:g})")
                reset = self._rejected or dt > c.gap_reset
            if reset:
                state = np.r_[measurement, np.zeros(3)]
                covariance = self._initial_covariance()
                support, recent, reason = 1, deque(maxlen=128), "warming up relative velocity"
            else:
                gain = np.linalg.solve(residual_covariance, covariance[:3, :]).T
                state = predicted + gain @ innovation
                projection = np.eye(6)
                projection[:, :3] -= gain
                covariance = _psd(projection @ covariance @ projection.T + gain @ self.R @ gain.T)
                support, recent, reason = self._support + 1, deque(self._recent, maxlen=128), "tracking"
                # A measured reversal beyond the noise floor invalidates the
                # old direction immediately, then requires fresh support.
                opposite_linear = (float(self._x[3:5] @ step[:2])
                                   < -c.reversal_sigma * math.sqrt(2.) * c.measurement_std_m * np.linalg.norm(self._x[3:5]))
                opposite_yaw = self._x[5] * step[2] < -c.reversal_sigma * math.sqrt(2.) * c.measurement_std_yaw * abs(self._x[5])
                if opposite_linear or opposite_yaw:
                    state[3:] = 0.
                    covariance[:3, 3:] = covariance[3:, :3] = 0.
                    covariance[3:, 3:] = self._initial_covariance()[3:, 3:]
                    support, recent, reason = 1, deque(maxlen=128), "direction change; warming up relative velocity"
            state[2] = _wrap(state[2])
            recent.append((captured, measurement.copy()))
            while len(recent) > 2 and recent[1][0] <= captured - c.stationary_window:
                recent.popleft()
            if len(recent) >= c.min_velocity_samples and captured - recent[0][0] >= c.stationary_window:
                positions = np.array([row[1][:2] for row in recent])
                angles = np.array([_wrap(row[1][2] - recent[-1][1][2]) for row in recent])
                translation_span = max(np.linalg.norm(position - positions[-1]) for position in positions)
                if translation_span <= c.stationary_sigma * c.measurement_std_m and np.ptp(angles) <= c.stationary_sigma * c.measurement_std_yaw:
                    state[3:] = 0.
                    covariance[:3, 3:] = covariance[3:, :3] = 0.
                    covariance[3:, 3:] = self._initial_covariance()[3:, 3:]
                    reason = "stationary within measurement noise"
            speed = float(np.linalg.norm(state[3:5]))
            if speed > c.max_speed:
                state[3:5] *= c.max_speed / speed
                reason = "relative velocity bounded"
            if abs(state[5]) > c.max_yaw_rate:
                state[5] = math.copysign(c.max_yaw_rate, state[5])
                reason = "relative velocity bounded"
            if not np.isfinite(state).all() or not np.isfinite(covariance).all():
                raise RelativeStateError("relative filter produced a nonfinite state")
        except (RelativeStateError, ValueError, np.linalg.LinAlgError) as exc:
            # Accepted pose/covariance/times remain unchanged on every failure.
            self._rejected, self._support = True, 0
            self._reason = "measurement rejected: " + str(exc)
            if isinstance(exc, RelativeStateError):
                raise
            raise RelativeStateError(str(exc)) from exc
        self._x, self._P = state.copy(), _psd(covariance)
        self._capture, self._receipts, self._sequence = captured, receipts, sequence
        self._raw_poses, self._raw_measurement = values, measurement.copy()
        self._support, self._recent, self._reason = support, recent, reason
        self._innovation, self._innovation_distance = innovation.copy(), distance
        self._rejected = False
        return True

    def initialize_stationary_pair(self, raw_poses, filtered_poses, captured_at, received_at, sequence):
        if self._x is not None:
            raise RelativeStateError("stationary initialization requires an empty relative estimator")
        if not isinstance(filtered_poses, dict) or set(filtered_poses) != set(ROBOTS):
            raise RelativeStateError("stationary pair must contain exactly both robots")
        filtered = self._measurement({name: _pose(filtered_poses[name]) for name in ROBOTS})
        self.update_pair(raw_poses, captured_at, received_at, sequence)
        self._x[:3], self._x[3:] = filtered, 0.
        self._reason = "stationary consensus; warming up relative velocity"

    def estimate(self, now, leader=None):
        now = _scalar(now, "estimate time")
        if leader is not None and leader != self.leader:
            raise RelativeStateError("estimate leader differs from the configured relative state")
        if self._x is None:
            raise RelativeStateError("both relative measurements are required")
        age = now - self._capture
        if any(now < receipt for receipt in self._receipts.values()):
            raise RelativeStateError("estimate time precedes a measurement receipt")
        if age < 0. or age > self.max_measurement_age:
            raise RelativeStateError("last valid relative base measurement has expired")
        c = self.config
        # Apply accepted-posterior motion confidence to extrapolation itself,
        # not just to the feedforward command. A stationary, noisy yaw must not
        # be predicted as motion merely because translation is well observed.
        # These gains are constant between accepted measurements, so the
        # returned velocity remains the derivative of the predicted pose.
        speed = float(np.linalg.norm(self._x[3:5]))
        direction = self._x[3:5] / speed if speed > 0. else np.zeros(2)
        direction_variance = float(direction @ self._P[3:5, 3:5] @ direction)
        signal = np.array([speed / math.sqrt(max(direction_variance, 1e-18)),
                           abs(self._x[5]) / math.sqrt(max(self._P[5, 5], 1e-18))])
        confidence_signal = np.clip((signal - c.velocity_snr_min)
                                    / (c.velocity_snr_full - c.velocity_snr_min), 0., 1.)
        confidence_support = float(np.clip((self._support - c.min_velocity_samples + 1)
                                           / c.min_velocity_samples, 0., 1.))
        prediction_gain = confidence_signal[[0, 0, 1]] * confidence_support
        if self._rejected:
            prediction_gain[:] = 0.
        # Integrate a short CV prediction and then continuously bring its
        # velocity to zero. Position and raw velocity are one model at one time.
        half = .5 * c.prediction_horizon
        extra = min(max(age - half, 0.), half)
        effective_dt = min(age, half) + extra - extra**2 / c.prediction_horizon
        decay = 1. if age <= half else max(0., 1. - extra / half)
        transition = np.eye(6)
        transition[:3, 3:] = np.diag(prediction_gain) * effective_dt
        transition[3:, 3:] = np.diag(prediction_gain) * decay
        state = transition @ self._x
        state[2] = _wrap(state[2])
        # Limiting the mean's extrapolation is not new evidence that its
        # uncertainty disappeared. Retain the full CV uncertainty envelope,
        # including process uncertainty throughout the actual measurement age.
        uncertainty_transition = np.eye(6)
        uncertainty_transition[:3, 3:] = np.eye(3) * age
        covariance = _psd(uncertainty_transition @ self._P @ uncertainty_transition.T + self._noise(age))
        physical = np.eye(4)
        cosine, sine = math.cos(state[2]), math.sin(state[2])
        physical[:3, :3] = [[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]
        physical[:2, 3] = state[:2]
        left, right = self.offsets[self.follower], inverse(self.offsets[self.leader])
        relation = left @ physical @ right
        angular = np.array([0., 0., state[5]])
        origin = left[:3, :3] @ (np.r_[state[3:5], 0.] + np.cross(angular, physical[:3, :3] @ right[:3, 3]))
        raw_velocity = np.r_[origin, left[:3, :3] @ angular]
        std = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
        confidence_covariance = min(1., 3. * c.measurement_std_m / max(std[0], std[1]),
                                    3. * c.measurement_std_yaw / std[2])
        confidence_age = max(0., 1. - age / c.prediction_horizon)
        # Signal/support already scale pose prediction and its raw derivative.
        # lambda applies only the remaining uncertainty/age validity factor;
        # multiplying signal/support here as well would count them twice.
        confidence = (confidence_covariance * confidence_age
                      if np.any(prediction_gain > 0.) else 0.)
        reason = self._reason
        if self._rejected:
            pass
        elif age >= c.prediction_horizon:
            reason = "prediction horizon elapsed; velocity suppressed"
        elif self._support < c.min_velocity_samples:
            reason = "warming up relative velocity"
        elif not np.any(confidence_signal > 0.):
            reason = "velocity below measurement confidence"
        anchor = self._raw_poses[self.follower].copy()
        poses = {self.follower: anchor, self.leader: anchor @ relation}
        diagnostics = {"G_control": relation.copy(), "raw_origin_velocity": raw_velocity.copy(),
                       "origin_velocity": confidence * raw_velocity, "state": state.copy(),
                       "state_frame": self.follower + "_physical_base",
                       "velocity_frame": self.follower + "_output_base",
                       "state_order": ["x", "y", "yaw", "vx", "vy", "yaw_rate"],
                       "covariance": covariance.copy(), "lambda": float(confidence), "reason": reason,
                       "support": self._support, "innovation": self._innovation.copy(),
                       "innovation_distance": self._innovation_distance, "capture_age": age,
                       "prediction_dt": effective_dt, "prediction_velocity_decay": decay,
                       "prediction_gain": prediction_gain.copy(),
                       "posterior_state": self._x.copy(),
                       "rejected": self._rejected,
                       "confidence_factors": {"signal": float(max(confidence_signal)),
                                              "signal_xy": float(confidence_signal[0]),
                                              "signal_yaw": float(confidence_signal[1]),
                                              "covariance": confidence_covariance,
                                              "support": confidence_support, "age": confidence_age},
                       "config": asdict(c)}
        return RelativeSnapshot(at=now, base_poses=poses,
            base_twists={name: np.zeros(6) for name in ROBOTS}, relative_pose=inverse(relation),
            relative_twist=np.zeros(6), measurement_age={name: age for name in ROBOTS},
            measurement_times={name: self._capture for name in ROBOTS}, received_times=self._receipts.copy(),
            prediction_dt={name: effective_dt for name in ROBOTS}, relative_filter=diagnostics)
