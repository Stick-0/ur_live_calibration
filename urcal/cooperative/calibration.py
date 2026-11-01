"""Bounded ACTIVE calibration observation and gradual coordinate replacement.

This object owns no transport. Its observation deadline cannot be refreshed by
new evidence. Blending changes controller coordinates, never camera timestamps
or measured robot velocities; callers retain the normal motion checks.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose
from urcal.control.live_calibration import LiveCalibrationCandidate
from urcal.control.policy import CalibrationUnavailable, ControlFault, ROBOTS


class LiveCalibrationWindow:
    def __init__(self, timeout=2., blend_time=.5, max_shift_m=.15, max_shift_deg=15.):
        self.timeout = self._positive(timeout, "calibration recovery timeout")
        self.blend_time = self._positive(blend_time, "calibration blend time")
        self.candidate = LiveCalibrationCandidate(max_shift_m=max_shift_m, max_shift_deg=max_shift_deg)
        self.linear_rate, self.angular_rate = .10, .35
        self.reacquisitions = 0
        self.reset()

    @staticmethod
    def _positive(value, label):
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(label + " must be finite and positive") from exc
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(label + " must be finite and positive")
        return scalar

    @staticmethod
    def _time(value):
        try:
            now = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ControlFault("live calibration time must be finite and nonnegative") from exc
        if not np.isfinite(now) or now < 0:
            raise ControlFault("live calibration time must be finite and nonnegative")
        return now

    @staticmethod
    def _bases(snapshot):
        if not isinstance(snapshot.base_poses, dict) or set(snapshot.base_poses) != set(ROBOTS):
            raise ControlFault("live calibration requires both base transforms")
        result = {}
        for name, pose in snapshot.base_poses.items():
            try:
                matrix_pose(pose)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ControlFault("live calibration base must be a finite proper transform") from exc
            result[name] = np.asarray(pose, dtype=float).copy()
        return result

    @property
    def waiting(self):
        return self.started_at is not None

    @property
    def blending(self):
        return self._blend_current is not None

    def reset(self):
        self.started_at = None
        self.frozen = None
        self.reason = ""
        self._last_check = None
        self._blend_current = None
        self._blend_final = None
        self._blend_started = self._blend_last_at = None
        self.candidate.reset()

    def check_deadline(self, now):
        now = self._time(now)
        if self._last_check is not None and now < self._last_check:
            raise ControlFault("live calibration time moved backwards")
        self._last_check = now
        if self.waiting and now - self.started_at > self.timeout:
            raise CalibrationUnavailable(
                f"live calibration observation exceeded {self.timeout:g}s: {self.reason}")

    def start(self, snapshot, now, reason):
        self.check_deadline(now)
        if not self.waiting:
            self._bases(snapshot)
            self.frozen = deepcopy(snapshot)
            self.started_at = float(now)
            self._blend_current = None
            self._blend_final = None
            self._blend_started = self._blend_last_at = None
            self.candidate.reset()
        self.reason = str(reason)

    def complete(self, now):
        self.check_deadline(now)
        if not self.waiting:
            raise ControlFault("live calibration completion requires an active observation window")
        self._blend_current = self._bases(self.frozen)
        self._blend_final = None
        self._blend_started = self._blend_last_at = float(now)
        self.started_at = None
        self.reason = "fresh calibration reacquired; blending coordinates"
        self.reacquisitions += 1
        self.candidate.reset()

    @staticmethod
    def _limited(vector, maximum):
        norm = np.linalg.norm(vector)
        return vector * (maximum / norm) if norm > maximum else vector

    def blend(self, snapshot, now, leader):
        now = self._time(now)
        if leader not in ROBOTS:
            raise ControlFault("live calibration leader must be ur5e or ur10e")
        target = self._bases(snapshot)
        if not self.blending:
            if self._blend_final is not None and now <= self._blend_last_at:
                return self._with_bases(snapshot, self._blend_final, leader)
            return deepcopy(snapshot)
        # Preview, freshness and send validation can request the same instant
        # repeatedly. None may consume the transition twice or turn time back.
        if now > self._blend_last_at:
            dt = now - self._blend_last_at
            remaining = max(0., self.blend_time - (self._blend_last_at - self._blend_started))
            weight = dt / max(remaining, dt)
            for name in ROBOTS:
                current = self._blend_current[name]
                dp = (target[name][:3, 3] - current[:3, 3]) * weight
                dr = Rotation.from_matrix(target[name][:3, :3] @ current[:3, :3].T).as_rotvec() * weight
                current[:3, 3] += self._limited(dp, self.linear_rate * dt)
                current[:3, :3] = Rotation.from_rotvec(
                    self._limited(dr, self.angular_rate * dt)).as_matrix() @ current[:3, :3]
            self._blend_last_at = now
        bases = deepcopy(self._blend_current)
        result = self._with_bases(snapshot, bases, leader)
        reached = all(np.linalg.norm(target[name][:3, 3] - bases[name][:3, 3]) <= 1e-10
                      and Rotation.from_matrix(target[name][:3, :3] @ bases[name][:3, :3].T).magnitude() <= 1e-10
                      for name in ROBOTS)
        if reached and now >= self._blend_started + self.blend_time:
            self._blend_final = deepcopy(bases)
            self._blend_current = None
            self.reason = "live calibration coordinates restored"
        return result

    @staticmethod
    def _with_bases(snapshot, bases, leader):
        follower = next(name for name in ROBOTS if name != leader)
        return deepcopy(replace(snapshot, base_poses=bases,
                                base_twists={name: np.zeros(6) for name in ROBOTS},
                                relative_pose=inverse(bases[leader]) @ bases[follower],
                                relative_twist=np.zeros(6)))

    def status(self, now):
        now = self._time(now)
        return {"waiting": self.waiting, "elapsed": max(0., now - self.started_at) if self.waiting else 0.,
                "timeout": self.timeout, "reason": self.reason,
                "candidate_count": self.candidate.count, "candidate_reason": self.candidate.reason,
                "blending": self.blending,
                "blend_elapsed": max(0., now - self._blend_started) if self.blending else 0.,
                "reacquisitions": self.reacquisitions}
