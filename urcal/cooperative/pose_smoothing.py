"""Robust, continuous application of visual relative-base coordinates.

The camera does not measure robot velocity. This filter changes only the
coordinates used for following: it never changes capture times, grants freshness,
or limits encoder/RTDE leader feedforward. A common rigid camera motion cancels
in the relative transform before filtering. The current leader camera pose is
retained for display and workspace checks.
"""

from collections import deque
from dataclasses import replace
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose


class RelativePoseSmoother:
    def __init__(self, window=5, time_constant=.35, deadband_m=.005,
                 deadband_rad=np.deg2rad(.5), linear_rate=.02,
                 angular_rate=.05, max_dt=.08):
        if (not isinstance(window, Integral) or isinstance(window, bool)
                or window < 3 or window > 31 or window % 2 != 1):
            raise ValueError("base pose window must be an odd integer from 3 to 31")
        for name, value in (("time_constant", time_constant), ("linear_rate", linear_rate),
                            ("angular_rate", angular_rate), ("max_dt", max_dt)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
            setattr(self, name, float(value))
        for name, value in (("deadband_m", deadband_m), ("deadband_rad", deadband_rad)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
            setattr(self, name, float(value))
        self.window = int(window)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.window)
        self.current = self.target = None
        self.last_at = self.last_capture = None
        self.last_times = None
        self.pending_m = self.pending_rad = 0.

    def _center(self):
        # A medoid is one actual pose, not an elementwise matrix median. Its
        # distances are invariant under a rigid change of the leader axes.
        poses = [pose for _, pose in self.samples]
        points = np.array([pose[:3, 3] for pose in poses])
        rotations = np.array([pose[:3, :3] for pose in poses])
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        distances /= max(self.deadband_m, .005)
        for index, rotation in enumerate(rotations):
            distances[index] += Rotation.from_matrix(
                rotations @ rotation.T).magnitude() / max(self.deadband_rad, np.deg2rad(.5))
        # Oldest wins ties, so a single new sample cannot replace the center.
        return poses[int(np.argmin(distances.sum(axis=1)))].copy()

    @staticmethod
    def _increment(error, deadband, weight, maximum):
        length = np.linalg.norm(error)
        if length <= deadband:
            return np.zeros(3)
        distance = min((length - deadband) * weight, maximum)
        return error * (distance / length)

    def apply(self, snapshot, now, leader):
        if not np.isfinite(now) or (self.last_at is not None and now < self.last_at):
            raise ValueError("base pose smoothing clock moved backwards or is invalid")
        names = tuple(snapshot.base_poses)
        if len(names) != 2 or leader not in names:
            raise ValueError("base pose smoothing requires both robot poses")
        for pose in snapshot.base_poses.values():
            matrix_pose(pose)
        follower = next(name for name in names if name != leader)
        times = tuple(float(snapshot.measurement_times[name]) for name in sorted(names))
        if (not np.isfinite(times).all() or max(times) > now
                or (self.last_times is not None and any(a < b for a, b in zip(times, self.last_times)))):
            raise ValueError("base pose smoothing capture times are invalid or moved backwards")
        relative = inverse(snapshot.base_poses[leader]) @ snapshot.base_poses[follower]
        matrix_pose(relative)
        if self.current is None:
            self.current, self.target = relative.copy(), relative.copy()
        if times != self.last_times:
            capture = min(times)
            # Do not accumulate old evidence over an outage. The caller still
            # owns actual freshness and observation deadlines.
            while self.samples and capture - self.samples[0][0] > 1.:
                self.samples.popleft()
            if not self.samples:
                # Expired evidence must not continue pulling toward the old
                # medoid while the replacement window is still filling.
                self.target = self.current.copy()
            self.samples.append((capture, relative.copy()))
            self.last_capture, self.last_times = capture, times
            if len(self.samples) >= 3:
                self.target = self._center()
        if self.last_at is not None and now > self.last_at:
            # Time spent stopped, in observation or in a slow RPC is not a
            # credit that can be spent as a large one-frame coordinate jump.
            dt = min(float(now - self.last_at), self.max_dt)
            weight = -np.expm1(-dt / self.time_constant)
            dp = self.target[:3, 3] - self.current[:3, 3]
            dr = Rotation.from_matrix(self.target[:3, :3] @ self.current[:3, :3].T).as_rotvec()
            self.current[:3, 3] += self._increment(dp, self.deadband_m, weight, self.linear_rate * dt)
            self.current[:3, :3] = Rotation.from_rotvec(
                self._increment(dr, self.deadband_rad, weight, self.angular_rate * dt)
            ).as_matrix() @ self.current[:3, :3]
        self.last_at = float(now)
        self.pending_m = float(np.linalg.norm(relative[:3, 3] - self.current[:3, 3]))
        self.pending_rad = float(Rotation.from_matrix(
            relative[:3, :3] @ self.current[:3, :3].T).magnitude())
        bases = {leader: snapshot.base_poses[leader].copy()}
        bases[follower] = bases[leader] @ self.current
        return replace(snapshot, base_poses=bases, relative_pose=self.current.copy(),
                       base_twists={name: np.zeros(6) for name in names}, relative_twist=np.zeros(6))

    def diagnostics(self):
        return {"samples": len(self.samples), "window": self.window,
                "pending_m": self.pending_m, "pending_deg": float(np.rad2deg(self.pending_rad)),
                "linear_rate": self.linear_rate, "angular_rate": self.angular_rate,
                "deadband_m": self.deadband_m, "deadband_deg": float(np.rad2deg(self.deadband_rad))}
