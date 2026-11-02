"""Robust pose acquisition for the stationary phases around initial alignment.

This is downstream of PnP. It is deliberately not a live moving-base filter:
the caller must leave it once tracking starts. Rejected images never refresh
the selected capture time, and all members of a cluster agree pairwise.
"""

from collections import deque
from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse
from urcal.control.policy import CalibrationFrame, CalibrationTracker, ControlFault, ROBOTS


class StationaryCalibrationAcquisition:
    def __init__(self, template, window=1.0, fraction=.6):
        if not np.isfinite(window) or window <= 0 or not .5 < fraction <= 1:
            raise ValueError("invalid stationary acquisition window/quorum")
        self.settings = {key: getattr(template, key) for key in (
            "max_age", "stable_frames", "alpha", "max_jump_m", "max_jump_rad",
            "max_mean_px", "max_pixel_px", "min_points", "check_reprojection", "check_individual_jumps")}
        self.window, self.fraction = float(window), float(fraction)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=max(32, 2*self.settings["stable_frames"]))
        self.sequence = self.capture = self.now = None
        self.frames = []
        self.result = None
        self.reason = "collecting stationary calibration"

    def _distance(self, a, b):
        poses_a = [a[name] for name in ROBOTS]
        poses_b = [b[name] for name in ROBOTS]
        poses_a.append(inverse(poses_a[0]) @ poses_a[1])
        poses_b.append(inverse(poses_b[0]) @ poses_b[1])
        distance = 0.
        for left, right in zip(poses_a, poses_b):
            position = np.linalg.norm(left[:3, 3] - right[:3, 3]) / self.settings["max_jump_m"]
            angle = Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude() / self.settings["max_jump_rad"]
            distance = max(distance, position, angle)
        return distance

    def offer(self, frame, now):
        """Return a fresh consensus tracker, or None without erasing good samples.

        At least stable_frames and 60% of the recent observations must agree.
        A bad frame counts against that quorum; duplicates never add evidence.
        The returned capture time belongs to the newest actual inlier image.
        """
        if not np.isfinite(now) or (self.now is not None and now < self.now):
            raise ControlFault("stationary acquisition clock moved backwards or is invalid")
        self.now = now
        if not isinstance(frame, CalibrationFrame):
            raise ControlFault("expected a CalibrationFrame")
        seq = frame.sequence
        if not isinstance(seq, Integral) or isinstance(seq, bool) or seq < 0:
            raise ControlFault("invalid calibration sequence")
        if self.sequence is not None and seq == self.sequence:
            if self.result is not None:
                try:
                    self.result.check_fresh(now)
                except ControlFault as exc:
                    self.reason = str(exc)
                    return None
            return deepcopy(self.result)
        if self.sequence is not None and seq < self.sequence:
            self.reason = "out-of-order calibration"
            return None
        self.sequence = int(seq)
        self.frames = []
        self.result = None
        validator = CalibrationTracker(**self.settings)
        try:
            validator.update(frame, now)
            if self.capture is not None and frame.captured_at <= self.capture:
                raise ControlFault("calibration capture time did not increase")
            self.capture = frame.captured_at
            sample = (deepcopy(frame), now)
        except ControlFault as exc:
            self.reason = str(exc)
            sample = None
        self.samples.append((now, sample))
        while self.samples and now - self.samples[0][0] > self.window:
            self.samples.popleft()
        valid = [entry for _, entry in self.samples if entry is not None
                 and now - entry[0].captured_at <= self.window]
        if not valid:
            return None
        n = len(valid)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                distances[i, j] = distances[j, i] = self._distance(valid[i][0].poses, valid[j][0].poses)
        # Search bounded pairwise clusters, not chains that can drift across
        # the whole scene. Prefer the newest cluster when support ties.
        clusters = []
        for seed in range(n):
            indices = [seed]
            for j in np.argsort(distances[seed], kind="stable"):
                if j != seed and all(distances[j, i] <= 1. for i in indices):
                    indices.append(int(j))
            clusters.append(sorted(indices))
        selected = max(clusters, key=lambda ids: (len(ids), valid[ids[-1]][0].captured_at))
        self.frames = [valid[i] for i in selected]
        count, total = len(selected), len(self.samples)
        self.reason = (f"stationary calibration consensus {count}/{total} images "
                       f"(need {self.settings['stable_frames']}, {self.fraction:.0%} agreement)")
        if count < self.settings["stable_frames"] or count < self.fraction * total:
            return None
        latest, receipt = self.frames[-1]
        try:
            validator._check_age(latest.captured_at, now)
        except ControlFault as exc:
            self.reason = str(exc)
            return None
        poses = {}
        for name in ROBOTS:
            members = [entry[0].poses[name] for entry in self.frames]
            pose = np.eye(4)
            pose[:3, 3] = np.mean([p[:3, 3] for p in members], axis=0)
            pose[:3, :3] = Rotation.from_matrix(np.array([p[:3, :3] for p in members])).mean().as_matrix()
            poses[name] = pose
        result = CalibrationTracker(**self.settings)
        result.update(latest, receipt)
        result._poses = deepcopy(poses)
        # Preserve the actual last raw image for the next live jump check.
        # The consensus is a state estimate, not a replacement observation.
        result._last_sequence = self.sequence
        result.stable_count = count
        self.result = deepcopy(result)
        return result


def stationary_estimator(factory, tracker, mappings, now):
    """Seed zero velocity at rest; do not differentiate acquisition noise."""
    estimator = factory()
    estimator.initialize_stationary_pair(
        {name: pose @ mappings[name] for name, pose in tracker._raw_poses.items()},
        {name: pose @ mappings[name] for name, pose in tracker.poses.items()},
        tracker.last_capture, now, tracker._last_sequence)
    return estimator
