"""Fresh paired calibration candidates during a bounded live observation wait.

This helper collects evidence only. It does not command motion, extend a
capture timestamp, modify the accepted tracker, or authorize resumption. The
caller must handle deceleration, loss timeout, and bounded pose blending.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, rotation_distance
from urcal.control.policy import (
    CalibrationFrame, CalibrationTracker, ControlFault, ROBOTS, _finite_scalar,
    _transform,
)


class LiveCalibrationCandidate:
    """Find a coherent replacement without requiring a return to the old pose.

    Every member must satisfy the normal input quality/time rules, the normal
    pairwise jump limits within the candidate cluster, and separate replacement
    limits relative to the captured accepted baseline. Both individual robot
    transforms and their relative transform are checked. A single distant or
    alternating estimate cannot become a replacement by refreshing timestamps.
    """

    def __init__(self, min_frames=3, min_span=.15, max_shift_m=.15, max_shift_deg=15.):
        if not isinstance(min_frames, Integral) or isinstance(min_frames, bool) or min_frames < 3:
            raise ValueError("live calibration candidate requires at least three frames")
        for name, value in (("min_span", min_span), ("max_shift_m", max_shift_m),
                            ("max_shift_deg", max_shift_deg)):
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(name + " must be finite and positive") from exc
            if not np.isfinite(scalar) or scalar <= 0:
                raise ValueError(name + " must be finite and positive")
            setattr(self, name, scalar)
        self.min_frames = int(min_frames)
        self.max_shift_rad = np.deg2rad(self.max_shift_deg)
        self.reset()

    def reset(self):
        self._samples = []
        self._reference = None
        self._reference_capture = None
        self._options = None
        self._last_sequence = None
        self._last_capture = None
        self._last_now = None
        self.reason = "collecting fresh live calibration"

    @property
    def count(self):
        return len(self._samples)

    @property
    def samples(self):
        """Independent (actual frame, actual receipt time) candidate members."""
        return deepcopy(self._samples)

    @staticmethod
    def _settings(template):
        return {name: getattr(template, name) for name in (
            "max_age", "stable_frames", "alpha", "max_jump_m", "max_jump_rad",
            "max_mean_px", "max_pixel_px", "min_points", "check_reprojection", "check_individual_jumps")}

    @staticmethod
    def _check_poses(reference, candidate, translation_limit, rotation_limit, label):
        comparisons = [(name, reference[name], candidate[name]) for name in ROBOTS]
        comparisons.append(("relative base", inverse(reference[ROBOTS[0]]) @ reference[ROBOTS[1]],
                            inverse(candidate[ROBOTS[0]]) @ candidate[ROBOTS[1]]))
        for name, before, after in comparisons:
            distance = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
            angle = rotation_distance(before, after)
            if distance > translation_limit or angle > rotation_limit:
                raise ControlFault(
                    f"{name} {label}: {distance:.4f}m, {np.rad2deg(angle):.2f}deg "
                    f"(limits {translation_limit:.4f}m, {np.rad2deg(rotation_limit):.2f}deg)")

    def _clear_cluster(self, reason):
        # Observation watermarks survive a rejected payload. A corrected copy
        # of that same image/sequence cannot count as new supporting evidence.
        self._samples = []
        self.reason = str(reason)

    def offer(self, frame, now, template):
        """Return a fresh independent ready tracker, or None with ``reason``.

        Collection starts immediately, including while the old baseline is
        still fresh. A coherent new pose can replace a rejected old baseline;
        neither stale nor duplicated observations refresh its evidence. The
        result retains the latest real raw pair/capture, with an averaged pose
        estimate over the qualifying fresh cluster.
        """
        try:
            now = _finite_scalar(now, "current time")
            if now < 0 or (self._last_now is not None and now < self._last_now):
                raise ControlFault("live calibration candidate time moved backwards or is invalid")
            self._last_now = now
            if not isinstance(template, CalibrationTracker) or template.last_capture is None:
                raise ControlFault("live candidate requires an accepted calibration baseline")
            options = self._settings(template)
            reference = {name: _transform(template._raw_poses[name], name + " accepted baseline")
                         for name in ROBOTS}
            capture = _finite_scalar(template.last_capture, "accepted capture time")
            if capture > now:
                raise ControlFault("accepted calibration capture time is in the future")
            changed = (self._reference is None or options != self._options
                       or capture != self._reference_capture
                       or any(not np.array_equal(reference[name], self._reference[name]) for name in ROBOTS))
            if changed:
                self._clear_cluster("accepted baseline changed; collecting fresh live calibration")
                self._reference, self._reference_capture, self._options = reference, capture, options
            if not isinstance(frame, CalibrationFrame):
                raise ControlFault("expected a CalibrationFrame")
            sequence = frame.sequence
            if not isinstance(sequence, Integral) or isinstance(sequence, bool) or sequence < 0:
                raise ControlFault("calibration sequence must be a nonnegative integer")
            if (template._last_sequence is not None and sequence < template._last_sequence):
                raise ControlFault("calibration sequence precedes accepted tracker observations")
            if self._last_sequence is not None:
                if sequence < self._last_sequence:
                    raise ControlFault("calibration sequence moved backwards")
                if sequence == self._last_sequence:
                    if self._samples and now - self._samples[-1][0].captured_at > options["max_age"]:
                        self._clear_cluster("live calibration candidate is stale")
                    else:
                        self.reason = "duplicate calibration image; waiting for new evidence"
                    return None
            self._last_sequence = int(sequence)
            captured_at = _finite_scalar(frame.captured_at, "capture time")
            validator = CalibrationTracker(**options)
            validator._check_age(captured_at, now)
            if (captured_at <= capture
                    or (self._last_capture is not None and captured_at <= self._last_capture)):
                raise ControlFault("calibration capture time did not increase")
            previous_capture = self._last_capture
            self._last_capture = captured_at
            # This fresh validator checks input quality, but deliberately has
            # no old-pose jump reference. That separate replacement bound is
            # checked below against the original accepted raw pair.
            validator.update(frame, now)
            self._check_poses(self._reference, validator._raw_poses,
                              self.max_shift_m, self.max_shift_rad, "replacement shift")
            restart_reason = ""
            if (previous_capture is not None and captured_at - previous_capture > options["max_age"]):
                restart_reason = "live calibration capture gap; collecting a new cluster"
                self._clear_cluster(restart_reason)
            # Every retained member must itself still be fresh. This prevents
            # slow delivery from using ancient support plus one recent image.
            self._samples = [entry for entry in self._samples
                             if now - entry[0].captured_at <= options["max_age"]]
            try:
                for member, _receipt in self._samples:
                    self._check_poses(member.poses, validator._raw_poses,
                                      options["max_jump_m"], options["max_jump_rad"], "candidate cluster jump")
            except ControlFault as exc:
                restart_reason = str(exc) + "; collecting a new cluster"
                self._clear_cluster(restart_reason)
            self._samples.append((deepcopy(frame), now))
            needed = max(self.min_frames, options["stable_frames"])
            # Bounded storage covers camera-rate delivery; retain a spread of
            # actual captures if a much faster source supplies surplus frames.
            capacity = max(64, 2 * needed)
            if len(self._samples) > capacity:
                del self._samples[1]
            span = captured_at - self._samples[0][0].captured_at
            progress = (f"live calibration candidate {self.count}/{needed} frames, "
                        f"span {span:.3f}/{self.min_span:.3f}s")
            self.reason = (restart_reason + "; " if restart_reason else "") + progress
            if self.count < needed or span + 1e-12 < self.min_span:
                return None
            poses = {}
            for name in ROBOTS:
                members = [entry[0].poses[name] for entry in self._samples]
                pose = np.eye(4)
                pose[:3, 3] = np.mean([value[:3, 3] for value in members], axis=0)
                pose[:3, :3] = Rotation.from_matrix(np.asarray([value[:3, :3] for value in members])).mean().as_matrix()
                poses[name] = pose
            self._check_poses(self._reference, poses, self.max_shift_m, self.max_shift_rad,
                              "averaged replacement shift")
            for member, _receipt in self._samples:
                self._check_poses(member.poses, poses, options["max_jump_m"], options["max_jump_rad"],
                                  "averaged candidate cluster jump")
            # The latest validator already owns the actual capture and raw
            # observation. Only its filtered estimate and evidence count are
            # replaced; timestamps and input quality are never fabricated.
            validator._poses = deepcopy(poses)
            validator.stable_count = self.count
            validator.check_fresh(now)
            self.reason = "fresh live calibration candidate ready"
            return validator
        except (ControlFault, ValueError, TypeError, KeyError, AttributeError) as exc:
            self._clear_cluster(str(exc))
            return None
