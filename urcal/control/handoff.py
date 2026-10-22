"""Fresh camera acquisition while an aligned robot remains at rest.

Only images captured after the measured alignment endpoint may initialize the
live controller. The caller keeps sending checked zero joint velocities on its
normal control ticks; this helper never sleeps or issues hardware commands.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np

from urcal.control.policy import CalibrationFrame, CalibrationTracker, CalibrationUnavailable, ControlFault
from urcal.control.acquisition import StationaryCalibrationAcquisition, stationary_estimator


class CalibrationHandoff:
    def __init__(self, tracker_template, started_at, *, timeout=15.,
                 estimator_factory=None, mappings=None, leader="ur10e",
                 consensus=False, wait_at_rest=False):
        if not np.isfinite(started_at) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid calibration handoff timing")
        self.started_at, self.timeout = float(started_at), float(timeout)
        self._settings = {key: getattr(tracker_template, key) for key in (
            "max_age", "stable_frames", "alpha", "max_jump_m", "max_jump_rad",
            "max_mean_px", "max_pixel_px", "min_points", "check_reprojection")}
        self._estimator_factory, self._mappings, self._leader = estimator_factory, mappings, leader
        self._sequence = None
        self._capture = None
        self._validated_sequence = None
        self.acquisition = StationaryCalibrationAcquisition(tracker_template) if consensus else None
        self.wait_at_rest, self.wait_expired = wait_at_rest, False
        self.reason = "waiting for images captured after alignment"
        self._reset_cluster()

    def _reset_cluster(self):
        self.tracker = CalibrationTracker(**self._settings)
        self.estimator = self._estimator_factory() if self._estimator_factory else None
        self.frames = []

    def check_timeout(self, now):
        if not np.isfinite(now) or now < self.started_at:
            raise ControlFault("invalid calibration handoff clock")
        if now - self.started_at > self.timeout:
            self.wait_expired = True
            if self.wait_at_rest:
                return  # Caller continues checked zero commands; never re-aligns here.
            raise CalibrationUnavailable(f"post-alignment calibration wait exceeded {self.timeout:g}s: "
                               + self.reason + "; motion stopped, a new Arm is required")

    @property
    def status(self):
        prefix = "alignment retained; holding position; " if self.wait_expired else ""
        return prefix + self.reason

    def ingest(self, packet, now):
        self.check_timeout(now)
        try:
            if not isinstance(packet, CalibrationFrame):
                raise ControlFault("expected a CalibrationFrame")
            if not np.isfinite(packet.captured_at):
                raise ControlFault("invalid calibration capture time")
            # In-flight inference may arrive later but still belong to alignment.
            if packet.captured_at <= self.started_at:
                return False
            if self.acquisition is not None:
                previous_sequence = self.acquisition.sequence
                candidate = self.acquisition.offer(packet, now)
                self.reason = self.acquisition.reason
                self.frames = self.acquisition.frames
                if candidate is None:
                    self.tracker.reject(packet.sequence, self.reason)
                    return False
                self.tracker = candidate
                if self._estimator_factory is not None:
                    self.estimator = stationary_estimator(
                        self._estimator_factory, candidate, self._mappings, now)
                return previous_sequence != packet.sequence and candidate.last_capture == packet.captured_at
            seq = packet.sequence
            if not isinstance(seq, Integral) or isinstance(seq, bool) or seq < 0:
                raise ControlFault("invalid calibration sequence")
            if self._sequence is not None and seq == self._sequence:
                return False  # duplicates never count or refresh capture time
            if self._sequence is not None and seq < self._sequence:
                raise ControlFault("calibration sequence moved backwards")
            self._sequence = int(seq)
            if self._capture is not None and packet.captured_at <= self._capture:
                raise ControlFault("calibration capture time did not increase")
            if (self.tracker.last_capture is not None
                    and packet.captured_at - self.tracker.last_capture > self.tracker.max_age):
                self._reset_cluster()
            candidate = deepcopy(self.tracker)
            accepted = candidate.update(packet, now)
            estimator = deepcopy(self.estimator)
            if accepted and estimator is not None:
                bases = {name: pose @ self._mappings[name] for name, pose in packet.poses.items()}
                estimator.update_pair(bases, packet.captured_at, now, seq)
                estimator.estimate(now, leader=self._leader)
            if accepted:
                self.tracker, self.estimator = candidate, estimator
                self._capture = packet.captured_at
                self.frames.append((deepcopy(packet), now))
                self.reason = f"collecting fresh calibration ({candidate.stable_count}/{candidate.stable_frames} frames)"
            return accepted
        except (ControlFault, ValueError, TypeError, KeyError) as exc:
            self.reason = str(exc)
            # A rejected sample cannot seed a replacement coordinate baseline.
            self._reset_cluster()
            return False

    def ready(self, now):
        self.check_timeout(now)
        try:
            self.tracker.check_fresh(now)
        except ControlFault as exc:
            if self.frames:
                self.reason = str(exc)
            if self.frames and now - self.frames[-1][0].captured_at > self.tracker.max_age:
                self._reset_cluster()
            return False
        return True

    def try_validate(self, now, validate):
        """Retry temporary validation faults at rest, on a new accepted pair.

        The configured timeout policy applies to retries as well. Callers must
        keep sending checked zero commands and install a trial baseline only
        on success. wait_at_rest never authorizes motion from stale data.
        """
        if not self.ready(now):
            return False
        sequence = self.tracker._last_sequence if self.acquisition is not None else self.frames[-1][0].sequence
        if sequence == self._validated_sequence:
            return False
        self._validated_sequence = sequence
        from urcal.control.recovery import is_recoverable_fault
        from urcal.cooperative.task import MotionInfeasible
        try:
            validate()
        except ControlFault as exc:
            if not (is_recoverable_fault(exc) or isinstance(exc, MotionInfeasible)):
                raise
            self.reason = "waiting for next fresh calibration: " + str(exc)
            return False
        return True
