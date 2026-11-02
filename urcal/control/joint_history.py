"""Bounded, exposure-time joint interpolation using receive-only RTDE samples.

Robot/host clock alignment uses first-observation receive intervals. It is an
approximation, not hardware synchronization or a known bound on network latency.
Cached robot timestamps never refresh a sample or its host clock observation.
"""

from collections import deque
import threading
import time

import numpy as np


class JointAlignmentUnavailable(ValueError):
    """This exposure has no sufficiently fresh, close pair of joint samples."""


class JointHistory:
    def __init__(self, maxlen=512):
        self.samples = deque(maxlen=maxlen)
        self.offset = None

    def append(self, stamp, joints, before, after):
        stamp, before, after = float(stamp), float(before), float(after)
        joints = np.asarray(joints, dtype=float)
        if (not np.isfinite([stamp, before, after]).all() or stamp < 0 or after < before
                or joints.shape != (6,) or not np.isfinite(joints).all()):
            raise ValueError("invalid timestamped RTDE joints")
        if self.samples and stamp < self.samples[-1][0]:
            raise ValueError("RTDE timestamp moved backwards")
        if self.samples and after < self.samples[-1][3]:
            raise ValueError("joint receive host clock moved backwards")
        if self.samples and stamp == self.samples[-1][0]:
            return False
        midpoint = (before + after) / 2.
        observed_offset = midpoint - stamp
        self.offset = observed_offset if self.offset is None else min(self.offset, observed_offset)
        self.samples.append((stamp, joints.copy(), before, after))
        return True

    def diagnostics(self, now):
        details = {"buffer_samples": len(self.samples), "robot_to_host_offset_s": self.offset}
        if self.samples:
            low, high = self.samples[0], self.samples[-1]
            details.update(available_robot_timestamps_s=[low[0], high[0]],
                           available_host_times_s=[low[0] + self.offset, high[0] + self.offset],
                           latest_first_received_at=high[3], last_unique_sample_age_s=now - high[3])
        return details

    def interpolate(self, captured, now, max_gap, max_age):
        if (not np.isfinite([captured, now, max_gap, max_age]).all()
                or max_gap <= 0 or max_age <= 0 or captured > now):
            raise JointAlignmentUnavailable("invalid exposure or interpolation limits")
        if not self.samples:
            raise JointAlignmentUnavailable("no timestamped joint samples")
        if now - self.samples[-1][3] > max_age:
            raise JointAlignmentUnavailable("joint history is stale")
        requested = captured - self.offset
        stamps = np.fromiter((row[0] for row in self.samples), dtype=float)
        right = int(np.searchsorted(stamps, requested))
        # A sample exactly at the exposure is already synchronized; no temporal
        # extrapolation or replacement with an arbitrary latest sample occurs.
        if right < len(stamps) and abs(stamps[right] - requested) <= 1e-9:
            left = right
        elif right > 0 and abs(stamps[right - 1] - requested) <= 1e-9:
            left = right = right - 1
        elif right == 0 or right == len(stamps):
            raise JointAlignmentUnavailable("exposure has no bracketing joint samples")
        else:
            left = right - 1
        low, high = self.samples[left], self.samples[right]
        gap = high[0] - low[0]
        mapped = [low[0] + self.offset, high[0] + self.offset]
        if gap > max_gap + 1e-12:
            raise JointAlignmentUnavailable(f"joint interpolation gap {gap:.4f}s exceeds {max_gap:.4f}s")
        if now - mapped[0] > max_age:
            raise JointAlignmentUnavailable("exposure joint bracket is stale")
        fraction = 0. if gap == 0 else float(np.clip((requested - low[0]) / gap, 0., 1.))
        delta = np.arctan2(np.sin(high[1] - low[1]), np.cos(high[1] - low[1]))
        joints = low[1] + fraction * delta
        offsets = [(row[2] + row[3]) / 2. - row[0] for row in self.samples]
        return joints, {
            "robot_timestamp_s": float(requested), "bracket_robot_timestamps_s": [low[0], high[0]],
            "bracket_host_times_s": mapped, "bracket_received_intervals_s": [[low[2], low[3]], [high[2], high[3]]],
            "bracket_gap_s": float(gap), "fraction": fraction,
            "max_bracket_distance_s": float(max(abs(captured - value) for value in mapped)),
            "robot_to_host_offset_s": float(self.offset),
            "sdk_read_half_interval_s": float(max(low[3] - low[2], high[3] - high[2]) / 2.),
            "observed_arrival_offset_range_s": float(max(offsets) - min(offsets)),
            "latest_first_received_at": float(self.samples[-1][3]),
            "buffer_samples": len(self.samples), "interpolation": "shortest_angular_linear",
            "clock_mapping": "minimum first-observation midpoint offset; unknown constant transport latency",
        }


class JointSampler:
    """One owner of both RTDE receivers; capture thread only reads its history."""

    def __init__(self, receivers, stop, hz=100., max_age=.5, *, clock=time.monotonic):
        if not np.isfinite([hz, max_age]).all() or hz <= 0 or max_age <= 0:
            raise ValueError("joint history rate and maximum age must be positive")
        self.receivers, self.stop_event, self.clock = receivers, stop, clock
        self.period, self.max_age = 1. / hz, max_age
        self.histories = {name: JointHistory() for name in receivers}
        self.last_advance = {name: clock() for name in receivers}
        self.incoherent_samples = {name: 0 for name in receivers}
        self.condition = threading.Condition()
        self.error = None
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="joint-history")
        self.thread.start()

    def _sample_once(self):
        for name, receiver in self.receivers.items():
            if not receiver.isConnected():
                raise RuntimeError(f"{name}: joint history RTDE disconnected")
            coherent = False
            for _ in range(3):
                before = self.clock()
                first = float(receiver.getTimestamp())
                joints = receiver.getActualQ()
                last = float(receiver.getTimestamp())
                after = self.clock()
                if not np.isfinite([first, last]).all() or last < first:
                    raise RuntimeError(f"{name}: invalid joint history RTDE timestamp")
                if first == last:
                    with self.condition:
                        if self.histories[name].append(first, joints, before, after):
                            self.last_advance[name] = after
                        self.condition.notify_all()
                    coherent = True
                    break
                self.incoherent_samples[name] += 1
            if self.clock() - self.last_advance[name] > self.max_age:
                reason = "timestamp stopped advancing" if coherent else "no coherent timestamp/q sample"
                raise RuntimeError(f"{name}: joint history stale ({reason})")

    def _run(self):
        try:
            while not self.stop_event.is_set() and not self.closing.is_set():
                start = self.clock()
                self._sample_once()
                self.closing.wait(max(0., self.period - (self.clock() - start)))
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def interpolate(self, captured, *, wait=.03, max_gap=.05):
        if not np.isfinite(wait) or wait < 0:
            raise ValueError("joint alignment wait must be finite and nonnegative")
        deadline = self.clock() + wait
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError(self.error)
                if self.stop_event.is_set() or self.closing.is_set():
                    raise RuntimeError("joint sampler stopped")
                now = self.clock()
                try:
                    values = [self.histories[name].interpolate(captured, now, max_gap, self.max_age)
                              for name in self.receivers]
                    metadata = {name: value[1] for name, value in zip(self.receivers, values)}
                    for name, item in metadata.items():
                        item["incoherent_samples_discarded"] = self.incoherent_samples[name]
                    return [value[0].tolist() for value in values], metadata
                except JointAlignmentUnavailable as exc:
                    remaining = deadline - now
                    if remaining <= 0:
                        error = JointAlignmentUnavailable(str(exc))
                        error.diagnostics = {name: history.diagnostics(now) for name, history in self.histories.items()}
                        raise error from exc
                    self.condition.wait(min(remaining, .005))

    def close(self):
        self.closing.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=1.5)
