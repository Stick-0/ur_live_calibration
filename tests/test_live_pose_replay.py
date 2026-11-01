"""Offline camera-coordinate replay; never imports or commands RTDE.

This is not a closed-loop robot simulation: recorded vision depends on the
original moving arm. It checks how the revised controller applies those same
measurements to its coordinate target, without claiming new physical accuracy.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_cooperative_controller import settings
from urcal.control.geometry import inverse
from urcal.control.policy import CalibrationFrame, ControlFault
from urcal.cooperative.controller import CooperativeController


FIXTURE = Path(__file__).parent / "fixtures" / "calibration_depth_switching_20261031.json"


def matrix(rows):
    result = np.eye(4)
    result[:3] = rows
    return result


def variation(points):
    steps = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    return {"travel": float(np.sum(steps)), "step_p95": float(np.percentile(steps, 95)),
            "step_max": float(np.max(steps))}


def replay(data, *, window=5):
    """Use production gate, observation deadlines, state and coordinate filter."""
    engine = CooperativeController(settings(
        initial_align=True, calibration_acquisition="consensus", stable_frames=3,
        base_motion_mode="pose", leader_response="bounded", live_calibration_recovery=True,
        filter_alpha=.2, target_tolerance_m=.05, target_tolerance_deg=10.,
        max_calibration_jump_m=.05, max_calibration_jump_deg=5.,
        max_calibration_age=.5, max_invalid_duration=.3, base_pose_window=window,
        base_pose_smoothing_time=.35, base_pose_deadband_m=.005,
        base_pose_deadband_deg=.5, base_pose_update_speed=.02,
        base_pose_update_angular_speed=.05))
    offset = 10.

    def frame(row):
        return CalibrationFrame(
            sequence=row["sequence"], captured_at=offset + row["captured"],
            poses={name: matrix(pose) @ inverse(engine.mappings[name])
                   for name, pose in row["bases"].items()},
            raw_mean_px=deepcopy(row["mean_px"]), raw_max_px=deepcopy(row["max_px"]),
            valid_counts=deepcopy(row["counts"]))

    for row in data["initial_measurements"]:
        engine.ingest(frame(row), offset + row["at"], allow_reacquire=True)
    engine.state(offset, active=True)
    target_in_leader_base = matrix(data["leader_tcp"]) @ matrix(data["relative_target"])
    measurements = iter(data["measurements"])
    next_measurement = next(measurements, None)
    goals, at, captures = [], [], []
    observation_starts = set()
    braking = 0
    fault = None

    def remember_observation():
        if engine.live_calibration.waiting:
            observation_starts.add(engine.live_calibration.started_at)

    for control in data["controls"]:
        now = offset + control["at"]
        try:
            while next_measurement is not None and next_measurement["at"] <= control["at"]:
                engine.ingest(frame(next_measurement), offset + next_measurement["at"],
                              active_recovery=True)
                remember_observation()
                next_measurement = next(measurements, None)
            if engine.check_live_calibration(now):
                snapshot = engine.state(now, active=True)
                # Filtered geometry may lag, but freshness remains the real
                # latest accepted image and is never renewed by interpolation.
                for name in snapshot.measurement_times:
                    assert snapshot.measurement_times[name] == engine.gate.last_capture
                    assert abs(snapshot.measurement_age[name]
                               - (now - engine.gate.last_capture)) < 1e-9
                    np.testing.assert_array_equal(snapshot.base_twists[name], np.zeros(6))
                captures.append(engine.gate.last_capture - offset)
            else:
                remember_observation()
                braking += 1
                snapshot = engine.live_calibration.frozen
            goals.append((inverse(snapshot.relative_pose) @ target_in_leader_base)[:3, 3])
            at.append(control["at"])
        except ControlFault as exc:
            fault = str(exc)
            break
    return {"goals": np.asarray(goals), "at": np.asarray(at), "captures": captures,
            "observations": len(observation_starts), "braking": braking, "fault": fault}


class LivePoseReplayTests(unittest.TestCase):
    def test_recorded_depth_switching_reduces_coordinate_target_variation(self):
        data = json.loads(FIXTURE.read_text())
        filtered = replay(data)
        original = np.asarray([row["goal"] for row in data["controls"]])
        self.assertIsNone(filtered["fault"])
        self.assertEqual(len(filtered["goals"]), len(original))
        before, after = variation(original), variation(filtered["goals"])
        self.assertLess(after["travel"], .7 * before["travel"])
        self.assertLess(after["step_p95"], .5 * before["step_p95"])
        self.assertLess(after["step_max"], .5 * before["step_max"])
        self.assertLess(filtered["observations"], data["original_observation_count"])
        # No regression may obtain smoothness by freezing every measurement.
        self.assertGreater(len(set(filtered["captures"])), 100)
        self.assertGreater(np.linalg.norm(filtered["goals"][-1] - filtered["goals"][0]), .005)


if __name__ == "__main__":
    unittest.main()
