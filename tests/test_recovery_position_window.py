"""Stationary TCP noise must not repeatedly erase post-stop calibration."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.recovery import RecoveryStopGate
import tests.test_follow_fault_recovery as follow_fixture


FIXTURE = Path(__file__).with_name("fixtures") / "recovery_stationary_pose_noise.json"


def sample(stamp, x=0.):
    tcp = np.eye(4)
    tcp[0, 3] = x
    return {"ur5e": SimpleNamespace(
        tcp=tcp, joints=np.zeros(6), speed=np.zeros(6),
        joint_velocity=np.zeros(6), robot_timestamp=stamp,
    )}


class RecoveryPositionWindowTests(unittest.TestCase):
    def test_recorded_zero_velocity_pose_variations_do_not_revoke_readiness(self):
        recorded = json.loads(FIXTURE.read_text())["samples"]
        positions = np.array([np.array(row["tcp"])[:3, 3] for row in recorded])
        # At 25 Hz, observed tens-of-micrometre position differences can
        # violate the old consecutive-sample 1 mm/s translation check.
        self.assertGreater(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) / .04, .001)
        self.assertLess(np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2)), .0002)

        def observed(row, stamp):
            np.testing.assert_array_equal(row["qd"], np.zeros(6))
            np.testing.assert_array_equal(row["tcp_speed"], np.zeros(6))
            return {"ur5e": SimpleNamespace(
                tcp=np.array(row["tcp"]), joints=np.array(row["q"]),
                speed=np.array(row["tcp_speed"]), joint_velocity=np.array(row["qd"]),
                robot_timestamp=stamp,
            )}

        gate = RecoveryStopGate(("ur5e",), observed(recorded[0], 50.), 10.)
        first_ready = None
        for tick in range(1, 4 * len(recorded)):
            ready = gate.observe(observed(recorded[tick % len(recorded)], 50. + tick * .04),
                                 10. + tick * .04)
            if tick >= 6:
                self.assertTrue(ready, (tick, gate.reason))
                if first_ready is None:
                    first_ready = gate.ready_at
                self.assertEqual(gate.ready_at, first_ready)

    def test_real_position_step_revokes_ready_despite_zero_reported_velocity(self):
        gate = RecoveryStopGate(("ur5e",), sample(50.), 10.)
        for tick in range(1, 7):
            gate.observe(sample(50. + tick * .04), 10. + tick * .04)
        self.assertTrue(gate.ready)
        original_ready = gate.ready_at
        self.assertFalse(gate.observe(sample(50.28, .001), 10.28))
        self.assertIsNone(gate.ready_at)
        # Keep observing the new pose. The movement must age out of the
        # window, then a new stationary dwell must finish.
        for tick in range(8, 20):
            gate.observe(sample(50. + tick * .04, .001), 10. + tick * .04)
        self.assertTrue(gate.ready)
        self.assertGreater(gate.ready_at, original_ready + .2)

    def test_accumulating_small_position_steps_cannot_hide_sustained_drift(self):
        for velocity in (.0011, .002, -.002):
            with self.subTest(velocity=velocity):
                gate = RecoveryStopGate(("ur5e",), sample(50.), 10.)
                for tick in range(1, 100):
                    self.assertFalse(gate.observe(
                        sample(50. + tick * .04, velocity * tick * .04), 10. + tick * .04))

    def test_returning_to_start_does_not_cancel_large_between_sample_excursions(self):
        gate = RecoveryStopGate(("ur5e",), sample(50.), 10.)
        for tick in range(1, 100):
            self.assertFalse(gate.observe(sample(50. + tick * .04, .0003 if tick % 2 else 0.),
                                          10. + tick * .04))

    def test_reported_motion_still_revokes_ready_without_position_change(self):
        for field, value in (("speed", [.002, 0, 0, 0, 0, 0]),
                             ("speed", [0, 0, 0, .006, 0, 0]),
                             ("joint_velocity", [.011, 0, 0, 0, 0, 0])):
            with self.subTest(field=field, value=value):
                gate = RecoveryStopGate(("ur5e",), sample(50.), 10.)
                for tick in range(1, 7):
                    gate.observe(sample(50. + tick * .04), 10. + tick * .04)
                self.assertTrue(gate.ready)
                moving = sample(50.28)
                setattr(moving["ur5e"], field, np.asarray(value))
                self.assertFalse(gate.observe(moving, 10.28))

    def test_follow_recovery_keeps_new_calibration_through_stationary_pose_noise(self):
        original_reader = follow_fixture.RecoveryReader

        class NoisyReader(original_reader):
            def read(self, now):
                state = super().read(now)
                if self.stopped_at is not None:
                    tick = round((now - self.stopped_at) / .04)
                    state.tcp = deepcopy(state.tcp)
                    state.tcp[0, 3] += .00003 if tick % 2 else -.00003
                return state

        case = follow_fixture.FollowFaultRecoveryTests()
        with patch.object(follow_fixture, "RecoveryReader", NoisyReader):
            result = case.run_runtime(stop_after=70)
        self.assertIsNone(result.caught)
        stationary = [row for row in result.records if row["kind"] == "recovery_stationary"]
        self.assertEqual(len(stationary), 1)
        recovery_statuses = [status for _, status in result.statuses if status["recovery_required"]]
        first_ready = next(index for index, status in enumerate(recovery_statuses) if status["ready"])
        self.assertTrue(all(status["ready"] for status in recovery_statuses[first_ready:]))
        self.assertEqual(result.factory.call_count, 1)
        self.assertEqual(result.sent, [])


if __name__ == "__main__":
    unittest.main()
