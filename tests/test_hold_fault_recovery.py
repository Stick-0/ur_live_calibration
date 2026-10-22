"""Hold recovery lifecycle with real camera/stop gates and fake transports."""

from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np

import urcal.control.runtime as runtime
from urcal.control.policy import CalibrationFrame, ControlFault, PlanningUnavailable
import tests.test_live_alignment_runtime as lifecycle


class HoldFaultRecoveryTests(unittest.TestCase):
    def run_recovery(self, *, response="preview", **kwargs):
        statuses = []
        configure = lifecycle.config
        publish = runtime.put_latest

        def config(*args, **options):
            result = configure(*args, **options)
            result.fault_response = response
            return result

        def status(queue, item):
            if isinstance(item, dict) and "scene" in item:
                statuses.append(deepcopy(item))
            return publish(queue, item)

        fixture = lifecycle.HoldAlignmentRuntimeTests()
        with patch.object(lifecycle, "config", config), patch.object(runtime, "put_latest", status):
            result = fixture.run_fake(enabled=kwargs.pop("enabled", False), handoff_frames=False, **kwargs)
        result.history = statuses
        return result

    def test_expired_calibration_stops_motion_and_keeps_preview_for_new_arm(self):
        result = self.run_recovery(stop_after=36, calibration_events={i: 0. for i in range(20, 36)})
        self.assertIsNone(result.caught)
        result.session.stop_for_recovery.assert_called_once()
        self.assertEqual(result.motion_factory.call_count, 1)
        self.assertGreater(result.session.send.call_count, 0)
        recovering = [status for status in result.history if status["recovery_required"]]
        self.assertTrue(recovering)
        self.assertTrue(all(not status["active"] for status in recovering))
        self.assertTrue(any(status["ready"] for status in recovering))
        self.assertTrue(any("calibration is stale" in status["recovery_reason"] for status in recovering))
        self.assertTrue(result.planner.target_applied)
        result.planner.apply_target.assert_called_once()
        for origin in result.planner.initial_tcp.values():
            np.testing.assert_array_equal(origin, np.eye(4))

    def test_waiting_or_delayed_arm_clicks_cannot_restart_but_a_new_ready_click_can(self):
        result = self.run_recovery(
            stop_after=36, calibration_events={i: 0. for i in range(20, 36)},
            later_commands={1: ["arm"], 14: [("arm", 10.55)], 23: [("arm", 10.9)],
                            28: [("arm", 10.7)], 30: [("arm", 11.2)]},
        )
        self.assertIsNone(result.caught)
        result.session.stop_for_recovery.assert_called_once()
        self.assertEqual(result.motion_factory.call_count, 2)
        self.assertGreaterEqual(result.factory_attempts[1], 11.2 - 1e-8)
        self.assertTrue(any(status["active"] for status in result.history[-5:]))

    def test_four_post_stop_frames_are_not_a_stable_pair(self):
        result = self.run_recovery(
            stop_after=29, calibration_events={i: 0. for i in range(21, 25)},
            later_commands={1: ["arm"], 26: [("arm", 11.04)]},
        )
        self.assertIsNone(result.caught)
        self.assertEqual(result.motion_factory.call_count, 1)
        self.assertFalse(any(status["ready"] for status in result.history if status["recovery_required"]))

    def test_delayed_inference_from_before_stop_cannot_rebuild_readiness(self):
        transforms = {name: np.eye(4) for name in runtime.ROBOTS}
        events = {
            i: CalibrationFrame(100 + i, 10.4, deepcopy(transforms),
                                {name: 1. for name in runtime.ROBOTS},
                                {name: 2. for name in runtime.ROBOTS},
                                {name: 7 for name in runtime.ROBOTS})
            for i in range(20, 32)
        }
        result = self.run_recovery(stop_after=33, calibration_events=events,
                                   later_commands={1: ["arm"], 30: [("arm", 11.2)]})
        self.assertIsNone(result.caught)
        self.assertEqual(result.motion_factory.call_count, 1)
        self.assertFalse(any(status["ready"] for status in result.history if status["recovery_required"]))

    def test_observed_motion_during_recovery_discards_calibration_cluster(self):
        original = lifecycle.Reader

        class MovingReader(original):
            def read(self, now):
                result = super().read(now)
                if 10.87 < now < 10.89:
                    result.speed[0] = .002
                return result

        with patch.object(lifecycle, "Reader", MovingReader):
            result = self.run_recovery(
                stop_after=33, calibration_events={i: 0. for i in range(20, 28)},
                later_commands={1: ["arm"], 30: [("arm", 11.2)]},
            )
        self.assertIsNone(result.caught)
        self.assertEqual(result.motion_factory.call_count, 1)
        self.assertFalse(any(status["ready"] for status in result.history if status["recovery_required"]))

    def test_planning_failure_stops_before_send_and_requires_new_arm(self):
        injected = False

        def reject(*_):
            nonlocal injected
            if not injected:
                injected = True
                raise PlanningUnavailable("final flange target exceeds the allowed position/orientation error")

        result = self.run_recovery(stop_after=20, fresh_hook=reject,
                                   calibration_events={i: 0. for i in range(4, 20)})
        self.assertIsNone(result.caught)
        self.assertEqual(result.motion_factory.call_count, 1)
        result.session.stop_for_recovery.assert_called_once()
        self.assertNotIn("speedL", result.order)
        self.assertTrue(any(status["ready"] for status in result.history if status["recovery_required"]))

    def test_unknown_controller_fault_is_fatal_and_never_relabels_as_recovery(self):
        def fail(*_):
            raise ControlFault("RTDE receive disconnected")

        result = self.run_recovery(stop_after=20, fresh_hook=fail)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("disconnected", str(result.caught))
        result.session.stop_for_recovery.assert_not_called()
        self.assertNotIn("speedL", result.order)

    def test_alignment_planning_failure_discards_old_path_without_reapplying_target(self):
        def fail(*_):
            raise PlanningUnavailable("initial alignment predicted path outside workspace")

        result = self.run_recovery(enabled=True, stop_after=22, fresh_hook=fail,
                                   calibration_events={i: 0. for i in range(4, 22)})
        self.assertIsNone(result.caught)
        result.session.stop_for_recovery.assert_called_once()
        result.alignment.commit.assert_not_called()
        result.planner.apply_target.assert_called_once()
        self.assertEqual(result.motion_factory.call_count, 1)
        recovering = [status for status in result.history if status["recovery_required"]]
        self.assertTrue(any(status["ready"] for status in recovering))
        self.assertTrue(all(status["alignment_progress"] is None for status in recovering))

    def test_malformed_camera_transform_is_not_treated_as_a_recoverable_outlier(self):
        transforms = {name: np.eye(4) for name in runtime.ROBOTS}
        transforms["ur5e"][:3, :3] = 0.
        malformed = CalibrationFrame(100, 10.079, transforms,
                                     {name: 1. for name in runtime.ROBOTS},
                                     {name: 2. for name in runtime.ROBOTS},
                                     {name: 7 for name in runtime.ROBOTS})
        result = self.run_recovery(stop_after=20, calibration_events={2: malformed})
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("rigid transform", str(result.caught))
        result.session.stop_for_recovery.assert_not_called()

    def test_exit_response_preserves_terminal_calibration_fault(self):
        result = self.run_recovery(response="exit", stop_after=20)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("calibration is stale", str(result.caught))
        result.session.stop_for_recovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
