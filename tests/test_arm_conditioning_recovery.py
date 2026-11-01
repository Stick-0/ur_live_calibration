"""A refused pre-motion IK candidate leaves preview usable, never auto-arms."""

from copy import deepcopy
import unittest

import numpy as np

from urcal.control.policy import ControlFault
from urcal.control.runtime import ArmConditioningUnavailable, take_latest
import tests.test_live_alignment_runtime as fixtures


def rejected_conditioning():
    return ArmConditioningUnavailable({
        "position_tolerance_m": .05, "rotation_tolerance_deg": 10., "max_attempts": 3,
        "attempts": [{"attempt": index + 1, "conditioning_scale": scale,
                      "native_joints": {"ur5e": [0.] * 6},
                      "captured_min_singular_value": .008,
                      "live_min_singular_value": .009, "outcome": "conditioning_rejected"}
                     for index, scale in enumerate((2., 4., 6.))]})


class ArmConditioningRecoveryTests(unittest.TestCase):
    def test_fault_copies_candidate_diagnostics_and_names_the_existing_allowance(self):
        original = rejected_conditioning().details
        expected = deepcopy(original)
        fault = ArmConditioningUnavailable(original)
        original["attempts"][0]["native_joints"]["ur5e"][0] = 1.
        original["attempts"].append({"attempt": 4})
        self.assertEqual(fault.details, expected)
        self.assertEqual(fault.category, "arm_conditioning")
        self.assertIn("50mm / 10deg", str(fault))
        self.assertIn("3 candidate checks", str(fault))
        self.assertIn("No motion started", str(fault))
        self.assertIn("press Arm again", str(fault))

    def test_follow_refusal_cancels_queued_arm_and_requires_fresh_explicit_retry(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                fault = rejected_conditioning()
                later = {1: [("arm", 10.)]}
                if retry:
                    later[2] = [("arm", 10.08)]
                result = fixtures.LiveAlignmentRuntimeTests().run_fake(
                    plan_error=fault, plan_error_once=True, preflight_commands=("arm", "arm"),
                    later_commands=later, stop_after=4)
                self.assertIsNone(result.caught)
                self.assertEqual(result.plan.call_count, 2 if retry else 1)
                self.assertEqual(result.motion_factory.call_count, 2 if retry else 1)
                np.testing.assert_allclose(result.factory_attempts, [10., 10.08] if retry else [10.])
                result.session.enable_watchdog.assert_not_called()
                result.session.send.assert_not_called()
                self.assertEqual(result.session.send_joints.call_count, int(retry))
                self.assertEqual(result.alignment.commit.call_count, int(retry))
                self.assertGreaterEqual(result.session.close.call_count, 1)
                record = next(item for item in result.records
                              if item["kind"] == "wait" and item["category"] == "arm_conditioning")
                self.assertEqual(record["details"], fault.details)
                status = take_latest(result.gui_statuses)
                if retry:
                    self.assertNotIn("ARM BLOCKED", status["message"])
                    self.assertLess(result.order.index("motion_close"), result.order.index("speedJ"))
                else:
                    self.assertTrue(status["ready"])
                    self.assertFalse(status["active"])
                    self.assertEqual(status["fault_category"], "arm_conditioning")
                    self.assertIn("ARM BLOCKED", status["message"])

    def test_rejected_preflight_preserves_follow_target_edits_and_stop(self):
        case = fixtures.LiveAlignmentRuntimeTests()
        target = {"type": "target", "gap": .35}
        result = case.run_fake(plan_error=rejected_conditioning(), plan_error_once=True,
                               preflight_commands=("arm", target, "arm", "stop"), stop_after=8)
        self.assertIsNone(result.caught)
        result.plan.assert_called_once()
        result.motion_factory.assert_called_once()
        result.session.enable_watchdog.assert_not_called()
        result.session.send.assert_not_called()
        result.session.send_joints.assert_not_called()
        result.alignment.commit.assert_not_called()
        self.assertLess(result.stop.waits, 8)
        result.engine.apply_target.assert_called_once_with(target)

    def test_recoverable_exception_type_during_active_follow_still_stops(self):
        result = fixtures.LiveAlignmentRuntimeTests().run_fake(send_error=rejected_conditioning())
        self.assertIsInstance(result.caught, ControlFault)
        result.alignment.commit.assert_not_called()
        result.engine.commit.assert_not_called()
        result.session.send.assert_not_called()
        self.assertNotIn("speedJ", result.order)
        self.assertIn("motion_close", result.order)

if __name__ == "__main__":
    unittest.main()
