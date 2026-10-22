"""Idle native preflight may reacquire vision, never silently recapture a robot."""

from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.test_cooperative_controller import ROBOTS, packet, scene
from urcal.control.policy import ControlFault
from urcal.control.startup import (
    AlignmentCalibrationPending, AlignmentCalibrationTimeout, prepare_alignment,
    preflight_calibration_bases,
)


class ReacquisitionRig:
    """Real paired-calibration gate, Cartesian FK, and no hardware transport."""

    def __init__(self, delivery="fresh"):
        self.engine, self.bases, self.states = scene(stable_frames=5)
        self.gate = self.engine.gate
        for sequence, at in enumerate((9.80, 9.84, 9.88, 9.92, 9.96), 1):
            self.gate.update(packet(self.engine, self.bases, at, sequence), at)
        self.original_bases, self.original_states = deepcopy(self.bases), deepcopy(self.states)
        self.current_bases, self.current_states = deepcopy(self.bases), deepcopy(self.states)
        self.now = 10.70  # The native connection took longer than the .5s image age.
        self.delivery, self.sequence, self.last_delivery = delivery, 5, -np.inf
        self.duplicate, self.stabilities, self.waits, self.reports = None, [], [], []
        self.stopped = False
        self.on_wait = self.on_solve = self.on_path = self.on_watchdog = None
        self.reads = 0
        self.alignment = SimpleNamespace(
            samples=[{name: state.joints.copy() for name, state in self.states.items()} for _ in range(2)],
            models=self.engine.models, guard=self.engine.guard, command_speed_fraction=.9,
        )
        self.factory = Mock(return_value=self.alignment)
        self.session = Mock()
        self.session.solve_joint_targets.side_effect = self.solve
        self.session.validate_joint_path.side_effect = self.path
        self.session.enable_watchdog.side_effect = self.watchdog

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        if self.on_wait is not None:
            self.on_wait()

    def read_current(self):
        self.reads += 1
        if self.stopped:
            raise ControlFault("operator stop during initial alignment")
        current = deepcopy(self.current_states)
        for state in current.values():
            state.robot_timestamp = self.now + 50.
        if self.delivery != "none" and self.now > self.last_delivery:
            self.last_delivery = self.now
            if self.delivery == "duplicate" and self.duplicate is not None:
                frame = self.duplicate
            else:
                self.sequence += 1
                observed_bases = deepcopy(self.current_bases)
                if self.delivery == "rejected":
                    observed_bases["ur5e"][0, 3] += .5149
                frame = packet(self.engine, observed_bases, self.now, self.sequence)
                self.duplicate = frame
            try:
                self.gate.update(frame, self.now)
            except ControlFault:
                # Runtime consumes the rejected pair, then the shared helper
                # reports pending vision with fresh encoders and accepted bases.
                pass
            self.stabilities.append(self.gate.stable_count)
        bases = preflight_calibration_bases(
            self.gate, self.engine.mappings, current, self.now)
        return bases, current

    def solve(self, goals, seeds, check_fresh, *, report=None):
        if self.on_solve is not None:
            self.on_solve()
        check_fresh()
        return {name: self.states[name].joints.copy() for name in goals}

    def path(self, samples, check_fresh, *, report=None):
        if self.on_path is not None:
            self.on_path()
        check_fresh()
        return [{"ur5e": self.states["ur5e"].tcp.copy()} for _ in samples]

    def watchdog(self, check_fresh):
        if self.on_watchdog is not None:
            self.on_watchdog()
        check_fresh()

    def prepare(self):
        with patch.dict(sys.modules, {"urcal.control.alignment": SimpleNamespace(LiveAlignment=self.factory)}):
            return prepare_alignment(
                self.session, self.engine.args, self.engine.offsets, self.bases, self.states,
                {"ur5e": self.states["ur5e"].tcp.copy()}, self.read_current, self.reports.append,
                wait=self.wait, clock=lambda: self.now,
            )


class AlignmentCalibrationReacquisitionTests(unittest.TestCase):
    def assert_no_motion(self, rig):
        rig.session.send.assert_not_called()
        rig.session.send_joints.assert_not_called()

    def test_capture_gap_requires_five_new_pairs_before_native_checks_without_recapturing_start(self):
        rig = ReacquisitionRig()
        self.assertTrue(rig.gate.ready)
        stable_at_ik = []
        rig.on_solve = lambda: stable_at_ik.append(rig.gate.stable_count)
        self.assertIs(rig.prepare(), rig.alignment)
        self.assertEqual(rig.stabilities, [1, 2, 3, 4, 5])
        self.assertEqual(stable_at_ik, [5])
        self.assertAlmostEqual(sum(rig.waits), .08)
        rig.session.enable_watchdog.assert_called_once()
        self.assertTrue(any("calibration stable" in message for message in rig.reports))
        self.assert_no_motion(rig)
        for name in ROBOTS:
            np.testing.assert_array_equal(rig.bases[name], rig.original_bases[name])
            np.testing.assert_array_equal(rig.states[name].joints, rig.original_states[name].joints)
            np.testing.assert_array_equal(rig.states[name].tcp, rig.original_states[name].tcp)

    def test_repeated_packet_cannot_replace_five_independent_accepted_frames(self):
        rig = ReacquisitionRig(delivery="duplicate")
        with self.assertRaises(ControlFault):
            rig.prepare()
        self.assertEqual(rig.stabilities[0], 1)
        self.assertEqual(set(rig.stabilities), {0, 1})
        self.assertAlmostEqual(sum(rig.waits), 5., places=8)
        self.assertEqual(rig.sequence, 6)
        self.assertFalse(rig.gate.ready)
        rig.session.solve_joint_targets.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_rejected_pair_during_native_path_waits_for_five_new_pairs_without_recapturing_target(self):
        rig = ReacquisitionRig()
        path_entry_stability = []

        def reject_at_path_start():
            path_entry_stability.append(rig.gate.stable_count)
            rig.now += .02
            rig.delivery = "rejected"

            def restore_vision():
                rig.delivery = "fresh"
                # A small accepted change may recover readiness but must not
                # replace the Arm snapshot or the already validated target.
                rig.current_bases["ur5e"][0, 3] = rig.bases["ur5e"][0, 3] + .002

            rig.on_wait = restore_vision

        rig.on_path = reject_at_path_start
        self.assertIs(rig.prepare(), rig.alignment)
        self.assertEqual(path_entry_stability, [5])
        self.assertEqual(rig.stabilities, [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(sum(rig.waits), .18)
        self.assertTrue(any("PREFLIGHT WAIT: rejected camera pair: ur5e calibration jump" in report
                            for report in rig.reports))
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_called_once()
        rig.session.enable_watchdog.assert_called_once()
        factory_bases, factory_states, factory_targets = rig.factory.call_args.args[2:5]
        for name in ROBOTS:
            np.testing.assert_array_equal(factory_bases[name], rig.original_bases[name])
            np.testing.assert_array_equal(factory_states[name].tcp, rig.original_states[name].tcp)
        np.testing.assert_array_equal(factory_targets["ur5e"], rig.original_states["ur5e"].tcp)
        np.testing.assert_array_equal(
            rig.session.solve_joint_targets.call_args.args[0]["ur5e"], factory_targets["ur5e"])
        self.assertAlmostEqual(rig.gate.poses["ur5e"][0, 3], rig.current_bases["ur5e"][0, 3])
        self.assert_no_motion(rig)

    def test_duplicate_recovery_packet_after_rejection_cannot_complete_stability(self):
        rig = ReacquisitionRig()
        path_waits = []

        def reject_at_path_start():
            rig.now += .02
            rig.delivery = "rejected"

            def deliver_one_new_pair_then_duplicates():
                path_waits.append(rig.now)
                rig.delivery = "fresh" if len(path_waits) == 1 else "duplicate"

            rig.on_wait = deliver_one_new_pair_then_duplicates

        rig.on_path = reject_at_path_start
        with self.assertRaises(AlignmentCalibrationTimeout):
            rig.prepare()
        self.assertEqual(rig.stabilities[:7], [1, 2, 3, 4, 5, 0, 1])
        self.assertEqual(set(rig.stabilities[6:]), {0, 1})
        self.assertEqual(rig.sequence, 12)  # Five fresh, one rejected, one recovered.
        self.assertAlmostEqual(rig.gate.last_capture, path_waits[0])
        self.assertAlmostEqual(sum(rig.waits), 5., places=8)
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_called_once()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_repeated_rejected_pairs_share_the_five_second_budget_across_native_callbacks(self):
        rig = ReacquisitionRig()

        def reject_during_ik():
            rig.now += .02
            rig.delivery = "rejected"
            release_at = rig.now + 1.

            def restore_later():
                if rig.now >= release_at:
                    rig.delivery = "fresh"

            rig.on_wait = restore_later

        def reject_during_path():
            rig.now += .02
            rig.delivery = "rejected"
            rig.on_wait = None

        rig.on_solve, rig.on_path = reject_during_ik, reject_during_path
        with self.assertRaises(AlignmentCalibrationTimeout) as caught:
            rig.prepare()
        self.assertIn("calibration jump", str(caught.exception))
        self.assertAlmostEqual(sum(rig.waits), 5., places=8)
        self.assertLessEqual(max(rig.waits), .02)
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_called_once()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_missing_calibration_times_out_with_bounded_wait_and_no_watchdog_or_motion(self):
        rig = ReacquisitionRig(delivery="none")
        with self.assertRaises(AlignmentCalibrationTimeout) as caught:
            rig.prepare()
        self.assertEqual(caught.exception.category, "arm_calibration")
        self.assertIn("No motion started", str(caught.exception))
        self.assertAlmostEqual(sum(rig.waits), 5., places=9)
        self.assertLessEqual(max(rig.waits), .02)
        rig.session.solve_joint_targets.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_wait_budget_is_shared_across_callbacks_and_excludes_native_query_time(self):
        rig = ReacquisitionRig(delivery="none")

        def release_vision():
            if rig.now >= 12.70:
                rig.delivery = "fresh"

        rig.on_wait = release_vision

        def slow_native_query_then_missing_vision():
            rig.now += 3.  # Controller computation is outside the wait budget.
            rig.delivery = "none"
            rig.on_wait = None

        rig.on_solve = slow_native_query_then_missing_vision
        with self.assertRaises(AlignmentCalibrationTimeout):
            rig.prepare()
        self.assertAlmostEqual(sum(rig.waits), 5., places=8)
        self.assertAlmostEqual(rig.now - 10.70, 8., places=8)
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_pending_vision_cannot_hide_either_arm_velocity_pose_drift_or_operator_stop(self):
        for name in ROBOTS:
            for change in ("joint_velocity", "tcp_velocity", "joint_position", "tcp_position", "stop"):
                with self.subTest(robot=name, change=change):
                    rig = ReacquisitionRig(delivery="none")

                    def change_after_one_wait():
                        state = rig.current_states[name]
                        if change == "joint_velocity":
                            state.joint_velocity[0] = .006
                        elif change == "tcp_velocity":
                            state.speed[0] = .003
                        elif change == "joint_position":
                            state.joints[0] += .011
                        elif change == "tcp_position":
                            state.tcp[0, 3] += .004
                        else:
                            rig.stopped = True

                    rig.on_wait = change_after_one_wait
                    with self.assertRaises(ControlFault) as caught:
                        rig.prepare()
                    self.assertNotIsInstance(caught.exception, AlignmentCalibrationTimeout)
                    self.assertEqual(len(rig.waits), 1)
                    rig.session.solve_joint_targets.assert_not_called()
                    rig.session.enable_watchdog.assert_not_called()
                    self.assert_no_motion(rig)

    def test_accepted_but_unstable_base_drift_is_compared_to_original_arm_snapshot(self):
        rig = ReacquisitionRig()

        def drift():
            # Each 4mm observation is below the 15mm pair jump threshold;
            # accumulated displacement must still fail against the Arm pose.
            rig.current_bases["ur5e"][0, 3] += .004

        rig.on_wait = drift
        with self.assertRaisesRegex(ControlFault, "base moved during initial alignment"):
            rig.prepare()
        self.assertEqual(rig.stabilities, [1, 2, 3, 4])
        self.assertEqual(rig.gate.last_rejection, "")
        rig.session.solve_joint_targets.assert_not_called()
        self.assert_no_motion(rig)

    def test_rejected_vision_cannot_hide_either_arm_velocity_pose_drift_or_operator_stop(self):
        for name in ROBOTS:
            for change in ("joint_velocity", "tcp_velocity", "joint_position", "tcp_position", "stop"):
                with self.subTest(robot=name, change=change):
                    rig = ReacquisitionRig(delivery="rejected")

                    def change_after_one_wait():
                        state = rig.current_states[name]
                        if change == "joint_velocity":
                            state.joint_velocity[0] = .006
                        elif change == "tcp_velocity":
                            state.speed[0] = .003
                        elif change == "joint_position":
                            state.joints[0] += .011
                        elif change == "tcp_position":
                            state.tcp[0, 3] += .004
                        else:
                            rig.stopped = True

                    rig.on_wait = change_after_one_wait
                    with self.assertRaises(ControlFault) as caught:
                        rig.prepare()
                    self.assertNotIsInstance(caught.exception, AlignmentCalibrationPending)
                    self.assertNotIsInstance(caught.exception, AlignmentCalibrationTimeout)
                    self.assertEqual(len(rig.waits), 1)
                    self.assertIn("calibration jump", rig.gate.last_rejection)
                    rig.session.solve_joint_targets.assert_not_called()
                    rig.session.enable_watchdog.assert_not_called()
                    self.assert_no_motion(rig)

    def test_rejected_pair_keeps_last_accepted_base_drift_visible_to_snapshot_check(self):
        rig = ReacquisitionRig(delivery="rejected")
        rig.current_bases["ur5e"][0, 3] += .012
        rig.sequence += 1
        rig.gate.update(packet(rig.engine, rig.current_bases, 10.68, rig.sequence), 10.68)
        # This 12mm accepted change is below the pair jump threshold but above
        # the 10mm total preflight drift bound. A later rejection cannot hide it.
        with self.assertRaisesRegex(ControlFault, "base moved during initial alignment preflight"):
            rig.prepare()
        self.assertIn("calibration jump", rig.gate.last_rejection)
        self.assertEqual(rig.waits, [])
        rig.session.solve_joint_targets.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_watchdog_callback_never_waits_for_calibration_even_after_successful_reacquisition(self):
        rig = ReacquisitionRig()
        waits_before_watchdog = []

        def expire_at_watchdog_start():
            waits_before_watchdog.append(len(rig.waits))
            rig.now += .6
            rig.delivery = "none"

        rig.on_watchdog = expire_at_watchdog_start
        with self.assertRaises(AlignmentCalibrationPending):
            rig.prepare()
        self.assertEqual(waits_before_watchdog, [len(rig.waits)])
        self.assertAlmostEqual(sum(rig.waits), .08)
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_called_once()
        rig.session.enable_watchdog.assert_called_once()
        self.assert_no_motion(rig)

    def test_rejection_at_watchdog_never_reenters_preflight_wait(self):
        rig = ReacquisitionRig()
        waits_before_watchdog = []

        def reject_at_watchdog_start():
            waits_before_watchdog.append(len(rig.waits))
            rig.now += .02
            rig.delivery = "rejected"

        rig.on_watchdog = reject_at_watchdog_start
        with self.assertRaisesRegex(AlignmentCalibrationPending, "calibration jump"):
            rig.prepare()
        self.assertEqual(waits_before_watchdog, [len(rig.waits)])
        self.assertAlmostEqual(sum(rig.waits), .08)
        rig.session.solve_joint_targets.assert_called_once()
        rig.session.validate_joint_path.assert_called_once()
        rig.session.enable_watchdog.assert_called_once()
        self.assert_no_motion(rig)


if __name__ == "__main__":
    unittest.main()
