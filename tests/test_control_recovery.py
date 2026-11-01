"""Recovery never substitutes a preview for confirmed robot stopping."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from urcal.control.handoff import CalibrationHandoff
from urcal.control.policy import (
    ActiveCalibrationGuard, CalibrationFrame, CalibrationTracker,
    CalibrationUnavailable, ControlFault, PlanningUnavailable,
)
from urcal.control.recovery import (
    RecoveryStopGate, fresh_calibration_tracker, is_recoverable_fault,
)
from urcal.control.runtime import MotionSession
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.task import MotionInfeasible


def frame(sequence=1, captured_at=10.):
    return CalibrationFrame(sequence, captured_at,
                            {name: np.eye(4) for name in ("ur5e", "ur10e")},
                            {name: 1. for name in ("ur5e", "ur10e")},
                            {name: 2. for name in ("ur5e", "ur10e")},
                            {name: 7 for name in ("ur5e", "ur10e")})


def states(stamp=50., **changes):
    result = {}
    for name in ("ur5e", "ur10e"):
        state = SimpleNamespace(tcp=np.eye(4), joints=np.zeros(6), speed=np.zeros(6),
                                joint_velocity=np.zeros(6), robot_timestamp=stamp)
        for key, value in changes.items():
            setattr(state, key, value)
        result[name] = state
    return result


class RecoveryClassificationTests(unittest.TestCase):
    def test_only_known_planning_and_measurement_faults_are_recoverable(self):
        for category in ("measurement_unavailable", "measurement_expired", "measurement_invalid",
                         "tracking_error", "tracking_infeasible", "transition_infeasible",
                         "compensation_infeasible", "reachability_infeasible", "singularity",
                         "workspace", "clearance", "workspace_infeasible", "clearance_infeasible"):
            self.assertTrue(is_recoverable_fault(ResearchFault(category, "reason")), category)
        self.assertTrue(is_recoverable_fault(CalibrationUnavailable("stale")))
        self.assertTrue(is_recoverable_fault(PlanningUnavailable("target unavailable")))
        for category in ("joint_speed", "tcp_speed", "robot_state", "robot_state_expired",
                         "process_exit", "operator_stop", "control_deadline", "frame_mismatch",
                         "unknown", "timing"):
            self.assertFalse(is_recoverable_fault(ResearchFault(category, "reason")), category)
        for exc in (ControlFault("calibration is stale"), RuntimeError("workspace"),
                    ValueError("target invalid"), KeyboardInterrupt()):
            self.assertFalse(is_recoverable_fault(exc))

    def test_governor_planning_failures_can_recover_but_hardware_categories_cannot(self):
        for category in ("compensation_infeasible", "transition_infeasible",
                         "tracking_infeasible", "singularity"):
            with self.subTest(category=category):
                self.assertTrue(is_recoverable_fault(MotionInfeasible("reason", category)))
        for category in ("joint_speed", "tcp_speed", "robot_state", "control_deadline",
                         "operator_stop", "timing", "unknown"):
            with self.subTest(category=category):
                self.assertFalse(is_recoverable_fault(MotionInfeasible("reason", category)))

    def test_a_planning_category_on_an_unrecognized_exception_does_not_enable_recovery(self):
        for exc in (ControlFault("planner failed"), RuntimeError("planner failed")):
            exc.category = "compensation_infeasible"
            self.assertFalse(is_recoverable_fault(exc))

    def test_quality_faults_typed_and_malformed_inputs_remain_fatal(self):
        empty = CalibrationTracker(stable_frames=1)
        with self.assertRaises(CalibrationUnavailable):
            empty.check_fresh(10.)
        empty.update(frame(), 10.)
        with self.assertRaises(CalibrationUnavailable):
            empty.check_fresh(10.51)
        for changes in ({"captured_at": float("nan")}, {"captured_at": 11.},
                        {"sequence": -1}, {"poses": {}}):
            packet = frame()
            packet = CalibrationFrame(**{**vars(packet), **changes})
            with self.assertRaises(ControlFault) as context:
                CalibrationTracker().update(packet, 10.)
            self.assertFalse(is_recoverable_fault(context.exception))
        packet = frame()
        packet.poses["ur5e"][:3, 3] = [.1, 0., 0.]
        tracker = CalibrationTracker(stable_frames=1)
        tracker.update(frame(), 10.)
        packet = CalibrationFrame(**{**vars(packet), "sequence": 2, "captured_at": 10.01})
        with self.assertRaises(CalibrationUnavailable):
            tracker.update(packet, 10.01)

    def test_only_valid_insufficient_keypoint_counts_can_recover(self):
        for count, recoverable in ((3, True), (0, True), (-1, False), (8, False), (True, False), (2.5, False)):
            packet = frame()
            packet.valid_counts["ur5e"] = count
            with self.assertRaises(ControlFault) as context:
                CalibrationTracker().update(packet, 10.)
            self.assertEqual(is_recoverable_fault(context.exception), recoverable)

    def test_active_grace_limit_is_unchanged_and_requires_new_arm(self):
        tracker = CalibrationTracker(stable_frames=1, max_age=1.)
        tracker.update(frame(), 10.)
        guard = ActiveCalibrationGuard(max_invalid_duration=.3)
        guard.arm(tracker, 10.)
        tracker.reject(2, "jump")
        self.assertFalse(guard.check(tracker, 10.1))
        self.assertFalse(guard.check(tracker, 10.39))
        with self.assertRaises(CalibrationUnavailable):
            guard.check(tracker, 10.401)
        self.assertFalse(guard.armed)
        with self.assertRaisesRegex(ControlFault, "explicit Arm"):
            guard.check(tracker, 10.402)

    def test_handoff_timeout_is_recoverable_but_bad_clock_is_not(self):
        handoff = CalibrationHandoff(CalibrationTracker(), 10., timeout=5.)
        with self.assertRaises(CalibrationUnavailable):
            handoff.check_timeout(15.01)
        with self.assertRaises(ControlFault) as context:
            handoff.check_timeout(9.)
        self.assertFalse(is_recoverable_fault(context.exception))

    def test_fresh_tracker_discards_all_old_history_preserving_limits(self):
        previous = CalibrationTracker(stable_frames=1, check_reprojection=False, max_age=.6)
        previous.update(frame(), 10.)
        fresh = fresh_calibration_tracker(previous)
        self.assertFalse(fresh.ready)
        self.assertIsNone(fresh.last_capture)
        self.assertEqual(fresh.poses, {})
        self.assertEqual(fresh.max_age, .6)
        self.assertFalse(fresh.check_reprojection)
        self.assertTrue(previous.ready)


class RecoveryStopGateTests(unittest.TestCase):
    def gate(self, **kwargs):
        return RecoveryStopGate(("ur5e", "ur10e"), states(), 10., **kwargs)

    def test_advancing_stationary_samples_are_required_for_dwell(self):
        gate = self.gate()
        self.assertFalse(gate.observe(states(), 10.01))
        for t in (10.04, 10.08, 10.12, 10.16, 10.20):
            self.assertFalse(gate.observe(states(40. + t), t))
        self.assertTrue(gate.observe(states(50.24), 10.24))
        self.assertEqual(gate.ready_at, 10.24)
        self.assertTrue(gate.observe(states(50.28), 10.28))
        self.assertEqual(gate.ready_at, 10.24)

    def test_one_frozen_robot_cannot_confirm_stop(self):
        gate = self.gate()
        for t in (10.04, 10.08, 10.12):
            samples = states(40. + t)
            samples["ur10e"].robot_timestamp = 50.
            self.assertFalse(gate.observe(samples, t))
        samples = states(50.16)
        samples["ur10e"].robot_timestamp = 50.
        with self.assertRaisesRegex(ControlFault, "stopped updating"):
            gate.observe(samples, 10.16)

    def test_reported_tcp_or_joint_motion_restarts_dwell(self):
        for key, value in (("speed", [.002, 0, 0, 0, 0, 0]),
                           ("speed", [0, 0, 0, .01, 0, 0]),
                           ("joint_velocity", [.02, 0, 0, 0, 0, 0])):
            gate = self.gate()
            gate.observe(states(50.04), 10.04)
            self.assertFalse(gate.observe(states(50.08, **{key: value}), 10.08))
            for step in range(3, 8):
                self.assertFalse(gate.observe(states(50. + .04 * step), 10. + .04 * step))
            self.assertTrue(gate.observe(states(50.32), 10.32))

    def test_joint_and_tcp_changes_detect_motion_despite_zero_velocity(self):
        for kind in ("joints", "position", "rotation"):
            gate = self.gate()
            samples = states(50.04)
            if kind == "joints":
                samples["ur5e"].joints[0] = .001
            elif kind == "position":
                samples["ur5e"].tcp[0, 3] = .001
            else:
                from scipy.spatial.transform import Rotation
                samples["ur5e"].tcp[:3, :3] = Rotation.from_rotvec([.001, 0., 0.]).as_matrix()
            self.assertFalse(gate.observe(samples, 10.04))
            self.assertIn("become stationary", gate.reason)

    def test_motion_after_ready_revokes_gate_and_changes_ready_timestamp(self):
        gate = self.gate(stationary_duration=.08)
        for index in (1, 2, 3):
            gate.observe(states(50. + .04 * index), 10. + .04 * index)
        self.assertTrue(gate.ready)
        self.assertFalse(gate.observe(states(50.16, speed=[.01, 0, 0, 0, 0, 0]), 10.16))
        self.assertIsNone(gate.ready_at)
        for index in (5, 6, 7):
            gate.observe(states(50. + .04 * index), 10. + .04 * index)
        self.assertTrue(gate.ready)
        self.assertAlmostEqual(gate.ready_at, 10.28)

    def test_invalid_or_backwards_state_is_fatal(self):
        for change in ({"robot_timestamp": 49.}, {"robot_timestamp": float("nan")},
                       {"speed": [0.] * 5}, {"joints": [float("nan")] * 6},
                       {"tcp": np.zeros((4, 4))}, {"joint_velocity": [0.] * 5}):
            with self.assertRaises(ControlFault) as context:
                self.gate().observe(states(**change), 10.04)
            self.assertFalse(is_recoverable_fault(context.exception))
        with self.assertRaises(ControlFault):
            self.gate().observe({}, 10.04)
        with self.assertRaises(ControlFault):
            self.gate().observe(states(50.04), 9.)

    def test_hold_reader_without_qdot_still_uses_joint_positions(self):
        gate = self.gate(stationary_duration=.04)
        self.assertFalse(gate.observe(states(50.04, joint_velocity=None), 10.04))
        self.assertTrue(gate.observe(states(50.08, joint_velocity=None), 10.08))

    def test_stop_that_never_becomes_stationary_has_fatal_deadline(self):
        gate = self.gate(confirmation_timeout=.3)
        for index in range(1, 8):
            self.assertFalse(gate.observe(states(50. + .04 * index, speed=[.002, 0, 0, 0, 0, 0]),
                                          10. + .04 * index))
        with self.assertRaisesRegex(ControlFault, "not confirmed within") as context:
            gate.observe(states(50.32), 10.32)
        self.assertFalse(is_recoverable_fault(context.exception))

    def test_manual_reposition_after_confirmed_stop_can_wait_for_rest_again(self):
        gate = self.gate(stationary_duration=.04, confirmation_timeout=.3)
        gate.observe(states(50.04), 10.04)
        self.assertTrue(gate.observe(states(50.08), 10.08))
        for index in range(3, 12):
            self.assertFalse(gate.observe(states(50. + .04 * index, speed=[.002, 0, 0, 0, 0, 0]),
                                          10. + .04 * index))
        self.assertFalse(gate.observe(states(50.48), 10.48))
        self.assertTrue(gate.observe(states(50.52), 10.52))


class AlignmentRecoveryClassificationTests(unittest.TestCase):
    def test_blocked_predicted_geometry_recovers_but_observed_geometry_is_fatal(self):
        from test_live_alignment import fixture, make_plan
        from urcal.control.alignment import AlignmentFault, AlignmentPlanningFault
        values = fixture("follow")
        plan = make_plan(values)
        plant = values[2]
        observed = plant.states(0.)
        joints = {name: state.joints for name, state in observed.items()}
        with patch.object(plan.guard, "check", return_value=(False, "blocked corridor", {})):
            with self.assertRaises(AlignmentPlanningFault) as context:
                plan._check_configuration(joints, plant.bases)
            self.assertTrue(is_recoverable_fault(context.exception))
            with self.assertRaises(AlignmentFault) as context:
                plan._check_configuration(joints, plant.bases, observed=observed)
            self.assertFalse(is_recoverable_fault(context.exception))

    def test_exhausted_bounded_candidate_search_is_recoverable_without_commit(self):
        from test_live_alignment import fixture, make_plan
        from urcal.control.alignment import AlignmentFault, AlignmentPlanningFault
        values = fixture("follow")
        plan = make_plan(values)
        failure = AlignmentFault("predicted alignment TCP acceleration exceeds limit",
                                 {"arm": "ur5e", "metric": "linear_acceleration"})
        with patch.object(plan, "_predicted_motion", side_effect=failure) as predict:
            with self.assertRaises(AlignmentPlanningFault) as context:
                plan.propose(values[2].states(0.), values[2].bases, .04)
        self.assertGreater(predict.call_count, 1)
        self.assertTrue(is_recoverable_fault(context.exception))
        self.assertIn("search_failures", context.exception.details)
        self.assertEqual(plan.elapsed, 0.)
        self.assertEqual(plan.progress, 0.)

    def test_measured_overspeed_encoder_mismatch_and_bad_period_remain_fatal(self):
        from test_live_alignment import fixture, make_plan
        from urcal.control.alignment import AlignmentFault
        values = fixture("follow")
        plan = make_plan(values)
        for field, value in (("speed", [1., 0, 0, 0, 0, 0]),
                             ("joint_velocity", [10., 0, 0, 0, 0, 0])):
            observed = values[2].states(0.)
            setattr(observed["ur5e"], field, np.asarray(value))
            with self.assertRaisesRegex(AlignmentFault, "measured speed") as context:
                plan.propose(observed, values[2].bases, .04)
            self.assertFalse(is_recoverable_fault(context.exception))
        observed = values[2].states(0.)
        observed["ur5e"].tcp[0, 3] += 1.
        with self.assertRaisesRegex(AlignmentFault, "kinematics") as context:
            plan.propose(observed, values[2].bases, .04)
        self.assertFalse(is_recoverable_fault(context.exception))
        with self.assertRaisesRegex(AlignmentFault, "control period") as context:
            plan.propose(values[2].states(0.), values[2].bases, -1.)
        self.assertFalse(is_recoverable_fault(context.exception))

    def test_fixed_leader_motion_remains_fatal_but_endpoint_timeout_recovers(self):
        from test_live_alignment import fixture, make_plan
        from urcal.control.alignment import AlignmentFault, AlignmentPlanningFault
        values = fixture("follow")
        plan = make_plan(values)
        observed = values[2].states(0.)
        observed["ur10e"].joints[0] += .02
        observed["ur10e"].tcp = plan.models["ur10e"].pose_and_jacobian(observed["ur10e"].joints)[0]
        with self.assertRaisesRegex(AlignmentFault, "fixed leader moved") as context:
            plan.propose(observed, values[2].bases, .04)
        self.assertFalse(is_recoverable_fault(context.exception))
        plan.elapsed = plan.timeout + .1
        with self.assertRaisesRegex(AlignmentPlanningFault, "timed out") as context:
            plan.propose(values[2].states(0.), values[2].bases, .04)
        self.assertTrue(is_recoverable_fault(context.exception))


class RecoveryTransportTests(unittest.TestCase):
    def session(self):
        args = SimpleNamespace(ur5e_ip="test5", ur10e_ip="test10", watchdog_hz=10.,
                               stop_accel=.1, max_joint_accel=.5)
        controls = {name: Mock() for name in ("ur5e", "ur10e")}
        for control in controls.values():
            for method in ("reuploadScript", "isProgramRunning", "isConnected", "setWatchdog",
                           "speedStop", "stopScript"):
                getattr(control, method).return_value = True
            control.getTCPOffset.return_value = [0.] * 6
        session = MotionSession(args, tuple(controls), {name: np.eye(4) for name in controls},
                                factory=lambda ip: controls["ur5e" if ip == "test5" else "ur10e"])
        return session, controls

    def test_confirmed_stops_release_both_arms_and_correct_stop_modes(self):
        session, controls = self.session()
        session._motion_modes["ur5e"] = "joint"
        self.assertIsNone(session.stop_for_recovery())
        controls["ur5e"].speedStop.assert_called_once_with(.5)
        controls["ur10e"].speedStop.assert_called_once_with(.1)
        for control in controls.values():
            control.stopScript.assert_called_once()
            control.disconnect.assert_called_once()
        self.assertFalse(session.controls)
        self.assertFalse(session._watchdog_enabled)
        session.close()  # Final cleanup remains harmless and idempotent.

    def test_unconfirmed_stop_still_attempts_every_stop_and_disconnect(self):
        for method in ("speedStop", "stopScript"):
            for failure in (False, None, RuntimeError("transport unavailable")):
                if method == "stopScript" and failure is None:
                    continue  # The installed SDK's stopScript API is void.
                session, controls = self.session()
                operation = getattr(controls["ur5e"], method)
                if isinstance(failure, BaseException):
                    operation.side_effect = failure
                else:
                    operation.return_value = failure
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ControlFault, "could not be confirmed") as context:
                    session.stop_for_recovery()
                self.assertFalse(is_recoverable_fault(context.exception))
                self.assertFalse(session.controls)
                for control in controls.values():
                    control.speedStop.assert_called_once()
                    control.stopScript.assert_called_once()
                    control.disconnect.assert_called_once()

    def test_void_script_shutdown_matches_installed_sdk_contract(self):
        session, controls = self.session()
        for control in controls.values():
            control.stopScript.return_value = None
            control.disconnect.return_value = None
        self.assertIsNone(session.stop_for_recovery())
        self.assertFalse(session.controls)

    def test_stop_requests_precede_script_shutdown(self):
        session, controls = self.session()
        events = []
        for name, control in controls.items():
            control.speedStop.side_effect = lambda *args, n=name: events.append((n, "stop")) or True
            control.stopScript.side_effect = lambda n=name: events.append((n, "script")) or True
        session.stop_for_recovery()
        self.assertEqual([event[1] for event in events], ["stop", "stop", "script", "script"])

    def test_cleanup_exception_or_already_closed_session_cannot_enable_recovery(self):
        session, controls = self.session()
        controls["ur5e"].disconnect.side_effect = RuntimeError("disconnect error")
        with redirect_stdout(io.StringIO()), self.assertRaises(ControlFault):
            session.stop_for_recovery()
        with self.assertRaisesRegex(ControlFault, "session is closed"):
            session.stop_for_recovery()


if __name__ == "__main__":
    unittest.main()
