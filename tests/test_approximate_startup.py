"""Bounded goals use native validation; fresh handoff preserves nominal task."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.test_cooperative_controller import packet, scene
from tests.test_target_approximation import recorded_request
from urcal.control.alignment import CapturedKinematics
from urcal.control.approximation import TargetApproximationUnavailable, flange_adjustment
from urcal.control.geometry import inverse, pose_matrix
from urcal.control.policy import ControlFault
from urcal.control.runtime import ArmIKUnavailable
from urcal.control.startup import check_handoff_snapshot, check_start_snapshot, prepare_alignment
from urcal.cooperative.controller import CooperativeController


class ApproximateStartupTests(unittest.TestCase):
    def rig(self):
        args, offsets, bases, states, targets = recorded_request()
        defaults = vars(scene()[0].args).copy()
        defaults.update(vars(args))
        args = SimpleNamespace(**defaults)
        for state in states.values():
            state.robot_timestamp = 10.
        models = {name: CapturedKinematics(name, state.joints, state.tcp, offsets[name])
                  for name, state in states.items()}
        exact_error = ArmIKUnavailable("ur5e", targets["ur5e"], states["ur5e"].joints, offsets["ur5e"])
        calls = []

        def solve(request, seeds, fresh, *, report=None):
            fresh()
            calls.append(deepcopy(request))
            if len(calls) == 1:
                raise exact_error
            # A robot-free transport stand-in checks that the proposed seed
            # solves precisely the selected TCP before accepting native IK.
            for name, target in request.items():
                np.testing.assert_allclose(models[name].pose_and_jacobian(seeds[name])[0], target, atol=1e-9)
            return deepcopy(seeds)

        def path(samples, fresh, *, report=None):
            fresh()
            return [{name: models[name].pose_and_jacobian(q[name])[0] for name in targets}
                    for q in samples]

        session = Mock()
        session.solve_joint_targets.side_effect = solve
        session.validate_joint_path.side_effect = path
        session.enable_watchdog.side_effect = lambda check_fresh: check_fresh()
        return SimpleNamespace(args=args, offsets=offsets, bases=bases, states=states,
                               targets=targets, session=session, calls=calls, exact_error=exact_error)

    def prepare(self, rig, fresh=None):
        return prepare_alignment(rig.session, rig.args, rig.offsets, rig.bases, rig.states,
                                 rig.targets, fresh or (lambda: (deepcopy(rig.bases), deepcopy(rig.states))))

    def test_recorded_exact_rejection_passes_selected_ik_and_all_81_path_samples(self):
        rig = self.rig()
        original = deepcopy(rig.targets)
        alignment = self.prepare(rig)
        self.assertEqual(len(rig.calls), 2)
        self.assertEqual(len(alignment.samples), 81)
        rig.session.enable_watchdog.assert_called_once()
        rig.session.send.assert_not_called()
        rig.session.send_joints.assert_not_called()
        for name in original:
            np.testing.assert_array_equal(rig.targets[name], original[name])
            np.testing.assert_array_equal(alignment.nominal_targets[name], original[name])
            np.testing.assert_array_equal(rig.calls[1][name], alignment.targets[name])
            distance, angle = flange_adjustment(original[name], alignment.targets[name], rig.offsets[name])
            self.assertLess(distance, .05)
            self.assertLess(angle, np.deg2rad(10))
        self.assertGreater(alignment.target_adjustments["adjustments"]["ur5e"]["position_m"], .01)
        # The same selected endpoint must enter live following without asking
        # exact IK for the original, rejected nominal pose again.
        endpoint = deepcopy(rig.states)
        endpoint["ur5e"].joints = alignment.goal_joints["ur5e"].copy()
        endpoint["ur5e"].tcp = alignment.models["ur5e"].pose_and_jacobian(endpoint["ur5e"].joints)[0]
        engine = CooperativeController(rig.args)
        self.assertTrue(engine.ingest(packet(engine, rig.bases), 10.))
        with patch.object(engine.models["ur5e"], "inverse", side_effect=AssertionError("exact IK during tolerant tracking")):
            proposal = engine.prepare(10., endpoint, .04)
        self.assertTrue(proposal.record["within_target_tolerance"])
        np.testing.assert_allclose(proposal.twists["ur5e"], 0., atol=1e-12)
        np.testing.assert_allclose(proposal.record["target_tcp"], original["ur5e"], atol=1e-10)
        self.assertGreater(proposal.record["relative_error_m"], .01)

    def test_zero_allowances_retain_original_native_rejection(self):
        rig = self.rig()
        rig.args.target_tolerance_m = rig.args.target_tolerance_deg = 0.
        with patch("urcal.control.approximation.select_alignment_candidate") as selector:
            with self.assertRaises(ArmIKUnavailable) as caught:
                self.prepare(rig)
        self.assertIs(caught.exception, rig.exact_error)
        selector.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()

    def test_unavailable_candidate_is_recoverable_but_stop_propagates_unchanged(self):
        for fault in (TargetApproximationUnavailable("no candidate within bounds"), ControlFault("operator stop")):
            with self.subTest(fault=str(fault)):
                rig = self.rig()
                with patch("urcal.control.approximation.select_alignment_candidate", side_effect=fault):
                    with self.assertRaises(ControlFault) as caught:
                        self.prepare(rig)
                if isinstance(fault, TargetApproximationUnavailable):
                    self.assertIs(caught.exception, rig.exact_error)
                    self.assertIn("Bounded target search", str(caught.exception))
                else:
                    self.assertIs(caught.exception, fault)
                rig.session.enable_watchdog.assert_not_called()
                rig.session.validate_joint_path.assert_not_called()

    def test_unbounded_candidate_is_rejected_before_second_native_call(self):
        rig = self.rig()
        candidate = deepcopy(rig.targets)
        candidate["ur5e"][0, 3] += .051
        with patch("urcal.control.approximation.select_alignment_candidate",
                   return_value=(candidate, {"ur5e": rig.states["ur5e"].joints}, {})):
            with self.assertRaisesRegex(ControlFault, "exceeds nominal flange allowance"):
                self.prepare(rig)
        self.assertEqual(len(rig.calls), 1)
        rig.session.enable_watchdog.assert_not_called()


class ApproximateHandoffTests(unittest.TestCase):
    def setUp(self):
        self.controller, self.bases, self.states = scene()
        self.args = self.controller.args
        self.args.target_tolerance_m, self.args.target_tolerance_deg = .05, 10.
        self.args.handoff_max_shift_m, self.args.handoff_max_shift_deg = .05, 10.
        self.offsets = self.controller.offsets
        self.targets = {"ur5e": self.states["ur5e"].tcp.copy()}

    def check(self, bases=None, states=None, live=None):
        return check_handoff_snapshot(
            self.args, self.bases, self.states, self.bases if bases is None else bases,
            self.states if states is None else states, offsets=self.offsets,
            nominal_targets=self.targets, live_targets=self.targets if live is None else live)

    def test_new_stable_estimate_is_bounded_by_nominal_task_effect(self):
        bases, live = deepcopy(self.bases), deepcopy(self.targets)
        bases["ur5e"][0, 3] += .025
        live["ur5e"][0, 3] -= .025
        with self.assertRaisesRegex(ControlFault, "base moved"):
            check_start_snapshot(self.args, self.bases, self.states, bases, self.states)
        self.check(bases=bases, live=live)
        np.testing.assert_array_equal(self.targets["ur5e"], self.states["ur5e"].tcp)

    def test_handoff_over_task_allowance_or_moved_encoders_still_stops(self):
        for change in ("position", "rotation", "joint", "tcp", "speed"):
            states, live = deepcopy(self.states), deepcopy(self.targets)
            if change == "position":
                live["ur5e"][0, 3] += .051
            elif change == "rotation":
                live["ur5e"] = live["ur5e"] @ pose_matrix([0, 0, 0, 0, 0, np.deg2rad(11)])
            elif change == "joint":
                states["ur10e"].joints[0] += .011
            elif change == "tcp":
                states["ur10e"].tcp[0, 3] += .004
            else:
                states["ur10e"].joint_velocity[0] = self.args.align_leader_max_joint_speed + .001
            with self.subTest(change=change), self.assertRaises(ControlFault):
                self.check(states=states, live=live)

    def test_long_tcp_does_not_turn_allowed_flange_rotation_into_position_rejection(self):
        self.offsets["ur5e"] = pose_matrix([.5, .2, .1, .2, -.1, .3])
        flange = self.targets["ur5e"] @ inverse(self.offsets["ur5e"])
        live = {"ur5e": flange @ pose_matrix([0, 0, 0, 0, 0, np.deg2rad(9)]) @ self.offsets["ur5e"]}
        self.assertGreater(np.linalg.norm(live["ur5e"][:3, 3] - self.targets["ur5e"][:3, 3]), .05)
        self.check(live=live)

    def test_zero_tolerances_keep_old_strict_base_handoff(self):
        self.args.target_tolerance_m = self.args.target_tolerance_deg = 0.
        self.args.handoff_max_shift_m = self.args.handoff_max_shift_deg = 0.
        bases = deepcopy(self.bases)
        bases["ur5e"][0, 3] += .02
        with self.assertRaisesRegex(ControlFault, "base calibration changed"):
            self.check(bases=bases)

    def test_one_zero_component_has_native_numerical_floor_not_bit_equality(self):
        self.args.target_tolerance_m = 0.
        self.args.handoff_max_shift_m = 0.
        live = deepcopy(self.targets)
        live["ur5e"][0, 3] += .0004
        self.check(live=live)
        live["ur5e"][0, 3] += .0002
        with self.assertRaisesRegex(ControlFault, "handoff allowance"):
            self.check(live=live)


if __name__ == "__main__":
    unittest.main()
