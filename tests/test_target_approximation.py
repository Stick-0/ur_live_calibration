"""Approximate flange goals stay bounded and require ordinary native preflight."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.alignment import CapturedKinematics, LiveAlignment
from urcal.control.approximation import TargetApproximationUnavailable, flange_adjustment, select_alignment_candidate
from urcal.control.geometry import inverse, pose_matrix
from urcal.control.policy import ControlFault
from urcal.control.runtime import parse_args
from urcal.cooperative.kinematics import WorkspaceGuard


def recorded_request():
    """Self-contained measured Arm snapshot; no run logs or hardware needed."""
    args = SimpleNamespace(target_tolerance_m=.05, target_tolerance_deg=10., min_singular_value=.01,
                           workspace_config=None, max_travel=.15, min_gap=.15)
    q = {
        "ur5e": [3.096818447113037, -2.6065155468382777, 2.218376938496725,
                 -2.7296954594054164, .3624587059020996, .4965832233428955],
        "ur10e": [-1.5981858412372034, -1.187269614343979, -2.1152279376983643,
                  -2.1691962681212367, -.06351071992983037, .000015977866496541537],
    }
    tcp = {
        "ur5e": [-.02488434608978659, .22791134755530845, .6281743265811436,
                 -1.5933143893775887, .16317422953669522, .6538629303043809],
        "ur10e": [-.2983685420233177, -.41975619828657623, .5775823709813881,
                  .5943988862326178, -1.4459961849873548, -.5863981920743176],
    }
    bases = {
        "ur5e": pose_matrix([-.28719609253267353, .6907255455653344, 2.40751141732757,
                             1.640333948263176, 1.3199552565469126, -.9038588585207064]),
        "ur10e": pose_matrix([.387053256779813, .3785249323522763, 3.015785520456429,
                              2.0116708057127237, -.28717351472197566, .11250216786385372]),
    }
    states = {name: SimpleNamespace(joints=np.array(q[name]), tcp=pose_matrix(tcp[name]),
                                    joint_velocity=np.zeros(6), speed=np.zeros(6)) for name in q}
    offsets = {name: np.eye(4) for name in q}
    targets = {"ur5e": pose_matrix([-.054947011988522254, .2104095943817929, .5738299175621109,
                                    -1.4664688301949331, -.6640885799819646, -.5355349822821479])}
    return args, offsets, bases, states, targets


class TargetApproximationTests(unittest.TestCase):
    def test_recorded_rejected_pose_has_bounded_nonsingular_candidate_without_mutation(self):
        values = recorded_request()
        args, offsets, bases, states, targets = values
        before = deepcopy(values)
        selected, seeds, diagnostics = select_alignment_candidate(*values)
        self.assertEqual(set(selected), {"ur5e"})
        self.assertEqual(set(seeds), {"ur5e"})
        dp, dr = flange_adjustment(targets["ur5e"], selected["ur5e"], offsets["ur5e"])
        self.assertGreater(dp, .01)
        self.assertLess(dp, .03)
        self.assertLess(dr, np.deg2rad(10) - .002)
        models = {name: CapturedKinematics(name, states[name].joints, states[name].tcp, offsets[name])
                  for name in states}
        tcp, jacobian = models["ur5e"].pose_and_jacobian(seeds["ur5e"])
        np.testing.assert_allclose(tcp, selected["ur5e"], atol=1e-12)
        self.assertGreaterEqual(np.linalg.svd(jacobian, compute_uv=False)[-1], args.min_singular_value)
        joints = {name: state.joints for name, state in states.items()}
        joints.update(seeds)
        self.assertTrue(WorkspaceGuard(models).check(bases, joints)[0])
        self.assertGreater(diagnostics["residual_evaluations"], 0)
        self.assertAlmostEqual(diagnostics["adjustments"]["ur5e"]["position_m"], dp)
        for name in states:
            np.testing.assert_array_equal(offsets[name], before[1][name])
            np.testing.assert_array_equal(bases[name], before[2][name])
            np.testing.assert_array_equal(states[name].joints, before[3][name].joints)
            np.testing.assert_array_equal(states[name].tcp, before[3][name].tcp)
        np.testing.assert_array_equal(targets["ur5e"], before[4]["ur5e"])

    def test_tolerance_is_euclidean_flange_error_even_with_long_rotated_tcp_offset(self):
        nominal_flange = pose_matrix([.1, .2, .3, .1, .2, .3])
        selected_flange = nominal_flange @ pose_matrix([.02, -.01, .01, 0, .12, 0])
        offset = pose_matrix([.5, .2, -.3, .4, -.2, .1])
        nominal_tcp, selected_tcp = nominal_flange @ offset, selected_flange @ offset
        dp, dr = flange_adjustment(nominal_tcp, selected_tcp, offset)
        self.assertAlmostEqual(dp, np.sqrt(.0006))
        self.assertAlmostEqual(dr, .12)
        self.assertGreater(np.linalg.norm(selected_tcp[:3, 3] - nominal_tcp[:3, 3]), dp)

    def test_nonzero_tcp_offset_search_preserves_flange_tolerance(self):
        args, offsets, bases, states, targets = recorded_request()
        offset = pose_matrix([.08, -.03, .05, .3, -.1, .2])
        for name in states:
            offsets[name] = offset.copy()
            states[name].tcp = states[name].tcp @ offset
        targets["ur5e"] = targets["ur5e"] @ offset
        selected, seeds, diagnostics = select_alignment_candidate(args, offsets, bases, states, targets)
        dp, dr = flange_adjustment(targets["ur5e"], selected["ur5e"], offset)
        reserve = .0005 + 2. * np.sin(.002 / 2.) * np.linalg.norm(offset[:3, 3])
        self.assertAlmostEqual(diagnostics["flange_position_reserves_m"]["ur5e"], reserve)
        self.assertLess(dp, .05 - reserve)
        self.assertLess(dr, np.deg2rad(10) - .002)
        expected = CapturedKinematics("ur5e", states["ur5e"].joints, states["ur5e"].tcp, offset)
        np.testing.assert_allclose(expected.pose_and_jacobian(seeds["ur5e"])[0], selected["ur5e"])

    def test_strict_bounds_fail_instead_of_silently_widening(self):
        values = recorded_request()
        values[0].target_tolerance_m = .001
        values[0].target_tolerance_deg = .2
        with self.assertRaisesRegex(TargetApproximationUnavailable, "no approximate alignment candidate"):
            select_alignment_candidate(*values)
        self.assertEqual(values[0].target_tolerance_m, .001)
        self.assertEqual(values[0].target_tolerance_deg, .2)

    def test_stop_or_encoder_fault_interrupts_optimizer_immediately(self):
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 4:
                raise ControlFault("operator stop")

        with self.assertRaisesRegex(ControlFault, "operator stop") as caught:
            select_alignment_candidate(*recorded_request(), check_fresh=check)
        self.assertEqual(len(calls), 4)
        self.assertNotIsInstance(caught.exception, TargetApproximationUnavailable)

    def test_existing_geometry_and_travel_constraints_are_not_relaxed(self):
        values = recorded_request()
        with patch.object(WorkspaceGuard, "check", return_value=(False, "capsule collision", -.1)):
            with self.assertRaisesRegex(ControlFault, "current configuration: capsule collision") as caught:
                select_alignment_candidate(*values)
            self.assertNotIsInstance(caught.exception, TargetApproximationUnavailable)
        with patch.object(WorkspaceGuard, "pose_allowed", return_value=False):
            with self.assertRaisesRegex(ControlFault, "TCP workspace"):
                select_alignment_candidate(*values)
        values[0].min_gap = .5
        with self.assertRaisesRegex(ControlFault, "gap below min_gap"):
            select_alignment_candidate(*values)
        values[0].min_gap = .15
        values[0].max_travel = .001
        with self.assertRaisesRegex(ControlFault, "travel envelope"):
            select_alignment_candidate(*values)

    def test_candidate_endpoint_geometry_rejection_is_not_only_checked_at_start(self):
        values = recorded_request()
        original = WorkspaceGuard.check

        def at_start_only(guard, bases, joints):
            if np.linalg.norm(joints["ur5e"] - values[3]["ur5e"].joints) > .01:
                return False, "goal obstacle", -.1
            return original(guard, bases, joints)

        with patch.object(WorkspaceGuard, "check", at_start_only):
            with self.assertRaisesRegex(TargetApproximationUnavailable, "goal obstacle"):
                select_alignment_candidate(*values)

    def test_disabled_legacy_and_invalid_tolerances_do_not_search(self):
        for distance, angle in ((0, 0), (-.1, 10), (.05, float("nan")), (.0001, 10)):
            with self.subTest(distance=distance, angle=angle):
                values = recorded_request()
                values[0].target_tolerance_m, values[0].target_tolerance_deg = distance, angle
                with patch("urcal.control.approximation.least_squares", side_effect=AssertionError("no search")):
                    with self.assertRaises(ControlFault):
                        select_alignment_candidate(*values)

    def test_search_is_deterministic(self):
        first = select_alignment_candidate(*recorded_request())
        second = select_alignment_candidate(*recorded_request())
        np.testing.assert_array_equal(first[0]["ur5e"], second[0]["ur5e"])
        np.testing.assert_array_equal(first[1]["ur5e"], second[1]["ur5e"])

    def test_recorded_candidate_passes_the_complete_existing_geometric_alignment_path(self):
        for mode in ("follow", "hold"):
            with self.subTest(mode=mode):
                _, offsets, bases, states, targets = recorded_request()
                if mode == "hold":
                    targets["ur10e"] = states["ur10e"].tcp.copy()
                args = parse_args(["--mode", mode, "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2"])
                args.target_tolerance_m, args.target_tolerance_deg = .05, 10.
                selected, seeds, _ = select_alignment_candidate(args, offsets, bases, states, targets)
                self.assertEqual(set(selected), set(targets))
                for name in targets:
                    dp, dr = flange_adjustment(targets[name], selected[name], offsets[name])
                    self.assertLess(dp, .05)
                    self.assertLess(dr, np.deg2rad(10.))
                # Pure kinematic path validation only: these candidate seeds
                # still need native IK/FK before becoming motion goals.
                plan = LiveAlignment(args, offsets, bases, states, selected, seeds)
                self.assertEqual(len(plan.samples), 81)
                np.testing.assert_array_equal(plan.samples[0]["ur5e"], states["ur5e"].joints)
                np.testing.assert_allclose(plan.samples[-1]["ur5e"], seeds["ur5e"], rtol=0, atol=1e-15)
                for sample in plan.samples:
                    self.assertTrue(plan.guard.check(bases, sample)[0])
                    if mode == "follow":
                        np.testing.assert_array_equal(sample["ur10e"], states["ur10e"].joints)
                self.assertLess(plan.estimated_duration, 600.)


if __name__ == "__main__":
    unittest.main()
