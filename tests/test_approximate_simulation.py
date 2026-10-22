"""Virtual alignment and hold use the same declared flange task allowance."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from test_simulation_runtime import options, synthetic_snapshot
from test_target_approximation import recorded_request
from urcal.control.approximation import flange_adjustment
from urcal.control.geometry import flange_from_tcp, inverse, model_from_controller_base
from urcal.simulation.approach import InitialApproach, InitialApproachIKUnavailable
from urcal.simulation.plant import SimulationFault
from urcal.simulation.runtime import SimulationSession


def recorded_snapshot():
    _, offsets, bases, states, _ = recorded_request()
    initial, _ = synthetic_snapshot()
    initial["bases"] = deepcopy(bases)
    for name, state in states.items():
        initial["robots"][name]["q"] = state.joints.copy()
        initial["robots"][name]["tcp"] = state.tcp.copy()
        initial["robots"][name]["tcp_offset"] = offsets[name].copy()
        initial["calibration"]["poses"][name] = bases[name] @ inverse(model_from_controller_base(name))
    return initial


def tolerant_options(mode="follow"):
    return options(mode, "none", "--target_tolerance_m", ".05", "--target_tolerance_deg", "10")


class ApproximateSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = recorded_snapshot()
        original = InitialApproach.__init__
        calls = []

        def exact_rejected_once(plan, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                # The recorded nominal target has no exact IK. Model that
                # typed outcome here; the chosen replacement must undergo the
                # actual simulator IK, 81-point path and timing validation.
                raise InitialApproachIKUnavailable("ur5e: exact recorded target unavailable")
            return original(plan, *args, **kwargs)

        with patch.object(InitialApproach, "__init__", exact_rejected_once):
            cls.planned = SimulationSession(tolerant_options(), cls.initial)
        if cls.planned.approach is None or not cls.planned.ready:
            raise AssertionError(cls.planned.message)

    def test_selected_command_and_nominal_goal_are_separate_and_capture_is_unchanged(self):
        session = deepcopy(self.planned)
        scene = session.status()["scene"]
        self.assertIn("command_goals", scene)
        nominal = session.approach_nominal_targets["ur5e"]
        selected = session.approach.targets["ur5e"]
        dp, dr = flange_adjustment(nominal, selected, session.plant.offsets["ur5e"])
        self.assertGreater(dp, .01)
        self.assertLess(dp, .05)
        self.assertLess(dr, np.deg2rad(10.))
        np.testing.assert_allclose(scene["goals"]["ur5e"], scene["nominal_goals"]["ur5e"])
        self.assertGreater(np.linalg.norm(scene["command_goals"]["ur5e"] - scene["goals"]["ur5e"]), .01)
        for name in session.plant.joints:
            np.testing.assert_array_equal(session.plant.joints[name], self.initial["robots"][name]["q"])
            np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
        self.assertEqual(session.steps, 0)

    def test_selected_virtual_alignment_reaches_endpoint_then_tolerant_follow_starts(self):
        session = deepcopy(self.planned)
        nominal = session.approach_nominal_targets["ur5e"].copy()
        path_start = session.engine.path.start.copy()
        session.command("arm")
        budget = int(np.ceil(session.approach.duration / session.dt)) + 3
        while not session.approach.done and budget:
            self.assertTrue(session.tick(), session.fault or session.message)
            budget -= 1
        self.assertTrue(session.approach.done)
        self.assertTrue(session.active, session.message)
        states = session.plant.states(session.now)
        dp, dr = flange_adjustment(nominal, states["ur5e"].tcp, session.plant.offsets["ur5e"])
        self.assertGreater(dp, .01)
        self.assertLess(dp, .05)
        self.assertLess(dr, np.deg2rad(10.))
        self.assertTrue(session.tick(), session.fault or session.message)
        self.assertTrue(session.last_record["within_target_tolerance"])
        self.assertGreater(session.last_record["relative_error_m"], .01)
        np.testing.assert_array_equal(session.engine.path.start, path_start)

    def test_path_or_geometry_fault_cannot_trigger_goal_adjustment(self):
        with patch.object(InitialApproach, "__init__", side_effect=SimulationFault("path obstruction")), \
                patch("urcal.control.approximation.select_alignment_candidate") as selector:
            session = SimulationSession(tolerant_options(), self.initial)
            self.assertFalse(session.ready)
            self.assertIn("path obstruction", session.message)
            selector.assert_not_called()

    def test_zero_allowance_preserves_exact_ik_refusal(self):
        with patch.object(InitialApproach, "__init__", side_effect=InitialApproachIKUnavailable("exact IK unavailable")), \
                patch("urcal.control.approximation.select_alignment_candidate") as selector:
            session = SimulationSession(options(), self.initial)
            self.assertFalse(session.ready)
            self.assertIn("exact IK unavailable", session.message)
            selector.assert_not_called()

    def test_tolerated_hold_does_not_query_exact_nominal_ik_or_erase_target(self):
        initial, _ = synthetic_snapshot()
        session = SimulationSession(tolerant_options("hold"), initial)
        states = session.plant.states(session.now)
        session.planner.hold_targets = {
            name: session.plant.bases[name] @ flange_from_tcp(state.tcp, session.plant.offsets[name])
            for name, state in states.items()}
        for target in session.planner.hold_targets.values():
            target[0, 3] += .03
        nominal = deepcopy(session.planner.hold_targets)
        with patch.object(session.plant.models["ur5e"], "inverse", side_effect=AssertionError("no nominal IK")), \
                patch.object(session.plant.models["ur10e"], "inverse", side_effect=AssertionError("no nominal IK")):
            goals, _, twists, qdots = session._hold_proposal(states)
        for name in states:
            np.testing.assert_allclose(twists[name], 0., atol=1e-12)
            np.testing.assert_allclose(qdots[name], 0., atol=1e-12)
            np.testing.assert_array_equal(goals[name], nominal[name])
            np.testing.assert_array_equal(session.planner.hold_targets[name], nominal[name])

    def test_hold_checks_midpoint_and_endpoint_joint_geometry(self):
        initial, _ = synthetic_snapshot()
        session = SimulationSession(tolerant_options("hold"), initial)
        states = session.plant.states(session.now)
        original = session.plant.guard.check
        calls = []

        def reject_midpoint(bases, joints):
            calls.append(1)
            if len(calls) == 2:
                return False, "midpoint obstacle", -.1
            return original(bases, joints)

        with patch.object(session.plant.guard, "check", reject_midpoint):
            with self.assertRaisesRegex(SimulationFault, "midpoint obstacle"):
                session._hold_proposal(states)
        self.assertEqual(len(calls), 2)
        self.assertEqual(session.steps, 0)

    def test_external_virtual_base_motion_cannot_bypass_minimum_gap(self):
        initial, _ = synthetic_snapshot()
        session = SimulationSession(tolerant_options("hold"), initial)
        states = session.plant.states(session.now)
        flanges = {name: session.plant.bases[name] @ flange_from_tcp(
            state.tcp, session.plant.offsets[name]) for name, state in states.items()}
        next_bases = deepcopy(session.plant.bases)
        next_bases["ur5e"][:3, 3] += flanges["ur10e"][:3, 3] - flanges["ur5e"][:3, 3]
        original = deepcopy(session.plant.bases)
        with self.assertRaisesRegex(SimulationFault, "base motion.*below min_gap"):
            session._check_hold_base_sweep({name: np.zeros(6) for name in states}, next_bases)
        for name in states:
            np.testing.assert_array_equal(session.plant.bases[name], original[name])


if __name__ == "__main__":
    unittest.main()
