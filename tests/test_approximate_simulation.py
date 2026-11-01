"""Virtual alignment and following share the declared flange task allowance."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from test_simulation_runtime import options, synthetic_snapshot
from test_target_approximation import recorded_request
from urcal.control.approximation import flange_adjustment
from urcal.control.geometry import inverse, model_from_controller_base
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
    return options(mode, "none", "--target_tolerance_m", ".05", "--target_tolerance_deg", "10",
                   "--gap", ".30", "--max_travel", ".15")


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





if __name__ == "__main__":
    unittest.main()
