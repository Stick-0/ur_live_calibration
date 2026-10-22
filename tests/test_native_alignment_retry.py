"""Preflight retries controller-returned IK, not merely the search seed."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.test_native_candidate_margin import distinct_model_solution, latest_request
from urcal.control.approximation import select_alignment_candidate
from urcal.control.policy import ControlFault
from urcal.control.runtime import ArmConditioningUnavailable, ArmIKUnavailable
from urcal.control.startup import prepare_alignment
from urcal.cooperative.kinematics import NominalKinematics


class NativeAlignmentRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = {scale: select_alignment_candidate(*latest_request(), conditioning_scale=scale)
                          for scale in (2., 4., 6.)}
        cls.native_joints = {
            scale: distinct_model_solution(candidate[0]["ur5e"], candidate[1]["ur5e"])
            for scale, candidate in cls.candidates.items()}

    def rig(self, *, exact_rejected=True, repeat_bad_candidate=False, solve_error=None,
            path_error=None, retry_hook=None, mutate_diagnostic=False):
        args, offsets, bases, states, targets = latest_request()
        if not exact_rejected:
            targets = deepcopy(self.candidates[2.][0])
        session = Mock()
        session.last_ik_tcp = {}
        records, reports, searches, solved, events = [], [], [], [], []
        models = {name: NominalKinematics(name, offsets[name]) for name in states}

        def current():
            return deepcopy(bases), deepcopy(states)

        rig = SimpleNamespace(args=args, offsets=offsets, bases=bases, states=states, targets=targets,
                              session=session, records=records, reports=reports, searches=searches,
                              solved=solved, events=events, current=current)

        def search(*supplied, check_fresh, conditioning_scale):
            searches.append(conditioning_scale)
            events.append("search_" + str(conditioning_scale))
            if retry_hook is not None and conditioning_scale == 4.:
                retry_hook(rig, check_fresh)
            check_fresh()
            return deepcopy(self.candidates[2. if repeat_bad_candidate else conditioning_scale])

        def solve(request, seeds, fresh, *, report=None):
            session.last_ik_tcp = {}
            fresh()
            solved.append(deepcopy((request, seeds)))
            events.append("solve_" + str(len(solved)))
            if len(solved) == 1 and exact_rejected:
                raise ArmIKUnavailable("ur5e", request["ur5e"], seeds["ur5e"], offsets["ur5e"])
            if solve_error is not None and len(solved) == 3:
                raise solve_error
            scale = 2. if len(solved) == 1 or repeat_bad_candidate else searches[-1]
            q = self.native_joints[scale].copy()
            native = models["ur5e"].pose_and_jacobian(q)[0]
            # The second model must solve the selected TCP pose independently;
            # it does not echo the candidate's joints back as a controller.
            np.testing.assert_allclose(native, request["ur5e"], atol=1e-8)
            session.last_ik_tcp = {"ur5e": native.copy()}
            fresh()
            return {"ur5e": q}

        def path(samples, fresh, *, report=None):
            events.append("path")
            fresh()
            if path_error is not None:
                raise path_error
            return [{name: models[name].pose_and_jacobian(sample[name])[0] for name in targets}
                    for sample in samples]

        def diagnostic(record):
            records.append(deepcopy(record))
            if mutate_diagnostic:
                # A consumer must not gain aliases to nominal targets, solver
                # arrays, or the retained attempt history used after this call.
                record.get("nominal_tcp_goals", {}).get("ur5e", np.zeros((4, 4)))[:] = -100.
                record.get("command_tcp_goals", {}).get("ur5e", np.zeros((4, 4)))[:] = -200.
                if "conditioning" in record:
                    record["conditioning"]["robots"]["ur5e"]["native_q_rad"][:] = 100.
                record["outcome"] = "consumer mutation"

        session.solve_joint_targets.side_effect = solve
        session.validate_joint_path.side_effect = path
        session.enable_watchdog.side_effect = lambda check_fresh: (check_fresh(), events.append("watchdog"))
        rig.search, rig.diagnostic = search, diagnostic
        return rig

    def prepare(self, rig):
        with patch("urcal.control.approximation.select_alignment_candidate", side_effect=rig.search):
            return prepare_alignment(rig.session, rig.args, rig.offsets, rig.bases, rig.states,
                                     rig.targets, rig.current, rig.reports.append, diagnostic=rig.diagnostic)

    def assert_no_motion(self, rig):
        rig.session.send.assert_not_called()
        rig.session.send_joints.assert_not_called()

    def test_native_returned_bad_condition_retries_before_path_and_preserves_nominal(self):
        rig = self.rig()
        before = deepcopy((rig.targets, rig.bases, rig.states))
        alignment = self.prepare(rig)
        self.assertEqual(rig.searches, [2., 4.])
        self.assertEqual(len(rig.solved), 3)
        self.assertEqual([item["outcome"] for item in rig.records],
                         ["native_ik_unavailable", "conditioning_rejected", "accepted"])
        poor = rig.records[1]["conditioning"]["robots"]["ur5e"]
        self.assertGreater(poor["seed_singular_value"], .01)
        self.assertLess(poor["captured_singular_value"], .01)
        np.testing.assert_allclose(poor["native_q_rad"], self.native_joints[2.], atol=1e-12)
        self.assertGreater(np.linalg.norm(poor["seed_q_rad"] - poor["native_q_rad"]), .01)
        good = rig.records[2]["conditioning"]["robots"]["ur5e"]
        self.assertGreaterEqual(min(good["captured_singular_value"], good["tracking_singular_value"]), .01)
        np.testing.assert_allclose(alignment.goal_joints["ur5e"], self.native_joints[4.], atol=1e-12)
        np.testing.assert_array_equal(alignment.targets["ur5e"], self.candidates[4.][0]["ur5e"])
        np.testing.assert_array_equal(alignment.nominal_targets["ur5e"], before[0]["ur5e"])
        np.testing.assert_array_equal(rig.targets["ur5e"], before[0]["ur5e"])
        for name in rig.states:
            np.testing.assert_array_equal(rig.bases[name], before[1][name])
            np.testing.assert_array_equal(rig.states[name].joints, before[2][name].joints)
            np.testing.assert_array_equal(rig.states[name].tcp, before[2][name].tcp)
        rig.session.validate_joint_path.assert_called_once()
        self.assertEqual(len(rig.session.validate_joint_path.call_args.args[0]), 81)
        rig.session.enable_watchdog.assert_called_once()
        self.assertLess(rig.events.index("solve_3"), rig.events.index("path"))
        self.assertLess(rig.events.index("path"), rig.events.index("watchdog"))
        self.assert_no_motion(rig)

    def test_exact_native_ik_success_with_bad_condition_also_uses_bounded_fallback(self):
        rig = self.rig(exact_rejected=False)
        alignment = self.prepare(rig)
        self.assertEqual(rig.records[0]["outcome"], "conditioning_rejected")
        self.assertEqual(rig.records[0]["attempt"], 0)
        self.assertEqual(rig.searches, [2., 4.])
        self.assertEqual(len(rig.solved), 3)
        np.testing.assert_array_equal(alignment.nominal_targets["ur5e"], self.candidates[2.][0]["ur5e"])
        np.testing.assert_array_equal(alignment.targets["ur5e"], self.candidates[4.][0]["ur5e"])
        rig.session.validate_joint_path.assert_called_once()
        self.assert_no_motion(rig)

    def test_candidate_exhaustion_is_finite_recoverable_and_never_validates_a_path(self):
        rig = self.rig(repeat_bad_candidate=True)
        with self.assertRaises(ArmConditioningUnavailable) as caught:
            self.prepare(rig)
        self.assertEqual(rig.searches, [2., 4., 6.])
        self.assertEqual(len(rig.solved), 4)  # One exact request plus three bounded searches.
        self.assertEqual(caught.exception.details["max_attempts"], 3)
        self.assertEqual(len(caught.exception.details["attempts"]), 4)
        self.assertEqual(caught.exception.details["position_tolerance_m"], .05)
        self.assertEqual(caught.exception.details["rotation_tolerance_deg"], 10.)
        rig.records[-1]["conditioning"]["robots"]["ur5e"]["native_q_rad"][:] = 123.
        np.testing.assert_array_equal(
            caught.exception.details["attempts"][-1]["conditioning"]["robots"]["ur5e"]["native_q_rad"],
            self.native_joints[2.])
        rig.session.validate_joint_path.assert_not_called()
        rig.session.enable_watchdog.assert_not_called()
        self.assert_no_motion(rig)

    def test_stop_or_encoder_change_during_retry_is_not_reclassified_or_ignored(self):
        for cause in ("stop", "encoder"):
            with self.subTest(cause=cause):
                stop_error = ControlFault("operator stop during retry")

                def retry(rig, fresh):
                    if cause == "stop":
                        # The selector's ordinary Stop check must propagate
                        # immediately rather than trying a third candidate.
                        raise stop_error
                    moved = deepcopy(rig.states)
                    moved["ur10e"].joints[0] += .02
                    # Change what the original reader returns, without moving
                    # the original Arm snapshot held in prepare_alignment.
                    reader.return_value = deepcopy(rig.bases), moved
                    fresh()

                rig = self.rig(retry_hook=retry)
                reader = Mock(side_effect=None, return_value=(deepcopy(rig.bases), deepcopy(rig.states)))
                rig.current = reader
                with self.assertRaises(ControlFault) as caught:
                    self.prepare(rig)
                self.assertNotIsInstance(caught.exception, ArmConditioningUnavailable)
                if cause == "stop":
                    self.assertIs(caught.exception, stop_error)
                else:
                    self.assertIn("robot moved", str(caught.exception))
                self.assertEqual(rig.searches, [2., 4.])
                self.assertEqual(len(rig.solved), 2)
                rig.session.validate_joint_path.assert_not_called()
                rig.session.enable_watchdog.assert_not_called()
                self.assert_no_motion(rig)

    def test_native_fk_safety_and_path_faults_are_not_swallowed_as_retryable_conditioning(self):
        for message, at_path in (("native IK/FK target residual exceeds 0.5 mm / 0.002 rad", False),
                                 ("native IK joints outside controller safety limits", False),
                                 ("native IK TCP outside controller safety limits", False),
                                 ("native path sample outside controller safety limits", True)):
            with self.subTest(message=message):
                fault = ControlFault(message)
                rig = self.rig(**({"path_error": fault} if at_path else {"solve_error": fault}))
                with self.assertRaises(ControlFault) as caught:
                    self.prepare(rig)
                self.assertIs(caught.exception, fault)
                self.assertEqual(rig.searches, [2., 4.])
                self.assertEqual(len(rig.solved), 3)
                self.assertEqual(rig.session.validate_joint_path.call_count, int(at_path))
                rig.session.enable_watchdog.assert_not_called()
                self.assert_no_motion(rig)

    def test_diagnostic_consumer_cannot_mutate_internal_targets_or_attempt_history(self):
        rig = self.rig(mutate_diagnostic=True)
        original = deepcopy(rig.targets)
        alignment = self.prepare(rig)
        np.testing.assert_array_equal(rig.targets["ur5e"], original["ur5e"])
        np.testing.assert_array_equal(alignment.nominal_targets["ur5e"], original["ur5e"])
        np.testing.assert_array_equal(alignment.targets["ur5e"], self.candidates[4.][0]["ur5e"])
        self.assertEqual([item["outcome"] for item in alignment.target_adjustments["native_attempts"]],
                         ["native_ik_unavailable", "conditioning_rejected", "accepted"])
        np.testing.assert_array_equal(
            alignment.target_adjustments["native_attempts"][-1]["conditioning"]["robots"]["ur5e"]["native_q_rad"],
            self.native_joints[4.])
        self.assert_no_motion(rig)


if __name__ == "__main__":
    unittest.main()
