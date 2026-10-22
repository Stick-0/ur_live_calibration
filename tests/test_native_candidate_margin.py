"""Offline regression for controller/local model endpoint conditioning mismatch."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

from urcal.control.alignment import CapturedKinematics, LiveAlignment
from urcal.control.approximation import flange_adjustment, select_alignment_candidate
from urcal.control.geometry import pose_delta
from urcal.control.policy import ControlFault
from urcal.control.runtime import parse_args
from urcal.cooperative.kinematics import NominalKinematics


# Recorded Arm request, copied into this test so no runtime logs are required.
# The real controller's returned IK joints were not recorded. A separate
# nominal FK model below reproduces the kind of local/controller mismatch;
# it does not claim to reproduce the unrecorded factory-calibrated IK result.
SNAPSHOT = {'tcp_goals': {'ur5e': [[0.7171825740098767,
                         0.6917388732007855,
                         -0.08453689632032584,
                         -0.05231752369445952],
                        [0.07394132328550101,
                         0.045089042266533604,
                         0.99624277110465,
                         0.21291361662156108],
                        [0.6929515396096285,
                         -0.7207387248999402,
                         -0.018811012260135523,
                         0.575109673917575],
                        [0.0, 0.0, 0.0, 1.0]]},
 'initial_bases': {'ur5e': [[0.1883099818220469,
                             0.9812186750255195,
                             -0.04182418591366257,
                             -0.2865159256791208],
                            [0.3856093087829481,
                             -0.11303595279401317,
                             -0.9157119276038111,
                             0.6922312854940376],
                            [-0.9032412810130624,
                             0.1563099010207286,
                             -0.39965285325738903,
                             2.415066554268625],
                            [0.0, 0.0, 0.0, 1.0]],
                   'ur10e': [[0.9665924652160847,
                              -0.2520097123737956,
                              -0.046798622381096604,
                              0.38684252578125666],
                             [-0.15256063132879003,
                              -0.4189282051395222,
                              -0.8951113968144611,
                              0.3786381926421836],
                             [0.20597150277660495,
                              0.8723475590656814,
                              -0.44337960737509463,
                              3.0213778416806645],
                             [0.0, 0.0, 0.0, 1.0]]},
 'initial_robots': {'ur5e': {'q': [3.0968029499053955,
                                   -2.6065122089781703,
                                   2.2183736006366175,
                                   -2.7296906910338343,
                                   0.3624587059020996,
                                   0.49656057357788086],
                             'qd': [0.0, 0.0, -0.0, -0.0, 0.0, -0.0],
                             'tcp': [[0.8242078998483231,
                                      -0.4738107970831725,
                                      -0.31013652863702684,
                                      -0.02487961253256724],
                                     [0.27253299797063096,
                                      -0.1481867890143712,
                                      0.9506663140023162,
                                      0.22791168077351887],
                                     [-0.4963941003323435,
                                      -0.8680691240500489,
                                      0.006992354841129866,
                                      0.6281757030375558],
                                     [0.0, 0.0, 0.0, 1.0]],
                             'speed': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                    'ur10e': {'q': [-1.598137680684225,
                                    -1.1872293514064332,
                                    -2.1151793003082275,
                                    -2.169176240960592,
                                    -0.06348973909486944,
                                    -1.2222920553028871e-05],
                              'qd': [-0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
                              'tcp': [[0.040459055186824244,
                                       0.010705563720520395,
                                       -0.9991238440547875,
                                       -0.2983481773779976],
                                      [-0.6881431916440408,
                                       0.7252966000517032,
                                       -0.02009452033248882,
                                       -0.41976282295995593],
                                      [0.7244460039556727,
                                       0.6883532762026092,
                                       0.0367117759552954,
                                       0.5776315507285472],
                                      [0.0, 0.0, 0.0, 1.0]],
                              'speed': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}}


def latest_request():
    args = parse_args(["--mode", "follow", "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2"])
    args.target_tolerance_m, args.target_tolerance_deg = .05, 10.
    args.min_singular_value = .01
    offsets = {name: np.eye(4) for name in SNAPSHOT["initial_robots"]}
    bases = {name: np.array(value) for name, value in SNAPSHOT["initial_bases"].items()}
    states = {name: SimpleNamespace(joints=np.array(value["q"]), tcp=np.array(value["tcp"]),
                                   joint_velocity=np.array(value["qd"]), speed=np.array(value["speed"]))
              for name, value in SNAPSHOT["initial_robots"].items()}
    targets = {name: np.array(value) for name, value in SNAPSHOT["tcp_goals"].items()}
    return args, offsets, bases, states, targets


def distinct_model_solution(target, seed):
    # This second, slightly different offline model accepts the same exact TCP
    # target. Near singularity its joints can differ materially from the seed.
    model = NominalKinematics("ur5e")

    def residual(q):
        pose, _ = model.pose_and_jacobian(q)
        dp, dr = pose_delta(pose, target)
        return np.concatenate((dp, dr))

    result = least_squares(residual, seed, max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if np.linalg.norm(result.fun) > 1e-8:
        raise AssertionError("offline model failed to reproduce exact endpoint")
    return result.x


class NativeCandidateMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = latest_request()
        cls.candidates = {2.: select_alignment_candidate(*cls.values)}
        cls.candidates.update({scale: select_alignment_candidate(*cls.values, conditioning_scale=scale)
                               for scale in (4., 6.)})
        cls.solved = {scale: distinct_model_solution(selected["ur5e"], seeds["ur5e"])
                      for scale, (selected, seeds, _) in cls.candidates.items()}

    def test_default_preference_preserves_recorded_candidate(self):
        args, offsets, _, states, targets = self.values
        selected, seeds, diagnostics = self.candidates[2.]
        dp, dr = flange_adjustment(targets["ur5e"], selected["ur5e"], offsets["ur5e"])
        self.assertAlmostEqual(dp, .017113665865960342, places=6)
        self.assertAlmostEqual(np.rad2deg(dr), 1.281472155725067, places=3)
        self.assertEqual(diagnostics["conditioning_scale"], 2.)
        self.assertEqual(diagnostics["singular_preference"], .02)
        model = CapturedKinematics("ur5e", states["ur5e"].joints, states["ur5e"].tcp, offsets["ur5e"])
        seed_sigma = np.linalg.svd(model.pose_and_jacobian(seeds["ur5e"])[1], compute_uv=False)[-1]
        returned_sigma = np.linalg.svd(model.pose_and_jacobian(self.solved[2.])[1], compute_uv=False)[-1]
        self.assertGreater(seed_sigma, args.min_singular_value)
        self.assertLess(returned_sigma, args.min_singular_value)

    def test_stronger_preferences_keep_task_limits_and_validate_complete_geometric_paths(self):
        args, offsets, bases, states, targets = self.values
        before = deepcopy(self.values)
        model = CapturedKinematics("ur5e", states["ur5e"].joints, states["ur5e"].tcp, offsets["ur5e"])
        for scale in (4., 6.):
            with self.subTest(scale=scale):
                selected, seeds, diagnostics = self.candidates[scale]
                dp, dr = flange_adjustment(targets["ur5e"], selected["ur5e"], offsets["ur5e"])
                self.assertLess(dp, args.target_tolerance_m - .0005)
                self.assertLess(dr, np.deg2rad(args.target_tolerance_deg) - .002)
                self.assertEqual(diagnostics["conditioning_scale"], scale)
                self.assertEqual(diagnostics["singular_preference"], scale * args.min_singular_value)
                returned = self.solved[scale]
                sigma = np.linalg.svd(model.pose_and_jacobian(returned)[1], compute_uv=False)[-1]
                self.assertGreater(sigma, args.min_singular_value)
                plan = LiveAlignment(args, offsets, bases, states, selected, {"ur5e": returned})
                self.assertEqual(len(plan.samples), 81)
                for sample in plan.samples:
                    self.assertTrue(plan.guard.check(bases, sample)[0])
                    np.testing.assert_array_equal(sample["ur10e"], states["ur10e"].joints)
                np.testing.assert_allclose(plan.samples[-1]["ur5e"], returned, atol=1e-15, rtol=0)
        self.assertEqual(vars(args), vars(before[0]))
        for name in states:
            np.testing.assert_array_equal(offsets[name], before[1][name])
            np.testing.assert_array_equal(bases[name], before[2][name])
            np.testing.assert_array_equal(states[name].tcp, before[3][name].tcp)
            np.testing.assert_array_equal(states[name].joints, before[3][name].joints)
        np.testing.assert_array_equal(targets["ur5e"], before[4]["ur5e"])
        self.assertEqual(args.min_singular_value, .01)
        self.assertEqual(args.target_tolerance_m, .05)
        self.assertEqual(args.target_tolerance_deg, 10.)

    def test_invalid_conditioning_preferences_are_rejected_before_search(self):
        for scale in (0., .99, -1., float("nan"), float("inf"), None, "bad"):
            with self.subTest(scale=scale):
                with patch("urcal.control.approximation.least_squares", side_effect=AssertionError("no search")):
                    with self.assertRaisesRegex(ControlFault, "conditioning scale"):
                        select_alignment_candidate(*latest_request(), conditioning_scale=scale)

    def test_stronger_preference_preserves_stop_callback(self):
        def stop():
            raise ControlFault("operator stop")

        with patch("urcal.control.approximation.least_squares", side_effect=AssertionError("no search")):
            with self.assertRaisesRegex(ControlFault, "operator stop"):
                select_alignment_candidate(*latest_request(), conditioning_scale=6.,
                                           check_fresh=stop)


if __name__ == "__main__":
    unittest.main()
