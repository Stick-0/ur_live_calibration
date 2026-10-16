"""Full EMPnP regression against the preserved pre-optimization implementation.

Uses synthetic projections of both live programs' robot FK keypoints. No camera,
robot, weights, or GPU is needed. Run from the bundle with:
    ../.venv/bin/python -m unittest discover -s tests -p test_pnp_equivalence.py -v
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load_module("pnp_reference", BUNDLE / "reference" / "my_pnp.py")
from urcal.inference import my_pnp as optimized
# Importing these modules defines FK functions; hardware connections only happen
# when main() is called. Reuse the actual FK definitions to cover both robots.
from urcal.inference import infer_live as ur5e
from urcal.inference import infer_live_ur10e as ur10e

K = np.array([[615.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.01, -0.005, 0.0002, -0.0001, 0.001], dtype=np.float64)


def synthetic_cases():
    """144 seeded cases: 2 robots × 6 configurations × 4 masks × 3 noises."""
    rng = np.random.default_rng(20261012)
    joints = [
        np.zeros(6),
        np.array([0.2, -1.1, 0.9, -1.7, 0.5, 0.8]),
        np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]),
        *rng.uniform(-2.5, 2.5, size=(3, 6)),
    ]
    masks = [[0, 2, 4, 6], [0, 1, 3, 4, 6], [0, 1, 2, 4, 5, 6], list(range(7))]
    robots = [
        ("ur5e", ur5e.ur5e_keypoints_from_joints),
        ("ur10e", ur10e.ur10e_keypoints_from_joints),
    ]
    case_id = 0
    for robot, fk in robots:
        for configuration, q in enumerate(joints):
            all_points = fk(q)
            r_true = np.array([0.15, -0.10, 0.20]) + rng.normal(0, 0.15, 3)
            t_true = np.array([0.08, -0.05, 2.5]) + rng.normal(0, 0.05, 3)
            for mask in masks:
                points = np.ascontiguousarray(all_points[mask], dtype=np.float64)
                clean = cv2.projectPoints(points, r_true, t_true, K, DIST)[0].reshape(-1, 2)
                for disturbance in ("clean", "noise", "outlier"):
                    observed = clean.copy()
                    previous = clean + rng.normal(0, 0.2, clean.shape)
                    if disturbance != "clean":
                        observed += rng.normal(0, 0.5, clean.shape)
                    if disturbance == "outlier":
                        observed[-1] += [12.0, -9.0]
                    prior = 0.9 if case_id % 2 == 0 else rng.uniform(0.1, 1, len(points))
                    yield {
                        "name": (robot, configuration, len(points), disturbance),
                        "seed": 1000 + case_id,
                        "points": points,
                        "observed": observed,
                        "previous": previous,
                        "prior": prior,
                        # Exercise the live default (50) and shorter solves.
                        "iterations": 50 if case_id % 6 == 0 else 10,
                    }
                    case_id += 1


def fit_case(implementation, case):
    solver = implementation.EMPnP(K.copy(), DIST.copy())
    cv2.setRNGSeed(case["seed"])
    try:
        result = solver.fit(
            case["points"].copy(),
            case["observed"].copy(),
            case["previous"].copy(),
            pi_prior=np.copy(case["prior"]) if not np.isscalar(case["prior"]) else case["prior"],
            max_iter=case["iterations"],
        )
    except (RuntimeError, ValueError, cv2.error) as exc:
        return solver, None, (type(exc), str(exc))
    return solver, result, None


class EMPnPEquivalenceTests(unittest.TestCase):
    def test_full_fit_matches_reference(self):
        successes = failures = 0
        for case in synthetic_cases():
            with self.subTest(case=case["name"]):
                old, old_result, old_failure = fit_case(baseline, case)
                new, new_result, new_failure = fit_case(optimized, case)
                self.assertEqual(old_failure, new_failure)
                if old_failure is not None:
                    failures += 1
                    continue
                successes += 1
                for expected, actual in zip(old_result, new_result):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(new.pi_prior, old.pi_prior)
                np.testing.assert_array_equal(cv2.Rodrigues(new.rvec)[0], cv2.Rodrigues(old.rvec)[0])
                np.testing.assert_array_equal(
                    new.project(new.rvec, new.tvec, case["points"]),
                    old.project(old.rvec, old.tvec, case["points"]),
                )
        self.assertEqual(successes + failures, 144)
        # RANSAC legitimately rejects some 4-point/outlier cases. Ensure that
        # the regression actually exercises enough completed full EM fits.
        self.assertGreaterEqual(successes, 100)

    def test_m_step_zero_and_uneven_weights_match_reference(self):
        rng = np.random.default_rng(817)
        points = ur10e.ur10e_keypoints_from_joints([0.2, -1.1, 0.9, -1.7, 0.5, 0.8])
        r = np.array([0.15, -0.1, 0.2])
        t = np.array([0.08, -0.05, 2.5])
        u = cv2.projectPoints(points, r, t, K, DIST)[0].reshape(-1, 2)
        u += rng.normal(0, 0.5, u.shape)
        for w in (np.ones(7), np.zeros(7), np.array([0, 0.1, 0.7, 1, 0.2, 0, 0.9])):
            with self.subTest(weights=w):
                old = baseline.EMPnP(K, DIST).m_step(r + 0.02, t + 0.03, points, u, w)
                new = optimized.EMPnP(K, DIST).m_step(r + 0.02, t + 0.03, points, u, w)
                np.testing.assert_array_equal(new, old)


if __name__ == "__main__":
    unittest.main()
