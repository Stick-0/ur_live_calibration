"""Bare-flange endpoint geometry without changing radii, margins or tool bounds."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from urcal.control.alignment import CapturedKinematics
from urcal.control.geometry import facing_follower, inverse, pose_matrix
from urcal.cooperative.kinematics import (
    NominalKinematics, WorkspaceGuard, _capsule_support,
    _terminal_pair_clearance, _terminal_wrist_planes, segment_distance,
)
from urcal.simulation.plant import SeededKinematics


def facing_scene(model_type=NominalKinematics, *, gap=.15, offset=None, correction=None):
    offset = np.eye(4) if offset is None else offset
    correction = np.eye(4) if correction is None else correction
    joints = {name: np.array([-.6, -1.2, 1.8, -1.4, 1.57, 0.]) for name in ("ur5e", "ur10e")}
    models = {}
    for name, q in joints.items():
        nominal = NominalKinematics(name, offset)
        if model_type is NominalKinematics:
            models[name] = nominal
        else:
            flange = nominal.pose_and_jacobian(q, flange=True)[0]
            models[name] = model_type(name, q, flange @ correction @ offset, offset)
    flanges = {name: model.pose_and_jacobian(joints[name], flange=True)[0]
               for name, model in models.items()}
    bases = {"ur10e": np.eye(4),
             "ur5e": facing_follower(flanges["ur10e"], gap) @ inverse(flanges["ur5e"])}
    return models, bases, joints


class TerminalSupportTests(unittest.TestCase):
    def test_flat_end_retains_full_cylinder_side_and_proximal_hemisphere(self):
        a, b, radius = np.array([0., 0., -.1]), np.zeros(3), .06
        clip = (np.array([0., 0., 1.]), b)
        for vector, expected in (([0, 0, 1], 0.), ([0, 0, -1], .16),
                                 ([1, 0, 0], .06), ([.6, 0, .8], .036)):
            self.assertAlmostEqual(_capsule_support(a, b, radius, np.asarray(vector), clip), expected)

    def test_oblique_plane_support_bound_contains_every_sampled_retained_point(self):
        rng = np.random.default_rng(540)
        a, b, radius = np.array([-.03, .01, -.11]), np.array([.008, -.004, 0.]), .08
        normal = np.array([.1, -.08, 1.]); normal /= np.linalg.norm(normal)
        origin = np.array([.002, -.003, .001])
        directions = rng.normal(size=(20000, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = a + rng.uniform(0., 1., size=(len(directions), 1))*(b-a) + radius*directions
        points = points[(points-origin) @ normal <= 0.]
        for direction in directions[:30]:
            upper = _capsule_support(a, b, radius, direction, (normal, origin))
            self.assertLessEqual(float(np.max(points @ direction)), upper+1e-12)

    def test_side_and_proximal_contact_are_not_cleared_by_flat_distal_end(self):
        a, b, radius = np.array([0., 0., -.1]), np.zeros(3), .06
        clip = (np.array([0., 0., 1.]), b)
        for other in (np.array([.11, 0., -.05]), np.array([0., 0., -.2])):
            original = segment_distance(a, b, other, other)-2*radius
            refined = _terminal_pair_clearance(a, b, radius, other, other, radius, clip, None, original)
            self.assertLess(refined, .02)


class TerminalWorkspaceTests(unittest.TestCase):
    def test_facing_bare_flanges_at_15cm_pass_with_unchanged_default_sizes(self):
        models, bases, joints = facing_scene()
        guard = WorkspaceGuard(models)
        self.assertEqual(guard.radii, {"ur5e": .06, "ur10e": .08})
        self.assertEqual(guard.tool_radii, {"ur5e": .04, "ur10e": .04})
        self.assertEqual(guard.margin, .02)
        allowed, reason, clearance = guard.check(bases, joints)
        self.assertTrue(allowed, (reason, clearance))
        self.assertGreaterEqual(clearance, guard.margin)
        # The original terminal hemispheres alone require at least 16cm.
        self.assertLess(.15-guard.radii["ur5e"]-guard.radii["ur10e"], guard.margin)

    def test_captured_extra_segment_and_tilted_corrections_keep_actual_flange_plane(self):
        for model_type in (CapturedKinematics, SeededKinematics):
            for correction in (np.eye(4), pose_matrix([.004, -.002, .003, .02, -.01, .005])):
                models, bases, joints = facing_scene(model_type, correction=correction)
                for name, model in models.items():
                    tip, points = model.pose_and_points(joints[name])
                    planes = _terminal_wrist_planes(model, tip, points)
                    self.assertEqual(set(planes), {5, 6})
                    np.testing.assert_allclose(planes[6][1], points[-2])
                allowed, reason, clearance = WorkspaceGuard(models).check(bases, joints)
                self.assertTrue(allowed, (model_type, reason, clearance))

    def test_tool_protrusion_and_flange_contact_still_reject(self):
        for options in ({"offset": pose_matrix([0., 0., .06, 0., 0., 0.])}, {"gap": .08}):
            models, bases, joints = facing_scene(**options)
            allowed, reason, clearance = WorkspaceGuard(models).check(bases, joints)
            self.assertFalse(allowed)
            self.assertEqual(reason, "inter-arm capsule clearance")
            self.assertLess(clearance, .02)

    def test_unknown_geometry_retains_original_capsule_rejection(self):
        class OtherModel:
            def __init__(self, model):
                self.model = model
            def pose_and_points(self, q):
                return self.model.pose_and_points(q)
        models, bases, joints = facing_scene()
        guard = WorkspaceGuard({name: OtherModel(model) for name, model in models.items()})
        self.assertFalse(guard.check(bases, joints)[0])

    def test_obstacle_checks_keep_full_capsules(self):
        models, bases, joints = facing_scene(gap=.3)
        flange = models["ur10e"].pose_and_jacobian(joints["ur10e"], flange=True)[0]
        center = flange[:3, 3]+.09*flange[:3, 2]
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/"workspace.json"
            config.write_text(json.dumps({"obstacles": [{"center": center.tolist(), "radius": .01}]}))
            allowed, reason, _ = WorkspaceGuard(models, config).check(bases, joints)
        self.assertFalse(allowed)
        self.assertIn("obstacle clearance", reason)

    def test_common_camera_transform_preserves_refined_decision(self):
        models, bases, joints = facing_scene()
        guard = WorkspaceGuard(models)
        original = guard.check(bases, joints)
        change = pose_matrix([1., -2., 3., .4, -.7, .2])
        changed = guard.check({name: change @ base for name, base in bases.items()}, joints)
        self.assertEqual(original[:2], changed[:2])
        self.assertAlmostEqual(original[2], changed[2], places=12)


if __name__ == "__main__":
    unittest.main()
