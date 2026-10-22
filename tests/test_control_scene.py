"""Display geometry tests with injected kinematics; no GUI or hardware."""

from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose

from urcal.control.geometry import pose_matrix
from urcal.control.scene import (SceneError, build_scene, fit_view, orbit_basis,
                                 project_points, scene_points)


class FakeModel:
    def __init__(self, points=None):
        self.local_points = np.array([[0., 0., 0.], [.1, .2, .3], [.3, .2, .4]]
                                     if points is None else points, dtype=float)
        self.last_q = None

    def points(self, q):
        self.last_q = q.copy()
        q[:] = 99.  # Even an impolite injected model cannot mutate the packet.
        return self.local_points


def fixture():
    models = {name: FakeModel() for name in ("ur5e", "ur10e")}
    snapshot = {
        "at": 100.,
        "bases": {"ur5e": pose_matrix([-.4, .2, 1.5, .1, .2, -.1]),
                  "ur10e": pose_matrix([.5, .1, 1.8, -.2, .1, .3])},
        "robots": {name: {"q": np.arange(6.) / 10.,
                          "tcp": pose_matrix([.3, .1, .5, .2, -.3, .1]),
                          "tcp_offset": np.eye(4)} for name in models},
        "goals": {}, "measurement_age": {"ur5e": .12, "ur10e": .12},
    }
    return snapshot, models


class SceneTests(unittest.TestCase):
    def test_wait_without_goals_keeps_both_robot_polylines(self):
        packet, models = fixture()
        del packet["goals"]
        scene = build_scene(packet, models)
        self.assertEqual(set(scene["robots"]), {"ur5e", "ur10e"})
        for name, robot in scene["robots"].items():
            base = packet["bases"][name]
            expected = models[name].local_points @ base[:3, :3].T + base[:3, 3]
            assert_allclose(robot["joint_polyline_C"], expected)
            self.assertIsNone(robot["target_C"])
            self.assertIsNone(robot["error_m"])
            self.assertIsNone(robot["error_rad"])
        self.assertEqual(scene["measurement_age"], packet["measurement_age"])
        self.assertTrue(scene["calibration_valid"])
        self.assertEqual(scene["calibration_mode"], "live")
        self.assertIsNone(scene["calibration_captured_at"])
        self.assertEqual(scene["reason"], "")
        self.assertIsNone(scene["task"])

    def test_frozen_calibration_metadata_preserves_actual_camera_age(self):
        packet, models = fixture()
        packet.update(calibration_mode="frozen", calibration_captured_at=20., measurement_age=80.)
        scene = build_scene(packet, models)
        self.assertEqual(scene["calibration_mode"], "frozen")
        self.assertEqual(scene["calibration_captured_at"], 20.)
        self.assertEqual(scene["measurement_age"], 80.)
        self.assertEqual(scene["at"], 100.)

    def test_actual_tcp_nonzero_rotated_offset_recovers_flange(self):
        packet, models = fixture()
        local_flange = pose_matrix([.2, -.15, .6, -.2, .1, .4])
        offset = pose_matrix([.04, -.02, .12, .3, -.4, .1])
        packet["robots"]["ur5e"]["tcp_offset"] = offset
        packet["robots"]["ur5e"]["tcp"] = local_flange @ offset
        expected = packet["bases"]["ur5e"] @ local_flange
        packet["goals"]["ur5e"] = expected.copy()
        robot = build_scene(packet, models)["robots"]["ur5e"]
        assert_allclose(robot["flange_C"], expected, atol=1e-15)
        assert_allclose(robot["tcp_C"], expected @ offset, atol=1e-15)
        self.assertAlmostEqual(robot["error_m"], 0.)
        self.assertAlmostEqual(robot["error_rad"], 0.)
        self.assertGreater(np.linalg.norm(robot["tcp_C"][:3, 3] - robot["flange_C"][:3, 3]), .1)

    def test_errors_use_camera_flange_and_shortest_rotation(self):
        packet, models = fixture()
        packet["bases"]["ur5e"] = np.eye(4)
        packet["robots"]["ur5e"]["tcp"] = pose_matrix([.1, .2, .3, 0, 0, np.deg2rad(179)])
        target = pose_matrix([.13, .24, .3, 0, 0, np.deg2rad(-179)])
        packet["goals"] = {"ur5e": target}
        robot = build_scene(packet, models)["robots"]["ur5e"]
        self.assertAlmostEqual(robot["error_m"], .05)
        self.assertAlmostEqual(robot["error_rad"], np.deg2rad(2))

    def test_scene_has_no_input_aliases_and_copies_task_metadata(self):
        packet, models = fixture()
        packet["goals"]["ur5e"] = np.eye(4)
        packet.update(calibration_valid=False, reason="measurement expired",
                      task={"path": "line", "samples": [np.eye(4)]})
        original = deepcopy(packet)
        scene = build_scene(packet, models)
        self.assertFalse(scene["calibration_valid"])
        self.assertEqual(scene["reason"], "measurement expired")
        scene["task"]["samples"][0][0, 3] = 88
        scene["measurement_age"]["ur5e"] = 42
        robot = scene["robots"]["ur5e"]
        for field in ("base_C", "flange_C", "tcp_C", "target_C", "joint_polyline_C"):
            robot[field][...] = 77
        for name in models:
            assert_allclose(packet["robots"][name]["q"], original["robots"][name]["q"])
            assert_allclose(models[name].last_q, original["robots"][name]["q"])
            assert_allclose(packet["bases"][name], original["bases"][name])
        assert_allclose(packet["goals"]["ur5e"], original["goals"]["ur5e"])
        assert_allclose(packet["task"]["samples"][0], np.eye(4))
        self.assertEqual(packet["measurement_age"], original["measurement_age"])
        assert_allclose(models["ur5e"].local_points, FakeModel().local_points)

    def test_invalid_pair_pose_state_and_metadata_are_rejected(self):
        cases = [
            ("missing robot", lambda p: p["robots"].pop("ur10e")),
            ("missing base", lambda p: p["bases"].pop("ur5e")),
            ("unknown goal", lambda p: p["goals"].update(other=np.eye(4))),
            ("bad time", lambda p: p.update(at=float("nan"))),
            ("bad q", lambda p: p["robots"]["ur5e"].update(q=[0.] * 5)),
            ("bad tcp", lambda p: p["robots"]["ur5e"].update(tcp=np.zeros((4, 4)))),
            ("bad offset", lambda p: p["robots"]["ur5e"].update(tcp_offset=np.diag([-1, 1, 1, 1]))),
            ("bad goal", lambda p: p["goals"].update(ur10e=np.eye(3))),
            ("bad age", lambda p: p.update(measurement_age=-.1)),
            ("partial ages", lambda p: p.update(measurement_age={"ur5e": 0.})),
            ("bad validity", lambda p: p.update(calibration_valid="false")),
            ("bad calibration mode", lambda p: p.update(calibration_mode="unknown")),
            ("bad capture time", lambda p: p.update(calibration_captured_at=float("nan"))),
            ("negative capture time", lambda p: p.update(calibration_captured_at=-1.)),
            ("bad reason", lambda p: p.update(reason=4)),
            ("bad task", lambda p: p.update(task="line")),
        ]
        for label, change in cases:
            with self.subTest(label=label):
                packet, models = fixture()
                change(packet)
                with self.assertRaises(SceneError):
                    build_scene(packet, models)

    def test_invalid_nominal_points_and_missing_models_are_rejected(self):
        for points in ([[0., 0., 0.]], [[0., 0.], [1., 1.]], [[0., 0., 0.], [np.nan, 0., 0.]]):
            with self.subTest(points=points):
                packet, models = fixture()
                models["ur5e"] = FakeModel(points)
                with self.assertRaises(SceneError):
                    build_scene(packet, models)
        packet, models = fixture()
        del models["ur10e"]
        with self.assertRaises(SceneError):
            build_scene(packet, models)

    def test_scene_fit_points_include_targets_without_camera_origin(self):
        packet, models = fixture()
        for name in models:
            packet["bases"][name] = pose_matrix([10., 20., 30., 0, 0, 0])
        packet["goals"]["ur5e"] = pose_matrix([12., 23., 32., .2, -.3, .1])
        scene = build_scene(packet, models)
        points = scene_points(scene)
        self.assertTrue(np.all(points[:, 0] >= 10.))
        self.assertTrue(np.all(points[:, 1] >= 20.))
        self.assertTrue(np.all(points[:, 2] >= 30.))
        self.assertTrue(np.any(np.all(points == [12., 23., 32.], axis=1)))
        assert_allclose(scene["bounds"][0], points.min(axis=0))
        assert_allclose(scene["bounds"][1], points.max(axis=0))
        self.assertGreater(scene["bounds"][1][0], 12.)


class ProjectionTests(unittest.TestCase):
    def test_front_projection_preserves_camera_image_axes_and_depth(self):
        center = np.array([.4, .5, 2.])
        xy, depth = project_points(center + np.eye(3), center, 100., 0., 0., (800, 600))
        assert_allclose(xy, [[500, 300], [400, 400], [400, 300]])
        assert_allclose(depth, [0, 0, 1])

    def test_orbit_basis_is_proper_rotation_including_top_and_side(self):
        for azimuth, elevation in ((0., 0.), (-.55, .35), (0., np.pi / 2), (np.pi / 2, 0.), (2., -np.pi / 2)):
            with self.subTest(azimuth=azimuth, elevation=elevation):
                basis = orbit_basis(azimuth, elevation)
                assert_allclose(basis @ basis.T, np.eye(3), atol=1e-15)
                self.assertAlmostEqual(np.linalg.det(basis), 1.)
        top_xy, top_depth = project_points(np.eye(3), np.zeros(3), 1., 0., np.pi / 2, (2, 2))
        assert_allclose(top_xy, [[2, 1], [1, 1], [1, 0]], atol=1e-15)
        assert_allclose(top_depth, [0, 1, 0], atol=1e-15)

    def test_fit_contains_rotated_scene_and_centers_projected_bounds(self):
        points = np.array([[-4., 0., 2.], [1., 2., 6.], [2., -3., 1.], [4., 1., 5.]])
        viewport = np.array([900., 550.])
        for azimuth, elevation in ((0., 0.), (-.55, .35), (0., np.pi / 2), (np.pi / 2, 0.)):
            view = fit_view(points, viewport, azimuth, elevation, padding=.15)
            xy, _ = project_points(points, view["center"], view["scale"], azimuth, elevation, viewport)
            self.assertTrue(np.all(xy >= viewport * .15 - 1e-10))
            self.assertTrue(np.all(xy <= viewport * .85 + 1e-10))
            assert_allclose((xy.min(axis=0) + xy.max(axis=0)) / 2, viewport / 2.)

    def test_fit_degenerate_points_is_finite_and_empty_projection_allowed(self):
        points = np.repeat([[2., 3., 4.]], 3, axis=0)
        view = fit_view(points, (800, 600))
        self.assertTrue(np.isfinite(view["scale"]))
        self.assertGreater(view["scale"], 0)
        xy, _ = project_points(points, view["center"], view["scale"], 0, 0, (800, 600))
        assert_allclose(xy, [[400, 300]] * 3)
        xy, depth = project_points(np.empty((0, 3)), np.zeros(3), 1., 0, 0, (10, 10))
        self.assertEqual(xy.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))

    def test_bad_view_parameters_are_explicitly_rejected(self):
        calls = [
            lambda: orbit_basis(np.nan, 0),
            lambda: fit_view([], (100, 100)),
            lambda: fit_view([[0, 0, 0]], (100, 0)),
            lambda: fit_view([[0, 0, 0]], (100, 100), padding=.5),
            lambda: fit_view([[0, 0, 0]], (100, 100), min_span=0),
            lambda: project_points([[0, 0, 0]], [0, 0], 1., 0, 0, (100, 100)),
            lambda: project_points([[0, 0, 0]], [0, 0, 0], -1., 0, 0, (100, 100)),
            lambda: project_points([[0, 0, np.inf]], [0, 0, 0], 1., 0, 0, (100, 100)),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(SceneError):
                call()


if __name__ == "__main__":
    unittest.main()
