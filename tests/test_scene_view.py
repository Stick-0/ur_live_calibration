"""Canvas scene rendering and interaction without a display or robot drivers."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from urcal.control.geometry import facing_follower, pose_matrix
from urcal.control.scene import build_scene, orbit_basis
from urcal.control.scene_view import (AXIS_COLORS, COLORS, DRAFT_COLOR, TARGET_COLOR,
                                      RobotSceneView, task_path)


class FakeCanvas:
    def __init__(self, width=800, height=600):
        self.width, self.height = width, height
        self.items = []
        self.clear_count = 0

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, tag):
        assert tag == "all"
        self.items.clear()
        self.clear_count += 1

    def _add(self, kind, coords, options):
        assert np.isfinite(coords).all(), (kind, coords)
        self.items.append({"kind": kind, "coords": np.asarray(coords), **options})
        return len(self.items)

    def create_line(self, *coords, **options):
        return self._add("line", coords, options)

    def create_oval(self, *coords, **options):
        return self._add("oval", coords, options)

    def create_text(self, *coords, **options):
        return self._add("text", coords, options)


class Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class Model:
    def points(self, q):
        return np.array([[0., 0., 0.], [.1, .1, .25], [0., .1, .45]])


def snapshot():
    bases = {"ur5e": pose_matrix([-.4, 0, 1.5, 0, 0, 0]),
             "ur10e": pose_matrix([.4, 0, 1.5, .2, -.1, .3])}
    robots = {name: {"q": np.zeros(6), "tcp": pose_matrix([0., .1, .5, .1, .2, -.1]),
                     "tcp_offset": pose_matrix([0, 0, .05, 0, 0, 0])} for name in bases}
    return {"at": 100., "measurement_age": .1, "bases": bases, "robots": robots,
            "goals": {"ur5e": pose_matrix([-.2, .1, 1.9, .2, .3, .1])},
            "task": {"leader": "ur10e", "kind": "line", "gap": .3,
                     "delta": [.08, -.02, 0, 0, 0, .2], "radius": .02, "duration": 10.}}


def fake_view():
    view = object.__new__(RobotSceneView)
    view.canvas = FakeCanvas()
    view._details = Value()
    view.max_measurement_age, view.max_robot_age = .5, .15
    view.scene = None
    view._models = {name: Model() for name in COLORS}
    view._draft, view._draft_path, view._path = {}, {}, {}
    view._missing = True
    view._notice = "Waiting for calibration"
    view._dirty, view._need_fit, view._last_draw = True, True, 0.
    view.azimuth, view.elevation = -.55, .35
    view.center, view.scale = np.zeros(3), 250.
    view._drag = None
    return view


class PathTests(unittest.TestCase):
    def test_line_path_is_in_live_leader_flange_coordinates(self):
        packet = snapshot()
        scene = build_scene(packet, {name: Model() for name in COLORS})
        path = task_path(scene)
        self.assertEqual(set(path), {"ur5e"})
        self.assertEqual(path["ur5e"].shape, (49, 3))
        leader = scene["robots"]["ur10e"]["flange_C"]
        start = facing_follower(leader, .3)
        assert_allclose(path["ur5e"][0], start[:3, 3])
        assert_allclose(path["ur5e"][-1], start[:3, 3] + leader[:3, :3] @ [.08, -.02, 0])
        self.assertEqual(packet["task"], snapshot()["task"])

    def test_circle_closes_and_hold_has_no_path(self):
        scene = build_scene(snapshot(), {name: Model() for name in COLORS})
        circle = {**scene["task"], "kind": "circle"}
        path = task_path(scene, circle)["ur5e"]
        assert_allclose(path[0], path[-1], atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(path[24] - path[0]), .04)
        self.assertEqual(task_path(scene, {"kind": "hold"}), {})


class RenderTests(unittest.TestCase):
    def test_waiting_message_without_scene(self):
        view = fake_view()
        view.draw(100.)
        self.assertEqual(len(view.canvas.items), 1)
        self.assertIn("Waiting for both robot poses", view.canvas.items[0]["text"])

    def test_live_current_target_path_and_draft_have_distinct_visuals(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        draft = pose_matrix([-.1, .2, 2., -.1, .2, .3])
        view.set_draft({"ur5e": draft}, {"ur5e": np.array([[0., 0., 2.], [.1, 0, 2], [.2, .1, 2.]])})
        view.draw(100.01)
        lines = [item for item in view.canvas.items if item["kind"] == "line"]
        for color in COLORS.values():
            self.assertEqual(sum(item.get("width") == 8 and item["fill"] == color for item in lines), 2)
        for color in AXIS_COLORS:
            self.assertEqual(sum(item.get("width") == 3 and item.get("arrow") == "last"
                                 and item["fill"] == color for item in lines), 2)
            self.assertTrue(any(item.get("dash") == (4, 3) and item["fill"] == color for item in lines))
        self.assertEqual(sum(item["fill"] == "#986fc0" and item.get("dash") == (3, 3) for item in lines), 48)
        self.assertEqual(sum(item["fill"] == DRAFT_COLOR and item.get("dash") == (4, 3) for item in lines), 2)
        outlines = [item["outline"] for item in view.canvas.items if item["kind"] == "oval"]
        self.assertIn(TARGET_COLOR, outlines)
        self.assertIn(DRAFT_COLOR, outlines)
        labels = [item["text"] for item in view.canvas.items if item["kind"] == "text"]
        for label in ("UR5E NOW", "UR10E NOW", "UR5E TARGET", "UR5E EDIT", "LIVE ESTIMATED POSE"):
            self.assertIn(label, labels)
        self.assertIn("error", view._details.get())
        self.assertIn("GOAL", view._details.get())

    def test_missing_snapshot_retains_geometry_but_marks_stale(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        previous = view.scene
        view.set_snapshot(None)
        self.assertIs(view.scene, previous)
        self.assertTrue(view._stale(100.01))
        view.draw(100.01)
        links = [item for item in view.canvas.items if item["kind"] == "line" and item.get("width") == 8]
        self.assertEqual(len(links), 4)
        self.assertTrue(all(item["fill"] == "#64748b" for item in links))
        self.assertTrue(any("STALE / UNAVAILABLE" in item.get("text", "") for item in view.canvas.items))
        self.assertIn("last view retained", view._details.get())

    def test_stale_tool_line_is_dimmed_with_links(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        view.draw(100.2)
        tools = [item for item in view.canvas.items if item["kind"] == "line"
                 and item.get("width") == 3 and "arrow" not in item]
        self.assertEqual(len(tools), 2)
        self.assertTrue(all(item["fill"] == "#64748b" for item in tools))

    def test_distant_links_draw_before_near_links(self):
        view = fake_view()
        packet = snapshot()
        packet["bases"]["ur5e"] = pose_matrix([0, 0, 1, 0, 0, 0])
        packet["bases"]["ur10e"] = pose_matrix([0, 0, 3, 0, 0, 0])
        view.set_snapshot(packet)
        view.preset(0, 0)
        view.draw(100.01)
        order = [item["fill"] for item in view.canvas.items
                 if item["kind"] == "line" and item.get("width") == 8]
        self.assertEqual(order, [COLORS["ur10e"], COLORS["ur10e"], COLORS["ur5e"], COLORS["ur5e"]])

    def test_invalid_snapshot_keeps_previous_valid_geometry(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        previous = view.scene
        invalid = snapshot()
        invalid["robots"]["ur5e"]["tcp"][0, 0] = 7
        view.set_snapshot(invalid)
        self.assertIs(view.scene, previous)
        self.assertTrue(view._missing)
        self.assertIn("rejected", view._notice)
        view.set_snapshot(snapshot())
        self.assertFalse(view._missing)

    def test_ages_invalid_calibration_and_missing_age_expire_display(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        self.assertFalse(view._stale(100.01))
        self.assertTrue(view._stale(99.99))
        self.assertTrue(view._stale(100.151))
        packet = snapshot()
        packet["measurement_age"] = {"ur5e": .49, "ur10e": .1}
        view.set_snapshot(packet)
        self.assertTrue(view._stale(100.02))
        packet["measurement_age"] = None
        view.set_snapshot(packet)
        self.assertTrue(view._stale(100.01))
        packet["measurement_age"] = .1
        packet["calibration_valid"] = False
        view.set_snapshot(packet)
        self.assertTrue(view._stale(100.01))

    def test_idle_refresh_eventually_redraws_expiration(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        view.refresh(100.)
        self.assertEqual(view.canvas.clear_count, 1)
        view.refresh(100.01)
        self.assertEqual(view.canvas.clear_count, 1)
        view.refresh(100.21)
        self.assertEqual(view.canvas.clear_count, 2)
        self.assertTrue(any("STALE / UNAVAILABLE" in item.get("text", "") for item in view.canvas.items))

    def test_frozen_camera_age_does_not_expire_fresh_encoder_geometry(self):
        view = fake_view()
        packet = snapshot()
        packet.update(calibration_mode="frozen", calibration_captured_at=20., measurement_age=80.)
        view.set_snapshot(packet)
        self.assertFalse(view._stale(100.01))
        self.assertEqual(view.scene["measurement_age"], 80.)
        view.draw(100.01)
        labels = [item["text"] for item in view.canvas.items if item["kind"] == "text"]
        self.assertIn("CAPTURED CALIBRATION + LIVE JOINTS", labels)
        self.assertTrue(any("camera and bases assumed fixed" in text for text in labels))
        self.assertNotIn("LIVE ESTIMATED POSE", labels)
        self.assertTrue(view._stale(100.151))
        self.assertTrue(view._stale(99.99))

    def test_frozen_mode_does_not_hide_invalid_or_missing_camera_state(self):
        view = fake_view()
        packet = snapshot()
        packet.update(calibration_mode="frozen", measurement_age=80.)
        for change in ({"calibration_valid": False}, {"measurement_age": None}):
            view.set_snapshot({**packet, **change})
            self.assertTrue(view._stale(100.01))
        view.set_snapshot(packet)
        view.set_snapshot(None)
        self.assertTrue(view._stale(100.01))

    def test_live_camera_age_expiry_returns_after_frozen_alignment(self):
        view = fake_view()
        packet = snapshot()
        packet.update(calibration_mode="frozen", measurement_age={"ur5e": 80., "ur10e": 79.})
        view.set_snapshot(packet)
        self.assertFalse(view._stale(100.01))
        packet["calibration_mode"] = "live"
        view.set_snapshot(packet)
        self.assertTrue(view._stale(100.01))

    def test_simulation_keeps_existing_caption(self):
        view = fake_view()
        view._simulation = True
        packet = snapshot()
        packet["calibration_mode"] = "frozen"
        view.set_snapshot(packet)
        view.draw(100.01)
        labels = [item["text"] for item in view.canvas.items if item["kind"] == "text"]
        self.assertIn("SIMULATED POSE", labels)
        self.assertIn("Camera coordinates (m) · virtual joints + snapshot / virtual bases", labels)
        self.assertNotIn("CAPTURED CALIBRATION + LIVE JOINTS", labels)


class InteractionTests(unittest.TestCase):
    def test_fit_covers_live_applied_and_draft_paths(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        view.set_draft({"ur5e": pose_matrix([3., 2., 1., 0, 0, 0])},
                       {"ur5e": np.array([[2, 2, 1], [3, 3, 2]])})
        view.fit()
        for points in ([pose[:3, 3] for pose in view._draft.values()], *view._path.values(), *view._draft_path.values()):
            xy, _ = view._project(points)
            self.assertTrue(np.all(xy > 0))
            self.assertTrue(np.all(xy < np.array(view._viewport())))

    def test_snapshot_refresh_does_not_reset_manual_orbit_or_zoom(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        view.azimuth, view.elevation = 1., .2
        view.center = np.array([.2, .4, 1.8])
        view.scale = 433.
        packet = snapshot()
        packet["at"] = 100.1
        view.set_snapshot(packet)
        self.assertEqual((view.azimuth, view.elevation, view.scale), (1., .2, 433.))
        assert_allclose(view.center, [.2, .4, 1.8])

    def test_orbit_pan_zoom_and_preset_are_view_only(self):
        view = fake_view()
        view.set_snapshot(snapshot())
        original = deepcopy(view.scene["robots"]["ur5e"]["flange_C"])
        view._start_drag(SimpleNamespace(x=10, y=20), "orbit")
        initial_azimuth = view.azimuth
        view._move_drag(SimpleNamespace(x=30, y=0))
        self.assertAlmostEqual(view.azimuth, initial_azimuth + .16)
        view._move_drag(SimpleNamespace(x=30, y=-10000))
        self.assertEqual(view.elevation, np.pi / 2)
        view._end_drag()
        self.assertIsNone(view._drag)
        view.preset(0., 0.)
        center, scale = view.center.copy(), view.scale
        view._start_drag(SimpleNamespace(x=0, y=0), "pan")
        view._move_drag(SimpleNamespace(x=20, y=10))
        assert_allclose(view.center, center - orbit_basis(0, 0).T @ [20 / scale, 10 / scale, 0])
        view.zoom(1e10)
        self.assertEqual(view.scale, 20000.)
        view.zoom(1e-20)
        self.assertEqual(view.scale, 20.)
        self.assertEqual(view._wheel(SimpleNamespace(delta=0)), "break")
        assert_allclose(view.scene["robots"]["ur5e"]["flange_C"], original)


if __name__ == "__main__":
    unittest.main()
