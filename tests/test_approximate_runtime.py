"""Approximate flange goals in live runtimes, with no hardware or display."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.geometry import flange_from_tcp, pose_matrix
from urcal.control.policy import ControlFault, bounded_twist
from urcal.control.runtime import parse_args
from urcal.control.scene import SceneError, build_scene
from urcal.cooperative.runtime import cooperative_gui_scene
from urcal.cooperative.task import RelativePath
import tests.test_control_scene as scene_fixtures
import tests.test_live_alignment_runtime as alignment_fixtures


def arguments(*extra):
    return parse_args(["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10", *extra])


class ApproximateOptionsTests(unittest.TestCase):
    def test_new_cli_defaults_and_explicit_exact_mode(self):
        current = arguments()
        self.assertEqual((current.target_tolerance_m, current.target_tolerance_deg), (.05, 10.))
        exact = arguments("--target_tolerance_m", "0", "--target_tolerance_deg", "0")
        self.assertEqual((exact.target_tolerance_m, exact.target_tolerance_deg), (0., 0.))

    def test_invalid_or_tracking_cap_sized_tolerances_are_rejected(self):
        for key, value in (("--target_tolerance_m", "nan"), ("--target_tolerance_deg", "inf"),
                           ("--target_tolerance_m", "-0.01"), ("--target_tolerance_deg", "-1"),
                           ("--target_tolerance_m", "0.2"), ("--target_tolerance_deg", "25")):
            with self.subTest(key=key, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(key, value)


class ApproximatePosePolicyTests(unittest.TestCase):
    def command(self, current, target, *, offset=None, previous=None, **kwargs):
        return bounded_twist(current, target, np.zeros(6) if previous is None else previous, .04,
                             position_tolerance_m=.05, rotation_tolerance_rad=np.deg2rad(10.),
                             tcp_offset=np.eye(4) if offset is None else offset, **kwargs)

    def test_inside_flange_tube_is_zero_even_with_nonzero_rotated_tcp(self):
        offset = pose_matrix([.1, .04, .08, .2, -.1, .3])
        flange = pose_matrix([.2, -.1, .3, .1, .2, -.1])
        goal = flange @ pose_matrix([.03, .02, 0, 0, 0, np.deg2rad(7.)])
        np.testing.assert_allclose(self.command(flange @ offset, goal @ offset, offset=offset), np.zeros(6), atol=1e-15)

    def test_long_tcp_does_not_turn_allowed_flange_rotation_into_position_error(self):
        offset = pose_matrix([1., 0, 0, 0, 0, 0])
        goal = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(9.)]) @ offset
        self.assertGreater(np.linalg.norm(goal[:3, 3] - offset[:3, 3]), .1)
        np.testing.assert_allclose(self.command(offset, goal, offset=offset), np.zeros(6))

    def test_feedback_is_continuous_at_position_tube_boundary(self):
        for distance, expected in ((.049999, 0.), (.05, 0.), (.050001, .000001), (.052, .002)):
            with self.subTest(distance=distance):
                command = self.command(np.eye(4), pose_matrix([distance, 0, 0, 0, 0, 0]))
                np.testing.assert_allclose(command, [expected, 0, 0, 0, 0, 0], atol=1e-14)

    def test_flange_angular_feedback_is_shifted_to_tcp_point(self):
        offset = pose_matrix([.2, .05, 0, 0, 0, 0])
        goal = pose_matrix([0, 0, 0, 0, 0, np.deg2rad(12.)]) @ offset
        command = self.command(offset, goal, offset=offset, max_linear=1., max_angular=1.,
                               max_accel=100., max_angular_accel=100.)
        angular = np.array([0., 0., np.deg2rad(2.)])
        np.testing.assert_allclose(command[3:], angular, atol=1e-14)
        np.testing.assert_allclose(command[:3], np.cross(angular, offset[:3, 3]), atol=1e-14)

    def test_entering_tube_still_respects_deceleration_limit(self):
        previous = np.array([.01, 0, 0, 0, 0, .05])
        command = self.command(np.eye(4), np.eye(4), previous=previous)
        np.testing.assert_allclose(command, [.006, 0, 0, 0, 0, .03])

    def test_full_tracking_limit_is_checked_before_tolerance_subtraction(self):
        with self.assertRaisesRegex(ControlFault, "flange target exceeds"):
            self.command(np.eye(4), pose_matrix([.11, 0, 0, 0, 0, 0]))

class ApproximateSceneTests(unittest.TestCase):
    def test_nominal_error_is_not_erased_by_selected_feasible_command(self):
        packet, models = scene_fixtures.fixture()
        actual = packet["bases"]["ur5e"] @ packet["robots"]["ur5e"]["tcp"]
        nominal = actual.copy()
        nominal[0, 3] += .03
        packet["goals"] = {"ur5e": actual.copy()}
        packet["nominal_goals"] = {"ur5e": nominal.copy()}
        packet["command_goals"] = {"ur5e": actual.copy()}
        scene = build_scene(packet, models)
        robot = scene["robots"]["ur5e"]
        self.assertAlmostEqual(robot["error_m"], .03)
        self.assertAlmostEqual(robot["command_adjustment_m"], .03)
        np.testing.assert_allclose(robot["target_C"], nominal)
        np.testing.assert_allclose(robot["command_C"], actual)
        packet["command_goals"]["ur5e"][:] = 999
        np.testing.assert_allclose(robot["command_C"], actual)

    def test_command_scene_validates_proper_pose(self):
        packet, models = scene_fixtures.fixture()
        packet["command_goals"] = {"ur5e": np.full((4, 4), np.nan)}
        with self.assertRaises(SceneError):
            build_scene(packet, models)

    def test_follow_command_marker_uses_flange_and_keeps_nominal_goal(self):
        packet, _ = scene_fixtures.fixture()
        offset = pose_matrix([.04, .02, .1, .2, -.1, .3])
        offsets = {name: offset for name in packet["robots"]}
        states = {name: SimpleNamespace(tcp=item["tcp"], joints=item["q"], robot_timestamp=50.)
                  for name, item in packet["robots"].items()}
        readers = {name: SimpleNamespace(last_advance=100.) for name in states}
        engine = SimpleNamespace(invalid_since=None, last_rejection="", leader="ur10e", follower="ur5e",
                                 path=RelativePath(np.eye(4)), offsets=offsets, gap=.3, progress=0.)
        nominal = pose_matrix([.3, .2, 1., .2, .3, .1])
        next_tcp = states["ur5e"].tcp.copy()
        proposal = SimpleNamespace(goals={"ur5e": nominal}, record={"bases": packet["bases"],
                                   "measurement_age": .1, "next_commanded_target_tcp": next_tcp})
        scene = cooperative_gui_scene(100., engine, states, readers, {}, .15, proposal=proposal)
        np.testing.assert_allclose(scene["goals"]["ur5e"], nominal)
        np.testing.assert_allclose(scene["command_goals"]["ur5e"],
                                   packet["bases"]["ur5e"] @ flange_from_tcp(next_tcp, offset))


class ApproximateRuntimeIntegrationTests(unittest.TestCase):
    def test_selected_alignment_endpoint_and_nominal_task_are_both_logged_and_shown(self):
        selected = {"ur5e": pose_matrix([.025, 0, .3, 0, np.pi, 0])}
        result = alignment_fixtures.LiveAlignmentRuntimeTests().run_fake(
            selected_targets=selected, stop_after=2, done_after=99)
        self.assertIsNone(result.caught)
        armed = next(record for record in result.records if record["kind"] == "armed")
        np.testing.assert_allclose(armed["tcp_goals"]["ur5e"], selected["ur5e"])
        np.testing.assert_allclose(armed["nominal_tcp_goals"]["ur5e"], result.alignment.nominal_targets["ur5e"])
        self.assertFalse(np.allclose(armed["tcp_goals"]["ur5e"], armed["nominal_tcp_goals"]["ur5e"]))
        self.assertEqual(armed["target_adjustments"], result.alignment.target_adjustments)
        status = alignment_fixtures.runtime.take_latest(result.gui_statuses)
        np.testing.assert_allclose(status["scene"]["command_goals"]["ur5e"], selected["ur5e"])
        np.testing.assert_allclose(status["scene"]["nominal_goals"]["ur5e"], result.alignment.nominal_targets["ur5e"])

    def test_fresh_handoff_allows_small_nominal_target_shift_and_rejects_outside_tube(self):
        original_config = alignment_fixtures.config
        def tolerant_config(execute=False):
            args = original_config(execute)
            args.target_tolerance_m, args.target_tolerance_deg = .05, 10.
            args.handoff_max_shift_m, args.handoff_max_shift_deg = .05, 10.
            return args
        fixture = alignment_fixtures.LiveAlignmentRuntimeTests()
        for displacement in (.02, .06):
            with self.subTest(displacement=displacement), patch.object(alignment_fixtures, "config", tolerant_config):
                result = fixture.run_fake(done_after=1, stop_after=8,
                                           calibration_events={index: displacement for index in range(2, 7)})
                if displacement == .02:
                    self.assertIsNone(result.caught)
                    result.session.send.assert_called_once()
                    self.assertTrue(any(record["kind"] == "alignment_complete" for record in result.records))
                else:
                    self.assertIsNone(result.caught)
                    self.assertIn("handoff allowance", alignment_fixtures.dual.take_latest(result.gui_statuses)["message"])
                    result.session.send.assert_not_called()
                    self.assertFalse(any(record["kind"] == "alignment_complete" for record in result.records))


if __name__ == "__main__":
    unittest.main()
