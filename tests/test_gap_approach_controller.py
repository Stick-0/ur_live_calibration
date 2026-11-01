"""Active gap changes exercise the production planner, with no robot IO."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest

import numpy as np

from urcal.control.geometry import pose_matrix
from urcal.control.runtime import parse_args
from urcal.control.scene_view import task_path
from test_cooperative_controller import accept, packet, scene


def following(**overrides):
    options = dict(base_motion_mode="pose", leader_response="bounded",
                   target_tolerance_m=.05, target_tolerance_deg=10.)
    options.update(overrides)
    engine, bases, states = scene(**options)
    accept(engine, bases)
    engine.commit(engine.prepare(10., states, .04, active=True))
    return engine, bases, states


class GapApproachControllerTests(unittest.TestCase):
    def test_start_preserves_command_history_travel_origin_and_offset_reference(self):
        for z_offset in (-.02, .02):
            engine, bases, states = following()
            engine.tracking_offset = pose_matrix([.008, -.006, z_offset, .02, 0., .01])
            expected = engine.path.pose(0.) @ engine.tracking_offset
            attributes = ("previous_twist", "previous_qdot", "previous_compensation",
                          "previous_compensation_qdot", "previous_feedback",
                          "previous_base_compensation", "previous_base_compensation_qdot")
            for index, attribute in enumerate(attributes):
                getattr(engine, attribute)[:] = .001 * (index + 1)
            history = {attribute: getattr(engine, attribute).copy() for attribute in attributes}
            origin = engine.initial_tcp.copy()
            gate, estimator = engine.gate, engine.estimator
            engine.previous_task_scale = .4
            engine.handle_gap_approach("start", enabled=True)
            np.testing.assert_allclose(engine.path.pose(0.) @ engine.tracking_offset, expected)
            np.testing.assert_allclose(engine.path.pose(1.)[:2, 3], expected[:2, 3])
            np.testing.assert_allclose(engine.path.pose(1.)[:3, :3], expected[:3, :3])
            self.assertEqual(engine.path.pose(1.)[2, 3], .15)
            for attribute, values in history.items():
                np.testing.assert_array_equal(getattr(engine, attribute), values)
            np.testing.assert_array_equal(engine.initial_tcp, origin)
            self.assertIs(engine.gate, gate)
            self.assertIs(engine.estimator, estimator)
            self.assertEqual(engine.previous_task_scale, .4)

    def test_prepare_without_send_never_commits_approach_progress(self):
        engine, bases, states = following()
        engine.handle_gap_approach("start", enabled=True)
        proposal = engine.prepare(10., states, .04, active=True)
        self.assertGreater(proposal.next_progress, 0.)
        self.assertEqual(engine.progress, 0.)
        self.assertEqual(engine.gap_approach.info()["progress"], 0.)
        engine.commit(proposal)
        self.assertEqual(engine.progress, proposal.next_progress)
        self.assertEqual(engine.gap_approach.info()["progress"], proposal.next_progress)

    def test_pause_resume_and_duplicate_start_do_not_reset_reference(self):
        engine, bases, states = following()
        engine.handle_gap_approach("start", enabled=True)
        engine.commit(engine.prepare(10., states, .04, active=True))
        reference, path, progress = engine.path.pose(engine.progress), engine.path, engine.progress
        engine.handle_gap_approach("pause", enabled=False)
        engine.handle_gap_approach("start", enabled=True)
        self.assertEqual(engine.gap_approach.state, "paused")
        proposal = engine.prepare(10., states, .04, active=True)
        self.assertEqual(proposal.next_progress, progress)
        engine.handle_gap_approach("resume", enabled=True)
        engine.handle_gap_approach("resume", enabled=True)
        self.assertIs(engine.path, path)
        np.testing.assert_array_equal(engine.path.pose(engine.progress), reference)
        self.assertEqual(engine.gap_approach.state, "running")

    def test_recovery_preserves_path_paused_progress_and_folded_offset(self):
        engine, bases, states = following()
        engine.handle_gap_approach("start", enabled=True)
        engine.commit(engine.prepare(10., states, .04, active=True))
        path, progress, origin = engine.path, engine.progress, engine.initial_tcp.copy()
        engine.reset_after_stop()
        self.assertEqual(engine.gap_approach.state, "paused")
        self.assertTrue(engine.task_paused)
        self.assertIs(engine.path, path)
        self.assertEqual(engine.progress, progress)
        np.testing.assert_array_equal(engine.tracking_offset, np.eye(4))
        np.testing.assert_array_equal(engine.initial_tcp, origin)
        self.assertEqual((engine.path.pose(1.) @ engine.tracking_offset)[2, 3], .15)
        engine.apply_target({"type": "target", "gap": .3, "path": "hold"})
        self.assertEqual(engine.gap_approach.state, "idle")
        self.assertFalse(engine.task_paused)

    def test_invalid_modes_or_gaps_cannot_change_an_active_path(self):
        for overrides in ({"path": "circle"}, {"min_gap": .2}, {"gap": .15},
                          {"base_motion_mode": "velocity"}, {"leader_response": "strict"}):
            engine, _, _ = following(**overrides) if overrides.get("gap") != .15 else following()
            if overrides.get("gap") == .15:
                engine.apply_target({"type": "target", "gap": .15, "path": "hold"})
                engine.tracking_offset = np.eye(4)
            path = engine.path
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                engine.handle_gap_approach("start", enabled=True)
            self.assertIs(engine.path, path)
            self.assertEqual(engine.gap_approach.state, "idle")
        engine, _, _ = following()
        with self.assertRaises(ValueError):
            engine.handle_gap_approach("start", enabled=False)

    def test_estimated_gap_and_commanded_gap_are_independent_signed_axial_values(self):
        engine, bases, states = following()
        states[engine.follower].tcp[2, 3] += .01
        state = engine.gap_approach_status(enabled=True, states=states,
                                          snapshot=SimpleNamespace(base_poses=bases))
        self.assertAlmostEqual(state["commanded_gap_m"], .3)
        self.assertAlmostEqual(state["estimated_gap_m"], .31)
        self.assertEqual(state["target_gap_m"], .15)

    def test_transient_bad_calibration_holds_approach_clock_without_latching_pause(self):
        engine, bases, states = following(live_calibration_recovery=True)
        engine.handle_gap_approach("start", enabled=True)
        engine.commit(engine.prepare_live(10., states, .04))
        path, progress = engine.path, engine.progress
        engine.ingest(packet(engine, bases, 10.04, 2, reason="occlusion"), 10.04,
                      active_recovery=True)
        self.assertEqual(engine.gap_approach.state, "running")
        self.assertFalse(engine.task_paused)
        self.assertIs(engine.path, path)
        status = engine.gap_approach_status(enabled=True)
        self.assertTrue(status["enabled"], "the operator must still be able to pause")
        self.assertTrue(status["temporary_hold"])
        self.assertIn("automatically", status["reason"])
        held = engine.prepare_live(10.04, states, .04)
        self.assertEqual(held.decision.task_scale, 0.)
        self.assertEqual(held.next_progress, progress)
        self.assertFalse(held.record["calibration_braking"])
        engine.commit(held)
        self.assertTrue(engine.ingest(packet(engine, bases, 10.08, 3), 10.08,
                                     active_recovery=True))
        resumed = engine.prepare_live(10.08, states, .04)
        self.assertGreater(resumed.next_progress, progress)
        self.assertIs(engine.path, path)
        self.assertFalse(engine.gap_approach_status(enabled=True)["temporary_hold"])

    def test_bad_calibration_without_live_recovery_still_requires_explicit_resume(self):
        engine, bases, states = following(live_calibration_recovery=False)
        engine.handle_gap_approach("start", enabled=True)
        engine.ingest(packet(engine, bases, 10.04, 2, reason="occlusion"), 10.04)
        self.assertEqual(engine.gap_approach.state, "paused")
        self.assertTrue(engine.task_paused)
        self.assertFalse(engine.gap_approach_status(enabled=True)["temporary_hold"])
        engine.ingest(packet(engine, bases, 10.08, 3), 10.08)
        self.assertEqual(engine.gap_approach.state, "paused")
        self.assertEqual(engine.prepare(10.08, states, .04, active=True).next_progress, engine.progress)

    def test_path_display_starts_at_the_effective_reference(self):
        engine, _, _ = following()
        engine.tracking_offset = pose_matrix([.02, -.01, .015, 0., 0., .1])
        engine.handle_gap_approach("start", enabled=True)
        leader = pose_matrix([.1, .2, .8, 0., .3, 0.])
        task = {"leader": engine.leader, "kind": engine.path.kind,
                "gap": engine.gap, "delta": engine.path.delta, "radius": engine.path.radius,
                "duration": engine.path.duration, "start_pose": engine.path.start}
        points = task_path({"task": task, "robots": {engine.leader: {"flange_C": leader}}})[engine.follower]
        np.testing.assert_allclose(points[0], (leader @ engine.path.pose(0.))[:3, 3])
        np.testing.assert_allclose(points[-1], (leader @ engine.path.pose(1.))[:3, 3])

    def test_cli_approach_target_speed_is_positive_and_defaults_to_5_mm_s(self):
        args = parse_args(["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"])
        self.assertEqual(args.approach_speed, .005)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10", "--approach_speed", value])


if __name__ == "__main__":
    unittest.main()
