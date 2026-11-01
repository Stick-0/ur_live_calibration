"""Graph telemetry uses current TCPs and accepted targets, never predicted steps."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from urcal.control.geometry import flange_from_tcp, inverse, pose_matrix
from urcal.control.tracking_error import position_error_sample, transformed_tcp_target


def scene():
    return {"at": 10., "calibration_valid": True, "measurement_age": .1,
            "bases": {"ur5e": pose_matrix([.3, .1, 0., 0., 0., np.pi / 2]),
                      "ur10e": pose_matrix([-.2, .4, .1, .1, -.2, .3])},
            "robots": {"ur5e": {"tcp": pose_matrix([.1, -.2, .5, .2, .1, -.3]),
                                 "tcp_offset": pose_matrix([.03, .02, .1, .1, 0., .1])},
                       "ur10e": {"tcp": pose_matrix([.2, .1, .4, .3, -.2, .1]),
                                  "tcp_offset": pose_matrix([-.01, .02, .08, 0., .1, 0.])}}}


def sample(value, **kwargs):
    values = dict(now=10.04, leader="ur10e", follower="ur5e", phase="tracking",
                  relative_target=pose_matrix([.01, -.02, .3, np.pi, 0., .1]),
                  target_source="tracking_accepted_relative")
    values.update(kwargs)
    return position_error_sample(value, **values)


class TrackingErrorTests(unittest.TestCase):
    def test_signed_tcp_error_uses_follower_base_axes_and_both_offsets(self):
        value = scene()
        original = deepcopy(value)
        relative = pose_matrix([.01, -.02, .3, np.pi, 0., .1])
        target = (inverse(value["bases"]["ur5e"]) @ value["bases"]["ur10e"]
                  @ flange_from_tcp(value["robots"]["ur10e"]["tcp"], value["robots"]["ur10e"]["tcp_offset"])
                  @ relative @ value["robots"]["ur5e"]["tcp_offset"])
        result = sample(value, relative_target=relative)
        self.assertTrue(result["valid"])
        self.assertEqual(result["at"], 10.)
        np.testing.assert_allclose(result["error_m"], target[:3, 3] - value["robots"]["ur5e"]["tcp"][:3, 3])
        for name in value["robots"]:
            for key in ("tcp", "tcp_offset"):
                np.testing.assert_array_equal(value["robots"][name][key], original["robots"][name][key])

    def test_accepted_offset_changes_target_and_predicted_scene_goals_are_ignored(self):
        value = scene()
        nominal = pose_matrix([0., 0., .3, np.pi, 0., 0.])
        accepted = nominal @ pose_matrix([.03, .01, -.02, .1, 0., 0.])
        value["command_goals"] = {"ur5e": np.eye(4) * 999.}
        value["goals"] = {"ur5e": np.eye(4) * 999.}
        result = sample(value, relative_target=accepted)
        expected = transformed_tcp_target(value, "ur10e", "ur5e", accepted)
        np.testing.assert_allclose(result["error_m"], expected[:3, 3] - value["robots"]["ur5e"]["tcp"][:3, 3])
        self.assertFalse(np.allclose(result["error_m"], sample(value, relative_target=nominal)["error_m"]))

    def test_alignment_uses_selected_native_tcp_endpoint_with_frozen_calibration(self):
        value = scene()
        value.update(calibration_mode="frozen", measurement_age=80.)
        target = value["robots"]["ur5e"]["tcp"].copy()
        target[:3, 3] += [.03, -.02, .01]
        result = sample(value, target_tcp=target, phase="alignment", target_source="initial_alignment_endpoint")
        self.assertTrue(result["valid"])
        np.testing.assert_allclose(result["error_m"], [.03, -.02, .01], atol=1e-15)

    def test_missing_and_stale_measurements_emit_gaps_without_zero_error(self):
        for value in (None, {**scene(), "at": 9.}, {**scene(), "measurement_age": .51},
                      {**scene(), "calibration_valid": False, "reason": "PnP failed"},
                      {**scene(), "at": 11.}, {**scene(), "measurement_age": float("nan")}):
            with self.subTest(value=value):
                result = sample(value)
                self.assertFalse(result["valid"])
                self.assertNotIn("error_m", result)
                self.assertTrue(result["reason"])

    def test_polling_same_reading_never_relabels_timestamp(self):
        first = sample(scene(), now=10.1)
        second = sample(scene(), now=10.4)
        self.assertEqual(first["at"], second["at"])
        self.assertEqual(first["error_m"], second["error_m"])
        self.assertFalse(sample(scene(), now=10.6)["valid"])

    def test_invalid_transform_and_explicit_fault_are_gaps(self):
        malformed = scene()
        malformed["robots"]["ur5e"]["tcp"][0, 3] = float("nan")
        for value, kwargs in ((malformed, {}), (scene(), {"reason": "robot disconnected"}),
                              (scene(), {"target_tcp": np.zeros((4, 4))})):
            result = sample(value, **kwargs)
            self.assertFalse(result["valid"])
            self.assertNotIn("error_m", result)

    def test_simulation_clock_episode_and_pause_are_preserved(self):
        value = {**scene(), "at": 0.}
        result = sample(value, now=0., clock="simulation", episode=3, phase="paused")
        self.assertTrue(result["valid"])
        self.assertEqual((result["at"], result["clock"], result["episode"], result["phase"]),
                         (0., "simulation", 3, "paused"))


class SimulationTrackingErrorTests(unittest.TestCase):
    def build(self):
        from test_simulation_runtime import options, synthetic_snapshot, line_target
        from urcal.simulation.runtime import SimulationSession
        initial, _ = synthetic_snapshot()
        args = options("follow", "both", "--base_motion_mode", "pose", "--filter_alpha", "1", "--base_pose_window", "0")
        session = SimulationSession(args, initial)
        session.command(line_target())
        self.assertTrue(session.ready, session.message)
        session.command("arm")
        for _ in range(4):
            self.assertTrue(session.tick(), session.fault or session.message)
        return session

    def test_post_step_target_matches_post_step_robot_and_ignores_previous_prediction(self):
        session = self.build()
        status = session.status(wall_now=5000.)
        result = status["position_error"]
        relative = session.engine.path.pose(session.engine.progress)
        if session.engine.tracking_offset is not None:
            relative = relative @ session.engine.tracking_offset
        target = transformed_tcp_target(status["scene"], session.engine.leader, session.engine.follower, relative)
        current = session.plant.states(session.now)[session.engine.follower].tcp
        self.assertTrue(result["valid"])
        self.assertEqual(result["at"], session.elapsed)
        np.testing.assert_allclose(result["error_m"], target[:3, 3] - current[:3, 3])
        session.last_record["next_commanded_target_tcp"] = pose_matrix([9., 8., 7., 0., 0., 0.])
        np.testing.assert_array_equal(session.status(wall_now=6000.)["position_error"]["error_m"], result["error_m"])

    def test_pause_freezes_graph_time_and_reset_changes_episode(self):
        session = self.build()
        before = session.status()["position_error"]
        session.command("sim_pause")
        self.assertFalse(session.tick())
        paused = session.status(wall_now=9000.)["position_error"]
        self.assertEqual(paused["at"], before["at"])
        self.assertEqual(paused["phase"], "paused")
        np.testing.assert_array_equal(paused["error_m"], before["error_m"])
        session.command("sim_reset")
        reset = session.status()["position_error"]
        self.assertEqual(reset["at"], 0.)
        self.assertEqual(reset["episode"], before["episode"] + 1)


class LiveTrackingErrorQueueTests(unittest.TestCase):
    def test_native_endpoint_then_accepted_tracking_target_reach_gui_queue(self):
        import urcal.cooperative.runtime as runtime
        from test_live_alignment_runtime import LiveAlignmentRuntimeTests

        endpoint = pose_matrix([.025, -.01, .29, np.pi, 0., 0.])
        accepted = pose_matrix([.01, -.02, .28, np.pi, 0., 0.])
        published = []
        original_put = runtime.put_latest

        def capture(queue, value):
            if isinstance(value, dict) and "position_error" in value:
                published.append(deepcopy(value["position_error"]))
            return original_put(queue, value)

        def inject_tracking_reference(_read, _clock, _gate, _readers, engine):
            original_prepare = engine.prepare.side_effect

            def prepare(*args, **kwargs):
                proposal = original_prepare(*args, **kwargs)
                proposal.record["tracking_relative_target"] = accepted.copy()
                # Deliberately unrelated one-step prediction: graph must use
                # the accepted relative target with the observed current TCP.
                proposal.record["next_commanded_target_tcp"] = pose_matrix([9., 8., 7., 0., 0., 0.])
                return proposal

            engine.prepare.side_effect = prepare

        case = LiveAlignmentRuntimeTests("test_completion_waits_for_five_new_frames_then_zeroes_histories_before_follow")
        with patch.object(runtime, "put_latest", side_effect=capture):
            result = case.run_fake(done_after=1, stop_after=8,
                                   selected_targets={"ur5e": endpoint},
                                   preflight_hook=inject_tracking_reference,
                                   calibration_events={index: 0. for index in range(2, 7)})
        self.assertIsNone(result.caught)
        alignment = [row for row in published if row["valid"] and row["phase"] == "alignment"]
        tracking = [row for row in published if row["valid"] and row["phase"] == "tracking"]
        self.assertTrue(alignment, published)
        self.assertTrue(tracking, published)
        self.assertEqual(alignment[-1]["target_source"], "initial_alignment_endpoint")
        self.assertEqual(tracking[-1]["target_source"], "tracking_accepted_relative")
        np.testing.assert_allclose(alignment[-1]["error_m"], endpoint[:3, 3])
        np.testing.assert_allclose(tracking[-1]["error_m"], accepted[:3, 3])
        self.assertLess(alignment[-1]["at"], tracking[-1]["at"])
        self.assertEqual(tracking[-1]["clock"], "monotonic")
        self.assertEqual(result.session.send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
