"""Approach command lifecycle through the real follower runtime, with fake IO."""

import unittest
from unittest.mock import patch

from tests import test_live_calibration_runtime as fixtures


def request(action, at):
    return {"type": "gap_approach", "action": action, "requested_at": at}


class LiveGapApproachRuntimeTests(unittest.TestCase):
    run_runtime = fixtures.LiveCalibrationRuntimeTests.run_runtime

    def assert_active_session(self, result):
        self.assertIsNone(result.caught)
        self.assertEqual(result.stops, [])
        self.assertEqual(result.factory.call_count, 1)
        self.assertTrue(result.sends)
        self.assertEqual(len(result.sends), len(result.commits))

    @staticmethod
    def gap_statuses(result):
        return [(at, status["gap_approach"]) for at, status in result.statuses
                if isinstance(status.get("gap_approach"), dict)]

    def test_start_pause_resume_uses_same_tracking_session_and_freezes_only_approach(self):
        actions = {3: "start", 20: "pause", 30: "resume"}
        result = self.run_runtime(
            lambda tick, now: 0., stop_after=65,
            later_commands=lambda tick, now: [request(actions[tick], now)] if tick in actions else [])
        self.assert_active_session(result)
        values = self.gap_statuses(result)
        first = [value for at, value in values if 10.12 <= at < 10.79]
        paused = [value for at, value in values if 10.81 < at < 11.19]
        resumed = [value for at, value in values if at > 11.25]
        self.assertTrue(first and paused and resumed)
        self.assertTrue(all(value["state"] == "running" for value in first))
        self.assertLess(first[-1]["commanded_gap_m"], first[0]["commanded_gap_m"])
        self.assertTrue(all(value["state"] == "paused" for value in paused))
        for value in paused:
            self.assertAlmostEqual(value["commanded_gap_m"], paused[0]["commanded_gap_m"])
        self.assertTrue(all(value["state"] == "running" for value in resumed))
        self.assertLess(resumed[-1]["commanded_gap_m"], paused[-1]["commanded_gap_m"])
        self.assertTrue(any(10.81 < at < 11.19 for at, _, _ in result.sends))
        accepted = [row for row in result.records if row["kind"] == "gap_approach"]
        self.assertEqual(len(accepted), 3)

    def test_preview_request_is_rejected_without_creating_a_motion_session(self):
        result = self.run_runtime(
            lambda tick, now: 0., execute=False, arm=False, stop_after=8,
            later_commands=lambda tick, now: [request("start", now)] if tick == 3 else [])
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 0)
        self.assertEqual(result.sends, [])
        self.assertTrue(any(row["kind"] == "gap_approach_rejected" for row in result.records))
        self.assertTrue(all(value["state"] == "idle" for _, value in self.gap_statuses(result)))

    def test_paused_approach_still_follows_encoder_observed_leader_motion(self):
        plant = {}
        original_scene = fixtures.scene

        def capture_scene(**settings):
            engine, bases, states = original_scene(**settings)
            plant.update(engine=engine, states=states)
            return engine, bases, states

        def commands(tick, now):
            leader = plant["states"][plant["engine"].leader]
            leader.speed[0] = leader.joint_velocity[0] = .01 if 10 <= tick < 20 else 0.
            action = {3: "start", 10: "pause"}.get(tick)
            return [request(action, now)] if action else []

        with patch.object(fixtures, "scene", side_effect=capture_scene):
            result = self.run_runtime(lambda tick, now: 0., later_commands=commands, stop_after=30)
        self.assert_active_session(result)
        paused = [value for at, value in self.gap_statuses(result) if at > 10.41]
        self.assertTrue(paused)
        self.assertTrue(all(value["state"] == "paused" for value in paused))
        for value in paused:
            self.assertAlmostEqual(value["commanded_gap_m"], paused[0]["commanded_gap_m"])
        self.assertGreater(result.states[result.engine.leader].joints[0], .003)
        self.assertTrue(any(twists[result.engine.follower][0] > .001
                            for at, twists, _ in result.sends if 10.45 < at < 10.8))

    def test_invalid_old_future_and_stale_requests_are_nonfatal_and_do_not_start(self):
        cases = {
            "unknown action": lambda now: request("toggle", now),
            "missing timestamp": lambda now: {"type": "gap_approach", "action": "start"},
            "nan timestamp": lambda now: request("start", float("nan")),
            "infinite timestamp": lambda now: request("start", float("inf")),
            "non-numeric timestamp": lambda now: request("start", "bad"),
            "before session epoch": lambda now: request("start", 9.99),
            "future timestamp": lambda now: request("start", now + .1),
            "stale timestamp": lambda now: request("start", now - 1.1),
        }
        for label, command in cases.items():
            with self.subTest(case=label):
                result = self.run_runtime(
                    lambda tick, now: 0., stop_after=36,
                    later_commands=lambda tick, now: [command(now)] if tick == 30 else [])
                self.assert_active_session(result)
                rejected = [row for row in result.records if row["kind"] == "gap_approach_rejected"]
                self.assertEqual(len(rejected), 1)
                self.assertFalse(any(row["kind"] == "gap_approach" for row in result.records))
                self.assertTrue(all(value["state"] == "idle" for _, value in self.gap_statuses(result)))

    def test_duplicate_timestamp_cannot_pause_an_already_running_approach(self):
        requested_at = []

        def commands(tick, now):
            if tick == 3:
                requested_at.append(now)
                return [request("start", now)]
            if tick == 6:
                return [request("pause", requested_at[0])]
            return []

        result = self.run_runtime(lambda tick, now: 0., later_commands=commands, stop_after=20)
        self.assert_active_session(result)
        values = [value for at, value in self.gap_statuses(result) if at > 10.25]
        self.assertTrue(values)
        self.assertTrue(all(value["state"] == "running" for value in values))
        self.assertLess(values[-1]["commanded_gap_m"], values[0]["commanded_gap_m"])
        self.assertEqual(sum(row["kind"] == "gap_approach" for row in result.records), 1)
        self.assertEqual(sum(row["kind"] == "gap_approach_rejected" for row in result.records), 1)

    def test_request_created_before_arm_cannot_start_in_later_session(self):
        def commands(tick, now):
            if tick == 3:
                return ["arm"]
            if tick == 5:
                return [request("start", 10.08)]
            return []

        result = self.run_runtime(lambda tick, now: 0., arm=False, later_commands=commands, stop_after=16)
        self.assert_active_session(result)
        self.assertTrue(any(row["kind"] == "gap_approach_rejected" for row in result.records))
        self.assertTrue(all(value["state"] == "idle" for _, value in self.gap_statuses(result)))

    def test_single_55_mm_outlier_holds_only_approach_until_next_fresh_frame(self):
        result = self.run_runtime(
            lambda tick, now: .0553 if tick == 12 else 0., max_calibration_jump_m=.05,
            later_commands=lambda tick, now: [request("start", now)] if tick == 3 else [], stop_after=45)
        self.assert_active_session(result)
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))
        self.assertFalse(any(row["kind"] == "calibration_reacquired_live" for row in result.records))
        self.assertFalse(any(row.get("calibration_braking") for _, row in result.commits))
        values = [(at, value) for at, value in self.gap_statuses(result) if at > 10.13]
        self.assertTrue(values)
        self.assertTrue(all(value["state"] == "running" for _, value in values))
        held = [(at, row) for at, row in result.commits if row["live_calibration"]["grace"]]
        self.assertEqual(len(held), 1)
        held_at, row = held[0]
        self.assertAlmostEqual(held_at, 10.48)
        self.assertEqual(row["task_scale"], 0.)
        self.assertEqual(row["next_progress"], row["progress"])
        self.assertFalse(row["task_paused"])
        self.assertTrue(any(value["temporary_hold"] for _, value in values))
        later = [row for at, row in result.commits if at > held_at + 1e-9]
        self.assertTrue(later)
        self.assertGreater(later[0]["next_progress"], row["next_progress"])
        self.assertLess(values[-1][1]["commanded_gap_m"], values[0][1]["commanded_gap_m"])
        self.assertEqual(sum(row["kind"] == "gap_approach" for row in result.records), 1)

    def test_outlier_return_then_coherent_shift_blends_and_continues_without_another_click(self):
        # Reproduce the reported short rejection, recovery, then replacement
        # cluster. A fresh consensus gathered within grace does not brake the
        # follower, and must not latch a manual approach pause either.
        result = self.run_runtime(
            lambda tick, now: .0553 if tick == 12 or tick >= 20 else 0.,
            max_calibration_jump_m=.05,
            later_commands=lambda tick, now: [request("start", now)] if tick == 3 else [], stop_after=75)
        self.assert_active_session(result)
        self.assertEqual(sum(row["kind"] == "calibration_reacquired_live" for row in result.records), 1)
        self.assertFalse(any(row.get("calibration_braking") for _, row in result.commits))
        values = [(at, value) for at, value in self.gap_statuses(result) if at > 10.13]
        self.assertTrue(all(value["state"] == "running" for _, value in values))
        blending = [(at, row) for at, row in result.commits if row["live_calibration"]["blending"]]
        self.assertTrue(blending)
        for _, row in blending:
            self.assertEqual(row["task_scale"], 0.)
            self.assertEqual(row["next_progress"], row["progress"])
            self.assertFalse(row["task_paused"])
        self.assertTrue(any(value["temporary_hold"] for at, value in values if at > 10.8))
        blend_end = blending[-1][0]
        resumed = [row for at, row in result.commits if at > blend_end + 1e-9]
        self.assertTrue(resumed)
        self.assertGreater(resumed[-1]["next_progress"], blending[-1][1]["next_progress"])
        self.assertFalse(values[-1][1]["temporary_hold"])
        self.assertEqual(sum(row["kind"] == "gap_approach" for row in result.records), 1)

    def test_actual_observation_latches_pause_until_explicit_resume(self):
        result = self.run_runtime(
            lambda tick, now: {"reason": "occlusion"} if 12 <= tick < 24 else 0.,
            later_commands=lambda tick, now: [request({3: "start", 60: "resume"}[tick], now)]
            if tick in (3, 60) else [], stop_after=75)
        self.assert_active_session(result)
        self.assertTrue(any(row.get("calibration_braking") for _, row in result.commits))
        restored = [value for at, value in self.gap_statuses(result) if 11.85 < at < 12.39]
        resumed = [value for at, value in self.gap_statuses(result) if at > 12.45]
        self.assertTrue(restored and resumed)
        self.assertTrue(all(value["state"] == "paused" for value in restored))
        self.assertTrue(all(not value["temporary_hold"] for value in restored))
        for value in restored:
            self.assertAlmostEqual(value["commanded_gap_m"], restored[0]["commanded_gap_m"])
        self.assertTrue(all(value["state"] == "running" for value in resumed))
        self.assertLess(resumed[-1]["commanded_gap_m"], restored[-1]["commanded_gap_m"])
        self.assertEqual([row["action"] for row in result.records if row["kind"] == "gap_approach"],
                         ["start", "resume"])

    def test_operator_pause_survives_grace_and_coherent_calibration_blend(self):
        result = self.run_runtime(
            lambda tick, now: .0553 if tick == 12 or tick >= 20 else 0.,
            max_calibration_jump_m=.05,
            later_commands=lambda tick, now: [request({3: "start", 6: "pause"}[tick], now)]
            if tick in (3, 6) else [], stop_after=75)
        self.assert_active_session(result)
        self.assertEqual(sum(row["kind"] == "calibration_reacquired_live" for row in result.records), 1)
        self.assertFalse(any(row.get("calibration_braking") for _, row in result.commits))
        paused = [value for at, value in self.gap_statuses(result) if at > 10.25]
        self.assertTrue(paused)
        self.assertTrue(all(value["state"] == "paused" for value in paused))
        self.assertTrue(all(not value["temporary_hold"] for value in paused))
        for value in paused:
            self.assertAlmostEqual(value["commanded_gap_m"], paused[0]["commanded_gap_m"])
        self.assertEqual([row["action"] for row in result.records if row["kind"] == "gap_approach"],
                         ["start", "pause"])

    def test_resume_created_before_rejected_frame_cannot_resume_after_good_frame(self):
        pending = []

        def commands(tick, now):
            if tick == 3:
                return [request("start", now)]
            if tick == 6:
                return [request("pause", now)]
            if tick == 7:
                # The GUI creates this click before the rejected camera frame,
                # but IPC delivers it only after normal calibration returns.
                pending.append(request("resume", now))
            if tick == 14:
                return pending
            if tick == 22:
                return [request("resume", now)]
            return []

        result = self.run_runtime(
            lambda tick, now: {"reason": "temporary occlusion"} if tick == 12 else 0.,
            later_commands=commands, stop_after=32)
        self.assert_active_session(result)
        self.assertFalse(any(row.get("calibration_braking") for _, row in result.commits))
        paused = [value for at, value in self.gap_statuses(result) if 10.55 < at < 10.87]
        resumed = [value for at, value in self.gap_statuses(result) if at > 10.89]
        self.assertTrue(paused and resumed)
        self.assertTrue(all(value["state"] == "paused" for value in paused))
        self.assertTrue(all(value["state"] == "running" for value in resumed))
        rejected = [row for row in result.records if row["kind"] == "gap_approach_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("predates", rejected[0]["reason"])
        self.assertEqual([row["action"] for row in result.records if row["kind"] == "gap_approach"],
                         ["start", "pause", "resume"])


if __name__ == "__main__":
    unittest.main()
