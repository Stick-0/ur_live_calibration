"""ACTIVE vision recovery lifecycle with real control logic and virtual IO only.

These tests replace processes, clocks and RTDE transports. Camera frames still
pass through the production calibration/controller/runtime implementations.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import urcal.cooperative.runtime as runtime
from urcal.control.policy import ControlFault
from tests.test_cooperative_controller import accept, packet, scene
from tests.test_cooperative_runtime import Clock, Event, LocalQueue


class VirtualReader:
    def __init__(self, state, clock, frozen_after=None):
        self.template, self.clock = state, clock
        self.last_advance = clock.now
        self.frozen_after = frozen_after

    def read(self, now):
        if self.frozen_after is None or now < self.frozen_after:
            self.last_advance = now
        state = deepcopy(self.template)
        state.robot_timestamp = 50. + self.last_advance
        return state


class LiveCalibrationRuntimeTests(unittest.TestCase):
    def run_runtime(self, observations, *, stop_after=25, execute=True, arm=True,
                    frozen_after=None, delay_send=None, hardware_fault=None, later_commands=None, **overrides):
        settings = dict(base_motion_mode="pose", leader_response="bounded", initial_align=False,
                        execute=execute, fault_response="preview", stable_frames=3,
                        live_calibration_recovery=True, calibration_recovery_timeout=2.,
                        calibration_recovery_max_shift_m=.15, calibration_recovery_max_shift_deg=15.,
                        calibration_blend_time=.5,
                        filter_alpha=.2, max_calibration_age=.5, max_invalid_duration=.3,
                        max_robot_age=.5, max_calibration_jump_m=.015, max_calibration_jump_deg=2.,
                        max_target_m=.2, handoff_max_shift_m=.15, handoff_max_shift_deg=15.)
        settings.update(overrides)
        engine, bases, robot_states = scene(**settings)
        args = engine.args
        for index, at in enumerate((9.92, 9.96, 10.), 1):
            accept(engine, bases, at, index)
        clock = Clock()
        queues = [LocalQueue(maxsize=size) for size in (1, 8, 1, 1)]
        frames, commands, statuses, gui_statuses = queues
        if arm:
            commands.put_nowait("arm")
        sequence = 3
        records, seen_statuses, successful_sends, attempts, stops, sessions, commits = [], [], [], [], [], [], []
        readers = {name: VirtualReader(state, clock, frozen_after) for name, state in robot_states.items()}
        original_commit = engine.commit

        def commit(proposal):
            original_commit(proposal)
            commits.append((clock.now, deepcopy(proposal.record)))

        engine.commit = Mock(side_effect=commit)

        def later(tick):
            nonlocal sequence
            # Integrate the previously sent command for a single virtual period.
            for name, state in robot_states.items():
                state.joints += state.joint_velocity * .04
                state.tcp = engine.models[name].pose_and_jacobian(state.joints)[0]
            if later_commands is not None:
                for command in later_commands(tick, clock.now) or ():
                    commands.put_nowait(command)
            observation = observations(tick, clock.now) if callable(observations) else observations.get(tick)
            if observation is None:
                return
            sequence += 1
            observation = observation if isinstance(observation, dict) else {"shift": observation}
            observed_bases = deepcopy(bases)
            observed_bases[engine.leader][0, 3] += observation.get("shift", 0.)
            sample = packet(engine, observed_bases, clock.now - observation.get("age", 0.), sequence,
                            reason=observation.get("reason", ""))
            if observation.get("malformed"):
                sample.poses[engine.leader][0, 0] = np.nan
            runtime.put_latest(frames, sample)

        event = Event(clock, stop_after, later, .04)
        processes = [Mock(pid=701), Mock(pid=702)]
        for process in processes:
            process.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = queues
        context.Event.return_value = event
        context.Process.side_effect = processes
        receivers = [Mock(), Mock()]

        def construct(*unused, **kwargs):
            session = Mock()
            sessions.append(session)

            def send(targets, twists, check_fresh, safety_targets, **options):
                attempts.append((clock.now, event.waits))
                if delay_send is not None:
                    clock.now += delay_send(event.waits)
                if hardware_fault is not None:
                    fault = hardware_fault(event.waits)
                    if fault is not None:
                        raise fault
                check_fresh()
                successful_sends.append((clock.now, deepcopy(twists), deepcopy(targets)))
                for name, twist in twists.items():
                    robot_states[name].speed = np.asarray(twist).copy()
                    robot_states[name].joint_velocity = np.asarray(twist).copy()

            def stopped():
                stops.append(clock.now)
                for state in robot_states.values():
                    state.speed[:] = 0.
                    state.joint_velocity[:] = 0.

            session.send.side_effect = send
            session.stop_for_recovery.side_effect = stopped
            return session

        factory = Mock(side_effect=construct)
        logger = Mock()
        logger.log.side_effect = lambda row: records.append(deepcopy(row))
        real_put = runtime.put_latest

        def put(queue, value):
            if queue is gui_statuses:
                seen_statuses.append((clock.now, deepcopy(value)))
            return real_put(queue, value)

        caught = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(side_effect=receivers)),
                "rtde_control": SimpleNamespace(RTDEControlInterface=Mock(side_effect=AssertionError("hardware forbidden"))),
                "urcal.control.gui": SimpleNamespace(gui_worker=Mock()),
                "urcal.control.vision": SimpleNamespace(vision_worker=Mock()),
            }))
            stack.enter_context(patch.object(runtime.mp, "get_context", return_value=context))
            stack.enter_context(patch.object(runtime.time, "monotonic", side_effect=clock))
            stack.enter_context(patch.object(runtime, "CooperativeController", return_value=engine))
            stack.enter_context(patch.object(runtime, "StateReader", side_effect=[readers[name] for name in runtime.ROBOTS]))
            stack.enter_context(patch.object(runtime, "MotionSession", factory))
            stack.enter_context(patch.object(runtime, "ExperimentLogger", return_value=logger))
            stack.enter_context(patch.object(runtime, "cooperative_gui_scene", return_value=None))
            stack.enter_context(patch.object(runtime, "put_latest", side_effect=put))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                runtime.run(args)
            except BaseException as exc:
                caught = exc
        for receiver in receivers:
            receiver.disconnect.assert_called_once()
        logger.close.assert_called_once()
        for queue in queues:
            self.assertTrue(queue.cancelled and queue.closed)
        return SimpleNamespace(engine=engine, args=args, records=records, statuses=seen_statuses,
                               sends=successful_sends, attempts=attempts, commits=commits, stops=stops,
                               factory=factory, sessions=sessions, caught=caught, clock=clock,
                               bases=bases, states=robot_states, event=event)

    def test_healthy_fresh_pairs_leave_normal_following_and_one_session(self):
        result = self.run_runtime(lambda tick, now: 0., stop_after=12)
        self.assertIsNone(result.caught)
        self.assertEqual(result.stops, [])
        self.assertEqual(result.factory.call_count, 1)
        self.assertGreater(len(result.sends), 5)
        self.assertEqual(len(result.sends), len(result.commits))
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))

    def test_one_rejected_pair_recovers_without_stop_or_new_arm(self):
        observations = lambda tick, now: {"reason": "temporary occlusion"} if tick == 2 else 0.
        result = self.run_runtime(observations)
        self.assertIsNone(result.caught)
        self.assertEqual(result.stops, [])
        self.assertEqual(result.factory.call_count, 1)
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))
        self.assertFalse(any(row["kind"] == "calibration_reacquired_live" for row in result.records))
        self.assertFalse(any(row.get("calibration_braking") for _, row in result.commits))
        self.assertTrue(any(row["live_calibration"].get("grace") for _, row in result.commits))
        self.assertTrue(any(not row.get("calibration_braking") and at > 10.5 for at, row in result.commits))

    def test_consistent_shifted_cluster_replaces_old_baseline_with_gradual_coordinates(self):
        result = self.run_runtime(lambda tick, now: .04 if tick >= 2 else 0., stop_after=38)
        self.assertIsNone(result.caught)
        self.assertEqual(result.stops, [])
        self.assertEqual(result.factory.call_count, 1)
        recovered = [row for row in result.records if row["kind"] == "calibration_reacquired_live"]
        self.assertEqual(len(recovered), 1)
        raw_origin = result.bases[result.engine.leader][0, 3]
        values = [(at, row["bases"][result.engine.leader][0, 3] - raw_origin)
                  for at, row in result.commits if "bases" in row]
        moving = [(at, value) for at, value in values if value > 1e-6]
        self.assertTrue(moving)
        self.assertLess(moving[0][1], .04)
        self.assertGreater(moving[-1][1], .035)
        for (before_at, before), (at, value) in zip(values, values[1:]):
            self.assertLessEqual(abs(value - before), .10 * (at - before_at) + 1e-9)

    def test_lost_vision_stops_after_bounded_observation_and_later_frames_need_new_arm(self):
        def observations(tick, now):
            if tick == 2:
                return {"reason": "temporary occlusion"}
            return 0. if tick >= 54 else None
        result = self.run_runtime(observations, stop_after=80)
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stops), 1)
        self.assertGreaterEqual(result.stops[0], 12.08 - 1e-9)
        self.assertLessEqual(result.stops[0], 12.12 + 1e-9)
        self.assertEqual(result.factory.call_count, 1)
        self.assertFalse(any(at > result.stops[0] for at, _, _ in result.sends))
        self.assertTrue(any(status.get("recovery_required") for _, status in result.statuses))

    def test_fresh_pair_at_expired_deadline_cannot_refresh_observation(self):
        def observations(tick, now):
            if tick == 2:
                return {"reason": "temporary occlusion"}
            return 0. if tick >= 53 else None
        result = self.run_runtime(observations, stop_after=65)
        self.assertEqual(len(result.stops), 1)
        self.assertFalse(any(row["kind"] == "calibration_reacquired_live" for row in result.records))
        self.assertEqual(result.factory.call_count, 1)

    def test_send_freshness_check_rejects_observation_expiry_without_committing(self):
        observations = {2: {"reason": "temporary occlusion"}}
        result = self.run_runtime(observations, stop_after=58,
                                  delay_send=lambda tick: .08 if tick == 51 else 0.)
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stops), 1)
        self.assertTrue(any(tick == 51 for _, tick in result.attempts))
        self.assertEqual(len(result.attempts), len(result.sends) + 1)
        self.assertEqual(len(result.sends), len(result.commits))
        self.assertLess(result.sends[-1][0], result.stops[0])

    def test_calibration_expires_during_native_checks_discards_tracking_then_brakes(self):
        # The last capture is 10.0: preparation at 10.48 is valid, while
        # native checks finish at 10.52, beyond the 0.5 s measurement age.
        # Encoder state is still current. The unsent proposal must not commit.
        result = self.run_runtime(
            lambda tick, now: 0. if tick >= 13 else None,
            delay_send=lambda tick: .04 if tick == 12 else 0., stop_after=30)
        self.assertIsNone(result.caught)
        self.assertEqual(result.stops, [])
        self.assertEqual(result.factory.call_count, 1)
        discarded = [row for row in result.records if row["kind"] == "calibration_command_discarded"]
        self.assertEqual(len(discarded), 1)
        self.assertAlmostEqual(discarded[0]["at"], 10.52)
        self.assertEqual(len(result.attempts), len(result.sends) + 1)
        self.assertEqual(len(result.sends), len(result.commits))
        self.assertFalse(any(abs(at - 10.52) < 1e-9 for at, _ in result.commits))
        resumed = [(at, row) for at, row in result.commits if at > 10.52 + 1e-9]
        self.assertTrue(resumed)
        self.assertAlmostEqual(resumed[0][0], 10.56)
        self.assertTrue(resumed[0][1]["calibration_braking"])
        self.assertTrue(any(row["kind"] == "calibration_reacquired_live" for row in result.records))
        self.assertTrue(any(not row.get("calibration_braking") and at > 11. for at, row in resumed))

    def test_controller_program_failure_is_not_vision_recovery(self):
        result = self.run_runtime(
            lambda tick, now: {"reason": "temporary occlusion"} if tick >= 2 else 0.,
            hardware_fault=lambda tick: ControlFault("ur5e: control connection/program stopped") if tick == 3 else None)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("program stopped", str(result.caught))
        self.assertEqual(result.factory.call_count, 1)
        self.assertFalse(any(row["kind"] == "calibration_reacquired_live" for row in result.records))
        self.assertLess(max(at for at, _, _ in result.sends), 10.12 + 1e-9)

    def test_stale_encoder_stream_still_faults_during_vision_observation(self):
        result = self.run_runtime(
            lambda tick, now: {"reason": "temporary occlusion"} if tick >= 2 else 0.,
            frozen_after=10.2, stop_after=30)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("expired", str(result.caught))
        self.assertEqual(result.factory.call_count, 1)
        self.assertLess(result.clock.now, 12.)

    def test_disabling_live_recovery_preserves_previous_short_expiry(self):
        result = self.run_runtime({2: {"reason": "temporary occlusion"}},
                                  live_calibration_recovery=False, stop_after=20)
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stops), 1)
        self.assertLess(result.stops[0], 10.6)
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))

    def test_preview_does_not_activate_observation_window_or_control_session(self):
        result = self.run_runtime(lambda tick, now: .04, execute=False, arm=False, stop_after=12)
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 0)
        self.assertEqual(result.sends, [])
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))

    def test_initial_alignment_preview_uses_unchanged_acquisition_path(self):
        result = self.run_runtime(lambda tick, now: .04, execute=False, arm=False,
                                  initial_align=True, stop_after=12)
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 0)
        self.assertEqual(result.sends, [])
        self.assertFalse(any(row["kind"] == "calibration_observation" for row in result.records))


if __name__ == "__main__":
    unittest.main()
