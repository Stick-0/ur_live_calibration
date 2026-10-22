"""Recoverable live follow faults through the runtime, with no hardware or GUI.

The camera gate, estimator, command engine and recovery state machine are real.
Only process/clock/RTDE boundaries and the Cartesian fixture robot are replaced.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import urcal.cooperative.runtime as runtime
from urcal.control.geometry import inverse
from urcal.control.policy import CalibrationFrame, ControlFault, PlanningUnavailable
from urcal.cooperative.controller import ResearchFault
from tests.test_cooperative_controller import accept, packet, scene
from tests.test_cooperative_runtime import Clock, Event, LocalQueue


FIXTURE = Path(__file__).with_name("fixtures") / "follow_calibration_rejection.json"


class RecoveryReader:
    def __init__(self, state, clock, *, frozen_after_stop=False, moving_after_stop=False):
        self.template, self.clock = state, clock
        self.last_advance = clock.now
        self.stopped_at = None
        self.frozen_after_stop, self.moving_after_stop = frozen_after_stop, moving_after_stop

    def read(self, now):
        state = deepcopy(self.template)
        frozen = self.frozen_after_stop and self.stopped_at is not None
        if not frozen:
            self.last_advance = now
        state.robot_timestamp = 50. + (self.stopped_at if frozen else now)
        moving = self.moving_after_stop
        if isinstance(moving, tuple):
            moving = self.stopped_at is not None and moving[0] <= now - self.stopped_at < moving[1]
        if moving and self.stopped_at is not None:
            state.speed[0], state.joint_velocity[0] = .01, .02
        return state


class FollowFaultRecoveryTests(unittest.TestCase):
    def run_runtime(self, *, fault="send_expired", stop_after=24, arm_steps=(),
                    stop_error=None, frozen_after_stop=False, moving_after_stop=False,
                    recorded=False, fault_response="preview", automatic_frames=True,
                    pre_stop_capture=False, unknown_error=False, process_exit=False,
                    initial_align=False):
        engine, bases, robot_states = scene(
            stable_frames=5, max_invalid_duration=.3, max_robot_age=.15,
            filter_alpha=.25, velocity_alpha=.2, prediction_horizon=.5,
            path="line" if initial_align else "hold", execute=True,
            fault_response=fault_response,
        )
        args = engine.args
        args.initial_align = initial_align
        engine.progress, engine.task_paused = .37, True
        engine.initial_tcp = robot_states[engine.follower].tcp.copy()
        engine.tracking_offset = np.eye(4)
        clock, records, order, statuses_seen, sent = Clock(), [], [], [], []
        queues = [LocalQueue(maxsize=size) for size in (1, 8, 1, 1)]
        frames, commands, statuses, gui_statuses = queues
        commands.put_nowait("arm")
        sequence, pending_recorded = 5, []
        if recorded:
            data = json.loads(FIXTURE.read_text())["measurements"]
            frame_map = {name: bases[name] @ inverse(np.asarray(data[4]["poses_camera_model"][name]))
                         for name in runtime.ROBOTS}
            converted = []
            for raw in data:
                camera_poses = {name: frame_map[name] @ np.asarray(raw["poses_camera_model"][name])
                                @ inverse(engine.mappings[name]) for name in runtime.ROBOTS}
                converted.append((10. + raw["received_at"], CalibrationFrame(
                    raw["sequence"], 10. + raw["captured_at"], camera_poses,
                    raw["raw_mean_px"], raw["raw_max_px"], raw["valid_counts"])))
            for received, calibration in converted[:5]:
                self.assertTrue(engine.ingest(calibration, received))
            pending_recorded = converted[5:]
            sequence = data[-1]["sequence"]
        else:
            for index, at in enumerate((9.84, 9.88, 9.92, 9.96, 10.), 1):
                accept(engine, bases, at, index)

        readers = {name: RecoveryReader(state, clock, frozen_after_stop=frozen_after_stop,
                                        moving_after_stop=moving_after_stop)
                   for name, state in robot_states.items()}
        sessions, stop_times, successful_commits, alignments = [], [], [], []
        initial_path = engine.path
        initial_origin = engine.initial_tcp.copy()
        original_commit = engine.commit

        def commit(proposal):
            original_commit(proposal)
            successful_commits.append(clock.now)

        engine.commit = Mock(side_effect=commit)
        original_prepare = engine.prepare
        fault_injected = False

        def prepare(now, states, dt, active=False):
            nonlocal fault_injected
            if active and fault == "planner" and not fault_injected:
                fault_injected = True
                raise ResearchFault("tracking_error", "relative target exceeds allowed tracking error")
            return original_prepare(now, states, dt, active=active)

        engine.prepare = Mock(side_effect=prepare)

        def later(count):
            nonlocal sequence
            if count in arm_steps:
                commands.put_nowait("arm")
            if process_exit and count == 1:
                processes[0].is_alive.return_value = False
            if pending_recorded and clock.now >= pending_recorded[0][0]:
                _, item = pending_recorded.pop(0)
                frames.put_nowait(item)
            elif automatic_frames and (not recorded or stop_times):
                sequence += 1
                capture = (stop_times[0] - .001 if pre_stop_capture and stop_times else clock.now)
                item = packet(engine, bases, capture, sequence)
                frames.put_nowait(item)

        event = Event(clock, stop_after, later, .04)
        processes = [Mock(pid=501), Mock(pid=502)]
        for child in processes:
            child.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = queues
        context.Event.return_value = event
        context.Process.side_effect = processes
        receivers = [Mock(), Mock()]

        def construct(*unused, **kwargs):
            session = Mock()
            sessions.append(session)
            index = len(sessions)
            order.append(("connect", clock.now, index))

            def send(targets, twists, check_fresh, safety_targets, **options):
                nonlocal fault_injected
                if index == 1 and not fault_injected and not recorded:
                    if unknown_error:
                        fault_injected = True
                        raise RuntimeError("injected native controller transport failure")
                    if fault == "send_expired":
                        fault_injected = True
                        # A reset must erase histories from the interrupted
                        # session, not merely preserve already-zero arrays.
                        for name in ("previous_twist", "previous_qdot", "previous_compensation",
                                     "previous_compensation_qdot", "previous_feedback",
                                     "previous_base_compensation", "previous_base_compensation_qdot"):
                            getattr(engine, name)[:] = .02
                        clock.now += .51
                        # A click already in flight cannot authorize a new session.
                        commands.put_nowait("arm")
                check_fresh()
                sent.append((clock.now, index, deepcopy(twists)))
                order.append(("send", clock.now, index))

            def stopped():
                stop_times.append(clock.now)
                order.append(("controlled_stop", clock.now, index))
                for reader in readers.values():
                    reader.stopped_at = clock.now
                if stop_error:
                    raise stop_error

            session.send.side_effect = send

            def send_joints(qdots, next_joints, check_fresh, **options):
                check_fresh()
                sent.append((clock.now, index, deepcopy(qdots)))
                order.append(("send_joints", clock.now, index))

            session.send_joints.side_effect = send_joints
            session.stop_for_recovery.side_effect = stopped
            session.close.side_effect = lambda: order.append(("close", clock.now, index))
            return session

        factory = Mock(side_effect=construct)
        logger = Mock()
        logger.log.side_effect = lambda row: records.append(deepcopy(row))
        original_put = runtime.put_latest

        def preflight(session, supplied_args, offsets, live_bases, states, targets,
                      read_current, report, **options):
            read_current()
            index = len(alignments)
            alignment = SimpleNamespace(
                goal_joints={engine.follower: states[engine.follower].joints.copy()},
                predicted_joints={engine.follower: states[engine.follower].joints.copy()},
                joint_accelerations={engine.follower: .05},
                targets=deepcopy(targets), progress=0., done=False,
            )
            alignments.append(alignment)

            def propose(states, bases, dt):
                if index == 0:
                    raise PlanningUnavailable("alignment path left the allowed workspace")
                return {engine.follower: np.zeros(6)}

            alignment.propose = Mock(side_effect=propose)
            alignment.commit = Mock()
            return alignment

        alignment_plan = Mock(side_effect=preflight)
        alignment_check = Mock(wraps=runtime.check_alignment_request)

        def put(queue, value):
            if queue is gui_statuses:
                statuses_seen.append((clock.now, deepcopy(value)))
            return original_put(queue, value)

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
            if initial_align:
                stack.enter_context(patch.object(runtime, "prepare_alignment", alignment_plan))
                stack.enter_context(patch.object(runtime, "check_alignment_request", alignment_check))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                runtime.run(args)
            except BaseException as exc:
                caught = exc
        self.assertTrue(event.is_set())
        for receiver in receivers:
            receiver.disconnect.assert_called_once()
        for queue in queues:
            self.assertTrue(queue.cancelled and queue.closed)
        logger.close.assert_called_once()
        return SimpleNamespace(engine=engine, args=args, caught=caught, records=records, sent=sent,
                               statuses=statuses_seen, sessions=sessions, factory=factory,
                               stop_times=stop_times, order=order, initial_path=initial_path,
                               initial_origin=initial_origin, commits=successful_commits,
                               processes=processes, event=event, alignments=alignments,
                               alignment_plan=alignment_plan, alignment_check=alignment_check,
                               robot_states=robot_states)

    def test_expired_inside_native_preflight_stops_without_sending_or_committing_then_stays_open(self):
        result = self.run_runtime()
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stop_times), 1)
        self.assertEqual(result.sent, [])
        self.assertEqual(result.commits, [])
        self.assertEqual(result.factory.call_count, 1)
        self.assertTrue(any(status.get("recovery_required") for _, status in result.statuses))
        self.assertTrue(any(status.get("ready") for _, status in result.statuses if not status.get("active")))
        self.assertIs(result.engine.path, result.initial_path)
        self.assertEqual(result.engine.progress, .37)
        self.assertTrue(result.engine.task_paused)
        np.testing.assert_array_equal(result.engine.initial_tcp, result.initial_origin)
        stationary = next(row for row in result.records if row["kind"] == "recovery_stationary")
        self.assertGreaterEqual(stationary["at"] - result.stop_times[0], .2)
        first_ready = next(at for at, status in result.statuses
                           if at > result.stop_times[0] and status.get("ready"))
        fresh = [row for row in result.records if row["kind"] == "measurement" and row["accepted"]
                 and stationary["at"] < row["captured_at"] <= first_ready]
        self.assertEqual(len(fresh), 5)
        self.assertIsNone(result.engine.tracking_offset)
        for name in ("previous_twist", "previous_qdot", "previous_compensation", "previous_compensation_qdot", "previous_feedback",
                     "previous_base_compensation", "previous_base_compensation_qdot"):
            np.testing.assert_array_equal(getattr(result.engine, name), np.zeros(6))

    def test_queued_and_early_arm_cannot_resume_but_later_explicit_arm_can(self):
        # Step 12 is exactly the first READY tick, so that click is stale too.
        result = self.run_runtime(arm_steps=(2, 3, 5, 7, 12, 18), stop_after=23)
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 2)
        connects = [row for row in result.order if row[0] == "connect"]
        self.assertGreater(connects[1][1] - result.stop_times[0], .4)
        self.assertTrue(result.sent)
        self.assertTrue(all(index == 2 for _, index, _ in result.sent))
        self.assertGreater(result.sent[0][0], connects[1][1])
        self.assertEqual(len(result.commits), len(result.sent))
        ready_at = next(at for at, status in result.statuses
                        if at > result.stop_times[0] and status.get("ready"))
        self.assertLess(ready_at, connects[1][1])

    def test_old_in_flight_images_cannot_count_as_post_stop_calibration(self):
        result = self.run_runtime(pre_stop_capture=True)
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 1)
        self.assertFalse(any(status.get("ready") for at, status in result.statuses
                             if at > result.stop_times[0]))
        self.assertEqual(result.sent, [])

    def test_recognized_planner_fault_returns_to_preview_without_changing_task(self):
        result = self.run_runtime(fault="planner")
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stop_times), 1)
        self.assertEqual(result.factory.call_count, 1)
        self.assertEqual(result.sent, [])
        self.assertEqual(result.engine.progress, .37)
        self.assertIs(result.engine.path, result.initial_path)

    def test_failed_controlled_stop_is_terminal_and_never_resets_into_ready(self):
        failure = ControlFault("ur5e: speedStop failed; stop not confirmed")
        result = self.run_runtime(stop_error=failure)
        self.assertIs(result.caught, failure)
        self.assertEqual(len(result.stop_times), 1)
        self.assertFalse(any(status.get("ready") for at, status in result.statuses
                             if at > result.stop_times[0]))
        self.assertEqual(result.sent, [])

    def test_frozen_encoder_timestamps_cannot_confirm_stationary_stop(self):
        result = self.run_runtime(frozen_after_stop=True)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertEqual(result.factory.call_count, 1)
        self.assertEqual(result.sent, [])
        self.assertFalse(any(status.get("ready") for at, status in result.statuses
                             if at > result.stop_times[0]))

    def test_nonstationary_robot_cannot_reset_calibration_or_reenable_arm(self):
        result = self.run_runtime(moving_after_stop=True, arm_steps=(8, 14, 18))
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 1)
        self.assertEqual(result.sent, [])
        self.assertFalse(any(row["kind"] == "recovery_stationary" for row in result.records))
        self.assertFalse(any(status.get("ready") for at, status in result.statuses
                             if at > result.stop_times[0]))

    def test_motion_after_recovery_ready_invalidates_the_old_calibration_and_arm(self):
        result = self.run_runtime(moving_after_stop=(.50, .64), arm_steps=(14, 33), stop_after=37)
        self.assertIsNone(result.caught)
        self.assertEqual(result.factory.call_count, 2)
        stationary = [row for row in result.records if row["kind"] == "recovery_stationary"]
        self.assertEqual(len(stationary), 2)
        self.assertGreater(stationary[1]["at"] - result.stop_times[0], .8)
        connections = [row for row in result.order if row[0] == "connect"]
        self.assertGreater(connections[1][1], stationary[1]["at"] + .19)
        self.assertTrue(all(index == 2 for _, index, _ in result.sent))

    def test_unknown_transport_and_process_failure_remain_terminal(self):
        for options in ({"unknown_error": True}, {"process_exit": True}):
            with self.subTest(options=options):
                result = self.run_runtime(**options)
                self.assertIsNotNone(result.caught)
                self.assertEqual(result.stop_times, [])
                self.assertEqual(result.factory.call_count, 1)

    def test_explicit_exit_response_preserves_terminal_research_behavior(self):
        result = self.run_runtime(fault_response="exit")
        self.assertIsInstance(result.caught, ControlFault)
        self.assertEqual(result.stop_times, [])
        self.assertEqual(result.sent, [])

    def test_recorded_four_rejections_stop_then_accept_a_new_stationary_cluster_without_autoresume(self):
        result = self.run_runtime(recorded=True, fault="camera", arm_steps=(1,), stop_after=34)
        self.assertIsNone(result.caught)
        rejected = [row for row in result.records if row["kind"] == "measurement" and not row["accepted"]]
        self.assertEqual(len(rejected), 4)
        self.assertTrue(all("calibration jump" in row["rejection"] for row in rejected))
        self.assertEqual(len(result.stop_times), 1)
        self.assertEqual(result.factory.call_count, 1)
        self.assertTrue(any(status.get("ready") for at, status in result.statuses
                            if at > result.stop_times[0]))
        self.assertTrue(all(at < result.stop_times[0] for at, _, _ in result.sent))

    def test_initial_alignment_fault_replans_preserved_path_progress_after_new_arm(self):
        result = self.run_runtime(initial_align=True, fault="alignment", arm_steps=(3, 7, 18), stop_after=22)
        self.assertIsNone(result.caught)
        self.assertEqual(len(result.stop_times), 1)
        self.assertEqual(result.factory.call_count, 2)
        self.assertEqual(result.alignment_plan.call_count, 2)
        self.assertIsNot(result.alignments[0], result.alignments[1])
        self.assertEqual(result.alignments[0].propose.call_count, 1)
        result.alignments[0].commit.assert_not_called()
        self.assertGreater(result.alignments[1].propose.call_count, 0)
        self.assertTrue(result.sent)
        self.assertTrue(all(index == 2 for _, index, _ in result.sent))
        self.assertEqual(result.engine.progress, .37)
        self.assertIs(result.engine.path, result.initial_path)
        np.testing.assert_array_equal(result.engine.initial_tcp, result.initial_origin)
        target = result.robot_states[result.engine.leader].tcp @ result.initial_path.pose(.37)
        target_at_start = result.robot_states[result.engine.leader].tcp @ result.initial_path.pose(0.)
        self.assertGreater(np.linalg.norm(target[:3, 3] - target_at_start[:3, 3]), .01)
        for call in result.alignment_plan.call_args_list:
            np.testing.assert_allclose(call.args[5][result.engine.follower], target, atol=1e-12)
        for call in result.alignment_check.call_args_list:
            np.testing.assert_array_equal(call.args[7][result.engine.follower], result.initial_origin)


if __name__ == "__main__":
    unittest.main()
