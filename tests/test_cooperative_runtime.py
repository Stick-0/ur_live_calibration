"""Research runtime ordering and failure handling with no robot/process access."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from queue import Queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))

import urcal.cooperative.runtime as runtime
import urcal.control.runtime as control_dual
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.task import RelativePath
from urcal.control.geometry import facing_follower
from urcal.control.policy import CalibrationFrame, ControlFault


class LocalQueue(Queue):
    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now


class Event:
    def __init__(self, clock, stop_after, callback, step):
        self.clock, self.stop_after, self.callback, self.step = clock, stop_after, callback, step
        self.stopped, self.waits = False, 0

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, timeout):
        self.clock.now += self.step
        self.waits += 1
        self.callback(self.waits)
        if self.waits >= self.stop_after:
            self.set()
        return self.stopped


class Reader:
    def __init__(self, frozen=False):
        self.last_advance = 10.
        self.frozen = frozen

    def read(self, now):
        if not self.frozen:
            self.last_advance = now
        return SimpleNamespace(tcp=np.eye(4), joints=np.zeros(6), speed=np.zeros(6),
                               joint_velocity=np.zeros(6), robot_timestamp=60. if self.frozen else 50. + now)


def config(execute=False):
    arguments = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10",
                 "--ur5e_tcp_offset", *(["0"] * 6), "--ur10e_tcp_offset", *(["0"] * 6)]
    if execute:
        arguments.append("--execute")
    # These lifecycle fixtures intentionally exercise the original exact task
    # contract; approximate task behavior has separate integration coverage.
    return control_dual.parse_args(arguments + ["--max_robot_age", ".15",
                                               "--target_tolerance_m", "0", "--target_tolerance_deg", "0",
                                               "--handoff_max_shift_m", "0", "--handoff_max_shift_deg", "0",
                                               "--handoff_timeout", "5", "--max_target_m", ".1",
                                               "--calibration_acquisition", "consecutive", "--handoff_wait", "stop",
                                               "--max_calibration_jump_m", ".015", "--max_calibration_jump_deg", "2",
                                               "--fault_response", "exit"])


class ResearchRuntimeTests(unittest.TestCase):
    def mocked_run(self, *, execute=False, initial_commands=("arm",), later_commands=None,
                   send_hook=None, state_error=None, prepare_error=None, expected=None,
                   stop_after=2, step=.04, packet=None, logger_error_kind=None, frozen_joints=False,
                   later_packets=None, reacquire_on_packet=False, ingest_accepted=True,
                   cleanup_delay=0., initial_recording_commands=(), later_recording_commands=None,
                   recorder_status=None):
        args, clock, order, records = config(execute), Clock(), [], []
        queues = [LocalQueue(maxsize=n) for n in (1, 8, 1, 1, 8)]
        frames, commands, statuses, gui_statuses, recording_commands = queues
        for command in initial_commands:
            commands.put_nowait(command)
        for command in initial_recording_commands:
            recording_commands.put_nowait(command)
        if packet is not None:
            frames.put_nowait(packet)

        def later(wait_count):
            if wait_count in (later_packets or {}):
                frames.put_nowait(later_packets[wait_count])
            for command in (later_commands or {}).get(wait_count, ()):
                commands.put_nowait(command)
            for command in (later_recording_commands or {}).get(wait_count, ()):
                recording_commands.put_nowait(command)

        event = Event(clock, stop_after, later, step)
        processes = [Mock(pid=101), Mock(pid=102)]
        for child in processes:
            child.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = queues
        context.Event.return_value = event
        context.Process.side_effect = processes

        engine = Mock(leader="ur10e", follower="ur5e", commanded=("ur5e",),
                      offsets={name: np.eye(4) for name in runtime.ROBOTS},
                      path=RelativePath(facing_follower(np.eye(4), .3), kind="line"),
                      gap=.3, progress=0., task_paused=False, invalid_since=None, last_rejection="",
                      preview_recovery=Mock(count=0), reacquisition_count=0,
                      live_calibration=None)
        engine.gap_approach.state = "idle"
        engine.gap_approach_status.side_effect = lambda **_kwargs: {"state": "idle", "enabled": False}
        engine.gate.last_rejection = ""
        engine.gate.last_capture = 9.9
        preparations = []

        def prepare(now, states, dt, active=False):
            order.append("prepare_active" if active else "prepare_preview")
            preparations.append((active, engine.task_paused))
            if prepare_error is not None:
                raise prepare_error
            next_progress = engine.progress + (0. if engine.task_paused else .01)
            return SimpleNamespace(
                goals={"ur5e": np.eye(4)}, tcp_goals={"ur5e": np.eye(4)},
                future_tcp_goals={"ur5e": np.eye(4)},
                twists={"ur5e": np.full(6, .001)}, next_progress=next_progress,
                decision=SimpleNamespace(task_scale=0. if engine.task_paused else 1.),
                record={"kind": "control", "at": now,
                        "bases": {name: np.eye(4) for name in runtime.ROBOTS},
                        "measurement_age": {name: now - 9.9 for name in runtime.ROBOTS}},
            )

        def commit(proposal):
            order.append("commit")
            engine.progress = proposal.next_progress

        engine.prepare.side_effect, engine.commit.side_effect = prepare, commit
        engine.ingest.return_value = ingest_accepted
        if not ingest_accepted and packet is not None:
            engine.last_rejection = packet.reason
        if reacquire_on_packet:
            def ingest(frame, at, *, allow_reacquire=False, active_recovery=False):
                self.assertTrue(allow_reacquire)
                self.assertFalse(active_recovery)
                engine.reacquisition_count += 1
                return True
            engine.ingest.side_effect = ingest
        engine.state.side_effect = state_error
        engine.state.return_value = SimpleNamespace(
            base_poses={name: np.eye(4) for name in runtime.ROBOTS},
            measurement_age={name: .1 for name in runtime.ROBOTS})

        session = Mock()
        sent = []

        def send(targets, twists, check_fresh, safety_targets):
            order.append("send_preflight")
            self.assertEqual(set(safety_targets), {"ur5e"})
            if send_hook is not None:
                send_hook(clock, event)
            check_fresh()
            order.append("speed_sent")
            sent.append(deepcopy(twists))

        session.send.side_effect = send
        def close_motion():
            order.append("motion_close")
            clock.now += cleanup_delay

        session.close.side_effect = close_motion
        motion_factory = Mock(return_value=session)
        receivers = [Mock(), Mock()]
        for index, receiver in enumerate(receivers):
            receiver.isConnected.return_value = True
            receiver.getRobotMode.return_value = 7
            receiver.getSafetyMode.return_value = 1
            receiver.getRuntimeState.return_value = 2
            receiver.getTimestamp.return_value = 60.

            def disconnect(i=index, received=receiver):
                order.append("receiver_close_" + str(i))
                received.isConnected.return_value = False
                received.getRuntimeState.return_value = 0

            receiver.disconnect.side_effect = disconnect
        receive_factory = Mock(side_effect=receivers)
        fake_readers = [Reader(frozen_joints), Reader(frozen_joints)]
        for reader, receiver in zip(fake_readers, receivers):
            reader.receiver = receiver
        reader_factory = Mock(side_effect=fake_readers)
        logger = Mock()

        def log(record):
            order.append("log_" + record["kind"])
            if record["kind"] == logger_error_kind:
                raise OSError("injected log write failure")
            records.append(deepcopy(record))

        logger.log.side_effect = log
        recorder, error_samples = Mock(), []
        recorder.status.return_value = (recorder_status if recorder_status is not None else
                                       {"state": "idle", "path": None, "rows": 0,
                                        "valid_samples": 0, "elapsed_s": 0., "error": ""})
        recorder.record.side_effect = lambda sample, **kwargs: error_samples.append((deepcopy(sample), kwargs))
        recorder.close.side_effect = lambda *args, **kwargs: order.append("recording_close")
        caught = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=receive_factory),
                "rtde_control": SimpleNamespace(RTDEControlInterface=Mock(side_effect=AssertionError("hardware forbidden"))),
                "urcal.control.vision": SimpleNamespace(vision_worker=Mock()),
                "urcal.control.gui": SimpleNamespace(gui_worker=Mock()),
            }))
            stack.enter_context(patch.object(runtime.mp, "get_context", return_value=context))
            stack.enter_context(patch.object(runtime.time, "monotonic", side_effect=clock))
            stack.enter_context(patch.object(runtime, "CooperativeController", return_value=engine))
            stack.enter_context(patch.object(runtime, "StateReader", reader_factory))
            stack.enter_context(patch.object(runtime, "MotionSession", motion_factory))
            stack.enter_context(patch.object(runtime, "ExperimentLogger", return_value=logger))
            stack.enter_context(patch.object(runtime, "PositionErrorRecorder", return_value=recorder))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if expected is None:
                runtime.run(args)
            else:
                with self.assertRaises(expected) as caught_context:
                    runtime.run(args)
                caught = caught_context.exception

        self.assertTrue(event.is_set())
        logger.close.assert_called_once()
        recorder.close.assert_called_once()
        for receiver in receivers:
            receiver.disconnect.assert_called_once()
        for queue in queues:
            self.assertTrue(queue.cancelled)
            self.assertTrue(queue.closed)
        for call in reader_factory.call_args_list:
            self.assertTrue(call.kwargs["require_joint_velocity"])
        return SimpleNamespace(engine=engine, session=session, motion_factory=motion_factory,
                               records=records, order=order, preparations=preparations,
                               event=event, sent=sent, statuses=statuses, gui_statuses=gui_statuses,
                               caught=caught, clock=clock, processes=processes,
                               recorder=recorder, error_samples=error_samples,
                               recording_commands=recording_commands)

    def test_preview_arm_never_constructs_controller_or_commits_task_progress(self):
        result = self.mocked_run(stop_after=1)
        result.motion_factory.assert_not_called()
        result.engine.commit.assert_not_called()
        self.assertEqual(result.engine.progress, 0.)
        self.assertIn("PREVIEW ONLY", runtime.take_latest(result.statuses)["message"])

    def test_execute_without_arm_remains_receive_only(self):
        result = self.mocked_run(execute=True, initial_commands=())
        result.motion_factory.assert_not_called()
        result.engine.commit.assert_not_called()
        self.assertTrue(all(not active for active, paused in result.preparations))

    def test_successful_command_commits_only_after_send_returns(self):
        result = self.mocked_run(execute=True)
        self.assertEqual(result.preparations, [(False, False), (True, False)])
        result.motion_factory.assert_called_once()
        self.assertEqual(result.motion_factory.call_args.args[1], ("ur5e",))
        result.session.send.assert_called_once()
        result.engine.commit.assert_called_once()
        self.assertLess(result.order.index("log_command_proposal"), result.order.index("speed_sent"))
        self.assertLess(result.order.index("speed_sent"), result.order.index("commit"))
        self.assertAlmostEqual(result.engine.progress, .01)
        self.assertTrue([r for r in result.records if r["kind"] == "control"][-1]["command_sent"])

    def test_failed_send_never_commits_and_stops_before_receiver_disconnect(self):
        def fail(clock, event):
            raise ControlFault("injected speedL send failure")

        result = self.mocked_run(execute=True, send_hook=fail, expected=ControlFault)
        result.engine.commit.assert_not_called()
        self.assertEqual(result.engine.progress, 0.)
        self.assertEqual(result.sent, [])
        self.assertLess(result.order.index("motion_close"), result.order.index("receiver_close_1"))
        fault = [r for r in result.records if r["kind"] == "fault"][0]
        self.assertEqual(fault["category"], "control_fault")
        self.assertTrue(fault["external_base_stop_required"])

    def test_slow_controller_preflight_rechecks_robot_age_before_speed_send(self):
        def delay(clock, event):
            clock.now += .2

        result = self.mocked_run(execute=True, send_hook=delay, expected=ResearchFault)
        self.assertEqual(result.caught.category, "robot_state_expired")
        result.engine.state.assert_called_once_with(result.clock.now, active=True)
        result.engine.commit.assert_not_called()
        self.assertEqual(result.sent, [])

    def test_fault_retains_pre_cleanup_time_and_receive_status(self):
        def fail(clock, event):
            raise ControlFault("ur5e: control program stopped")

        result = self.mocked_run(execute=True, send_hook=fail, expected=ControlFault,
                                 cleanup_delay=1.7)
        fault = next(record for record in result.records if record["kind"] == "fault")
        self.assertAlmostEqual(fault["at"] - fault["details"]["detected_at"], 1.7)
        for status in fault["details"]["receive_status"].values():
            self.assertTrue(status["connected"])
            self.assertEqual(status["runtime_state"], 2)
            self.assertEqual(status["safety_mode"], 1)
            self.assertEqual(status["at"], fault["details"]["detected_at"])
        result.engine.commit.assert_not_called()

    def test_calibration_expiring_during_preflight_keeps_its_fault_category(self):
        fault = ResearchFault("measurement_expired", "prediction expired during preflight")
        result = self.mocked_run(execute=True, state_error=fault, expected=ResearchFault)
        self.assertEqual(result.caught.category, "measurement_expired")
        self.assertEqual(result.records[-1]["reason"], "measurement_expired")
        self.assertEqual(result.sent, [])
        result.engine.commit.assert_not_called()

    def test_stop_during_preflight_blocks_send_and_commit(self):
        result = self.mocked_run(execute=True, send_hook=lambda clock, event: event.set(), expected=ResearchFault)
        self.assertEqual(result.caught.category, "operator_stop")
        result.engine.state.assert_not_called()
        self.assertEqual(result.sent, [])
        result.engine.commit.assert_not_called()

    def test_pause_resume_commands_preserve_active_compensation_session(self):
        stop_during_send = []
        result = self.mocked_run(execute=True, later_commands={1: ["pause_task"], 2: ["resume_task"]},
                                 stop_after=3, send_hook=lambda clock, event: stop_during_send.append(event.is_set()))
        self.assertEqual(result.preparations, [(False, False), (True, True), (True, False)])
        self.assertEqual(stop_during_send, [False, False])
        self.assertEqual(len(result.sent), 2)
        self.assertEqual(result.engine.commit.call_count, 2)
        self.assertAlmostEqual(result.engine.progress, .01)
        self.assertEqual([r["paused"] for r in result.records if r["kind"] == "task_pause"], [True, False])

    def test_apply_cancels_arm_in_same_batch_and_is_forbidden_while_active(self):
        target = {"type": "target", "gap": .35, "path": "line"}
        for initial in ([target, "arm"], ["arm", target]):
            with self.subTest(initial=initial):
                result = self.mocked_run(execute=True, initial_commands=initial)
                result.engine.apply_target.assert_called_once_with(target)
                result.motion_factory.assert_not_called()
        result = self.mocked_run(execute=True, later_commands={1: [target]}, expected=ResearchFault)
        self.assertEqual(result.caught.category, "target_change")
        result.engine.apply_target.assert_not_called()
        result.engine.commit.assert_not_called()

    def test_measurement_logging_preserves_raw_capture_and_host_receipt_times(self):
        packet = CalibrationFrame(9, 9.8, {name: np.eye(4) for name in runtime.ROBOTS},
                                  {name: .2 for name in runtime.ROBOTS},
                                  {name: .4 for name in runtime.ROBOTS},
                                  {name: 7 for name in runtime.ROBOTS})
        result = self.mocked_run(packet=packet, initial_commands=(), stop_after=1)
        result.engine.ingest.assert_called_once_with(packet, 10., allow_reacquire=True, active_recovery=False)
        record = [r for r in result.records if r["kind"] == "measurement"][0]
        self.assertEqual(record["captured_at"], 9.8)
        self.assertEqual(record["received_at"], 10.)
        self.assertEqual(record["sequence"], 9)
        self.assertEqual(record["poses_camera_model"]["ur5e"], np.eye(4).tolist())
        self.assertIsNone(record["pose_diagnostics"])

    def test_measurement_logs_pose_inputs_for_success_and_failure_without_aliases(self):
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                points = np.arange(14., dtype=float).reshape(7, 2)
                points[2] = np.nan
                diagnostics = {"schema": "urcal.pnp-initialization.v1", "K": np.eye(3),
                               "robots": {"ur5e": {"detected_px": points,
                                                    "empnp_ok": np.bool_(accepted)}}}
                reason = "" if accepted else "ur5e: EMPnP failed"
                packet = CalibrationFrame(9, 9.8, {}, {}, {}, {"ur5e": 7, "ur10e": 7},
                                          reason, diagnostics)
                result = self.mocked_run(packet=packet, initial_commands=(), stop_after=1,
                                         ingest_accepted=accepted)
                record = next(r for r in result.records if r["kind"] == "measurement")
                self.assertEqual(record["accepted"], accepted)
                self.assertEqual(record["rejection"], reason)
                self.assertEqual(record["captured_at"], 9.8)
                self.assertEqual(record["received_at"], 10.)
                saved = record["pose_diagnostics"]["robots"]["ur5e"]
                self.assertEqual(saved["empnp_ok"], accepted)
                self.assertEqual(saved["detected_px"][2], [None, None])
                points[:] = -1
                diagnostics["K"][:] = 0
                self.assertEqual(saved["detected_px"][0], [0., 1.])
                self.assertEqual(record["pose_diagnostics"]["K"], np.eye(3).tolist())

    def test_reacquisition_is_enabled_in_preview_but_disabled_after_explicit_arm(self):
        def frame(sequence, at):
            return CalibrationFrame(sequence, at, {name: np.eye(4) for name in runtime.ROBOTS},
                                    {name: 1. for name in runtime.ROBOTS},
                                    {name: 2. for name in runtime.ROBOTS},
                                    {name: 7 for name in runtime.ROBOTS})

        first, active = frame(1, 9.99), frame(2, 10.04)
        result = self.mocked_run(execute=True, packet=first, later_packets={1: active}, stop_after=2)
        calls = result.engine.ingest.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, (first, 10.))
        self.assertTrue(calls[0].kwargs["allow_reacquire"])
        self.assertFalse(calls[1].kwargs["allow_reacquire"])
        result.motion_factory.assert_called_once()

    def test_new_preview_baseline_cancels_queued_arm_until_another_explicit_arm(self):
        frame = CalibrationFrame(1, 10., {name: np.eye(4) for name in runtime.ROBOTS},
                                 {name: 1. for name in runtime.ROBOTS},
                                 {name: 2. for name in runtime.ROBOTS},
                                 {name: 7 for name in runtime.ROBOTS})
        for arm_again in (False, True):
            with self.subTest(arm_again=arm_again):
                result = self.mocked_run(execute=True, packet=frame, reacquire_on_packet=True,
                                         later_commands={1: ["arm"]} if arm_again else {}, stop_after=3)
                self.assertEqual(result.engine.reacquisition_count, 1)
                self.assertEqual(result.motion_factory.call_count, int(arm_again))
                self.assertEqual(len(result.sent), int(arm_again))
    def test_log_failure_before_send_blocks_unlogged_command(self):
        result = self.mocked_run(execute=True, logger_error_kind="command_proposal", expected=OSError)
        result.session.send.assert_not_called()
        result.engine.commit.assert_not_called()
        self.assertIn("motion_close", result.order)

    def test_loop_deadline_failure_does_not_prepare_or_commit_again(self):
        result = self.mocked_run(execute=True, step=.2, expected=ResearchFault)
        self.assertEqual(result.caught.category, "control_deadline")
        self.assertEqual(result.preparations, [(False, False)])
        result.engine.commit.assert_not_called()
        result.session.send.assert_not_called()

    def test_preview_quality_failure_is_wait_status_with_category(self):
        result = self.mocked_run(prepare_error=ResearchFault("measurement_unavailable", "not enough valid frames"),
                                 stop_after=1)
        status = runtime.take_latest(result.gui_statuses)
        self.assertFalse(status["ready"])
        self.assertFalse(status["active"])
        self.assertEqual(status["fault_category"], "measurement_unavailable")
        result.motion_factory.assert_not_called()

    def test_gui_scene_survives_rejected_ik_without_relaxing_readiness(self):
        result = self.mocked_run(prepare_error=ResearchFault("reachability_infeasible", "no local IK"),
                                 stop_after=1)
        vision = runtime.take_latest(result.statuses)
        gui = runtime.take_latest(result.gui_statuses)
        self.assertNotIn("scene", vision)
        self.assertFalse(gui["ready"])
        self.assertEqual(gui["fault_category"], "reachability_infeasible")
        scene = gui["scene"]
        self.assertEqual(scene["at"], 10.)
        self.assertEqual(set(scene["robots"]), set(runtime.ROBOTS))
        np.testing.assert_allclose(scene["goals"]["ur5e"], facing_follower(np.eye(4), .3))
        self.assertEqual(scene["measurement_age"], {name: .1 for name in runtime.ROBOTS})
        self.assertEqual(scene["task"]["leader"], "ur10e")
        self.assertEqual(scene["task"]["kind"], "line")
        self.assertEqual(scene["task"]["progress"], 0.)
        result.motion_factory.assert_not_called()
        result.engine.commit.assert_not_called()

    def test_gui_scene_is_omitted_if_calibration_state_is_unavailable(self):
        result = self.mocked_run(prepare_error=ResearchFault("measurement_expired", "stale"),
                                 state_error=ResearchFault("measurement_expired", "stale"), stop_after=1)
        self.assertIsNone(runtime.take_latest(result.gui_statuses)["scene"])
        result.motion_factory.assert_not_called()

    def test_initial_accepted_pair_can_be_shown_without_bypassing_stability_or_age(self):
        from test_cooperative_controller import accept, packet, scene

        engine, bases, states = scene(stable_frames=5)
        accept(engine, bases, at=10.)
        readers = {name: Reader() for name in runtime.ROBOTS}
        displayed = runtime.cooperative_gui_scene(10., engine, states, readers, {}, .15)
        self.assertIsNotNone(displayed)
        self.assertFalse(displayed["calibration_valid"])
        self.assertIn("not stable", displayed["reason"])
        self.assertEqual(displayed["measurement_age"], {name: 0. for name in runtime.ROBOTS})
        self.assertFalse(engine.gate.ready)
        self.assertEqual(engine.gate.stable_count, 1)
        self.assertEqual(engine.progress, 0.)
        for reader in readers.values():
            reader.last_advance = 10.51
        self.assertIsNone(runtime.cooperative_gui_scene(10.51, engine, states, readers, {}, .15))

        engine, bases, states = scene(stable_frames=5)
        accept(engine, bases, at=10.)
        self.assertFalse(engine.ingest(packet(engine, bases, at=10.01, sequence=2, reason="invalid image"), 10.02))
        for reader in readers.values():
            reader.last_advance = 10.03
        self.assertIsNone(runtime.cooperative_gui_scene(10.03, engine, states, readers, {}, .15))

    def test_gui_scene_does_not_refresh_frozen_encoder_timestamps(self):
        result = self.mocked_run(initial_commands=(), frozen_joints=True, stop_after=2)
        self.assertIsNone(runtime.take_latest(result.gui_statuses)["scene"])
        self.assertNotIn("scene", runtime.take_latest(result.statuses))

    def test_successful_scene_preserves_measurement_age_and_copies_task_delta(self):
        result = self.mocked_run(initial_commands=(), stop_after=1)
        scene = runtime.take_latest(result.gui_statuses)["scene"]
        self.assertAlmostEqual(scene["measurement_age"]["ur5e"], .1)
        scene["task"]["delta"][:] = 999.
        np.testing.assert_array_equal(result.engine.path.delta, np.zeros(6))

    def test_finite_record_preserves_rejected_measurements_without_json_nan(self):
        result = runtime.finite_record({"raw": np.array([np.nan, np.inf, -np.inf, .2]),
                                        "sequence": np.int64(7), "flag": np.bool_(False)})
        self.assertEqual(result, {"raw": [None, None, None, .2], "sequence": 7, "flag": False})

    def test_recording_receives_each_produced_sample_before_latest_gui_queue_overwrite(self):
        result = self.mocked_run(initial_commands=(), stop_after=8)
        self.assertEqual(len(result.error_samples), 8)
        values = [sample for sample, _ in result.error_samples]
        self.assertTrue(all(sample["valid"] for sample in values))
        self.assertEqual(len({sample["at"] for sample in values}), 8)
        # This fixture's target is 300mm away. The graph range must not clamp
        # the producer data, and its one-slot queue retains only the last tick.
        self.assertTrue(all(np.linalg.norm(sample["error_m"]) > .05 for sample in values))
        latest = runtime.take_latest(result.gui_statuses)
        self.assertEqual(latest["position_error"], values[-1])
        self.assertIsNone(runtime.take_latest(result.gui_statuses))
        self.assertEqual(latest["error_recording"], result.recorder.status.return_value)

    def test_recording_buttons_use_separate_queue_and_cannot_arm_or_change_target(self):
        start = {"action": "start", "request_id": "record-1"}
        stop = {"action": "stop", "request_id": "record-2"}
        result = self.mocked_run(execute=True, initial_commands=(), stop_after=4,
            initial_recording_commands=(start,), later_recording_commands={2: [stop]})
        self.assertEqual([call.args[0] for call in result.recorder.request.call_args_list], [start, stop])
        result.motion_factory.assert_not_called()
        result.engine.apply_target.assert_not_called()
        result.engine.commit.assert_not_called()
        self.assertEqual(len(result.error_samples), 4)

    def test_recording_retains_unavailable_error_as_invalid_instead_of_fabricating_value(self):
        result = self.mocked_run(initial_commands=(), frozen_joints=True, stop_after=3)
        values = [sample for sample, _ in result.error_samples]
        self.assertEqual(len(values), 3)
        self.assertTrue(values[0]["valid"])
        self.assertFalse(values[-1]["valid"])
        self.assertNotIn("error_m", values[-1])
        self.assertTrue(values[-1]["reason"])

    def test_recording_shutdown_drains_only_after_motion_is_closed_on_transport_fault(self):
        def failure(clock, event):
            raise ControlFault("injected transport failure")
        result = self.mocked_run(execute=True, send_hook=failure, expected=ControlFault)
        self.assertLess(result.order.index("motion_close"), result.order.index("recording_close"))

    def test_recording_error_is_displayed_without_interrupting_an_existing_motion_session(self):
        failed = {"state": "error", "path": "failed.csv", "rows": 2,
                  "valid_samples": 2, "elapsed_s": .04, "error": "disk write failed"}
        result = self.mocked_run(execute=True, stop_after=4, recorder_status=failed)
        self.assertGreater(len(result.sent), 0)
        self.assertEqual(len(result.sent), result.engine.commit.call_count)
        self.assertEqual(runtime.take_latest(result.gui_statuses)["error_recording"], failed)


class DispatchTests(unittest.TestCase):
    def test_existing_main_dispatches_default_and_explicit_follow_to_research(self):
        for mode_arguments in ([], ["--mode", "follow"]):
            with self.subTest(mode_arguments=mode_arguments), patch.object(runtime, "run") as follow:
                control_dual.main([*mode_arguments, "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"])
                follow.assert_called_once()
                self.assertEqual(follow.call_args.args[0].mode, "follow")

    def test_research_entrypoint_defaults_to_follow(self):
        with patch.object(runtime, "run") as run:
            runtime.main(["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"])
        self.assertEqual(run.call_args.args[0].mode, "follow")
        self.assertEqual(run.call_args.args[0].method, "separated")


class FutureSafetyTests(unittest.TestCase):
    """Exercise the real shared MotionSession against a controller API mock."""

    def session(self):
        controller = Mock()
        controller.isConnected.return_value = True
        controller.isProgramRunning.return_value = True
        controller.reuploadScript.return_value = True
        controller.setWatchdog.return_value = True
        controller.getTCPOffset.return_value = [0.] * 6
        controller.isPoseWithinSafetyLimits.return_value = True
        controller.speedL.return_value = True
        session = control_dual.MotionSession(config(True), ("ur5e",), {"ur5e": np.eye(4)},
                                             factory=Mock(return_value=controller))
        self.addCleanup(session.close)
        return session, controller

    def test_next_target_safety_failure_blocks_current_command_and_commit_opportunity(self):
        session, controller = self.session()
        controller.isPoseWithinSafetyLimits.side_effect = [True, False]
        freshness = Mock()
        with self.assertRaisesRegex(ControlFault, "next goal"):
            session.send({"ur5e": np.eye(4)}, {"ur5e": np.zeros(6)}, check_fresh=freshness,
                         safety_targets={"ur5e": np.eye(4)})
        controller.speedL.assert_not_called()
        freshness.assert_not_called()

    def test_freshness_is_checked_after_both_current_and_next_target_preflight(self):
        session, controller = self.session()
        order = []
        controller.isPoseWithinSafetyLimits.side_effect = lambda pose: order.append("safety") or True
        controller.speedL.side_effect = lambda *args: order.append("speed") or True
        session.send({"ur5e": np.eye(4)}, {"ur5e": np.zeros(6)},
                     check_fresh=lambda: order.append("freshness"), safety_targets={"ur5e": np.eye(4)})
        self.assertEqual(order, ["safety", "safety", "freshness", "speed"])


if __name__ == "__main__":
    unittest.main()
