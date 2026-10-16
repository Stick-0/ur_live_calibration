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
    def __init__(self):
        self.last_advance = 10.

    def read(self, now):
        self.last_advance = now
        return SimpleNamespace(tcp=np.eye(4), joints=np.zeros(6), speed=np.zeros(6),
                               joint_velocity=np.zeros(6), robot_timestamp=50. + now)


def config(execute=False):
    arguments = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10",
                 "--ur5e_tcp_offset", *(["0"] * 6), "--ur10e_tcp_offset", *(["0"] * 6)]
    if execute:
        arguments.append("--execute")
    return control_dual.parse_args(arguments)


class ResearchRuntimeTests(unittest.TestCase):
    def mocked_run(self, *, execute=False, initial_commands=("arm",), later_commands=None,
                   send_hook=None, state_error=None, prepare_error=None, expected=None,
                   stop_after=2, step=.04, packet=None, logger_error_kind=None):
        args, clock, order, records = config(execute), Clock(), [], []
        queues = [LocalQueue(maxsize=n) for n in (1, 8, 1, 1)]
        frames, commands, statuses, gui_statuses = queues
        for command in initial_commands:
            commands.put_nowait(command)
        if packet is not None:
            frames.put_nowait(packet)

        def later(wait_count):
            for command in (later_commands or {}).get(wait_count, ()):
                commands.put_nowait(command)

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
                      path=SimpleNamespace(kind="line"), progress=0., task_paused=False,
                      last_rejection="")
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
                        "measurement_age": {name: now - 9.9 for name in runtime.ROBOTS}},
            )

        def commit(proposal):
            order.append("commit")
            engine.progress = proposal.next_progress

        engine.prepare.side_effect, engine.commit.side_effect = prepare, commit
        engine.ingest.return_value = True
        engine.state.side_effect = state_error

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
        session.close.side_effect = lambda: order.append("motion_close")
        motion_factory = Mock(return_value=session)
        receivers = [Mock(), Mock()]
        for index, receiver in enumerate(receivers):
            receiver.disconnect.side_effect = lambda i=index: order.append("receiver_close_" + str(i))
        receive_factory = Mock(side_effect=receivers)
        reader_factory = Mock(side_effect=[Reader(), Reader()])
        logger = Mock()

        def log(record):
            order.append("log_" + record["kind"])
            if record["kind"] == logger_error_kind:
                raise OSError("injected log write failure")
            records.append(deepcopy(record))

        logger.log.side_effect = log
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
            stack.enter_context(redirect_stdout(io.StringIO()))
            if expected is None:
                runtime.run(args)
            else:
                with self.assertRaises(expected) as caught_context:
                    runtime.run(args)
                caught = caught_context.exception

        self.assertTrue(event.is_set())
        logger.close.assert_called_once()
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
                               caught=caught, clock=clock, processes=processes)

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
        result.engine.ingest.assert_called_once_with(packet, 10.)
        record = [r for r in result.records if r["kind"] == "measurement"][0]
        self.assertEqual(record["captured_at"], 9.8)
        self.assertEqual(record["received_at"], 10.)
        self.assertEqual(record["sequence"], 9)
        self.assertEqual(record["poses_camera_model"]["ur5e"], np.eye(4).tolist())

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

    def test_finite_record_preserves_rejected_measurements_without_json_nan(self):
        result = runtime.finite_record({"raw": np.array([np.nan, np.inf, -np.inf, .2]),
                                        "sequence": np.int64(7), "flag": np.bool_(False)})
        self.assertEqual(result, {"raw": [None, None, None, .2], "sequence": 7, "flag": False})


class DispatchTests(unittest.TestCase):
    def test_existing_main_dispatches_follow_to_research_and_hold_to_fixed_pose(self):
        for mode in ("hold", "follow"):
            with self.subTest(mode=mode), patch.object(control_dual, "run") as hold, patch.object(runtime, "run") as follow:
                control_dual.main(["--mode", mode, "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"])
                selected, other = (follow, hold) if mode == "follow" else (hold, follow)
                selected.assert_called_once()
                self.assertEqual(selected.call_args.args[0].mode, mode)
                other.assert_not_called()

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
