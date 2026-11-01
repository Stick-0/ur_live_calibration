"""Explicitly armed live alignment ordering, using only fake robot transports."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import urcal.control.runtime as dual
import urcal.control.startup as startup
import urcal.cooperative.runtime as runtime
from urcal.control.geometry import facing_follower
from urcal.control.policy import CalibrationFrame, CalibrationTracker, ControlFault
from urcal.cooperative.controller import ResearchFault
from urcal.cooperative.state import RelativeStateEstimator
from urcal.cooperative.task import RelativePath
from tests.test_cooperative_runtime import Clock, Event, LocalQueue, Reader, config
from tests.test_cooperative_controller import scene


def calibration_stream(args, frames, clock, events):
    """Inject actual accepted/rejected pairs into the runtime's camera queue."""
    gate = CalibrationTracker(max_age=args.max_calibration_age, stable_frames=5, alpha=1.)
    sequence = 0

    def packet(at, displacement):
        nonlocal sequence
        sequence += 1
        poses = {name: np.eye(4) for name in runtime.ROBOTS}
        poses["ur5e"][0, 3] = displacement
        return CalibrationFrame(sequence, at, poses, {name: 1. for name in runtime.ROBOTS},
                                {name: 2. for name in runtime.ROBOTS},
                                {name: 7 for name in runtime.ROBOTS})

    for at in (9.92, 9.94, 9.96, 9.98, 10.):
        gate.update(packet(at, 0.), at)

    def feed(count):
        if count in events:
            value = events[count]
            frames.put_nowait(value if isinstance(value, CalibrationFrame) else packet(clock.now, value))

    return gate, feed


def check_preflight_reader(test, read_current, clock, gate, readers, engine=None):
    """Arm bases are immutable while fresh encoder failures still propagate."""
    with patch.object(gate, "check_fresh", side_effect=ControlFault("calibration is stale")) as fresh:
        gate.last_rejection = "ur10e calibration jump"
        bases, states = read_current()
        for name in runtime.ROBOTS:
            np.testing.assert_array_equal(bases[name], np.eye(4))
            test.assertEqual(states[name].robot_timestamp, 50. + clock.now)
        bases["ur5e"][0, 3] = 7.
        copied, _ = read_current()
        np.testing.assert_array_equal(copied["ur5e"], np.eye(4))
        encoder_fault = ControlFault("fresh encoder read failed")
        with patch.object(readers[0], "read", side_effect=encoder_fault):
            with test.assertRaises(ControlFault) as encoder:
                read_current()
        test.assertIs(encoder.exception, encoder_fault)
        fresh.assert_not_called()
        gate.last_rejection = ""

    if engine is not None:
        with patch.object(engine, "state", side_effect=ResearchFault("frame_mismatch", "invalid live estimate")) as state:
            read_current()
            state.assert_not_called()


def preflight_jump_packet():
    poses = {name: np.eye(4) for name in runtime.ROBOTS}
    poses["ur5e"][0, 3] = .5149
    return CalibrationFrame(100, 10.05, poses, {name: 1. for name in runtime.ROBOTS},
                            {name: 2. for name in runtime.ROBOTS},
                            {name: 7 for name in runtime.ROBOTS})


def check_queued_preflight_jump(test, read_current, clock, gate, *_):
    """A queued camera outlier cannot replace the copied Arm calibration."""
    accepted, captured = deepcopy(gate.poses), gate.last_capture
    bases, states = read_current()
    test.assertEqual(gate.stable_count, 5)
    test.assertEqual(gate.last_capture, captured)
    for name in runtime.ROBOTS:
        np.testing.assert_array_equal(gate.poses[name], accepted[name])
        np.testing.assert_array_equal(bases[name], accepted[name])
        test.assertEqual(states[name].robot_timestamp, 50. + clock.now)


class LiveAlignmentRuntimeTests(unittest.TestCase):
    def run_fake(self, *, enabled=True, execute=True, commands=("arm",), stop_after=3,
                 done_after=2, plan_error=None, propose_error=None, send_error=None,
                 fresh_hook=None, preflight_seconds=0., later_commands=None, process_failure=False,
                 factory_error=None, tracking_ready=False, plan_error_once=False, preflight_hook=None,
                 calibration_events=None, preflight_frame=None, preflight_commands=(), tracking_error=None,
                 selected_targets=None, cartesian_acceleration=None):
        args = config(execute)
        args.stable_frames = 5  # Match the explicit five-frame calibration stream below.
        args.initial_align = enabled
        clock, order, records = Clock(), [], []
        queues = [LocalQueue(maxsize=size) for size in (1, 8, 1, 1)]
        frames, command_queue, statuses, gui_statuses = queues
        for command in commands:
            command_queue.put_nowait(command)
        real_gate, feed = calibration_stream(args, frames, clock, calibration_events or {})

        def later(count):
            feed(count)
            for command in (later_commands or {}).get(count, ()):
                command_queue.put_nowait(command)

        stop = Event(clock, stop_after, later, .04)
        processes = [Mock(pid=401), Mock(pid=402)]
        for process in processes:
            process.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = queues
        context.Event.return_value = stop
        context.Process.side_effect = processes
        readers = [Reader(), Reader()]
        receivers = [Mock(), Mock()]
        for index, receiver in enumerate(receivers):
            receiver.disconnect.side_effect = lambda i=index: order.append("receiver_close_" + str(i))

        engine = Mock(leader="ur10e", follower="ur5e", commanded=("ur5e",),
                      offsets={name: np.eye(4) for name in runtime.ROBOTS},
                      path=RelativePath(facing_follower(np.eye(4), .3), kind="hold"),
                      gap=.3, progress=0., task_paused=False, invalid_since=None, last_rejection="",
                      initial_tcp=None, settle_elapsed=3., finished=True,
                      preview_recovery=Mock(count=0), reacquisition_count=0,
                      live_calibration=None)
        engine.gap_approach.state = "idle"
        engine.gap_approach_status.side_effect = lambda **_kwargs: {"state": "idle", "enabled": False}
        histories = ("previous_twist", "previous_qdot", "previous_compensation",
                     "previous_compensation_qdot", "previous_feedback",
                     "previous_base_compensation", "previous_base_compensation_qdot")
        for attribute in histories:
            setattr(engine, attribute, np.full(6, .3))
        engine.state.return_value = SimpleNamespace(base_poses={name: np.eye(4) for name in runtime.ROBOTS},
                                                    measurement_age={name: .1 for name in runtime.ROBOTS})
        engine.gate.last_rejection = ""
        engine.gate.last_capture = 10.
        engine.gate.poses = {name: np.eye(4) for name in runtime.ROBOTS}
        engine.mappings = {name: np.eye(4) for name in runtime.ROBOTS}
        engine.ingest.return_value = True
        engine._new_estimator.side_effect = lambda: RelativeStateEstimator(
            mode="predict", alpha=args.filter_alpha, velocity_alpha=args.velocity_alpha,
            max_prediction_age=args.prediction_horizon, max_measurement_age=args.max_calibration_age,
            max_linear_speed=args.max_base_speed, max_angular_speed=args.max_base_angular_speed)
        if real_gate is not None:
            engine.gate = real_gate

            def ingest(frame, at, *, allow_reacquire=False, active_recovery=False):
                self.assertFalse(active_recovery)
                try:
                    accepted = engine.gate.update(frame, at)
                except ControlFault as exc:
                    engine.last_rejection = str(exc)
                    return False
                engine.last_rejection = ""
                return accepted

            def state(at, active=False):
                if not active:
                    engine.gate.check_fresh(at)
                return SimpleNamespace(base_poses=engine.gate.poses,
                                       measurement_age={name: at - engine.gate.last_capture for name in runtime.ROBOTS})

            engine.ingest.side_effect = ingest
            engine.state.side_effect = state
        observed_bases = []
        alignment = SimpleNamespace(goal_joints={"ur5e": np.ones(6) * .02},
                                    predicted_joints={"ur5e": np.ones(6) * .001},
                                    joint_accelerations={"ur5e": .0125},
                                    progress=0., done=False, commits=0)

        def propose(states, bases, dt):
            order.append("alignment_propose")
            observed_bases.append(deepcopy(bases))
            self.assertGreater(clock.now, 10. + preflight_seconds)
            self.assertEqual(states["ur5e"].robot_timestamp, 50. + clock.now)
            if propose_error:
                raise propose_error
            return {"ur5e": np.ones(6) * .001}

        def alignment_commit():
            order.append("alignment_commit")
            alignment.commits += 1
            alignment.progress = alignment.commits / done_after
            alignment.done = alignment.commits >= done_after

        alignment.propose = Mock(side_effect=propose)
        def propose_hold(states, bases, dt):
            order.append("alignment_hold")
            self.assertTrue(alignment.done)
            self.assertEqual(states["ur5e"].robot_timestamp, 50. + clock.now)
            return {"ur5e": np.zeros(6)}

        alignment.propose_hold = Mock(side_effect=propose_hold)
        alignment.commit = Mock(side_effect=alignment_commit)
        preparations = []

        def prepare(now, states, dt, active=False):
            order.append("prepare")
            preparations.append({"at": now, "active": active,
                                 "histories": {key: getattr(engine, key).copy() for key in histories},
                                 "progress": engine.progress, "settle_elapsed": engine.settle_elapsed,
                                 "finished": engine.finished})
            if not alignment.done and not tracking_ready:
                raise ResearchFault("tracking_error", "original pose is outside tracking limits")
            if tracking_error is not None:
                raise tracking_error
            return SimpleNamespace(goals={"ur5e": np.eye(4)}, tcp_goals={"ur5e": np.eye(4)},
                                   future_tcp_goals={"ur5e": np.eye(4)}, twists={"ur5e": np.full(6, .001)},
                                   next_progress=.01, decision=SimpleNamespace(task_scale=1.),
                                   record={"kind": "control", "at": now,
                                           "bases": {name: np.eye(4) for name in runtime.ROBOTS},
                                           "measurement_age": {name: .1 for name in runtime.ROBOTS},
                                           **({"speedl_acceleration": cartesian_acceleration}
                                              if cartesian_acceleration is not None else {})})

        def commit(proposal):
            order.append("tracking_commit")
            engine.progress = proposal.next_progress

        engine.prepare.side_effect = prepare
        engine.commit.side_effect = commit
        session = Mock()

        def send_joints(qdots, next_joints, check_fresh, accelerations):
            order.append("joint_preflight")
            self.assertEqual(set(qdots), {"ur5e"})
            self.assertEqual(set(next_joints), {"ur5e"})
            self.assertEqual(accelerations, alignment.joint_accelerations)
            if fresh_hook is not None:
                fresh_hook(clock, stop, engine, readers)
            if process_failure:
                processes[0].is_alive.return_value = False
            check_fresh()
            if send_error:
                raise send_error
            order.append("speedJ")

        def send(targets, twists, check_fresh, safety_targets, *, accelerations=None):
            self.assertEqual(set(targets), {"ur5e"})
            self.assertEqual(set(twists), {"ur5e"})
            self.assertEqual(set(safety_targets), {"ur5e"})
            self.assertEqual(accelerations, None if cartesian_acceleration is None
                             else {"ur5e": cartesian_acceleration})
            if fresh_hook is not None:
                fresh_hook(clock, stop, engine, readers)
            check_fresh()
            order.append("speedL")

        session.send_joints.side_effect = send_joints
        session.send.side_effect = send
        session.close.side_effect = lambda: order.append("motion_close")
        factory_attempts = []

        def construct(*_args, **_kwargs):
            factory_attempts.append(clock.now)
            if factory_error is not None and len(factory_attempts) == 1:
                raise factory_error
            return session

        motion_factory = Mock(side_effect=construct)

        def preflight(supplied_session, supplied_args, offsets, bases, states, targets, read_current, report, *, wait,
                      diagnostic=None):
            order.append("native_preflight")
            self.assertIs(supplied_session, session)
            self.assertEqual(set(targets), {"ur5e"})
            self.assertEqual(wait, stop.wait)
            session.send.assert_not_called()
            session.send_joints.assert_not_called()
            clock.now += preflight_seconds
            if preflight_frame is not None:
                frames.put_nowait(preflight_frame)
            if preflight_hook is not None:
                preflight_hook(read_current, clock, engine.gate, readers, engine)
            read_current()
            report("PREFLIGHT: native path checked; no motion yet")
            if plan.call_count == 1:
                for command in preflight_commands:
                    command_queue.put_nowait(deepcopy(command))
            if plan_error and (not plan_error_once or plan.call_count == 1):
                raise plan_error
            if selected_targets is not None:
                alignment.targets = deepcopy(selected_targets)
                alignment.nominal_targets = deepcopy(targets)
                alignment.target_adjustments = {"adjustments": {"ur5e": {"position_m": .025}}}
            return alignment

        plan = Mock(side_effect=preflight)
        check_request = Mock()
        logger = Mock()

        def log(record):
            order.append("log_" + record["kind"])
            records.append(deepcopy(record))

        logger.log.side_effect = log
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
            stack.enter_context(patch.object(runtime, "StateReader", side_effect=readers))
            stack.enter_context(patch.object(runtime, "MotionSession", motion_factory))
            stack.enter_context(patch.object(runtime, "ExperimentLogger", return_value=logger))
            stack.enter_context(patch.object(runtime, "prepare_alignment", plan))
            stack.enter_context(patch.object(runtime, "check_alignment_request", check_request))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                runtime.run(args)
            except (ControlFault, ValueError) as exc:
                caught = exc
        for receiver in receivers:
            receiver.disconnect.assert_called_once()
        for queue in queues:
            self.assertTrue(queue.cancelled and queue.closed)
        logger.close.assert_called_once()
        self.assertTrue(stop.is_set())
        return SimpleNamespace(args=args, caught=caught, engine=engine, alignment=alignment, session=session,
                               motion_factory=motion_factory, plan=plan, records=records, order=order,
                               preparations=preparations, statuses=statuses, gui_statuses=gui_statuses,
                               factory_attempts=factory_attempts, observed_bases=observed_bases, stop=stop)

    def test_parser_alignment_is_off_by_default_and_requires_explicit_flag(self):
        self.assertFalse(config().initial_align)
        args = dual.parse_args(["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10",
                                "--initial_align"])
        self.assertTrue(args.initial_align)
        self.assertFalse(args.execute)

    def test_disabled_alignment_preserves_tracking_error_and_receive_only_wait(self):
        result = self.run_fake(enabled=False, stop_after=1)
        self.assertIsNone(result.caught)
        result.plan.assert_not_called()
        result.motion_factory.assert_not_called()
        self.assertEqual(result.engine.prepare.call_count, 1)
        self.assertEqual(runtime.take_latest(result.gui_statuses)["fault_category"], "tracking_error")

    def test_execute_still_requires_arm_and_preview_arm_cannot_open_motion(self):
        for execute, commands in ((True, ()), (False, ("arm",))):
            with self.subTest(execute=execute):
                result = self.run_fake(execute=execute, commands=commands, stop_after=1)
                self.assertIsNone(result.caught)
                result.motion_factory.assert_not_called()
                result.plan.assert_not_called()
                result.engine.prepare.assert_not_called()
                result.engine.commit.assert_not_called()

    def test_arm_performs_only_native_preflight_with_deferred_watchdog(self):
        result = self.run_fake(stop_after=1, preflight_seconds=.4)
        self.assertIsNone(result.caught)
        result.motion_factory.assert_called_once()
        self.assertEqual(result.motion_factory.call_args.args[1], ("ur5e",))
        self.assertTrue(result.motion_factory.call_args.kwargs["defer_watchdog"])
        result.plan.assert_called_once()
        result.session.send.assert_not_called()
        result.session.send_joints.assert_not_called()
        result.engine.prepare.assert_not_called()
        result.alignment.commit.assert_not_called()
        self.assertEqual([record["phase"] for record in result.records if record["kind"] == "armed"], ["initial_alignment"])
        armed = next(record for record in result.records if record["kind"] == "armed")
        self.assertAlmostEqual(armed["initial_state_at"], 10.)
        self.assertAlmostEqual(armed["at"], 10.4)
        request = next(record for record in result.records if record["kind"] == "alignment_request")
        self.assertAlmostEqual(request["at"], 10.)
        self.assertAlmostEqual(request["initial_state_at"], 10.)
        self.assertLess(result.order.index("log_alignment_request"), result.order.index("native_preflight"))
        for name, target in result.plan.call_args.args[5].items():
            np.testing.assert_array_equal(request["tcp_goals"][name], target)
        initial_states = result.plan.call_args.args[4]
        for name in runtime.ROBOTS:
            np.testing.assert_array_equal(armed["initial_robots"][name]["q"], initial_states[name].joints)
            np.testing.assert_array_equal(armed["initial_robots"][name]["tcp"], initial_states[name].tcp)
            np.testing.assert_array_equal(armed["initial_bases"][name], result.plan.call_args.args[3][name])
            np.testing.assert_array_equal(request["initial_robots"][name]["q"], initial_states[name].joints)
            np.testing.assert_array_equal(request["initial_bases"][name], result.plan.call_args.args[3][name])

    def test_alignment_uses_next_fresh_tick_and_commits_only_after_joint_send(self):
        result = self.run_fake(stop_after=3, done_after=5, preflight_seconds=.4)
        self.assertIsNone(result.caught)
        result.engine.prepare.assert_not_called()
        result.engine.commit.assert_not_called()
        result.session.send.assert_not_called()
        self.assertEqual(result.session.send_joints.call_count, 2)
        self.assertEqual(result.alignment.commit.call_count, 2)
        self.assertLess(result.order.index("native_preflight"), result.order.index("alignment_propose"))
        self.assertLess(result.order.index("log_alignment_proposal"), result.order.index("speedJ"))
        self.assertLess(result.order.index("speedJ"), result.order.index("alignment_commit"))
        for record in result.records:
            if record["kind"] in ("alignment_proposal", "alignment"):
                self.assertAlmostEqual(record["dt"], .04)
            if record["kind"] == "alignment_proposal":
                self.assertEqual(set(record["bases"]), set(runtime.ROBOTS))
                self.assertTrue(set(record["qdots"]) <= set(record["predicted_joints"]))

    def test_completion_waits_for_five_new_frames_then_zeroes_histories_before_follow(self):
        result = self.run_fake(done_after=1, stop_after=8,
                               calibration_events={index: 0. for index in range(2, 7)})
        self.assertIsNone(result.caught)
        self.assertEqual(result.session.send_joints.call_count, 6)
        self.assertEqual(result.alignment.commit.call_count, 1)
        self.assertEqual(result.alignment.propose_hold.call_count, 5)
        self.assertEqual(result.session.send.call_count, 1)
        self.assertEqual(result.engine.commit.call_count, 1)
        self.assertEqual(len(result.preparations), 2)
        self.assertFalse(result.preparations[0]["active"])
        self.assertAlmostEqual(result.preparations[0]["at"], 10.24)
        self.assertTrue(result.preparations[1]["active"])
        self.assertAlmostEqual(result.preparations[1]["at"], 10.28)
        for prepared in result.preparations:
            self.assertEqual(prepared["progress"], 0.)
            self.assertEqual(prepared["settle_elapsed"], 0.)
            self.assertFalse(prepared["finished"])
            for value in prepared["histories"].values():
                np.testing.assert_array_equal(value, np.zeros(6))
        for call in result.session.send_joints.call_args_list[1:]:
            np.testing.assert_array_equal(call.args[0]["ur5e"], np.zeros(6))
        self.assertLess(result.order.index("alignment_commit"), result.order.index("log_alignment_endpoint"))
        self.assertLess(result.order.index("log_alignment_endpoint"), result.order.index("prepare"))
        self.assertLess(result.order.index("prepare"), result.order.index("log_alignment_complete"))
        self.assertLess(result.order.index("log_alignment_complete"), result.order.index("speedL"))
        self.assertLess(result.order.index("speedL"), result.order.index("tracking_commit"))
        self.assertEqual(result.engine._new_estimator.call_count, 1)
        estimate = result.engine.estimator.estimate(10.28, leader="ur10e")
        for velocity in estimate.base_twists.values():
            np.testing.assert_array_equal(velocity, np.zeros(6))

    def test_bounded_follow_acceleration_reaches_transport_after_fresh_handoff(self):
        result = self.run_fake(done_after=1, stop_after=8, cartesian_acceleration=.027,
                               calibration_events={index: 0. for index in range(2, 7)})
        self.assertIsNone(result.caught)
        self.assertEqual(result.session.send.call_count, 1)
        self.assertEqual(result.session.send.call_args.kwargs["accelerations"], {"ur5e": .027})
        self.assertEqual(result.engine.commit.call_count, 1)
        self.assertLess(result.order.index("speedL"), result.order.index("tracking_commit"))

    def test_plan_and_proposal_failures_close_motion_without_velocity_or_commit(self):
        for field in ("plan_error", "propose_error"):
            with self.subTest(field=field):
                result = self.run_fake(**{field: ControlFault("injected alignment failure")})
                self.assertIsInstance(result.caught, ControlFault)
                result.session.send.assert_not_called()
                result.session.send_joints.assert_not_called()
                result.alignment.commit.assert_not_called()
                result.engine.commit.assert_not_called()
                self.assertLess(result.order.index("motion_close"), result.order.index("receiver_close_1"))
                if field == "propose_error":
                    fault = next(record for record in result.records if record["kind"] == "fault")
                    self.assertAlmostEqual(fault["control_at"], 10.04)
                    self.assertAlmostEqual(fault["control_dt"], .04)
                    self.assertGreaterEqual(fault["at"], fault["control_at"])

    def test_joint_send_failure_never_commits_or_starts_follow(self):
        for fault in (ControlFault("speedJ failed"), dual.ArmConnectionError("ur5e", "192.0.2.5"),
                      startup.AlignmentCalibrationTimeout("expired after active command"),
                      dual.ArmIKUnavailable("ur5e", np.eye(4), np.zeros(6), np.eye(4))):
            with self.subTest(fault=type(fault).__name__):
                result = self.run_fake(send_error=fault)
                self.assertIsInstance(result.caught, ControlFault)
                result.alignment.commit.assert_not_called()
                result.engine.prepare.assert_not_called()
                result.engine.commit.assert_not_called()
                result.session.send.assert_not_called()
                self.assertNotIn("speedJ", result.order)
                self.assertIn("motion_close", result.order)

    def test_stop_or_stale_robot_during_send_blocks_joint_command(self):
        def stopped(clock, event, engine, readers):
            event.set()

        def expired(clock, event, engine, readers):
            clock.now += .2

        for hook in (stopped, expired):
            with self.subTest(hook=hook.__name__):
                result = self.run_fake(fresh_hook=hook)
                self.assertIsInstance(result.caught, ControlFault)
                self.assertNotIn("speedJ", result.order)
                result.alignment.commit.assert_not_called()
                result.engine.commit.assert_not_called()
                result.session.send.assert_not_called()
                self.assertLess(result.order.index("motion_close"), result.order.index("receiver_close_1"))

    def test_process_death_during_joint_preflight_blocks_send_without_shared_stop(self):
        result = self.run_fake(process_failure=True)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertNotIn("speedJ", result.order)
        result.alignment.commit.assert_not_called()
        self.assertIn("motion_close", result.order)

    def test_remote_mode_rejection_preserves_follow_preview_without_automatic_retry(self):
        for enabled in (False, True):
            with self.subTest(initial_align=enabled):
                fault = dual.ArmConnectionError("ur5e", "192.0.2.5")
                result = self.run_fake(enabled=enabled, tracking_ready=True, factory_error=fault, stop_after=4)
                self.assertIsNone(result.caught)
                self.assertEqual(result.motion_factory.call_count, 1)
                self.assertEqual(result.factory_attempts, [10.])
                result.plan.assert_not_called()
                result.session.send.assert_not_called()
                result.session.send_joints.assert_not_called()
                result.engine.commit.assert_not_called()
                status = runtime.take_latest(result.gui_statuses)
                self.assertTrue(status["ready"])
                self.assertFalse(status["active"])
                self.assertIn("ARM BLOCKED", status["message"])
                self.assertIn("192.0.2.5", status["message"])

    def test_new_explicit_arm_retries_follow_connection_and_then_can_move(self):
        for enabled in (False, True):
            with self.subTest(initial_align=enabled):
                fault = dual.ArmConnectionError("ur5e", "192.0.2.5")
                result = self.run_fake(enabled=enabled, tracking_ready=True, factory_error=fault,
                                       later_commands={2: ["arm"]}, stop_after=4)
                self.assertIsNone(result.caught)
                self.assertEqual(result.motion_factory.call_count, 2)
                np.testing.assert_allclose(result.factory_attempts, [10., 10.08])
                self.assertEqual(result.session.send_joints.call_count, int(enabled))
                self.assertEqual(result.session.send.call_count, int(not enabled))
                self.assertNotIn("ARM BLOCKED", runtime.take_latest(result.gui_statuses)["message"])

    def test_follow_preflight_calibration_timeout_closes_then_waits_for_new_arm(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                fault = startup.AlignmentCalibrationTimeout("five seconds without stable calibration")
                result = self.run_fake(plan_error=fault, plan_error_once=True,
                                       later_commands={2: ["arm"]} if retry else {}, stop_after=4)
                self.assertIsNone(result.caught)
                self.assertEqual(result.motion_factory.call_count, 2 if retry else 1)
                self.assertEqual(result.plan.call_count, 2 if retry else 1)
                self.assertGreaterEqual(result.session.close.call_count, 1)
                result.session.send.assert_not_called()
                self.assertEqual(result.session.send_joints.call_count, int(retry))
                if not retry:
                    status = runtime.take_latest(result.gui_statuses)
                    self.assertTrue(status["ready"])
                    self.assertFalse(status["active"])
                    self.assertEqual(status["fault_category"], "arm_calibration")
                    self.assertIn("ARM BLOCKED", status["message"])
                    result.alignment.commit.assert_not_called()

    def test_follow_native_ik_unavailable_preserves_preview_until_explicit_arm_retry(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                fault = dual.ArmIKUnavailable("ur5e", np.eye(4), np.zeros(6), np.eye(4))
                result = self.run_fake(plan_error=fault, plan_error_once=True,
                                       later_commands={2: ["arm"]} if retry else {}, stop_after=4)
                self.assertIsNone(result.caught)
                self.assertEqual(result.motion_factory.call_count, 2 if retry else 1)
                self.assertEqual(result.plan.call_count, 2 if retry else 1)
                np.testing.assert_allclose(result.factory_attempts, [10., 10.08] if retry else [10.])
                self.assertGreaterEqual(result.session.close.call_count, 1)
                self.assertLess(result.order.index("motion_close"), result.order.index("receiver_close_1"))
                result.session.send.assert_not_called()
                result.session.enable_watchdog.assert_not_called()
                self.assertEqual(result.session.send_joints.call_count, int(retry))
                self.assertEqual(result.alignment.commit.call_count, int(retry))
                result.engine.commit.assert_not_called()
                status = runtime.take_latest(result.gui_statuses)
                if retry:
                    self.assertNotIn("ARM BLOCKED", status["message"])
                    self.assertLess(result.order.index("motion_close"), result.order.index("speedJ"))
                else:
                    self.assertTrue(status["ready"])
                    self.assertFalse(status["active"])
                    self.assertEqual(status["fault_category"], "arm_ik")
                    self.assertIn("ARM BLOCKED", status["message"])
                    self.assertIn("ur5e", status["message"].lower())
                    self.assertFalse(any(record["kind"] == "armed" for record in result.records))

                requests = [record for record in result.records if record["kind"] == "alignment_request"]
                self.assertEqual(len(requests), 2 if retry else 1)
                rejected = next(record for record in result.records
                                if record["kind"] == "wait" and record["category"] == "arm_ik")
                np.testing.assert_array_equal(rejected["details"]["target_tcp"], np.eye(4))

    def test_follow_failed_preflight_discards_queued_arms_but_allows_later_arm(self):
        for fault in (dual.ArmIKUnavailable("ur5e", np.eye(4), np.zeros(6), np.eye(4)),
                      startup.AlignmentCalibrationTimeout("unstable during preflight")):
            for retry in (False, True):
                with self.subTest(fault=type(fault).__name__, retry=retry):
                    # A multiprocessing feeder can deliver a pre-failure click
                    # after the immediate queue drain has already finished.
                    later = {1: [("arm", 10.)]}
                    if retry:
                        later[2] = [("arm", 10.08)]
                    result = self.run_fake(plan_error=fault, plan_error_once=True,
                                           preflight_commands=("arm", "arm"),
                                           later_commands=later, stop_after=4)
                    self.assertIsNone(result.caught)
                    self.assertEqual(result.plan.call_count, 2 if retry else 1)
                    np.testing.assert_allclose(result.factory_attempts, [10., 10.08] if retry else [10.])
                    self.assertEqual(result.session.send_joints.call_count, int(retry))
                    self.assertEqual(result.alignment.commit.call_count, int(retry))
                    result.session.send.assert_not_called()
                    if not retry:
                        status = runtime.take_latest(result.gui_statuses)
                        self.assertFalse(status["active"])
                        self.assertIn("ARM BLOCKED", status["message"])

    def test_follow_failed_preflight_preserves_queued_target_and_stop(self):
        target = {"type": "target", "gap": .35}
        result = self.run_fake(plan_error=dual.ArmIKUnavailable(
            "ur5e", np.eye(4), np.zeros(6), np.eye(4)), plan_error_once=True,
            preflight_commands=("arm", target, "arm", "stop"), stop_after=8)
        self.assertIsNone(result.caught)
        result.plan.assert_called_once()
        result.motion_factory.assert_called_once()
        result.engine.apply_target.assert_called_once_with(target)
        self.assertLess(result.stop.waits, 8)
        result.session.send.assert_not_called()
        result.session.send_joints.assert_not_called()
        result.alignment.commit.assert_not_called()
        result.engine.commit.assert_not_called()

    def test_follow_preflight_reader_keeps_arm_snapshot_despite_expired_or_rejected_vision(self):
        result = self.run_fake(stop_after=1, preflight_seconds=.7,
                               preflight_hook=lambda *args: check_preflight_reader(self, *args))
        self.assertIsNone(result.caught)
        result.plan.assert_called_once()
        result.session.send.assert_not_called()
        result.session.send_joints.assert_not_called()

    def test_follow_preflight_reader_ignores_actual_queued_jump(self):
        result = self.run_fake(stop_after=1, preflight_seconds=.1, calibration_events={},
                               preflight_frame=preflight_jump_packet(),
                               preflight_hook=lambda *args: check_queued_preflight_jump(self, *args))
        self.assertIsNone(result.caught)
        self.assertEqual(result.engine.gate.stable_count, 5)
        result.session.send.assert_not_called()
        result.session.send_joints.assert_not_called()

    def test_follow_rejects_jump_pose_but_continues_until_one_accepted_frame_recovers(self):
        for enabled in (False, True):
            with self.subTest(initial_align=enabled):
                result = self.run_fake(enabled=enabled, tracking_ready=True, done_after=99, stop_after=4,
                                       calibration_events={1: .119, 2: .003})
                self.assertIsNone(result.caught)
                result.motion_factory.assert_called_once()
                gate = result.engine.gate
                measurements = [r for r in result.records if r["kind"] == "measurement"]
                if enabled:
                    self.assertEqual(gate.stable_count, 5)
                    self.assertAlmostEqual(gate.last_capture, 10.)
                    self.assertEqual(measurements, [])
                    self.assertEqual(result.alignment.commit.call_count, 3)
                    np.testing.assert_array_equal([b["ur5e"][0, 3] for b in result.observed_bases], [0., 0., 0.])
                    result.session.send.assert_not_called()
                else:
                    self.assertEqual(gate.stable_count, 1)
                    self.assertFalse(gate.ready)
                    self.assertAlmostEqual(gate.last_capture, 10.08)
                    self.assertEqual(gate.last_rejection, "")
                    self.assertAlmostEqual(gate.poses["ur5e"][0, 3], .003)
                    self.assertEqual([r["accepted"] for r in measurements], [False, True])
                    self.assertEqual(result.engine.commit.call_count, 3)
                    self.assertEqual(result.session.send.call_count, 3)
                    # Ordinary following retains the controller's active state/prediction path.
                    self.assertTrue(all(call.kwargs["active"] for call in result.engine.state.call_args_list))

    def test_follow_invalid_timeout_is_terminal_even_if_a_good_frame_arrives_late(self):
        for late_recovery in (False, True):
            with self.subTest(late_recovery=late_recovery):
                events = {1: .119, **({9: .003} if late_recovery else {})}
                result = self.run_fake(enabled=False, tracking_ready=True, stop_after=12, calibration_events=events)
                self.assertIsInstance(result.caught, ControlFault)
                self.assertIn("remained invalid", str(result.caught))
                self.assertGreater(result.engine.commit.call_count, 1)
                self.assertLess(result.engine.commit.call_count, 11)
                result.motion_factory.assert_called_once()
                self.assertIn("motion_close", result.order)
                result.session.send_joints.assert_not_called()

    def test_follow_last_good_capture_age_still_expires_inside_rejection_grace(self):
        result = self.run_fake(enabled=False, tracking_ready=True, stop_after=16, calibration_events={11: .119})
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("stale", str(result.caught))
        self.assertAlmostEqual(result.engine.gate.last_capture, 10.)
        self.assertGreater(result.engine.commit.call_count, 1)
        self.assertIn("motion_close", result.order)

    def test_follow_send_callback_rechecks_rejection_deadline_before_velocity_and_commit(self):
        def cross_deadline(clock, stop, engine, readers):
            if clock.now >= 10.31:
                clock.now += .04  # Encoder age remains below .15s.

        result = self.run_fake(enabled=False, tracking_ready=True, stop_after=12, calibration_events={1: .119},
                               fresh_hook=cross_deadline)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("remained invalid", str(result.caught))
        self.assertEqual(result.session.send.call_count, result.engine.commit.call_count + 1)
        self.assertEqual(result.order.count("speedL"), result.engine.commit.call_count)
        result.alignment.commit.assert_not_called()

    def test_follow_preview_rejection_and_single_recovery_frame_cannot_arm(self):
        result = self.run_fake(commands=(), calibration_events={1: .119, 2: .003},
                               later_commands={1: ["arm"], 2: ["arm"]}, stop_after=4)
        self.assertIsNone(result.caught)
        result.motion_factory.assert_not_called()
        result.plan.assert_not_called()
        self.assertEqual(result.engine.gate.stable_count, 1)

    def test_follow_native_preflight_and_alignment_camera_frames_do_not_change_live_gate(self):
        frame = CalibrationFrame(1, 10., {name: np.eye(4) for name in runtime.ROBOTS},
                                 {name: 1. for name in runtime.ROBOTS},
                                 {name: 2. for name in runtime.ROBOTS},
                                 {name: 7 for name in runtime.ROBOTS})
        preflight = self.run_fake(stop_after=1, preflight_frame=frame)
        self.assertIsNone(preflight.caught)
        preflight.engine.ingest.assert_not_called()
        active = self.run_fake(done_after=99, stop_after=4, calibration_events={1: .119, 2: .003})
        self.assertIsNone(active.caught)
        active.engine.ingest.assert_not_called()
        ignored = [record for record in active.records if record["kind"] == "alignment_vision"]
        self.assertEqual(len(ignored), 2)
        self.assertTrue(all(not record["accepted"] for record in ignored))
        self.assertEqual(active.engine.gate.stable_count, 5)

    def test_follow_alignment_runs_with_captured_calibration_beyond_camera_age_and_rejection_deadline(self):
        result = self.run_fake(done_after=99, stop_after=18, calibration_events={1: .5149})
        self.assertIsNone(result.caught)
        self.assertEqual(result.alignment.commit.call_count, 17)
        self.assertEqual(result.engine.gate.last_capture, 10.)
        np.testing.assert_array_equal([base["ur5e"][0, 3] for base in result.observed_bases], np.zeros(17))
        result.engine.prepare.assert_not_called()
        result.session.send.assert_not_called()
        status = runtime.take_latest(result.gui_statuses)
        self.assertEqual(status["scene"]["calibration_mode"], "frozen")
        self.assertGreater(status["scene"]["measurement_age"], .5)

    def test_follow_handoff_needs_five_fresh_frames_and_sends_zero_joint_commands(self):
        result = self.run_fake(done_after=1, stop_after=6,
                               calibration_events={index: 0. for index in range(2, 6)})
        self.assertIsNone(result.caught)
        self.assertEqual(result.alignment.commit.call_count, 1)
        self.assertEqual(result.alignment.propose_hold.call_count, 4)
        result.session.send.assert_not_called()
        result.engine.prepare.assert_not_called()
        self.assertFalse(any(record["kind"] == "alignment_complete" for record in result.records))
        for call in result.session.send_joints.call_args_list[1:]:
            np.testing.assert_array_equal(call.args[0]["ur5e"], np.zeros(6))

    def test_follow_handoff_ignores_inflight_frames_captured_before_endpoint(self):
        old = replace(preflight_jump_packet(), captured_at=10.)
        result = self.run_fake(done_after=1, stop_after=13,
                               calibration_events={**{index: old for index in range(2, 7)},
                                                   **{index: 0. for index in range(7, 12)}})
        self.assertIsNone(result.caught)
        handoff = [record for record in result.records if record["kind"] == "handoff_measurement"]
        self.assertEqual([record["accepted"] for record in handoff], [False]*5 + [True]*5)
        completion = next(record for record in result.records if record["kind"] == "alignment_complete")
        self.assertAlmostEqual(completion["at"], 10.44)
        self.assertEqual(result.session.send.call_count, 1)

    def test_follow_handoff_missing_camera_times_out_without_following(self):
        result = self.run_fake(done_after=1, stop_after=130)
        self.assertIsInstance(result.caught, ControlFault)
        self.assertIn("post-alignment calibration wait exceeded", str(result.caught))
        self.assertEqual(result.alignment.commit.call_count, 1)
        self.assertGreater(result.alignment.propose_hold.call_count, 100)
        result.session.send.assert_not_called()
        result.engine.prepare.assert_not_called()
        self.assertIn("motion_close", result.order)

    def test_follow_handoff_rejects_changed_base_frame_before_following(self):
        result = self.run_fake(done_after=1, stop_after=8,
                               calibration_events={index: .02 for index in range(2, 7)})
        self.assertIsNone(result.caught)
        self.assertIn("base calibration changed", runtime.take_latest(result.gui_statuses)["message"])
        result.session.send.assert_not_called()
        result.engine.prepare.assert_not_called()
        self.assertFalse(any(record["kind"] == "alignment_complete" for record in result.records))

    def test_follow_handoff_checks_original_tracking_task_before_first_speedl(self):
        fault = ResearchFault("tracking_error", "live target exceeds tracking error after alignment")
        result = self.run_fake(done_after=1, stop_after=8, tracking_error=fault,
                               calibration_events={index: 0. for index in range(2, 7)})
        self.assertIsNone(result.caught)
        self.assertIn(str(fault), runtime.take_latest(result.gui_statuses)["message"])
        self.assertEqual(result.engine.prepare.call_count, 1)
        self.assertFalse(result.engine.prepare.call_args.kwargs["active"])
        result.session.send.assert_not_called()
        result.engine.commit.assert_not_called()
        self.assertFalse(any(record["kind"] == "alignment_complete" for record in result.records))

    def test_follow_handoff_zero_command_still_checks_stop_and_encoder_age(self):
        for stop_operator in (False, True):
            def fault_in_hold(clock, event, engine, readers):
                if clock.now > 10.06:
                    if stop_operator:
                        event.set()
                    else:
                        clock.now += .2

            with self.subTest(stop=stop_operator):
                result = self.run_fake(done_after=1, stop_after=5, fresh_hook=fault_in_hold)
                self.assertIsInstance(result.caught, ControlFault)
                self.assertEqual(result.alignment.commit.call_count, 1)
                self.assertEqual(result.order.count("speedJ"), 1)
                result.session.send.assert_not_called()

    def test_follow_after_handoff_uses_new_live_frames_and_retains_rejection_grace(self):
        result = self.run_fake(done_after=1, stop_after=10,
                               calibration_events={**{index: 0. for index in range(2, 7)},
                                                   7: .119, 8: .003})
        self.assertIsNone(result.caught)
        measurements = [record for record in result.records if record["kind"] == "measurement"]
        self.assertEqual([record["accepted"] for record in measurements], [False, True])
        self.assertEqual(result.engine.gate.stable_count, 1)
        self.assertAlmostEqual(result.engine.gate.poses["ur5e"][0, 3], .003)
        self.assertEqual(result.session.send.call_count, 3)
        self.assertEqual(runtime.take_latest(result.gui_statuses)["scene"]["calibration_mode"], "live")


class NativeAlignmentPreparationTests(unittest.TestCase):
    def prepare(self, *, incomplete=False, changed_state=False, native_error=False):
        engine, bases, states = scene()
        targets = {"ur5e": states["ur5e"].tcp.copy()}
        order = []
        samples = [{name: value.joints.copy() for name, value in states.items()} for _ in range(2)]
        alignment = SimpleNamespace(samples=samples, models=engine.models, guard=engine.guard,
                                    command_speed_fraction=.9)
        factory = Mock(side_effect=lambda *args: (order.append("build_path") or alignment))
        session = Mock()
        reporter = Mock()

        def read_current():
            order.append("fresh")
            current = deepcopy(states)
            if changed_state:
                current["ur10e"].joints[0] += .02
            return deepcopy(bases), current

        def solve(goals, seeds, check_fresh, *, report=None):
            order.append("native_IK")
            self.assertEqual(set(goals), {"ur5e"})
            self.assertEqual(set(seeds), {"ur5e"})
            self.assertIs(report, reporter)
            check_fresh()
            return {"ur5e": states["ur5e"].joints.copy()}

        def validate(path, check_fresh, *, report=None):
            order.append("native_path_FK")
            self.assertIs(report, reporter)
            check_fresh()
            if native_error:
                raise ControlFault("native path rejected")
            poses = [{"ur5e": states["ur5e"].tcp.copy()} for _ in path]
            return poses[:-1] if incomplete else poses

        def watchdog(check_fresh):
            order.append("watchdog")
            check_fresh()

        session.solve_joint_targets.side_effect = solve
        session.validate_joint_path.side_effect = validate
        session.enable_watchdog.side_effect = watchdog
        caught, returned = None, None
        with patch.dict(sys.modules, {"urcal.control.alignment": SimpleNamespace(LiveAlignment=factory)}):
            try:
                returned = startup.prepare_alignment(session, engine.args, engine.offsets, bases, states,
                                                     targets, read_current, reporter)
            except ControlFault as exc:
                caught = exc
        session.send.assert_not_called()
        session.send_joints.assert_not_called()
        return SimpleNamespace(session=session, alignment=alignment, returned=returned, caught=caught,
                               order=order, reporter=reporter)

    def test_native_ik_then_path_fk_then_watchdog_never_sends_velocity(self):
        result = self.prepare()
        self.assertIsNone(result.caught)
        self.assertIs(result.returned, result.alignment)
        self.assertLess(result.order.index("native_IK"), result.order.index("build_path"))
        self.assertLess(result.order.index("build_path"), result.order.index("native_path_FK"))
        self.assertLess(result.order.index("native_path_FK"), result.order.index("watchdog"))
        self.assertGreaterEqual(result.order.count("fresh"), 6)
        speed_reports = [call.args[0] for call in result.reporter.call_args_list
                         if "alignment speed" in call.args[0]]
        self.assertEqual(len(speed_reports), 1)
        self.assertIn("alignment speed x2", speed_reports[0])
        self.assertIn("command caps", speed_reports[0])
        # This synthetic scene uses 0.2 m/s and 0.5 rad/s global caps.
        self.assertIn("180mm/s", speed_reports[0])
        self.assertIn("25.78deg/s", speed_reports[0])
        self.assertIn("90% of hard limits; measured limits unchanged", speed_reports[0])
        self.assertIn("no motion yet", speed_reports[0])

    def test_incomplete_native_path_failure_or_moving_leader_prevents_watchdog(self):
        for option in ("incomplete", "changed_state", "native_error"):
            with self.subTest(option=option):
                result = self.prepare(**{option: True})
                self.assertIsInstance(result.caught, ControlFault)
                self.assertIsNone(result.returned)
                result.session.enable_watchdog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
