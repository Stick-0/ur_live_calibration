"""Evaluation telemetry follows virtual control cycles without changing motion."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from urcal.control.evaluation import EvaluationRecorder
from urcal.control.geometry import calibration_output_mapping, inverse
from urcal.cooperative.base_plane import project_output_bases
from urcal.cooperative.logging import ExperimentLogger
from urcal.simulation.plant import SimulationFault
from urcal.simulation.runtime import SimulationSession, run
from urcal.simulation.snapshot import save_initial_state
from test_simulation_runtime import line_target, options, synthetic_snapshot


class EvaluationSpy:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.requests, self.records, self.events = [], [], []
        self.active, self.closed = False, False
        self.target = None
        self.instances.append(self)

    def request(self, command, **metadata):
        entry = copy.deepcopy({"command": command, **metadata})
        self.requests.append(entry)
        self.events.append(("request", entry))
        if command["action"] == "start":
            self.active = True
            self.target = copy.deepcopy(metadata["target"])
        elif command["action"] == "stop":
            self.active = False
        return True

    def record(self, **metadata):
        if self.active:
            entry = copy.deepcopy(metadata)
            self.records.append(entry)
            self.events.append(("record", entry))

    def status(self):
        return {"state": "recording" if self.active else "saved", "rows": len(self.records)}

    def close(self, reason="shutdown"):
        self.closed = True
        self.close_reason = reason
        self.active = False


class EvaluationSimulationTests(unittest.TestCase):
    def session(self, *, line=True, motion="none", initial=None):
        if initial is None:
            initial, _ = synthetic_snapshot()
        session = SimulationSession(options("follow", motion), initial)
        if line:
            session.command(line_target())
        self.assertTrue(session.ready, session.message)
        recorder = EvaluationSpy()
        session.evaluation_recorder = recorder
        self.assertTrue(session.evaluation_request(
            {"type": "evaluation", "action": "start", "request_id": "evaluation-start"}))
        return session, recorder

    def test_each_tick_pairs_pre_step_state_with_only_the_applied_simulated_command(self):
        session, recorder = self.session()
        target = recorder.target.copy()
        context = recorder.requests[-1]["command"]["runtime_context"]
        self.assertEqual(context["path"]["kind"], "line")
        self.assertEqual(context["progress"], 0.)
        np.testing.assert_array_equal(context["path"]["start"], target)
        self.assertFalse(session.active)  # Evaluation start cannot arm motion.
        session.command("arm")
        for _ in range(3):
            at = session.now
            before = session.plant.states(at)
            measurement_sequence = session.sequence
            self.assertTrue(session.tick(), session.fault)
            row = recorder.records[-1]
            self.assertEqual(row["at"], at)
            self.assertEqual(row["scene"]["at"], at)
            self.assertEqual(row["control_record"]["at"], at)
            self.assertEqual(row["measurement"]["sequence"], measurement_sequence)
            self.assertEqual(row["measurement"]["captured_at"], at)
            self.assertEqual(row["phase"], "tracking")
            self.assertEqual(row["episode"], session.episode)
            self.assertTrue(row["transport"]["simulated"])
            self.assertTrue(row["transport"]["simulated_command_applied"])
            self.assertFalse(row["transport"]["hardware_commands"])
            self.assertFalse(row["control_record"]["command_sent"])
            self.assertTrue(row["control_record"]["simulated_command_applied"])
            for name, state in before.items():
                np.testing.assert_array_equal(row["scene"]["robots"][name]["q"], state.joints)
                np.testing.assert_array_equal(row["scene"]["robots"][name]["qd"], state.joint_velocity)
                np.testing.assert_array_equal(row["scene"]["robots"][name]["tcp_speed"], state.speed)
            follower = session.engine.follower
            command = row["transport"]["commands"][follower]
            np.testing.assert_array_equal(command["joint_velocity"], session.plant.joint_velocity[follower])
            np.testing.assert_array_equal(row["scene"]["bases"][follower], row["control_record"]["bases"][follower])
        np.testing.assert_array_equal(recorder.target, target)
        self.assertEqual(len(recorder.requests), 1)

    def test_evaluation_does_not_change_control_trajectory_or_target(self):
        recorded, recorder = self.session(motion="both")
        initial, _ = synthetic_snapshot()
        baseline = SimulationSession(options("follow", "both"), initial)
        baseline.command(line_target())
        for session in (recorded, baseline):
            session.command("arm")
        for _ in range(12):
            self.assertTrue(recorded.tick(), recorded.fault)
            self.assertTrue(baseline.tick(), baseline.fault)
            for name in ("ur5e", "ur10e"):
                np.testing.assert_array_equal(recorded.plant.joints[name], baseline.plant.joints[name])
                np.testing.assert_array_equal(recorded.plant.bases[name], baseline.plant.bases[name])
            self.assertEqual(recorded.engine.progress, baseline.engine.progress)
            np.testing.assert_array_equal(recorded.engine.path.start, baseline.engine.path.start)
        self.assertEqual(len(recorder.records), 12)

    def test_pause_preserves_duplicate_virtual_times_and_has_no_previous_command(self):
        session, recorder = self.session()
        session.command("arm")
        self.assertTrue(session.tick(), session.fault)
        session.command("sim_pause")
        at, progress = session.now, session.engine.progress
        for _ in range(2):
            self.assertFalse(session.tick())
        for row in recorder.records[-2:]:
            self.assertEqual(row["at"], at)
            self.assertEqual(row["phase"], "paused")
            self.assertIn("paused", row["reason"])
            self.assertIsNone(row["control_record"])
            self.assertIsNone(row["transport"])
        self.assertEqual(session.now, at)
        self.assertEqual(session.engine.progress, progress)

    def test_rejected_plant_step_keeps_candidate_separate_from_applied_command(self):
        session, recorder = self.session()
        session.command("arm")
        self.assertTrue(session.tick(), session.fault)
        at, progress = session.now, session.engine.progress
        with mock.patch.object(session.plant, "step", side_effect=SimulationFault("blocked sample")):
            self.assertFalse(session.tick())
        row = recorder.records[-1]
        self.assertEqual(row["phase"], "fault")
        self.assertEqual(row["reason"], "blocked sample")
        self.assertFalse(row["transport"]["simulated_command_applied"])
        self.assertFalse(row["control_record"]["command_sent"])
        self.assertFalse(row["control_record"]["simulated_command_applied"])
        self.assertEqual(session.now, at)
        self.assertEqual(session.engine.progress, progress)
        self.assertFalse(session.tick())
        self.assertIsNone(recorder.records[-1]["control_record"])
        self.assertIsNone(recorder.records[-1]["transport"])
        self.assertEqual(recorder.records[-1]["reason"], "blocked sample")

    def test_reset_ends_evaluation_before_clock_rewind_and_next_start_is_new_episode(self):
        session, recorder = self.session()
        session.command("arm")
        self.assertTrue(session.tick(), session.fault)
        at, episode = session.now, session.episode
        old_target = recorder.target.copy()
        session.command("sim_reset")
        self.assertEqual(recorder.events[-2][0], "record")
        self.assertEqual(recorder.events[-2][1]["phase"], "reset")
        stop = recorder.requests[-1]
        self.assertEqual(stop["command"]["action"], "stop")
        self.assertEqual(stop["command"]["reason"], "simulation_reset")
        self.assertEqual(stop["now"], at)
        self.assertEqual(stop["episode"], episode)
        self.assertFalse(recorder.active)
        self.assertEqual(session.episode, episode + 1)
        self.assertLess(session.now, at)
        np.testing.assert_array_equal(recorder.target, old_target)
        session.evaluation_request({"type": "evaluation", "action": "start", "request_id": "next-evaluation"})
        self.assertEqual(recorder.requests[-1]["episode"], episode + 1)

    def test_alignment_records_joint_command_as_simulated_and_no_hardware_send(self):
        initial, demo = synthetic_snapshot()
        row = initial["robots"]["ur10e"]
        row["q"][5] += 2.
        row["tcp"] = demo.models["ur10e"].pose_and_jacobian(row["q"], flange=True)[0] @ row["tcp_offset"]
        session, recorder = self.session(line=False, initial=initial)
        self.assertIsNotNone(session.approach)
        session.command("arm")
        at = session.now
        self.assertTrue(session.tick(), session.fault)
        row = recorder.records[-1]
        self.assertEqual(row["phase"], "alignment")
        self.assertEqual(row["at"], at)
        self.assertTrue(row["control_record"]["active"])
        self.assertFalse(row["control_record"]["command_sent"])
        self.assertTrue(row["transport"]["simulated_command_applied"])
        command = row["transport"]["commands"][session.engine.follower]
        self.assertEqual(command["method"], "speedJ")
        np.testing.assert_array_equal(command["command"], session.plant.joint_velocity[session.engine.follower])

    def test_real_export_preserves_pauses_invalid_samples_and_reset_boundary(self):
        initial, _ = synthetic_snapshot()
        session = SimulationSession(options("follow", "both"), initial)
        session.command(line_target())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            with ExperimentLogger(raw) as logger:
                recorder = EvaluationRecorder(logger, raw, root / "evaluations", settings=vars(session.args),
                                              leader=session.engine.leader, follower=session.engine.follower,
                                              clock="simulation")
                session.evaluation_recorder = recorder
                self.assertTrue(session.evaluation_request(
                    {"type": "evaluation", "action": "start", "request_id": "start"}))
                session.command("arm")
                for _ in range(2):
                    self.assertTrue(session.tick(), session.fault)
                session.command("sim_pause")
                self.assertFalse(session.tick())
                # Inject stale accepted calibration to exercise explicit invalid
                # rows while virtual time stays frozen; recording must not repair it.
                session.engine.gate.last_capture -= session.args.max_calibration_age + .01
                self.assertFalse(session.tick())
                session.command("sim_reset")
                recorder.close()
                status = recorder.status()
                self.assertEqual(status["state"], "saved", status)
            events = [json.loads(line) for line in raw.read_text().splitlines()]
            cycles = [event for event in events if event["kind"] == "evaluation_cycle"]
            self.assertEqual([event["phase"] for event in cycles],
                             ["tracking", "tracking", "paused", "paused", "reset"])
            self.assertEqual([event["metric_valid"] for event in cycles], [True, True, True, False, False])
            self.assertEqual(events[-1]["reason"], "simulation_reset")
            self.assertTrue(all(event["episode"] == 0 for event in events))
            self.assertTrue(all(event["frozen_target"] == events[0]["frozen_target"] for event in cycles))
            self.assertIsNone(cycles[-1]["position_error_norm_m"])
            self.assertEqual(len({event["at"] for event in cycles[-3:]}), 1)
            with Path(status["csv_path"]).open(newline="") as stream:
                exported = [row for row in csv.DictReader(stream) if row["kind"] == "evaluation_cycle"]
            self.assertEqual(len(exported), len(cycles))
            self.assertEqual([row["metric_valid"] for row in exported], ["1", "1", "1", "0", "0"])
            self.assertTrue(all(row["simulated"] == "1" and row["command_sent"] == "0" for row in exported))
            self.assertTrue(all(row["final_sent_velocity_0"] == "" for row in exported))
            self.assertEqual(exported[0]["simulated_applied_method"], "speedL")
            self.assertEqual(exported[0]["simulated_integrator_method"], "joint_velocity_step")
            self.assertNotEqual(exported[0]["simulated_applied_joint_velocity_0"], "")
            self.assertEqual(exported[-1]["position_error_norm_m"], "")
            self.assertEqual(exported[-1]["simulated_applied_velocity_0"], "")
            self.assertEqual(json.loads(exported[-1]["raw_record_json"]), cycles[-1])

    def test_simulation_ui_routes_evaluation_without_arming_and_keeps_position_csv_separate(self):
        initial, _ = synthetic_snapshot()
        initial["calibration"]["common_base_plane"] = True
        mappings = {name: calibration_output_mapping(name) for name in initial["bases"]}
        initial["bases"] = project_output_bases(initial["bases"], mappings)
        initial["calibration"]["poses"] = {
            name: base @ inverse(mappings[name]) for name, base in initial["bases"].items()}
        EvaluationSpy.instances = []
        error_recorder = mock.Mock()
        error_recorder.status.return_value = {"state": "idle"}

        class Panel:
            def __init__(self, args, command, stop, *, recording_command=None):
                self.root, self._view, self.stop = object(), mock.Mock(), stop
                self.callback, self.polls = recording_command, 0
                self.callback({"type": "evaluation", "action": "start", "request_id": "start"})
                self.callback({"action": "start", "request_id": "position-start"})

            def poll(self, status):
                self.polls += 1
                self.active = status["active"]
                if self.active:
                    raise AssertionError("recording armed the virtual robot")
                if "evaluation_recording" not in status:
                    raise AssertionError("evaluation status was not forwarded")
                if self.polls == 1:
                    self.callback({"type": "evaluation", "action": "table_start", "request_id": "table-start"})
                elif self.polls == 2:
                    self.callback({"type": "evaluation", "action": "table_stop", "request_id": "table-stop"})
                    self.callback({"type": "evaluation", "action": "stop", "request_id": "stop"})
                    self.stop.set()

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_initial_state(root / "initial.json", initial)
            args = options("follow", "none", "--base_motion_mode", "relative_kalman", "--common_base_plane")
            args.initial_state = path
            with mock.patch("urcal.simulation.runtime.BUNDLE", root), \
                    mock.patch("urcal.simulation.runtime.EvaluationRecorder", EvaluationSpy), \
                    mock.patch("urcal.simulation.runtime.PositionErrorRecorder", return_value=error_recorder), \
                    mock.patch("urcal.control.gui.TargetPanel", Panel), \
                    mock.patch("tkinter.ttk.Label"), mock.patch("builtins.print"):
                run(args)
        recorder, = EvaluationSpy.instances
        self.assertEqual([request["command"]["action"] for request in recorder.requests],
                         ["start", "table_start", "table_stop", "stop"])
        self.assertTrue(all(request["now"] == 1000. for request in recorder.requests))
        self.assertEqual(recorder.kwargs["clock"], "simulation")
        self.assertFalse(recorder.kwargs["settings"]["execute"])
        self.assertIsInstance(recorder.kwargs["settings"]["relative_kalman_parameters"], dict)
        self.assertIn("prediction_horizon", recorder.kwargs["settings"]["relative_kalman_parameters"])
        self.assertTrue(recorder.closed)
        error_recorder.request.assert_called_once_with({"action": "start", "request_id": "position-start"})


if __name__ == "__main__":
    unittest.main()
