"""Virtual UR controller integration with entirely invented initial conditions.

The fixture uses the offline viewer's nominal joint poses and constructed
camera transforms, not measurements or evidence of real-robot performance.
No camera, robot connection, or trained checkpoint is used by these tests.
"""

import builtins
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from urcal.control.geometry import flange_from_tcp, inverse, model_from_controller_base
from urcal.control.runtime import parse_args
from urcal.simulation.plant import SimulationFault
from urcal.simulation.runtime import SimulationSession, run
from urcal.simulation.snapshot import SCHEMA, load_initial_state, save_initial_state
from urcal.tools.preview_gui import DemoState


ROBOTS = ("ur5e", "ur10e")


def synthetic_snapshot():
    """Nominal FK with invented, exactly facing base placement and TCP offsets."""
    demo = DemoState(now=0.)
    value = {
        "schema": SCHEMA, "captured_at_utc": "2026-10-18T00:00:00Z",
        "calibration": {
            "sequence": 10, "captured_at_monotonic": 123.,
            "poses": {name: base @ inverse(model_from_controller_base(name))
                      for name, base in demo.bases.items()},
            "raw_mean_px": {name: 2. for name in ROBOTS},
            "raw_max_px": {name: 3. for name in ROBOTS},
            "valid_counts": {name: 7 for name in ROBOTS},
        },
        "bases": copy.deepcopy(demo.bases),
        "robots": {name: {
            "q": q.copy(), "qd": np.full(6, .01), "speed": np.full(6, .001),
            "tcp": demo.models[name].pose_and_jacobian(q, flange=True)[0] @ demo.offsets[name],
            "tcp_offset": demo.offsets[name].copy(), "tcp_offset_source": "explicit",
            "robot_timestamp": 100.,
        } for name, q in demo.joints.items()},
    }
    return value, demo


def options(mode="follow", motion="none", *extra):
    return parse_args(["--simulate", "--mode", mode, "--initial_state", "synthetic.json",
                       "--sim_motion", motion, "--target_tolerance_m", "0", "--target_tolerance_deg", "0",
                       "--gap", ".30", "--max_travel", ".15", "--base_motion_mode", "velocity", *extra])


def line_target():
    return {"type": "target", "gap": .30, "path": "line", "path_delta": [.015, 0, 0, 0, 0, 0],
            "path_radius": .02, "path_duration": 5.}


def world_flanges(session):
    states = session.plant.states(session.now)
    return {name: session.plant.bases[name] @ flange_from_tcp(states[name].tcp, session.plant.offsets[name])
            for name in ROBOTS}


class SimulationSessionTests(unittest.TestCase):
    def setUp(self):
        self.initial, self.demo = synthetic_snapshot()

    def start_line(self, motion="none"):
        session = SimulationSession(options("follow", motion), self.initial)
        session.command(line_target())
        self.assertTrue(session.ready, session.message)
        session.command("arm")
        return session

    def assert_tick(self, session, count):
        for index in range(count):
            self.assertTrue(session.tick(), f"step {index}: {session.fault or session.message}")

    def endpoint_error(self, session):
        flanges = world_flanges(session)
        goal = flanges[session.engine.leader] @ session.engine.path.pose(1.)
        return float(np.linalg.norm(goal[:3, 3] - flanges[session.engine.follower][:3, 3]))

    def test_snapshot_q_tcp_offsets_and_camera_bases_are_used_without_captured_velocity(self):
        session = SimulationSession(options(), self.initial)
        states = session.plant.states(session.now)
        for name in ROBOTS:
            np.testing.assert_array_equal(states[name].joints, self.initial["robots"][name]["q"])
            np.testing.assert_allclose(states[name].tcp, self.initial["robots"][name]["tcp"], atol=1e-12)
            np.testing.assert_array_equal(states[name].joint_velocity, np.zeros(6))
            np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
        before = {name: state.joints.copy() for name, state in states.items()}
        self.assertFalse(session.tick())
        self.assertEqual(session.elapsed, 0.)
        for name in ROBOTS:
            np.testing.assert_array_equal(session.plant.joints[name], before[name])

    def test_saved_output_convention_overrides_cli_and_preserves_virtual_base_sensor(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                initial = copy.deepcopy(self.initial)
                if not legacy:
                    initial["calibration"]["ur10e_output_correction"] = False
                    initial["calibration"]["poses"] = copy.deepcopy(initial["bases"])
                args = options()
                args.ur10e_output_correction = not legacy
                session = SimulationSession(args, initial)
                self.assertEqual(session.args.ur10e_output_correction, legacy)
                self.assertEqual(args.ur10e_output_correction, not legacy)
                sensor = session._calibration()
                state = session.engine.state(session.now, active=False)
                for name in ROBOTS:
                    np.testing.assert_allclose(sensor[name], initial["calibration"]["poses"][name], atol=1e-12)
                    np.testing.assert_allclose(state.base_poses[name], initial["bases"][name], atol=1e-12)
                session.command(line_target())
                self.assertTrue(session.ready, session.message)
                session.command("arm")
                self.assert_tick(session, 3)

    def test_follow_line_reaches_endpoint_with_static_or_both_external_motions(self):
        for motion in ("none", "both"):
            with self.subTest(motion=motion):
                session = self.start_line(motion)
                before = self.endpoint_error(session)
                self.assert_tick(session, 150)  # Six virtual seconds, including endpoint settling.
                self.assertGreater(before, .014)
                self.assertLess(self.endpoint_error(session), .002)
                self.assertEqual(session.engine.progress, 1.)
                self.assertTrue(session.engine.finished)
                if motion == "none":
                    np.testing.assert_array_equal(session.plant.joints["ur10e"], self.initial["robots"]["ur10e"]["q"])
                else:
                    self.assertGreater(np.linalg.norm(session.plant.joints["ur10e"] - self.initial["robots"]["ur10e"]["q"]), .01)
                    self.assertGreater(np.linalg.norm(session.plant.bases["ur10e"] - self.initial["bases"]["ur10e"]), .005)

    def test_pause_freezes_time_motion_and_controller_history_then_resume_continues(self):
        session = self.start_line("both")
        self.assert_tick(session, 8)
        session.command("sim_pause")
        before = copy.deepcopy((session.plant.joints, session.plant.joint_velocity, session.plant.bases))
        progress, clock, previous = session.engine.progress, session.now, session.engine.previous_qdot.copy()
        for _ in range(3):
            self.assertFalse(session.tick())
            self.assertTrue(session.status(wall_now=9999.)["sim_paused"])
        self.assertEqual(session.now, clock)
        self.assertEqual(session.engine.progress, progress)
        np.testing.assert_array_equal(session.engine.previous_qdot, previous)
        for index, collection in enumerate((session.plant.joints, session.plant.joint_velocity, session.plant.bases)):
            for name in ROBOTS:
                np.testing.assert_array_equal(collection[name], before[index][name])
        session.command("sim_pause")
        self.assert_tick(session, 1)
        self.assertGreater(session.now, clock)
        self.assertGreater(session.engine.progress, progress)

    def test_reset_restores_original_plant_and_retains_applied_target(self):
        for mode in ("follow",):
            with self.subTest(mode=mode):
                session = SimulationSession(options(mode), self.initial)
                target = line_target()
                session.command(target)
                session.command("arm")
                self.assert_tick(session, 5)
                with self.assertRaisesRegex(ValueError, "Reset simulation"):
                    session.command(target)
                session.command("sim_reset")
                self.assertFalse(session.active)
                self.assertFalse(session.paused)
                self.assertEqual(session.elapsed, 0.)
                self.assertTrue(session.ready, session.message)
                for name in ROBOTS:
                    np.testing.assert_array_equal(session.plant.joints[name], self.initial["robots"][name]["q"])
                    np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
                    np.testing.assert_array_equal(session.plant.joint_velocity[name], np.zeros(6))
                if session.engine:
                    self.assertEqual(session.engine.path.kind, "line")
                    np.testing.assert_array_equal(session.engine.path.delta, target["path_delta"])
                    self.assertEqual(session.engine.progress, 0.)

    def test_rejected_plant_step_never_commits_task_or_advances_virtual_clock(self):
        session = self.start_line()
        self.assert_tick(session, 3)
        before = copy.deepcopy(session.plant.joints)
        progress, clock = session.engine.progress, session.now
        with mock.patch.object(session.plant, "step", side_effect=SimulationFault("test collision")), \
                mock.patch.object(session.engine, "commit", wraps=session.engine.commit) as commit:
            self.assertFalse(session.tick())
            commit.assert_not_called()
        self.assertEqual(session.engine.progress, progress)
        self.assertEqual(session.now, clock)
        self.assertFalse(session.active)
        self.assertIn("test collision", session.fault)
        for name in ROBOTS:
            np.testing.assert_array_equal(session.plant.joints[name], before[name])
            np.testing.assert_array_equal(session.plant.joint_velocity[name], np.zeros(6))
        with self.assertRaisesRegex(ValueError, "Reset simulation"):
            session.command("arm")
        session.command("sim_reset")
        self.assertTrue(session.ready)
        session.command("arm")
        self.assert_tick(session, 1)

    def test_target_failure_remains_visible_without_motion_and_can_be_replaced(self):
        session = SimulationSession(options(), self.initial)
        session.command({"type": "target", "gap": 1.30})
        self.assertFalse(session.ready)
        self.assertIn("WAIT", session.message)
        self.assertIn("goals", session.status()["scene"])
        with self.assertRaises(ValueError):
            session.command("arm")
        self.assertEqual(session.steps, 0)
        session.command({"type": "target", "gap": .30})
        self.assertTrue(session.ready, session.message)

    def test_status_preserves_capture_provenance_and_returns_independent_geometry(self):
        session = self.start_line()
        status = session.status(wall_now=5000.)
        self.assertEqual(status["scene"]["at"], 5000.)
        self.assertIn(self.initial["captured_at_utc"], status["scene"]["reason"])
        self.assertIn("Ideal virtual", status["scene"]["reason"])
        self.assertIn("SIM", status["message"])
        status["scene"]["bases"]["ur5e"][:] = 0.
        status["scene"]["robots"]["ur5e"]["q"][:] = 0.
        np.testing.assert_array_equal(session.plant.bases["ur5e"], self.initial["bases"]["ur5e"])
        np.testing.assert_array_equal(session.plant.joints["ur5e"], self.initial["robots"]["ur5e"]["q"])

    def test_wrong_tcp_offset_or_execution_flag_cannot_reinterpret_snapshot(self):
        args = options("follow", "none", "--ur5e_tcp_offset", "0", "0", "0", "0", "0", "0")
        with self.assertRaisesRegex(ValueError, "TCP offset differs"):
            SimulationSession(args, self.initial)
        args = options()
        args.execute = True
        with self.assertRaisesRegex(ValueError, "without --execute"):
            SimulationSession(args, self.initial)

    def test_removed_fixed_target_requests_are_rejected_without_mutating_follow_task(self):
        session = SimulationSession(options(), self.initial)
        before = session.engine.path.pose(0.).copy()
        with self.assertRaisesRegex(ValueError, "fixed camera-center"):
            session.command({"type": "target", "gap": .35, "center_pose": [0, 0, 1, 0, 0, 0]})
        np.testing.assert_array_equal(session.engine.path.pose(0.), before)
        self.assertFalse(session.active)
        self.assertEqual(session.steps, 0)
        args = options()
        args.mode = "hold"
        with self.assertRaisesRegex(ValueError, "only follower"):
            SimulationSession(args, self.initial)

    def test_demo_rejects_removed_fixed_target_mode_and_commands(self):
        with self.assertRaisesRegex(ValueError, "only follower"):
            DemoState("hold")
        before = copy.deepcopy(self.demo.request)
        with self.assertRaisesRegex(ValueError, "fixed camera-center"):
            self.demo.apply({"type": "target", "gap": .35, "center_pose": [0, 0, 1, 0, 0, 0]})
        self.assertEqual(self.demo.request, before)
        self.assertNotIn("center_pose", self.demo.status())


class OfflineSimulationEntryTests(unittest.TestCase):
    def test_saved_state_run_skips_capture_and_sdk_imports_and_closes_gui(self):
        initial, _ = synthetic_snapshot()
        original_import = builtins.__import__
        forbidden = {"rtde_control", "rtde_receive", "pyrealsense2"}

        def no_hardware_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError("offline run attempted a hardware SDK import: " + name)
            return original_import(name, *args, **kwargs)

        class Panel:
            instances = []

            def __init__(self, args, command, stop):
                self.root, self.stop, self.closed = object(), stop, False
                self._view = mock.Mock()
                self.command = command
                self.instances.append(self)

            def poll(self, status):
                self.status = status
                self.stop.set()

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_initial_state(root / "captured.json", initial)
            args = parse_args(["--simulate", "--mode", "follow", "--initial_state", str(path)])
            with mock.patch("urcal.simulation.runtime.BUNDLE", root), \
                    mock.patch("urcal.simulation.runtime.capture_initial_state", side_effect=AssertionError("capture reached")) as capture, \
                    mock.patch("urcal.control.gui.TargetPanel", Panel), \
                    mock.patch("tkinter.ttk.Label"), \
                    mock.patch("builtins.__import__", side_effect=no_hardware_import), \
                    mock.patch("builtins.print"):
                run(args)
            capture.assert_not_called()
            self.assertEqual(len(Panel.instances), 1)
            self.assertTrue(Panel.instances[0].closed)
            self.assertIn("SIM", Panel.instances[0].status["message"])
            snapshots = list(root.glob("runs/simulation/*/initial_state.json"))
            logs = list(root.glob("runs/simulation/*/simulation.jsonl"))
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(len(logs), 1)
            saved = load_initial_state(snapshots[0])
            self.assertEqual(saved["captured_at_utc"], initial["captured_at_utc"])
            self.assertIn('"hardware_commands":false', logs[0].read_text())
            for name in ROBOTS:
                np.testing.assert_array_equal(saved["robots"][name]["q"], initial["robots"][name]["q"])


if __name__ == "__main__":
    unittest.main()
