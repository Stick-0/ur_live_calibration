"""Gap approach in the real virtual-joint session, without hardware or weights.

The initial placement is invented nominal FK, not evidence of physical robot
clearance or final tracking accuracy. The session keeps its real workspace
checks and commits a reference step only after the virtual plant accepts it.
"""

import builtins
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base
from urcal.simulation.plant import SimulationFault
from urcal.simulation.runtime import SimulationSession
from test_simulation_runtime import options, synthetic_snapshot, world_flanges


def nonsingular_zero_tcp_snapshot():
    """Invented facing placement with room for a 15 cm axial translation.

    The screen demo's shallow elbow configuration approaches a singularity
    after a few centimetres. This independent bent-elbow fixture keeps the
    original camera flange poses while constructing a different nominal base
    placement. It uses zero TCP offsets, as in the user's no-tool setup.
    """
    initial, demo = synthetic_snapshot()
    follower = initial["robots"]["ur5e"]
    camera_flange = initial["bases"]["ur5e"] @ follower["tcp"] @ inverse(follower["tcp_offset"])
    for row in initial["robots"].values():
        row["tcp"] = row["tcp"] @ inverse(row["tcp_offset"])
        row["tcp_offset"] = np.eye(4)
    follower["q"] = np.array([-.6, -1.2, 1.8, -1.4, 1.57, 0.])
    follower["tcp"] = demo.models["ur5e"].pose_and_jacobian(follower["q"], flange=True)[0]
    initial["bases"]["ur5e"] = camera_flange @ inverse(follower["tcp"])
    initial["calibration"]["poses"]["ur5e"] = (
        initial["bases"]["ur5e"] @ inverse(model_from_controller_base("ur5e")))
    return initial


class SimulationGapApproachTests(unittest.TestCase):
    def setUp(self):
        self.initial, _ = synthetic_snapshot()
        self.records = []

    def session(self, motion="none", *, arm=True):
        args = options("follow", motion, "--base_motion_mode", "pose", "--max_travel", ".3",
                       "--approach_speed", ".02")
        logger = SimpleNamespace(log=lambda value: self.records.append(copy.deepcopy(value)))
        session = SimulationSession(args, self.initial, logger=logger)
        self.assertTrue(session.ready, session.message)
        self.assertIsNone(session.approach)
        if arm:
            session.command("arm")
            self.ticks(session, 1)  # The first accepted command fixes the following reference.
        return session

    def request(self, session, action, **extra):
        session.command({"type": "gap_approach", "action": action, **extra})

    def ticks(self, session, count):
        for index in range(count):
            self.assertTrue(session.tick(), f"step {index}: {session.fault or session.message}")

    def test_start_requires_tracking_and_does_not_become_a_target_edit(self):
        session = self.session(arm=False)
        path = session.engine.path
        self.assertFalse(session.status()["gap_approach"]["enabled"])
        with self.assertRaises(ValueError):
            self.request(session, "start")
        self.assertIs(session.engine.path, path)
        self.assertIsNone(session._target_command)
        self.assertEqual(session.elapsed, 0.)
        self.assertEqual(session.fault, "")
        session.command("arm")
        self.assertFalse(session.status()["gap_approach"]["enabled"])
        self.ticks(session, 1)
        self.request(session, "start", requested_at=987654321.)
        self.assertEqual(session.status()["gap_approach"]["state"], "running")
        self.assertNotIn("error", session.status()["gap_approach"])
        self.assertIsNone(session._target_command)

    def test_initial_alignment_and_frozen_simulation_reject_approach_without_advancing(self):
        session = self.session()
        elapsed = session.elapsed
        # Phase-boundary check only; no fake alignment trajectory is executed.
        session.approach = SimpleNamespace(done=False)
        with self.assertRaises(ValueError):
            self.request(session, "start")
        self.assertEqual(session.engine.gap_approach_status(enabled=False)["state"], "idle")
        session.approach = None
        session.command("sim_pause")
        with self.assertRaises(ValueError):
            self.request(session, "start")
        self.assertEqual(session.elapsed, elapsed)
        self.assertFalse(session.status()["gap_approach"]["enabled"])
        session.command("sim_pause")
        self.request(session, "start")
        self.assertEqual(session.status()["gap_approach"]["state"], "running")

    def test_bad_timestamp_and_unknown_action_are_nonfatal_and_leave_reference_unchanged(self):
        session = self.session()
        elapsed = session.elapsed
        initial_goal = session.engine.path.pose(session.engine.progress)
        for stamp in (float("nan"), float("inf"), None, "not-a-clock"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                self.request(session, "start", requested_at=stamp)
        with self.assertRaises(ValueError):
            self.request(session, "invalid")
        self.assertTrue(session.active)
        self.assertEqual(session.fault, "")
        self.assertEqual(session.elapsed, elapsed)
        np.testing.assert_array_equal(session.engine.path.pose(session.engine.progress), initial_goal)
        self.assertTrue(all(not row["accepted"] for row in self.records
                            if row["kind"] == "simulation_gap_approach"))

    def test_pause_keeps_relative_reference_and_leader_following_then_resume_advances(self):
        session = self.session("leader")
        self.request(session, "start")
        self.ticks(session, 20)
        self.request(session, "pause")
        before = session.status()["gap_approach"]
        progress, clock = session.engine.progress, session.now
        follower = session.plant.joints[session.engine.follower].copy()
        leader = session.plant.joints[session.engine.leader].copy()
        self.ticks(session, 45)
        paused = session.status()["gap_approach"]
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(session.engine.progress, progress)
        self.assertEqual(paused["commanded_gap_m"], before["commanded_gap_m"])
        self.assertGreater(session.now, clock)
        self.assertGreater(np.linalg.norm(session.plant.joints[session.engine.leader] - leader), 1e-4)
        self.assertGreater(np.linalg.norm(session.plant.joints[session.engine.follower] - follower), 1e-4)
        flanges = world_flanges(session)
        target = flanges[session.engine.leader] @ session.engine.path.pose(session.engine.progress)
        self.assertLess(np.linalg.norm(target[:3, 3] - flanges[session.engine.follower][:3, 3]), .01)
        self.request(session, "resume")
        self.ticks(session, 10)
        resumed = session.status()["gap_approach"]
        self.assertEqual(resumed["state"], "running")
        self.assertGreater(session.engine.progress, progress)
        self.assertLess(resumed["commanded_gap_m"], before["commanded_gap_m"])

    def test_whole_simulation_pause_freezes_gap_clock_joints_and_history(self):
        session = self.session("both")
        self.request(session, "start")
        self.ticks(session, 12)
        session.command("sim_pause")
        before = copy.deepcopy((session.plant.joints, session.plant.joint_velocity, session.plant.bases))
        clock, progress = session.now, session.engine.progress
        previous = session.engine.previous_qdot.copy()
        for wall_time in (100000., 200000., 400000.):
            self.assertFalse(session.tick())
            status = session.status(wall_now=wall_time)
            self.assertTrue(status["sim_paused"])
            self.assertFalse(status["gap_approach"]["enabled"])
            self.assertEqual(status["gap_approach"]["state"], "running")
        self.assertEqual(session.now, clock)
        self.assertEqual(session.engine.progress, progress)
        np.testing.assert_array_equal(session.engine.previous_qdot, previous)
        for index, values in enumerate((session.plant.joints, session.plant.joint_velocity, session.plant.bases)):
            for name in values:
                np.testing.assert_array_equal(values[name], before[index][name])
        session.command("sim_pause")
        self.ticks(session, 1)
        self.assertGreater(session.engine.progress, progress)

    def test_existing_task_pause_buttons_keep_approach_status_in_sync(self):
        session = self.session("leader")
        self.request(session, "start")
        self.ticks(session, 8)
        session.command("pause_task")
        self.assertTrue(session.engine.task_paused)
        self.assertEqual(session.status()["gap_approach"]["state"], "paused")
        progress = session.engine.progress
        self.ticks(session, 5)
        self.assertEqual(session.engine.progress, progress)
        session.command("resume_task")
        self.assertFalse(session.engine.task_paused)
        self.assertEqual(session.status()["gap_approach"]["state"], "running")
        self.ticks(session, 2)
        self.assertGreater(session.engine.progress, progress)

    def test_rejected_plant_step_never_advances_reference_and_latches_approach_pause(self):
        session = self.session()
        self.request(session, "start")
        self.ticks(session, 8)
        progress, clock = session.engine.progress, session.now
        with mock.patch.object(session.plant, "step", side_effect=SimulationFault("invented obstruction")), \
                mock.patch.object(session.engine, "commit", wraps=session.engine.commit) as commit:
            self.assertFalse(session.tick())
            commit.assert_not_called()
        self.assertEqual(session.now, clock)
        self.assertEqual(session.engine.progress, progress)
        status = session.status()
        self.assertEqual(status["gap_approach"]["state"], "paused")
        self.assertFalse(status["gap_approach"]["enabled"])
        with self.assertRaisesRegex(ValueError, "Reset simulation"):
            session.command("arm")
        with self.assertRaises(ValueError):
            self.request(session, "resume")

    def test_reset_restores_capture_and_discards_unrequested_approach_without_hardware(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("rtde_control", "rtde_receive", "pyrealsense2"):
                raise AssertionError("simulation imported hardware: " + name)
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            session = self.session()
            self.request(session, "start")
            self.ticks(session, 10)
            session.command("sim_reset")
            status = session.status()
            self.assertEqual(status["gap_approach"]["state"], "idle")
            self.assertFalse(status["gap_approach"]["enabled"])
            self.assertEqual(status["gap_approach"]["commanded_gap_m"], .30)
            self.assertEqual(session.engine.path.kind, "hold")
            self.assertEqual(session.elapsed, 0.)
            self.assertFalse(session.active)
            self.assertEqual(session.fault, "")
            for name in self.initial["robots"]:
                np.testing.assert_array_equal(session.plant.joints[name], self.initial["robots"][name]["q"])
                np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
            session.command("arm")
            self.assertEqual(session.status()["gap_approach"]["state"], "idle")
            self.assertTrue(all(row["hardware_commands"] is False for row in self.records))

    def test_status_returns_reference_path_start_and_independent_geometry(self):
        session = self.session()
        self.request(session, "start")
        self.ticks(session, 5)
        status = session.status()
        np.testing.assert_array_equal(status["scene"]["task"]["start_pose"], session.engine.path.start)
        self.assertIsNotNone(status["gap_approach"]["estimated_gap_m"])
        start = session.engine.path.start
        status["scene"]["task"]["start_pose"][:] = 0.
        np.testing.assert_array_equal(session.engine.path.start, start)

    def test_default_speed_completes_full_approach_and_continues_following_at_endpoint(self):
        args = options("follow", "none", "--base_motion_mode", "pose", "--max_travel", ".3",
                       "--target_tolerance_m", ".05", "--target_tolerance_deg", "10")
        self.assertEqual(args.approach_speed, .005)
        session = SimulationSession(args, nonsingular_zero_tcp_snapshot())
        self.assertTrue(session.ready, session.message)
        self.assertIsNone(session.approach)
        session.command("arm")
        self.ticks(session, 1)
        self.request(session, "start")
        budget = int(np.ceil(session.engine.path.duration / session.dt)) + 200
        before_gap = session.status()["gap_approach"]["commanded_gap_m"]
        for index in range(budget):
            self.assertTrue(session.tick(), f"step {index}: {session.fault or session.message}")
            status = session.status()["gap_approach"]
            gap = status["commanded_gap_m"]
            self.assertGreaterEqual(gap, .15)
            self.assertLessEqual(gap, before_gap + 1e-12)
            self.assertLessEqual(before_gap - gap, args.approach_speed * session.dt + 1e-12)
            before_gap = gap
            if status["state"] == "complete":
                break
        else:
            self.fail("approach reference did not complete within its virtual-time budget")
        self.assertEqual(status["commanded_gap_m"], .15)
        self.assertEqual(session.engine.progress, 1.)
        self.assertTrue(session.active)
        self.assertFalse(status["enabled"])
        # Reaching the reference endpoint does not terminate normal following.
        self.ticks(session, 25)
        session.args.sim_motion = "leader"
        session._control_origin = session.elapsed
        leader_before = session.plant.joints[session.engine.leader].copy()
        follower_before = session.plant.joints[session.engine.follower].copy()
        self.ticks(session, 50)
        final = session.status()["gap_approach"]
        self.assertEqual(final["state"], "complete")
        self.assertEqual(final["commanded_gap_m"], .15)
        self.assertGreater(np.linalg.norm(session.plant.joints[session.engine.leader] - leader_before), 1e-4)
        self.assertGreater(np.linalg.norm(session.plant.joints[session.engine.follower] - follower_before), 1e-4)
        self.assertAlmostEqual(final["estimated_gap_m"], .15, delta=.002)
        self.assertTrue(session.active, session.fault)


if __name__ == "__main__":
    unittest.main()
