"""Initial-alignment integration using invented nominal UR poses only.

Rotating the fixture's UR10e last joint by two radians changes its flange roll
while preserving flange position/+Z. No captured user data or robot IO is used.
"""

import copy
import unittest
from unittest import mock

import numpy as np

from urcal.control.geometry import pose_delta
from urcal.simulation.plant import SimulationFault
from urcal.simulation.runtime import SimulationSession
from test_simulation_runtime import options, synthetic_snapshot, world_flanges


ROBOTS = ("ur5e", "ur10e")


class SimulationAlignmentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial, demo = synthetic_snapshot()
        row = cls.initial["robots"]["ur10e"]
        row["q"][5] += 2.
        row["tcp"] = demo.models["ur10e"].pose_and_jacobian(row["q"], flange=True)[0] @ row["tcp_offset"]
        # Reuse independently copied, unstarted plans. This avoids making every
        # pause/error test repeat the deterministic initial-path search.
        cls.planned = {}
        for mode in ("follow",):
            args = options(mode)
            if mode == "follow":
                args.path = "line"
                args.path_delta = [.015, 0, 0, 0, 0, 0]
                args.path_duration = 5.
            session = SimulationSession(args, cls.initial)
            cls.planned[mode] = session

    def session(self, mode, motion="none"):
        session = copy.deepcopy(self.planned[mode])
        session.args.sim_motion = motion
        self.assertIsNotNone(session.approach, session.message)
        self.assertTrue(session.ready, session.message)
        self.assertFalse(session.approach.done)
        self.assertEqual(session.args.max_target_deg, 25.)
        self.assertFalse(session.args.execute)
        return session

    def assert_tick(self, session):
        self.assertTrue(session.tick(), session.fault or session.message)

    def test_large_roll_aligns_without_teleport_then_hands_over_to_follow(self):
        for mode, motion in (("follow", "both"),):
            with self.subTest(mode=mode):
                session = self.session(mode, motion)
                goals = copy.deepcopy(session.status()["scene"]["goals"])
                current = world_flanges(session)
                self.assertGreater(max(np.linalg.norm(pose_delta(current[name], goal)[1])
                                       for name, goal in goals.items()), np.deg2rad(90.))
                for name in ROBOTS:
                    np.testing.assert_array_equal(session.plant.joints[name], self.initial["robots"][name]["q"])
                self.assertEqual(session.elapsed, 0.)
                self.assertFalse(session.tick())
                session.command("arm")
                self.assertIn("alignment", session.message.lower())
                first = True
                budget = int(np.ceil(session.approach.duration / session.dt)) + 10
                while not session.approach.done and budget:
                    before = copy.deepcopy(session.plant.joints)
                    self.assert_tick(session)
                    for name in ROBOTS:
                        np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
                        self.assertLessEqual(np.max(np.abs(session.plant.joints[name] - before[name])),
                                             session.args.max_joint_speed * session.dt + 1e-10)
                    if session.engine is not None:
                        self.assertEqual(session.engine.progress, 0.)
                        np.testing.assert_array_equal(session.plant.joints[session.engine.leader],
                                                      self.initial["robots"][session.engine.leader]["q"])
                    if first:
                        self.assertFalse(session.approach.done)
                        first = False
                    budget -= 1
                self.assertTrue(session.approach.done, "alignment did not complete within its planned duration")
                self.assertTrue(session.active, session.message)
                self.assertEqual(session.args.max_target_deg, 25.)
                self.assertEqual(session._control_origin, session.elapsed)
                self.assertIn("Alignment complete", session.message)
                reached = world_flanges(session)
                for name, goal in goals.items():
                    dp, dr = pose_delta(reached[name], goal)
                    self.assertLess(np.linalg.norm(dp), .0006)
                    self.assertLess(np.linalg.norm(dr), .0021)
                for name in ROBOTS:
                    np.testing.assert_allclose(session.plant.joint_velocity[name], np.zeros(6), atol=1e-12)
                displayed = session.status()["scene"]["goals"]
                for name in goals:
                    np.testing.assert_allclose(displayed[name], goals[name], atol=1e-12)
                # External sine motions restart at phase zero after alignment,
                # rather than jumping ahead by the time spent approaching.
                next_bases, next_velocity = session._external_motion()
                phase = .5 * (1 - np.cos(2 * np.pi / 20. * session.dt))
                for name, sign in (("ur5e", 1.), ("ur10e", -1.)):
                    np.testing.assert_allclose(next_bases[name][:3, 3] - self.initial["bases"][name][:3, 3],
                                               [0., sign * .01 * phase, 0.], atol=1e-12)
                if session.engine is not None:
                    np.testing.assert_allclose(next_velocity,
                                               np.array([0., .025, 0., .015, 0., 0.]) * phase / session.dt,
                                               atol=1e-12)
                self.assert_tick(session)
                if session.engine is not None:
                    self.assertGreater(session.engine.progress, 0.)

    def test_alignment_pauses_without_advancing_and_reset_restores_capture_and_target(self):
        session = self.session("follow")
        desired = copy.deepcopy(session.status()["scene"]["goals"])
        session.command("arm")
        for _ in range(4):
            self.assert_tick(session)
        session.command("sim_pause")
        before = copy.deepcopy((session.plant.joints, session.plant.joint_velocity, session.plant.bases))
        clock, progress = session.now, session.approach.progress
        for _ in range(3):
            self.assertFalse(session.tick())
        self.assertEqual(session.now, clock)
        self.assertEqual(session.approach.progress, progress)
        for index, values in enumerate((session.plant.joints, session.plant.joint_velocity, session.plant.bases)):
            for name in ROBOTS:
                np.testing.assert_array_equal(values[name], before[index][name])
        session.command("sim_pause")
        self.assert_tick(session)
        self.assertGreater(session.approach.progress, progress)
        session.command("sim_reset")
        self.assertFalse(session.active)
        self.assertFalse(session.paused)
        self.assertEqual(session.elapsed, 0.)
        self.assertEqual(session.approach.progress, 0.)
        self.assertTrue(session.ready, session.message)
        for name in ROBOTS:
            np.testing.assert_array_equal(session.plant.joints[name], self.initial["robots"][name]["q"])
            np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
            np.testing.assert_array_equal(session.plant.joint_velocity[name], np.zeros(6))
        for name, goal in desired.items():
            np.testing.assert_allclose(session.status()["scene"]["goals"][name], goal, atol=1e-12)

    def test_rejected_alignment_step_commits_neither_approach_nor_task(self):
        session = self.session("follow", "leader")
        session.command("arm")
        self.assert_tick(session)
        before = copy.deepcopy(session.plant.joints)
        clock, progress = session.now, session.approach.progress
        with mock.patch.object(session.plant, "step", side_effect=SimulationFault("alignment obstruction")), \
                mock.patch.object(session.approach, "commit", wraps=session.approach.commit) as commit, \
                mock.patch.object(session.engine, "commit", wraps=session.engine.commit) as task_commit:
            self.assertFalse(session.tick())
        commit.assert_not_called()
        task_commit.assert_not_called()
        self.assertEqual(session.now, clock)
        self.assertEqual(session.approach.progress, progress)
        self.assertEqual(session.engine.progress, 0.)
        self.assertFalse(session.active)
        self.assertIn("alignment obstruction", session.fault)
        for name in ROBOTS:
            np.testing.assert_array_equal(session.plant.joints[name], before[name])
            np.testing.assert_array_equal(session.plant.joint_velocity[name], np.zeros(6))

    def test_each_external_motion_mode_remains_frozen_while_alignment_progresses(self):
        for motion in ("leader", "bases", "both"):
            with self.subTest(motion=motion):
                session = self.session("follow", motion)
                desired = copy.deepcopy(session.status()["scene"]["goals"])
                session.command("arm")
                for _ in range(4):
                    self.assert_tick(session)
                self.assertGreater(session.approach.progress, 0.)
                self.assertEqual(session.engine.progress, 0.)
                leader = session.engine.leader
                np.testing.assert_array_equal(session.plant.joints[leader], self.initial["robots"][leader]["q"])
                for name in ROBOTS:
                    np.testing.assert_array_equal(session.plant.bases[name], self.initial["bases"][name])
                for name, pose in desired.items():
                    np.testing.assert_allclose(session.status()["scene"]["goals"][name], pose, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
