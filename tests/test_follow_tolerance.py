"""Recorded motion state and travel limits, without hardware connections."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.state import RelativeSnapshot
from test_bounded_follow import bounded_scene, move_joint
from test_cooperative_controller import accept


FIXTURE = Path(__file__).with_name("fixtures") / "follow_transition_limits.json"
CLI = ["--mode", "follow", "--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10"]


def recorded_case(index, **overrides):
    data = json.loads(FIXTURE.read_text())
    row = data["cases"][index]["last_control"]
    args = SimpleNamespace(**{**data["settings"], **overrides})
    controller = CooperativeController(args)
    controller.initial_tcp = np.array(data["initial_tcp"])
    controller.previous_twist = np.array(row["twist"])
    controller.previous_qdot = np.array(row["joint_command"])
    controller.previous_feedback = np.array(row["feedback"])
    controller.tracking_offset = np.array(row["tracking_offset"])
    snapshot = RelativeSnapshot(
        at=row["state_at"], base_poses={name: np.array(pose) for name, pose in row["bases"].items()},
        base_twists={name: np.array(value) for name, value in row["base_twists"].items()},
        relative_pose=np.array(row["relative_base_pose"]), relative_twist=np.array(row["relative_base_twist"]),
        measurement_age=row["measurement_age"], measurement_times=row["measurement_times"],
        received_times=row["received_times"], prediction_dt=row["prediction_dt"])
    controller.state = lambda now, active: snapshot
    states = {name: SimpleNamespace(joints=np.array(value["q"]), joint_velocity=np.array(value["qd"]),
                                   tcp=np.array(value["tcp"]), speed=np.array(value["tcp_speed"]),
                                   robot_timestamp=value["robot_timestamp"])
              for name, value in row["robots"].items()}
    return controller, states, row, data


class FollowToleranceTests(unittest.TestCase):
    def test_follow_defaults_and_explicit_overrides(self):
        for mode, expected in (("follow", (.75, .30)),):
            args = parse_args(["--mode", mode] + CLI[2:])
            self.assertEqual((args.max_joint_accel, args.max_travel), expected)
            tuned = parse_args(["--mode", mode] + CLI[2:] + ["--max_joint_accel", ".4", "--max_travel", ".08"])
            self.assertEqual((tuned.max_joint_accel, tuned.max_travel), (.4, .08))

    def test_recorded_history_at_next_nominal_tick_accepts_new_joint_acceleration(self):
        old, states, row, _ = recorded_case(0)
        with self.assertRaises(ResearchFault) as caught:
            old.prepare(row["at"] + .04, states, .04, active=True)
        self.assertEqual(caught.exception.category, "transition_infeasible")
        self.assertIn("measured_qdot", caught.exception.details)
        self.assertEqual(caught.exception.details["joint_acceleration_limit"], .5)

        args = parse_args(CLI)
        new, states, row, _ = recorded_case(0, max_joint_accel=args.max_joint_accel)
        proposal = new.prepare(row["at"] + .04, states, .04, active=True)
        self.assertGreater(np.linalg.norm(proposal.decision.twist[:3]), .05)
        self.assertLessEqual(np.max(np.abs(proposal.decision.qdot)), args.max_joint_speed + 1e-9)
        for previous in (new.previous_qdot, states[new.follower].joint_velocity):
            self.assertLessEqual(np.max(np.abs(proposal.decision.qdot - previous)), args.max_joint_accel * .04 + 1e-9)
        self.assertLessEqual(np.linalg.norm(proposal.decision.twist[:3] - new.previous_twist[:3]), args.max_accel * .04 + 1e-9)

    def test_recorded_travel_stop_is_inside_new_radius_with_reference_preserved(self):
        for radius, allowed in ((.10, False), (.30, True)):
            controller, states, row, data = recorded_case(1, max_travel=radius)
            for name, observed in data["cases"][1]["stopped_robots"].items():
                states[name].tcp = np.array(observed["tcp"])
                states[name].joints = np.array(observed["q"])
                states[name].speed = np.array(observed["tcp_speed"])
                states[name].joint_velocity = np.array(observed["qd"])
            # A fresh stationary preview has no active command history.
            controller.previous_qdot[:] = 0.
            controller.previous_twist[:] = 0.
            controller.previous_feedback[:] = 0.
            origin = deepcopy(controller.initial_tcp)
            if allowed:
                controller.prepare(row["at"], states, .04)
            else:
                with self.assertRaises(ResearchFault) as caught:
                    controller.prepare(row["at"], states, .04)
                self.assertEqual(caught.exception.category, "workspace")
                self.assertAlmostEqual(caught.exception.details["travel_m"], .100818, places=5)
                self.assertEqual(caught.exception.details["limit_m"], .10)
            np.testing.assert_array_equal(controller.initial_tcp, origin)

    def test_larger_radius_still_rejects_real_travel_beyond_its_boundary(self):
        controller, bases, states = bounded_scene(max_travel=.30)
        controller.initial_tcp = states[controller.follower].tcp.copy()
        for name in states:
            move_joint(controller, states, name, 0, .301)
        accept(controller, bases)
        with self.assertRaises(ResearchFault) as caught:
            controller.prepare(10., states, .04, active=True)
        self.assertEqual(caught.exception.category, "workspace")
        self.assertAlmostEqual(caught.exception.details["travel_m"], .301)
        self.assertEqual(caught.exception.details["limit_m"], .30)


if __name__ == "__main__":
    unittest.main()
