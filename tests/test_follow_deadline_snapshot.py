"""Replay the valid motion snapshot whose planning overran a live deadline.

Only the planning state is frozen here; acquisition/transport freshness has
separate runtime coverage. Recorded source timings document the incident, while
these tests compare commands and geometry call counts without wall-clock bounds.
No robot addresses, device handles, weights, or runtime log files are needed.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_batched_workspace_guard import scalar_guard_check
from urcal.cooperative.controller import CooperativeController
from urcal.cooperative.follow import govern_follow
from urcal.cooperative.state import RelativeSnapshot


FIXTURE = Path(__file__).parent / "fixtures" / "follow_deadline_snapshot.json"


def captured_controller(damping_time=.12):
    data = json.loads(FIXTURE.read_text())
    args = SimpleNamespace(**data["settings"], follow_damping_time=damping_time)
    engine = CooperativeController(args)
    row, previous = data["current"], data["previous"]
    snapshot = RelativeSnapshot(
        at=row["state_at"],
        base_poses={name: np.array(value) for name, value in row["bases"].items()},
        base_twists={name: np.array(value) for name, value in row["base_twists"].items()},
        relative_pose=np.array(row["relative_base_pose"]),
        relative_twist=np.array(row["relative_base_twist"]),
        measurement_age=deepcopy(row["measurement_age"]),
        measurement_times=deepcopy(row["measurement_times"]),
        received_times=deepcopy(row["received_times"]),
        prediction_dt=deepcopy(row["prediction_dt"]))
    engine.state = lambda *args, **kwargs: snapshot
    engine.previous_twist = np.array(previous["twist"])
    engine.previous_qdot = np.array(previous["joint_command"])
    engine.previous_feedback = np.array(previous["feedback"])
    engine.previous_compensation = np.array(previous["applied_compensation"])
    engine.previous_compensation_qdot = np.zeros(6)
    engine.previous_base_compensation = np.array(previous["base_compensation"])
    engine.previous_base_compensation_qdot = np.zeros(6)
    engine.previous_task_scale = previous["task_scale"]
    engine.tracking_offset = np.array(previous["tracking_offset"])
    engine.initial_tcp = np.array(data["initial_tcp"])
    engine.progress = row["progress"]
    states = {name: SimpleNamespace(
        joints=np.array(value["q"]), joint_velocity=np.array(value["qd"]),
        tcp=np.array(value["tcp"]), speed=np.array(value["tcp_speed"]),
        robot_timestamp=value["robot_timestamp"])
        for name, value in row["robots"].items()}
    return engine, states, data


def replay(engine, states, data, *, original_geometry=False, original_cache=False):
    # Capture the production bound method before installing the counter.
    if original_geometry:
        guard_check = lambda bases, joints: scalar_guard_check(engine.guard, bases, joints)
    else:
        guard_check = engine.guard.check
    calls = []

    def checked(bases, joints):
        calls.append({name: np.array(q, copy=True) for name, q in joints.items()})
        return guard_check(bases, joints)

    def governor(*args, **kwargs):
        if original_cache:
            kwargs["extra_feasible_depends_on_rho"] = True
        return govern_follow(*args, **kwargs)

    row = data["current"]
    with patch.object(engine.guard, "check", checked), \
            patch("urcal.cooperative.controller.govern_follow", governor):
        result = engine.prepare(row["at"], states, row["control_dt"], active=True)
    return result, calls


class RecordedDeadlineSnapshotTests(unittest.TestCase):
    def test_equivalent_geometry_and_cache_preserve_recorded_command_with_less_work(self):
        counts, outputs = [], []
        for original in (True, False):
            engine, states, data = captured_controller(damping_time=0.)
            result, calls = replay(engine, states, data,
                                   original_geometry=original, original_cache=original)
            np.testing.assert_allclose(result.decision.twist, data["expected"]["twist"], atol=1e-14)
            np.testing.assert_allclose(result.decision.qdot, data["expected"]["joint_command"], atol=1e-14)
            self.assertEqual(result.decision.reason, data["expected"]["decision"])
            counts.append(len(calls))
            outputs.append(result)
        self.assertGreaterEqual(counts[1], 3, "current scene and future sweep remain checked")
        self.assertLess(counts[1], counts[0] / 4)
        np.testing.assert_array_equal(outputs[0].decision.twist, outputs[1].decision.twist)

    def test_current_damped_law_produces_a_checked_bounded_proposal(self):
        engine, states, data = captured_controller(damping_time=.12)
        result, calls = replay(engine, states, data)
        args, row = engine.args, data["current"]
        dt, follower = row["control_dt"], states[engine.follower]
        twist, qdot = result.decision.twist, result.decision.qdot
        self.assertTrue(np.isfinite(twist).all())
        self.assertTrue(np.isfinite(qdot).all())
        self.assertLessEqual(np.linalg.norm(twist[:3]), engine.limits.linear_speed + 1e-12)
        self.assertLessEqual(np.linalg.norm(twist[3:]), args.max_angular_speed + 1e-12)
        self.assertLessEqual(np.max(abs(qdot)), args.max_joint_speed + 1e-12)
        for old_twist, old_qdot in ((follower.speed, follower.joint_velocity),
                                    (engine.previous_twist, engine.previous_qdot)):
            self.assertLessEqual(np.linalg.norm(twist[:3] - old_twist[:3]), args.max_accel * dt + 1e-12)
            self.assertLessEqual(np.linalg.norm(twist[3:] - old_twist[3:]), args.max_angular_accel * dt + 1e-12)
            self.assertLessEqual(np.max(abs(qdot - old_qdot)), args.max_joint_accel * dt + 1e-12)
        self.assertGreaterEqual(len(calls), 3)
        self.assertLess(len(calls), 30)
        self.assertTrue(any(np.linalg.norm(q[engine.follower] - follower.joints) > 0.
                            for q in calls), "future commanded joints must receive geometry checks")
        self.assertEqual(result.record["follow_damping_time"], .12)
        self.assertGreater(result.record["feedback_braking_linear_accel"], 0.)
        self.assertLessEqual(result.record["feedback_braking_linear_accel"], args.max_accel)
        np.testing.assert_allclose(
            result.record["feedback_relative_tcp_velocity"],
            follower.speed - result.record["base_compensation"] - result.record["leader_compensation"]
            - result.record["task_twist_at_command"] * data["previous"]["task_scale"], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
