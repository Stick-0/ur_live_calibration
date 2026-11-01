"""Robust visual coordinate application without hardware or robot-velocity inference."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urcal.control.geometry import inverse, matrix_pose, pose_matrix, rotation_distance
from urcal.cooperative.pose_smoothing import RelativePoseSmoother
from urcal.cooperative.state import RelativeSnapshot


def pose(x=0., y=0., z=0., rx=0., ry=0., rz=0.):
    return pose_matrix([x, y, z, rx, ry, rz])


def snapshot(t, relative=None, leader=None):
    relative = pose(.9, 0., 0.) if relative is None else relative
    leader = pose(.2, -.1, 2., .1, -.2, .3) if leader is None else leader
    return RelativeSnapshot(
        at=t + .02,
        base_poses={"ur5e": leader @ relative, "ur10e": leader.copy()},
        base_twists={name: np.zeros(6) for name in ("ur5e", "ur10e")},
        relative_pose=relative.copy(), relative_twist=np.zeros(6),
        measurement_age={"ur5e": .02, "ur10e": .02},
        measurement_times={"ur5e": t, "ur10e": t},
        received_times={"ur5e": t + .01, "ur10e": t + .015},
        prediction_dt={"ur5e": 0., "ur10e": 0.})


class RelativePoseSmoothingTests(unittest.TestCase):
    def apply(self, smoother, t, relative=None, leader=None):
        return smoother.apply(snapshot(t, relative, leader), t + .02, "ur10e")

    def test_first_valid_capture_seeds_geometry_without_artificial_ramp(self):
        smoother = RelativePoseSmoother()
        source = snapshot(10., pose(.9, -.1, .2, .2, -.1, .4))
        result = smoother.apply(source, 10.02, "ur10e")
        np.testing.assert_allclose(result.relative_pose, source.relative_pose, atol=1e-14)
        for name in source.base_poses:
            np.testing.assert_allclose(result.base_poses[name], source.base_poses[name], atol=1e-14)

    def test_single_depth_solution_switch_cannot_move_accepted_geometry(self):
        smoother = RelativePoseSmoother()
        baseline = pose(.9, -.1, .2)
        for i in range(15):
            measured = baseline.copy()
            if i == 7:
                measured[2, 3] += .30  # Same class of depth branch jump as the live log.
            result = self.apply(smoother, 10. + .04 * i, measured)
            np.testing.assert_allclose(result.relative_pose, baseline, atol=1e-14)

    def test_two_isolated_outliers_in_five_frames_do_not_replace_majority(self):
        smoother = RelativePoseSmoother()
        baseline = pose(.9, -.1, .2, .1, .2, -.1)
        for i in range(14):
            measured = baseline if i not in (5, 7) else pose(.7, .2, .5, .5, -.2, .7)
            result = self.apply(smoother, 10. + .04 * i, measured)
            np.testing.assert_allclose(result.relative_pose, baseline, atol=1e-14)

    def test_sustained_base_step_is_applied_monotonically_with_bounded_rate(self):
        smoother = RelativePoseSmoother()
        positions = []
        for i in range(220):
            measured = pose(.9 + (.05 if i >= 6 else 0.), 0., 0.)
            positions.append(self.apply(smoother, 10. + .04 * i, measured).relative_pose[0, 3])
        increments = np.diff(positions)
        self.assertGreater(positions[-1] - positions[0], .044)
        self.assertLessEqual(abs(.95 - positions[-1]), smoother.deadband_m + 1e-5)
        self.assertTrue(np.all(increments >= -1e-14))
        self.assertLessEqual(float(max(increments)), smoother.linear_rate * .04 + 1e-12)
        self.assertEqual(positions[6], positions[5])  # First changed frame is insufficient evidence.

    def test_continuing_slow_base_motion_is_not_frozen_as_stationary_noise(self):
        smoother = RelativePoseSmoother()
        positions = []
        for i in range(251):
            t = .04 * i
            positions.append(self.apply(smoother, 10. + t, pose(.9 + .005 * t)).relative_pose[0, 3])
        self.assertGreater(positions[-1] - positions[0], .04)
        self.assertLess(.95 - positions[-1], .01)
        self.assertTrue(np.all(np.diff(positions) >= -1e-14))

    def test_deadband_uses_vector_distance_not_independent_axis_thresholds(self):
        smoother = RelativePoseSmoother(deadband_m=.005)
        baseline = pose(.9)
        delta = np.array([.004, .004, 0.])  # Each component < 5 mm, vector length > 5 mm.
        for i in range(80):
            measured = baseline.copy()
            if i >= 5:
                measured[:3, 3] += delta
            result = self.apply(smoother, 10. + .04 * i, measured)
        step = result.relative_pose[:3, 3] - baseline[:3, 3]
        self.assertGreater(np.linalg.norm(step), .0005)
        np.testing.assert_allclose(step[0], step[1], atol=1e-13)
        self.assertLessEqual(np.linalg.norm(delta - step), .00501)

    def test_common_camera_motion_cancels_before_smoothing(self):
        plain, moved = RelativePoseSmoother(), RelativePoseSmoother()
        initial_leader = pose(.2, -.1, 2., .1, -.2, .3)
        for i in range(80):
            t = .04 * i
            relative = pose(.9 + .01 * t, -.1, .2, .1, .05 * t, -.1)
            camera_change = pose(.15 * t, -.08 * t, .1 * t, .2 * t, -.1 * t, .3 * t)
            ordinary = self.apply(plain, 10. + t, relative, initial_leader)
            transformed = self.apply(moved, 10. + t, relative, camera_change @ initial_leader)
            np.testing.assert_allclose(transformed.relative_pose, ordinary.relative_pose, atol=1e-12)
            for name in ordinary.base_poses:
                np.testing.assert_allclose(transformed.base_poses[name],
                                           camera_change @ ordinary.base_poses[name], atol=2e-12)

    def test_rotation_is_rate_bounded_and_geometry_remains_consistent_at_tool_lever_arm(self):
        smoother = RelativePoseSmoother(deadband_m=0., deadband_rad=0.)
        axis = np.array([1., 2., -1.]) / np.sqrt(6.)
        before = None
        tool_in_follower = pose(.45, -.2, .35, .1, -.2, .3)
        positions = []
        for i in range(160):
            rotation = axis * (.20 if i >= 5 else 0.)
            result = self.apply(smoother, 10. + .04 * i, pose(.9, -.1, .2, *rotation))
            matrix_pose(result.relative_pose)  # No elementwise rotation-matrix blending.
            if before is not None:
                self.assertLessEqual(rotation_distance(before, result.relative_pose),
                                     smoother.angular_rate * .04 + 1e-12)
            expected_tool = result.base_poses["ur10e"] @ result.relative_pose @ tool_in_follower
            np.testing.assert_allclose(result.base_poses["ur5e"] @ tool_in_follower,
                                       expected_tool, atol=2e-13)
            positions.append((result.relative_pose @ tool_in_follower)[:3, 3])
            before = result.relative_pose
        self.assertLess(rotation_distance(before, pose(.9, -.1, .2, *(axis * .2))), .001)
        self.assertGreater(np.linalg.norm(positions[-1] - positions[0]), .05)

    def test_rotation_crossing_pi_uses_short_arc_and_not_angle_component_average(self):
        smoother = RelativePoseSmoother(deadband_rad=0.)
        start = pose(.9, rz=np.deg2rad(179.))
        destination = pose(.9, rz=np.deg2rad(-179.))
        for i in range(100):
            result = self.apply(smoother, 10. + .04 * i, start if i < 5 else destination)
            self.assertLess(rotation_distance(start, result.relative_pose), np.deg2rad(2.001))
        self.assertLess(rotation_distance(destination, result.relative_pose), .0001)

    def test_filter_never_refreshes_capture_age_receipt_or_prediction_metadata(self):
        smoother = RelativePoseSmoother()
        source = snapshot(10.)
        result = smoother.apply(source, 10.02, "ur10e")
        self.assertEqual(result.at, source.at)
        for field in ("measurement_age", "measurement_times", "received_times", "prediction_dt"):
            self.assertIs(getattr(result, field), getattr(source, field))
        repeated = smoother.apply(source, 10.3, "ur10e")
        self.assertEqual(repeated.measurement_times, source.measurement_times)
        self.assertEqual(repeated.measurement_age, source.measurement_age)
        self.assertEqual(smoother.diagnostics()["samples"], 1)

    def test_duplicate_control_calls_do_not_add_evidence_or_spend_more_time(self):
        smoother = RelativePoseSmoother()
        for i in range(10):
            source = snapshot(10. + .04 * i, pose(.9 + (.05 if i >= 5 else 0.)))
            once = smoother.apply(source, source.at, "ur10e")
            count = smoother.diagnostics()["samples"]
            for _ in range(20):
                repeated = smoother.apply(source, source.at, "ur10e")
                np.testing.assert_array_equal(repeated.relative_pose, once.relative_pose)
                self.assertEqual(smoother.diagnostics()["samples"], count)

    def test_changed_state_with_repeated_capture_time_cannot_manufacture_votes(self):
        smoother = RelativePoseSmoother()
        baseline = pose(.9)
        for i in range(5):
            self.apply(smoother, 10. + .04 * i, baseline)
        source = snapshot(10.2, pose(1.2))
        first = smoother.apply(source, source.at, "ur10e")
        for i in range(20):
            altered = snapshot(10.2, pose(1.3 + i * .01))
            repeated = smoother.apply(altered, source.at, "ur10e")
            np.testing.assert_array_equal(repeated.relative_pose, first.relative_pose)
        np.testing.assert_allclose(first.relative_pose, baseline, atol=1e-14)

    def test_outage_cannot_be_spent_as_large_coordinate_motion_on_next_tick(self):
        smoother = RelativePoseSmoother(deadband_m=0., deadband_rad=0.)
        for i in range(10):
            result = self.apply(smoother, 10. + .04 * i, pose(.9 + (.2 if i >= 5 else 0.),
                                                           rz=.3 if i >= 5 else 0.))
        before = result.relative_pose.copy()
        after = self.apply(smoother, 25., pose(1.1, rz=.3)).relative_pose
        self.assertLessEqual(np.linalg.norm(after[:3, 3] - before[:3, 3]),
                             smoother.linear_rate * smoother.max_dt + 1e-12)
        self.assertLessEqual(rotation_distance(after, before),
                             smoother.angular_rate * smoother.max_dt + 1e-12)
        self.assertEqual(smoother.diagnostics()["samples"], 1)

    def test_expired_window_cannot_keep_pulling_toward_an_obsolete_target(self):
        smoother = RelativePoseSmoother(deadband_m=0., deadband_rad=0.)
        for i in range(10):
            result = self.apply(smoother, 10. + .04 * i, pose(.9 + (.2 if i >= 5 else 0.),
                                                           rz=.3 if i >= 5 else 0.))
        before = result.relative_pose.copy()
        # The first observations after an outage disagree with the old target.
        # They must not grant an extra correction toward that expired target.
        for i in range(2):
            result = self.apply(smoother, 25. + .04 * i, pose(.8, rz=-.2))
            np.testing.assert_allclose(result.relative_pose, before, atol=1e-14)
        confirmed = self.apply(smoother, 25.08, pose(.8, rz=-.2))
        self.assertLess(confirmed.relative_pose[0, 3], before[0, 3])
        self.assertLess(rotation_distance(confirmed.relative_pose, pose(.8, rz=-.2)),
                        rotation_distance(before, pose(.8, rz=-.2)))

    def test_mutating_outputs_does_not_change_internal_or_source_geometry(self):
        smoother = RelativePoseSmoother()
        source = snapshot(10.)
        expected = {name: value.copy() for name, value in source.base_poses.items()}
        output = smoother.apply(source, source.at, "ur10e")
        output.relative_pose[:] = 0.
        output.base_poses["ur5e"][:] = 0.
        output.base_poses["ur10e"][:] = 0.
        output.base_twists["ur5e"][:] = 100.
        again = smoother.apply(source, source.at, "ur10e")
        for name in expected:
            np.testing.assert_allclose(again.base_poses[name], expected[name], atol=1e-14)
            np.testing.assert_array_equal(source.base_poses[name], expected[name])

    def test_visual_pose_filter_does_not_generate_velocity_from_geometry_changes(self):
        smoother = RelativePoseSmoother()
        for i in range(20):
            result = self.apply(smoother, 10. + .04 * i, pose(.9 + .02 * i, rz=.02 * i))
            np.testing.assert_array_equal(result.relative_twist, np.zeros(6))
            for twist in result.base_twists.values():
                np.testing.assert_array_equal(twist, np.zeros(6))

    def test_reset_starts_from_new_capture_and_discards_prior_votes_and_clock(self):
        smoother = RelativePoseSmoother()
        for i in range(10):
            self.apply(smoother, 10. + .04 * i, pose(.9 + .02 * i))
        smoother.reset()
        result = self.apply(smoother, 1., pose(.4, .2, .1, .2, .1, .3))
        np.testing.assert_allclose(result.relative_pose, pose(.4, .2, .1, .2, .1, .3), atol=1e-14)
        self.assertEqual(smoother.diagnostics()["samples"], 1)

    def test_nonfinite_invalid_rate_and_window_configuration_is_rejected(self):
        for window in (True, 1, 2, 4, 33, 5., np.nan):
            with self.subTest(window=window), self.assertRaises(ValueError):
                RelativePoseSmoother(window=window)
        for field in ("time_constant", "linear_rate", "angular_rate", "max_dt"):
            for value in (0., -1., np.inf, np.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    RelativePoseSmoother(**{field: value})
        for field in ("deadband_m", "deadband_rad"):
            for value in (-1., np.inf, np.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    RelativePoseSmoother(**{field: value})

    def test_invalid_or_reordered_clock_is_rejected_without_advancing_state(self):
        smoother = RelativePoseSmoother()
        initial = self.apply(smoother, 10.)
        for now in (10., np.nan, np.inf):
            with self.subTest(now=now), self.assertRaises(ValueError):
                smoother.apply(snapshot(10.), now, "ur10e")
        later = snapshot(10.04)
        result = smoother.apply(later, later.at, "ur10e")
        np.testing.assert_allclose(result.relative_pose, initial.relative_pose, atol=1e-14)

    def test_invalid_capture_times_and_robot_pair_are_rejected(self):
        smoother = RelativePoseSmoother()
        self.apply(smoother, 10.)
        for times in ({"ur5e": np.nan, "ur10e": 10.04},
                      {"ur5e": 10.07, "ur10e": 10.04},
                      {"ur5e": 9.99, "ur10e": 10.04}):
            with self.subTest(times=times), self.assertRaises(ValueError):
                smoother.apply(replace(snapshot(10.04), measurement_times=times), 10.06, "ur10e")
        with self.assertRaises(ValueError):
            smoother.apply(snapshot(10.04), 10.06, "other")
        with self.assertRaises(ValueError):
            smoother.apply(replace(snapshot(10.04), base_poses={"ur5e": np.eye(4)}), 10.06, "ur5e")

    def test_invalid_transform_does_not_poison_next_valid_sample(self):
        for name in ("ur5e", "ur10e"):
            for malformed in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
                with self.subTest(name=name, shape=malformed.shape):
                    smoother = RelativePoseSmoother()
                    self.apply(smoother, 10.)
                    invalid = snapshot(10.04)
                    invalid.base_poses[name] = malformed
                    with self.assertRaises(ValueError):
                        smoother.apply(invalid, 10.06, "ur10e")
                    result = self.apply(smoother, 10.04)
                    np.testing.assert_allclose(result.relative_pose, pose(.9), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
