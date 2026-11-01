"""Pure relative-reference approach geometry and explicit action lifecycle."""

import unittest

import numpy as np

from urcal.control.geometry import facing_follower, pose_matrix
from urcal.cooperative.approach import GapApproach, build_gap_approach
from urcal.cooperative.task import RelativePath


def reference(gap=.3):
    return RelativePath(facing_follower(np.eye(4), gap))


class GapApproachGeometryTests(unittest.TestCase):
    def test_folds_existing_offset_without_changing_commanded_start(self):
        source = reference()
        offset = pose_matrix([.025, -.012, .03, .04, -.08, .06])
        initial, original_offset = source.pose(0.) @ offset, offset.copy()
        path = build_gap_approach(source, 0., offset)
        self.assertEqual(path.kind, "line")
        np.testing.assert_allclose(path.pose(0.), initial, atol=1e-15)
        final = path.pose(1.)
        self.assertEqual(final[2, 3], .15)
        np.testing.assert_allclose(final[:2, 3], initial[:2, 3])
        np.testing.assert_allclose(final[:3, :3], initial[:3, :3])
        np.testing.assert_array_equal(offset, original_offset)
        np.testing.assert_array_equal(source.pose(0.), facing_follower(np.eye(4), .3))

    def test_axial_motion_is_monotonic_and_never_crosses_exact_gap_boundary(self):
        source = reference(.3173)
        offset = pose_matrix([.02, -.01, -.0179, .01, -.02, .03])
        path = build_gap_approach(source, .4, offset)
        phases = np.r_[np.linspace(0., 1., 1001), np.nextafter(1., 0.)]
        poses = [path.pose(float(u)) for u in phases]
        for pose in poses:
            self.assertGreaterEqual(pose[2, 3], .15)
            self.assertGreaterEqual(np.linalg.norm(pose[:3, 3]), .15)
            np.testing.assert_array_equal(pose[:2, 3], path.start[:2, 3])
            np.testing.assert_array_equal(pose[:3, :3], path.start[:3, :3])
        self.assertTrue(np.all(np.diff([pose[2, 3] for pose in poses[:-1]]) <= 1e-15))
        self.assertEqual(path.pose(1.)[2, 3], .15)
        np.testing.assert_array_equal(path.derivative(0.), np.zeros(6))
        np.testing.assert_array_equal(path.derivative(1.), np.zeros(6))

    def test_quintic_peak_speed_and_acceleration_obey_requested_budgets(self):
        for speed, acceleration in ((.005, .01), (.5, .01)):
            with self.subTest(speed=speed):
                path = build_gap_approach(reference(.31), 0., None,
                                          speed=speed, acceleration=acceleration)
                phases = np.linspace(0., 1., 2001)
                velocities = np.array([path.derivative(float(u)) for u in phases])
                accelerations = np.diff(velocities, axis=0) / (path.duration / (len(phases) - 1))
                self.assertLessEqual(np.max(np.linalg.norm(velocities[:, :3], axis=1)), speed + 1e-12)
                self.assertLessEqual(np.max(np.linalg.norm(accelerations[:, :3], axis=1)), acceleration + 1e-10)
                np.testing.assert_array_equal(velocities[:, [0, 1, 3, 4, 5]], 0.)
                self.assertTrue(np.all(velocities[:, 2] <= 0.))
                self.assertGreater(np.max(abs(velocities[:, 2])), .001)

    def test_nonstationary_relative_tasks_cannot_be_replaced(self):
        for kind in ("line", "circle"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "stationary relative"):
                build_gap_approach(RelativePath(reference().start, kind=kind), .5, None)

    def test_higher_minimum_or_already_close_signed_gap_cannot_start(self):
        with self.assertRaisesRegex(ValueError, "minimum flange gap"):
            build_gap_approach(reference(), 0., None, min_gap=.16)
        for gap in (.15, .14):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, "already"):
                build_gap_approach(reference(gap), 0., None)
        # A lateral offset does not turn a negative/too-small axial gap into
        # permission to move to the opposite side of the leader flange.
        shifted = pose_matrix([.4, 0., .1, 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "below"):
            build_gap_approach(RelativePath(shifted), 0., None)

    def test_nonfinite_settings_and_bad_transform_are_rejected(self):
        for key, value in (("speed", 0.), ("speed", -.01), ("speed", np.nan),
                           ("speed", np.inf), ("acceleration", 0.), ("min_gap", np.nan),
                           ("target_gap", np.inf)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_gap_approach(reference(), 0., None, **{key: value})
        offset = np.eye(4)
        offset[0, 0] = 2.
        with self.assertRaises(ValueError):
            build_gap_approach(reference(), 0., offset)


class GapApproachLifecycleTests(unittest.TestCase):
    def test_start_pause_resume_preserves_same_reference_and_phase(self):
        approach = GapApproach()
        self.assertEqual(approach.state, "idle")
        self.assertFalse(approach.pause())
        self.assertFalse(approach.resume())
        path = approach.start(reference(), 0., None)
        approach.commit(.37)
        before = path.pose(.37)
        self.assertTrue(approach.pause("calibration observation"))
        self.assertFalse(approach.pause("operator"))
        paused = approach.info()
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["reason"], "calibration observation")
        self.assertAlmostEqual(paused["progress"], .37)
        self.assertTrue(approach.resume())
        self.assertEqual(approach.info()["progress"], .37)
        np.testing.assert_array_equal(path.pose(.37), before)
        self.assertEqual(approach.info()["state"], "running")
        self.assertFalse(approach.resume())

    def test_status_rendering_cannot_complete_or_advance_the_action(self):
        approach = GapApproach()
        approach.start(reference(), 0., None)
        approach.commit(.2)
        shown = approach.info(1.)
        self.assertEqual(shown["reference_gap_m"], .15)
        self.assertEqual(approach.state, "running")
        self.assertEqual(approach.info()["progress"], .2)
        approach.commit(1.)
        self.assertEqual(approach.state, "complete")
        self.assertIn("reference", approach.info()["reason"])
        self.assertFalse(approach.pause())
        self.assertFalse(approach.resume())

    def test_configured_action_cannot_be_restarted_or_reset_by_invalid_start(self):
        approach = GapApproach()
        with self.assertRaises(ValueError):
            approach.start(reference(), 0., None, speed=0.)
        self.assertEqual(approach.state, "idle")
        self.assertIsNone(approach.info()["duration_s"])
        approach.start(reference(), 0., None)
        for action in (lambda: None, lambda: approach.pause(), lambda: approach.commit(1.)):
            action()
            with self.assertRaisesRegex(ValueError, "already configured"):
                approach.start(reference(.4), 0., None)
            self.assertEqual(approach.info()["start_gap_m"], .3)
        approach.reset()
        self.assertEqual(approach.state, "idle")
        self.assertIsNone(approach.info()["reference_gap_m"])
        approach.start(reference(.4), 0., None)
        self.assertEqual(approach.info()["start_gap_m"], .4)


if __name__ == "__main__":
    unittest.main()
