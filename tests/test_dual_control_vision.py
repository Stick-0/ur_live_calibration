"""Raw-detection acceptance and capture-clock checks without any hardware."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))

from urcal.control.policy import CalibrationTracker, ControlFault
from urcal.control.vision import LatestCapture, calibration_frame


class CalibrationFrameTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[600., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.points3d = np.array([
            [-.15, -.12, 0.], [.13, -.13, .04], [.17, .14, .02],
            [-.14, .13, -.03], [.02, -.02, .12], [.08, .07, .07], [-.09, .04, .05],
        ])
        self.joints = [[0.] * 6, [.2] * 6]
        self.results = []
        self.estimators = []
        for translation in ([-.2, 0., 1.1], [.3, .1, 1.3]):
            rvec = np.array([.12, -.18, .07])
            tvec = np.asarray(translation)
            points = cv2.projectPoints(self.points3d, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
            self.results.append(SimpleNamespace(
                ok=True, rvec=rvec, tvec=tvec, points=points,
                valid=np.ones(7, dtype=bool), reproj_mean=987654.,
            ))
            self.estimators.append(SimpleNamespace(fk=Mock(return_value=self.points3d.copy())))

    def frame(self, skew=.04):
        return calibration_frame(17, 100., self.results, self.estimators,
                                 self.joints, self.K, self.dist, .08, skew)

    def test_reprojection_uses_original_points_and_ignores_reported_solver_error(self):
        perturbations = np.array([[3., 4.], [0., 2.], [0., 0.], [-3., 0.],
                                  [0., 0.], [0., -4.], [0., 0.]])
        self.results[0].points += perturbations
        self.results[0].reproj_mean = 0.
        frame = self.frame()
        self.assertEqual(frame.sequence, 17)
        self.assertEqual(frame.captured_at, 100.)
        self.assertFalse(frame.reason)
        self.assertEqual(frame.valid_counts, {"ur5e": 7, "ur10e": 7})
        self.assertAlmostEqual(frame.raw_mean_px["ur5e"], 2.)
        self.assertAlmostEqual(frame.raw_max_px["ur5e"], 5.)
        self.assertAlmostEqual(frame.raw_mean_px["ur10e"], 0.)
        self.assertAlmostEqual(frame.raw_max_px["ur10e"], 0.)
        for estimator, joints in zip(self.estimators, self.joints):
            estimator.fk.assert_called_once_with(joints)

    def test_hidden_raw_outlier_cannot_pass_tracker_with_zero_solver_error(self):
        self.results[0].points[0] += [12., 16.]
        self.results[0].reproj_mean = 0.
        frame = self.frame()
        self.assertAlmostEqual(frame.raw_max_px["ur5e"], 20.)
        self.assertAlmostEqual(frame.raw_mean_px["ur5e"], 20. / 7)
        tracker = CalibrationTracker(stable_frames=1)
        with self.assertRaisesRegex(ControlFault, "raw reprojection"):
            tracker.update(frame, 100.1)
        self.assertFalse(tracker.ready)

    def test_invalid_and_nonfinite_detections_do_not_enter_error_statistics(self):
        self.results[0].points[0] = np.nan
        self.results[0].points[1] += [10000., -10000.]
        self.results[0].valid[1] = False
        frame = self.frame()
        self.assertFalse(frame.reason)
        self.assertEqual(frame.valid_counts["ur5e"], 5)
        self.assertAlmostEqual(frame.raw_max_px["ur5e"], 0.)
        self.assertAlmostEqual(frame.raw_mean_px["ur5e"], 0.)

    def test_one_failed_robot_rejects_the_pair(self):
        self.results[1].ok = False
        frame = self.frame()
        self.assertIn("ur10e", frame.reason)
        self.assertEqual(set(frame.poses), {"ur5e"})
        with self.assertRaises(ControlFault):
            CalibrationTracker(stable_frames=1).update(frame, 100.1)

    def test_nonpositive_and_nonfinite_depths_are_rejected(self):
        self.results[0].rvec = np.zeros(3)
        self.results[0].tvec = np.array([0., 0., 1.])
        for z in (-2., -1., float("nan")):
            with self.subTest(z=z):
                points = self.points3d.copy()
                points[0, 2] = z
                self.estimators[0].fk.return_value = points
                frame = self.frame()
                self.assertIn("behind camera", frame.reason)
                self.assertNotIn("ur5e", frame.poses)

    def test_collinear_or_tiny_detection_layout_is_rejected(self):
        original = self.results[0].points.copy()
        for points in (np.column_stack((np.arange(7) * 30., np.ones(7) * 250.)),
                       original.mean(axis=0) + (original - original.mean(axis=0)) * .001):
            with self.subTest(points=points):
                self.results[0].points = points
                frame = self.frame()
                self.assertIn("collinear/small", frame.reason)
                self.assertNotIn("ur5e", frame.poses)

    def test_too_few_original_points_is_rejected(self):
        self.results[0].valid[:] = [True, True, True, False, False, False, False]
        frame = self.frame()
        self.assertEqual(frame.valid_counts["ur5e"], 3)
        self.assertIn("too few original detections", frame.reason)

    def test_excessive_and_nonfinite_joint_skew_rejects_even_good_poses(self):
        for skew in (.081, float("inf"), float("nan")):
            with self.subTest(skew=skew):
                frame = self.frame(skew)
                self.assertIn("skew", frame.reason)
                with self.assertRaises(ControlFault):
                    CalibrationTracker(stable_frames=1).update(frame, 100.1)


class CaptureClockTests(unittest.TestCase):
    def simulate(self, *, domain="hardware", frame_count=12, sensor_age=.05,
                 last_delay=0., frozen_robot=False, repeated_frame=False,
                 repeated_sensor=False, invalid_sensor=False):
        """Run the real capture loop with a deterministic fake SDK and clock."""
        clock = {"mono": 100., "wall": 1700000000., "index": 0}
        stop = threading.Event()
        domains = SimpleNamespace(hardware_clock="hardware", system_time="system", global_time="global")
        data = np.zeros((8, 12, 3), dtype=np.uint8)

        class Pipeline:
            def wait_for_frames(self, timeout_ms):
                self.timeout_ms = timeout_ms
                clock["index"] += 1
                index = clock["index"]
                nominal_mono = 100. + index / 30.
                clock["mono"] = nominal_mono + (last_delay if index == frame_count else 0.)
                clock["wall"] = 1700000000. + clock["mono"] - 100.
                number = index - 1 if repeated_frame and index == frame_count else index
                sensor_index = index - 1 if repeated_sensor and index == frame_count else index
                stamp = ((20. + sensor_index / 30.) if domain == "hardware"
                         else clock["wall"] - sensor_age)
                if invalid_sensor and index == frame_count:
                    stamp = float("nan")
                data[:] = index
                color = SimpleNamespace(
                    get_frame_number=lambda: number,
                    get_timestamp=lambda: stamp * 1000.,
                    get_frame_timestamp_domain=lambda: domain,
                    get_data=lambda: data,
                )
                if index == frame_count:
                    stop.set()
                return SimpleNamespace(get_color_frame=lambda: color)

        stream = SimpleNamespace(pipeline=Pipeline(), rs=SimpleNamespace(timestamp_domain=domains))
        receivers = {}
        for name in ("ur5e", "ur10e"):
            receiver = Mock()
            receiver.isConnected.return_value = True
            receiver.getActualQ.return_value = [0.] * 6
            if frozen_robot and name == "ur10e":
                receiver.getTimestamp.return_value = 25.
            else:
                receiver.getTimestamp.side_effect = lambda: 25. + clock["index"] / 30.
            receivers[name] = receiver
        with patch("urcal.control.vision.time.monotonic", side_effect=lambda: clock["mono"]), \
                patch("urcal.control.vision.time.time", side_effect=lambda: clock["wall"]), \
                patch("urcal.control.vision.threading.Thread") as thread:
            capture = LatestCapture(stream, receivers, stop, 30)
            thread.return_value.start.assert_called_once_with()
            capture._run()
        return capture, clock, data

    def test_hardware_clock_maps_to_monotonic_and_publishes_copied_latest_frame(self):
        capture, clock, data = self.simulate()
        sample = capture.read_after(None)
        self.assertIsNotNone(sample)
        sequence, captured, image, joints, skew = sample
        self.assertEqual(sequence, 12)
        self.assertAlmostEqual(captured, clock["mono"] - 1. / 30.)
        self.assertAlmostEqual(skew, 1. / 30.)
        self.assertEqual(joints, [[0.] * 6, [0.] * 6])
        self.assertIsNone(capture.read_after(sequence))
        data[:] = 0
        self.assertTrue(np.all(image == 12))

    def test_capture_warmup_does_not_publish_initial_buffered_frames(self):
        capture, _, _ = self.simulate(frame_count=9)
        self.assertIsNone(capture.read_after(None))
        capture, _, _ = self.simulate(frame_count=10)
        self.assertEqual(capture.read_after(None)[0], 10)

    def test_hardware_arrival_delay_increases_skew_instead_of_retimestamping_image(self):
        capture, clock, _ = self.simulate(last_delay=.2)
        _, captured, _, _, skew = capture.read_after(None)
        self.assertAlmostEqual(captured, clock["mono"] - .2 - 1. / 30.)
        self.assertAlmostEqual(skew, .2 + 1. / 30.)

    def test_system_and_global_timestamp_age_maps_to_monotonic(self):
        for domain in ("system", "global"):
            with self.subTest(domain=domain):
                capture, clock, _ = self.simulate(domain=domain, sensor_age=.075)
                _, captured, _, _, skew = capture.read_after(None)
                self.assertAlmostEqual(captured, clock["mono"] - .075, places=6)
                self.assertAlmostEqual(skew, .075, places=6)

    def test_system_clock_ahead_of_host_is_rejected(self):
        capture, _, _ = self.simulate(domain="system", sensor_age=-.1)
        with self.assertRaisesRegex(RuntimeError, "ahead of host"):
            capture.read_after(None)

    def test_frozen_rtde_timestamp_is_rejected_despite_advancing_camera(self):
        capture, _, _ = self.simulate(frozen_robot=True)
        with self.assertRaisesRegex(RuntimeError, "ur10e: capture RTDE state is stale"):
            capture.read_after(None)

    def test_repeated_frame_or_sensor_timestamp_latches_error_over_old_good_frame(self):
        for flags in ({"repeated_frame": True}, {"repeated_sensor": True}):
            with self.subTest(flags=flags):
                capture, _, _ = self.simulate(**flags)
                self.assertIsNotNone(capture.latest)
                with self.assertRaisesRegex(RuntimeError, "stopped advancing"):
                    capture.read_after(None)

    def test_nonfinite_camera_timestamp_is_rejected(self):
        capture, _, _ = self.simulate(invalid_sensor=True)
        with self.assertRaisesRegex(RuntimeError, "invalid camera timestamp"):
            capture.read_after(None)


if __name__ == "__main__":
    unittest.main()
