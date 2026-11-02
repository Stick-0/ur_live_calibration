"""Raw-detection acceptance and capture-clock checks without any hardware."""

from pathlib import Path
from contextlib import ExitStack, redirect_stdout
import io
from queue import Queue
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
from urcal.control.joint_history import JointAlignmentUnavailable


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
        self.assertEqual(frame.valid_counts["ur10e"], 7)
        self.assertEqual(set(frame.poses), {"ur5e"})
        with self.assertRaises(ControlFault):
            CalibrationTracker(stable_frames=1).update(frame, 100.1)

    def test_pose_diagnostics_preserve_failed_inputs_without_changing_acceptance(self):
        regions = ((0, 0, 320, 480), (320, 0, 640, 480))
        for name, estimator in zip(("ur5e", "ur10e"), self.estimators):
            estimator.key_names = [name + f"_link{i}" for i in range(7)]
        self.results[0].ok = False
        self.results[0].rvec = self.results[0].tvec = None
        plain = self.frame()
        logged = calibration_frame(17, 100., self.results, self.estimators, self.joints,
                                   self.K, self.dist, .08, .04, regions=regions)
        self.assertEqual(logged.reason, plain.reason)
        self.assertEqual(logged.captured_at, plain.captured_at)
        self.assertEqual(logged.valid_counts, plain.valid_counts)
        self.assertEqual(logged.raw_mean_px, plain.raw_mean_px)
        self.assertEqual(logged.raw_max_px, plain.raw_max_px)
        self.assertEqual(set(logged.poses), {"ur10e"})
        np.testing.assert_array_equal(logged.poses["ur10e"], plain.poses["ur10e"])
        for frame in (plain, logged):
            with self.assertRaisesRegex(ControlFault, "ur5e: EMPnP failed"):
                CalibrationTracker(stable_frames=1).update(frame, 100.1)
        self.assertIsNone(plain.pose_diagnostics)
        robot = logged.pose_diagnostics["robots"]["ur5e"]
        self.assertFalse(robot["empnp_ok"])
        np.testing.assert_array_equal(robot["detected_px"], self.results[0].points)
        np.testing.assert_array_equal(robot["q_rad"], self.joints[0])
        self.assertEqual(robot["region"], list(regions[0]))

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

    def test_interpolation_metadata_and_capture_joints_reach_logged_pose_diagnostics(self):
        for name, estimator in zip(("ur5e", "ur10e"), self.estimators):
            estimator.key_names = [name + f"_link{i}" for i in range(7)]
        alignment = {"aligned": True, "captured_at": 100., "robots": {"ur5e": {"fraction": .4}}}
        packet = calibration_frame(17, 100., self.results, self.estimators, self.joints,
                                   self.K, self.dist, .08, .04,
                                   regions=((0, 0, 320, 480), (320, 0, 640, 480)),
                                   joint_alignment=alignment)
        self.assertEqual(packet.pose_diagnostics["joint_alignment"], alignment)
        np.testing.assert_array_equal(packet.pose_diagnostics["robots"]["ur10e"]["q_rad"], self.joints[1])


class CaptureClockTests(unittest.TestCase):
    def simulate(self, *, domain="hardware", frame_count=12, sensor_age=.05,
                 last_delay=0., frozen_robot=False, repeated_frame=False,
                 repeated_sensor=False, invalid_sensor=False, align_joints=False, alignment_failure=False):
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
        sampler = Mock()
        sampler.error = None

        def interpolate(captured, **_kwargs):
            if alignment_failure:
                raise JointAlignmentUnavailable("exposure has no bracketing joint samples")
            return ([[captured] * 6, [captured + 1.] * 6],
                    {name: {"max_bracket_distance_s": .005, "sdk_read_half_interval_s": .001}
                     for name in receivers})

        sampler.interpolate.side_effect = interpolate
        with patch("urcal.control.vision.time.monotonic", side_effect=lambda: clock["mono"]), \
                patch("urcal.control.vision.time.time", side_effect=lambda: clock["wall"]), \
                patch("urcal.control.vision.JointSampler", return_value=sampler) as sampler_factory, \
                patch("urcal.control.vision.threading.Thread") as thread:
            capture = LatestCapture(stream, receivers, stop, 30, align_joints=align_joints)
            thread.return_value.start.assert_called_once_with()
            capture._run()
        if not align_joints:
            sampler_factory.assert_not_called()
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

    def test_new_mode_interpolates_at_exposure_and_keeps_metadata_with_that_frame(self):
        capture, clock, data = self.simulate(align_joints=True)
        sequence, captured, image, joints, skew, metadata = capture.read_after(None)
        self.assertEqual(sequence, 12)
        np.testing.assert_allclose(joints, [[captured] * 6, [captured + 1.] * 6])
        self.assertTrue(metadata["aligned"])
        self.assertFalse(metadata["hardware_synchronized"])
        self.assertEqual(metadata["captured_at"], captured)
        self.assertAlmostEqual(captured, clock["mono"])
        self.assertAlmostEqual(metadata["camera"]["exposure_time_uncertainty_s"], 1. / 30.)
        self.assertAlmostEqual(skew, 1. / 30. + .006)
        self.assertEqual(capture.joint_sampler.interpolate.call_args.args, (captured,))
        for receiver in capture.receivers.values():
            receiver.getActualQ.assert_not_called()  # Single RTDE owner is the sampler.
        data[:] = 0
        self.assertTrue(np.all(image == 12))
        capture.close()
        capture.joint_sampler.close.assert_called_once()

    def test_new_mode_uses_mapped_sensor_time_without_subtracting_uncertainty(self):
        for domain in ("system", "global"):
            with self.subTest(domain=domain):
                capture, clock, _ = self.simulate(domain=domain, sensor_age=.01, align_joints=True)
                _, captured, _, joints, _, metadata = capture.read_after(None)
                self.assertAlmostEqual(captured, clock["mono"] - .01, places=6)
                np.testing.assert_allclose(joints[0], np.full(6, clock["mono"] - .01), atol=3e-7)
                self.assertEqual(capture.joint_sampler.interpolate.call_args.args, (captured,))
                self.assertAlmostEqual(metadata["camera"]["exposure_time_uncertainty_s"], 1. / 30.)

    def test_new_mode_transport_delay_preserves_old_capture_and_never_becomes_q_fallback(self):
        capture, clock, _ = self.simulate(last_delay=.2, align_joints=True)
        _, captured, _, joints, _, metadata = capture.read_after(None)
        self.assertAlmostEqual(captured, clock["mono"] - .2)
        np.testing.assert_allclose(joints[0], captured)
        self.assertIn("transport-biased", metadata["camera"]["mapping"])
        failed, _, _ = self.simulate(last_delay=.2, align_joints=True, alignment_failure=True)
        self.assertIsNone(failed.read_after(None))
        self.assertIn("bracketing", failed.read_rejection_after(None)[2]["reason"])

    def test_new_mode_invalid_future_camera_timestamp_is_rejected_before_interpolation(self):
        capture, _, _ = self.simulate(domain="global", sensor_age=-.1, align_joints=True)
        with self.assertRaisesRegex(RuntimeError, "ahead of host"):
            capture.read_after(None)
        capture.joint_sampler.interpolate.assert_not_called()

    def test_missing_joint_bracket_skips_image_and_exposes_rejection_metadata(self):
        capture, _, _ = self.simulate(align_joints=True, alignment_failure=True)
        self.assertIsNone(capture.read_after(None))
        sequence, captured, metadata = capture.read_rejection_after(None)
        self.assertEqual(sequence, 12)
        self.assertEqual(metadata["captured_at"], captured)
        self.assertFalse(metadata["aligned"])
        self.assertIn("bracketing", metadata["reason"])
        self.assertIsNone(capture.read_rejection_after(sequence))


class VisionDisplayTests(unittest.TestCase):
    def run_worker(self, keys, *, align_joints=False, alignment_rejected=False):
        import urcal.control.vision as vision
        import urcal.inference as inference
        from urcal.control.runtime import parse_args

        fixture = CalibrationFrameTests("test_one_failed_robot_rejects_the_pair")
        fixture.setUp()
        for result in fixture.results:
            result.base_plane, result.constraint_error = None, ""
        args = parse_args(["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10", "--device", "cpu", "--show_seg"])
        if align_joints:
            args.base_motion_mode = "relative_kalman"
        source = np.zeros((480, 640, 3), np.uint8)
        regions = ((0, 0, 320, 480), (320, 0, 640, 480))
        stream = SimpleNamespace(K=fixture.K, dist=fixture.dist, stop=Mock())
        fake_dual = SimpleNamespace(
            BUNDLE=BUNDLE, MODEL_HW=(480, 640), require_weights=Mock(),
            make_transform=Mock(), PVNetVoteConfig=Mock(), load_model=Mock(),
            UR5E_KEY_NAMES=["ur5e_link" + str(i) for i in range(7)],
            UR10E_KEY_NAMES=["ur10e_link" + str(i) for i in range(7)],
            ur5e_keypoints_from_joints=fixture.estimators[0].fk,
            ur10e_keypoints_from_joints=fixture.estimators[1].fk,
            RobotEstimator=lambda name, model, names, fk, config: SimpleNamespace(name=name, key_names=names, fk=fk),
            D435ColorStream=Mock(return_value=stream),
            process_frame=Mock(return_value=(regions, fixture.results)),
            draw_results=Mock(side_effect=lambda *a, **kw: (a[0].copy(), a[0].copy())),
            draw_pose_axes=Mock(side_effect=lambda image, *a: image))
        capture = Mock()
        metadata = {"aligned": True, "captured_at": 99.97, "test_capture": "timestamped interpolation"}
        capture.read_after.side_effect = [
            (i, 99.97, source.copy(), fixture.joints, .03) + ((metadata,) if align_joints else ())
            for i in range(len(keys))]
        capture.read_rejection_after.return_value = None
        if alignment_rejected:
            capture.read_after.side_effect = None
            capture.read_after.return_value = None
            capture.read_rejection_after.return_value = (7, 99.97, {**metadata, "aligned": False,
                                                                     "reason": "no bracketing joint samples"})
        frames, commands, statuses = Queue(), Queue(), Queue()
        target = np.eye(4)
        target[2, 3] = 1.
        statuses.put({"message": "ACTIVE controller-only status", "goals": {"ur5e": target}})
        stop, output = threading.Event(), io.StringIO()
        fake_torch = SimpleNamespace(hub=SimpleNamespace(set_dir=Mock()), device=lambda value: value)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {
                "torch": fake_torch,
                "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock(return_value=Mock())),
                "urcal.inference.infer_live_dual": fake_dual}))
            stack.enter_context(patch.object(inference, "infer_live_dual", fake_dual, create=True))
            stack.enter_context(patch.object(vision, "LatestCapture", return_value=capture))
            if alignment_rejected:
                stack.enter_context(patch.object(stop, "wait", side_effect=lambda *_: stop.set()))
            stack.enter_context(patch.object(vision.time, "monotonic", return_value=100.))
            stack.enter_context(patch.object(vision.cv2, "waitKey", side_effect=keys))
            stack.enter_context(patch.object(vision.cv2, "destroyAllWindows"))
            shown = stack.enter_context(patch.object(vision.cv2, "imshow"))
            text = stack.enter_context(patch.object(vision.cv2, "putText", wraps=cv2.putText))
            save = stack.enter_context(patch.object(vision, "save_calibration_debug", return_value=Path("snapshot.json")))
            stack.enter_context(redirect_stdout(output))
            vision.vision_worker(args, frames, commands, statuses, stop)
        self.assertNotIn("[vision STOP]", output.getvalue())
        self.assertTrue(stop.is_set())
        fake_dual.frames = frames
        return fake_dual, shown, text, save, output.getvalue()

    def test_live_control_has_no_text_and_reports_inference_only_to_terminal(self):
        dual, shown, text, save, output = self.run_worker([ord("q")])
        text.assert_not_called()
        save.assert_not_called()
        dual.draw_pose_axes.assert_called_once()
        self.assertEqual(shown.call_count, 2)  # Main view and segmentation.
        self.assertIn("a: Arm", output)
        self.assertIn("Magenta crosses", output)
        self.assertIn("raw mean/max=", output)
        self.assertNotIn("ACTIVE controller-only status", output)
        self.assertEqual(output.count("[vision] FPS"), 1)

    def test_explicit_save_keeps_detailed_overlay_off_the_live_image(self):
        _dual, shown, text, save, output = self.run_worker([ord("d"), ord("q")])
        save.assert_called_once()
        self.assertTrue(text.called)
        saved_overlay = save.call_args.args[2]
        live_image = shown.call_args_list[0].args[1]
        self.assertFalse(np.array_equal(saved_overlay, live_image))
        self.assertEqual(save.call_args.args[4], "ACTIVE controller-only status")
        self.assertEqual(output.count("[vision] FPS"), 1)  # No frame-by-frame terminal spam.

    def test_aligned_capture_metadata_survives_inference_and_calibration_packet(self):
        dual, _shown, _text, _save, output = self.run_worker([ord("q")], align_joints=True)
        packet = dual.frames.get_nowait()
        self.assertTrue(packet.pose_diagnostics["joint_alignment"]["aligned"])
        self.assertEqual(packet.pose_diagnostics["joint_alignment"]["test_capture"], "timestamped interpolation")
        self.assertEqual(dual.process_frame.call_args.args[1], [[0.] * 6, [.2] * 6])
        self.assertIn("not hardware synchronized", output)

    def test_unbracketed_frame_is_logged_as_rejected_without_model_inference(self):
        dual, _shown, _text, _save, output = self.run_worker([], align_joints=True, alignment_rejected=True)
        dual.process_frame.assert_not_called()
        packet = dual.frames.get_nowait()
        self.assertEqual(packet.sequence, 7)
        self.assertEqual(packet.poses, {})
        self.assertFalse(packet.pose_diagnostics["joint_alignment"]["aligned"])
        self.assertIn("no bracketing", packet.reason)
        self.assertIn("[vision] WAIT:", output)


if __name__ == "__main__":
    unittest.main()
