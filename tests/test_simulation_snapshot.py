"""Snapshot persistence and receive-only acquisition with no hardware access."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from urcal.control.geometry import model_from_controller_base, pose_matrix
from urcal.control.policy import CalibrationFrame, ControlFault
from urcal.control.runtime import parse_args
from urcal.simulation import snapshot as module


def initial_state():
    poses = {name: pose_matrix([index * .8, -.2, 1.2, .1, -.2, .05])
             for index, name in enumerate(module.ROBOTS)}
    return {
        "schema": module.SCHEMA,
        "captured_at_utc": "2026-10-18T09:15:00Z",
        "calibration": {"sequence": 12, "captured_at_monotonic": 20., "poses": poses,
                        "raw_mean_px": {name: 12. for name in module.ROBOTS},
                        "raw_max_px": {name: 24. for name in module.ROBOTS},
                        "valid_counts": {name: 7 for name in module.ROBOTS}},
        "bases": {name: pose @ model_from_controller_base(name) for name, pose in poses.items()},
        "robots": {name: {"q": np.arange(6) * .1, "qd": np.arange(6) * .01,
                          "speed": np.arange(6) * .001,
                          "tcp": pose_matrix([.2, .1, .4, 0, .2, 0]),
                          "tcp_offset": np.eye(4), "tcp_offset_source": "zero_assumed",
                          "robot_timestamp": 100. + index}
                   for index, name in enumerate(module.ROBOTS)},
    }


def config():
    result = parse_args(["--mode", "follow", "--ur5e_ip", "192.0.2.1", "--ur10e_ip", "192.0.2.2",
                         "--stable_frames", "5"])
    result.sim_initial_timeout = .3
    return result


class SnapshotValidationTests(unittest.TestCase):
    def test_roundtrip_and_exclusive_save_preserve_frames_and_values(self):
        original = initial_state()
        original["robots"]["ur5e"]["tcp_offset"] = pose_matrix([.01, .02, .03, .1, -.2, .3])
        original["robots"]["ur5e"]["tcp_offset_source"] = "explicit"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session" / "initial.json"
            self.assertEqual(module.save_initial_state(path, original), path)
            saved_bytes = path.read_bytes()
            result = module.load_initial_state(path)
            for name in module.ROBOTS:
                np.testing.assert_allclose(result["bases"][name], original["bases"][name])
                for key in ("q", "qd", "speed", "tcp", "tcp_offset"):
                    np.testing.assert_allclose(result["robots"][name][key], original["robots"][name][key])
            self.assertEqual(result["calibration"]["raw_mean_px"]["ur5e"], 12.)
            self.assertTrue(result["calibration"]["ur10e_output_correction"])
            with self.assertRaises(FileExistsError):
                module.save_initial_state(path, original)
            self.assertEqual(path.read_bytes(), saved_bytes)

    def test_raw_output_snapshot_roundtrip_preserves_unmodified_bases(self):
        original = initial_state()
        original["calibration"]["ur10e_output_correction"] = False
        original["bases"] = {name: pose.copy() for name, pose in original["calibration"]["poses"].items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"
            module.save_initial_state(path, original)
            result = module.load_initial_state(path)
        self.assertFalse(result["calibration"]["ur10e_output_correction"])
        for name in module.ROBOTS:
            np.testing.assert_array_equal(result["bases"][name], original["calibration"]["poses"][name])

    def test_snapshot_cannot_silently_change_stored_output_convention(self):
        sample = initial_state()
        sample["calibration"]["ur10e_output_correction"] = False
        with self.assertRaisesRegex(ValueError, "model-to-controller"):
            module.validate_initial_state(sample)

    def test_validation_result_does_not_alias_input(self):
        original = initial_state()
        result = module.validate_initial_state(original)
        result["robots"]["ur5e"]["q"][0] = 500
        result["calibration"]["poses"]["ur10e"][0, 3] = 500
        result["bases"]["ur5e"][0, 3] = 500
        result["calibration"]["raw_mean_px"]["ur5e"] = 500
        self.assertEqual(original["robots"]["ur5e"]["q"][0], 0)
        self.assertLess(original["calibration"]["poses"]["ur10e"][0, 3], 2)
        self.assertLess(original["bases"]["ur5e"][0, 3], 2)
        self.assertEqual(original["calibration"]["raw_mean_px"]["ur5e"], 12.)

    def test_invalid_structure_timestamps_and_statistics_are_rejected(self):
        cases = [
            (("schema",), "unknown"),
            (("captured_at_utc",), "2026-10-18T09:15:00"),
            (("captured_at_utc",), "2026-10-18T09:15:00+09:00"),
            (("calibration", "sequence"), True),
            (("calibration", "ur10e_output_correction"), "false"),
            (("calibration", "ur10e_output_correction"), 0),
            (("calibration", "captured_at_monotonic"), float("inf")),
            (("calibration", "raw_max_px", "ur5e"), 1.),
            (("calibration", "valid_counts", "ur5e"), 3),
            (("robots", "ur5e", "robot_timestamp"), float("nan")),
            (("robots", "ur10e", "tcp_offset_source"), "read_from_controller"),
            (("robots", "ur5e", "q"), [0.] * 5),
            (("robots", "ur5e", "qd"), [0.] * 5 + [float("nan")]),
            (("robots", "ur5e", "speed"), [0.] * 5 + [float("inf")]),
        ]
        for keys, value in cases:
            with self.subTest(keys=keys):
                sample = initial_state()
                target = sample
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    module.validate_initial_state(sample)
        for field in ("bases", "robots"):
            sample = initial_state()
            del sample[field]["ur10e"]
            with self.assertRaises(ValueError):
                module.validate_initial_state(sample)

    def test_invalid_rotations_and_base_mapping_are_rejected(self):
        for field in ("tcp", "tcp_offset"):
            sample = initial_state()
            sample["robots"]["ur5e"][field][0, 0] = -1
            with self.assertRaisesRegex(ValueError, "rigid"):
                module.validate_initial_state(sample)
        sample = initial_state()
        sample["bases"]["ur10e"] = sample["calibration"]["poses"]["ur10e"].copy()
        with self.assertRaisesRegex(ValueError, "model-to-controller"):
            module.validate_initial_state(sample)
        sample = initial_state()
        sample["robots"]["ur5e"]["tcp_offset"][0, 3] = .1
        with self.assertRaisesRegex(ValueError, "assumed-zero"):
            module.validate_initial_state(sample)

    def test_invalid_input_is_rejected_before_creating_file(self):
        sample = initial_state()
        sample["robots"]["ur5e"]["tcp"][0, 0] = 2
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            with self.assertRaises(ValueError):
                module.save_initial_state(path, sample)
            self.assertFalse(path.exists())

    def test_import_and_saved_state_loading_do_not_import_robot_sdks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = module.save_initial_state(Path(folder) / "initial.json", initial_state())
            code = ("import sys; from urcal.simulation.snapshot import load_initial_state; "
                    "load_initial_state(sys.argv[1]); "
                    "assert 'rtde_receive' not in sys.modules; "
                    "assert 'rtde_control' not in sys.modules; "
                    "assert 'pyrealsense2' not in sys.modules; "
                    "assert 'torch' not in sys.modules")
            subprocess.run([sys.executable, "-B", "-c", code, str(path)], check=True,
                           cwd=Path(__file__).resolve().parents[1], capture_output=True)


class FakeQueue(Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.cancelled = self.closed = False

    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class AcquisitionRig:
    """Synthetic clock, process and robot receives; no worker code is run."""
    def __init__(self, *, frozen=False, failing_start=False, stubborn=False):
        self.now = 10.
        self.queues = []
        self.sequence = 0
        self.frozen = frozen
        self.failing_start = failing_start
        self.stubborn = stubborn
        self.worker_stop = Mock()
        self.worker_stop.is_set.side_effect = lambda: self.stopped
        self.worker_stop.set.side_effect = lambda: setattr(self, "stopped", True)
        self.worker_stop.wait.side_effect = self.advance
        self.stopped = False
        self.worker = Mock(pid=None)
        self.worker.start.side_effect = self.start
        self.worker.is_alive.side_effect = lambda: self.alive
        self.worker.join.side_effect = self.join
        self.worker.terminate.side_effect = lambda: setattr(self, "alive", False)
        self.alive = False
        self.context = Mock()
        self.context.Queue.side_effect = self.queue
        self.context.Event.return_value = self.worker_stop
        self.context.Process.return_value = self.worker
        self.receivers = [self.receiver(), self.receiver()]
        self.receive_factory = Mock(side_effect=self.receivers)
        self.control_factory = Mock(side_effect=AssertionError("control SDK is forbidden"))
        self.on_wait = None

    def queue(self, maxsize):
        value = FakeQueue(maxsize)
        self.queues.append(value)
        return value

    def receiver(self):
        value = Mock()
        value.isConnected.return_value = True
        value.getTimestamp.side_effect = lambda: 123. if self.frozen else self.now
        value.getSafetyMode.return_value = 1
        value.getRobotMode.return_value = 7
        value.getActualQ.return_value = np.arange(6) * .1
        value.getActualQd.return_value = np.arange(6) * .01
        value.getActualTCPSpeed.return_value = np.arange(6) * .001
        value.getActualTCPPose.return_value = [.1, -.2, .3, .1, .2, 0]
        return value

    def start(self):
        if self.failing_start:
            raise RuntimeError("worker startup failed")
        self.worker.pid = 1234
        self.alive = True
        self.publish()

    def publish(self):
        sample = initial_state()["calibration"]
        frame = CalibrationFrame(self.sequence, self.now, sample["poses"], sample["raw_mean_px"],
                                 sample["raw_max_px"], sample["valid_counts"])
        self.sequence += 1
        from urcal.control.runtime import put_latest
        put_latest(self.queues[0], frame)

    def advance(self, duration):
        self.now += duration
        self.publish()
        if self.on_wait is not None:
            self.on_wait()

    def join(self, timeout):
        if not self.stubborn:
            self.alive = False

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(module.mp, "get_context", return_value=self.context))
        stack.enter_context(patch.object(module.time, "monotonic", side_effect=lambda: self.now))
        stack.enter_context(patch.dict(sys.modules, {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=self.receive_factory),
            "rtde_control": SimpleNamespace(RTDEControlInterface=self.control_factory),
        }))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def assert_closed(self, testcase):
        for receiver in self.receivers:
            receiver.disconnect.assert_called_once()
        self.control_factory.assert_not_called()
        self.worker_stop.set.assert_called_once()
        testcase.assertTrue(all(queue.closed and queue.cancelled for queue in self.queues))


class AcquisitionTests(unittest.TestCase):
    def test_live_capture_waits_for_stable_pair_and_advancing_states_then_closes_everything(self):
        rig = AcquisitionRig()
        args = config()
        args.ur5e_tcp_offset = [.01, .02, .03, .1, .2, .3]
        callback = Mock()
        caller_stop = threading.Event()
        with rig.patches():
            result = module.capture_initial_state(args, callback, caller_stop)
        self.assertEqual(result["calibration"]["sequence"], 4)
        self.assertFalse(caller_stop.is_set())
        self.assertEqual(result["robots"]["ur5e"]["tcp_offset_source"], "explicit")
        self.assertEqual(result["robots"]["ur10e"]["tcp_offset_source"], "zero_assumed")
        np.testing.assert_allclose(result["robots"]["ur5e"]["tcp_offset"], pose_matrix(args.ur5e_tcp_offset))
        np.testing.assert_allclose(result["robots"]["ur10e"]["qd"], np.arange(6) * .01)
        self.assertTrue(result["calibration"]["ur10e_output_correction"])
        np.testing.assert_allclose(result["bases"]["ur10e"],
                                   result["calibration"]["poses"]["ur10e"] @ model_from_controller_base("ur10e"))
        self.assertGreater(result["robots"]["ur5e"]["robot_timestamp"], 10.)
        self.assertTrue(all(isinstance(call.args[0], str) for call in callback.call_args_list))
        rig.assert_closed(self)

    def test_capture_records_explicitly_disabled_output_mapping(self):
        rig = AcquisitionRig()
        args = config()
        args.ur10e_output_correction = False
        with rig.patches():
            result = module.capture_initial_state(args)
        self.assertFalse(result["calibration"]["ur10e_output_correction"])
        np.testing.assert_allclose(result["bases"]["ur10e"],
                                   result["calibration"]["poses"]["ur10e"])
        rig.assert_closed(self)

    def test_execute_rejected_before_any_hardware_or_context_access(self):
        args = config()
        args.execute = True
        with patch.object(module.mp, "get_context") as context:
            with self.assertRaisesRegex(ValueError, "receive-only"):
                module.capture_initial_state(args)
            context.assert_not_called()

    def test_precancelled_capture_does_not_open_anything(self):
        stop = threading.Event()
        stop.set()
        with patch.object(module.mp, "get_context") as context:
            with self.assertRaisesRegex(ControlFault, "cancelled"):
                module.capture_initial_state(config(), stop_event=stop)
            context.assert_not_called()

    def test_frozen_robot_timestamp_cannot_be_captured(self):
        rig = AcquisitionRig(frozen=True)
        with rig.patches(), self.assertRaisesRegex(ControlFault, "timed out"):
            module.capture_initial_state(config())
        rig.assert_closed(self)

    def test_checked_reprojection_rejection_is_respected_until_timeout(self):
        rig = AcquisitionRig()
        args = config()
        args.check_reprojection = True
        with rig.patches(), self.assertRaisesRegex(ControlFault, "reprojection error"):
            module.capture_initial_state(args)
        rig.assert_closed(self)

    def test_external_cancellation_closes_worker_and_receivers(self):
        rig = AcquisitionRig()
        caller_stop = threading.Event()
        rig.on_wait = caller_stop.set
        with rig.patches(), self.assertRaisesRegex(ControlFault, "cancelled"):
            module.capture_initial_state(config(), stop_event=caller_stop)
        rig.assert_closed(self)

    def test_cancellation_during_shutdown_does_not_start_simulation(self):
        rig = AcquisitionRig()
        caller_stop = threading.Event()

        def report(message):
            if "initial state captured" in message:
                caller_stop.set()

        with rig.patches(), self.assertRaisesRegex(ControlFault, "cancelled"):
            module.capture_initial_state(config(), on_status=report, stop_event=caller_stop)
        rig.assert_closed(self)

    def test_worker_stop_cannot_return_partial_snapshot(self):
        rig = AcquisitionRig()
        rig.on_wait = lambda: setattr(rig, "stopped", True)
        with rig.patches(), self.assertRaisesRegex(ControlFault, "camera capture stopped"):
            module.capture_initial_state(config())
        rig.assert_closed(self)

    def test_worker_start_failure_and_stubborn_worker_cleanup(self):
        rig = AcquisitionRig(failing_start=True)
        with rig.patches(), self.assertRaisesRegex(RuntimeError, "startup failed"):
            module.capture_initial_state(config())
        rig.assert_closed(self)
        rig.worker.join.assert_not_called()
        rig = AcquisitionRig(stubborn=True)
        with rig.patches():
            module.capture_initial_state(config())
        rig.worker.terminate.assert_called_once()
        rig.assert_closed(self)

    def test_still_running_camera_never_hands_snapshot_to_simulation(self):
        rig = AcquisitionRig(stubborn=True)
        rig.worker.terminate.side_effect = None
        rig.worker.kill.side_effect = None
        with rig.patches(), self.assertRaisesRegex(ControlFault, "simulation will not start"):
            module.capture_initial_state(config())
        rig.worker.kill.assert_called_once()
        rig.assert_closed(self)

    def test_only_one_advancing_robot_is_not_a_valid_pair(self):
        rig = AcquisitionRig()
        rig.receivers[1].getTimestamp.side_effect = None
        rig.receivers[1].getTimestamp.return_value = 123.
        args = config()
        args.stable_frames = 1
        with rig.patches(), self.assertRaisesRegex(ControlFault, "timed out"):
            module.capture_initial_state(args)
        rig.assert_closed(self)

    def test_second_connection_failure_disconnects_first(self):
        rig = AcquisitionRig()
        rig.receive_factory.side_effect = [rig.receivers[0], RuntimeError("no route")]
        with rig.patches(), self.assertRaisesRegex(RuntimeError, "no route"):
            module.capture_initial_state(config())
        rig.receivers[0].disconnect.assert_called_once()
        rig.receivers[1].disconnect.assert_not_called()
        rig.worker.start.assert_not_called()
        rig.control_factory.assert_not_called()
        self.assertTrue(all(queue.closed for queue in rig.queues))

    def test_callback_failure_also_closes_resources(self):
        rig = AcquisitionRig()
        callback = Mock(side_effect=[None, None, RuntimeError("UI closed")])
        with rig.patches(), self.assertRaisesRegex(RuntimeError, "UI closed"):
            module.capture_initial_state(config(), on_status=callback)
        rig.assert_closed(self)


if __name__ == "__main__":
    unittest.main()
