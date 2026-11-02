"""Exposure-time RTDE interpolation with synthetic timestamps and fake receivers."""

import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from urcal.control.joint_history import JointAlignmentUnavailable, JointHistory, JointSampler


class JointHistoryTests(unittest.TestCase):
    def history(self):
        history = JointHistory()
        for index in range(3):
            history.append(1000. + .01 * index, np.full(6, .1 * index),
                           100. + .01 * index, 100.002 + .01 * index)
        return history

    def test_interpolates_capture_time_not_latest_arrival(self):
        history = self.history()
        joints, metadata = history.interpolate(100.016, 100.024, .05, .5)
        np.testing.assert_allclose(joints, np.full(6, .15), atol=1e-11)
        self.assertAlmostEqual(metadata["fraction"], .5)
        self.assertAlmostEqual(metadata["bracket_gap_s"], .01)
        self.assertAlmostEqual(metadata["robot_to_host_offset_s"], -899.999)
        self.assertAlmostEqual(metadata["sdk_read_half_interval_s"], .001)
        self.assertEqual(metadata["buffer_samples"], 3)

    def test_exact_sample_is_allowed_but_missing_brackets_never_use_latest_q(self):
        history = self.history()
        joints, metadata = history.interpolate(100.011, 100.024, .05, .5)
        np.testing.assert_allclose(joints, .1)
        self.assertEqual(metadata["bracket_gap_s"], 0.)
        for exposure in (99.99, 100.023):
            with self.subTest(exposure=exposure), self.assertRaisesRegex(JointAlignmentUnavailable, "bracketing"):
                history.interpolate(exposure, 100.03, .05, .5)

    def test_duplicate_timestamp_keeps_first_clock_observation_and_joint_value(self):
        history = self.history()
        offset, last = history.offset, history.samples[-1]
        self.assertFalse(history.append(last[0], np.full(6, 9.), 110., 110.01))
        self.assertEqual(history.offset, offset)
        self.assertEqual(len(history.samples), 3)
        self.assertEqual(history.samples[-1][3], last[3])
        np.testing.assert_array_equal(history.samples[-1][1], last[1])
        with self.assertRaisesRegex(JointAlignmentUnavailable, "stale"):
            history.interpolate(100.016, 110.01, .05, .5)

    def test_large_gap_old_capture_and_future_capture_are_rejected(self):
        history = self.history()
        with self.assertRaisesRegex(JointAlignmentUnavailable, "gap"):
            history.interpolate(100.016, 100.024, .005, .5)
        with self.assertRaisesRegex(JointAlignmentUnavailable, "stale"):
            history.interpolate(100.016, 100.024, .05, .01)
        with self.assertRaisesRegex(JointAlignmentUnavailable, "invalid exposure"):
            history.interpolate(101., 100.024, .05, .5)

    def test_joint_angle_wrap_interpolates_near_pi_not_zero(self):
        history = JointHistory()
        history.append(1., np.full(6, np.pi - .01), 10., 10.)
        history.append(1.02, np.full(6, -np.pi + .01), 10.02, 10.02)
        joints, _ = history.interpolate(10.01, 10.03, .05, .5)
        np.testing.assert_allclose(joints, np.pi)

    def test_buffer_is_bounded_and_invalid_samples_do_not_change_it(self):
        history = JointHistory(maxlen=3)
        for index in range(10):
            history.append(index, np.zeros(6), index + 10., index + 10.)
        self.assertEqual(len(history.samples), 3)
        self.assertEqual(history.samples[0][0], 7.)
        for stamp, q in ((8., np.zeros(6)), (float("nan"), np.zeros(6)), (10., np.full(6, np.nan)),
                         (10., np.zeros(5))):
            with self.assertRaises(ValueError):
                history.append(stamp, q, 20., 20.)
        self.assertEqual(history.samples[-1][0], 9.)


class JointSamplerTests(unittest.TestCase):
    def sampler(self, receivers, clock):
        with patch("urcal.control.joint_history.threading.Thread"):
            return JointSampler(receivers, threading.Event(), max_age=.2, clock=clock)

    def receiver(self, stamp=1., joints=None):
        receiver = Mock()
        receiver.isConnected.return_value = True
        receiver.getTimestamp.return_value = stamp
        receiver.getActualQ.return_value = np.zeros(6) if joints is None else joints
        return receiver

    def test_timestamp_q_timestamp_bracket_retries_cross_packet_read(self):
        receiver = self.receiver()
        receiver.getTimestamp.side_effect = [1., 1.01, 1.01, 1.01]
        receiver.getActualQ.side_effect = [np.ones(6), np.full(6, 2.)]
        sampler = self.sampler({"ur5e": receiver}, lambda: 10.)
        sampler._sample_once()
        self.assertEqual(sampler.incoherent_samples["ur5e"], 1)
        joints, metadata = sampler.interpolate(10., wait=0.)
        np.testing.assert_array_equal(joints, [np.full(6, 2.)])
        self.assertEqual(metadata["ur5e"]["incoherent_samples_discarded"], 1)

    def test_both_robots_interpolate_independently_to_one_exposure(self):
        now = [100.]
        receivers = {"ur5e": self.receiver(), "ur10e": self.receiver(20.)}
        sampler = self.sampler(receivers, lambda: now[0])
        sampler._sample_once()
        now[0] = 100.02
        for receiver, stamp, q in ((receivers["ur5e"], 1.02, .1), (receivers["ur10e"], 20.02, .2)):
            receiver.getTimestamp.return_value = stamp
            receiver.getActualQ.return_value = np.full(6, q)
        sampler._sample_once()
        joints, metadata = sampler.interpolate(100.01, wait=0.)
        np.testing.assert_allclose(joints, [np.full(6, .05), np.full(6, .1)])
        self.assertEqual(set(metadata), set(receivers))

    def test_cached_timestamp_or_permanent_incoherence_cannot_stay_fresh(self):
        for incoherent in (False, True):
            with self.subTest(incoherent=incoherent):
                now = [10.]
                receiver = self.receiver()
                sampler = self.sampler({"ur5e": receiver}, lambda: now[0])
                sampler._sample_once()
                now[0] = 10.3
                if incoherent:
                    receiver.getTimestamp.side_effect = [1., 1.01] * 3
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    sampler._sample_once()

    def test_missing_bracket_wait_is_bounded_and_fault_or_close_interrupts_wait(self):
        sampler = self.sampler({"ur5e": self.receiver()}, time.monotonic)
        start = time.monotonic()
        with self.assertRaises(JointAlignmentUnavailable):
            sampler.interpolate(start - .01, wait=.01)
        self.assertLess(time.monotonic() - start, .2)
        sampler.error = "fake disconnected"
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            sampler.interpolate(start, wait=.03)
        sampler.error = None
        sampler.close()
        sampler.thread.join.assert_called_once_with(timeout=1.5)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            sampler.interpolate(start, wait=.03)

    def test_background_sampler_fault_is_published_and_thread_stops(self):
        receiver = self.receiver()
        receiver.isConnected.return_value = False
        sampler = JointSampler({"ur5e": receiver}, threading.Event())
        sampler.thread.join(timeout=.2)
        self.assertFalse(sampler.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            sampler.interpolate(time.monotonic(), wait=0.)
        sampler.close()


if __name__ == "__main__":
    unittest.main()
