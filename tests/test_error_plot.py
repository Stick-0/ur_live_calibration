"""Position-error history and window lifecycle without Tk display or hardware."""

from copy import deepcopy
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from urcal.control.error_plot import PositionErrorHistory, PositionErrorWindow


def sample(at=10., **changes):
    result = {"at": at, "valid": True, "follower": "ur5e", "error_m": [.003, -.004, .012],
              "phase": "tracking", "target_source": "tracking_accepted_relative",
              "reason": "", "clock": "monotonic"}
    result.update(changes)
    return result


class PositionErrorHistoryTests(unittest.TestCase):
    def test_signed_xyz_and_norm_are_mm_and_packet_is_not_mutated(self):
        history = PositionErrorHistory()
        packet = sample(20.)
        original = deepcopy(packet)
        self.assertTrue(history.update(packet, received_at=20.))
        packet["error_m"][0] = 100.
        data = history.view(now=20.1)
        self.assertEqual(history.points[0].error_mm, (.003 * 1000, -.004 * 1000, .012 * 1000))
        self.assertEqual(data["norm_mm"], 13.)
        self.assertTrue(data["fresh"])
        self.assertEqual(original["error_m"], [.003, -.004, .012])
        data["phase"] = "changed"
        self.assertEqual(history.view(now=20.1)["phase"], "tracking")

    def test_duplicate_timestamp_does_not_add_replace_or_refresh_value(self):
        history = PositionErrorHistory()
        history.update(sample(), received_at=20.)
        for wall in (20.5, 21., 22.):
            self.assertFalse(history.update(sample(error_m=[1., 1., 1.]), received_at=wall))
        self.assertEqual(len(history.points), 1)
        data = history.view(now=22.)
        self.assertFalse(data["fresh"])
        self.assertEqual(data["error_mm"], (3., -4., 12.))
        self.assertIn("STALE", data["status"])

    def test_invalid_and_recovered_samples_never_connect(self):
        history = PositionErrorHistory()
        history.update(sample(1.), received_at=1.)
        history.update(sample(2., valid=False, reason="camera state expired"), received_at=2.)
        data = history.view(now=2.)
        self.assertFalse(data["valid"])
        self.assertFalse(data["fresh"])
        self.assertEqual(data["status"], "camera state expired")
        history.update(sample(3.), received_at=3.)
        self.assertNotEqual(history.points[0].segment, history.points[1].segment)

    def test_phase_and_target_source_changes_break_trace(self):
        history = PositionErrorHistory()
        for at, phase, source in ((1., "preview", "preview_nominal_relative"),
                                  (1.1, "alignment", "initial_alignment_endpoint"),
                                  (1.2, "tracking", "tracking_accepted_relative"),
                                  (1.3, "tracking", "preview_nominal_relative")):
            history.update(sample(at, phase=phase, target_source=source), received_at=at)
        self.assertEqual(len({point.segment for point in history.points}), 4)

    def test_duplicate_state_change_cannot_relabel_old_value_as_new_target(self):
        history = PositionErrorHistory()
        history.update(sample(), received_at=10.)
        history.update(sample(target_source="initial_alignment_endpoint"), received_at=10.1)
        self.assertFalse(history.view(now=10.1)["valid"])
        self.assertEqual(len(history.points), 1)
        history.update(sample(10.2, target_source="initial_alignment_endpoint"), received_at=10.2)
        self.assertTrue(history.view(now=10.2)["valid"])
        self.assertNotEqual(history.points[0].segment, history.points[-1].segment)

    def test_missing_packets_expire_values_and_reception_gap_breaks_trace(self):
        history = PositionErrorHistory()
        history.update(sample(10.), received_at=10.)
        self.assertFalse(history.view(now=11.1)["fresh"])
        history.update(sample(11.2), received_at=11.2)
        self.assertNotEqual(history.points[0].segment, history.points[-1].segment)
        self.assertEqual(history.view(now=50.)["points"], ())

    def test_simulation_pause_holds_time_and_does_not_forge_samples(self):
        history = PositionErrorHistory(simulation=True)
        history.update(sample(.5, clock="simulation"), received_at=10.)
        for wall in (10.1, 11., 12.):
            history.update(sample(.5, clock="simulation", phase="paused"), received_at=wall)
        data = history.view(now=12.)
        self.assertEqual(len(history.points), 1)
        self.assertEqual(data["end_at"], .5)
        self.assertTrue(data["valid"])
        self.assertFalse(data["fresh"])
        self.assertTrue(data["time_held"])
        self.assertIn("paused", data["status"])
        history.update(sample(.6, clock="simulation"), received_at=12.1)
        self.assertTrue(history.view(now=12.1)["fresh"])
        self.assertNotEqual(history.points[0].segment, history.points[-1].segment)

    def test_simulation_pause_does_not_relabel_changed_target(self):
        history = PositionErrorHistory(simulation=True)
        history.update(sample(.5, clock="simulation"), received_at=10.)
        history.update(sample(.5, clock="simulation", phase="paused", target_source="other"), received_at=10.1)
        self.assertFalse(history.view(now=10.1)["valid"])

    def test_reset_episode_and_simulation_time_reverse_clear_history(self):
        for change in ({"episode": 1, "at": 0.}, {"at": 0.}, {"clock": "monotonic", "at": 15.}):
            history = PositionErrorHistory()
            history.update(sample(1., clock="simulation"), received_at=10.)
            packet = sample(2., clock="simulation")
            packet.update(change)
            self.assertTrue(history.update(packet, received_at=max(11., packet["at"])))
            self.assertEqual(len(history.points), 1)
            self.assertEqual(history.points[0].at, packet["at"])

    def test_live_out_of_order_invalidates_without_overwriting_or_resetting(self):
        history = PositionErrorHistory()
        history.update(sample(10.), received_at=10.)
        self.assertFalse(history.update(sample(9.), received_at=10.1))
        self.assertEqual(history.points[0].at, 10.)
        self.assertIn("Out-of-order", history.view(now=10.1)["status"])
        history.update(sample(10.2), received_at=10.2)
        self.assertNotEqual(history.points[0].segment, history.points[-1].segment)

    def test_queued_old_live_sample_is_stale_immediately_and_axis_is_current(self):
        history = PositionErrorHistory()
        history.update(sample(10.), received_at=15.)
        data = history.view(now=15.)
        self.assertEqual(data["receipt_age"], 0.)
        self.assertEqual(data["sample_age"], 5.)
        self.assertFalse(data["fresh"])
        self.assertEqual(data["end_at"], 15.)
        self.assertIn("STALE", data["status"])

    def test_future_live_timestamp_is_invalid(self):
        history = PositionErrorHistory()
        self.assertFalse(history.update(sample(11.), received_at=10.))
        self.assertFalse(history.view(now=10.)["valid"])
        self.assertIn("future", history.view(now=10.)["status"])
        self.assertEqual(history.points, ())

    def test_time_window_and_hard_capacity_both_bound_memory(self):
        history = PositionErrorHistory(max_points=5)
        for index in range(100):
            history.update(sample(index / 10.), received_at=index / 10.)
        self.assertEqual(len(history.points), 5)
        history.update(sample(50.), received_at=50.)
        self.assertEqual(len(history.points), 1)
        history = PositionErrorHistory()
        for index in range(3101):
            history.update(sample(index / 1000.), received_at=index / 1000.)
        self.assertEqual(len(history.points), 3000)

    def test_malformed_samples_display_unavailable_and_make_a_gap(self):
        malformed = (None, {}, sample(valid=1), sample(at=float("nan")), sample(at=True),
                     sample(error_m=[1., 2.]), sample(error_m=[1., float("inf"), 3.]),
                     sample(error_m=[1e308, 2., 3.]), sample(phase=None), sample(episode=True))
        for packet in malformed:
            with self.subTest(packet=packet):
                history = PositionErrorHistory()
                history.update(sample(1.), received_at=1.)
                self.assertFalse(history.update(packet, received_at=1.1))
                self.assertFalse(history.view(now=1.1)["valid"])
                history.update(sample(1.2), received_at=1.2)
                self.assertNotEqual(history.points[0].segment, history.points[-1].segment)


class FakeWidget:
    def __init__(self, parent=None, **kwargs):
        self.parent, self.calls, self.protocols = parent, [], {}
        self.options = dict(kwargs)
        self.hidden = False

    def grid(self, **kwargs):
        pass

    def bind(self, *args):
        pass

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def invoke(self):
        if self.options.get("state") != "disabled":
            self.options["command"]()

    def title(self, text):
        self.title_text = text

    def geometry(self, text):
        pass

    def minsize(self, *args):
        pass

    def protocol(self, name, callback):
        self.protocols[name] = callback

    def columnconfigure(self, *args, **kwargs):
        pass

    def rowconfigure(self, *args, **kwargs):
        pass

    def winfo_width(self):
        return 900

    def winfo_height(self):
        return 350

    def withdraw(self):
        self.hidden = True

    def deiconify(self):
        self.hidden = False

    def lift(self):
        pass

    def delete(self, *args):
        self.calls.append(("delete", args, {}))

    def create_line(self, *args, **kwargs):
        self.calls.append(("line", args, kwargs))

    def create_text(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))

    def create_oval(self, *args, **kwargs):
        self.calls.append(("oval", args, kwargs))


class FakeStringVar:
    def __init__(self, parent=None, value=""):
        self.value = value

    def set(self, value):
        self.value = value


class PositionErrorWindowTests(unittest.TestCase):
    def setUp(self):
        ttk = SimpleNamespace(Label=FakeWidget, Frame=FakeWidget, Button=FakeWidget)
        tk = SimpleNamespace(Toplevel=FakeWidget, Canvas=FakeWidget, StringVar=FakeStringVar, ttk=ttk)
        self.tk_patch = patch.dict(sys.modules, {"tkinter": tk, "tkinter.ttk": ttk})
        self.tk_patch.start()
        self.addCleanup(self.tk_patch.stop)
        self.clock_patch = patch("urcal.control.error_plot.time.monotonic", return_value=10.)
        self.clock = self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def test_window_opens_then_close_only_hides_and_show_restores(self):
        parent = object()  # No stop/destroy/command methods exist.
        window = PositionErrorWindow(parent)
        self.assertFalse(window.window.hidden)
        self.assertIs(window.window.parent, parent)
        self.clock.return_value = 10.2
        window.update(sample())
        window.window.protocols["WM_DELETE_WINDOW"]()
        self.assertTrue(window.window.hidden)
        self.clock.return_value = 10.4
        window.update(sample(10.4))
        self.assertEqual(len(window.history.points), 2)
        window.show()
        self.assertFalse(window.window.hidden)
        self.assertIn("distance to target", window._heading.value)
        self.assertEqual("Position error: 13.00 mm", window._values.value)

    def test_render_throttle_staleness_and_single_distance_trace(self):
        window = PositionErrorWindow(object())
        window.update(sample())
        drawn = len(window.canvas.calls)
        self.clock.return_value = 10.05
        window.refresh()
        self.assertEqual(len(window.canvas.calls), drawn)
        self.clock.return_value = 10.2
        window.update(sample(10.2))
        colored = [call for call in window.canvas.calls if call[0] in ("line", "oval")
                   and call[2].get("fill") in ("#ff657a", "#64d68e", "#68aaff", "#facc15", "#2563eb")]
        self.assertEqual({call[2]["fill"] for call in colored}, {"#2563eb"})
        self.assertTrue(all(math.isfinite(number) for call in colored for number in call[1]))
        self.clock.return_value = 11.3
        window.refresh()
        self.assertIn("LAST SAMPLE", window._values.value)
        self.assertIn("STALE", window._status.value)
        self.clock.return_value = 11.5
        window.update(sample(11.5, valid=False, reason="robot state stale"))
        self.assertEqual("Position error: — mm", window._values.value)
        self.assertIn("robot state stale", window._status.value)

    def test_invalid_period_is_not_connected_by_colored_canvas_line(self):
        window = PositionErrorWindow(object())
        window.update(sample(10.))
        self.clock.return_value = 10.2
        window.update(sample(10.2, valid=False, reason="no goal"))
        self.clock.return_value = 10.4
        window.canvas.calls.clear()
        window.update(sample(10.4))
        color_lines = [call for call in window.canvas.calls if call[0] == "line"
                       and call[2].get("fill") in ("#ff657a", "#64d68e", "#68aaff", "#facc15", "#2563eb")]
        self.assertEqual(color_lines, [])

    def test_recording_is_disabled_without_a_producer(self):
        window = PositionErrorWindow(object())
        self.assertEqual(window._record_start.options["state"], "disabled")
        self.assertEqual(window._record_stop.options["state"], "disabled")
        self.assertIn("unavailable", window._recording_text.value)
        window._request_recording("start")
        self.assertIsNone(window._recording_pending)

    def test_recording_requests_require_ack_and_save_completion_before_next_start(self):
        send = Mock()
        window = PositionErrorWindow(object(), recording_command=send)
        window._record_start.invoke()
        start = send.call_args.args[0]
        self.assertEqual(start["action"], "start")
        self.assertTrue(start["request_id"])
        self.assertEqual(window._record_start.options["state"], "disabled")
        self.assertEqual(window._record_stop.options["state"], "disabled")
        window._request_recording("start")
        window.set_recording_status({"state": "idle", "request_id": "old"})
        window._request_recording("stop")
        send.assert_called_once_with(start)
        window.set_recording_status({"state": "recording", "request_id": start["request_id"],
                                     "path": "/tmp/segment.csv", "rows": 1200, "elapsed_s": 60.5})
        self.assertEqual(window._record_stop.options["state"], "normal")
        self.assertIn("1200 rows", window._recording_text.value)
        self.assertIn("60.50 s", window._recording_text.value)
        window._record_stop.invoke()
        stop = send.call_args.args[0]
        self.assertEqual(stop["action"], "stop")
        self.assertNotEqual(stop["request_id"], start["request_id"])
        window.set_recording_status({"state": "saving", "request_id": stop["request_id"]})
        self.assertEqual(window._record_start.options["state"], "disabled")
        self.assertEqual(window._record_stop.options["state"], "disabled")
        window.set_recording_status({"state": "saved", "request_id": stop["request_id"],
                                     "path": "/tmp/segment.csv", "rows": 1200, "elapsed_s": 60.5})
        self.assertEqual(window._record_start.options["state"], "normal")
        self.assertIn("CSV saved", window._recording_text.value)
        self.assertIn("/tmp/segment.csv", window._recording_text.value)

    def test_hiding_and_redrawing_never_stops_or_feeds_the_recorder(self):
        send = Mock()
        window = PositionErrorWindow(object(), recording_command=send)
        window._record_start.invoke()
        command = send.call_args.args[0]
        window.set_recording_status({"state": "recording", "request_id": command["request_id"]})
        window.close()
        for at in (10.1, 10.2, 45.):
            self.clock.return_value = at
            window.update(sample(at, error_m=[.1, 0., 0.]))
        window.set_recording_status({"state": "recording", "request_id": command["request_id"], "rows": 875})
        self.assertIn("875 rows", window._recording_text.value)
        window.show()
        self.assertIn("100.00 mm", window._values.value)
        send.assert_called_once_with(command)
        self.assertEqual(window._record_stop.options["state"], "normal")

    def test_queue_error_can_retry_without_changing_recording_or_propagating(self):
        send = Mock(side_effect=RuntimeError("queue full"))
        window = PositionErrorWindow(object(), recording_command=send)
        window._record_start.invoke()
        self.assertIsNone(window._recording_pending)
        self.assertIn("queue full", window._recording_text.value)
        self.assertEqual(window._record_start.options["state"], "normal")
        send.side_effect = None
        window._record_start.invoke()
        command = send.call_args.args[0]
        window.set_recording_status({"state": "recording", "request_id": command["request_id"]})
        send.side_effect = RuntimeError("queue full")
        window._record_stop.invoke()
        self.assertEqual(window._record_stop.options["state"], "normal")
        self.assertEqual(window._record_start.options["state"], "disabled")
        self.assertIn("RECORDING", window._recording_text.value)

    def test_writer_error_displays_incomplete_file_and_respects_busy_retry_flag(self):
        window = PositionErrorWindow(object(), recording_command=Mock())
        error = {"state": "error", "can_start": False, "can_stop": False,
                 "error": "write failed", "incomplete": True, "path": "/tmp/partial.csv"}
        window.set_recording_status(error)
        self.assertEqual(window._record_start.options["state"], "disabled")
        self.assertIn("INCOMPLETE", window._recording_text.value)
        self.assertIn("write failed", window._recording_text.value)
        window.set_recording_status({**error, "can_start": True})
        self.assertEqual(window._record_start.options["state"], "normal")


if __name__ == "__main__":
    unittest.main()
