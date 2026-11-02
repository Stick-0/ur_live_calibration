"""Read-only follower position-error history and an optional Tk graph window.

No robot or motion-command queue is referenced here. Optional recording buttons
send requests to a separate data recorder; graph redraws never supply its data.
The supplied error is target minus current position in the follower's base axes.
Tk is imported only when a window is constructed; the history is headless.
"""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral
import time
import uuid


_Y_MAX_MM = 50.


@dataclass(frozen=True)
class ErrorPoint:
    at: float
    error_mm: tuple
    norm_mm: float
    segment: int


class PositionErrorHistory:
    """Bounded samples without bridging invalid, changed, or missing states.

    ``at`` belongs to the packet clock. ``received_at`` and ``view(now=...)``
    use a local monotonic wall clock, independently of simulation time.
    Repeated timestamps do not add points or refresh measured values. A paused
    simulation keeps its time axis fixed and explicitly shows its held value.
    """

    def __init__(self, *, window_seconds=30., max_points=3000, stale_after=1.,
                 follower="ur5e", simulation=False):
        if (not math.isfinite(window_seconds) or window_seconds <= 0
                or not math.isfinite(stale_after) or stale_after <= 0):
            raise ValueError("history duration and freshness must be positive")
        if not isinstance(max_points, Integral) or isinstance(max_points, bool) or max_points < 1:
            raise ValueError("max_points must be a positive integer")
        self.window_seconds, self.stale_after = float(window_seconds), float(stale_after)
        self._points = deque(maxlen=int(max_points))
        self._segment = 0
        self._identity = None
        self._last_at = self._last_received = None
        self._signature = None
        self._latest = {"at": None, "valid": False, "follower": follower,
                        "phase": "waiting", "target_source": "unavailable",
                        "reason": "Waiting for position-error samples",
                        "clock": "simulation" if simulation else "monotonic", "episode": 0}

    @property
    def points(self):
        return tuple(self._points)

    @staticmethod
    def _finite(value, name):
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(name + " must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(name + " must be finite")
        return result

    @classmethod
    def _sample(cls, sample):
        if not isinstance(sample, dict):
            raise ValueError("sample must be a mapping")
        if not isinstance(sample.get("valid"), bool):
            raise ValueError("valid must be boolean")
        result = {"at": cls._finite(sample.get("at"), "sample time"), "valid": sample["valid"]}
        for key, default in (("follower", "ur5e"), ("phase", "unknown"),
                             ("target_source", "unavailable"), ("reason", "")):
            result[key] = sample.get(key, default)
            if not isinstance(result[key], str):
                raise ValueError(key + " must be text")
        if not result["follower"]:
            raise ValueError("follower must be nonempty")
        result["clock"] = sample.get("clock", "monotonic")
        if result["clock"] not in ("monotonic", "simulation"):
            raise ValueError("clock must be monotonic or simulation")
        result["episode"] = sample.get("episode", 0)
        if (not isinstance(result["episode"], Integral) or isinstance(result["episode"], bool)
                or result["episode"] < 0):
            raise ValueError("episode must be a nonnegative integer")
        if result["valid"]:
            try:
                values = tuple(sample["error_m"])
            except (KeyError, TypeError) as exc:
                raise ValueError("error_m must contain three finite values") from exc
            if len(values) != 3:
                raise ValueError("error_m must contain three finite values")
            error_mm = tuple(cls._finite(value, "error component") * 1000. for value in values)
            norm_mm = math.hypot(*error_mm)
            if not all(math.isfinite(value) for value in (*error_mm, norm_mm)):
                raise ValueError("error magnitude is not finite")
            result.update(error_mm=error_mm, norm_mm=norm_mm)
        return result

    def update(self, sample, *, received_at=None):
        """Accept a unique-time sample; return whether a graph point was added.

        Bad display packets invalidate the displayed value instead of raising
        into the GUI loop. Clock reversal is a simulation reset; reversal of a
        live timestamp is rejected. Explicit episode changes reset either mode.
        """
        now = time.monotonic() if received_at is None else self._finite(received_at, "receipt time")
        try:
            candidate = self._sample(sample)
            if candidate["clock"] == "monotonic" and candidate["at"] > now + 1e-6:
                raise ValueError("live sample time is in the future")
        except ValueError as exc:
            if self._latest["valid"]:
                self._segment += 1
            self._latest = {**self._latest, "valid": False, "reason": "Invalid graph sample: " + str(exc)}
            self._signature = None
            return False
        identity = (candidate["clock"], candidate["episode"], candidate["follower"])
        reset = identity != self._identity
        if not reset and self._last_at is not None and candidate["at"] < self._last_at:
            if candidate["clock"] == "simulation":
                reset = True
            else:
                self._segment += 1
                self._latest = {**self._latest, "valid": False,
                                "reason": "Out-of-order position-error sample"}
                self._signature = None
                return False
        if reset:
            self._points.clear()
            self._segment += 1
            self._last_at = self._last_received = None
            self._signature = None
            self._identity = identity
        signature = (candidate["valid"], candidate["phase"], candidate["target_source"])
        changed = signature != self._signature
        if changed or (self._last_received is not None and now - self._last_received > self.stale_after):
            self._segment += 1
        self._signature = signature
        if self._last_at is not None and candidate["at"] == self._last_at:
            # Metadata changes are visible even when paused simulation time
            # cannot advance. A duplicate never replaces an error value.
            same_source = candidate["target_source"] == self._latest["target_source"]
            self._latest.update({key: candidate[key] for key in ("phase", "target_source", "reason")})
            paused = (candidate["clock"] == "simulation" and candidate["phase"] == "paused"
                      and same_source)
            if not candidate["valid"] or (changed and not paused):
                self._latest["valid"] = False
                if candidate["valid"]:
                    self._latest["reason"] = "Waiting for a new sample after state change"
            return False
        self._last_at, self._last_received = candidate["at"], now
        self._latest = candidate
        while self._points and self._points[0].at < candidate["at"] - self.window_seconds:
            self._points.popleft()
        if not candidate["valid"]:
            return False
        self._points.append(ErrorPoint(candidate["at"], candidate["error_mm"],
                                       candidate["norm_mm"], self._segment))
        return True

    def view(self, *, now=None):
        """Return a detached display snapshot; never change sample freshness."""
        wall = time.monotonic() if now is None else self._finite(now, "display time")
        result = self._latest.copy()
        age = None if self._last_received is None else max(0., wall - self._last_received)
        sample_age = (None if self._last_at is None or result["clock"] == "simulation"
                      else max(0., wall - self._last_at))
        result["fresh"] = bool(result["valid"] and age is not None and age <= self.stale_after
                               and (sample_age is None or sample_age <= self.stale_after))
        result["receipt_age"], result["sample_age"] = age, sample_age
        result["time_held"] = (result["clock"] == "simulation" and age is not None
                               and (result["phase"] == "paused" or age > .2))
        result["status"] = (result["reason"] or "Position error unavailable") if not result["valid"] else (
            "Simulation time held — last sample" if result["time_held"] else
            "Live sample" if result["fresh"] else "STALE — no new position-error sample")
        if result["valid"] and not result["fresh"] and result["clock"] == "simulation":
            result["status"] = "Simulation time held / no new sample — last value"
        if result["valid"] and result["clock"] == "simulation" and result["phase"] == "paused":
            result["status"] = "Simulation paused — last sample"
        result["end_at"] = self._last_at
        if self._last_at is not None and result["clock"] == "monotonic":
            result["end_at"] = max(self._last_at, wall)
        earliest = float("-inf") if result["end_at"] is None else result["end_at"] - self.window_seconds
        result["points"] = tuple(point for point in self._points if point.at >= earliest)
        return result


class PositionErrorWindow:
    """A separate, automatically opened graph. Closing only hides this window.

    The GUI owner calls ``refresh`` on its normal UI tick, including ticks with
    no new telemetry. Rendering is throttled to 10 Hz independently of sampling.
    """

    def __init__(self, parent, follower="ur5e", simulation=False, *, recording_command=None):
        import tkinter as tk
        from tkinter import ttk

        self.history = PositionErrorHistory(follower=follower, simulation=simulation)
        self._recording_command = recording_command
        self._recording = {"state": "idle" if recording_command is not None else "unavailable"}
        self._recording_pending = None
        self._recording_error = ""
        self.window = tk.Toplevel(parent)
        self.window.title("Follower TCP position error" + (" — SIMULATION" if simulation else ""))
        self.window.geometry("900x620")
        self.window.minsize(560, 450)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        self._heading = tk.StringVar(self.window)
        self._values = tk.StringVar(self.window, value="Waiting for position-error samples...")
        self._status = tk.StringVar(self.window)
        heading = ttk.Label(self.window, textvariable=self._heading, font=("TkDefaultFont", 12, "bold"),
                            wraplength=850, justify="left")
        heading.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        values = ttk.Label(self.window, textvariable=self._values, font=("TkFixedFont", 11),
                           wraplength=850, justify="left")
        values.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
        self.canvas = tk.Canvas(self.window, background="#ffffff", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=10)
        status = ttk.Label(self.window, textvariable=self._status, wraplength=850, justify="left")
        status.grid(row=3, column=0, sticky="ew", padx=12, pady=8)
        controls = ttk.Frame(self.window)
        controls.grid(row=4, column=0, sticky="w", padx=12)
        self._record_start = ttk.Button(controls, text="Start recording",
                                       command=lambda: self._request_recording("start"))
        self._record_start.grid(row=0, column=0, padx=(0, 8))
        self._record_stop = ttk.Button(controls, text="Stop & save",
                                      command=lambda: self._request_recording("stop"))
        self._record_stop.grid(row=0, column=1)
        self._recording_text = tk.StringVar(self.window)
        recording = ttk.Label(self.window, textvariable=self._recording_text,
                              wraplength=850, justify="left")
        recording.grid(row=5, column=0, sticky="ew", padx=12, pady=(6, 4))
        note = ttk.Label(self.window, text="Closing this graph keeps recording. Stop & save ends the CSV interval.",
                         wraplength=850, justify="left")
        note.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 10))
        def reflow(event):
            if event.widget is self.window:
                for label in (heading, values, status, recording, note):
                    label.configure(wraplength=max(300, event.width - 24))
        self.window.bind("<Configure>", reflow)
        self._hidden = False
        self._last_draw = float("-inf")
        self._refresh_recording()
        self.refresh()

    def set_recording_status(self, status):
        """Display producer acknowledgements independently of plot throttling."""
        if not isinstance(status, dict):
            return
        self._recording = dict(status)
        if (self._recording_pending is not None
                and status.get("request_id") == self._recording_pending["request_id"]):
            self._recording_pending = None
            self._recording_error = ""
        self._refresh_recording()

    def _request_recording(self, action):
        state = self._recording.get("state", "idle")
        allowed = (self._recording.get("can_start", state in ("idle", "saved", "error"))
                   if action == "start" else self._recording.get("can_stop", state in ("starting", "recording")))
        if self._recording_command is None or self._recording_pending is not None or not allowed:
            return
        command = {"action": action, "request_id": uuid.uuid4().hex}
        self._recording_pending = command
        self._recording_error = ""
        self._refresh_recording()
        try:
            self._recording_command(command)
        except Exception as exc:
            # Optional data recording must never send a robot Stop or escape
            # the Tk callback because its queue or filesystem is unavailable.
            self._recording_pending = None
            self._recording_error = "Recording request failed: " + str(exc)
            self._refresh_recording()

    def _refresh_recording(self):
        state = self._recording.get("state", "idle")
        connected = self._recording_command is not None
        waiting = self._recording_pending is not None
        self._record_start.configure(state="normal" if connected and not waiting
                                     and self._recording.get("can_start", state in ("idle", "saved", "error")) else "disabled")
        self._record_stop.configure(state="normal" if connected and not waiting
                                    and self._recording.get("can_stop", state in ("starting", "recording")) else "disabled")
        if not connected:
            text = "CSV recording is unavailable in this viewer."
        elif waiting:
            action = self._recording_pending["action"]
            text = "Waiting for recorder to " + ("start..." if action == "start" else "stop and save...")
        else:
            labels = {"idle": "CSV ready", "starting": "Opening CSV", "recording": "RECORDING",
                      "saving": "Saving CSV", "saved": "CSV saved", "error": "Recording error"}
            text = labels.get(state, str(state))
            rows = self._recording.get("rows", self._recording.get("samples", 0))
            duration = self._recording.get("elapsed_s", 0.)
            if state != "idle":
                text += f" | {rows} rows | {float(duration):.2f} s"
            if self._recording.get("error"):
                text += " | " + str(self._recording["error"])
            if self._recording.get("incomplete"):
                text += " | INCOMPLETE"
        if self._recording_error:
            text += "\n" + self._recording_error
        if self._recording.get("path"):
            text += "\n" + str(self._recording["path"])
        self._recording_text.set(text)

    def update(self, sample):
        self.history.update(sample)
        self.refresh()

    def show(self):
        self._hidden = False
        self.window.deiconify()
        self.window.lift()
        self._last_draw = float("-inf")
        self.refresh()

    def close(self):
        self._hidden = True
        self.window.withdraw()

    def refresh(self):
        now = time.monotonic()
        if self._hidden or now - self._last_draw < .1:
            return
        self._last_draw = now
        data = self.history.view(now=now)
        follower = data["follower"].upper()
        self._heading.set(f"{follower} TCP position error: distance to target")
        if data["valid"]:
            qualifier = "" if data["fresh"] and not data["time_held"] else "LAST SAMPLE  "
            overflow = " (above plot range)" if data["norm_mm"] > _Y_MAX_MM else ""
            self._values.set(f"{qualifier}Position error: {data['norm_mm']:.2f} mm{overflow}")
        else:
            self._values.set("Position error: — mm")
        phase, source = data["phase"].replace("_", " "), data["target_source"].replace("_", " ")
        self._status.set(f"{phase} | Target: {source}\n{data['status']}")
        self._draw(data)

    def _draw(self, data):
        canvas = self.canvas
        width, height = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
        canvas.delete("all")
        if width < 220 or height < 120:
            return
        left, top, right, bottom = 66., 35., width - 18., height - 42.
        points = data["points"]
        span = _Y_MAX_MM
        end = data["end_at"] or 0.
        start = end - self.history.window_seconds
        xscreen = lambda at: left + (at - start) / self.history.window_seconds * (right - left)
        # Clamp drawing only; history and the numeric readout retain the real error.
        yscreen = lambda value: top + (1. - min(span, max(0., value)) / span) * (bottom - top)
        for index in range(6):
            value = span * (1 - index / 5)
            y = yscreen(value)
            canvas.create_line(left, y, right, y, fill="#94a3b8" if index == 5 else "#e2e8f0")
            canvas.create_text(left - 8, y, text=f"{value:.1f}", fill="#334155", anchor="e")
        for seconds in range(0, 31, 5):
            x = xscreen(end - seconds)
            canvas.create_line(x, top, x, bottom, fill="#e2e8f0")
            canvas.create_text(x, bottom + 16, text=f"−{seconds}" if seconds else "0", fill="#334155")
        canvas.create_text(12, top, text="mm", fill="#334155", anchor="w")
        clock = "simulation" if data["clock"] == "simulation" else "live"
        canvas.create_text((left + right)/2, height - 9, text=f"Seconds relative to current {clock} time", fill="#334155")
        color = "#2563eb"
        canvas.create_text(left, 14, text="Position error", fill=color, anchor="w")
        segment, coordinates = None, []
        def flush():
            if len(coordinates) >= 4:
                canvas.create_line(*coordinates, fill=color, width=2)
            elif coordinates:
                x, y = coordinates
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
        for point in points:
            if point.segment != segment:
                flush()
                coordinates = []
                segment = point.segment
            coordinates.extend((xscreen(point.at), yscreen(point.norm_mm)))
        flush()
