"""Evaluation recording controls with no robot-command or hardware access.

Only the separate recording callback receives requests. Authoritative recorder
status, including request acknowledgements, controls every enabled action.
Tk is imported lazily so importing the GUI remains safe without a display.
"""

import time
import uuid


class EvaluationPanel:
    """A small data-only panel embedded in the live or simulation target GUI."""

    def __init__(self, parent, *, recording_command=None):
        import tkinter as tk
        from tkinter import ttk

        self._recording_command = recording_command
        self._status = {"state": "idle", "can_start": False, "can_stop": False}
        self._received_status = False
        self._pending = None
        self._request_error = ""
        self.frame = ttk.Frame(parent, padding=(0, 8))
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        ttk.Label(self.frame, text="Cooperative evaluation", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self.frame, text="Trial ID (optional; blank uses an automatic ID)", wraplength=480).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._trial_id = tk.StringVar(parent, value="")
        self._trial_entry = ttk.Entry(self.frame, textvariable=self._trial_id)
        self._trial_entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        self._start_button = ttk.Button(self.frame, text="Start evaluation", command=lambda: self._request("start"))
        self._start_button.grid(row=3, column=0, sticky="ew", padx=(0, 8))
        self._stop_button = ttk.Button(self.frame, text="Stop & export", command=lambda: self._request("stop"))
        self._stop_button.grid(row=3, column=1, sticky="ew")
        self._table_start_button = ttk.Button(self.frame, text="Table movement start",
                                              command=lambda: self._request("table_start"))
        self._table_start_button.grid(row=4, column=0, sticky="ew", padx=(0, 8), pady=(4, 0))
        self._table_stop_button = ttk.Button(self.frame, text="Table movement end",
                                             command=lambda: self._request("table_stop"))
        self._table_stop_button.grid(row=4, column=1, sticky="ew", pady=(4, 0))
        ttk.Label(self.frame, text="Evaluation target: nominal relative flange target, frozen at evaluation start.\n"
                  "These controls only record data and mark table movement intervals.",
                  wraplength=480, justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._status_text = tk.StringVar(parent, value="")
        ttk.Label(self.frame, textvariable=self._status_text, wraplength=480, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._refresh()

    def set_status(self, status):
        """Apply only a new recorder packet; an unrelated status cannot ack."""
        if not isinstance(status, dict):
            return
        self._status = dict(status)
        self._received_status = True
        if self._pending is not None and status.get("request_id") == self._pending["request_id"]:
            self._pending = None
            self._request_error = ""
        self._refresh()

    def _allowed(self, action):
        if (self._recording_command is None or not self._received_status
                or self._pending is not None):
            return False
        state = self._status.get("state", "idle")
        active = bool(self._status.get("active", state == "recording"))
        if state == "saving":
            return False
        if action == "start":
            return (not active and not self._status.get("busy", False)
                    and bool(self._status.get("can_start", state in ("idle", "saved", "error"))))
        if action == "stop":
            return bool(self._status.get("can_stop", active))
        if action in ("table_start", "table_stop"):
            moving = bool(self._status.get("table_moving", False))
            return active and state == "recording" and (not moving if action == "table_start" else moving)
        return False

    def _request(self, action):
        if not self._allowed(action):
            return
        request = {"type": "evaluation", "action": action, "request_id": uuid.uuid4().hex,
                   "requested_at": time.monotonic()}
        if action == "start":
            request["trial_id"] = self._trial_id.get().strip()
        self._pending = request
        self._request_error = ""
        self._refresh()
        try:
            self._recording_command(request)
        except Exception as exc:
            # Queue failure affects this data request only: do not stop, arm,
            # change a target, or pretend the recorder accepted the request.
            self._pending = None
            self._request_error = "Evaluation request failed: " + str(exc)
            self._refresh()

    def _refresh(self):
        for action, widget in (("start", self._start_button), ("stop", self._stop_button),
                               ("table_start", self._table_start_button), ("table_stop", self._table_stop_button)):
            widget.configure(state="normal" if self._allowed(action) else "disabled")
        self._trial_entry.configure(state="normal" if self._allowed("start") else "disabled")
        state = self._status.get("state", "idle")
        if self._recording_command is None:
            text = "Evaluation recording is unavailable in this viewer."
        elif not self._received_status:
            text = "Waiting for evaluation recorder status."
        else:
            labels = {"idle": "Evaluation ready", "recording": "EVALUATION RECORDING",
                      "saving": "Exporting evaluation", "saved": "Evaluation exported", "error": "Evaluation error"}
            text = labels.get(state, str(state))
            if self._status.get("table_moving"):
                text += " | Table movement marked active"
            if self._status.get("incomplete"):
                text += " | INCOMPLETE"
            if self._status.get("error"):
                text += "\n" + str(self._status["error"])
        if self._pending is not None:
            pending = {"start": "start evaluation", "stop": "stop and export",
                       "table_start": "mark table movement start", "table_stop": "mark table movement end"}
            text += "\nWaiting for recorder to " + pending[self._pending["action"]] + "..."
        for key, label in (("trial_id", "Trial"), ("session_id", "Session"),
                           ("raw_path", "Raw"), ("csv_path", "CSV")):
            if self._status.get(key):
                text += "\n" + label + ": " + str(self._status[key])
        if self._request_error:
            text += "\n" + self._request_error
        self._status_text.set(text)
