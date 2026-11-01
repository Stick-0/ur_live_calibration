"""Follower task controls; Tk is loaded only when a panel is created.

The GUI owns no hardware interface. Its process sends target/arm requests and
uses a shared event for Stop, so stopping does not depend on a command queue.
"""

from queue import Empty, Full
import time

import numpy as np

from urcal.control.geometry import facing_follower
from urcal.control.error_plot import PositionErrorWindow
from urcal.control.scene_view import RobotSceneView, task_path


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _triple(values, label):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} needs three finite numbers") from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} needs three finite numbers")
    return array


def build_target_command(mode, gap_cm, min_gap, *,
                         path=None, path_delta_cm=None, path_rotation_degrees=None,
                         path_radius_cm=2., path_duration=10.):
    """Validate UI units and return a plain, serializable target request."""
    if mode != "follow":
        raise ValueError("only follower mode is supported")
    minimum = _number(min_gap, "Minimum gap")
    if minimum <= 0:
        raise ValueError("Minimum gap must be positive")
    gap = _number(gap_cm, "Gap (cm)") / 100.
    if gap < minimum:
        raise ValueError(f"Gap must be at least {minimum * 100:g} cm")
    command = {"type": "target", "gap": gap}
    if path is not None:
        if path not in ("hold", "line", "circle"):
            raise ValueError("Relative path must be hold, line or circle in follow mode")
        delta = _triple([0., 0., -5.] if path_delta_cm is None else path_delta_cm, "Path delta X, Y, Z (cm)") / 100.
        rotation = np.deg2rad(_triple([0., 0., 0.] if path_rotation_degrees is None else path_rotation_degrees,
                                    "Path rotation-vector Rx, Ry, Rz (degrees)"))
        radius = _number(path_radius_cm, "Circle radius (cm)") / 100.
        duration = _number(path_duration, "Path duration (s)")
        if radius <= 0 or duration <= 0:
            raise ValueError("Circle radius and path duration must be positive")
        command.update(path=path, path_delta=np.concatenate((delta, rotation)).tolist(),
                       path_radius=radius, path_duration=duration)
    return command


def research_status_text(status):
    """Format research diagnostics without assuming all states are ready."""
    def number(value, scale=1., suffix=""):
        try:
            return f"{_number(value, 'status') * scale:.1f}{suffix}"
        except ValueError:
            return "--"

    age = status.get("measurement_age")
    if isinstance(age, dict):
        try:
            age = max(_number(value, "measurement age") for value in age.values())
        except (ValueError, TypeError):
            age = None
    text = (f"Progress: {number(status.get('progress'), 100., '%')}   "
            f"Task speed: {number(status.get('task_scale'), 100., '%')}   "
            f"Measurement age: {number(age, 1000., ' ms')}")
    fault = status.get("fault_category", status.get("faultcategory"))
    return text + (f"\nFault: {fault}" if fault else "")


def gap_approach_status_text(status):
    """Keep the commanded flange gap separate from the visual estimate."""
    approach = status.get("gap_approach")
    if not isinstance(approach, dict):
        approach = {}

    def number(key, scale, unit):
        try:
            return f"{_number(approach.get(key), key) * scale:.1f}{unit}"
        except ValueError:
            return "--"

    state = ("waiting for coordinates (automatic continuation)" if approach.get("temporary_hold")
             else {"idle": "not started", "running": "running", "paused": "paused (press Resume)",
                   "complete": "target set"}.get(approach.get("state"), "not started"))
    text = (f"Approach: {state}\n"
            f"Commanded gap: {number('commanded_gap_m', 100., ' cm')}   "
            f"Estimated gap: {number('estimated_gap_m', 100., ' cm')}\n"
            f"Approach progress: {number('progress', 100., '%')}")
    reason = approach.get("error") or approach.get("reason")
    return text + (f"\n{reason}" if reason else "")


def draft_geometry(scene, request, mode, leader="ur10e"):
    """Preview form values in camera coordinates without sending any command."""
    robots = scene["robots"]
    if mode != "follow":
        raise ValueError("only follower mode is supported")
    follower = "ur5e" if leader == "ur10e" else "ur10e"
    from urcal.cooperative.task import RelativePath
    relative = RelativePath(facing_follower(np.eye(4), request["gap"]),
                            kind=request.get("path", "hold"), delta=request.get("path_delta", [0.]*6),
                            radius=request.get("path_radius", .02), duration=request.get("path_duration", 10.))
    # Show the edited END pose, including line rotation increments. The actual
    # controller target remains separately visible at its current path phase.
    goals = {follower: robots[leader]["flange_C"] @ relative.pose(1.)}
    path = task_path(scene, {"leader": leader, "kind": request.get("path", "hold"),
                            "gap": request["gap"], "delta": request.get("path_delta", [0.]*6),
                            "radius": request.get("path_radius", .02), "duration": request.get("path_duration", 10.)})
    return goals, path


class TargetPanel:
    """Pump with poll(status); requests remain subject to controller validation."""

    def __init__(self, args, send_command, stop_event):
        if getattr(args, "mode", "follow") != "follow":
            raise ValueError("only follower mode is supported")
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError("Target GUI requires Tkinter (install python3-tk)") from exc
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("Target GUI needs an available graphical display (check DISPLAY)") from exc
        self._tk = tk
        self.args, self.send_command, self.stop_event = args, send_command, stop_event
        self.mode = args.mode
        self._simulation = bool(getattr(args, "simulate", False))
        self._research = getattr(args, "method", None) is not None
        self._task_paused = False
        self._closed = False
        self._draft_dirty = False
        self._status = {"active": False, "ready": False, "message": "Waiting for calibration"}
        self.root.title("SIMULATION — captured initial robot state" if self._simulation
                        else "Dual-arm live pose & target viewer")
        self.root.resizable(True, True)
        self.root.geometry(f"{max(1000, min(1380, self.root.winfo_screenwidth() - 40))}x"
                           f"{max(650, min(900, self.root.winfo_screenheight() - 80))}")
        self.root.minsize(1000, 650)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._request_stop)
        self.root.bind("<Escape>", lambda _event: self._request_stop())
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")
        self._view = RobotSceneView(panes, getattr(args, "max_calibration_age", .5),
                                   getattr(args, "max_robot_age", .15), simulation=self._simulation)
        panes.add(self._view.frame, weight=3)
        controls = ttk.Frame(panes, width=550)
        panes.add(controls, weight=0)
        def keep_controls_visible(_event=None):
            # Keep all task fields and STOP reachable at the minimum window size.
            def clamp_sash():
                if self._closed:
                    return
                try:
                    width = panes.winfo_width()
                    if width >= 900:
                        current = panes.sashpos(0)
                        bounded = max(340, min(current, width - 550))
                        if current != bounded:
                            panes.sashpos(0, bounded)
                except tk.TclError:
                    pass
            self.root.after_idle(clamp_sash)
        panes.bind("<Configure>", keep_controls_visible)
        panes.bind("<B1-Motion>", keep_controls_visible)
        panes.bind("<ButtonRelease-1>", keep_controls_visible)
        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)
        scroll = tk.Canvas(controls, width=525, highlightthickness=0)
        scroll.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(controls, orient="vertical", command=scroll.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        scroll.configure(yscrollcommand=scrollbar.set)
        body = ttk.Frame(scroll, padding=14)
        body_window = scroll.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _event: scroll.configure(scrollregion=scroll.bbox("all")))
        scroll.bind("<Configure>", lambda event: scroll.itemconfigure(body_window, width=event.width))
        ttk.Label(body, text="Camera frame: X right, Y down, Z forward", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        self._gap = tk.StringVar(self.root, value=f"{(args.gap if args.gap is not None else .30) * 100:g}")
        self._message = tk.StringVar(self.root, value="Waiting for calibration")
        self._error = tk.StringVar(self.root, value="")
        self._progress_message = tk.StringVar(self.root, value="")
        self._approach_message = tk.StringVar(self.root, value=gap_approach_status_text({}))
        self._editable = []
        row = 1
        leader = getattr(args, "leader", "ur10e")
        follower = "ur5e" if leader == "ur10e" else "ur10e"
        ttk.Label(body, text=f"Follow: leader {leader.upper()}  ->  follower {follower.upper()}",
                  font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 8))
        row += 1
        ttk.Label(body, text="Follower stays along the leader flange +Z; flange +Z axes face each other.",
                  wraplength=480).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 9))
        row += 1
        ttk.Button(body, text="Position error graph", command=self._show_error_plot).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 6), padx=(0, 10))
        row += 1
        ttk.Label(body, text="Flange gap (cm)").grid(row=row, column=0, sticky="w", pady=(8, 0))
        gap_entry = ttk.Entry(body, textvariable=self._gap, width=15)
        gap_entry.grid(row=row + 1, column=0, sticky="w", pady=(2, 8))
        self._editable.append(gap_entry)
        ttk.Label(body, text=f"Minimum: {args.min_gap * 100:g} cm").grid(row=row + 1, column=1, columnspan=2, sticky="w")
        row += 2
        if self._research:
            ttk.Label(body, text=f"Method: {args.method}; follow law: {getattr(args, 'follow_law', 'feedforward')} "
                      "(selected at startup)").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(4, 3))
            row += 1
            self._path = tk.StringVar(self.root, value=getattr(args, "path", "hold"))
            ttk.Label(body, text="Relative task path").grid(row=row, column=0, sticky="w")
            self._path_selector = ttk.Combobox(body, textvariable=self._path,
                                               values=("hold", "line", "circle"), state="readonly", width=13)
            self._path_selector.grid(row=row, column=1, sticky="w", pady=(0, 6))
            row += 1
            delta = np.asarray(getattr(args, "path_delta", [0., 0., -.05, 0., 0., 0.]), dtype=float)
            self._delta_vars = [tk.StringVar(self.root, value=f"{value * 100:.9g}") for value in delta[:3]]
            self._delta_rotation_vars = [tk.StringVar(self.root, value=f"{np.rad2deg(value):.9g}") for value in delta[3:]]
            ttk.Label(body, text="Line delta is expressed in the leader flange frame.").grid(
                row=row, column=0, columnspan=4, sticky="w")
            row += 1
            for labels, variables in ((["Delta X (cm)", "Delta Y (cm)", "Delta Z (cm)"], self._delta_vars),
                                      (["Rx (deg)", "Ry (deg)", "Rz (deg)"], self._delta_rotation_vars)):
                for index, (label, variable) in enumerate(zip(labels, variables)):
                    ttk.Label(body, text=label).grid(row=row, column=index, sticky="w")
                    entry = ttk.Entry(body, textvariable=variable, width=15)
                    entry.grid(row=row + 1, column=index, sticky="w", padx=(0, 10), pady=(2, 6))
                    self._editable.append(entry)
                row += 2
            ttk.Label(body, text="Rx/Ry/Rz are rotation-vector (axis-angle) components, not Euler angles.",
                      wraplength=480).grid(row=row, column=0, columnspan=4, sticky="w")
            row += 1
            self._radius = tk.StringVar(self.root, value=f"{getattr(args, 'path_radius', .02) * 100:.9g}")
            self._duration = tk.StringVar(self.root, value=f"{getattr(args, 'path_duration', 10.):.9g}")
            for index, (label, variable) in enumerate((("Circle radius (cm)", self._radius),
                                                       ("Duration (s)", self._duration))):
                ttk.Label(body, text=label).grid(row=row, column=index, sticky="w")
                entry = ttk.Entry(body, textvariable=variable, width=15)
                entry.grid(row=row + 1, column=index, sticky="w", padx=(0, 10), pady=(2, 8))
                self._editable.append(entry)
            row += 2
        self._apply_button = ttk.Button(body, text="Apply target", command=self._apply)
        self._apply_button.grid(row=row, column=0, sticky="ew", padx=(0, 10))
        self._arm_button = ttk.Button(body, text="Start simulation" if self._simulation else "Arm motion", command=self._arm)
        self._arm_button.grid(row=row, column=1, sticky="ew", padx=(0, 10))
        ttk.Button(body, text="STOP / Exit", command=self._request_stop).grid(row=row, column=2, sticky="ew")
        row += 1
        self._approach_button = ttk.Button(body, text="Approach to 15 cm", command=self._request_gap_approach)
        self._approach_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=(0, 10))
        row += 1
        ttk.Label(body, textvariable=self._approach_message, wraplength=480).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
        row += 1
        ttk.Label(body, text="Pause freezes the gap target; leader following continues.\n"
                  "15 cm is the target flange gap, not a contact or insertion depth.",
                  wraplength=480).grid(row=row, column=0, columnspan=4, sticky="w", pady=(4, 0))
        row += 1
        if self._simulation:
            self._sim_pause_button = ttk.Button(body, text="Pause simulation", command=self._toggle_simulation_pause)
            self._sim_pause_button.grid(row=row, column=0, sticky="ew", pady=(8, 0), padx=(0, 10))
            ttk.Button(body, text="Reset simulation", command=self._reset_simulation).grid(
                row=row, column=1, sticky="ew", pady=(8, 0), padx=(0, 10))
            row += 1
            ttk.Label(body, text="SIMULATION: initial state captured once. No hardware commands.\n"
                      "Large pose differences use initial alignment before control.\n"
                      "Pause freezes virtual time. Reset restores the captured pose and unlocks target editing.",
                      wraplength=480, foreground="#8a4100").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(8, 0))
            row += 1
        if self._research:
            self._pause_button = ttk.Button(body, text="Pause task", command=self._toggle_task_pause)
            self._pause_button.grid(row=row, column=0, sticky="ew", pady=(8, 0), padx=(0, 10))
            ttk.Label(body, text="Pause freezes path progress; base compensation continues.",
                      wraplength=330).grid(row=row, column=1, columnspan=3, sticky="w", pady=(8, 0))
            row += 1
            ttk.Label(body, textvariable=self._progress_message, wraplength=480).grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
            row += 1
        if not args.execute and not self._simulation:
            ttk.Label(body, text="Preview only. --execute is required to arm motion.").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(8, 0))
            row += 1
        if getattr(args, "initial_align", False) and not self._simulation:
            ttk.Label(body, text="Initial alignment: Arm checks the joint path, then aligns slowly.\n"
                      "Keep the leader and both bases stationary until ALIGNED.\n"
                      "STOP / Exit stops alignment; Pause task only affects the later relative path.",
                      wraplength=480, foreground="#8a4100").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(8, 0))
            row += 1
        if getattr(args, "check_reprojection", True):
            quality_text = (f"Reprojection rejection: ON (mean {getattr(args, 'max_reproj_px', 5):g} px, "
                            f"max {getattr(args, 'max_point_reproj_px', 8):g} px)")
        else:
            quality_text = "Reprojection rejection: OFF (mean/max displayed only)"
        ttk.Label(body, text=quality_text, wraplength=480).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
        row += 1
        tolerance_text = (f"Estimated flange task tolerance: {getattr(args, 'target_tolerance_m', 0.) * 100:g} cm / "
                          f"{getattr(args, 'target_tolerance_deg', 0.):g} deg.\n"
                          "Magenta: nominal target. Cyan: selected/next command.\n"
                          "Tolerance is not a calibration accuracy guarantee.")
        ttk.Label(body, text=tolerance_text, wraplength=480).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
        row += 1
        ttk.Label(body, text="Target editing is available while motion is inactive.").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(5, 0))
        ttk.Label(body, textvariable=self._message, wraplength=480).grid(row=row + 1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(body, textvariable=self._error, foreground="#b00020", wraplength=480).grid(
            row=row + 2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        variables = [self._gap]
        if self._research:
            variables.extend([self._path, *self._delta_vars, *self._delta_rotation_vars, self._radius, self._duration])
        for variable in variables:
            variable.trace_add("write", self._on_target_edit)
        self._refresh_enabled()
        self._error_plot = None
        self._show_error_plot()

    def _show_error_plot(self):
        if self._closed:
            return
        try:
            if getattr(self, "_error_plot", None) is None:
                leader = getattr(self.args, "leader", "ur10e")
                self._error_plot = PositionErrorWindow(
                    self.root, follower="ur5e" if leader == "ur10e" else "ur10e",
                    simulation=getattr(self, "_simulation", False))
            else:
                self._error_plot.show()
        except self._tk.TclError as exc:
            # This optional display never sends motion or stop requests.
            self._error.set(f"Position error graph unavailable: {exc}")

    def _on_target_edit(self, *_):
        self._draft_dirty = True
        self._refresh_enabled()

    def _target_request(self):
        path_options = {}
        if getattr(self, "_research", False):
            path_options = {
                "path": self._path.get(),
                "path_delta_cm": [variable.get() for variable in self._delta_vars],
                "path_rotation_degrees": [variable.get() for variable in self._delta_rotation_vars],
                "path_radius_cm": self._radius.get(), "path_duration": self._duration.get(),
            }
        return build_target_command(
            self.mode, self._gap.get(), self.args.min_gap, **path_options)

    def _preview_edits(self):
        view = getattr(self, "_view", None)
        if view is None or view.scene is None:
            return
        if not getattr(self, "_draft_dirty", False) or self._status.get("active", False):
            if view._draft:
                view.set_draft()
            return
        try:
            goals, path = draft_geometry(view.scene, self._target_request(), self.mode,
                                         getattr(self.args, "leader", "ur10e"))
            view.set_draft(goals, path)
        except (ValueError, KeyError, TypeError):
            # Incomplete numeric edits are common; never retain a misleading draft.
            view.set_draft()

    def _refresh_enabled(self):
        active = bool(self._status.get("active", False))
        for widget in self._editable + [self._apply_button]:
            widget.configure(state="disabled" if active else "normal")
        can_arm = ((self.args.execute or getattr(self, "_simulation", False))
                   and self._status.get("ready", False) and not active
                   and not getattr(self, "_draft_dirty", False))
        self._arm_button.configure(state="normal" if can_arm else "disabled")
        if getattr(self, "_research", False):
            self._path_selector.configure(state="disabled" if active else "readonly")
            self._pause_button.configure(state="normal" if active else "disabled",
                                         text="Resume task" if self._task_paused else "Pause task")
        if getattr(self, "_simulation", False):
            self._sim_pause_button.configure(state="normal" if active else "disabled",
                                             text="Resume simulation" if self._status.get("sim_paused") else "Pause simulation")
        if hasattr(self, "_approach_button"):
            action, label = self._gap_approach_action()
            self._approach_button.configure(state="normal" if action is not None else "disabled", text=label)
            self._approach_message.set(gap_approach_status_text(self._status))

    def _gap_approach_action(self):
        approach = self._status.get("gap_approach")
        if not isinstance(approach, dict):
            approach = {}
        state = approach.get("state", "idle")
        action, label = {
            "idle": ("start", "Approach to 15 cm"),
            "running": ("pause", "Pause approach"),
            "paused": ("resume", "Resume approach"),
            "complete": (None, "15 cm target set"),
        }.get(state, (None, "Approach to 15 cm"))
        # The controller owns phase/path/calibration eligibility. Require an
        # active execution session here too; preview cannot issue this action.
        enabled = (approach.get("enabled") is True and self._status.get("active", False)
                   and (self.args.execute or getattr(self, "_simulation", False))
                   and not self._status.get("sim_paused", False))
        return action if enabled else None, label

    def _request_gap_approach(self):
        action, _ = self._gap_approach_action()
        if action is None:
            return
        try:
            self.send_command({"type": "gap_approach", "action": action, "requested_at": time.monotonic()})
        except Exception as exc:
            self._error.set(str(exc))
        else:
            # Only authoritative status changes the action. Repeated clicks
            # before acknowledgement repeat the same idempotent request.
            self._error.set("Approach request sent; leader following continues")

    def _apply(self):
        if self._status.get("active", False):
            self._error.set("Target editing is disabled while motion is active")
            return
        try:
            request = self._target_request()
            self.send_command(request)
        except Exception as exc:
            self._error.set(str(exc))
        else:
            self._draft_dirty = False
            if getattr(self, "_view", None) is not None:
                self._view.set_draft()
            self._refresh_enabled()
            self._error.set("" if getattr(self, "_simulation", False)
                            else "Target request sent; waiting for controller validation")

    def _arm(self):
        if (not (self.args.execute or getattr(self, "_simulation", False))
                or not self._status.get("ready", False) or self._status.get("active", False)
                or getattr(self, "_draft_dirty", False)):
            return
        try:
            self.send_command("arm")
        except Exception as exc:
            self._error.set(str(exc))
        else:
            self._error.set("")

    def _toggle_simulation_pause(self):
        if not getattr(self, "_simulation", False) or not self._status.get("active", False):
            return
        try:
            self.send_command("sim_pause")
        except Exception as exc:
            self._error.set(str(exc))

    def _reset_simulation(self):
        if not getattr(self, "_simulation", False):
            return
        try:
            self.send_command("sim_reset")
        except Exception as exc:
            self._error.set(str(exc))

    def _request_stop(self):
        self.stop_event.set()
        self.close()

    def _toggle_task_pause(self):
        if not getattr(self, "_research", False) or not self._status.get("active", False):
            return
        try:
            self.send_command("resume_task" if self._task_paused else "pause_task")
        except Exception as exc:
            self._error.set(str(exc))
        else:
            self._task_paused = not self._task_paused
            self._error.set("Task pause/resume requested; base compensation remains enabled")
            self._refresh_enabled()

    def poll(self, status=None):
        if self._closed:
            return
        if self.stop_event.is_set():
            self.close()
            return
        if status is not None:
            self._status.update(status)
            if getattr(self, "_view", None) is not None:
                self._view.set_snapshot(status.get("scene"))
            self._message.set(str(self._status.get("message", "")))
            if getattr(self, "_research", False):
                if "task_paused" in status:
                    self._task_paused = bool(status["task_paused"])
                self._progress_message.set(research_status_text(self._status))
            self._refresh_enabled()
        self._preview_edits()
        if getattr(self, "_view", None) is not None:
            self._view.refresh()
        plot = getattr(self, "_error_plot", None)
        if plot is not None:
            try:
                if status is not None:
                    # Do not revive an old sample from the merged UI status.
                    plot.update(status.get("position_error"))
                plot.refresh()
            except self._tk.TclError as exc:
                self._error.set(f"Position error graph unavailable: {exc}")
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError as exc:
            if self._closed:
                return
            self.stop_event.set()
            self.close()
            raise RuntimeError("Target GUI closed unexpectedly") from exc

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.root.destroy()
            except self._tk.TclError:
                pass


def gui_worker(args, commands, statuses, stop):
    """Run Tk in its own process so inference cannot block Stop or target entry."""
    panel = None

    def send_command(command):
        try:
            if command == "arm" and not getattr(args, "simulate", False):
                command = ("arm", time.monotonic())
            commands.put_nowait(command)
        except Full as exc:
            raise RuntimeError("Controller command queue is full; retry when it has caught up") from exc

    try:
        panel = TargetPanel(args, send_command, stop)
        while not stop.is_set():
            latest = None
            while True:
                try:
                    latest = statuses.get_nowait()
                except Empty:
                    break
            panel.poll(latest)
            stop.wait(.02)
    except Exception as exc:
        print(f"[GUI STOP] {exc}", flush=True)
    finally:
        stop.set()
        if panel is not None:
            panel.close()
