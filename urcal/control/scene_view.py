"""Interactive orthographic robot view, drawn with Tk Canvas in the GUI process.

Only visualizes snapshots; it has no controller or command-queue reference.
Tk and nominal robot models are loaded lazily when the view is used.
"""

import time

import numpy as np

from urcal.control.scene import build_scene, fit_view, orbit_basis, project_points, scene_points


COLORS = {"ur5e": "#50b5ff", "ur10e": "#ffab54"}
AXIS_COLORS = ("#ff657a", "#64d68e", "#68aaff")
TARGET_COLOR = "#e879f9"
COMMAND_COLOR = "#67e8f9"
DRAFT_COLOR = "#facc15"


def task_path(scene, task=None):
    """Return camera-frame path points; never perform IK or change task progress."""
    task = scene.get("task") if task is None else task
    if not task or task.get("kind", "hold") == "hold":
        return {}
    from urcal.control.geometry import facing_follower
    from urcal.cooperative.task import RelativePath

    leader = task["leader"]
    follower = "ur5e" if leader == "ur10e" else "ur10e"
    start = task.get("start_pose")
    if start is None:
        start = facing_follower(np.eye(4), task["gap"])
    path = RelativePath(start, kind=task["kind"],
                        delta=task["delta"], radius=task["radius"], duration=task["duration"])
    origin = scene["robots"][leader]["flange_C"]
    return {follower: np.stack([(origin @ path.pose(u))[:3, 3] for u in np.linspace(0., 1., 49)])}


class RobotSceneView:
    def __init__(self, parent, max_measurement_age=.5, max_robot_age=.15, *, simulation=False):
        import tkinter as tk
        from tkinter import ttk

        self._simulation = bool(simulation)
        self.frame = ttk.Frame(parent)
        self.frame.rowconfigure(2, weight=1)
        self.frame.columnconfigure(0, weight=1)
        title = "Simulated robots & flange targets" if self._simulation else "Live robots & flange targets"
        ttk.Label(self.frame, text=title, font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 3))
        toolbar = ttk.Frame(self.frame)
        toolbar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for label, angles, width in (("3D", (-.55, .35), 5), ("Camera XY", (0., 0.), 10),
                                     ("XZ", (0., np.pi/2), 5), ("YZ", (np.pi/2, 0.), 5)):
            ttk.Button(toolbar, text=label, command=lambda a=angles: self.preset(*a), width=width).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Fit all", command=self.fit, width=8).pack(side="left", padx=(10, 2))
        self.canvas = tk.Canvas(self.frame, width=700, height=570, background="#111827", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=8)
        self._details = tk.StringVar(parent, value="Waiting for paired calibration and joint states...")
        details = ttk.Label(self.frame, textvariable=self._details, font=("TkFixedFont", 10),
                            justify="left", wraplength=760)
        details.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 4))
        help_text = ttk.Label(self.frame, text="Drag: orbit   |   Right-drag: pan   |   Wheel: zoom   |   Axes: X red, Y green, Z blue",
                              wraplength=760)
        help_text.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 4))
        legend = ttk.Label(self.frame, text="Blue: UR5e   Orange: UR10e   Magenta: nominal target   Cyan: command   Yellow: edit",
                           wraplength=760)
        legend.grid(row=5, column=0, sticky="w", padx=12, pady=(0, 8))
        def reflow(event):
            if event.widget is self.frame:
                for label in (details, help_text, legend):
                    label.configure(wraplength=max(280, event.width - 24))
        self.frame.bind("<Configure>", reflow)
        self.max_measurement_age = float(max_measurement_age)
        self.max_robot_age = float(max_robot_age)
        self.scene = None
        self._models = None
        self._draft = {}
        self._draft_path = {}
        self._path = {}
        self._missing = True
        self._notice = "Waiting for calibration"
        self._dirty = True
        self._need_fit = True
        self._last_draw = 0.
        self.azimuth, self.elevation = -.55, .35
        self.center, self.scale = np.zeros(3), 250.
        self._drag = None
        self.canvas.bind("<Configure>", lambda _event: self._invalidate())
        self.canvas.bind("<ButtonPress-1>", lambda event: self._start_drag(event, "orbit"))
        self.canvas.bind("<B1-Motion>", self._move_drag)
        self.canvas.bind("<ButtonPress-3>", lambda event: self._start_drag(event, "pan"))
        self.canvas.bind("<B3-Motion>", self._move_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: self._end_drag())
        self.canvas.bind("<ButtonRelease-3>", lambda _event: self._end_drag())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.zoom(1.12))
        self.canvas.bind("<Button-5>", lambda _event: self.zoom(1/1.12))

    def _invalidate(self):
        self._dirty = True

    def _viewport(self):
        return max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())

    def set_snapshot(self, snapshot):
        if snapshot is None:
            self._missing = True
            self._notice = "No fresh paired state — last view retained"
            self._dirty = True
            return
        try:
            if self._models is None:
                from urcal.cooperative.kinematics import NominalKinematics
                self._models = {name: NominalKinematics(name) for name in COLORS}
            candidate = build_scene(snapshot, self._models)
            path = task_path(candidate)
        except (ValueError, KeyError, TypeError) as exc:
            self._missing = True
            self._notice = "Display data rejected: " + str(exc)
            self._dirty = True
            return
        self.scene, self._path = candidate, path
        self._missing = False
        self._notice = candidate.get("reason", "")
        if self._need_fit:
            self.fit()
            self._need_fit = self.canvas.winfo_width() <= 100
        self._dirty = True

    def set_draft(self, goals=None, path=None):
        self._draft = {} if goals is None else goals
        self._draft_path = {} if path is None else path
        self._dirty = True

    def fit(self):
        if self.scene is None:
            return
        points = [scene_points(self.scene)]
        points.extend(pose[:3, 3][None] for pose in self._draft.values())
        points.extend(self._path.values())
        points.extend(self._draft_path.values())
        fitted = fit_view(np.concatenate(points), self._viewport(), self.azimuth, self.elevation, padding=.17)
        self.center, self.scale = fitted["center"], fitted["scale"]
        self._dirty = True

    def preset(self, azimuth, elevation):
        self.azimuth, self.elevation = azimuth, elevation
        self.fit()
        self._dirty = True

    def zoom(self, factor):
        self.scale = float(np.clip(self.scale * factor, 20., 20000.))
        self._dirty = True
        return "break"

    def _wheel(self, event):
        if event.delta:
            return self.zoom(1.12 if event.delta > 0 else 1/1.12)
        return "break"

    def _start_drag(self, event, mode):
        self._drag = (event.x, event.y, mode)

    def _end_drag(self):
        self._drag = None

    def _move_drag(self, event):
        if self._drag is None:
            return
        x, y, mode = self._drag
        dx, dy = event.x - x, event.y - y
        if mode == "orbit":
            self.azimuth += dx * .008
            self.elevation = float(np.clip(self.elevation - dy * .008, -np.pi/2, np.pi/2))
        else:
            basis = orbit_basis(self.azimuth, self.elevation)
            self.center -= basis.T @ np.array([dx / self.scale, dy / self.scale, 0.])
        self._drag = (event.x, event.y, mode)
        self._dirty = True

    def _project(self, points):
        return project_points(np.asarray(points), self.center, self.scale,
                              self.azimuth, self.elevation, self._viewport())

    def _stale(self, now):
        if self.scene is None or self._missing:
            return True
        elapsed = max(0., now - self.scene["at"])
        age = self.scene.get("measurement_age", 0.)
        if age is None or now < self.scene["at"]:
            return True
        if isinstance(age, dict):
            age = max(age.values(), default=0.)
        frozen = self.scene.get("calibration_mode") == "frozen"
        return (not self.scene.get("calibration_valid", True) or elapsed > self.max_robot_age
                or (not frozen and age + elapsed > self.max_measurement_age))

    def refresh(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self._last_draw < 1/20:
            return
        # Redraw at least every 200 ms so stale snapshots visibly expire.
        if not self._dirty and now - self._last_draw < .2:
            return
        self._last_draw, self._dirty = now, False
        self.draw(now)

    def draw(self, now):
        canvas = self.canvas
        canvas.delete("all")
        width, height = self._viewport()
        if self.scene is None:
            canvas.create_text(width/2, height/2, text="Waiting for both robot poses\n\nLive links and flange targets will appear here",
                               fill="#b8c6dc", font=("TkDefaultFont", 15), justify="center")
            self._details.set(self._notice)
            return
        stale = self._stale(now)
        dim = "#64748b"
        primitives, labels = [], []

        def line(points, color, line_width=2, dash=(), arrow=None):
            xy, depth = self._project(points)
            options = {"fill": color, "width": line_width}
            if dash:
                options["dash"] = dash
            if arrow:
                options["arrow"] = arrow
            primitives.append((float(np.mean(depth)), "line", xy.ravel().tolist(), options))

        def circle(point, radius, color, hollow=False):
            xy, depth = self._project(np.asarray(point)[None])
            x, y = xy[0]
            primitives.append((float(depth[0]) - .0001, "oval", [x-radius, y-radius, x+radius, y+radius],
                               {"outline": color, "fill": "#111827" if hollow else color, "width": 2}))

        def axes(pose, length, dashed=False):
            origin = pose[:3, 3]
            for axis, color in enumerate(AXIS_COLORS):
                line([origin, origin + length * pose[:3, axis]], dim if stale else color,
                     2 if dashed else 3, (4, 3) if dashed else (), "last")

        def label(point, text, color, offset=(10, -16)):
            xy, _ = self._project(np.asarray(point)[None])
            labels.append((xy[0] + offset, text, color))

        # Camera-XY reference grid through the scene center, not a physical table.
        span = max(.2, min(width, height) / self.scale)
        grid_step = 10 ** np.floor(np.log10(span / 8))
        grid_step *= 5 if span / grid_step > 32 else 2 if span / grid_step > 16 else 1
        extent = grid_step * 6
        for value in np.arange(-6, 7) * grid_step:
            line([self.center + [-extent, value, 0], self.center + [extent, value, 0]], "#253247", 1)
            line([self.center + [value, -extent, 0], self.center + [value, extent, 0]], "#253247", 1)
        for path in self._path.values():
            for a, b in zip(path[:-1], path[1:]):
                line([a, b], dim if stale else "#986fc0", 1, (3, 3))
        for path in self._draft_path.values():
            for a, b in zip(path[:-1], path[1:]):
                line([a, b], dim if stale else DRAFT_COLOR, 1, (4, 3))
        for name, robot in self.scene["robots"].items():
            color = dim if stale else COLORS[name]
            points = robot["joint_polyline_C"]
            for a, b in zip(points[:-1], points[1:]):
                line([a, b], color, 8)
            for point in points:
                circle(point, 5, color)
            circle(robot["base_C"][:3, 3], 10, color, True)
            label(robot["base_C"][:3, 3], name.upper() + " BASE", color, (10, 18))
            flange = robot["flange_C"]
            axes(flange, .09)
            circle(flange[:3, 3], 6, color)
            label(flange[:3, 3], name.upper() + " NOW", color, (10, -22))
            tcp = robot["tcp_C"]
            if np.linalg.norm(tcp[:3, 3] - flange[:3, 3]) > .001:
                line([flange[:3, 3], tcp[:3, 3]], dim if stale else "#cbd5e1", 3)
                label(tcp[:3, 3], "TCP", dim if stale else "#cbd5e1", (8, 14))
            target = robot["target_C"]
            if target is not None:
                color = dim if stale else TARGET_COLOR
                line([flange[:3, 3], target[:3, 3]], color, 1, (4, 4))
                axes(target, .12, True)
                circle(target[:3, 3], 10, color, True)
                label(target[:3, 3], name.upper() + " TARGET", color, (10, 23))
            command = robot.get("command_C")
            if command is not None:
                color = dim if stale else COMMAND_COLOR
                axes(command, .075, True)
                circle(command[:3, 3], 7, color, True)
                label(command[:3, 3], name.upper() + " COMMAND", color, (10, 39))
            draft = self._draft.get(name)
            if draft is not None:
                color = dim if stale else DRAFT_COLOR
                axes(draft, .10, True)
                circle(draft[:3, 3], 13, color, True)
                label(draft[:3, 3], name.upper() + " EDIT", color, (12, -38))
        # Orthographic depth sorting: distant primitives first.
        for _, kind, coords, options in sorted(primitives, key=lambda item: item[0], reverse=True):
            getattr(canvas, "create_" + kind)(*coords, **options)
        for xy, text, color in labels:
            canvas.create_text(*xy, text=text, fill=color, anchor="w", font=("TkDefaultFont", 10, "bold"))
        simulation = getattr(self, "_simulation", False)
        frozen = self.scene.get("calibration_mode") == "frozen"
        badge = ("STALE / UNAVAILABLE — last pose" if stale
                 else "SIMULATED POSE" if simulation
                 else "CAPTURED CALIBRATION + LIVE JOINTS" if frozen else "LIVE ESTIMATED POSE")
        canvas.create_text(16, 18, text=badge, fill="#fbbf24" if stale else "#8ae3b0", anchor="nw",
                           font=("TkDefaultFont", 11, "bold"))
        caption = ("Camera coordinates (m) · virtual joints + snapshot / virtual bases" if simulation
                   else "Camera coordinates (m) · camera and bases assumed fixed" if frozen
                   else "Camera coordinates (m) · links from joints + estimated bases")
        canvas.create_text(16, 42, text=caption,
                           fill="#a7b5cb", anchor="nw", font=("TkDefaultFont", 9))
        if self._draft:
            canvas.create_text(16, 62, text="EDIT PREVIEW — Apply target to submit", fill=DRAFT_COLOR,
                               anchor="nw", font=("TkDefaultFont", 10, "bold"))
        canvas.create_text(16, height-16, text=f"Reference grid: {grid_step:g} m · orthographic view",
                           fill="#94a3b8", anchor="sw", font=("TkDefaultFont", 9))
        # Fixed corner triad shows the current view direction without fitting the camera origin.
        axes_2d = orbit_basis(self.azimuth, self.elevation) @ np.eye(3)
        origin = np.array([width-68., height-65.])
        for i, color in enumerate(AXIS_COLORS):
            end = origin + 35 * axes_2d[:2, i]
            canvas.create_line(*origin, *end, fill=color, width=2, arrow="last")
            canvas.create_text(*(end + [5, -5]), text="XYZ"[i], fill=color, font=("TkDefaultFont", 9, "bold"))
        rows = []
        for name, robot in self.scene["robots"].items():
            xyz = robot["flange_C"][:3, 3]
            rows.append(f"{name.upper():5s} NOW  {xyz[0]:+.3f}  {xyz[1]:+.3f}  {xyz[2]:+.3f} m")
            if robot["target_C"] is not None:
                xyz = robot["target_C"][:3, 3]
                rows.append(f"      GOAL {xyz[0]:+.3f}  {xyz[1]:+.3f}  {xyz[2]:+.3f} m   "
                            f"error {robot['error_m']*1000:.1f} mm / {np.rad2deg(robot['error_rad']):.1f} deg")
            if robot.get("command_C") is not None and robot.get("command_adjustment_m") is not None:
                rows.append(f"      COMMAND vs nominal: {robot['command_adjustment_m']*1000:.1f} mm / "
                            f"{np.rad2deg(robot['command_adjustment_rad']):.1f} deg")
        if stale:
            rows.append(self._notice or "State expired; displayed geometry is retained for reference.")
        self._details.set("\n".join(rows))
