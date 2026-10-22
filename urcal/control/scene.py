"""Validated dual-arm display geometry, independent of Tk and robot drivers.

All scene coordinates are in the camera frame, in metres. Current flange/TCP
markers come from controller poses; the link polyline comes from injected
nominal kinematics and can differ slightly from those markers. This module
only prepares a view and never validates a motion command.

The orthographic orbit uses radians. At azimuth=elevation=0, camera X points
right on screen, camera Y points down, and positive Z is farther away. Points
with larger returned depth should therefore be drawn first when depth sorting.
"""

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from urcal.control.geometry import flange_from_tcp, matrix_pose, rotation_distance


ROBOTS = ("ur5e", "ur10e")


class SceneError(ValueError):
    """A display snapshot or view parameter cannot be interpreted safely."""


def _number(value, label, *, positive=False, nonnegative=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneError(f"{label} must be finite") from exc
    if not np.isfinite(number):
        raise SceneError(f"{label} must be finite")
    if positive and number <= 0:
        raise SceneError(f"{label} must be positive")
    if nonnegative and number < 0:
        raise SceneError(f"{label} must be nonnegative")
    return number


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneError(f"{label} must have finite shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise SceneError(f"{label} must have finite shape {shape}")
    return result.copy()


def _points(value, label="points", minimum=1):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneError(f"{label} must be finite Nx3 points") from exc
    if (result.ndim != 2 or result.shape[1] != 3 or len(result) < minimum
            or not np.isfinite(result).all()):
        raise SceneError(f"{label} must be finite Nx3 points, with N >= {minimum}")
    return result.copy()


def _pose(value, label):
    try:
        result = np.asarray(value, dtype=float)
        matrix_pose(result)
    except (ValueError, TypeError, OverflowError) as exc:
        raise SceneError(f"{label} must be a proper finite 4x4 rigid transform") from exc
    return result.copy()


def _pair(value, label):
    if not isinstance(value, Mapping) or set(value) != set(ROBOTS):
        raise SceneError(f"{label} must contain both ur5e and ur10e")
    return value


def _age(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = _pair(value, "measurement_age")
        return {name: _number(value[name], f"{name} measurement_age", nonnegative=True)
                for name in ROBOTS}
    return _number(value, "measurement_age", nonnegative=True)


def build_scene(snapshot, models):
    """Build copied display data from one complete paired status snapshot.

    Required snapshot keys are ``at``, ``bases`` (T_C_B for both robots), and
    ``robots`` (both robots, each with q[6], tcp=T_B_TCP, tcp_offset=T_F_TCP).
    ``goals`` is an optional subset of camera-frame FLANGE targets; missing
    goals are valid, including WAIT states.
    ``nominal_goals`` overrides the displayed task target/error when the
    controller selected an approximate endpoint. ``command_goals`` contains
    separately drawn selected or near-term flange command poses. Both maps
    use camera coordinates and are copied and validated like ordinary goals.
    ``measurement_age`` is optional
    None, a nonnegative scalar, or a pair of nonnegative seconds. Optional
    calibration_valid(bool), calibration_mode('live' or 'frozen'), optional
    calibration_captured_at(monotonic seconds), reason(str), and task(mapping
    or None) metadata are copied. A frozen calibration retains its actual
    measurement age; it does not imply fresh camera measurements.

    Models must provide points(q) in their controller base frame. No models
    are imported or instantiated here. Output robot fields are base_C,
    joint_polyline_C, flange_C, tcp_C, target_C, error_m, error_rad. Target and
    error fields are None when absent. Errors compare the measured flange
    with its target, not the TCP or last nominal link point. ``bounds`` gives
    camera-frame min/max XYZ including display axes of length 0.08 m.

    Invalid input is rejected as SceneError; no partial scene is returned.
    No snapshot/model arrays are modified or shared with output arrays.
    """
    if not isinstance(snapshot, Mapping):
        raise SceneError("snapshot must be a mapping")
    try:
        at = _number(snapshot["at"], "at", nonnegative=True)
        bases = _pair(snapshot["bases"], "bases")
        states = _pair(snapshot["robots"], "robots")
    except KeyError as exc:
        raise SceneError(f"snapshot missing {exc.args[0]}") from exc
    if not isinstance(models, Mapping) or not set(ROBOTS).issubset(models):
        raise SceneError("models must provide ur5e and ur10e kinematics")
    # The ordinary target/error always describes the requested nominal task.
    # A reachable selected command is separately drawn and never relabelled
    # as a perfect task result.
    goals = snapshot.get("nominal_goals", snapshot.get("goals", {}))
    if not isinstance(goals, Mapping) or set(goals) - set(ROBOTS):
        raise SceneError("goals must map known robot names to camera flange targets")
    commands = snapshot.get("command_goals", {})
    if not isinstance(commands, Mapping) or set(commands) - set(ROBOTS):
        raise SceneError("command_goals must map known robot names to camera flange targets")
    calibration_valid = snapshot.get("calibration_valid", True)
    calibration_mode = snapshot.get("calibration_mode", "live")
    calibration_captured_at = snapshot.get("calibration_captured_at")
    reason = snapshot.get("reason", "")
    task = snapshot.get("task")
    if not isinstance(calibration_valid, bool):
        raise SceneError("calibration_valid must be boolean")
    if calibration_mode not in ("live", "frozen"):
        raise SceneError("calibration_mode must be live or frozen")
    if calibration_captured_at is not None:
        calibration_captured_at = _number(calibration_captured_at, "calibration_captured_at", nonnegative=True)
    if not isinstance(reason, str):
        raise SceneError("reason must be a string")
    if task is not None and not isinstance(task, Mapping):
        raise SceneError("task metadata must be a mapping or None")
    scene = {"at": at, "measurement_age": _age(snapshot.get("measurement_age")),
             "calibration_valid": calibration_valid, "calibration_mode": calibration_mode,
             "calibration_captured_at": calibration_captured_at, "reason": reason,
             "task": deepcopy(task), "robots": {}}
    for name in ROBOTS:
        state = states[name]
        if not isinstance(state, Mapping):
            raise SceneError(f"{name} robot state must be a mapping")
        base = _pose(bases[name], name + " base")
        try:
            q = _array(state["q"], (6,), name + " joints")
            tcp = _pose(state["tcp"], name + " TCP")
            offset = _pose(state["tcp_offset"], name + " TCP offset")
        except KeyError as exc:
            raise SceneError(f"{name} state missing {exc.args[0]}") from exc
        model = models[name]
        if not callable(getattr(model, "points", None)):
            raise SceneError(f"{name} model must provide points(q)")
        try:
            local_points = _points(model.points(q.copy()), name + " nominal links", minimum=2)
        except Exception as exc:
            raise SceneError(f"{name} nominal link geometry failed: {exc}") from exc
        flange_c = base @ flange_from_tcp(tcp, offset)
        tcp_c = base @ tcp
        target = _pose(goals[name], name + " target") if name in goals else None
        command = _pose(commands[name], name + " command") if name in commands else None
        scene["robots"][name] = {
            "base_C": base,
            "joint_polyline_C": local_points @ base[:3, :3].T + base[:3, 3],
            "flange_C": flange_c,
            "tcp_C": tcp_c,
            "target_C": target,
            "command_C": command,
            "command_adjustment_m": None if target is None or command is None else float(
                np.linalg.norm(target[:3, 3] - command[:3, 3])),
            "command_adjustment_rad": None if target is None or command is None else rotation_distance(target, command),
            "error_m": None if target is None else float(np.linalg.norm(target[:3, 3] - flange_c[:3, 3])),
            "error_rad": None if target is None else rotation_distance(flange_c, target),
        }
    points = scene_points(scene)
    scene["bounds"] = (points.min(axis=0), points.max(axis=0))
    return scene


def scene_points(scene, axis_length=.08):
    """Collect links and base/flange/TCP/target axes for fit-to-scene.

    The camera origin is deliberately not included: it can be far away from
    the robot pair and should be drawn separately as a corner orientation key.
    """
    length = _number(axis_length, "axis_length", nonnegative=True)
    if not isinstance(scene, Mapping) or "robots" not in scene:
        raise SceneError("scene must contain robots")
    robots = _pair(scene["robots"], "scene robots")
    result = []
    for name in ROBOTS:
        robot = robots[name]
        if not isinstance(robot, Mapping):
            raise SceneError(f"{name} scene robot must be a mapping")
        try:
            result.append(_points(robot["joint_polyline_C"], name + " scene links", minimum=2))
            for key in ("base_C", "flange_C", "tcp_C", "target_C", "command_C"):
                pose = robot.get(key) if key == "command_C" else robot[key]
                if key in ("target_C", "command_C") and pose is None:
                    continue
                pose = _pose(pose, name + " " + key)
                origin = pose[:3, 3]
                result.append(np.vstack((origin, origin + length * pose[:3, :3].T)))
        except KeyError as exc:
            raise SceneError(f"{name} scene missing {exc.args[0]}") from exc
    return np.concatenate(result, axis=0)


def orbit_basis(azimuth=0., elevation=0.):
    """Return rows (screen-right, screen-down, depth) in camera coordinates.

    Azimuth rotates around camera Y. Positive elevation tilts the view toward
    positive camera Y. Front=(0,0), top=(0,pi/2), side=(pi/2,0). The formula
    remains well-defined at top/bottom views; there is no look-at pole case.
    """
    a = _number(azimuth, "azimuth")
    e = _number(elevation, "elevation")
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    return np.array([[ca, 0., -sa], [-sa * se, ce, -ca * se], [sa * ce, se, ca * ce]])


def _viewport(value):
    viewport = _array(value, (2,), "viewport (width, height)")
    if np.any(viewport <= 0):
        raise SceneError("viewport width and height must be positive")
    return viewport


def project_points(points, center, scale, azimuth, elevation, viewport):
    """Return (pixel_XY[N,2], depth[N]) for a finite Nx3 point array.

    Scale is pixels/metre and center is the orbit pivot in camera coordinates.
    Projection is orthographic: depth never changes a point's scale and no
    near-plane clipping occurs. Empty (0,3) arrays are accepted for renderers.
    """
    points = _points(points, minimum=0)
    center = _array(center, (3,), "view center")
    scale = _number(scale, "view scale", positive=True)
    viewport = _viewport(viewport)
    view = (points - center) @ orbit_basis(azimuth, elevation).T
    xy = view[:, :2] * scale + viewport / 2.
    if not np.isfinite(xy).all() or not np.isfinite(view).all():
        raise SceneError("projection exceeds finite numeric range")
    return xy, view[:, 2].copy()


def fit_view(points, viewport, azimuth=0., elevation=0., padding=.1, min_span=.1):
    """Return {center: camera XYZ, scale: pixels/metre} fitting finite points.

    Padding is a fraction of each viewport edge, in [0,.5). min_span in metres
    prevents extreme magnification for coincident points or a flat projection.
    Bounds are fitted after orbit rotation, so targets remain inside the view.
    """
    points = _points(points)
    viewport = _viewport(viewport)
    padding = _number(padding, "padding", nonnegative=True)
    if padding >= .5:
        raise SceneError("padding must be less than 0.5")
    min_span = _number(min_span, "min_span", positive=True)
    basis = orbit_basis(azimuth, elevation)
    view = points @ basis.T
    lo, hi = view.min(axis=0), view.max(axis=0)
    midpoint = lo / 2. + hi / 2.
    span = np.maximum(hi[:2] - lo[:2], min_span)
    scale = float(np.min(viewport * (1. - 2. * padding) / span))
    center = midpoint @ basis
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise SceneError("view bounds exceed finite numeric range")
    return {"center": center, "scale": scale}
