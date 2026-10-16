"""Nominal differential kinematics and geometric feasibility for research control.

Link capsules are configurable proxies, not mesh collision certification. The
robot controller still checks its own calibrated reachability and safety limits.
All public transforms use controller base / zero-TCP flange conventions.
"""

import json
from pathlib import Path

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base, pose_delta, pose_matrix


JOINT_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")


class NominalKinematics:
    def __init__(self, robot, tcp_offset=None):
        if robot == "ur5e":
            from urcal.inference.infer_live import JOINT_NAMES, compute_ur5e_link_transforms
            self.fk = compute_ur5e_link_transforms
            self.forearm_length = .3922
        elif robot == "ur10e":
            from urcal.inference.infer_live_ur10e import JOINT_NAMES, compute_ur10e_link_transforms
            self.fk = compute_ur10e_link_transforms
            self.forearm_length = .57155
        else:
            raise ValueError("unknown robot")
        self.names = JOINT_NAMES
        self.base_from_model = inverse(model_from_controller_base(robot))
        self.tcp_offset = np.eye(4) if tcp_offset is None else inverse(inverse(tcp_offset))

    def links(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("joints must be six finite radians")
        return {name: self.base_from_model @ T for name, T in self.fk(dict(zip(self.names, q))).items()}

    def pose_and_jacobian(self, q, flange=False):
        links = self.links(q)
        tip = links["wrist_3_link"] @ (np.eye(4) if flange else self.tcp_offset)
        axes = np.stack([links[name][:3, 2] for name in JOINT_LINKS], axis=1)
        origins = np.stack([links[name][:3, 3] for name in JOINT_LINKS], axis=1)
        J = np.vstack((np.cross(axes.T, tip[:3, 3] - origins.T).T, axes))
        return tip, J

    def inverse(self, goal, seed, lower, upper, max_iterations=16):
        inverse(goal)  # validate SE(3)
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(max_iterations):
            current, J = self.pose_and_jacobian(q)
            dp, dr = pose_delta(current, goal)
            if np.linalg.norm(dp) < .0005 and np.linalg.norm(dr) < .002:
                return q if np.all(q >= lower) and np.all(q <= upper) else None
            error = np.concatenate((dp, dr))
            step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), error)
            norm = np.linalg.norm(step)
            if norm > .2:
                step *= .2 / norm  # numerical IK trust region, never an actuator command
            q += step
            if np.any(q < lower) or np.any(q > upper):
                return None
        return None

    def points(self, q):
        links = self.links(q)
        elbow_end = links["forearm_link"] @ pose_matrix([-self.forearm_length, 0, 0, 0, 0, 0])
        tip = links["wrist_3_link"] @ self.tcp_offset
        # Include the forearm's real elbow-to-wrist bend instead of a diagonal.
        return np.stack([links["base_link_inertia"][:3, 3], links["shoulder_link"][:3, 3],
                         links["forearm_link"][:3, 3], elbow_end[:3, 3],
                         *[links[name][:3, 3] for name in JOINT_LINKS[-3:]], tip[:3, 3]])


def segment_distance(a, b, c, d):
    """Minimum Euclidean distance of two closed 3D segments, including points."""
    a, b, c, d = (np.asarray(point, dtype=float) for point in (a, b, c, d))
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in (a, b, c, d)):
        raise ValueError("segment endpoints must be finite 3D points")
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
    if uu < 1e-18 and vv < 1e-18:
        return float(np.linalg.norm(w))
    if uu < 1e-18:
        t = np.clip(v @ w / vv, 0, 1)
        return float(np.linalg.norm(w - t * v))
    if vv < 1e-18:
        s = np.clip(-u @ w / uu, 0, 1)
        return float(np.linalg.norm(w + s * u))
    den = uu * vv - uv * uv
    s = np.clip((uv * (v @ w) - vv * (u @ w)) / den, 0, 1) if den > 1e-18 else 0.
    t = (uv * s + v @ w) / vv
    if t < 0:
        t, s = 0., np.clip(-u @ w / uu, 0, 1)
    elif t > 1:
        t, s = 1., np.clip((uv - u @ w) / uu, 0, 1)
    return float(np.linalg.norm(w + s * u - t * v))


class WorkspaceGuard:
    """Joint limits, TCP boxes, inter-arm capsules and camera-frame keep-outs.

    Optional JSON keys: joint_lower/upper (robot -> six radians), tcp_boxes
    (robot -> [minXYZ,maxXYZ] in its base), link_radius (robot -> metres),
    tool_radius (robot -> metres), margin (metres), obstacles ([{center,radius}]
    in the camera frame). Table geometry can be supplied as keep-out spheres.
    """

    def __init__(self, models, config_path=None):
        cfg = {} if config_path is None else json.loads(Path(config_path).read_text())
        if not isinstance(cfg, dict):
            raise ValueError("workspace config must be a JSON object")
        allowed = {"joint_lower", "joint_upper", "tcp_boxes", "link_radius", "tool_radius", "margin", "obstacles"}
        if set(cfg) - allowed:
            raise ValueError("unknown workspace config keys: " + str(sorted(set(cfg) - allowed)))
        self.models, self.lower, self.upper, self.boxes = models, {}, {}, {}
        self.radii, self.tool_radii = {}, {}
        for name in models:
            self.lower[name] = self._vector(cfg.get("joint_lower", {}).get(name, [-2*np.pi]*6), 6)
            self.upper[name] = self._vector(cfg.get("joint_upper", {}).get(name, [2*np.pi]*6), 6)
            if np.any(self.lower[name] >= self.upper[name]):
                raise ValueError("joint lower limits must be below upper limits")
            box = cfg.get("tcp_boxes", {}).get(name)
            if box is not None:
                lo, hi = (self._vector(row, 3) for row in box)
                if np.any(lo >= hi):
                    raise ValueError("invalid TCP workspace box")
                self.boxes[name] = (lo, hi)
            self.radii[name] = float(cfg.get("link_radius", {}).get(name, .06 if name == "ur5e" else .08))
            self.tool_radii[name] = float(cfg.get("tool_radius", {}).get(name, .04))
        self.margin = float(cfg.get("margin", .02))
        if not np.isfinite([*self.radii.values(), *self.tool_radii.values(), self.margin]).all() or min(*self.radii.values(), *self.tool_radii.values()) <= 0 or self.margin < 0:
            raise ValueError("radii must be positive and margin nonnegative")
        self.obstacles = []
        for entry in cfg.get("obstacles", []):
            center, radius = self._vector(entry["center"], 3), float(entry["radius"])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("obstacle radius must be finite and positive")
            self.obstacles.append((center, radius))

    @staticmethod
    def _vector(value, size):
        value = np.asarray(value, dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"expected {size} finite numbers")
        return value

    def pose_allowed(self, robot, pose):
        inverse(pose)
        if robot in self.boxes:
            lo, hi = self.boxes[robot]
            return bool(np.all(pose[:3, 3] >= lo) and np.all(pose[:3, 3] <= hi))
        return True

    def check(self, bases, joints):
        segments = {}
        minimum = float("inf")
        for name, model in self.models.items():
            inverse(bases[name])
            q = self._vector(joints[name], 6)
            if np.any(q < self.lower[name]) or np.any(q > self.upper[name]):
                return False, f"{name}: joint position limit", None
            tip, _ = model.pose_and_jacobian(q)
            if not self.pose_allowed(name, tip):
                return False, f"{name}: TCP workspace box", None
            points = model.points(q)
            points = points @ bases[name][:3, :3].T + bases[name][:3, 3]
            segments[name] = [(a, b, self.tool_radii[name] if i == len(points)-2 else self.radii[name])
                              for i, (a, b) in enumerate(zip(points[:-1], points[1:]))]
            for a, b, radius in segments[name]:
                for center, obstacle_radius in self.obstacles:
                    clearance = segment_distance(a, b, center, center) - radius - obstacle_radius
                    minimum = min(minimum, clearance)
                    if clearance < self.margin:
                        return False, f"{name}: obstacle clearance", clearance
        names = list(segments)
        for i, name in enumerate(names):
            for other in names[i+1:]:
                for a, b, ra in segments[name]:
                    for c, d, rb in segments[other]:
                        clearance = segment_distance(a, b, c, d) - ra - rb
                        minimum = min(minimum, clearance)
                        if clearance < self.margin:
                            return False, "inter-arm capsule clearance", clearance
        return True, "feasible", None if np.isinf(minimum) else minimum
