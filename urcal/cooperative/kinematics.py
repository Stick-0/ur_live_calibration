"""Nominal differential kinematics and geometric feasibility for research control.

Link capsules are configurable proxies, not mesh collision certification. The
robot controller still checks its own calibrated reachability and safety limits.
All public transforms use controller base / zero-TCP flange conventions.
"""

import json
from pathlib import Path

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base, pose_delta


JOINT_LINKS = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")


class NominalKinematics:
    def __init__(self, robot, tcp_offset=None):
        if robot == "ur5e":
            from urcal.inference.infer_live import JOINT_NAMES, compute_ur5e_link_transforms
            self.fk = compute_ur5e_link_transforms
            self.forearm_length = .3922
            self.terminal_wrist_length = .0996
        elif robot == "ur10e":
            from urcal.inference.infer_live_ur10e import JOINT_NAMES, compute_ur10e_link_transforms
            self.fk = compute_ur10e_link_transforms
            self.forearm_length = .57155
            self.terminal_wrist_length = .11655
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
        tip = links["wrist_3_link"] @ self.tcp_offset
        return self._points_from_links(links, tip)

    def pose_and_points(self, q):
        """TCP and capsule vertices from one FK evaluation of the same joints."""
        # Captured alignment/simulation models override these methods with a
        # measured flange correction. Preserve that public geometry unless
        # the subclass explicitly supplies its own combined implementation.
        if (type(self).pose_and_jacobian is not NominalKinematics.pose_and_jacobian
                or type(self).points is not NominalKinematics.points):
            return self.pose_and_jacobian(q)[0], self.points(q)
        links = self.links(q)
        tip = links["wrist_3_link"] @ self.tcp_offset
        return tip, self._points_from_links(links, tip)

    def _points_from_links(self, links, tip):
        elbow_end = (links["forearm_link"][:3, 3]
                     - self.forearm_length * links["forearm_link"][:3, 0])
        # Include the forearm's real elbow-to-wrist bend instead of a diagonal.
        return np.stack([links["base_link_inertia"][:3, 3], links["shoulder_link"][:3, 3],
                         links["forearm_link"][:3, 3], elbow_end,
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


def _segment_distances(a, b, c, d):
    """Broadcast the scalar segment calculation over arrays ending in XYZ.

    Keep point segments and parallel-segment clamping identical to
    ``segment_distance``. Workspace checks can inspect every capsule pair
    without repeating Python loops and endpoint validation for each pair.
    """
    endpoints = [np.asarray(point, dtype=float) for point in (a, b, c, d)]
    if any(point.ndim < 1 or point.shape[-1] != 3 or not np.isfinite(point).all()
           for point in endpoints):
        raise ValueError("segment endpoints must be finite 3D points")
    a, b, c, d = np.broadcast_arrays(*endpoints)
    u, v, w = b - a, d - c, a - c
    dot = lambda x, y: np.einsum("...i,...i->...", x, y)
    uu, vv, uv, uw, vw = dot(u, u), dot(v, v), dot(u, v), dot(u, w), dot(v, w)
    den = uu * vv - uv * uv
    s = np.zeros_like(den)
    np.divide(uv * vw - vv * uw, den, out=s, where=den > 1e-18)
    np.clip(s, 0., 1., out=s)
    t = np.zeros_like(den)
    np.divide(uv * s + vw, vv, out=t, where=vv >= 1e-18)
    below, above = t < 0., t > 1.
    endpoint_s = np.zeros_like(den)
    np.divide(-uw, uu, out=endpoint_s, where=uu >= 1e-18)
    s = np.where(below, np.clip(endpoint_s, 0., 1.), s)
    np.divide(uv - uw, uu, out=endpoint_s, where=uu >= 1e-18)
    s = np.where(above, np.clip(endpoint_s, 0., 1.), s)
    np.clip(t, 0., 1., out=t)
    # A point in either argument follows its scalar projection rule; a pair
    # of points retains zero parameters and its direct Euclidean distance.
    first_point, second_point = uu < 1e-18, vv < 1e-18
    s = np.where(first_point, 0., s)
    point_s = np.zeros_like(den)
    np.divide(-uw, uu, out=point_s, where=uu >= 1e-18)
    s = np.where(second_point, np.clip(point_s, 0., 1.), s)
    point_t = np.zeros_like(den)
    np.divide(vw, vv, out=point_t, where=vv >= 1e-18)
    t = np.where(first_point, np.clip(point_t, 0., 1.), t)
    t = np.where(second_point, 0., t)
    return np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1)


def _terminal_wrist_planes(model, tip, points):
    """Identify only known UR terminal body capsules and their flange plane.

    The nominal points have wrist_2/wrist_3 at indices 5/6. Captured FK
    retains those points and inserts its corrected flange before the TCP.
    Unknown point layouts keep the full capsules. The corrected flange plane
    can be tilted, but wrist_2 must remain on its proximal side.
    Inputs here share the camera frame; the TCP offset remains flange-local.
    """
    if not isinstance(model, NominalKinematics) or len(points) not in (8, 9):
        return {}
    axis = points[6] - points[5]
    length = float(np.linalg.norm(axis))
    if not np.isclose(length, model.terminal_wrist_length, atol=1e-9, rtol=0.):
        return {}
    flange = tip @ inverse(model.tcp_offset)
    if not np.allclose(flange[:3, 3], points[-2], atol=1e-9, rtol=0.):
        return {}
    correction = getattr(model, "flange_correction", None)
    nominal_flange = flange if correction is None else flange @ inverse(correction)
    if (not np.allclose(nominal_flange[:3, 3], points[6], atol=1e-9, rtol=0.)
            or not np.allclose(nominal_flange[:3, 2], axis/length, atol=1e-9, rtol=0.)):
        return {}
    axis = flange[:3, 2]
    if float(axis @ (flange[:3, 3] - points[5])) <= 0.:
        return {}
    plane = (axis, flange[:3, 3])
    return {index: plane for index in range(5, len(points)-2)}


def _capsule_support(a, b, radius, direction, clip=None):
    """Upper bound on support of a capsule cut by its actual flange plane.

    For S intersect {n.x <= c}, its support h(v) is at most
    h_S(v-lambda*n) + lambda*c for every lambda >= 0. A finite minimum of
    these upper bounds is conservative; no optimizer or exact mesh is needed.
    The flange-relative origin makes c zero and avoids subtracting large
    camera coordinates. In the axial case lambda=v.n gives the distal circle
    support, retaining the full cylinder radius and proximal hemisphere.
    """
    if clip is None:
        return max(float(direction @ a), float(direction @ b)) + radius*float(np.linalg.norm(direction))
    axis, plane_origin = clip
    first, last = a-plane_origin, b-plane_origin
    values = []
    for multiplier in (0., max(0., float(direction @ axis)), 1., 2.):
        shifted = direction-multiplier*axis
        values.append(max(float(shifted @ first), float(shifted @ last))
                      + radius*float(np.linalg.norm(shifted)))
    return float(direction @ plane_origin) + min(values)


def _terminal_pair_clearance(a, b, ra, c, d, rb, clip_a, clip_b, original):
    """Conservative separation certificate; never infer clearance from overlap.

    A projected interval gap in any unit direction is a lower bound on 3D
    surface distance. Taking the maximum of these bounds can certify a pair
    rejected solely by its fictitious distal hemisphere. Unproven pairs keep
    the old conservative rejection. No radius or clearance margin is reduced.
    """
    directions = [d-b]
    if clip_a is not None:
        directions.append(clip_a[0])
    if clip_b is not None:
        directions.append(clip_b[0])
    result = float(original)
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        unit = direction/norm
        for sign in (1., -1.):
            n = sign*unit
            gap = (-_capsule_support(c, d, rb, -n, clip_b)
                   - _capsule_support(a, b, ra, n, clip_a))
            result = max(result, float(gap))
    return result


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
        terminal_geometry, clip_planes = {}, {}
        minimum = float("inf")
        for name, model in self.models.items():
            inverse(bases[name])
            q = self._vector(joints[name], 6)
            if np.any(q < self.lower[name]) or np.any(q > self.upper[name]):
                return False, f"{name}: joint position limit", None
            geometry = getattr(model, "pose_and_points", None)
            if geometry is None:
                tip, _ = model.pose_and_jacobian(q)
                points = None
            else:
                tip, points = geometry(q)
            if not self.pose_allowed(name, tip):
                return False, f"{name}: TCP workspace box", None
            if points is None:
                points = model.points(q)
            points = points @ bases[name][:3, :3].T + bases[name][:3, 3]
            radii = np.full(max(0, len(points) - 1), self.radii[name])
            if len(radii):
                radii[-1] = self.tool_radii[name]
            segments[name] = (points[:-1], points[1:], radii)
            terminal_geometry[name] = (model, bases[name] @ tip, points)
            if self.obstacles and len(radii):
                centers = np.asarray([entry[0] for entry in self.obstacles])
                obstacle_radii = np.asarray([entry[1] for entry in self.obstacles])
                clearances = (_segment_distances(points[:-1, None], points[1:, None], centers, centers)
                              - radii[:, None] - obstacle_radii)
                violated = np.flatnonzero(clearances < self.margin)
                if len(violated):
                    return False, f"{name}: obstacle clearance", float(clearances.flat[violated[0]])
                minimum = min(minimum, float(np.min(clearances)))
        names = list(segments)
        for i, name in enumerate(names):
            for other in names[i+1:]:
                a, b, ra = segments[name]
                c, d, rb = segments[other]
                if not len(ra) or not len(rb):
                    continue
                clearances = _segment_distances(a[:, None], b[:, None], c, d) - ra[:, None] - rb
                violated = np.flatnonzero(clearances < self.margin)
                # Common distant pairs retain the batched capsule-only path.
                # Refine failed pairs only, using known terminal body geometry.
                if len(violated):
                    for robot in (name, other):
                        if robot not in clip_planes:
                            clip_planes[robot] = _terminal_wrist_planes(*terminal_geometry[robot])
                    for flat_index in violated:
                        ia, ib = np.unravel_index(flat_index, clearances.shape)
                        ca, cb = clip_planes[name].get(ia), clip_planes[other].get(ib)
                        if ca is not None or cb is not None:
                            clearances[ia, ib] = _terminal_pair_clearance(
                                a[ia], b[ia], ra[ia], c[ib], d[ib], rb[ib],
                                ca, cb, clearances[ia, ib])
                    violated = np.flatnonzero(clearances < self.margin)
                if len(violated):
                    return False, "inter-arm capsule clearance", float(clearances.flat[violated[0]])
                minimum = min(minimum, float(np.min(clearances)))
        return True, "feasible", None if np.isinf(minimum) else minimum
