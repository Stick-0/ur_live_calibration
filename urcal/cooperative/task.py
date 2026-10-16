"""Frozen relative paths and a governor that slows task progress, not compensation.

All twists here are geometric twists ``[origin velocity, angular velocity]``;
the linear part is the velocity of the named frame's origin, not a spatial
twist's moment about the coordinate origin. There are no robot/model imports
or hardware side effects.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, matrix_pose
from urcal.control.policy import ControlFault


def _scalar(value, name, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()


def _transform(value, name):
    result = _array(value, (4, 4), name)
    matrix_pose(result)  # Validate homogeneous row and proper SO(3) rotation.
    return result


class RelativePath:
    """A relative follower-flange path fixed in the leader-flange coordinates.

    ``pose(u)`` takes dimensionless progress in [0, 1]. ``derivative(u)``
    returns the geometric twist at nominal progress rate ``du/dt=1/duration``.
    An executing caller advances u by ``rho * dt / duration`` only after a
    successful command; rendering this path never changes its progress.

    Line translation and its left-multiplied rotation delta use the leader's
    flange axes. Rotation follows the shortest equivalent SO(3) delta. Circle
    makes one XY lap with offset [r*(cos(theta)-1), r*sin(theta), 0], keeping
    the starting orientation. Quintic easing gives zero endpoint velocities
    and accelerations. The initial relative pose is copied and never recaptured.
    """

    def __init__(self, start, kind="hold", delta=(0.,) * 6, radius=0.02, duration=10.0):
        if kind not in ("hold", "line", "circle"):
            raise ValueError("path kind must be hold, line, or circle")
        self._start = _transform(start, "relative path start")
        self._start.setflags(write=False)
        self._delta = _array(delta, (6,), "relative path delta")
        self._delta[3:] = Rotation.from_rotvec(self._delta[3:]).as_rotvec()
        self._delta.setflags(write=False)
        self.kind = kind
        self.radius = _scalar(radius, "circle radius", positive=True)
        self.duration = _scalar(duration, "path duration", positive=True)

    @property
    def start(self):
        return self._start.copy()

    @property
    def delta(self):
        return self._delta.copy()

    @staticmethod
    def _phase(progress):
        u = _scalar(progress, "path progress")
        if not 0 <= u <= 1:
            raise ValueError("path progress must be in [0, 1]")
        s = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        ds_du = 30.0 * u**2 * (1.0 - u)**2
        return u, s, ds_du

    def pose(self, progress):
        u, s, _ = self._phase(progress)
        result = self._start.copy()
        if self.kind == "line":
            result[:3, 3] += s * self._delta[:3]
            result[:3, :3] = Rotation.from_rotvec(s * self._delta[3:]).as_matrix() @ self._start[:3, :3]
        elif self.kind == "circle" and u not in (0., 1.):
            angle = 2.0 * np.pi * s
            result[:3, 3] += self.radius * np.array([np.cos(angle) - 1.0, np.sin(angle), 0.])
        return result

    def derivative(self, progress):
        _u, s, ds_du = self._phase(progress)
        rate = ds_du / self.duration
        result = np.zeros(6)
        if self.kind == "line":
            result = self._delta * rate
        elif self.kind == "circle":
            angle = 2.0 * np.pi * s
            result[:3] = 2.0 * np.pi * self.radius * rate * np.array([-np.sin(angle), np.cos(angle), 0.])
        return result


def target_components(base_poses, base_twists, leader, follower,
                      leader_flange_local, leader_flange_twist_local,
                      relative_target, path_derivative, follower_tcp_offset):
    """Return (target TCP, base compensation, leader motion, task motion) in Bf.

    ``base_poses[name]`` is T_camera_controller_base. ``base_twists[name]``
    contains that base origin's linear/angular velocity expressed in CAMERA.
    ``leader_flange_local`` is T_leader_base_leader_flange, and its twist is
    the flange origin's velocity relative to that base, expressed in LEADER
    BASE. ``relative_target`` is T_leader_flange_follower_flange; its derivative
    is a nominal per-second geometric twist expressed in LEADER FLANGE.

    The returned twists are velocities of the desired active TCP origin in
    FOLLOWER BASE. The base component includes both bases' translations and
    rotations. Leader and task components include the lever arm to the TCP,
    including a rotated, nonzero active TCP offset. Sum the base and leader
    terms as mandatory compensation; only the task term is scaled by rho.
    """
    if leader == follower:
        raise ValueError("leader and follower must be different robots")
    try:
        A = _transform(base_poses[leader], "leader base pose")
        B = _transform(base_poses[follower], "follower base pose")
        base_leader = _array(base_twists[leader], (6,), "leader base twist")
        base_follower = _array(base_twists[follower], (6,), "follower base twist")
    except KeyError as exc:
        raise ValueError("both robots require base poses and base twists") from exc
    L = _transform(leader_flange_local, "leader local flange pose")
    S = _transform(relative_target, "relative flange target")
    O = _transform(follower_tcp_offset, "follower TCP offset")
    local_leader = _array(leader_flange_twist_local, (6,), "leader local flange twist")
    local_task = _array(path_derivative, (6,), "relative path derivative")

    T_C_L = A @ L
    T_C_target = T_C_L @ S @ O
    target_tcp = inverse(B) @ T_C_target
    camera_to_follower = B[:3, :3].T
    target_position = T_C_target[:3, 3]
    base_linear = (base_leader[:3] + np.cross(base_leader[3:], target_position - A[:3, 3])
                   - base_follower[:3] - np.cross(base_follower[3:], target_position - B[:3, 3]))
    base_angular = base_leader[3:] - base_follower[3:]
    base_component = np.r_[camera_to_follower @ base_linear, camera_to_follower @ base_angular]

    tcp_from_leader_flange = S[:3, 3] + S[:3, :3] @ O[:3, 3]
    leader_to_follower = camera_to_follower @ A[:3, :3]
    leader_linear = local_leader[:3] + np.cross(local_leader[3:], L[:3, :3] @ tcp_from_leader_flange)
    leader_component = np.r_[leader_to_follower @ leader_linear, leader_to_follower @ local_leader[3:]]

    flange_to_follower = camera_to_follower @ T_C_L[:3, :3]
    task_linear = local_task[:3] + np.cross(local_task[3:], S[:3, :3] @ O[:3, 3])
    task_component = np.r_[flange_to_follower @ task_linear, flange_to_follower @ local_task[3:]]
    if not np.isfinite(np.r_[target_tcp.ravel(), base_component, leader_component, task_component]).all():
        raise ValueError("target components are not finite")
    return target_tcp, base_component, leader_component, task_component


@dataclass(frozen=True)
class MotionLimits:
    linear_speed: float = 0.02
    angular_speed: float = 0.1
    linear_accel: float = 0.1
    angular_accel: float = 0.5
    joint_speed: np.ndarray = field(default_factory=lambda: np.full(6, 0.2))
    joint_accel: np.ndarray = field(default_factory=lambda: np.full(6, 0.5))
    min_singular_value: float = 0.01

    def __post_init__(self):
        for name in ("linear_speed", "angular_speed", "linear_accel", "angular_accel", "min_singular_value"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, positive=True))
        for name in ("joint_speed", "joint_accel"):
            values = _array(getattr(self, name), (6,), name)
            if np.any(values <= 0):
                raise ValueError(f"{name} must contain six positive limits")
            values.setflags(write=False)
            object.__setattr__(self, name, values)


class MotionInfeasible(ControlFault):
    """A pre-command feasibility failure, classified for recording/stop logic."""

    def __init__(self, reason, category):
        self.category = category
        super().__init__(reason)


@dataclass(frozen=True)
class MotionDecision:
    twist: np.ndarray
    qdot: np.ndarray
    task_scale: float
    reason: str
    components: Dict[str, object]


def govern_task(compensation, feedback, task, J, previous_twist, previous_qdot, dt,
                limits, rho_cap=1.0,
                extra_feasible: Optional[Callable[[float, np.ndarray, np.ndarray], bool]] = None,
                grid_samples=101, previous_compensation=None,
                previous_compensation_qdot=None, allow_deceleration=False):
    """Choose rho before sending ``compensation + feedback + rho*task``.

    Compensation is never reduced or clipped. Its acceleration is checked
    against its OWN previous component, which defaults to zero, not the prior
    total command. Its prior joint component defaults to J-solved previous
    compensation. Pass both histories when J changes between control samples.

    Cartesian norm balls and per-joint intervals give an exact convex feasible
    interval along rho. Acceleration may impose a positive lower bound while
    decelerating an established task; zero is not assumed feasible. Normally
    the maximum feasible rho <= rho_cap is selected. If allow_deceleration is
    True and only acceleration prevents honoring the requested cap, the lowest
    feasible rho is used temporarily, with reason ``task_decelerating``. This
    never relaxes physical bounds or a cap infeasible even without acceleration.

    A nonlinear ``extra_feasible`` callback can check a candidate's next phase,
    workspace, or collision margin. A bounded grid selects the highest sampled
    feasible rho (lowest during deceleration). Narrow feasible intervals may be
    missed; callback convexity is not assumed. It receives copies so it cannot
    alter commands. No compensation, feedback, or output component is clipped.

    J maps joint velocity to geometric active-TCP twist in follower-base axes.
    Singular Jacobians are rejected before any solve/command. Returned qdot is
    a feasibility prediction, not a replacement for measured joint velocities.
    """
    if not isinstance(limits, MotionLimits):
        raise ValueError("limits must be MotionLimits")
    compensation = _array(compensation, (6,), "compensation")
    feedback = _array(feedback, (6,), "feedback")
    task = _array(task, (6,), "task twist")
    previous_twist = _array(previous_twist, (6,), "previous twist")
    previous_qdot = _array(previous_qdot, (6,), "previous joint velocity")
    previous_compensation = _array(np.zeros(6) if previous_compensation is None else previous_compensation,
                                   (6,), "previous compensation")
    J = _array(J, (6, 6), "Jacobian")
    dt = _scalar(dt, "control interval", positive=True)
    rho_cap = _scalar(rho_cap, "task rate cap")
    if not 0 <= rho_cap <= 1:
        raise ValueError("task rate cap must be in [0, 1]")
    if not isinstance(grid_samples, Integral) or isinstance(grid_samples, bool) or grid_samples < 2:
        raise ValueError("grid_samples must be an integer of at least 2")
    U, singular_values, Vh = np.linalg.svd(J)
    if singular_values[-1] < limits.min_singular_value:
        raise MotionInfeasible(
            f"Jacobian singular value {singular_values[-1]:.6g} is below {limits.min_singular_value:.6g}",
            "singularity",
        )

    def joint_velocity(twist):
        result = Vh.T @ ((U.T @ twist) / singular_values)
        if not np.isfinite(result).all():
            raise MotionInfeasible("joint velocity solve is not finite", "singularity")
        return result

    previous_compensation_qdot = (
        joint_velocity(previous_compensation) if previous_compensation_qdot is None
        else _array(previous_compensation_qdot, (6,), "previous compensation joint velocity")
    )

    def violations(twist, qdot, previous, previous_joints):
        checks = (
            ("linear speed", np.linalg.norm(twist[:3]), limits.linear_speed),
            ("angular speed", np.linalg.norm(twist[3:]), limits.angular_speed),
            ("linear acceleration", np.linalg.norm(twist[:3] - previous[:3]), limits.linear_accel * dt),
            ("angular acceleration", np.linalg.norm(twist[3:] - previous[3:]), limits.angular_accel * dt),
        )
        for label, value, bound in checks:
            if not np.isfinite(value) or value > bound + 1e-12:
                return label
        if np.any(np.abs(qdot) > limits.joint_speed + 1e-12):
            return "joint speed"
        if np.any(np.abs(qdot - previous_joints) > limits.joint_accel * dt + 1e-12):
            return "joint acceleration"
        return None

    compensation_qdot = joint_velocity(compensation)
    failure = violations(compensation, compensation_qdot, previous_compensation, previous_compensation_qdot)
    if failure:
        raise MotionInfeasible(f"compensation alone exceeds {failure}", "compensation_infeasible")
    baseline = compensation + feedback
    baseline_qdot = joint_velocity(baseline)
    task_qdot = joint_velocity(task)

    def candidate(rho):
        return baseline + rho * task, baseline_qdot + rho * task_qdot

    def norm_interval(a, b, bound):
        """Solve ||a + rho*b|| <= bound without assuming rho=0 is feasible."""
        if not np.isfinite(bound):
            raise ValueError("velocity/acceleration bound overflow")
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), abs(bound), np.finfo(float).tiny)
        an, bn, rn = a / scale, b / scale, bound / scale
        quadratic = float(bn @ bn)
        if quadratic == 0.:
            return (-np.inf, np.inf) if np.linalg.norm(a) <= bound + 1e-12 else None
        linear = 2. * float(an @ bn)
        constant = float(an @ an) - rn * rn
        discriminant = linear * linear - 4. * quadratic * constant
        rounding = 16. * np.finfo(float).eps * max(linear * linear, abs(4. * quadratic * constant), 1.)
        if discriminant < -rounding:
            return None
        root = np.sqrt(max(0., discriminant))
        # This form avoids subtracting almost equal values for one of the roots.
        q = -0.5 * (linear + np.copysign(root, linear))
        if q == 0.:
            first = second = -linear / (2. * quadratic)
        else:
            first, second = q / quadratic, constant / q
        return min(first, second), max(first, second)

    def intersect(interval, constraints):
        lower, upper = interval
        for a, b, bound in constraints:
            permitted = norm_interval(np.atleast_1d(a), np.atleast_1d(b), float(bound))
            if permitted is None:
                return None
            lower, upper = max(lower, permitted[0]), min(upper, permitted[1])
            if lower > upper:
                return None
        return lower, upper

    static_constraints = [
        (baseline[:3], task[:3], limits.linear_speed),
        (baseline[3:], task[3:], limits.angular_speed),
        *((baseline_qdot[i], task_qdot[i], limits.joint_speed[i]) for i in range(6)),
    ]
    static_interval = intersect((0., 1.), static_constraints)
    if static_interval is None or static_interval[0] > rho_cap:
        raise MotionInfeasible("compensation/feedback/task cannot meet speed limits at the requested rate cap",
                               "tracking_infeasible")
    acceleration_constraints = [
        (baseline[:3] - previous_twist[:3], task[:3], limits.linear_accel * dt),
        (baseline[3:] - previous_twist[3:], task[3:], limits.angular_accel * dt),
        *((baseline_qdot[i] - previous_qdot[i], task_qdot[i], limits.joint_accel[i] * dt) for i in range(6)),
    ]
    interval = intersect(static_interval, acceleration_constraints)
    if interval is None:
        raise MotionInfeasible("no task rate can meet the velocity transition limits", "transition_infeasible")
    lower, upper = interval
    decelerating = lower > rho_cap
    if decelerating and not allow_deceleration:
        raise MotionInfeasible("requested task rate cap cannot be reached within acceleration limits",
                               "transition_infeasible")
    if not decelerating:
        upper = min(upper, rho_cap)
    rho = lower if decelerating else upper
    if extra_feasible is not None:
        rho = None
        candidates = np.linspace(lower, upper, int(grid_samples)) if decelerating else np.linspace(upper, lower, int(grid_samples))
        for sampled in candidates:
            twist, qdot = candidate(float(sampled))
            if bool(extra_feasible(float(sampled), twist.copy(), qdot.copy())):
                rho = float(sampled)
                break
        if rho is None:
            raise MotionInfeasible("no sampled task rate meets workspace/collision feasibility", "tracking_infeasible")
    twist, qdot = candidate(rho)
    failure = violations(twist, qdot, previous_twist, previous_qdot)
    if failure:
        raise MotionInfeasible(f"candidate fails final {failure} verification", "transition_infeasible")
    reason = ("task_decelerating" if decelerating else "task_paused" if rho <= 1e-12
              else "full_rate" if rho >= 1.0 - 1e-12 else "task_slowed")
    return MotionDecision(twist, qdot, float(rho), reason, {
        "compensation": compensation.copy(),
        "feedback": feedback.copy(),
        "task_requested": task.copy(),
        "task_applied": rho * task,
        "singular_values": singular_values.copy(),
        "compensation_qdot": compensation_qdot.copy(),
        "requested_task_scale": rho_cap,
        "feasible_task_interval": np.asarray(interval),
    })
