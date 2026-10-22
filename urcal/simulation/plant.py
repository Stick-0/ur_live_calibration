"""Joint-velocity simulation initialized from captured UR encoder/TCP poses.

This is a kinematic integrator, not a dynamics or factory-calibration model.
Every model retains its nominal links and a fixed local flange correction that
matches the captured controller TCP exactly at the captured joints. Away from
that pose the correction is an approximation. No robot driver is constructed.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import inverse, pose_delta
from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard


ROBOTS = ("ur5e", "ur10e")


class SimulationFault(ValueError):
    """A virtual step was rejected without changing the plant state."""


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SimulationFault(f"{label}: expected six finite values") from exc
    if result.shape != (6,) or not np.isfinite(result).all():
        raise SimulationFault(f"{label}: expected six finite values")
    return result.copy()


def _transform(value):
    inverse(value)
    return np.asarray(value, dtype=float).copy()


def _positive(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationFault(f"{label} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise SimulationFault(f"{label} must be finite and positive")
    return result


class SeededKinematics(NominalKinematics):
    """Nominal links with a constant local tool-flange correction at capture.

    ``tcp0`` and ``tcp_offset`` are respectively T_base_TCP and T_flange_TCP.
    The correction is T_nominal_flange_actual_flange, so both the corrected
    flange and active TCP geometric Jacobians include the additional lever arm.
    Numerical IK uses this same corrected FK/Jacobian. The short nominal solver
    is tried first, followed by a bounded least-squares solve when it cannot
    converge or its intermediate joint branch leaves the joint limits.
    """

    def __init__(self, robot, q0, tcp0, tcp_offset):
        q0 = _vector(q0, "initial joints")
        tcp0, tcp_offset = _transform(tcp0), _transform(tcp_offset)
        super().__init__(robot, tcp_offset)
        nominal, _ = super().pose_and_jacobian(q0, flange=True)
        self.flange_correction = inverse(nominal) @ tcp0 @ inverse(tcp_offset)

    def pose_and_jacobian(self, q, flange=False):
        nominal, jacobian = super().pose_and_jacobian(q, flange=True)
        tip = nominal @ self.flange_correction
        if not flange:
            tip = tip @ self.tcp_offset
        result = jacobian.copy()
        shift = tip[:3, 3] - nominal[:3, 3]
        result[:3] += np.cross(jacobian[3:].T, shift).T
        return tip, result

    def points(self, q):
        # The original wrist endpoint is retained. Only the last nominal TCP
        # point is replaced with separate corrected-flange and actual-TCP tips.
        links = super().points(q)[:-1]
        flange, _ = self.pose_and_jacobian(q, flange=True)
        tcp = flange @ self.tcp_offset
        return np.vstack((links, flange[:3, 3], tcp[:3, 3]))

    def inverse(self, goal, seed, lower, upper, max_iterations=16):
        """Find a bounded virtual-model solution, without certifying its path.

        A short local Jacobian solve can fail for reachable goals near a wrist
        singularity, either by running out of iterations or leaving the bounds
        along one branch. A bounded trust-region solve can find another local
        solution. Its termination flag alone is insufficient: the existing
        position/rotation tolerances and joint limits must also be satisfied.
        This fallback is confined to the simulator, with no hardware IK calls.
        """
        goal = _transform(goal)
        seed = _vector(seed, "IK seed")
        lower, upper = _vector(lower, "IK lower bounds"), _vector(upper, "IK upper bounds")
        if np.any(lower >= upper):
            raise SimulationFault("IK lower bounds must be below upper bounds")
        if np.any(seed < lower) or np.any(seed > upper):
            return None
        result = super().inverse(goal, seed, lower, upper, max_iterations=max_iterations)
        if result is not None:
            return result
        from scipy.optimize import least_squares

        def residual(q):
            current, _ = self.pose_and_jacobian(q)
            return np.concatenate(pose_delta(current, goal))

        solution = least_squares(residual, seed, bounds=(lower, upper), max_nfev=256,
                                 xtol=1e-9, ftol=1e-9, gtol=1e-9)
        error = residual(solution.x)
        if (np.isfinite(solution.x).all() and np.all(solution.x >= lower) and np.all(solution.x <= upper)
                and np.linalg.norm(error[:3]) < .0005 and np.linalg.norm(error[3:]) < .002):
            return solution.x.copy()
        return None


class JointPlant:
    """Two virtual arms with atomic midpoint/end-point feasibility checks.

    Commands are piecewise-constant joint velocities. Captured velocities are
    deliberately not replayed: loading/reset starts both virtual arms at rest.
    ``stop`` freezes the virtual state immediately, rather than simulating an
    actuator's braking dynamics. The same workspace proxies as the research
    controller are retained; they are not mesh or swept-volume certification.
    """

    def __init__(self, initial, args, models=None, guard=None):
        try:
            self._initial_bases = {name: _transform(initial["bases"][name]) for name in ROBOTS}
            self._initial_joints = {name: _vector(initial["robots"][name]["q"], name + " joints")
                                    for name in ROBOTS}
            self.offsets = {name: _transform(initial["robots"][name]["tcp_offset"]) for name in ROBOTS}
            initial_tcp = {name: _transform(initial["robots"][name]["tcp"]) for name in ROBOTS}
        except (KeyError, TypeError, ValueError) as exc:
            raise SimulationFault("invalid paired simulation initial state: " + str(exc)) from exc
        self.max_joint_speed = _positive(args.max_joint_speed, "max_joint_speed")
        self.max_joint_accel = _positive(args.max_joint_accel, "max_joint_accel")
        self.models = (models if models is not None else {
            name: SeededKinematics(name, self._initial_joints[name], initial_tcp[name], self.offsets[name])
            for name in ROBOTS})
        if set(self.models) != set(ROBOTS):
            raise SimulationFault("simulation requires both robot models")
        self.guard = guard if guard is not None else WorkspaceGuard(self.models, getattr(args, "workspace_config", None))
        for name in ROBOTS:
            lower = _vector(self.guard.lower[name], name + " lower limits")
            upper = _vector(self.guard.upper[name], name + " upper limits")
            if np.any(lower >= upper):
                raise SimulationFault(name + ": joint lower limits must be below upper limits")
        self.reset()

    def reset(self):
        """Restore the capture, independent of caller-owned snapshot arrays."""
        self.bases = {name: value.copy() for name, value in self._initial_bases.items()}
        self.joints = {name: value.copy() for name, value in self._initial_joints.items()}
        self.joint_velocity = {name: np.zeros(6) for name in ROBOTS}

    def stop(self):
        """Pause the virtual integrator; this sends no robot stop command."""
        self.joint_velocity = {name: np.zeros(6) for name in ROBOTS}

    def states(self, now):
        from urcal.control.runtime import RobotState

        now = float(now)
        if not np.isfinite(now):
            raise SimulationFault("simulation timestamp must be finite")
        result = {}
        for name in ROBOTS:
            pose, jacobian = self.models[name].pose_and_jacobian(self.joints[name])
            pose = _transform(pose)
            if np.asarray(jacobian).shape != (6, 6) or not np.isfinite(jacobian).all():
                raise SimulationFault(name + ": invalid model Jacobian")
            result[name] = RobotState(tcp=pose, joints=self.joints[name].copy(),
                                      speed=jacobian @ self.joint_velocity[name],
                                      joint_velocity=self.joint_velocity[name].copy(), robot_timestamp=now)
        return result

    @staticmethod
    def _pair_names(values, label):
        if not isinstance(values, dict) or set(values) - set(ROBOTS):
            raise SimulationFault(label + " must map known robot names to values")

    def twists_to_qdots(self, twists, dt):
        """Virtual hold servo: damped least-squares TCP twist to joint velocity.

        One interpolation factor from the previous to the requested joint
        velocity satisfies every joint speed and slew limit. Unlike clipping
        each joint independently, this keeps the achieved Cartesian twist on
        the line between the previous and requested twists (up to damping).
        The resulting twist can differ near a limit/singularity.
        This conversion does not step the plant or authorize physical motion.
        Follow mode instead passes the research governor's qdot to ``step``.
        """
        self._pair_names(twists, "twists")
        dt = _positive(dt, "control period")
        result = {}
        for name in ROBOTS:
            desired = np.zeros(6)
            if name in twists:
                twist = _vector(twists[name], name + " TCP twist")
                _, jacobian = self.models[name].pose_and_jacobian(self.joints[name])
                if np.asarray(jacobian).shape != (6, 6) or not np.isfinite(jacobian).all():
                    raise SimulationFault(name + ": invalid model Jacobian")
                desired = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), twist)
            previous = self.joint_velocity[name]
            delta = desired - previous
            change = float(np.max(np.abs(delta)))
            alpha = min(1., self.max_joint_accel * dt / change) if change else 1.
            for value, direction in zip(previous, delta):
                if direction > 0:
                    alpha = min(alpha, (self.max_joint_speed - value) / direction)
                elif direction < 0:
                    alpha = min(alpha, (-self.max_joint_speed - value) / direction)
            result[name] = previous + max(0., alpha) * delta
        return result

    def step(self, qdots, dt, bases=None):
        """Validate the entire pair before atomically accepting one time step.

        Optional bases are target camera-from-base poses for this step. Their
        translation and shortest rotation are interpolated for the midpoint
        workspace check. Unspecified arm commands mean zero joint velocity.
        On any rejected command, no joints, velocities or bases are changed.
        """
        self._pair_names(qdots, "joint velocities")
        dt = _positive(dt, "control period")
        proposed_bases = {name: pose.copy() for name, pose in self.bases.items()}
        if bases is not None:
            if not isinstance(bases, dict) or set(bases) != set(ROBOTS):
                raise SimulationFault("base update must contain both robots")
            proposed_bases = {name: _transform(bases[name]) for name in ROBOTS}
        velocities, proposed_joints = {}, {}
        for name in ROBOTS:
            velocity = _vector(qdots.get(name, np.zeros(6)), name + " joint velocity")
            q = _vector(self.joints[name], name + " joints")
            if np.any(np.abs(velocity) > self.max_joint_speed + 1e-12):
                raise SimulationFault(name + ": joint speed limit")
            if np.any(np.abs(velocity - self.joint_velocity[name]) > self.max_joint_accel * dt + 1e-12):
                raise SimulationFault(name + ": joint acceleration limit")
            candidate = q + velocity * dt
            if not np.isfinite(candidate).all():
                raise SimulationFault(name + ": invalid integrated joints")
            if (np.any(candidate < self.guard.lower[name]) or np.any(candidate > self.guard.upper[name])):
                raise SimulationFault(name + ": joint position limit")
            velocities[name], proposed_joints[name] = velocity, candidate
        for fraction in (.5, 1.):
            trial_joints = {name: self.joints[name] + velocities[name] * dt * fraction for name in ROBOTS}
            trial_bases = {}
            for name in ROBOTS:
                before, after = self.bases[name], proposed_bases[name]
                trial = before.copy()
                trial[:3, 3] += fraction * (after[:3, 3] - before[:3, 3])
                angle = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec()
                trial[:3, :3] = Rotation.from_rotvec(fraction * angle).as_matrix() @ before[:3, :3]
                trial_bases[name] = trial
            allowed, reason, _ = self.guard.check(trial_bases, trial_joints)
            if not allowed:
                raise SimulationFault("virtual step rejected: " + reason)
        self.joints, self.joint_velocity, self.bases = proposed_joints, velocities, proposed_bases
