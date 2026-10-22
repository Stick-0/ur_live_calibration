"""Measured-state alignment to robot-controller-provided IK goals.

This module owns no transport and imports no robot SDK. The caller checks
native controller safety for samples and predicted steps, sends joint velocity
requests, and commits only after successful send. Link capsules are proxies.
"""

import numpy as np

from urcal.control.geometry import inverse, rotation_distance
from urcal.control.policy import ControlFault, PlanningUnavailable, stationary_velocity_limits
from urcal.cooperative.kinematics import NominalKinematics, WorkspaceGuard


ROBOTS = ("ur5e", "ur10e")
# Command budgets leave room for servo tracking and measured-speed variation.
# This is a design reserve, not a bound on hardware response. Actual measured
# speed still uses the unchanged configured hard limits.
COMMAND_SPEED_FRACTION_MAX = .9


class AlignmentFault(ControlFault):
    category = "initial_alignment"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = {} if details is None else dict(details)


class AlignmentPlanningFault(AlignmentFault, PlanningUnavailable):
    """No valid alignment command/path; stop before offering a new Arm."""


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlignmentFault(label + " must contain six finite values") from exc
    if result.shape != (6,) or not np.isfinite(result).all():
        raise AlignmentFault(label + " must contain six finite values")
    return result.copy()


def _pose(value, label):
    try:
        inverse(value)
        return np.asarray(value, dtype=float).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlignmentFault(label + " must be a finite rigid transform") from exc


class CapturedKinematics(NominalKinematics):
    """Local flange correction matches the initial measured TCP, not factory FK."""

    def __init__(self, robot, q, tcp, offset):
        super().__init__(robot, offset)
        nominal, _ = super().pose_and_jacobian(q, flange=True)
        self.flange_correction = inverse(nominal) @ tcp @ inverse(offset)

    def pose_and_jacobian(self, q, flange=False):
        nominal, jacobian = super().pose_and_jacobian(q, flange=True)
        tip = nominal @ self.flange_correction
        if not flange:
            tip = tip @ self.tcp_offset
        result = jacobian.copy()
        result[:3] += np.cross(jacobian[3:].T, tip[:3, 3] - nominal[:3, 3]).T
        return tip, result

    def points(self, q):
        points = super().points(q)[:-1]
        flange, _ = self.pose_and_jacobian(q, flange=True)
        return np.vstack((points, flange[:3, 3], (flange @ self.tcp_offset)[:3, 3]))


def _ball_interval(origin, direction, center, radius, interval):
    """Intersect alpha with ||origin + alpha*direction-center|| <= radius."""
    offset = origin - center
    a, b = float(direction @ direction), float(offset @ direction)
    c = float(offset @ offset - radius * radius)
    if a < 1e-24:
        return interval if c <= 1e-14 else None
    discriminant = b * b - a * c
    if discriminant < -1e-18:
        return None
    root = np.sqrt(max(0., discriminant))
    low, high = max(interval[0], (-b - root) / a), min(interval[1], (-b + root) / a)
    return (low, high) if low <= high + 1e-12 else None


class LiveAlignment:
    """Joint-corridor feedback with frozen bases and stopped uncommanded arms.

    Native IK goals are supplied by the transport owner. Samples use an
    approximate captured kinematic model; completion uses measured TCP poses.
    No clock-driven reference advances ahead of the measured robot.
    """

    def __init__(self, args, offsets, bases, states, targets, goal_joints, guard=None):
        if (set(offsets) != set(ROBOTS) or set(bases) != set(ROBOTS) or set(states) != set(ROBOTS)
                or not targets or set(targets) - set(ROBOTS) or set(goal_joints) != set(targets)):
            raise AlignmentFault("alignment needs both measured arms and matching native IK targets")
        self.commanded = tuple(name for name in ROBOTS if name in targets)
        self.fixed = tuple(name for name in ROBOTS if name not in targets)
        self.leader_rest_limits = stationary_velocity_limits(args, leader=bool(self.fixed))
        self.offsets = {name: _pose(offsets[name], name + " TCP offset") for name in ROBOTS}
        self.initial_bases = {name: _pose(bases[name], name + " base") for name in ROBOTS}
        self.initial_joints = {name: _vector(states[name].joints, name + " initial joints") for name in ROBOTS}
        self.initial_tcp = {name: _pose(states[name].tcp, name + " initial TCP") for name in ROBOTS}
        self.initial_flange = {name: self.initial_tcp[name] @ inverse(self.offsets[name]) for name in ROBOTS}
        self.targets = {name: _pose(targets[name], name + " target") for name in self.commanded}
        self.expected_targets = {name: value.copy() for name, value in self.targets.items()}
        self.goal_joints = {name: (_vector(goal_joints[name], name + " native IK") if name in self.commanded
                                   else self.initial_joints[name].copy()) for name in ROBOTS}
        self.limits = {key: float(getattr(args, key)) for key in (
            "max_joint_speed", "max_joint_accel", "max_speed", "max_angular_speed", "max_accel",
            "max_angular_accel", "max_travel", "min_gap", "frame_check_m", "frame_check_deg", "max_robot_age")}
        if not all(np.isfinite(value) and value > 0 for value in self.limits.values()):
            raise AlignmentFault("alignment limits must be finite and positive")
        # Legacy saved configurations predate this option. Only the alignment
        # current/future command speed budgets change; measured hard limits
        # and every acceleration constraint retain their original values.
        try:
            self.speed_scale = float(getattr(args, "align_speed_scale", 1.))
        except (ValueError, TypeError, OverflowError) as exc:
            raise AlignmentFault("align_speed_scale must be finite and in (0, 2]") from exc
        if not np.isfinite(self.speed_scale) or not 0 < self.speed_scale <= 2:
            raise AlignmentFault("align_speed_scale must be finite and in (0, 2]")
        self.command_speed_fraction = min(.5 * self.speed_scale, COMMAND_SPEED_FRACTION_MAX)
        self.base_distance_limit = min(.01, float(args.max_calibration_jump_m))
        self.base_angle_limit = np.deg2rad(min(1., float(args.max_calibration_jump_deg)))
        if not np.isfinite([self.base_distance_limit, self.base_angle_limit]).all() or min(self.base_distance_limit, self.base_angle_limit) <= 0:
            raise AlignmentFault("alignment base drift limits must be finite and positive")
        self.models = {name: CapturedKinematics(name, self.initial_joints[name], self.initial_tcp[name], self.offsets[name])
                       for name in ROBOTS}
        self.guard = guard if guard is not None else WorkspaceGuard(self.models, args.workspace_config)
        self._delta = {name: self.goal_joints[name] - self.initial_joints[name] for name in ROBOTS}
        self._line = np.concatenate([self._delta[name] for name in self.commanded])
        self._line_squared = float(self._line @ self._line)
        for name in ROBOTS:
            self._validate_state(name, states[name])
            if not self._role_at_rest(name, states[name]):
                raise AlignmentFault(name + ": stop the robot before initial alignment")
        for name, target in self.targets.items():
            predicted, _ = self.models[name].pose_and_jacobian(self.goal_joints[name])
            if (np.linalg.norm(predicted[:3, 3] - target[:3, 3]) > self.limits["frame_check_m"]
                    or rotation_distance(predicted, target) > np.deg2rad(self.limits["frame_check_deg"])):
                raise AlignmentFault(name + ": native IK endpoint disagrees with the captured kinematic model")
            if not self.guard.pose_allowed(name, target):
                raise AlignmentPlanningFault(name + ": initial target outside TCP workspace")
        self.samples = []
        duration = 0.
        for phase in np.linspace(0., 1., 81):
            trial = {name: self.initial_joints[name] + phase * self._delta[name] for name in ROBOTS}
            self._check_configuration(trial, self.initial_bases)
            self.samples.append({name: q.copy() for name, q in trial.items()})
            for name in self.commanded:
                _, jacobian = self.models[name].pose_and_jacobian(trial[name])
                twist = jacobian @ self._delta[name]
                duration = max(duration, np.max(np.abs(self._delta[name])) / (self.command_speed_fraction * self.limits["max_joint_speed"]),
                               np.linalg.norm(twist[:3]) / (self.command_speed_fraction * self.limits["max_speed"]),
                               np.linalg.norm(twist[3:]) / (self.command_speed_fraction * self.limits["max_angular_speed"]))
        self.estimated_duration = duration + 2. * np.log(max(1., np.sqrt(self._line_squared) / .005)) + 2.
        if self.estimated_duration > 600.:
            raise AlignmentPlanningFault("initial alignment would exceed the 600 s execution bound")
        self.timeout = min(600., max(60., self.estimated_duration * 4.))
        self.previous_commands = {name: np.zeros(6) for name in self.commanded}
        self.previous_twists = {name: np.zeros(6) for name in self.commanded}
        self.joint_accelerations = {name: min(1e-6, self.limits["max_joint_accel"])
                                    for name in self.commanded}
        self.predicted_joints = {name: q.copy() for name, q in self.initial_joints.items()}
        self.elapsed, self.progress = 0., 0.
        self.done, self._pending = False, None
        self._settled = 0
        self._endpoint_mismatch_time = 0.
        self.message = "Initial alignment ready; measured joint feedback"

    @staticmethod
    def _at_rest(state, limits=(.005, .002, .01)):
        joint, linear, angular = limits
        return (np.max(np.abs(state.joint_velocity)) <= joint
                and np.linalg.norm(state.speed[:3]) <= linear and np.linalg.norm(state.speed[3:]) <= angular)

    def _role_at_rest(self, name, state):
        return self._at_rest(state, self.leader_rest_limits) if name in self.fixed else self._at_rest(state)

    def _validate_state(self, name, state):
        q = _vector(state.joints, name + " joints")
        qd = _vector(state.joint_velocity, name + " joint velocity")
        speed = _vector(state.speed, name + " TCP speed")
        tcp = _pose(state.tcp, name + " TCP")
        for metric, value, limit, unit in (
                ("joint_speed", float(np.max(np.abs(qd))), self.limits["max_joint_speed"], "rad/s"),
                ("linear_speed", float(np.linalg.norm(speed[:3])), self.limits["max_speed"], "m/s"),
                ("angular_speed", float(np.linalg.norm(speed[3:])), self.limits["max_angular_speed"], "rad/s")):
            if value > limit + 1e-9:
                raise AlignmentFault(
                    f"{name}: measured speed exceeded alignment limits: "
                    f"{metric}={value:.9g} > {limit:.9g} {unit}",
                    {"arm": name, "metric": metric, "value": value, "limit": limit, "unit": unit})
        nominal, _ = self.models[name].pose_and_jacobian(q)
        if (np.linalg.norm(nominal[:3, 3] - tcp[:3, 3]) > self.limits["frame_check_m"]
                or rotation_distance(nominal, tcp) > np.deg2rad(self.limits["frame_check_deg"])):
            raise AlignmentFault(name + ": measured TCP disagrees with alignment kinematics")

    def _check_bases_and_fixed(self, states, bases):
        if set(states) != set(ROBOTS) or set(bases) != set(ROBOTS):
            raise AlignmentFault("initial alignment lost a complete robot/base pair")
        result = {}
        for name in ROBOTS:
            result[name] = _pose(bases[name], name + " base")
            before = self.initial_bases[name]
            if (np.linalg.norm(result[name][:3, 3] - before[:3, 3]) > self.base_distance_limit
                    or rotation_distance(result[name], before) > self.base_angle_limit):
                raise AlignmentFault(name + ": base moved during initial alignment; stop and replan")
            self._validate_state(name, states[name])
        for name in self.fixed:
            current = states[name]
            if not self._role_at_rest(name, current):
                joint, linear, angular = self.leader_rest_limits
                measured = (float(np.max(np.abs(current.joint_velocity))),
                            float(np.linalg.norm(current.speed[:3])), float(np.linalg.norm(current.speed[3:])))
                raise AlignmentFault(
                    f"{name}: fixed leader moved during initial alignment: "
                    f"TCP {measured[1] * 1000:.3f}/{linear * 1000:g} mm/s, "
                    f"angular {measured[2]:.5f}/{angular:g} rad/s, "
                    f"joint {measured[0]:.5f}/{joint:g} rad/s (actual/limit)",
                    {"arm": name, "metric": "leader_rest_speed", "joint_speed": measured[0],
                     "linear_speed": measured[1], "angular_speed": measured[2],
                     "joint_limit": joint, "linear_limit": linear, "angular_limit": angular})
            if (np.max(np.abs(current.joints - self.initial_joints[name])) > .01
                    or np.linalg.norm(current.tcp[:3, 3] - self.initial_tcp[name][:3, 3]) > .003
                    or rotation_distance(current.tcp, self.initial_tcp[name]) > np.deg2rad(.5)):
                raise AlignmentFault(name + ": fixed leader moved during initial alignment")
        return result

    def _check_configuration(self, joints, bases, observed=None):
        fault = AlignmentFault if observed is not None else AlignmentPlanningFault
        flanges = {}
        for name in ROBOTS:
            q = _vector(joints[name], name + " path joints")
            if np.any(q < self.guard.lower[name]) or np.any(q > self.guard.upper[name]):
                raise fault(name + ": initial alignment joint position limit")
            tcp = self.models[name].pose_and_jacobian(q)[0] if observed is None else observed[name].tcp
            flange = tcp @ inverse(self.offsets[name])
            for pose, origin in ((tcp, self.initial_tcp[name]), (flange, self.initial_flange[name])):
                if np.linalg.norm(pose[:3, 3] - origin[:3, 3]) > self.limits["max_travel"] + 1e-9:
                    raise fault(name + ": initial alignment start-position travel envelope")
            if not self.guard.pose_allowed(name, tcp):
                raise fault(name + ": initial alignment TCP workspace")
            flanges[name] = bases[name] @ flange
        if np.linalg.norm(flanges["ur5e"][:3, 3] - flanges["ur10e"][:3, 3]) < self.limits["min_gap"]:
            raise fault("initial alignment flange gap is below min_gap")
        allowed, reason, _ = self.guard.check(bases, joints)
        if not allowed:
            raise fault("initial alignment geometry: " + str(reason))

    def _corridor_progress(self, joints):
        displacement = np.concatenate([joints[name] - self.initial_joints[name] for name in self.commanded])
        phase = float(np.clip(displacement @ self._line / self._line_squared, 0., 1.)) if self._line_squared > 1e-18 else 1.
        if np.max(np.abs(displacement - phase * self._line)) > .03:
            raise AlignmentFault("measured joints left the initial alignment corridor")
        return phase

    def _at_goal(self, states):
        return all(np.max(np.abs(states[name].joints - self.goal_joints[name])) <= .01
                   and np.linalg.norm(states[name].tcp[:3, 3] - self.targets[name][:3, 3]) <= .002
                   and rotation_distance(states[name].tcp, self.targets[name]) <= np.deg2rad(1.)
                   for name in self.commanded)

    def _command_interval(self, origins, desired, states, jacobians, dt):
        """Intersect every original bound along one common command segment."""
        interval = (0., 1.)
        for name in self.commanded:
            origin = origins[name]
            previous = self.previous_commands[name]
            direction = desired[name] - origin
            constraints = []
            for index in range(6):
                part = slice(index, index + 1)
                for label, center, radius in (
                        ("speed", np.zeros(1), self.command_speed_fraction * self.limits["max_joint_speed"]),
                        ("acceleration_from_previous", previous[part], .5 * self.limits["max_joint_accel"] * dt),
                        ("acceleration_from_measured", states[name].joint_velocity[part], self.limits["max_joint_accel"] * dt)):
                    constraints.append((f"joint_{index + 1}_{label}", origin[part], direction[part], center, radius))
            origin_twist = jacobians[name] @ origin
            direction_twist = jacobians[name] @ direction
            for part, label, speed, accel in (
                    (slice(0, 3), "linear", "max_speed", "max_accel"),
                    (slice(3, 6), "angular", "max_angular_speed", "max_angular_accel")):
                for suffix, center, radius in (
                        ("speed", np.zeros(3), self.command_speed_fraction * self.limits[speed]),
                        ("acceleration_from_previous", self.previous_twists[name][part], .5 * self.limits[accel] * dt),
                        ("acceleration_from_measured", states[name].speed[part], self.limits[accel] * dt)):
                    constraints.append((f"{label}_{suffix}", origin_twist[part], direction_twist[part], center, radius))
            for metric, origin_value, direction_value, center, radius in constraints:
                narrowed = _ball_interval(origin_value, direction_value, center, radius, interval)
                if narrowed is None:
                    squared = float(direction_value @ direction_value)
                    closest = (float(np.clip(-((origin_value - center) @ direction_value) / squared,
                                             interval[0], interval[1])) if squared > 1e-24 else interval[0])
                    divisor = dt if "acceleration" in metric else 1.
                    value = float(np.linalg.norm(origin_value + closest * direction_value - center) / divisor)
                    limit = float(radius / divisor)
                    raise AlignmentFault(
                        f"{name}: measured motion cannot follow the bounded alignment command: "
                        f"{metric} minimum={value:.9g}, limit={limit:.9g}, dt={dt:.6f}s",
                        {"arm": name, "metric": metric, "value": value, "limit": limit, "dt": float(dt)})
                interval = narrowed
        return interval

    def _predicted_motion(self, current, commands, states, dt):
        """Check the selected command using its actual predicted Jacobian.

        This same test governs candidate selection and final acceptance; its
        speed/acceleration limits are not relaxed for a linearized boundary.
        """
        predicted = {name: current[name] + commands.get(name, np.zeros(6)) * dt for name in ROBOTS}
        for name in self.commanded:
            future_twist = self.models[name].pose_and_jacobian(predicted[name])[1] @ commands[name]
            for part, label, speed, accel in (
                    (slice(0, 3), "linear", "max_speed", "max_accel"),
                    (slice(3, 6), "angular", "max_angular_speed", "max_angular_accel")):
                velocity = float(np.linalg.norm(future_twist[part]))
                change = float(np.linalg.norm(future_twist[part] - states[name].speed[part]))
                command_limit = self.command_speed_fraction * self.limits[speed]
                if velocity > command_limit + 1e-9:
                    metric, value, limit = label + "_speed", velocity, command_limit
                elif change > self.limits[accel] * dt + 1e-9:
                    metric, value, limit = label + "_acceleration", change / dt, self.limits[accel]
                else:
                    continue
                raise AlignmentFault(
                    f"{name}: predicted alignment TCP {metric} exceeds limit "
                    f"({value:.9g} > {limit:.9g}; dt={dt:.6f}s)",
                    {"arm": name, "metric": metric, "value": value, "limit": limit, "dt": float(dt)})
        return predicted

    def _bounded_commands(self, current, desired, states, jacobians, dt):
        """Try command history first, then a measured-velocity braking segment.

        Current-Jacobian bounds define a feasible scalar interval. Its upper
        endpoint need not satisfy the future Jacobian at q + qdot*dt, so test
        a bounded set of interior candidates too. Every candidate retains all
        previous-command and measured-state constraints. There are at most ten
        predictions per segment, and one on the normal successful path.
        """
        failures = []
        measured = {name: states[name].joint_velocity for name in self.commanded}
        for source, origins in (("previous_command", self.previous_commands), ("measured_velocity", measured)):
            try:
                low, high = self._command_interval(origins, desired, states, jacobians, dt)
            except AlignmentFault as exc:
                failures.append(exc)
                continue
            directions = {name: desired[name] - origins[name] for name in self.commanded}
            previous_alpha = None
            for fraction in (0., .0001, .001, .01, .05, .1, .25, .5, .75, 1.):
                alpha = float(np.clip(high - fraction * (high - low), 0., 1.))
                if alpha == previous_alpha:
                    continue
                previous_alpha = alpha
                commands = {name: origins[name] + alpha * directions[name] for name in self.commanded}
                try:
                    predicted = self._predicted_motion(current, commands, states, dt)
                except AlignmentFault as exc:
                    exc.details.update({"command_origin": source, "alpha": alpha})
                    failures.append(exc)
                    continue
                return commands, predicted
        # No unchecked fallback is sent when both bounded searches fail.
        failure = failures[-1]
        failure.details["search_failures"] = [str(exc) for exc in failures]
        raise AlignmentPlanningFault(str(failure), failure.details) from failure

    def _ramp_accelerations(self, commands, states, jacobians, dt):
        """Bound speedJ's scalar ramp from the actual measured joint speed."""
        accelerations = {}
        for name in self.commanded:
            difference = commands[name] - states[name].joint_velocity
            peak = float(np.max(np.abs(difference)))
            if peak == 0.:
                acceleration = min(1e-6, self.limits["max_joint_accel"])
            else:
                direction = jacobians[name] @ (difference / peak)
                acceleration = min(peak / dt, self.limits["max_joint_accel"])
                for part, limit in ((slice(0, 3), "max_accel"), (slice(3, 6), "max_angular_accel")):
                    magnitude = float(np.linalg.norm(direction[part]))
                    if magnitude > 0.:
                        acceleration = min(acceleration, self.limits[limit] / magnitude)
            if not np.isfinite(acceleration) or acceleration <= 0.:
                raise AlignmentFault(name + ": invalid measured-to-command joint ramp acceleration")
            accelerations[name] = float(acceleration)
        return accelerations

    def propose(self, states, bases, dt):
        """Compute bounded qdot requests without sending or advancing completion."""
        self._pending = None
        if self.done:
            raise AlignmentFault("initial alignment is already complete")
        if not np.isfinite(dt) or dt <= 0 or dt > self.limits["max_robot_age"]:
            raise AlignmentFault("initial alignment control period is invalid or stale")
        if self.elapsed + dt > self.timeout:
            raise AlignmentPlanningFault("initial alignment timed out before measured endpoint settling")
        bases = self._check_bases_and_fixed(states, bases)
        current = {name: _vector(states[name].joints, name + " joints") for name in ROBOTS}
        phase = self._corridor_progress(current)
        self._check_configuration(current, bases, observed=states)
        at_goal = self._at_goal(states)
        mismatch = (not at_goal and all(np.max(np.abs(current[name] - self.goal_joints[name])) <= .001
                                      and self._at_rest(states[name]) for name in self.commanded))
        if mismatch and self._endpoint_mismatch_time + dt > 1.:
            raise AlignmentFault("native IK joints reached but measured TCP does not match the target")
        desired, jacobians = {}, {}
        scale = 1.
        for name in self.commanded:
            jacobians[name] = self.models[name].pose_and_jacobian(current[name])[1]
            desired[name] = np.zeros(6) if at_goal else .5 * (self.goal_joints[name] - current[name])
            twist = jacobians[name] @ desired[name]
            # Stay just inside the command budget so Jacobian changes over the
            # next step do not routinely land on its numerical boundary. The
            # separate 10% reserve below hard limits is for the real servo.
            fraction = .98 * self.command_speed_fraction
            scale = min(scale, fraction * self.limits["max_joint_speed"] / max(np.max(np.abs(desired[name])), 1e-15),
                        fraction * self.limits["max_speed"] / max(np.linalg.norm(twist[:3]), 1e-15),
                        fraction * self.limits["max_angular_speed"] / max(np.linalg.norm(twist[3:]), 1e-15))
        desired = {name: value * scale for name, value in desired.items()}
        commands, predicted = self._bounded_commands(current, desired, states, jacobians, dt)
        # speedJ's scalar acceleration must agree with this measured-to-command
        # ramp. A fixed larger value could reach the same discrete command too
        # quickly. These local-Jacobian bounds do not certify continuous
        # Cartesian acceleration (Jdot and the actual servo remain unmodeled).
        accelerations = self._ramp_accelerations(commands, states, jacobians, dt)
        self._corridor_progress(predicted)
        self._check_configuration({name: (current[name] + predicted[name]) * .5 for name in ROBOTS}, bases)
        self._check_configuration(predicted, bases)
        self.predicted_joints = {name: q.copy() for name, q in predicted.items()}
        self.joint_accelerations = accelerations
        settled = at_goal and all(self._at_rest(states[name]) and np.max(np.abs(commands[name])) < 1e-10
                                 for name in self.commanded)
        self._pending = {"dt": float(dt), "progress": phase, "commands": commands,
                         "twists": {name: jacobians[name] @ commands[name] for name in self.commanded},
                         "settled": settled, "mismatch": mismatch}
        return {name: value.copy() for name, value in commands.items()}

    def propose_hold(self, states, bases, dt):
        """Propose only zero speedJ while a completed alignment stays settled.

        No commit is needed or allowed: this operation does not advance the
        alignment clock, progress, or command history. The caller owns the
        handoff deadline, fresh encoder timestamps and stop/process checks,
        and must transmit the returned zeros on each normal control tick to
        feed the controller watchdog. A drift or infeasible zero transition
        raises instead of issuing a corrective joint command.
        """
        self._pending = None
        if not self.done:
            raise AlignmentFault("initial alignment must be complete before stationary handoff")
        if not np.isfinite(dt) or dt <= 0 or dt > self.limits["max_robot_age"]:
            raise AlignmentFault("stationary handoff control period is invalid or stale")
        bases = self._check_bases_and_fixed(states, bases)
        current = {name: _vector(states[name].joints, name + " joints") for name in ROBOTS}
        self._corridor_progress(current)
        self._check_configuration(current, bases, observed=states)
        if not self._at_goal(states):
            raise AlignmentFault("robot left the measured alignment endpoint during stationary handoff")
        if not all(self._role_at_rest(name, states[name]) for name in ROBOTS):
            raise AlignmentFault("both robots must remain stationary during calibration handoff")
        commands = {name: np.zeros(6) for name in self.commanded}
        jacobians = {name: self.models[name].pose_and_jacobian(current[name])[1]
                     for name in self.commanded}
        # Unlike the alignment governor, there is no nonzero braking fallback.
        # Require exact zero to satisfy every measured/history transition bound.
        self._command_interval(commands, commands, states, jacobians, dt)
        predicted = self._predicted_motion(current, commands, states, dt)
        self._check_configuration(predicted, bases)
        accelerations = self._ramp_accelerations(commands, states, jacobians, dt)
        self.predicted_joints = {name: value.copy() for name, value in predicted.items()}
        self.joint_accelerations = accelerations
        return commands

    def commit(self):
        """Record a transmitted proposal; failed sends must not call this."""
        if self._pending is None or self.done:
            raise AlignmentFault("alignment commit requires a successful proposal and send")
        pending, self._pending = self._pending, None
        self.elapsed += pending["dt"]
        self.progress = pending["progress"]
        self.previous_commands = {name: value.copy() for name, value in pending["commands"].items()}
        self.previous_twists = {name: value.copy() for name, value in pending["twists"].items()}
        self._settled = self._settled + 1 if pending["settled"] else 0
        self._endpoint_mismatch_time = self._endpoint_mismatch_time + pending["dt"] if pending["mismatch"] else 0.
        self.done = self._settled >= 5
        if self.done:
            self.progress = 1.
        self.message = ("Initial alignment settled at measured TCP target" if self.done else
                        f"Initial alignment {self.progress:.1%}; settling {self._settled}/5")
