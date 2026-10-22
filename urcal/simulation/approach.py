"""Bounded initial joint approach for the disconnected simulation only.

The requested final TCP poses are preserved. A local IK search finds joint
endpoints, then both arms follow the same quintic time law along straight joint
paths. This is not a general obstacle-avoiding planner: an obstructed path is
rejected. The plant still checks each proposed step before the caller commits.
"""

from itertools import product

import numpy as np

from urcal.control.geometry import inverse, rotation_distance
from urcal.simulation.plant import SimulationFault


ROBOTS = ("ur5e", "ur10e")


class InitialApproachIKUnavailable(SimulationFault):
    """No virtual-model IK candidate; distinct from an unsafe current/path pose."""


def _vector(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise SimulationFault(label + " must contain six finite numbers")
    return value.copy()


def _pose(value):
    inverse(value)
    return np.asarray(value, dtype=float).copy()


def _smooth(u):
    return u * u * u * (10. + u * (-15. + 6. * u))


class InitialApproach:
    """Plan once; propose using measured virtual joints; advance only on commit.

    ``targets`` contains final controller-base TCP poses for commanded arms.
    Bases and the other arm remain fixed throughout the approach. ``duration``
    includes one final zero-velocity settling step. The original capture and
    target arrays are never modified. Limits are taken from the existing args;
    no real controller setting or runtime is changed.
    """

    def __init__(self, models, guard, args, bases, states, targets):
        if (set(models) != set(ROBOTS) or set(bases) != set(ROBOTS)
                or set(states) != set(ROBOTS) or not targets or set(targets) - set(ROBOTS)):
            raise SimulationFault("initial approach requires a robot pair and known TCP targets")
        self.models, self.guard = models, guard
        self.bases = {name: _pose(bases[name]) for name in ROBOTS}
        self.targets = {name: _pose(target) for name, target in targets.items()}
        self.initial_joints = {name: _vector(states[name].joints, name + " joints") for name in ROBOTS}
        self.initial_tcp = {name: _pose(states[name].tcp) for name in ROBOTS}
        self.offsets = {name: _pose(models[name].tcp_offset) for name in ROBOTS}
        self.initial_flange = {name: self.initial_tcp[name] @ inverse(self.offsets[name]) for name in ROBOTS}
        self.lower = {name: _vector(guard.lower[name], name + " lower limits") for name in ROBOTS}
        self.upper = {name: _vector(guard.upper[name], name + " upper limits") for name in ROBOTS}
        self.limits = {key: float(getattr(args, key)) for key in
                       ("control_hz", "max_joint_speed", "max_joint_accel", "max_speed", "max_angular_speed",
                        "max_accel", "max_angular_accel", "min_gap", "max_travel")}
        if not all(np.isfinite(value) and value > 0 for value in self.limits.values()):
            raise SimulationFault("initial approach limits must be finite and positive")
        self.dt = 1. / self.limits["control_hz"]
        for name in ROBOTS:
            if np.any(self.lower[name] >= self.upper[name]):
                raise SimulationFault(name + ": invalid approach joint limits")
            if (np.linalg.norm(_vector(states[name].joint_velocity, name + " joint velocity")) > 1e-9
                    or np.linalg.norm(_vector(states[name].speed, name + " TCP velocity")) > 1e-9):
                raise SimulationFault("initial approach must start from rest")
        self._check_configuration(self.initial_joints)
        self.goal_joints = self._find_endpoints()
        self._delta = {name: self.goal_joints[name] - self.initial_joints[name] for name in ROBOTS}
        self._motion_steps = self._choose_timing()
        self.duration = (self._motion_steps + 1) * self.dt
        self._index = 0
        self._pending = False

    @property
    def progress(self):
        return min(1., self._index / (self._motion_steps + 1))

    @property
    def done(self):
        return self._index >= self._motion_steps + 1

    def _check_configuration(self, joints):
        flanges = {}
        for name in ROBOTS:
            q = _vector(joints[name], name + " approach joints")
            if np.any(q < self.lower[name] - 1e-10) or np.any(q > self.upper[name] + 1e-10):
                raise SimulationFault(name + ": initial approach joint position limit")
            tcp, _ = self.models[name].pose_and_jacobian(q)
            flange = tcp @ inverse(self.offsets[name])
            for pose, start, label in ((tcp, self.initial_tcp[name], "TCP"),
                                        (flange, self.initial_flange[name], "flange")):
                if np.linalg.norm(pose[:3, 3] - start[:3, 3]) > self.limits["max_travel"] + 1e-9:
                    raise SimulationFault(name + ": initial approach " + label + " travel envelope")
            if not self.guard.pose_allowed(name, tcp):
                raise SimulationFault(name + ": initial approach TCP workspace")
            flanges[name] = self.bases[name] @ flange
        gap = np.linalg.norm(flanges["ur5e"][:3, 3] - flanges["ur10e"][:3, 3])
        if gap < self.limits["min_gap"] - 1e-9:
            raise SimulationFault("initial approach flange separation below min_gap")
        allowed, reason, _ = self.guard.check(self.bases, joints)
        if not allowed:
            raise SimulationFault("initial approach path rejected: " + str(reason))

    def _canonical(self, name, q):
        q = _vector(q, name + " IK solution")
        for index in range(6):
            lower = int(np.ceil((self.lower[name][index] - q[index]) / (2 * np.pi)))
            upper = int(np.floor((self.upper[name][index] - q[index]) / (2 * np.pi)))
            if lower > upper:
                return None
            choices = q[index] + 2 * np.pi * np.arange(lower, upper + 1)
            q[index] = choices[np.argmin(np.abs(choices - self.initial_joints[name][index]))]
        return q

    def _solution(self, name, seed):
        q = self.models[name].inverse(self.targets[name], seed, self.lower[name], self.upper[name])
        if q is None:
            return None
        q = self._canonical(name, q)
        if q is None:
            return None
        pose, _ = self.models[name].pose_and_jacobian(q)
        if (np.linalg.norm(pose[:3, 3] - self.targets[name][:3, 3]) > .0005 + 1e-9
                or rotation_distance(pose, self.targets[name]) > .002 + 1e-9):
            return None
        return q

    def _path_allowed(self, goals):
        for phase in np.linspace(0., 1., 81):
            trial = {name: self.initial_joints[name] + phase * (goals[name] - self.initial_joints[name])
                     for name in ROBOTS}
            self._check_configuration(trial)

    def _find_endpoints(self):
        candidates = {name: [] for name in ROBOTS}
        for name in ROBOTS:
            q = self._solution(name, self.initial_joints[name]) if name in self.targets else self.initial_joints[name].copy()
            if q is not None:
                candidates[name].append(q)
        last_reason = "no local IK solution for the requested TCP target"
        if all(candidates.values()):
            primary = {name: candidates[name][0] for name in ROBOTS}
            try:
                self._path_allowed(primary)
                return {name: q.copy() for name, q in primary.items()}
            except SimulationFault as exc:
                last_reason = str(exc)
        # A different wrist branch can avoid an obstructed straight joint path.
        # Seeds are repeatable and never replace the captured initial joints.
        generator = np.random.default_rng(7)
        for name in self.targets:
            start = self.initial_joints[name]
            seeds = []
            for index in (3, 4, 5):
                for change in (-2 * np.pi, -np.pi, np.pi, 2 * np.pi):
                    seed = start.copy()
                    seed[index] += change
                    if np.all(seed >= self.lower[name]) and np.all(seed <= self.upper[name]):
                        seeds.append(seed)
            seeds.extend(np.clip(start + generator.normal(0., 1., 6), self.lower[name], self.upper[name])
                         for _ in range(12))
            for seed in seeds:
                solution = self._solution(name, seed)
                if solution is not None and not any(np.linalg.norm(solution - old) < .01 for old in candidates[name]):
                    candidates[name].append(solution)
            if not candidates[name]:
                raise InitialApproachIKUnavailable(name + ": initial approach IK failed for requested target")
            candidates[name].sort(key=lambda q: np.linalg.norm(q - start))
            candidates[name] = candidates[name][:8]
        combinations = sorted(product(*(candidates[name] for name in ROBOTS)),
                              key=lambda pair: sum(np.linalg.norm(q - self.initial_joints[name])
                                                   for name, q in zip(ROBOTS, pair)))
        for pair in combinations:
            goals = dict(zip(ROBOTS, pair))
            try:
                self._path_allowed(goals)
                return {name: q.copy() for name, q in goals.items()}
            except SimulationFault as exc:
                last_reason = str(exc)
        raise SimulationFault("no feasible initial approach joint path: " + last_reason)

    def _positions(self, name, phase):
        return self.initial_joints[name] + np.asarray(_smooth(phase))[..., None] * self._delta[name]

    def _choose_timing(self):
        phase = np.linspace(0., 1., 81)
        derivative = 30. * phase**2 * (1 - phase)**2
        second = 60. * phase * (1 - phase) * (1 - 2 * phase)
        duration = self.dt
        for name in ROBOTS:
            qd = derivative[:, None] * self._delta[name]
            qdd = second[:, None] * self._delta[name]
            twists = np.array([self.models[name].pose_and_jacobian(q)[1] @ velocity
                               for q, velocity in zip(self._positions(name, phase), qd)])
            acceleration = np.gradient(twists, phase, axis=0, edge_order=2)
            duration = max(duration, np.max(np.abs(qd)) / self.limits["max_joint_speed"],
                           np.sqrt(np.max(np.abs(qdd)) / self.limits["max_joint_accel"]))
            for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                       (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                duration = max(duration, np.linalg.norm(twists[:, part], axis=1).max() / self.limits[speed],
                               np.sqrt(np.linalg.norm(acceleration[:, part], axis=1).max() / self.limits[accel]))
        steps = max(1, int(np.ceil(duration * 1.15 / self.dt)))
        for _ in range(6):
            if steps > 100000:
                raise SimulationFault("initial approach needs more than 100000 steps at these speed limits")
            speed_ratio, accel_ratio = self._trajectory_ratios(steps)
            if max(speed_ratio, accel_ratio) <= 1. + 1e-8:
                return steps
            steps = max(steps + 1, int(np.ceil(steps * max(speed_ratio, np.sqrt(accel_ratio)) * 1.05)))
        raise SimulationFault("could not time the initial approach within configured motion limits")

    def _trajectory_ratios(self, steps):
        speed_ratio = accel_ratio = 0.
        for name in ROBOTS:
            positions = self._positions(name, np.linspace(0., 1., steps + 1))
            jacobians = np.array([self.models[name].pose_and_jacobian(q)[1] for q in positions])
            qdot = np.vstack((np.diff(positions, axis=0) / self.dt, np.zeros(6)))
            previous = np.vstack((np.zeros(6), qdot[:-1]))
            speed_ratio = max(speed_ratio, np.max(np.abs(qdot)) / self.limits["max_joint_speed"])
            accel_ratio = max(accel_ratio, np.max(np.abs(qdot - previous)) / self.dt / self.limits["max_joint_accel"])
            before = np.einsum("nij,nj->ni", jacobians, previous)
            command = np.einsum("nij,nj->ni", jacobians, qdot)
            after = np.einsum("nij,nj->ni", np.concatenate((jacobians[1:], jacobians[-1:])), qdot)
            for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                       (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                for velocity in (command, after):
                    speed_ratio = max(speed_ratio, np.linalg.norm(velocity[:, part], axis=1).max() / self.limits[speed])
                    accel_ratio = max(accel_ratio, np.linalg.norm((velocity - before)[:, part], axis=1).max()
                                      / self.dt / self.limits[accel])
        return speed_ratio, accel_ratio

    def propose(self, states):
        """Return both qdot commands without changing progress or joint state."""
        if self.done:
            raise SimulationFault("initial approach is already complete")
        self._pending = False
        qdots, current, following = {}, {}, {}
        for name in ROBOTS:
            q = _vector(states[name].joints, name + " current joints")
            velocity = _vector(states[name].joint_velocity, name + " measured joint velocity")
            measured = _vector(states[name].speed, name + " measured TCP velocity")
            current[name] = q
            if self._index == self._motion_steps:
                if np.max(np.abs(q - self.goal_joints[name])) > 1e-7:
                    raise SimulationFault(name + ": initial approach endpoint was not reached")
                command = np.zeros(6)
            elif name not in self.targets:
                if np.max(np.abs(q - self.initial_joints[name])) > 1e-7:
                    raise SimulationFault(name + ": fixed arm moved during initial approach")
                command = np.zeros(6)
            else:
                desired = self._positions(name, (self._index + 1) / self._motion_steps)
                command = (desired - q) / self.dt
            next_q = q + command * self.dt
            if (np.max(np.abs(command)) > self.limits["max_joint_speed"] + 1e-9
                    or np.max(np.abs(command - velocity)) > self.limits["max_joint_accel"] * self.dt + 1e-9):
                raise SimulationFault(name + ": initial approach joint speed/acceleration limit")
            for candidate in (q, next_q):
                _, jacobian = self.models[name].pose_and_jacobian(candidate)
                twist = jacobian @ command
                for part, speed, accel in ((slice(0, 3), "max_speed", "max_accel"),
                                           (slice(3, 6), "max_angular_speed", "max_angular_accel")):
                    if (np.linalg.norm(twist[part]) > self.limits[speed] + 1e-9
                            or np.linalg.norm(twist[part] - measured[part]) > self.limits[accel] * self.dt + 1e-9):
                        raise SimulationFault(name + ": initial approach TCP speed/acceleration limit")
            qdots[name], following[name] = command, next_q
        self._check_configuration(current)
        self._check_configuration({name: (current[name] + following[name]) / 2. for name in ROBOTS})
        self._check_configuration(following)
        self._pending = True
        return {name: value.copy() for name, value in qdots.items()}

    def commit(self):
        """Advance one step only after the caller has accepted the proposal."""
        if not self._pending or self.done:
            raise SimulationFault("initial approach commit requires a successful proposal")
        self._index += 1
        self._pending = False
