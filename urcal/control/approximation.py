"""Bounded preflight goal selection for uncertain camera calibration.

This module changes a requested flange goal, never a calibration measurement.
The captured nominal model only proposes a new precise TCP goal and IK seed;
native controller IK/FK, limits and the complete path still need validation.
It owns no robot transport and must not run in the velocity streaming loop.
"""

from itertools import product

import numpy as np
from scipy.optimize import least_squares

from urcal.control.alignment import CapturedKinematics
from urcal.control.geometry import inverse, pose_delta
from urcal.control.policy import ControlFault
from urcal.cooperative.kinematics import WorkspaceGuard


ROBOTS = ("ur5e", "ur10e")
POSITION_RESERVE_M = .0005
ROTATION_RESERVE_RAD = .002
MAX_EVALUATIONS = 80


class TargetApproximationUnavailable(ControlFault):
    """A bounded idle search found no candidate; a new Arm may retry it."""

    category = "arm_approximation"


def flange_adjustment(nominal_tcp, selected_tcp, offset):
    """Return translation norm and geodesic angle at the flange, in m/rad."""
    flange_offset = inverse(offset)
    dp, dr = pose_delta(nominal_tcp @ flange_offset, selected_tcp @ flange_offset)
    return float(np.linalg.norm(dp)), float(np.linalg.norm(dr))


def _vector(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ControlFault(label + " must contain six finite numbers")
    return value.copy()


def _transform(value):
    inverse(value)
    return np.asarray(value, dtype=float).copy()


def _bounds(args):
    try:
        distance = float(getattr(args, "target_tolerance_m", 0.))
        angle = np.deg2rad(float(getattr(args, "target_tolerance_deg", 0.)))
        singular = float(args.min_singular_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault("invalid approximate target bounds") from exc
    if (not np.isfinite([distance, angle, singular]).all() or min(distance, angle) < 0
            or singular <= 0):
        raise ControlFault("approximate target bounds must be finite and nonnegative")
    if distance == 0 and angle == 0:
        raise ControlFault("approximate target selection is disabled by zero tolerances")
    # Reserve room for the native IK/FK residual. A disabled component remains
    # numerically exact; it is not assigned a positive adjustment allowance.
    if 0 < distance <= POSITION_RESERVE_M or 0 < angle <= ROTATION_RESERVE_RAD:
        raise ControlFault("approximate target tolerance is smaller than the native validation reserve")
    position_bound = distance - POSITION_RESERVE_M if distance else 1e-9
    angle_bound = angle - ROTATION_RESERVE_RAD if angle else 1e-9
    return distance, angle, singular, position_bound, angle_bound


def select_alignment_candidate(args, offsets, bases, states, nominal_targets, *, check_fresh=None,
                               conditioning_scale=2.):
    """Select a reachable precise goal within each nominal flange tolerance.

    Five deterministic joint seeds per commanded arm and at most 80 optimizer
    evaluations per seed bound the work. The optional callback is invoked on
    every residual evaluation and before returning, so Stop/encoder checks
    remain active during this idle preflight calculation. It may raise.

    ``conditioning_scale`` increases only the optimizer's preference for an
    endpoint farther from singularity. It never changes the task allowance or
    the hard singular-value floor, and native validation remains mandatory.

    Returns ``(tcp_targets, joint_seeds, diagnostics)``. Inputs are never
    mutated. These are candidates, not authorization to issue motion.
    """
    distance, angle, minimum_singular, _, angle_bound = _bounds(args)
    try:
        conditioning_scale = float(conditioning_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault("conditioning scale must be finite and at least one") from exc
    if not np.isfinite(conditioning_scale) or conditioning_scale < 1.:
        raise ControlFault("conditioning scale must be finite and at least one")
    singular_preference = conditioning_scale * minimum_singular
    if not np.isfinite(singular_preference):
        raise ControlFault("conditioning preference must be finite")
    if (set(offsets) != set(ROBOTS) or set(bases) != set(ROBOTS) or set(states) != set(ROBOTS)
            or not nominal_targets or set(nominal_targets) - set(ROBOTS)):
        raise ControlFault("approximate alignment needs both robot states and valid commanded targets")
    commanded = tuple(name for name in ROBOTS if name in nominal_targets)
    offsets = {name: _transform(offsets[name]) for name in ROBOTS}
    bases = {name: _transform(bases[name]) for name in ROBOTS}
    targets = {name: _transform(nominal_targets[name]) for name in commanded}
    initial = {name: _vector(states[name].joints, name + " joints") for name in ROBOTS}
    measured_tcp = {name: _transform(states[name].tcp) for name in ROBOTS}
    measured_flange = {name: measured_tcp[name] @ inverse(offsets[name]) for name in ROBOTS}
    # Native validation bounds TCP translation and rotation independently. Its
    # rotation residual can displace the inferred flange with a nonzero TCP
    # offset, so reserve the worst-case lever-arm contribution as well.
    position_reserves = {
        name: POSITION_RESERVE_M + 2. * np.sin(ROTATION_RESERVE_RAD / 2.) * np.linalg.norm(offsets[name][:3, 3])
        for name in commanded}
    models = {name: CapturedKinematics(name, initial[name], measured_tcp[name], offsets[name])
              for name in ROBOTS}
    guard = WorkspaceGuard(models, args.workspace_config)
    evaluations = 0

    def fresh():
        if check_fresh is not None:
            check_fresh()

    def configuration(joints):
        flanges = {}
        for name in ROBOTS:
            tcp, _ = models[name].pose_and_jacobian(joints[name])
            flange = tcp @ inverse(offsets[name])
            if not guard.pose_allowed(name, tcp):
                return False, name + ": TCP workspace", None
            for pose, origin in ((tcp, measured_tcp[name]), (flange, measured_flange[name])):
                if np.linalg.norm(pose[:3, 3] - origin[:3, 3]) > args.max_travel + 1e-9:
                    return False, name + ": start-position travel envelope", None
            flanges[name] = bases[name] @ flange
        if np.linalg.norm(flanges[ROBOTS[0]][:3, 3] - flanges[ROBOTS[1]][:3, 3]) < args.min_gap:
            return False, "flange gap below min_gap", None
        return guard.check(bases, joints)

    fresh()
    allowed, reason, _ = configuration(initial)
    if not allowed:
        raise ControlFault("approximate alignment current configuration: " + reason)
    pools = {}
    for name in commanded:
        position_bound = distance - position_reserves[name] if distance else 1e-9
        if position_bound <= 0:
            raise TargetApproximationUnavailable(
                name + ": flange position tolerance leaves no room for native FK validation with this TCP offset")
        model = models[name]
        nominal_flange = targets[name] @ inverse(offsets[name])
        seed = initial[name]
        # Native IK can return a less well-conditioned endpoint than the local
        # model's seed. A retry may ask for more margin inside the same task
        # allowance; the hard acceptance threshold below stays unchanged.

        def residual(q):
            nonlocal evaluations
            fresh()
            evaluations += 1
            flange, _ = model.pose_and_jacobian(q, flange=True)
            _, jacobian = model.pose_and_jacobian(q)
            dp, dr = pose_delta(flange, nominal_flange)
            sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            return np.concatenate((dp / max(position_bound, 1e-5), dr / max(angle_bound, 1e-5),
                                   .02 * (q - seed), [4. * max(0., 1. - sigma / singular_preference)]))

        perturbations = (np.zeros(6), np.array([0, 0, 0, .4, .3, 0]),
                         np.array([0, 0, 0, -.4, -.3, 0]),
                         np.array([.4, -.3, 0, 0, 0, 0]),
                         np.array([-.4, .3, 0, 0, 0, 0]))
        candidates = []
        for perturbation in perturbations:
            start = np.clip(seed + perturbation, guard.lower[name] + 1e-10, guard.upper[name] - 1e-10)
            result = least_squares(residual, start, bounds=(guard.lower[name], guard.upper[name]),
                                   max_nfev=MAX_EVALUATIONS, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            q = _vector(result.x, name + " candidate joints")
            tcp, jacobian = model.pose_and_jacobian(q)
            dp, dr = flange_adjustment(targets[name], tcp, offsets[name])
            singular = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            if (dp > position_bound or dr > angle_bound or singular < minimum_singular
                    or not guard.pose_allowed(name, tcp)):
                continue
            score = (dp / max(position_bound, 1e-5)) ** 2 + (dr / max(angle_bound, 1e-5)) ** 2
            score += .0004 * float(np.sum((q - seed) ** 2))
            candidates.append({"tcp": tcp.copy(), "q": q, "score": score,
                               "position_m": dp, "rotation_deg": float(np.rad2deg(dr)),
                               "min_singular_value": singular})
        if not candidates:
            raise TargetApproximationUnavailable(
                name + ": no approximate alignment candidate within flange tolerances "
                f"{distance * 1000:g}mm / {np.rad2deg(angle):g}deg and existing limits")
        pools[name] = sorted(candidates, key=lambda item: item["score"])

    best, best_score, best_clearance = None, float("inf"), None
    last_reason = "no jointly feasible candidate"
    for combination in product(*(pools[name] for name in commanded)):
        fresh()
        joints = {name: value.copy() for name, value in initial.items()}
        joints.update({name: item["q"] for name, item in zip(commanded, combination)})
        allowed, last_reason, clearance = configuration(joints)
        score = sum(item["score"] for item in combination)
        if allowed and score < best_score:
            best, best_score, best_clearance = combination, score, clearance
    if best is None:
        raise TargetApproximationUnavailable(
            "approximate alignment candidates violate existing geometry/travel limits: " + last_reason)
    selected = {name: item["tcp"].copy() for name, item in zip(commanded, best)}
    seeds = {name: item["q"].copy() for name, item in zip(commanded, best)}
    diagnostics = {"position_tolerance_m": distance, "rotation_tolerance_deg": float(np.rad2deg(angle)),
                   "conditioning_scale": conditioning_scale,
                   "singular_preference": singular_preference,
                   "position_reserve_m": POSITION_RESERVE_M, "rotation_reserve_rad": ROTATION_RESERVE_RAD,
                   "flange_position_reserves_m": {name: float(value) for name, value in position_reserves.items()},
                   "residual_evaluations": evaluations, "goal_clearance_m": best_clearance,
                   "adjustments": {name: {key: value for key, value in item.items()
                                          if key not in ("tcp", "q", "score")}
                                   for name, item in zip(commanded, best)}}
    fresh()
    return selected, seeds, diagnostics
