"""Bounded leader tracking with mandatory moving-base compensation.

The base term retains the strict research governor's independent checks. Leader
tracking may lag while a feasible velocity transition is generated; callers must
retain a pose reference and enforce a separate tracking-error envelope. This
module does not capture targets, advance task time, or communicate with hardware.
"""

import numpy as np

from urcal.cooperative.task import MotionDecision, MotionInfeasible, govern_task


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(name + " must contain six finite values")
    return result.copy()


def _norm_interval(a, b, bound):
    """Exact scalar interval for ||a + fraction*b|| <= bound."""
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))),
                abs(float(bound)), np.finfo(float).tiny)
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
    q = -.5 * (linear + np.copysign(root, linear))
    first, second = ((-linear / (2. * quadratic),) * 2 if q == 0.
                     else (q / quadratic, constant / q))
    return min(first, second), max(first, second)


def govern_follow(base, tracking, task, J, measured_twist, measured_qdot,
                  previous_twist, previous_qdot, dt, limits, *, previous_base,
                  previous_base_qdot, rho_cap=1., extra_feasible=None):
    """Shape tracking first, then allocate optional relative-path progress.

    ``base`` is never scaled and must be feasible independently. The total
    tracking baseline lies on a segment from the previous non-base command to
    a statically feasible tracking velocity. That endpoint is selected along
    the requested tracking direction, with the full base contribution retained.
    A measured-motion anchor is a
    fallback when a changed Jacobian makes that segment infeasible. Both anchors
    preserve the new minus previous base contribution; measured motion cannot
    silently cancel a new base request.

    Every accepted total obeys TCP/joint speed limits and acceleration from BOTH
    measured motion and the previous transmitted command. Stops and reversals
    therefore ramp through the previous velocity instead of scaling a fresh
    request towards zero. A bounded interior search lets ``extra_feasible``
    reject candidates for future geometry. No candidate bypasses that callback.

    ``task_scale`` describes optional task progress only. ``tracking_scale`` is
    the interpolation fraction along the selected transition segment, not a
    multiplier of raw leader speed. The previous non-base command can contain
    residual task velocity while transitioning; callers must keep their original
    relative path and check actual/next tracking error throughout that transition.
    """
    base = _vector(base, "base compensation")
    tracking = _vector(tracking, "tracking demand")
    task = _vector(task, "task twist")
    measured_twist = _vector(measured_twist, "measured twist")
    measured_qdot = _vector(measured_qdot, "measured joint velocity")
    previous_twist = _vector(previous_twist, "previous twist")
    previous_qdot = _vector(previous_qdot, "previous joint velocity")
    previous_base = _vector(previous_base, "previous base compensation")
    previous_base_qdot = _vector(previous_base_qdot, "previous base joint velocity")
    # Reuse the strict governor for parameter, singularity, and mandatory-base
    # validation. The optional channel must never hide impossible base motion.
    base_check = govern_task(
        base, np.zeros(6), np.zeros(6), J, previous_base, previous_base_qdot,
        dt, limits, rho_cap=rho_cap, previous_compensation=previous_base,
        previous_compensation_qdot=previous_base_qdot)
    J = np.asarray(J, dtype=float)

    # Make the transition endpoint statically feasible before imposing slew
    # limits. Otherwise an over-speed demand that turns slightly away from a
    # current command on a speed ball can point outward everywhere along its
    # segment, preventing the follower from changing direction at all.
    tracking_qdot = np.linalg.solve(J, tracking)
    static_constraints = [
        (base[:3], tracking[:3], limits.linear_speed),
        (base[3:], tracking[3:], limits.angular_speed),
        *((base_check.qdot[i], tracking_qdot[i], limits.joint_speed[i]) for i in range(6)),
    ]
    static_lower, static_upper = 0., 1.
    for a, b, bound in static_constraints:
        interval = _norm_interval(a, b, bound)
        if interval is None:
            # govern_task has already checked the complete base against the
            # same limits, including numerical tolerance. Retain that verified
            # zero tracking endpoint if a boundary quadratic loses its root.
            static_lower = static_upper = 0.
            break
        static_lower = max(static_lower, interval[0])
        static_upper = min(static_upper, interval[1])
    if static_lower > static_upper:
        static_upper = 0.
    tracking_endpoint = tracking * static_upper

    def command_transition_ok(twist, qdot):
        return (np.linalg.norm(twist[:3] - previous_twist[:3]) <= limits.linear_accel * dt + 1e-12
                and np.linalg.norm(twist[3:] - previous_twist[3:]) <= limits.angular_accel * dt + 1e-12
                and np.all(np.abs(qdot - previous_qdot) <= limits.joint_accel * dt + 1e-12))

    feasibility_cache = {}

    def allowed(rho, twist, qdot):
        if not command_transition_ok(twist, qdot):
            return False
        if extra_feasible is None:
            return True
        key = (float(rho), twist.tobytes(), qdot.tobytes())
        if key not in feasibility_cache:
            feasibility_cache[key] = bool(extra_feasible(rho, twist.copy(), qdot.copy()))
        return feasibility_cache[key]

    def transition_interval(origin):
        initial = base + origin
        direction = tracking_endpoint - origin
        initial_qdot = np.linalg.solve(J, initial)
        direction_qdot = np.linalg.solve(J, direction)
        constraints = [
            (initial[:3], direction[:3], limits.linear_speed),
            (initial[3:], direction[3:], limits.angular_speed),
            *((initial_qdot[i], direction_qdot[i], limits.joint_speed[i]) for i in range(6)),
        ]
        for twist, qdot in ((measured_twist, measured_qdot), (previous_twist, previous_qdot)):
            constraints.extend([
                (initial[:3] - twist[:3], direction[:3], limits.linear_accel * dt),
                (initial[3:] - twist[3:], direction[3:], limits.angular_accel * dt),
                *((initial_qdot[i] - qdot[i], direction_qdot[i], limits.joint_accel[i] * dt)
                  for i in range(6)),
            ])
        # A previous command exactly on a speed ball may have a quadratic root
        # a few ulps below zero. Preserve the independently verified zero step
        # rather than treating harmless rounding as an empty transition set.
        origin_feasible = all(np.linalg.norm(np.atleast_1d(a)) <= bound + 1e-12
                              for a, _b, bound in constraints)
        lower, upper = 0., 1.
        for a, b, bound in constraints:
            interval = _norm_interval(a, b, bound)
            if interval is None:
                return (0., 0.) if origin_feasible else None
            lower, upper = max(lower, interval[0]), min(upper, interval[1])
            if lower > upper:
                return (0., 0.) if origin_feasible else None
        return lower, upper

    failure = None
    anchors = (("previous_command", previous_twist - previous_base),
               ("measured_motion", measured_twist - previous_base))
    visited = set()
    for origin_name, origin in anchors:
        interval = transition_interval(origin)
        if interval is None:
            continue
        lower, upper = interval
        for fraction in np.linspace(upper, lower, 9):
            applied = origin + float(fraction) * (tracking_endpoint - origin)
            key = applied.tobytes()
            if key in visited:
                continue
            visited.add(key)
            try:
                decision = govern_task(
                    base, applied, task, J, measured_twist, measured_qdot, dt, limits,
                    rho_cap=rho_cap, extra_feasible=allowed, grid_samples=9,
                    previous_compensation=previous_base,
                    previous_compensation_qdot=previous_base_qdot)
            except MotionInfeasible as exc:
                if exc.category in ("compensation_infeasible", "singularity"):
                    raise
                failure = exc
                continue
            limited = not np.allclose(applied, tracking, rtol=0., atol=1e-10)
            components = dict(decision.components)
            components.update({
                "compensation": base.copy(),
                "compensation_qdot": base_check.qdot.copy(),
                "tracking_requested": tracking.copy(),
                "tracking_applied": applied.copy(),
                "tracking_scale": float(fraction),
                "tracking_speed_scale": float(static_upper),
                "tracking_endpoint": tracking_endpoint.copy(),
                "tracking_limited": limited,
                "tracking_origin": origin_name,
                "feasible_tracking_interval": np.asarray(interval),
            })
            return MotionDecision(decision.twist, decision.qdot, decision.task_scale,
                                  "tracking_limited" if limited else decision.reason, components)
    if failure is not None:
        raise failure
    raise MotionInfeasible("no bounded tracking transition meets measured and command acceleration limits",
                           "transition_infeasible")
