"""Choose the scalar acceleration used by the existing ``speedL`` transport.

The ur_rtde script calls ``speedl(target, a=acceleration, t=time)``. URScript
defaults rotational acceleration to the same numerical value as positional
acceleration when ``aRot`` is omitted. The two Cartesian groups can finish
their ramps at different times, so checking only the final joint-velocity
change is insufficient.

This is a local, constant-Jacobian prediction. It does not bound unmodelled
servo response, changing Jacobians, or measured joint acceleration on hardware.
Sources: https://gitlab.com/sdurobotics/ur_rtde/-/blob/master/scripts/rtde_control.script
and https://www.universal-robots.com/manuals/EN/PDF/SW5_21/scriptmanualG5/script_directory_Poly5.pdf
"""

import numpy as np

from urcal.control.policy import ControlFault


def _array(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault(label + " must contain finite values") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ControlFault(label + f" must have shape {shape} and finite values")
    return result.copy()


def _positive(value, label):
    try:
        array = np.asarray(value, dtype=float)
        if array.shape != ():
            raise ValueError("not a scalar")
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlFault(label + " must be a positive finite scalar") from exc
    if not np.isfinite(result) or result <= 0.:
        raise ControlFault(label + " must be a positive finite scalar")
    return result


def _norm(vector):
    """Avoid underflow/overflow while normalizing small velocity changes."""
    scale = float(np.max(np.abs(vector)))
    if scale == 0.:
        return 0.
    with np.errstate(over="ignore", invalid="ignore"):
        value = scale * float(np.linalg.norm(vector / scale))
    if not np.isfinite(value):
        raise ControlFault("Cartesian velocity change is too large")
    return value


def bounded_speedl_acceleration(J, measured_twist, target_twist, dt, limits):
    """Return a positive scalar ramp acceleration for a validated speedL step.

    ``limits`` supplies ``linear_accel``, ``angular_accel``, six ``joint_accel``
    values, and ``min_singular_value``. Besides both Cartesian limits, the scalar
    is bounded by the larger Cartesian group's velocity change divided by dt,
    using the same scalar convention as speedL. Thus the longer ramp lasts at
    least one control period; a tiny change in the other group cannot stall it.
    The shorter ramp may finish earlier, which is why all phases are checked.

    For each possible ramp phase (translation, rotation, or both), ``J^-1`` maps
    the unit Cartesian acceleration to joint acceleration. All three phases must
    respect the per-joint bounds. No velocity, pose, or motion history is changed;
    the caller must separately validate the command and commit it after sending.
    """
    jacobian = _array(J, (6, 6), "speedL Jacobian")
    measured = _array(measured_twist, (6,), "measured Cartesian velocity")
    target = _array(target_twist, (6,), "target Cartesian velocity")
    period = _positive(dt, "speedL control interval")
    try:
        linear_accel = _positive(limits.linear_accel, "linear acceleration limit")
        angular_accel = _positive(limits.angular_accel, "angular acceleration limit")
        joint_accel = _array(limits.joint_accel, (6,), "joint acceleration limits")
        min_singular = _positive(limits.min_singular_value, "minimum singular value")
    except AttributeError as exc:
        raise ControlFault("speedL acceleration limits are incomplete") from exc
    if np.any(joint_accel <= 0.):
        raise ControlFault("joint acceleration limits must be positive")
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ControlFault("cannot evaluate speedL Jacobian conditioning") from exc
    if not np.isfinite(singular_values).all() or singular_values[-1] < min_singular:
        raise ControlFault("speedL Jacobian is below the configured singularity limit")

    with np.errstate(over="ignore", invalid="ignore"):
        change = target - measured
    if not np.isfinite(change).all():
        raise ControlFault("Cartesian velocity change is not finite")
    linear_norm, angular_norm = _norm(change[:3]), _norm(change[3:])
    acceleration = min(linear_accel, angular_accel)
    linear, angular = np.zeros(6), np.zeros(6)
    for magnitude, direction, selection in (
            (linear_norm, linear, slice(0, 3)),
            (angular_norm, angular, slice(3, 6))):
        if magnitude > 0.:
            direction[selection] = change[selection] / magnitude
    if linear_norm > 0. or angular_norm > 0.:
        with np.errstate(over="ignore", under="ignore"):
            acceleration = min(acceleration, max(linear_norm, angular_norm) / period)
        try:
            phase_joints = np.linalg.solve(jacobian, np.column_stack((linear, angular, linear + angular)))
        except np.linalg.LinAlgError as exc:
            raise ControlFault("cannot solve speedL ramp joint acceleration") from exc
        if not np.isfinite(phase_joints).all():
            raise ControlFault("speedL ramp joint acceleration is not finite")
        response = np.max(np.abs(phase_joints), axis=1)
        active = response > 0.
        if np.any(active):
            with np.errstate(over="ignore", under="ignore"):
                acceleration = min(acceleration, float(np.min(joint_accel[active] / response[active])))
    if not np.isfinite(acceleration) or acceleration <= 0.:
        raise ControlFault("no positive representable speedL ramp acceleration")
    return float(acceleration)
