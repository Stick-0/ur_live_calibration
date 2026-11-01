"""Pose-error feedback damped by measured robot motion, without visual velocity.

This module shapes a requested correction. It does not replace the caller's
Cartesian/joint speed, acceleration, geometry, or transport checks.
"""

import math

import numpy as np


def damped_feedback_velocity(error, relative_velocity, *, gain, speed_limit,
                             deceleration, reaction_time):
    """Return a three-axis velocity correction in the error's reference frame.

    ``error`` is target minus current position, or the corresponding rotation
    vector, after the caller applies its deadband. ``relative_velocity`` is the
    measured robot velocity minus target feedforward velocity in the same frame;
    positive velocity along the error is closing motion. It must come from robot
    encoders/RTDE, not finite differences of estimated calibration transforms.

    Predicting ``error - relative_velocity * reaction_time`` reduces correction
    before the robot passes the target, and damps sideways motion as well. A
    second cap reserves distance for the reaction interval and constant braking:
    ``v * reaction_time + v**2 / (2 * deceleration) <= norm(error)``. The caller
    must provide a conservative deceleration budget for its configured limits.

    Use metres, metres/second and metres/second squared for translation, or
    radians and the corresponding angular units for rotation. ``reaction_time``
    is seconds. Zero preserves legacy proportional feedback, including its lack
    of predictive braking. The returned demand can reverse before error crosses
    zero; the motion governor remains responsible for a feasible transition.
    """
    error = np.asarray(error, dtype=float)
    relative_velocity = np.asarray(relative_velocity, dtype=float)
    if (error.shape != (3,) or relative_velocity.shape != (3,)
            or not np.isfinite(error).all() or not np.isfinite(relative_velocity).all()):
        raise ValueError("feedback error and relative velocity must contain three finite values")
    values = {}
    for name, value in (("gain", gain), ("speed_limit", speed_limit),
                        ("deceleration", deceleration), ("reaction_time", reaction_time)):
        scalar = np.asarray(value, dtype=float)
        if (scalar.shape != () or not np.isfinite(scalar)
                or (float(scalar) < 0. if name == "reaction_time" else float(scalar) <= 0.)):
            raise ValueError(name + " must be finite and " + ("nonnegative" if name == "reaction_time" else "positive"))
        values[name] = float(scalar)
    gain, speed_limit, deceleration, reaction_time = (
        values[name] for name in ("gain", "speed_limit", "deceleration", "reaction_time"))
    distance = float(np.linalg.norm(error))
    if distance == 0.:
        return np.zeros(3)
    predicted = error.copy()
    if reaction_time > 0.:
        predicted -= relative_velocity * reaction_time
        reaction_speed = deceleration * reaction_time
        # Equivalent to sqrt((a*t)**2 + 2*a*d) - a*t, without cancellation
        # close to the target. hypot avoids squaring a potentially large a*t.
        braking_speed = (2. * deceleration * distance
                         / (math.hypot(reaction_speed, math.sqrt(2. * deceleration * distance))
                            + reaction_speed))
        speed_limit = min(speed_limit, braking_speed)
    requested = gain * predicted
    magnitude = float(np.linalg.norm(requested))
    if magnitude > speed_limit:
        requested *= speed_limit / magnitude
    return requested
