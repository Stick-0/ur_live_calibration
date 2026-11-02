"""Relative-base motion at the current follower flange, without hardware IO.

Only G's origin velocity and angular velocity enter this calculation. Leader
joint motion, target offsets and approach/path derivatives are separate terms.
The caller commits the returned flange velocity only after a successful send.
"""

import numpy as np

from urcal.control.geometry import inverse


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError(name + " must be a finite six-vector")
    return value.copy()


def _bounded(value, limit):
    length = np.linalg.norm(value)
    return value * min(1., limit / length) if length else value.copy()


def relative_base_compensation(G, flange, tcp, origin_velocity, confidence,
                               previous_flange, dt, *, linear_speed=.10,
                               angular_speed=.15, linear_accel=.20, angular_accel=.30):
    """Return independently bounded flange/TCP feedforward in follower axes.

    G is follower-base-from-leader-base at the SAME control time as flange/tcp.
    ``confidence`` weights the relative-state velocity, not a derivative of the
    complete target. Slew limits also apply when confidence decreases: stale
    velocity decays to zero instead of persisting or causing a command step.
    The complete robot command still requires the existing motion governor.
    """
    for pose in (G, flange, tcp):
        inverse(pose)
    velocity = _vector(origin_velocity, "relative origin velocity")
    previous = _vector(previous_flange, "previous flange compensation")
    values = [dt, linear_speed, angular_speed, linear_accel, angular_accel]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("compensation duration and limits must be finite and positive")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("relative-base confidence must be in [0, 1]")
    raw = velocity.copy()
    raw[:3] += np.cross(raw[3:], flange[:3, 3] - G[:3, 3])
    desired = raw * confidence
    desired[:3] = _bounded(desired[:3], linear_speed)
    desired[3:] = _bounded(desired[3:], angular_speed)
    effective = previous.copy()
    effective[:3] += _bounded(desired[:3] - previous[:3], linear_accel * dt)
    effective[3:] += _bounded(desired[3:] - previous[3:], angular_accel * dt)
    # The limits do not change during a session; a malformed prior command is
    # rejected rather than silently using it as an acceleration anchor.
    if (np.linalg.norm(previous[:3]) > linear_speed + 1e-10
            or np.linalg.norm(previous[3:]) > angular_speed + 1e-10):
        raise ValueError("previous base compensation exceeds its speed limits")
    effective_tcp = effective.copy()
    effective_tcp[:3] += np.cross(effective[3:], tcp[:3, 3] - flange[:3, 3])
    effective_origin = effective.copy()
    effective_origin[:3] -= np.cross(effective[3:], flange[:3, 3] - G[:3, 3])
    return {"raw_flange": raw, "requested_flange": desired,
            "effective_flange": effective, "effective_tcp": effective_tcp,
            "effective_origin": effective_origin, "lambda": float(confidence),
            "limited": not np.allclose(effective, raw * confidence, atol=1e-12, rtol=0.)}
