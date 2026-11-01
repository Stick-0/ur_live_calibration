"""Read-only TCP target error telemetry in the follower controller base frame.

Inputs are already validated display snapshots; this module never reads a robot,
changes a target, advances an estimator, or uses a predicted next control step.
"""

import numpy as np

from urcal.control.geometry import flange_from_tcp, inverse


def transformed_tcp_target(scene, leader, follower, relative_target):
    """Transform an accepted relative FLANGE target to follower-base TCP."""
    bases, robots = scene["bases"], scene["robots"]
    leader_flange = flange_from_tcp(robots[leader]["tcp"], robots[leader]["tcp_offset"])
    inverse(relative_target)
    inverse(bases[leader])
    inverse(robots[follower]["tcp_offset"])
    return (inverse(bases[follower]) @ bases[leader] @ leader_flange
            @ relative_target @ robots[follower]["tcp_offset"])


def position_error_sample(scene, *, now, leader, follower, phase,
                          relative_target=None, target_tcp=None,
                          target_source="unavailable", max_robot_age=.5,
                          max_calibration_age=.5, clock="monotonic", episode=None,
                          reason=""):
    """Return signed goal-minus-actual XYZ metres, or an explicit invalid gap.

    A valid sample retains ``scene['at']``: cached TCP readings never receive a
    new timestamp. ``measurement_age`` is the camera age evaluated at ``now``
    by the scene producer, not at the older RTDE timestamp. Frozen alignment
    bases are intentionally allowed beyond the live-camera age limit.
    Simulation callers supply a virtual-time scene.
    """
    payload = {"at": float(now), "valid": False, "follower": str(follower),
               "phase": str(phase), "target_source": str(target_source),
               "reason": str(reason), "clock": str(clock)}
    if episode is not None:
        payload["episode"] = int(episode)
    try:
        if reason:
            return payload
        if scene is None:
            raise ValueError("waiting for fresh paired robot states and calibration")
        stamp = float(scene["at"])
        if not np.isfinite([now, stamp]).all() or stamp > now or now - stamp > max_robot_age:
            raise ValueError("robot state is stale or has an invalid timestamp")
        if not scene.get("calibration_valid", False):
            raise ValueError(scene.get("reason") or "calibration is unavailable")
        ages = scene["measurement_age"]
        ages = list(ages.values()) if isinstance(ages, dict) else [ages]
        if not ages or not np.isfinite(ages).all() or min(ages) < 0:
            raise ValueError("invalid calibration age")
        if scene.get("calibration_mode") != "frozen" and max(ages) > max_calibration_age:
            raise ValueError("calibration is stale")
        current = np.asarray(scene["robots"][follower]["tcp"], dtype=float)
        inverse(current)
        if target_tcp is None:
            if relative_target is None:
                raise ValueError("tracking target is unavailable")
            target_tcp = transformed_tcp_target(scene, leader, follower, np.asarray(relative_target, dtype=float))
        target_tcp = np.asarray(target_tcp, dtype=float)
        inverse(target_tcp)
        error = target_tcp[:3, 3] - current[:3, 3]
        payload.update(at=stamp, valid=True, error_m=error.tolist(), reason="")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, np.linalg.LinAlgError) as exc:
        payload["reason"] = str(exc)
    return payload
