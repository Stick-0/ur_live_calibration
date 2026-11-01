"""Braking proposals while the live calibration is being reacquired.

Only the runtime may authorize this bounded transition using its elapsed-time
budget and fresh robot state. Frozen visual geometry is retained for motion
checks; it is never promoted to a new camera measurement or a tracking target.
"""

import copy

import numpy as np

from urcal.control.geometry import inverse, matrix_pose
from urcal.cooperative.state import RelativeSnapshot


def frozen_braking_snapshot(snapshot, now, *, leader):
    """Copy last-valid geometry without extrapolation or refreshed timestamps."""
    now = float(now)
    if not np.isfinite(now) or not np.isfinite(snapshot.at) or snapshot.at > now:
        raise ValueError("braking requires a finite current time after the saved snapshot")
    names = ("ur5e", "ur10e")
    if leader not in names:
        raise ValueError("braking leader must be ur5e or ur10e")
    bases, ages = {}, {}
    for name in names:
        pose = np.asarray(snapshot.base_poses[name], dtype=float)
        matrix_pose(pose)  # Reject non-finite or improper SE(3) geometry.
        bases[name] = pose.copy()
        captured = float(snapshot.measurement_times[name])
        received = float(snapshot.received_times[name])
        if not np.isfinite([captured, received]).all() or max(captured, received) > now:
            raise ValueError("braking snapshot timestamps must be finite and not in the future")
        ages[name] = now - captured
    follower = next(name for name in names if name != leader)
    return RelativeSnapshot(
        at=float(snapshot.at), base_poses=bases,
        base_twists={name: np.zeros(6) for name in names},
        relative_pose=inverse(bases[leader]) @ bases[follower],
        relative_twist=np.zeros(6), measurement_age=ages,
        measurement_times=copy.deepcopy(snapshot.measurement_times),
        received_times=copy.deepcopy(snapshot.received_times),
        prediction_dt={name: 0. for name in names})


def prepare_calibration_braking(controller, now, states, dt, snapshot):
    """Ask the normal controller checks for a zero-tracking, bounded brake.

    The controller's braking branch retains measured-state, workspace,
    Jacobian, swept geometry, and native-transport proposal checks. Runtime
    must still verify fresh robot state immediately before transmitting it,
    and commit only after successful transmission. This function never sends
    commands, commits state, or authorizes indefinite use of old calibration.
    """
    frozen = frozen_braking_snapshot(snapshot, now, leader=controller.leader)
    proposal = controller.prepare(now, states, dt, active=True,
                                  snapshot_override=frozen, calibration_braking=True)
    proposal.record.update({
        "calibration_braking": True,
        "calibration_geometry_frozen": True,
        "calibration_geometry_at": float(snapshot.at),
    })
    return proposal
