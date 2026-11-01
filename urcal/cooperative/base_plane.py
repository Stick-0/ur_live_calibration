"""Preserve the common physical mounting plane after temporal pose filtering.

The optional displayed/control output convention need not be the physical
controller-base convention. Project in the latter, then restore the selected
output convention. Camera capture/receipt times are never changed here.
"""

from dataclasses import replace

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base
from urcal.inference.coplanar import project_coplanar_bases


ROBOTS = ("ur5e", "ur10e")


def _physical_offsets(mappings):
    return {name: inverse(mappings[name]) @ model_from_controller_base(name) for name in ROBOTS}


def project_output_bases(bases, mappings):
    """Return coplanar bases in the caller's selected output convention."""
    offsets = _physical_offsets(mappings)
    physical = project_coplanar_bases({name: bases[name] @ offsets[name] for name in ROBOTS})
    return {name: physical[name] @ inverse(offsets[name]) for name in ROBOTS}


def constrain_base_snapshot(snapshot, leader, mappings):
    """Project poses and research-mode twists onto a shared mounting plane.

    Pose mode enters with zero visual twists and retains them. For the optional
    velocity research mode, project origin velocities onto the plane's tangent
    constraints, retaining common tilt and independent rotation about the normal.
    This removes incompatible velocity components, not encoder measurements.
    """
    offsets = _physical_offsets(mappings)
    physical = {name: snapshot.base_poses[name] @ offsets[name] for name in ROBOTS}
    projected = project_coplanar_bases(physical)
    velocities = {}
    for name in ROBOTS:
        velocity = np.asarray(snapshot.base_twists[name], dtype=float).copy()
        if velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError("base-plane projection requires finite six-component twists")
        # A constant local translation moves the twist's reference origin.
        velocity[:3] += np.cross(velocity[3:], physical[name][:3, 3]
                                - snapshot.base_poses[name][:3, 3])
        # Evaluate the same spatial motion at the projected origin. In
        # particular, a common rigid camera motion must remain common.
        velocity[:3] += np.cross(velocity[3:], projected[name][:3, 3] - physical[name][:3, 3])
        velocities[name] = velocity
    normal = projected["ur5e"][:3, 2]
    tilt = sum(velocity[3:] - normal * np.dot(normal, velocity[3:])
               for velocity in velocities.values()) / 2.
    for velocity in velocities.values():
        velocity[3:] = tilt + normal * np.dot(normal, velocity[3:])
    normal_rate = np.cross(tilt, normal)
    separation = projected["ur10e"][:3, 3] - projected["ur5e"][:3, 3]
    height_rate = (np.dot(normal, velocities["ur10e"][:3] - velocities["ur5e"][:3])
                   + np.dot(normal_rate, separation))
    velocities["ur5e"][:3] += .5 * height_rate * normal
    velocities["ur10e"][:3] -= .5 * height_rate * normal
    bases = {name: projected[name] @ inverse(offsets[name]) for name in ROBOTS}
    for name in ROBOTS:
        velocities[name][:3] += np.cross(velocities[name][3:], bases[name][:3, 3]
                                         - projected[name][:3, 3])
    follower = next(name for name in ROBOTS if name != leader)
    rotation = bases[leader][:3, :3].T
    displacement = bases[follower][:3, 3] - bases[leader][:3, 3]
    relative_twist = np.concatenate((
        rotation @ (velocities[follower][:3] - velocities[leader][:3]
                    - np.cross(velocities[leader][3:], displacement)),
        rotation @ (velocities[follower][3:] - velocities[leader][3:])))
    return replace(snapshot, base_poses=bases, base_twists=velocities,
                   relative_pose=inverse(bases[leader]) @ bases[follower],
                   relative_twist=relative_twist)
