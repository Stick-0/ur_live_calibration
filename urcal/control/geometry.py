"""Rigid transforms for dual-arm flange control, with no hardware side effects.

``T_A_B`` maps coordinates expressed in B into A; distances are metres and
rotation vectors are radians. ``F`` is the UR controller's zero-TCP tool
flange, not the differently oriented ROS attachment frame named ``flange``.

Nominal frame mapping follows the existing dataset FK and official UR sources:
https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_description/doc/robot_frames.html
https://raw.githubusercontent.com/UniversalRobots/Universal_Robots_ROS2_Description/rolling/urdf/ur_macro.xacro
https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-scriptmanual/G5/get_tcp_offset.htm

The repository's UR5e FK omits the standard base_link-to-base pi yaw, so its
model frame already has controller-base axes. UR10e retains that yaw and adds
a 5 mm model-world height offset. The official wrist_3-to-flange-to-tool0
rotations multiply to identity. These nominal transforms do not correct the
individual robot's factory kinematic calibration; compare against controller
FK before enabling motion. Importing this module never opens a connection.
"""

import numpy as np
from scipy.spatial.transform import Rotation


_ROTATION_TOLERANCE = 1e-6
_OPPOSING_ROTATION = np.diag([1.0, -1.0, -1.0])


def _finite_array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}")
    return result


def _validate_rotation(value, name="rotation"):
    R = _finite_array(value, (3, 3), name)
    if not np.allclose(R.T @ R, np.eye(3), atol=_ROTATION_TOLERANCE, rtol=0):
        raise ValueError(f"{name} must be an orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(R), 1.0, atol=_ROTATION_TOLERANCE, rtol=0):
        raise ValueError(f"{name} must be a proper rotation with determinant +1")
    return R


def _validate_transform(value, name="transform"):
    T = _finite_array(value, (4, 4), name)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    _validate_rotation(T[:3, :3], name + " rotation")
    return T


def _positive_gap(gap):
    try:
        value = float(gap)
    except (TypeError, ValueError) as exc:
        raise ValueError("gap must be a positive finite distance in metres") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("gap must be a positive finite distance in metres")
    return value


def pose_matrix(pose6):
    """Convert UR [x, y, z, rx, ry, rz] to a validated 4x4 transform."""
    pose = _finite_array(pose6, (6,), "pose")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return _validate_transform(result)


def matrix_pose(T):
    """Convert a proper 4x4 transform to a float64 UR pose ndarray of shape (6,)."""
    T = _validate_transform(T)
    return np.concatenate((T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()))


def inverse(T):
    """Invert a validated rigid transform, without modifying the input."""
    T = _validate_transform(T)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def rotation_distance(A, B):
    """Return the shortest angular distance in radians (3x3 rotations or SE(3))."""
    def rotation(value):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected a 3x3 rotation or 4x4 rigid transform") from exc
        if array.shape == (4, 4):
            return _validate_transform(array)[:3, :3]
        return _validate_rotation(array)

    return float(Rotation.from_matrix(rotation(B) @ rotation(A).T).magnitude())


def model_from_controller_base(robot):
    """Return the fixed dataset-FK mapping, independent of PnP output settings."""
    if robot not in ("ur5e", "ur10e"):
        raise ValueError("robot must be 'ur5e' or 'ur10e'")
    result = np.eye(4, dtype=np.float64)
    if robot == "ur10e":
        result[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        result[2, 3] = 0.005
    return result


def calibration_output_mapping(robot, *, ur10e_output_correction=True):
    """Optional right-multiplied transform on the camera's PnP output.

    Disabled means use the raw PnP pose as the control base estimate. This is
    a changed interpretation of the output frame, not a change to training
    keypoints or the controller-base kinematics. The fixed nominal mapping
    above must still be used to compute encoder FK/Jacobians.
    """
    if not isinstance(ur10e_output_correction, bool):
        raise ValueError("ur10e_output_correction must be a boolean")
    nominal = model_from_controller_base(robot)
    return nominal if ur10e_output_correction else np.eye(4, dtype=np.float64)


def model_flange_pose(robot, q):
    """Return nominal T_model_flange from six joints using the existing live FK.

    Imports the inference module only when called; does not instantiate its
    camera, neural network, or RTDE interfaces.
    """
    model_from_controller_base(robot)  # Reject unknown models before importing.
    q = _finite_array(q, (6,), "joint positions")
    if robot == "ur5e":
        from urcal.inference.infer_live import JOINT_NAMES, compute_ur5e_link_transforms

        fk = compute_ur5e_link_transforms
    else:
        from urcal.inference.infer_live_ur10e import JOINT_NAMES, compute_ur10e_link_transforms

        fk = compute_ur10e_link_transforms
    joints = {name: float(value) for name, value in zip(JOINT_NAMES, q)}
    return _validate_transform(fk(joints)["wrist_3_link"], "model flange pose").copy()


def flange_from_tcp(T_B_TCP, T_F_TCP):
    """Remove the active TCP offset: T_B_F = T_B_TCP @ inverse(T_F_TCP)."""
    return _validate_transform(T_B_TCP, "TCP pose") @ inverse(T_F_TCP)


def facing_follower(T_C_leader, gap):
    """Place the follower gap metres along leader +Z with opposed +Z normals.

    The frames share +X; the follower's +Y and +Z oppose the leader's axes.
    """
    T_C_leader = _validate_transform(T_C_leader, "leader flange pose")
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = _OPPOSING_ROTATION
    relative[2, 3] = _positive_gap(gap)
    return T_C_leader @ relative




def pose_delta(current, target):
    """Return (linear_error, angular_error_rotvec), expressed in the parent frame."""
    current = _validate_transform(current, "current pose")
    target = _validate_transform(target, "target pose")
    linear = target[:3, 3] - current[:3, 3]
    angular = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return linear, angular
