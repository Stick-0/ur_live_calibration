"""Joint image refinement for two upright bases mounted on one plane.

The plane is estimated in the camera frame: it need not be camera XY, and
the robots retain independent yaw and separation within that plane.  All
transforms use metres and map child coordinates into the parent frame.
This module performs no camera, network, model-loading or robot operations.
"""

from collections.abc import Mapping
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from urcal.control.geometry import _validate_transform, inverse, rotation_distance
from urcal.inference.display import import_opencv


cv2 = import_opencv()
ROBOTS = ("ur5e", "ur10e")
_MIN_DEPTH = 1e-6


def _transforms(values, name):
    if not isinstance(values, Mapping) or any(robot not in values for robot in ROBOTS):
        raise ValueError(f"{name} must contain ur5e and ur10e transforms")
    result = {robot: _validate_transform(values[robot], f"{name}[{robot}]").copy()
              for robot in ROBOTS}
    for pose in result.values():
        # Validation allows floating-point SO(3) tolerance.  Remove that tiny
        # roundoff before enforcing an exact common normal.
        pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    return result


def _common_normal(bases):
    normal = bases["ur5e"][:3, 2] + bases["ur10e"][:3, 2]
    length = float(np.linalg.norm(normal))
    if length < 1e-6:
        raise ValueError("opposing base +Z normals do not define an unambiguous common plane")
    return normal / length


def _normal_rotation(source, target):
    """Shortest proper rotation from one unit normal to another."""
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1., 1.))
    if sine < 1e-15:
        return np.eye(3)
    return Rotation.from_rotvec(cross * (np.arctan2(sine, cosine) / sine)).as_matrix()


def project_coplanar_bases(bases):
    """Symmetrically project camera-from-controller bases onto one plane.

    The plane passes through the midpoint of the two base origins.  Their
    mean +Z normal defines its normal; only the shortest tilt corrections
    and opposite normal translations are applied.  In-plane displacement
    and independent yaw are preserved.  Inputs are never modified.
    """
    result = _transforms(bases, "bases")
    normal = _common_normal(result)
    midpoint = (result["ur5e"][:3, 3] + result["ur10e"][:3, 3]) / 2.
    for robot in ROBOTS:
        pose = result[robot]
        pose[:3, :3] = _normal_rotation(pose[:3, 2], normal) @ pose[:3, :3]
        pose[:3, 3] -= normal * np.dot(normal, pose[:3, 3] - midpoint)
    return result


def _plane_diagnostics(bases):
    normal = _common_normal(bases)
    a, b = (bases[robot] for robot in ROBOTS)
    # atan2 retains sub-microradian precision near equal normals.
    angle = np.arctan2(np.linalg.norm(np.cross(a[:3, 2], b[:3, 2])),
                       np.clip(np.dot(a[:3, 2], b[:3, 2]), -1., 1.))
    return {"height_difference_m": float(abs(np.dot(normal, b[:3, 3] - a[:3, 3]))),
            "normal_angle_deg": float(np.rad2deg(angle))}


def _points(values, width, name):
    if not isinstance(values, Mapping) or any(robot not in values for robot in ROBOTS):
        raise ValueError(f"{name} must contain ur5e and ur10e point arrays")
    result = {}
    for robot in ROBOTS:
        array = np.asarray(values[robot], dtype=np.float64)
        if (array.ndim != 2 or array.shape[1] != width or array.shape[0] < 4
                or not np.isfinite(array).all()):
            raise ValueError(f"{name}[{robot}] requires at least four finite points of shape (N, {width})")
        result[robot] = np.ascontiguousarray(array).copy()
    return result


def _camera(K, dist):
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-9, rtol=0)):
        raise ValueError("K must be a finite camera matrix with positive focal lengths")
    distortion = np.asarray([] if dist is None else dist, dtype=np.float64).reshape(-1)
    if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("dist must contain 0, 4, 5, 8, 12 or 14 finite distortion coefficients")
    return K.copy(), distortion.copy()


def _project(pose, points, K, dist):
    return cv2.projectPoints(points, Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
                             pose[:3, 3], K, dist)[0].reshape(-1, 2)


def _positive_depth(poses, points):
    return all(np.all((points[robot] @ poses[robot][:3, :3].T
                       + poses[robot][:3, 3])[:, 2] > _MIN_DEPTH)
               for robot in ROBOTS)


def _parameters(bases):
    first = bases["ur5e"]
    relative = inverse(first) @ bases["ur10e"]
    return np.concatenate((first[:3, 3], Rotation.from_matrix(first[:3, :3]).as_rotvec(),
                           relative[:2, 3], [np.arctan2(relative[1, 0], relative[0, 0])]))


def _bases(parameters):
    first, relative = np.eye(4), np.eye(4)
    first[:3, 3] = parameters[:3]
    first[:3, :3] = Rotation.from_rotvec(parameters[3:6]).as_matrix()
    relative[:2, 3] = parameters[6:8]
    relative[:3, :3] = Rotation.from_rotvec([0., 0., parameters[8]]).as_matrix()
    return {"ur5e": first, "ur10e": first @ relative}


def _soft_l1_cost(residual):
    # Same loss and pixel scale as least_squares below; used to avoid returning
    # an interrupted solve that is worse than its valid constrained seed.
    return float(np.sum(9. * (np.sqrt(1. + (residual / 3.) ** 2) - 1.)))


def refine_coplanar_pair(poses_model, points_model, points_image, K, dist, mappings,
                         *, max_nfev=40):
    """Refine two model-frame poses jointly using original-image keypoints.

    ``mappings[r]`` is T_model_controller_base.  The nine parameters are one
    complete base pose and the other base's relative X, Y and yaw.  Thus all
    iterates satisfy parallel upright +Z axes and zero relative base height.
    Only current-frame pixel residuals are minimized (soft-L1, 3 px scale):
    there is no distance, yaw, previous-pose, or camera-horizontal prior.

    Caller must pass only valid correspondences, at least four per robot.
    A finite, positive-depth improvement may be used even at the evaluation
    limit.  Unusable solves fall back to a valid geometric projection, never
    the unconstrained input.  Invalid inputs or an invalid projected seed
    raise ValueError instead of claiming success.
    """
    started = time.perf_counter()
    poses = _transforms(poses_model, "poses_model")
    mappings = _transforms(mappings, "mappings")
    model = _points(points_model, 3, "points_model")
    image = _points(points_image, 2, "points_image")
    K, dist = _camera(K, dist)
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    for robot in ROBOTS:
        if len(model[robot]) != len(image[robot]):
            raise ValueError(f"{robot} model/image correspondence counts differ")
    if not _positive_depth(poses, model):
        raise ValueError("input poses must put every valid keypoint in front of the camera")
    raw_bases = {robot: poses[robot] @ mappings[robot] for robot in ROBOTS}
    seed_bases = project_coplanar_bases(raw_bases)
    inverse_mappings = {robot: inverse(mappings[robot]) for robot in ROBOTS}

    def model_poses(bases):
        return {robot: bases[robot] @ inverse_mappings[robot] for robot in ROBOTS}

    def residual(parameters):
        current = model_poses(_bases(parameters))
        return np.concatenate([(_project(current[robot], model[robot], K, dist)
                                - image[robot]).ravel() for robot in ROBOTS])

    seed_poses = model_poses(seed_bases)
    if not _positive_depth(seed_poses, model):
        raise ValueError("common-plane projection puts valid keypoints behind the camera")
    x0 = _parameters(seed_bases)
    seed_residual = residual(x0)
    if not np.isfinite(seed_residual).all():
        raise ValueError("common-plane seed has nonfinite pixel projections")
    final_bases = seed_bases
    solver_success, status, message, nfev, fallback = False, -1, "", 0, True
    try:
        optimized = least_squares(residual, x0, loss="soft_l1", f_scale=3.,
                                  x_scale="jac", max_nfev=int(max_nfev))
        solver_success, status = bool(optimized.success), int(optimized.status)
        message, nfev = str(optimized.message), int(optimized.nfev)
        if np.asarray(optimized.x).shape == (9,) and np.isfinite(optimized.x).all():
            candidate = _bases(optimized.x)
            candidate_poses = model_poses(candidate)
            candidate_residual = residual(optimized.x)
            if (np.isfinite(candidate_residual).all() and _positive_depth(candidate_poses, model)
                    and _soft_l1_cost(candidate_residual) <= _soft_l1_cost(seed_residual) + 1e-9):
                final_bases, fallback = candidate, False
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, cv2.error) as exc:
        message = str(exc)
    final_poses = _transforms(model_poses(final_bases), "refined poses")
    diagnostics = {
        "applied": True, "solver_success": solver_success, "solver_status": status,
        "solver_message": message, "nfev": nfev, "fallback": fallback,
        "raw_plane": _plane_diagnostics(raw_bases),
        "final_plane": _plane_diagnostics(final_bases),
        "reprojection": {}, "correction": {},
    }
    for robot in ROBOTS:
        before = np.linalg.norm(_project(poses[robot], model[robot], K, dist) - image[robot], axis=1)
        after = np.linalg.norm(_project(final_poses[robot], model[robot], K, dist) - image[robot], axis=1)
        diagnostics["reprojection"][robot] = {
            "before_mean_px": float(np.mean(before)), "before_max_px": float(np.max(before)),
            "after_mean_px": float(np.mean(after)), "after_max_px": float(np.max(after)),
        }
        diagnostics["correction"][robot] = {
            "translation_m": float(np.linalg.norm(final_bases[robot][:3, 3] - raw_bases[robot][:3, 3])),
            "rotation_deg": float(np.rad2deg(rotation_distance(raw_bases[robot], final_bases[robot]))),
        }
    diagnostics["elapsed_s"] = float(time.perf_counter() - started)
    return final_poses, diagnostics
