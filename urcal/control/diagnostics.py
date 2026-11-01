"""Read-only calibration diagnostics, separate from acceptance and motion.

Crosses are projections of the estimated pose onto the original camera image.
They visualize agreement with detections, not independent ground truth.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from urcal.inference.display import import_opencv

cv2 = import_opencv()


def pose_input_diagnostics(results, estimators, joints, K, dist, regions):
    """Snapshot PnP initialization inputs; do not estimate or accept any pose.

    ``solver_indices`` selects the 3D/2D rows for EMPnP.fit (at least four
    selected rows are required for the wrapper to call it):
    RobotEstimator first applies result.valid, then solve_pose_empnp filters
    nonfinite coordinates. FK is the same deterministic function evaluated
    with the captured joints. Covariance and previous-pose priors affect only
    subsequent EM iterations, so they are deliberately outside this record.
    RANSAC RNG state is not captured; replaying inputs need not reproduce its
    stochastic result. The 3 px value documents EMPnP.fit, not a new setting.
    """
    if any(len(values) != 2 for values in (results, estimators, joints, regions)):
        raise ValueError("PnP diagnostics require exactly two robot inputs")
    camera = np.asarray(K, dtype=float).copy()
    if camera.shape != (3, 3):
        raise ValueError("PnP diagnostics require a 3x3 camera matrix")
    robots = {}
    for name, result, estimator, q, region in zip(("ur5e", "ur10e"), results, estimators, joints, regions):
        q = np.asarray(q, dtype=float).copy()
        points = np.asarray(result.points, dtype=float).copy()
        valid = np.asarray(result.valid, dtype=bool).copy()
        if q.shape != (6,):
            raise ValueError(f"{name}: PnP diagnostics require six captured joints")
        points3d = np.asarray(estimator.fk(q), dtype=float).copy()
        if (points.ndim != 2 or points.shape[1] != 2
                or points3d.shape != (len(points), 3) or valid.shape != (len(points),)
                or len(estimator.key_names) != len(points) or len(region) != 4):
            raise ValueError(f"{name}: inconsistent PnP diagnostic point shapes")
        selected = valid & np.isfinite(points).all(axis=1) & np.isfinite(points3d).all(axis=1)
        robots[name] = {
            "key_names": list(estimator.key_names), "region": list(region),
            "q_rad": q, "points_3d_base_m": points3d,
            "detected_px": points, "valid": valid,
            "solver_indices": np.flatnonzero(selected), "empnp_ok": bool(result.ok),
        }
        raw_rvec, raw_tvec = getattr(result, "empnp_rvec", None), getattr(result, "empnp_tvec", None)
        robots[name].update({
            "empnp_ok": bool(result.ok or getattr(result, "constraint_error", "")),
            "pose_ok": bool(result.ok),
            "empnp_rvec": None if raw_rvec is None and result.rvec is None else np.asarray(
                result.rvec if raw_rvec is None else raw_rvec).copy(),
            "empnp_tvec": None if raw_tvec is None and result.tvec is None else np.asarray(
                result.tvec if raw_tvec is None else raw_tvec).copy(),
            "output_rvec": None if result.rvec is None else np.asarray(result.rvec).copy(),
            "output_tvec": None if result.tvec is None else np.asarray(result.tvec).copy(),
        })
    return {
        "schema": "urcal.pnp-initialization.v1", "K": camera,
        "dist": np.asarray(dist, dtype=float).copy(),
        "initialization": {
            "method": "cv2.solvePnPRansac", "opencv_version": cv2.__version__,
            "reprojection_error_px": 3.0, "other_options": "OpenCV defaults",
            "source": "urcal.inference.my_pnp.EMPnP.fit",
        },
        "robots": robots, "common_base_plane": getattr(results[0], "base_plane", None),
    }


def calibration_debug_record(frame, results, estimators, joints, K, dist, regions,
                             max_mean_px=5., max_pixel_px=8., min_points=5, check_reprojection=True):
    """Build a replayable diagnostic record without changing estimates or gates."""
    robots = {}
    for name, result, estimator, q, region in zip(("ur5e", "ur10e"), results, estimators, joints, regions):
        points = np.asarray(result.points, dtype=float)
        valid = np.asarray(result.valid, dtype=bool) & np.isfinite(points).all(axis=1)
        projected = np.full_like(points, np.nan)
        errors = np.full(len(points), np.nan)
        points3d = np.asarray(estimator.fk(q), dtype=float)
        if name in frame.poses:
            pose = frame.poses[name]
            rvec = cv2.Rodrigues(pose[:3, :3])[0]
            projected = cv2.projectPoints(points3d, rvec, pose[:3, 3], K, dist)[0].reshape(-1, 2)
            errors[valid] = np.linalg.norm(projected[valid] - points[valid], axis=1)
        finite = np.flatnonzero(np.isfinite(errors))
        worst = int(finite[np.argmax(errors[finite])]) if finite.size else None
        robots[name] = {
            "key_names": list(estimator.key_names), "region": list(region),
            "q_rad": np.asarray(q).copy(), "points_3d_base_m": points3d.copy(),
            "detected_px": points.copy(), "valid": valid.copy(),
            "empnp_ok": bool(result.ok),
            "empnp_rvec": None if result.rvec is None else np.asarray(result.rvec).copy(),
            "empnp_tvec": None if result.tvec is None else np.asarray(result.tvec).copy(),
            "camera_from_base": frame.poses[name].copy() if name in frame.poses else None,
            "projected_px": projected, "errors_px": errors, "worst_index": worst,
            "count": int(valid.sum()), "raw_mean_px": frame.raw_mean_px.get(name),
            "raw_max_px": frame.raw_max_px.get(name),
        }
        for component in ("rvec", "tvec"):
            raw = getattr(result, "empnp_" + component, None)
            if raw is not None:
                robots[name]["empnp_" + component] = np.asarray(raw).copy()
        robots[name]["pose_ok"] = bool(result.ok)
        robots[name]["empnp_ok"] = bool(result.ok or getattr(result, "constraint_error", ""))
    return {"schema": "urcal.calibration-debug.v1", "sequence": frame.sequence,
            "captured_at_monotonic": frame.captured_at,
            "K": np.asarray(K).copy(), "dist": np.asarray(dist).copy(),
            "frame_reason": frame.reason,
            "limits": {"mean_px": max_mean_px, "max_px": max_pixel_px, "min_points": min_points,
                       "check_reprojection": bool(check_reprojection)},
            "robots": robots, "common_base_plane": getattr(results[0], "base_plane", None)}


def diagnostic_lines(record):
    """Return compact raw-error summaries suitable for the 640 px video."""
    limits = record["limits"]
    checked = limits.get("check_reprojection", True)
    label = "RAW" if checked else "RAW limits OFF"
    lines = []
    for name, robot in record["robots"].items():
        mean, maximum = robot["raw_mean_px"], robot["raw_max_px"]
        if mean is None or maximum is None:
            state = "PnP failed" if not robot["empnp_ok"] else "invalid geometry"
            lines.append((f"{name} {label} n={robot['count']}/7: {state}", False))
            continue
        worst = robot["worst_index"]
        key = "?" if worst is None else robot["key_names"][worst].removeprefix(name + "_")
        okay = (checked and robot["count"] >= limits["min_points"] and 0 <= mean <= limits["mean_px"]
                and mean <= maximum <= limits["max_px"] and not record["frame_reason"])
        mean_text = f"{mean:.1f}/{limits['mean_px']:.1f}" if checked else f"{mean:.1f}"
        max_text = f"{maximum:.1f}/{limits['max_px']:.1f}" if checked else f"{maximum:.1f}"
        lines.append((f"{name} {label} n={robot['count']}/7 mean={mean_text} "
                      f"max={max_text}px worst={key}", okay))
    return lines


def draw_calibration_debug(image, record, show_keypoints=True, *, show_text=True):
    """Draw residual geometry; optional saved text does not establish READY."""
    height, width = image.shape[:2]
    if show_keypoints:
        for robot in record["robots"].values():
            for detected, projected, valid in zip(robot["detected_px"], robot["projected_px"], robot["valid"]):
                if not valid or not np.isfinite(projected).all():
                    continue
                # Clip before conversion to OpenCV integer coordinates, even for
                # finite but pathological rejected estimates.
                a = tuple(np.rint(np.clip(detected, -100000, 100000)).astype(int))
                b = tuple(np.rint(np.clip(projected, -100000, 100000)).astype(int))
                visible, a, b_clipped = cv2.clipLine((0, 0, width, height), a, b)
                if visible:
                    cv2.line(image, a, b_clipped, (255, 80, 255), 1, cv2.LINE_AA)
                if 0 <= b[0] < width and 0 <= b[1] < height:
                    cv2.drawMarker(image, b, (255, 80, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        if show_text:
            cv2.putText(image, "Magenta +: pose reprojection; lines: detection residual", (8, height - 77),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 160, 255), 1, cv2.LINE_AA)
    if show_text:
        for index, (text, okay) in enumerate(diagnostic_lines(record)):
            point = (8, height - 57 + index * 20)
            cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, .38,
                        (80, 230, 120) if okay else (80, 160, 255), 1, cv2.LINE_AA)
    return image


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_calibration_debug(directory, raw_image, overlay, record, control_message=""):
    """Save only on explicit keypress; never overwrite existing artifacts."""
    encoded = []
    for image in (raw_image, overlay):
        success, data = cv2.imencode(".png", image)
        if not success:
            raise ValueError("PNG encoding failed")
        encoded.append(data.tobytes())
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:12]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = {"raw": stem + "-raw.png", "overlay": stem + "-overlay.png"}
    payload = _json_safe({**record, "control_message": str(control_message), "images": images})
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for name, data in zip(images.values(), encoded):
        with (directory / name).open("xb") as handle:
            handle.write(data)
    destination = directory / (stem + ".json")
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return destination
