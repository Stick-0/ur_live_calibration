"""Offline audit and label-driven noise tuning for the planar relative filter.

Only paired camera measurements are used, never already-filtered control bases.
Joint motion is evidence of arm motion, not proof that a table/base was fixed.
R is fitted only to explicitly labelled stationary-base training intervals.
No camera, robot SDK or model checkpoint is opened. Source logs are read-only.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

import numpy as np

from urcal.control.geometry import inverse, model_from_controller_base


ROBOTS = ("ur5e", "ur10e")
MEASUREMENTS = {"measurement", "alignment_vision", "handoff_measurement"}
LABELS = {"base_stationary_arm_stationary", "base_stationary_arm_moving", "base_moving", "unknown"}
STATIONARY = LABELS - {"base_moving", "unknown"}


@dataclass
class Series:
    path: Path
    settings: dict
    times: np.ndarray
    values: np.ndarray
    joints: np.ndarray
    accepted: np.ndarray
    phases: tuple
    excluded: dict
    timing: dict = field(default_factory=dict)


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2. * np.pi) - np.pi


def planar_relative(model_poses, *, leader="ur10e", plane_tolerance=1e-5):
    """Physical follower-base -> leader-base transform: G = inv(B_f) B_l.

    Model-to-controller mappings are fixed dataset conventions, independent of
    the old display output toggle. A noncoplanar pair is rejected, not projected
    into a fabricated planar measurement by this analysis tool.
    """
    if leader not in ROBOTS:
        raise ValueError("leader must be ur5e or ur10e")
    follower = next(name for name in ROBOTS if name != leader)
    physical = {}
    for name in ROBOTS:
        pose = np.asarray(model_poses[name], dtype=float)
        inverse(pose)
        physical[name] = pose @ model_from_controller_base(name)
    relative = inverse(physical[follower]) @ physical[leader]
    if (abs(relative[2, 3]) > plane_tolerance
            or np.linalg.norm(relative[:3, 2] - [0., 0., 1.]) > plane_tolerance):
        raise ValueError("pair is not physically coplanar")
    return np.array([relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])])


def read_series(path, *, leader="ur10e"):
    """Read finite distinct paired captures, including gate-rejected poses.

    Rejecting upstream gate failures here would censor the very noise being
    audited. Missing poses, invalid geometry, or fewer than five original
    keypoints per robot cannot supply a valid planar sample and are counted.
    """
    path = Path(path)
    rows, settings, seen, excluded = [], {}, set(), Counter()
    receipt_ages, control_ages = [], []
    phase = "preview"
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid/incomplete JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            kind = record.get("kind")
            if kind == "metadata":
                settings = record.get("settings", {})
            elif kind == "armed":
                phase = "alignment" if record.get("phase") == "initial_alignment" else "tracking"
            elif kind == "alignment_complete":
                phase = "tracking"
            elif kind in ("motion_stopped", "recovery_stationary"):
                phase = "recovery"
            elif kind == "control":
                ages = record.get("measurement_age", {})
                ages = list(ages.values()) if isinstance(ages, dict) else [ages]
                control_ages.extend(float(age) for age in ages if age is not None and np.isfinite(age) and age >= 0)
            if kind not in MEASUREMENTS:
                continue
            try:
                stamp = float(record["captured_at"])
                if not np.isfinite(stamp):
                    raise ValueError("invalid capture time")
                key = (stamp, record.get("sequence"))
                if key in seen:
                    excluded["duplicate_capture"] += 1
                    continue
                seen.add(key)
                if any(record.get("valid_counts", {}).get(name, 0) < 5 for name in ROBOTS):
                    excluded["insufficient_keypoints"] += 1
                    continue
                value = planar_relative(record["poses_camera_model"], leader=leader)
                diag = record.get("pose_diagnostics", {}).get("robots", {})
                joints = np.array([diag.get(name, {}).get("q_rad", [np.nan] * 6) for name in ROBOTS])
                if joints.shape != (2, 6):
                    joints = np.full((2, 6), np.nan)
            except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
                excluded[str(exc)] += 1
                continue
            sample_phase = "alignment" if kind == "alignment_vision" else "handoff" if kind == "handoff_measurement" else phase
            rows.append((stamp, value, joints, bool(record.get("accepted", False)), sample_phase))
            age = float(record.get("received_at", record.get("at", stamp))) - stamp
            if np.isfinite(age) and age >= 0:
                receipt_ages.append(age)
    rows.sort(key=lambda row: row[0])
    unique = []
    for row in rows:
        if unique and row[0] == unique[-1][0]:
            excluded["duplicate_time"] += 1
        else:
            unique.append(row)
    return Series(path, settings, np.array([r[0] for r in unique]),
                  np.array([r[1] for r in unique]).reshape(-1, 3),
                  np.array([r[2] for r in unique]).reshape(-1, 2, 6),
                  np.array([r[3] for r in unique], dtype=bool),
                  tuple(r[4] for r in unique), dict(excluded),
                  {name: (np.percentile(values, [50, 95]).tolist() if values else None)
                   for name, values in (("receipt_age_median_p95_s", receipt_ages),
                                        ("control_age_median_p95_s", control_ages))})


def centered_residuals(values):
    """Position median plus locally unwrapped circular yaw median."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("need at least three finite XY/yaw samples")
    angles = values[:, 2]
    anchor = np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())
    local = values.copy()
    local[:, 2] = anchor + wrap_angle(angles - anchor)
    center = np.median(local, axis=0)
    residual = local - center
    residual[:, 2] = wrap_angle(residual[:, 2])
    center[2] = wrap_angle(center[2])
    return residual, center


def robust_covariance(residuals):
    """MAD marginal scales and winsorized, shrunk correlation; SPD by design.

    This describes spread about a labelled fixed pose, not absolute accuracy.
    It is deliberately not fitted from frame differences, which mix true base
    motion, time correlation, delayed joint samples and camera noise.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != 3 or len(residuals) < 10 or not np.isfinite(residuals).all():
        raise ValueError("at least ten finite residuals are needed")
    median = np.median(residuals, axis=0)
    scale = 1.4826022185 * np.median(abs(residuals - median), axis=0)
    scale = np.maximum(scale, [1e-6, 1e-6, 1e-6])
    clipped = np.clip((residuals - median) / scale, -4., 4.)
    covariance = np.cov(clipped, rowvar=False)
    denominator = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(denominator, denominator)
    correlation = .9 * correlation + .1 * np.eye(3)
    return np.diag(scale) @ correlation @ np.diag(scale), scale


def arm_motion(series, threshold=.01):
    """Adjacent recorded joint positions supply arm-motion evidence only."""
    moving = np.zeros(len(series.times), dtype=bool)
    known = np.zeros(len(series.times), dtype=bool)
    if len(series.times) > 1:
        dt = np.diff(series.times)
        good = (dt > 0) & (dt < .5) & np.isfinite(series.joints[1:]).all(axis=(1, 2)) & np.isfinite(series.joints[:-1]).all(axis=(1, 2))
        velocity = np.abs(wrap_angle(np.diff(series.joints, axis=0))) / dt[:, None, None]
        moving[1:] = good & (velocity.max(axis=(1, 2)) > threshold)
        known[1:] = good
    return moving, known


def audit(series):
    report = {"log": str(series.path), "samples": len(series.times), "excluded": series.excluded,
              "base_motion_label": "unknown", "ground_truth_available": False,
              "warning": "Joint motion and PnP changes do not establish whether a base/table moved.",
              "timing": series.timing, "settings": {key: series.settings.get(key) for key in
                           ("common_base_plane", "empnp_use_prev", "filter_alpha", "base_pose_window", "follow_law")}}
    if len(series.times) < 3:
        return report
    residual, _ = centered_residuals(series.values)
    spread = 1.4826022185 * np.median(abs(residual), axis=0)
    moving, known = arm_motion(series)
    report.update(capture_start=float(series.times[0]), duration_s=float(series.times[-1] - series.times[0]),
                  median_period_s=float(np.median(np.diff(series.times))),
                  accepted_fraction=float(series.accepted.mean()),
                  raw_spread_m_m_rad=spread.tolist(), raw_spread_is_noise_estimate=False,
                  arm_moving_fraction=float(moving.sum() / max(1, known.sum())),
                  joint_motion_known_samples=int(known.sum()), phases=[])
    for phase in dict.fromkeys(series.phases):
        mask = np.array(series.phases) == phase
        at = series.times[mask] - series.times[0]
        phase_report = {"phase": phase, "start_s": float(at.min()), "end_s": float(at.max()),
                                 "samples": int(mask.sum()), "arm_moving_samples": int((mask & moving).sum()),
                                 "base_motion_label": "unknown"}
        if mask.sum() >= 10:
            residual, _ = centered_residuals(series.values[mask])
            covariance, scale = robust_covariance(residual)
            correlation = []
            for axis in range(3):
                a, b = residual[:-1, axis], residual[1:, axis]
                correlation.append(float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-12 else 0.)
            phase_report.update(raw_spread_m_m_rad=scale.tolist(), lag1_correlation=correlation,
                                covariance_m_rad=covariance.tolist(), raw_spread_is_noise_estimate=False)
        report["phases"].append(phase_report)
    return report


def load_segments(manifest_path):
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "relative-kalman-segments-v1":
        raise ValueError("expected relative-kalman-segments-v1 manifest")
    leader = data.get("leader", "ur10e")
    segments, cache, ranges = [], {}, {}
    for index, entry in enumerate(data.get("segments", [])):
        if not isinstance(entry, dict) or entry.get("condition") not in LABELS:
            raise ValueError(f"segment {index}: invalid condition")
        if entry.get("partition") not in ("train", "validation"):
            raise ValueError(f"segment {index}: partition must be train or validation")
        if entry["condition"] != "unknown" and not str(entry.get("evidence", "")).strip():
            raise ValueError(f"segment {index}: explicit operator/reference evidence is required")
        start, end = float(entry["start_s"]), float(entry["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"segment {index}: invalid capture-relative time range")
        log = (path.parent / entry["log"]).resolve()
        for old_start, old_end in ranges.get(log, []):
            if max(start, old_start) < min(end, old_end):
                raise ValueError("segments overlap: training/validation leakage or duplicate labelling")
        ranges.setdefault(log, []).append((start, end))
        if log not in cache:
            cache[log] = read_series(log, leader=leader)
        full = cache[log]
        if not len(full.times):
            raise ValueError(f"{log}: no valid paired planar captures")
        mask = (full.times - full.times[0] >= start) & (full.times - full.times[0] < end)
        if mask.sum() < 10:
            raise ValueError(f"segment {index}: need at least ten paired captures")
        selected = Series(log, full.settings, full.times[mask], full.values[mask], full.joints[mask],
                          full.accepted[mask], tuple(np.array(full.phases)[mask]), {})
        segments.append((entry.copy(), selected))
    if not segments:
        raise ValueError("manifest has no segments")
    return segments


def fit_measurement_noise(segments):
    train = [(entry, series) for entry, series in segments
             if entry["partition"] == "train" and entry["condition"] in STATIONARY]
    if not train:
        raise ValueError("no explicitly labelled stationary-base training segment; unknown/PnP motion is not a noise label")
    residuals = np.concatenate([centered_residuals(series.values)[0] for _, series in train])
    covariance, scale = robust_covariance(residuals)
    diagnostics = []
    # Equal consideration per condition prevents long idle records from
    # obscuring the noise observed while the arm itself moves.
    condition_scales = []
    for condition in sorted(STATIONARY):
        subset = [centered_residuals(series.values)[0] for entry, series in train if entry["condition"] == condition]
        if subset:
            _, current_scale = robust_covariance(np.concatenate(subset))
            condition_scales.append(current_scale)
    conservative = np.max(condition_scales, axis=0)
    for entry, series in segments:
        residual, center = centered_residuals(series.values)
        spread = 1.4826022185 * np.median(abs(residual), axis=0)
        diagnostics.append({"log": str(series.path), "condition": entry["condition"],
                            "partition": entry["partition"], "samples": len(series.times),
                            "evidence": entry.get("evidence", ""), "center_m_m_rad": center.tolist(),
                            "spread_m_m_rad": spread.tolist(),
                            "spread_is_measurement_noise": entry["condition"] in STATIONARY})
    return {"measurement_std_m": float(max(.001, *conservative[:2])),
            "measurement_std_yaw": float(max(np.deg2rad(.1), conservative[2]))}, {
                "method": "labelled stationary-base median residuals; MAD marginals; winsorized shrunk correlation",
                "training_covariance_m_rad": covariance.tolist(), "training_scale_m_m_rad": scale.tolist(),
                "training_samples": len(residuals), "segments": diagnostics,
                "absolute_accuracy_validated": False,
                "warning": "Stationary spread includes pose-dependent estimator bias; this is not an accuracy guarantee."}


def replay(times, values, config, *, age=.18):
    """Replay the production filter at a declared capture age, no controller.

    This isolates the relative filter. It does not replay IK, the controller's
    calibration gate/recovery, robot dynamics or all historical scheduling.
    Invalid intervals remain missing; no stale pose is re-labelled as new.
    """
    from urcal.cooperative.relative_kalman import PlanarRelativeStateEstimator
    from urcal.cooperative.state import RelativeStateError
    times, values = np.asarray(times, float), np.asarray(values, float)
    if (times.ndim != 1 or values.shape != (len(times), 3) or not len(times)
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(age) or age < 0 or age > .5):
        raise ValueError("replay requires finite distinct increasing captures, XY/yaw and 0 <= age <= .5")
    estimator = PlanarRelativeStateEstimator(config=config)
    estimates, velocities, confidence, prediction_gain, reasons, rejected = [], [], [], [], [], 0
    for sequence, (stamp, value) in enumerate(zip(times, values)):
        g = np.eye(4)
        c, s = np.cos(value[2]), np.sin(value[2])
        g[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        g[:2, 3] = value[:2]
        try:
            estimator.update_pair({"ur5e": np.eye(4), "ur10e": g}, stamp, stamp + age, sequence)
        except RelativeStateError:
            rejected += 1
        try:
            detail = estimator.estimate(stamp + age).relative_filter
            estimates.append(detail["state"][:3].copy())
            velocities.append(detail["origin_velocity"].copy())
            confidence.append(detail["lambda"])
            prediction_gain.append(np.asarray(detail["prediction_gain"], float).copy())
            reasons.append(detail["reason"])
        except RelativeStateError as exc:
            estimates.append(np.full(3, np.nan))
            velocities.append(np.full(6, np.nan))
            confidence.append(0.)
            prediction_gain.append(np.zeros(3))
            reasons.append(str(exc))
    return {"estimates": np.asarray(estimates), "velocities": np.asarray(velocities),
            "confidence": np.asarray(confidence), "prediction_gain": np.asarray(prediction_gain),
            "reasons": reasons, "rejected": rejected}


def error_metrics(estimate, truth):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    valid = np.isfinite(estimate).all(axis=1) & np.isfinite(truth).all(axis=1)
    if not valid.any():
        return {"valid_samples": 0, "position_rmse_m": None, "yaw_rmse_rad": None}
    delta = estimate[valid] - truth[valid]
    delta[:, 2] = wrap_angle(delta[:, 2])
    return {"valid_samples": int(valid.sum()), "position_rmse_m": float(np.sqrt(np.mean(np.sum(delta[:, :2] ** 2, axis=1)))),
            "yaw_rmse_rad": float(np.sqrt(np.mean(delta[:, 2] ** 2)))}


def synthetic_truth(times):
    """Independent prescribed hold/start/stop/reversal path, crossing yaw pi."""
    t = np.asarray(times)
    travel = np.clip(t - 2., 0., 4.) - np.clip(t - 8., 0., 4.)
    return np.column_stack((1. + .04 * travel, .2 + .01 * travel,
                            wrap_angle(np.pi - .05 + .035 * travel)))


def synthetic_comparison(config, *, seed=2718, age=.18):
    """Known synthetic truth, never presented as real-table validation."""
    rng = np.random.default_rng(seed)
    times = np.r_[0., np.cumsum(rng.uniform(.065, .095, 205))]
    times = times[times <= 16.]
    truth = synthetic_truth(times + age)
    # Generate yaw separately with its angular variance, preserving units.
    measurements = synthetic_truth(times) + rng.normal(size=(len(times), 3)) * np.array(
        [config.measurement_std_m, config.measurement_std_m, config.measurement_std_yaw])
    measurements[:, 2] = wrap_angle(measurements[:, 2])
    schedules = {"stationary": times + age < 2., "moving": (times + age >= 2.) & (times + age < 6.),
                 "stopped": (times + age >= 6.) & (times + age < 8.),
                 "reversal": (times + age >= 8.) & (times + age < 12.), "final_stop": times + age >= 12.}
    comparisons = {}
    for name, current in (("q_half", replace(config, acceleration_std_m=config.acceleration_std_m * .5,
                                              acceleration_std_yaw=config.acceleration_std_yaw * .5)),
                          ("candidate", config),
                          ("yaw_q_half", replace(config, acceleration_std_yaw=config.acceleration_std_yaw * .5)),
                          ("q_double", replace(config, acceleration_std_m=config.acceleration_std_m * 2.,
                                                acceleration_std_yaw=config.acceleration_std_yaw * 2.)),
                          ("short_horizon", replace(config, prediction_horizon=.15))):
        result = replay(times, measurements, current, age=age)
        metrics = {phase: error_metrics(result["estimates"][mask], truth[mask]) for phase, mask in schedules.items()}
        compensation = {phase: {"nonzero_fraction": float((np.linalg.norm(result["velocities"][mask, :3], axis=1) > 1e-5).mean()),
                                "mean_lambda": float(result["confidence"][mask].mean()),
                                "mean_prediction_gain": result["prediction_gain"][mask].mean(axis=0).tolist(),
                                "mean_x_velocity_m_s": float(np.nanmean(result["velocities"][mask, 0])),
                                "mean_linear_m_s": float(np.nanmean(np.linalg.norm(result["velocities"][mask, :3], axis=1))),
                                "max_linear_m_s": float(np.nanmax(np.linalg.norm(result["velocities"][mask, :3], axis=1)))}
                        for phase, mask in schedules.items()}
        comparisons[name] = {"config": asdict(current), "errors": metrics, "rejected": result["rejected"],
                             "compensation": compensation,
                             "nonzero_compensation_fraction": float((np.linalg.norm(result["velocities"][:, :3], axis=1) > 1e-5).mean())}
    stationary_times = np.arange(0., 20., .08)
    stationary_truth = np.repeat([[1., .2, np.pi - .05]], len(stationary_times), axis=0)
    noise = rng.normal(size=stationary_truth.shape) * [config.measurement_std_m, config.measurement_std_m, config.measurement_std_yaw]
    stationary_tests = {}
    for name in ("white_noise", "slow_correlated_bias"):
        observed = stationary_truth + noise
        if name == "slow_correlated_bias":
            # This deliberately violates the white measurement-noise model.
            # A markerless filter alone cannot distinguish slow bias from motion.
            observed[:, 0] += .02 * np.sin(.35 * stationary_times)
            observed[:, 2] += np.deg2rad(.5) * np.sin(.35 * stationary_times)
        filtered = replay(stationary_times, observed, config, age=age)
        stationary_tests[name] = {"raw": error_metrics(observed, stationary_truth),
                                  "filtered": error_metrics(filtered["estimates"], stationary_truth),
                                  "false_linear_compensation_rms_m_s": float(np.sqrt(np.nanmean(np.sum(filtered["velocities"][:, :3] ** 2, axis=1)))),
                                  "false_yaw_compensation_rms_rad_s": float(np.sqrt(np.nanmean(filtered["velocities"][:, 5] ** 2))),
                                  "nonzero_compensation_fraction": float((np.linalg.norm(filtered["velocities"][:, :3], axis=1) > 1e-5).mean()),
                                  "rejected": filtered["rejected"]}
    return {"seed": seed, "kind": "synthetic_known_truth_not_hardware_validation", "age_s": age,
            "capture_samples": len(times), "raw_hold_errors": {phase: error_metrics(measurements[mask], truth[mask]) for phase, mask in schedules.items()},
            "comparisons": comparisons, "stationary_tests": stationary_tests}


def synthetic_validation(config, *, seeds=(31415, 57721, 16180), age=.18):
    """Check the chosen prior on independent seeds, without retuning it.

    The wider-yaw prior is a declared comparator, not an optimizer. Both use
    identical generated observations in each seed; seed 2718 is reserved for
    the exploratory comparison and cannot be called held-out validation.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or 2718 in seeds or min(seeds) < 0:
        raise ValueError("validation requires distinct nonnegative seeds excluding exploratory seed 2718")
    results = {}
    for label, current in (("candidate", config), ("wider_yaw_prior", replace(
            config, acceleration_std_yaw=config.acceleration_std_yaw * 2.))):
        runs = []
        for seed in seeds:
            comparison = synthetic_comparison(current, seed=seed, age=age)
            runs.append({"seed": seed, "capture_samples": comparison["capture_samples"],
                         "trajectory": comparison["comparisons"]["candidate"],
                         "stationary_tests": comparison["stationary_tests"]})
        aggregate = {}
        for phase in runs[0]["trajectory"]["errors"]:
            rows = [run["trajectory"]["errors"][phase] for run in runs]
            count = sum(row["valid_samples"] for row in rows)
            aggregate[phase] = {"valid_samples": count, **{
                metric: float(np.sqrt(sum(row["valid_samples"] * row[metric] ** 2 for row in rows) / count))
                for metric in ("position_rmse_m", "yaw_rmse_rad")}}
        stationary = {}
        for condition in runs[0]["stationary_tests"]:
            rows = [run["stationary_tests"][condition] for run in runs]
            stationary[condition] = {
                source: {metric: float(np.sqrt(np.mean([row[source][metric] ** 2 for row in rows])))
                         for metric in ("position_rmse_m", "yaw_rmse_rad")}
                for source in ("raw", "filtered")}
            stationary[condition]["false_linear_compensation_rms_m_s"] = float(np.sqrt(np.mean([
                row["false_linear_compensation_rms_m_s"] ** 2 for row in rows])))
            stationary[condition]["false_yaw_compensation_rms_rad_s"] = float(np.sqrt(np.mean([
                row["false_yaw_compensation_rms_rad_s"] ** 2 for row in rows])))
            stationary[condition]["nonzero_compensation_fraction"] = float(np.mean([
                row["nonzero_compensation_fraction"] for row in rows]))
        results[label] = {"config": asdict(current), "trajectory_errors": aggregate,
                          "stationary_tests": stationary,
                          "rejected": sum(run["trajectory"]["rejected"] for run in runs), "runs": runs}
    return {"kind": "heldout_synthetic_known_truth_not_hardware_validation", "age_s": age,
            "exploratory_seed": 2718, "validation_seeds": list(seeds),
            "config_retuned_by_validation": False, "comparisons": results}


def heldout_validation(segments, config, *, age=.18):
    """Validation data never change R; fixed-pose segments assess spread only.

    Motion without an independent reference is explicitly unscored. This
    prevents calling the raw vision signal the ground truth of its own filter.
    """
    results = []
    for entry, series in segments:
        if entry["partition"] != "validation":
            continue
        filtered = replay(series.times, series.values, config, age=age)
        row = {"log": str(series.path), "condition": entry["condition"], "rejected": filtered["rejected"],
               "samples": len(series.times), "position_accuracy_validated": False}
        if entry["condition"] in STATIONARY:
            _, center = centered_residuals(series.values)
            reference = np.repeat(center[None], len(series.times), axis=0)
            row.update(metric="spread about held-out stationary segment median; not absolute error",
                       raw=error_metrics(series.values, reference), filtered=error_metrics(filtered["estimates"], reference))
        else:
            row.update(metric="unscored: no independent moving-base ground truth", raw=None, filtered=None)
        results.append(row)
    return results


def provisional_config(audits):
    """Protocol-proxy noise plus observed timing, explicitly unverified.

    PREVIEW/ALIGNMENT are not true fixed-base labels. Their spreads only set
    cautious provisional floors; ACTIVE spread is not mistaken for noise.
    """
    from urcal.cooperative.relative_kalman import PlanarKalmanConfig
    scales = [phase["raw_spread_m_m_rad"] for item in audits for phase in item.get("phases", [])
              if phase["phase"] in ("preview", "alignment") and "raw_spread_m_m_rad" in phase]
    ages = [item.get("timing", {}).get("control_age_median_p95_s") for item in audits]
    max_age = max([a[1] for a in ages if a is not None] or [.25])
    scale = np.max(scales, axis=0) if scales else [.005, .005, np.deg2rad(.3)]
    linear = max(.007, np.ceil(max(scale[:2]) * 1000.) / 1000.)
    angular = max(np.deg2rad(.3), np.deg2rad(np.ceil(np.rad2deg(scale[2]) * 10.) / 10.))
    horizon = min(.45, max(.30, np.ceil((max_age + .05) / .05) * .05))
    config = PlanarKalmanConfig(measurement_std_m=float(linear), measurement_std_yaw=float(angular), prediction_horizon=float(horizon))
    return config, {"status": "provisional_unlabelled_protocol_proxy", "labels_available": False,
                    "assumption": "PREVIEW/ALIGNMENT bases were fixed per operating protocol; not independently verified",
                    "measurement_noise_validated": False, "process_noise_fitted": False,
                    "motion_tracking_improvement_validated": False,
                    "R_basis": "rounded-up marginal spreads in preview/alignment, minimum 7mm/0.3deg",
                    "horizon_basis": "observed control capture-age p95 plus at least 50ms, rounded to 50ms; capped .45s",
                    "Q_basis": "explicit constant-acceleration prior; synthetic comparisons only",
                    "logs": audits, "config": asdict(config)}


def replay_diagnostics(series, config, *, age=.18):
    result = replay(series.times, series.values, config, age=age)
    finite = np.isfinite(result["estimates"]).all(axis=1)
    phases = []
    for phase in dict.fromkeys(series.phases):
        mask = np.array(series.phases) == phase
        phases.append({"phase": phase, "samples": int(mask.sum()),
                       "finite_estimate_fraction": float(finite[mask].mean()),
                       "mean_lambda": float(result["confidence"][mask].mean()),
                       "mean_prediction_gain": result["prediction_gain"][mask].mean(axis=0).tolist(),
                       "linear_compensation_rms_m_s": float(np.sqrt(np.nanmean(np.sum(result["velocities"][mask, :3] ** 2, axis=1)))),
                       "nonzero_compensation_fraction": float((np.linalg.norm(result["velocities"][mask, :3], axis=1) > 1e-5).mean())})
    return {"log": str(series.path), "samples": len(series.times), "age_s": age,
            "filter_rejections": result["rejected"], "phases": phases,
            "interpretation": "diagnostic replay only; no ground truth, robot dynamics or upstream gate replay"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("audit", help="report unlabelled capture and arm-motion evidence without fitting R")
    inspect.add_argument("logs", type=Path, nargs="+")
    inspect.add_argument("--leader", choices=ROBOTS, default="ur10e")
    inspect.add_argument("--output", type=Path)
    fit = sub.add_parser("fit", help="fit only labelled stationary-base training intervals")
    fit.add_argument("--segments", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--report", type=Path, required=True)
    provisional = sub.add_parser("provisional", help="unverified protocol-proxy floors and timing, clearly marked provisional")
    provisional.add_argument("logs", type=Path, nargs="+")
    provisional.add_argument("--output", type=Path, required=True)
    provisional.add_argument("--report", type=Path, required=True)
    synth = sub.add_parser("synthetic", help="compare production filter with known synthetic stop/reversal truth")
    synth.add_argument("--config", type=Path)
    synth.add_argument("--age", type=float, default=.18)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--seed", type=int, default=2718)
    validate = sub.add_parser("validate-synthetic", help="validate fixed priors with independent seeds, without fitting")
    validate.add_argument("--config", type=Path)
    validate.add_argument("--age", type=float, default=.18)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--seeds", type=int, nargs="+", default=[31415, 57721, 16180])
    diagnostic = sub.add_parser("replay", help="diagnose production filter outputs without claiming ground-truth accuracy")
    diagnostic.add_argument("logs", type=Path, nargs="+")
    diagnostic.add_argument("--config", type=Path)
    diagnostic.add_argument("--age", type=float, default=.18)
    diagnostic.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    try:
        if options.command == "audit":
            result = {"schema": "relative-kalman-audit-v1", "logs": [audit(read_series(p, leader=options.leader)) for p in options.logs]}
            if options.output:
                if options.output.resolve() in [p.resolve() for p in options.logs]:
                    raise ValueError("output must not overwrite a source log")
                save_json(options.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif options.command == "fit":
            from urcal.cooperative.relative_kalman import PlanarKalmanConfig
            segments = load_segments(options.segments)
            sources = {series.path.resolve() for _, series in segments} | {options.segments.resolve()}
            if options.output.resolve() in sources or options.report.resolve() in sources or options.output.resolve() == options.report.resolve():
                raise ValueError("config/report must be distinct from each other and source logs/manifest")
            noise, report = fit_measurement_noise(segments)
            current = PlanarKalmanConfig(**noise)
            config = asdict(current)
            report.update(config=config, process_noise_fitted=False,
                          validation=heldout_validation(segments, current),
                          warning_process="Q/horizon/gates remain explicit priors; no motion truth was supplied.")
            save_json(options.output, config)
            save_json(options.report, report)
            print(f"Saved labelled-noise config {options.output} and report {options.report}")
        elif options.command == "provisional":
            sources = {p.resolve() for p in options.logs}
            if options.output.resolve() in sources or options.report.resolve() in sources or options.output.resolve() == options.report.resolve():
                raise ValueError("config/report must be distinct from source logs and each other")
            config, report = provisional_config([audit(read_series(p)) for p in options.logs])
            save_json(options.output, asdict(config))
            save_json(options.report, report)
            print(f"Saved PROVISIONAL unverified config {options.output}; evidence/assumptions: {options.report}")
        elif options.command in ("synthetic", "validate-synthetic"):
            from urcal.cooperative.relative_kalman import load_planar_kalman_config
            if options.config is not None and options.output.resolve() == options.config.resolve():
                raise ValueError("synthetic report must not overwrite the source config")
            current = load_planar_kalman_config(options.config)
            result = (synthetic_comparison(current, age=options.age, seed=options.seed)
                      if options.command == "synthetic" else
                      synthetic_validation(current, age=options.age, seeds=options.seeds))
            save_json(options.output, result)
            print(f"Saved synthetic-only comparison {options.output}")
        else:
            from urcal.cooperative.relative_kalman import load_planar_kalman_config
            sources = {p.resolve() for p in options.logs}
            if options.config is not None:
                sources.add(options.config.resolve())
            if options.output.resolve() in sources:
                raise ValueError("replay report must not overwrite source logs/config")
            config = load_planar_kalman_config(options.config)
            result = {"kind": "unlabelled_diagnostic_replay_not_accuracy", "config": asdict(config),
                      "logs": [replay_diagnostics(read_series(p), config, age=options.age) for p in options.logs]}
            save_json(options.output, result)
            print(f"Saved diagnostic replay {options.output}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
