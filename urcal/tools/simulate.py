"""Deterministic OFFLINE SYNTHETIC comparison of cooperative control methods.

This is a Cartesian six-coordinate, perfect velocity-integrator plant with an
empty workspace. It is NOT a UR dynamics simulator, collision validation,
camera/network benchmark, or experimental accuracy measurement. The production
CooperativeController.prepare/commit, relative-state processing and task governor
are exercised without constructing robot drivers or cameras. Only the method
changes between direct, predict and separated; sensor samples, plant, gains,
limits and experiment horizon are shared.

Every method writes its own plant-truth reference because different commands
produce different trajectories. This reference is independent of the estimator
but is SYNTHETIC, not a real-world measurement. Timestamps use the same declared
synthetic monotonic origin. Compare completion time, progress coverage and
active time together with errors; a slower or stopped method has not completed
the same work simply because its tracking error is smaller.

``finished`` means estimated endpoint attainment within the configured position
and rotation tolerances for a dwell interval of committed control. Path progress
reaching one alone is not completion. Independent synthetic-reference errors
remain separate and do not turn this criterion into real-hardware ground truth.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.tools.analyze_log import summarize_log
from urcal.cooperative.controller import CooperativeController, advance_origin_pose
from urcal.cooperative.logging import ExperimentLogger
from urcal.control.geometry import inverse, matrix_pose, model_from_controller_base, pose_matrix
from urcal.control.policy import CalibrationFrame, ControlFault


METHODS = ("direct", "predict", "separated")
ROBOTS = ("ur5e", "ur10e")
SCENARIOS = ("static", "bases", "leader", "follower", "rotation", "noise", "dropout", "infeasible")
CLOCK_ORIGIN = 1000.
SENSOR_PERIOD = .1
SENSOR_DELAY = .08
SIMULATION_NOTICE = (
    "SYNTHETIC CARTESIAN SIMULATION ONLY: perfect velocity integration, empty workspace; "
    "no UR dynamics, mesh collision, camera inference or real-world accuracy validation."
)


def _hat(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class CartesianModel:
    """Six synthetic coordinates [XYZ metres, rotation-vector radians].

    Orientation is Exp(q[3:]) times a fixed initial orientation. The angular
    Jacobian is the SO(3) left Jacobian, so qdot and base-frame geometric twist
    stay consistent. These coordinates are not UR revolute joints.
    """

    def __init__(self, initial_orientation):
        self.initial_orientation = np.asarray(initial_orientation, dtype=float).copy()

    def coordinates(self, pose):
        matrix_pose(pose)
        return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3] @ self.initial_orientation.T).as_rotvec()]

    def pose_and_jacobian(self, q, flange=False):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("synthetic coordinates must be six finite values")
        pose = pose_matrix(q)
        pose[:3, :3] = pose[:3, :3] @ self.initial_orientation
        theta = np.linalg.norm(q[3:])
        cross = _hat(q[3:])
        if theta < 1e-5:
            a, b = .5 - theta**2 / 24., 1. / 6. - theta**2 / 120.
        else:
            a, b = (1. - np.cos(theta)) / theta**2, (theta - np.sin(theta)) / theta**3
        jacobian = np.eye(6)
        jacobian[3:, 3:] += a * cross + b * cross @ cross
        return pose, jacobian

    def inverse(self, goal, seed, lower, upper):
        q = self.coordinates(goal)
        return q if np.all(q >= lower) and np.all(q <= upper) else None

    def points(self, q):
        pose, _ = self.pose_and_jacobian(q)
        return np.array([[0., 0., 0.], pose[:3, 3]])


class EmptySyntheticWorkspace:
    """No geometry: deliberately cannot validate UR links or collision margins."""

    def __init__(self):
        self.lower = {name: np.full(6, -10.) for name in ROBOTS}
        self.upper = {name: np.full(6, 10.) for name in ROBOTS}

    def pose_allowed(self, robot, pose):
        matrix_pose(pose)
        return bool(np.linalg.norm(pose[:3, 3]) < 10.)

    def check(self, bases, joints):
        for name in ROBOTS:
            inverse(bases[name])
            q = np.asarray(joints[name], dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                return False, "invalid synthetic coordinates", None
            if np.any(q < self.lower[name]) or np.any(q > self.upper[name]):
                return False, "synthetic coordinate bounds", None
        return True, "empty synthetic workspace (no collision validation)", None


def simulation_args(method, path="line", path_duration=8.):
    """One shared controller configuration; only method changes across trials."""
    return SimpleNamespace(
        method=method, leader="ur10e", ur5e_tcp_offset=[0.]*6, ur10e_tcp_offset=[0.]*6,
        workspace_config=None, max_calibration_age=.5, stable_frames=3,
        max_calibration_jump_m=.15, max_calibration_jump_deg=20., max_reproj_px=5.,
        max_point_reproj_px=8., check_reprojection=False, min_points=5, filter_alpha=.45, velocity_alpha=.4,
        prediction_horizon=.3, max_base_speed=2., max_base_angular_speed=1.,
        max_speed=.035, max_angular_speed=.25, max_accel=.25, max_angular_accel=1.,
        max_joint_speed=.5, max_joint_accel=2., min_singular_value=.05,
        gap=.30, min_gap=.15, path=path, path_delta=[.05, 0, 0, 0, 0, .08],
        path_radius=.015, path_duration=path_duration, frame_check_m=.001,
        frame_check_deg=.1, max_invalid_duration=.3, max_target_m=.2,
        max_target_deg=45., max_travel=.75, kp_linear=2., kp_angular=2.,
        feedback_speed=.03, feedback_angular_speed=.2, position_deadband=.0002,
        rotation_deadband_deg=.03, slow_error_m=.025, slow_error_deg=5.,
        finish_position_m=.005, finish_rotation_deg=2., finish_dwell=.5,
    )


def truth_bases(t, scenario):
    """Analytic externally moved bases; stationary before experiment time zero."""
    t = max(0., float(t))
    bases = {"ur10e": pose_matrix([-.35, 0, 1., 0, 0, 0]),
             "ur5e": pose_matrix([.35, 0, 1., 0, 0, 0])}
    wave = 1. - np.cos(.6 * t)
    if scenario in ("bases", "leader", "dropout"):
        bases["ur10e"][0, 3] += .03 * wave
    if scenario in ("bases", "follower", "dropout"):
        bases["ur5e"][0, 3] -= .02 * wave
    if scenario == "rotation":
        bases["ur10e"][:3, :3] = Rotation.from_rotvec([0, .06 * wave, .03 * wave]).as_matrix()
        bases["ur5e"][:3, :3] = Rotation.from_rotvec([0, -.04 * wave, -.02 * wave]).as_matrix()
    if scenario == "infeasible":
        bases["ur5e"][0, 3] += .25 * max(0., t - .5)**2
    return bases


def sensor_stream(duration, seed, scenario):
    """Same delayed, paired pose samples for all methods; no plant-state input."""
    rng = np.random.default_rng(seed)
    samples = []
    # Warm up while the plant/bases are stationary before the active trial.
    capture_times = np.arange(-.6, duration + SENSOR_PERIOD * .5, SENSOR_PERIOD)
    sigma_m, sigma_rad = ((.0005, .0004) if scenario == "noise" else (0., 0.))
    for sequence, captured in enumerate(capture_times):
        if scenario == "dropout" and .8 <= captured < 1.6:
            continue
        poses = {}
        for name, base in truth_bases(captured, scenario).items():
            noise = np.r_[rng.normal(0., sigma_m, 3), rng.normal(0., sigma_rad, 3)]
            measured = pose_matrix(noise) @ base
            poses[name] = measured @ inverse(model_from_controller_base(name))
        packet = CalibrationFrame(sequence, CLOCK_ORIGIN + float(captured), poses,
                                  {name: 1. for name in ROBOTS}, {name: 2. for name in ROBOTS},
                                  {name: 7 for name in ROBOTS})
        samples.append((CLOCK_ORIGIN + float(captured) + SENSOR_DELAY, packet))
    return samples


def _json_line(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def _run_method(output_dir, method, duration, dt, path, path_duration, scenario, seed, samples):
    args = simulation_args(method, path, path_duration)
    bases0 = truth_bases(0., scenario)
    leader_pose = pose_matrix([0, 0, .4, 0, 0, 0])
    relative0 = pose_matrix([0, 0, args.gap, np.pi, 0, 0])
    follower_pose = inverse(bases0["ur5e"]) @ bases0["ur10e"] @ leader_pose @ relative0
    poses = {"ur10e": leader_pose, "ur5e": follower_pose}
    models = {name: CartesianModel(pose[:3, :3]) for name, pose in poses.items()}
    engine = CooperativeController(args, models=models, guard=EmptySyntheticWorkspace())
    speed = {name: np.zeros(6) for name in ROBOTS}
    log_path, reference_path = output_dir / f"{method}.jsonl", output_dir / f"{method}.reference.jsonl"
    active, stopped, sample_index = False, False, 0
    times = np.arange(0., duration + dt * .1, dt)
    if len(times) == 0 or times[-1] < duration - 1e-10:
        times = np.r_[times, duration]
    with ExperimentLogger(log_path, max_pending=65536) as logger, reference_path.open("x", encoding="utf-8") as reference:
        logger.log({"kind": "metadata", "at": CLOCK_ORIGIN, "simulation": True,
                    "notice": SIMULATION_NOTICE, "synthetic_clock_origin": CLOCK_ORIGIN,
                    "scenario": scenario, "seed": seed, "duration_s": duration, "dt_s": dt,
                    "sensor_period_s": SENSOR_PERIOD, "sensor_delay_s": SENSOR_DELAY,
                    "method": method, "controller_arguments": vars(args),
                    "reference_file": reference_path.name,
                    "completion_definition": "finished is estimated endpoint attainment after committed dwell; independent reference errors are separate, not real-hardware truth"})
        for index, elapsed in enumerate(times):
            now = CLOCK_ORIGIN + float(elapsed)
            step = float(times[index + 1] - elapsed) if index + 1 < len(times) else dt
            while sample_index < len(samples) and samples[sample_index][0] <= now + 1e-10:
                available, packet = samples[sample_index]
                sample_index += 1
                # Warm-up samples retain their individual receipt times, not t=0.
                received = available if available < CLOCK_ORIGIN else now
                accepted = engine.ingest(packet, received) if not stopped else False
                logger.log({"kind": "measurement", "at": received,
                            "captured_at": packet.captured_at, "received_at": received,
                            "sequence": packet.sequence, "poses": packet.poses,
                            "raw_mean_px": packet.raw_mean_px, "raw_max_px": packet.raw_max_px,
                            "valid_counts": packet.valid_counts, "accepted": accepted,
                            "synthetic": True})
            states = {}
            for name in ROBOTS:
                q = models[name].coordinates(poses[name])
                _, jacobian = models[name].pose_and_jacobian(q)
                states[name] = SimpleNamespace(tcp=poses[name].copy(), joints=q, speed=speed[name].copy(),
                                              joint_velocity=np.linalg.solve(jacobian, speed[name]), robot_timestamp=now)
            truth = truth_bases(elapsed, scenario)
            actual_relative = inverse(truth["ur10e"] @ poses["ur10e"]) @ truth["ur5e"] @ poses["ur5e"]
            _json_line(reference, {"at": now, "relative_pose": actual_relative.tolist(), "synthetic": True})
            if not stopped:
                try:
                    # A valid preview proposal is required before activation.
                    if not active:
                        engine.prepare(now, states, step, active=False)
                        active = True
                    proposal = engine.prepare(now, states, step, active=True)
                    proposal.record["simulation"] = True
                    logger.log(proposal.record)
                    if index + 1 < len(times):
                        command = proposal.twists["ur5e"]
                        poses["ur5e"] = advance_origin_pose(poses["ur5e"], command, step)
                        speed["ur5e"] = command.copy()
                        engine.commit(proposal)
                except ControlFault as exc:
                    stopped, active = True, False
                    speed["ur5e"][:] = 0.
                    logger.log({"kind": "fault", "at": now,
                                "category": getattr(exc, "category", "control_fault"), "message": str(exc),
                                "progress": engine.progress, "simulation": True})
            if stopped:
                logger.log({"kind": "control", "at": now, "active": False, "simulation": True,
                            "progress": engine.progress, "task_scale": 0., "finished": False,
                            "method": method, "stopped": True})
        logger.log({"kind": "stop", "at": CLOCK_ORIGIN + duration,
                    "reason": "synthetic experiment horizon reached", "simulation": True})
    report = summarize_log(log_path, reference_path, reference_tolerance=max(1e-9, dt * .01))
    report["simulation"] = True
    report["reference_provenance"] = "independent synthetic plant truth; not real-world ground truth"
    report["requested_experiment_duration_s"] = duration
    return report


def run_comparison(output_dir, duration=20., seed=7, scenario="bases", dt=.04,
                   path="line", path_duration=8.):
    """Write reproducible method/reference JSONL artifacts and comparison.json."""
    for label, value in (("duration", duration), ("dt", dt), ("path_duration", path_duration)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if scenario not in SCENARIOS or path not in ("hold", "line", "circle"):
        raise ValueError("unknown synthetic scenario or path")
    if dt > SENSOR_PERIOD:
        raise ValueError("dt must be no greater than the synthetic sensor period (0.1 s)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{method}{suffix}" for method in METHODS for suffix in (".jsonl", ".reference.jsonl")]
    expected.append(output_dir / "comparison.json")
    if any(path.exists() for path in expected):
        raise FileExistsError("comparison output already exists; choose a fresh output directory")
    samples = sensor_stream(duration, seed, scenario)
    reports = {method: _run_method(output_dir, method, duration, dt, path, path_duration, scenario, seed, samples)
               for method in METHODS}
    result = {"simulation": True, "notice": SIMULATION_NOTICE, "scenario": scenario, "seed": seed,
              "duration_s": duration, "dt_s": dt, "path": path, "path_duration_s": path_duration,
              "completion_definition": "estimated endpoint attainment after committed dwell, not path-phase completion; independent synthetic truth is evaluated separately",
              "comparison_note": "Compare errors with completion time, active time and progress coverage; stopped/slow trials may perform less work.",
              "methods": reports}
    with (output_dir / "comparison.json").open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=SCENARIOS, default="bases")
    parser.add_argument("--dt", type=float, default=.04)
    parser.add_argument("--path", choices=("hold", "line", "circle"), default="line")
    parser.add_argument("--path_duration", "--path-duration", type=float, default=8.)
    args = parser.parse_args(argv)
    try:
        report = run_comparison(args.output_dir, args.duration, args.seed, args.scenario,
                                args.dt, args.path, args.path_duration)
    except (OSError, ValueError, ControlFault) as exc:
        parser.exit(2, f"Synthetic comparison failed: {exc}\n")
    print(SIMULATION_NOTICE)
    for method, result in report["methods"].items():
        independent = result["independent_real_error"]["position_m"]["rms"]
        print(f"{method}: synthetic reference position RMS={independent}, "
              f"progress={result['progress']['last']}, completion_s={result['completion_time_s']}, "
              f"faults={len(result['faults'])}")
    print(f"Report: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
