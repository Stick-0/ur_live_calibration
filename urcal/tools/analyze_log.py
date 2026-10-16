"""Summarize cooperative JSONL experiments without treating estimates as truth.

Control records use kind="control", at=<host monotonic seconds>, active=<bool>,
progress=<0..1>, task_scale=<0..1>, desired_relative_pose=<4x4 or UR pose6>,
and estimated_relative_pose=<4x4 or UR pose6>. Both transforms map follower
flange coordinates into the leader flange frame. path_finished marks completion
of the commanded path phase. Optional finished=true (or task_finished=true)
marks estimated endpoint attainment after the controller's committed dwell,
and requires progress=1. Hold/frozen-phase compensation is not completion.
Independent reference is evaluated separately; it does not determine finished.

An independent reference file must be supplied by the experimenter. Each line
has {"at": <seconds>, "relative_pose": <4x4 or UR pose6>}, using the SAME
monotonic time origin, units and flange convention as the control log. Align
external clocks and transform the independently measured tool frames BEFORE
analysis. Nearest samples within --reference-tolerance are compared to the
logged desired relation. The analyzer does not invent ground truth, estimate a
clock offset, or assume the pose estimator itself is an independent reference.

Error statistics cover ALL active control records, including maintenance after
completion. Interval times assume the left sample's values hold until the next
control sample. Time after the final sample is unknown and excluded. Slowdown
and mean task scale cover only unfinished non-hold path intervals; active holds
and post-phase maintenance have separate durations. Path-phase and estimated
endpoint completion times are both wall time from the first active sample.
For older records without path_finished, progress>=1 implies phase completion;
an absent path name is treated as non-hold. An explicit finished flag is still
required for endpoint completion, but older logs may not specify its criterion.
"""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path

import numpy as np

from urcal.control.geometry import matrix_pose, pose_matrix, rotation_distance


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _pose(value, label):
    try:
        array = np.asarray(value, dtype=float)
        if array.shape == (6,):
            return pose_matrix(array)
        matrix_pose(array)  # Validate homogeneous shape and proper rotation.
        return array
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a rigid 4x4 transform or finite UR pose6") from exc


def _read_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON (possibly an incomplete log)") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            records.append(record)
    return records


def _statistics(values):
    if not values:
        return {"count": 0, "mean": None, "rms": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(array.mean()),
            "rms": float(np.sqrt(np.mean(array ** 2))), "max": float(array.max())}


def _errors(desired, observed):
    return (float(np.linalg.norm(desired[:3, 3] - observed[:3, 3])),
            float(np.rad2deg(rotation_distance(desired, observed))))


def _error_summary(positions, angles):
    return {"position_m": _statistics(positions), "angle_deg": _statistics(angles)}


def _phase_finished(record):
    return record.get("path") != "hold" and record.get("path_finished", record.get("progress", 0.) >= 1.)


def summarize_log(path, reference_path=None, reference_tolerance=.03):
    """Return JSON-serializable statistics; independent_real_error is optional."""
    tolerance = _number(reference_tolerance, "reference_tolerance")
    if tolerance < 0:
        raise ValueError("reference_tolerance must be nonnegative")
    if reference_path is not None and Path(path).resolve() == Path(reference_path).resolve():
        raise ValueError("independent reference must be a separate externally measured file")
    records = _read_records(path)
    controls = [record for record in records if record.get("kind") == "control"]
    for index, record in enumerate(controls):
        record["at"] = _number(record.get("at"), "control at")
        if index and record["at"] < controls[index - 1]["at"]:
            raise ValueError("control timestamps must not move backwards")
        if not isinstance(record.get("active"), bool):
            raise ValueError("control active must be boolean")
        for key in ("progress", "task_scale"):
            if key in record:
                record[key] = _number(record[key], key)
                if not 0 <= record[key] <= 1:
                    raise ValueError(f"{key} must be between zero and one")
        for key in ("finished", "task_finished", "path_finished"):
            if key in record and not isinstance(record[key], bool):
                raise ValueError(f"{key} must be boolean")
    active = [record for record in controls if record["active"]]
    progress = [record["progress"] for record in active if "progress" in record]
    completed = next((record for record in active
                      if (record.get("finished") or record.get("task_finished"))
                      and record.get("path") != "hold" and record.get("progress", 0.) >= 1.), None)
    phase_completed = next((record for record in active if _phase_finished(record)), None)
    active_time = 0.
    path_active_time = 0.
    maintenance_time = 0.
    completion_active_time = 0.
    scale_time = 0.
    slowdown_time = 0.
    integrated_scale = 0.
    for left, right in zip(controls, controls[1:]):
        if not left["active"]:
            continue
        duration = right["at"] - left["at"]
        active_time += duration
        if completed is not None:
            completion_active_time += max(0., min(right["at"], completed["at"]) - left["at"])
        maintenance = left.get("path") == "hold" or _phase_finished(left)
        if maintenance:
            maintenance_time += duration
        else:
            path_active_time += duration
        if not maintenance and "task_scale" in left:
            scale_time += duration
            integrated_scale += duration * left["task_scale"]
            if left["task_scale"] < 1. - 1e-9:
                slowdown_time += duration
    positions, angles = [], []
    desired_samples = []
    for record in active:
        if "desired_relative_pose" not in record:
            continue
        desired = _pose(record["desired_relative_pose"], "desired_relative_pose")
        desired_samples.append((record["at"], desired))
        if "estimated_relative_pose" in record:
            position, angle = _errors(desired, _pose(record["estimated_relative_pose"], "estimated_relative_pose"))
            positions.append(position)
            angles.append(angle)
    independent = None
    if reference_path is not None:
        references = []
        for record in _read_records(reference_path):
            references.append((_number(record.get("at"), "reference at"),
                               _pose(record.get("relative_pose"), "reference relative_pose")))
        references.sort(key=lambda pair: pair[0])
        times = [pair[0] for pair in references]
        if any(first == second for first, second in zip(times, times[1:])):
            raise ValueError("reference timestamps must be unique")
        positions_ref, angles_ref, skews = [], [], []
        for at, desired in desired_samples:
            insert = bisect_left(times, at)
            candidates = [index for index in (insert - 1, insert) if 0 <= index < len(times)]
            if not candidates:
                continue
            closest = min(candidates, key=lambda index: abs(times[index] - at))
            skew = abs(times[closest] - at)
            if skew > tolerance + 1e-12:
                continue
            position, angle = _errors(desired, references[closest][1])
            positions_ref.append(position)
            angles_ref.append(angle)
            skews.append(skew)
        independent = _error_summary(positions_ref, angles_ref)
        independent.update({"reference_file": str(reference_path), "reference_records": len(references),
                            "matched_control_records": len(skews),
                            "unmatched_control_records": len(desired_samples) - len(skews),
                            "tolerance_s": tolerance, "matching_skew_s": _statistics(skews)})
    faults = []
    for record in records:
        if record.get("kind") == "fault" or (record.get("kind") in ("stop", "stop_fault")
                                              and record.get("fault_category")):
            faults.append({"at": record.get("at"),
                           "category": record.get("fault_category", record.get("category", "unspecified")),
                           "message": record.get("message", record.get("reason", ""))})
    nominal_duration = next((record["path_duration"] for record in active if "path_duration" in record), None)
    completion_basis = None
    for record in records:
        if record.get("kind") != "metadata":
            continue
        completion_basis = record.get("completion_definition", completion_basis)
        for key in ("controller_arguments", "args", "arguments"):
            config = record.get(key, {})
            if nominal_duration is None and isinstance(config, dict):
                nominal_duration = config.get("path_duration")
    if nominal_duration is not None:
        nominal_duration = _number(nominal_duration, "nominal path duration")
        if nominal_duration <= 0:
            raise ValueError("nominal path duration must be positive")
    if completion_basis is None:
        completion_basis = (
            "estimated endpoint attainment after committed dwell; independent reference is evaluated separately"
            if any("settle_elapsed" in record and "within_finish_tolerance" in record for record in active)
            else "explicit logged finished flag at progress=1; endpoint settling criterion not recorded"
        )
    return {
        "log_file": str(path), "records": len(records),
        "record_counts": dict(Counter(str(record.get("kind", "unknown")) for record in records)),
        "control_records": len(controls), "active_control_records": len(active),
        "elapsed_control_time_s": controls[-1]["at"] - controls[0]["at"] if controls else 0.,
        "elapsed_active_time_s": active_time,
        "path_active_time_s": path_active_time, "maintenance_time_s": maintenance_time,
        "nominal_path_duration_s": nominal_duration,
        "phase_completed": phase_completed is not None,
        "phase_completion_time_s": phase_completed["at"] - active[0]["at"] if phase_completed is not None else None,
        "task_completed": completed is not None,
        "completion_basis": completion_basis,
        "completion_time_s": completed["at"] - active[0]["at"] if completed is not None else None,
        "active_time_until_completion_s": completion_active_time if completed is not None else None,
        "progress": {"first": progress[0] if progress else None,
                     "last": progress[-1] if progress else None,
                     "min": min(progress) if progress else None,
                     "max": max(progress) if progress else None,
                     "coverage": max(progress) - min(progress) if progress else None,
                     "backwards_steps": sum(b < a for a, b in zip(progress, progress[1:]))},
        "slowdown_ratio": slowdown_time / scale_time if scale_time else None,
        "mean_task_scale": integrated_scale / scale_time if scale_time else None,
        "task_scale_observed_time_s": scale_time,
        "estimated_relative_error": _error_summary(positions, angles),
        "independent_real_error": independent,
        "faults": faults,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--reference", type=Path, help="Independent flange relation JSONL, already time/frame aligned")
    parser.add_argument("--reference-tolerance", type=float, default=.03, help="Nearest-reference matching tolerance, seconds")
    parser.add_argument("--output", type=Path, help="Also save JSON to a new file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = summarize_log(args.log, args.reference, args.reference_tolerance)
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(encoded)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
