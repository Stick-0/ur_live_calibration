"""Reproduce evaluation CSVs exclusively from the immutable source JSONL.

Usage: python -m urcal.tools.export_evaluation RAW --session-id ID --output CSV
Partial sessions and malformed lines are exported explicitly as incomplete;
CLI exit status 2 signals an incomplete result. Existing outputs are untouched.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


BASE_FIELDS = (
    "kind", "schema", "session_id", "trial_id", "request_id", "source_path", "source_line",
    "source_line_sha256", "export_incomplete", "at", "elapsed_s", "clock", "episode", "cycle", "phase",
    "leader", "follower", "reason", "metric_reason", "metric_valid", "robot_state_valid", "calibration_valid",
    "metric_source", "metric_at", "control_active", "preview", "simulated", "table_moving",
    "command_attempted", "command_sent", "transport_failed", "simulated_command_applied",
    "kalman_state_estimation_enabled", "prediction_enabled", "table_compensation_enabled",
    "position_error_norm_m", "position_error_norm_mm", "rotation_error_norm_rad", "rotation_error_norm_deg",
    "state_at", "control_at", "scene_at", "measurement_sequence", "measurement_captured_at", "measurement_received_at",
    "final_sent_method", "final_sent_at", "final_acceleration", "final_duration_s",
    "simulated_applied_method", "simulated_integrator_method",
)
VECTOR_FIELDS = ("position_error_m", "rotation_error_rad", "proposed_velocity", "proposed_joint_velocity",
                 "final_sent_velocity", "base_compensation", "leader_compensation", "feedback",
                 "task_twist_nominal", "task_twist_at_command", "requested_twist", "tracking_requested",
                 "requested_compensation", "applied_compensation", "base_compensation_nominal",
                 "leader_compensation_nominal", "simulated_applied_velocity", "simulated_applied_joint_velocity")
MATRIX_FIELDS = ("frozen_target", "actual_relative_pose", "G_control")
JSON_FIELDS = ("measurement_times", "received_times", "measurement_age", "prediction_dt", "relative_kalman",
               "relative_base_compensation", "flags", "proposed_command", "sent_commands", "command_attempts",
               "control_record", "transport", "stop_transport", "measurement", "scene", "source_links", "simulated_applied_commands", "raw_record")
FIELDS = (BASE_FIELDS
          + tuple(f"{key}_{index}" for key in VECTOR_FIELDS for index in range(3 if key in VECTOR_FIELDS[:2] else 6))
          + tuple(f"{key}_r{row}c{column}" for key in MATRIX_FIELDS for row in range(4) for column in range(4))
          + tuple(f"{robot}_{key}" for robot in ("ur5e", "ur10e")
                  for key in ("robot_timestamp", "host_sample_at", "measurement_at", "received_at"))
          + tuple(f"{robot}_{key}_{index}" for robot in ("ur5e", "ur10e")
                  for key in ("q", "qd", "tcp_speed") for index in range(6))
          + tuple(key + "_json" for key in JSON_FIELDS))


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _invalid_constant(value):
    raise ValueError("nonfinite JSON number " + value)


def _read_snapshot(path):
    """Snapshot a byte prefix; later raw appends never change this export."""
    rows, errors, sessions = [], [], set()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        size = source.seek(0, 2)
        source.seek(0)
        remaining, number = size, 0
        while remaining:
            raw = source.readline(remaining)
            if not raw:
                break
            remaining -= len(raw)
            digest.update(raw)
            number += 1
            text = raw.decode("utf-8", errors="replace")
            try:
                record = json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record is not an object")
            except (ValueError, TypeError) as exc:
                errors.append({"line": number, "error": str(exc), "raw_line": text})
                rows.append((number, {"kind": "evaluation_parse_error", "reason": str(exc),
                                      "raw_line": text}, hashlib.sha256(raw).hexdigest()))
                continue
            if not raw.endswith(b"\n"):
                errors.append({"line": number, "error": "final JSONL record has no newline (partial write possible)",
                               "raw_line": text})
            if str(record.get("kind", "")).startswith("evaluation_"):
                rows.append((number, record, hashlib.sha256(raw).hexdigest()))
                if record.get("session_id"):
                    sessions.add(record["session_id"])
    return rows, errors, sorted(sessions), size, digest.hexdigest()


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _flatten(record, start, *, source_path, line, digest, incomplete):
    row = {key: "" for key in FIELDS}
    for key in BASE_FIELDS:
        value = record.get(key)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (str, int, float)):
            row[key] = value
    row.update(source_path=str(source_path), source_line=line, source_line_sha256=digest,
               export_incomplete=int(incomplete), session_id=start.get("session_id", record.get("session_id", "")),
               trial_id=record.get("trial_id", start.get("trial_id", "")))
    if isinstance(record.get("at"), (int, float)) and isinstance(start.get("at"), (int, float)):
        row["elapsed_s"] = record["at"] - start["at"]
    flags = _mapping(record.get("flags", start.get("flags", {})))
    for key in ("kalman_state_estimation_enabled", "prediction_enabled", "table_compensation_enabled"):
        if key in flags:
            row[key] = int(flags[key])
    control = _mapping(record.get("control_record"))
    scene = _mapping(record.get("scene"))
    measurement = _mapping(record.get("measurement"))
    proposed = _mapping(record.get("proposed_command"))
    follower = record.get("follower", start.get("follower", "ur5e"))
    sent = _mapping(_mapping(record.get("sent_commands")).get(follower))
    command = _mapping(sent.get("command"))
    simulated_command = _mapping(_mapping(record.get("simulated_applied_commands")).get(follower))
    row.update(control_at=control.get("at", ""), scene_at=scene.get("at", ""),
               measurement_sequence=measurement.get("sequence", ""),
               measurement_captured_at=measurement.get("captured_at", ""),
               measurement_received_at=measurement.get("received_at", ""),
               final_sent_method=sent.get("stage", ""), final_sent_at=sent.get("finished_at", ""),
               final_acceleration=command.get("acceleration", ""), final_duration_s=command.get("duration_s", ""))
    row.update(simulated_applied_method=simulated_command.get("method", ""),
               simulated_integrator_method=simulated_command.get("applied_method", ""))
    vectors = {key: record.get(key, control.get(key)) for key in VECTOR_FIELDS}
    vectors.update(proposed_velocity=proposed.get("velocity"), proposed_joint_velocity=proposed.get("joint_velocity"),
                   final_sent_velocity=command.get("velocity"),
                   simulated_applied_velocity=simulated_command.get("command"),
                   simulated_applied_joint_velocity=simulated_command.get("joint_velocity"))
    for key, vector in vectors.items():
        if isinstance(vector, list):
            for index, value in enumerate(vector):
                if f"{key}_{index}" in row and isinstance(value, (int, float)):
                    row[f"{key}_{index}"] = value
    for key in MATRIX_FIELDS:
        matrix = record.get(key, start.get(key) if key == "frozen_target" else None)
        if isinstance(matrix, list):
            for i, values in enumerate(matrix[:4]):
                if isinstance(values, list):
                    for j, value in enumerate(values[:4]):
                        if isinstance(value, (int, float)):
                            row[f"{key}_r{i}c{j}"] = value
    for robot in ("ur5e", "ur10e"):
        state = {**_mapping(_mapping(scene.get("robots")).get(robot)),
                 **_mapping(_mapping(control.get("robots")).get(robot))}
        row[f"{robot}_robot_timestamp"] = state.get("robot_timestamp", "")
        row[f"{robot}_host_sample_at"] = _mapping(control.get("robot_state_host_times")).get(robot,
            next((state[key] for key in ("at", "host_sample_at", "sample_at", "sampled_at")
                  if state.get(key) is not None), ""))
        row[f"{robot}_measurement_at"] = _mapping(record.get("measurement_times")).get(robot, "")
        row[f"{robot}_received_at"] = _mapping(record.get("received_times")).get(robot, "")
        for key in ("q", "qd", "tcp_speed"):
            values = state.get(key, [])
            for index, value in enumerate(values[:6] if isinstance(values, list) else []):
                if isinstance(value, (int, float)):
                    row[f"{robot}_{key}_{index}"] = value
    if isinstance(record.get("position_error_norm_m"), (int, float)):
        row["position_error_norm_mm"] = record["position_error_norm_m"] * 1000.
    if isinstance(record.get("rotation_error_norm_rad"), (int, float)):
        row["rotation_error_norm_deg"] = record["rotation_error_norm_rad"] * 180. / 3.141592653589793
    for key in JSON_FIELDS:
        value = record if key == "raw_record" else record.get(key, control.get(key))
        if value is not None:
            row[key + "_json"] = _json(value)
    return row


def export_session(raw_path, output, *, session_id=None, wait_for_stop=0.):
    """Export every session event/cycle plus explicit file parse failures.

    Returns metadata including incomplete/issues. Partial data is useful and
    preserved, never relabelled complete. Parsing failures anywhere in the
    byte-prefix are included conservatively because their session is unknown.
    """
    raw_path, output = Path(raw_path), Path(output)
    if not isinstance(wait_for_stop, (int, float)) or not math.isfinite(wait_for_stop) or wait_for_stop < 0:
        raise ValueError("wait_for_stop must be nonnegative")
    deadline = time.monotonic() + wait_for_stop
    while True:
        rows, errors, sessions, size, source_hash = _read_snapshot(raw_path)
        chosen = session_id
        if chosen is None and len(sessions) == 1:
            chosen = sessions[0]
        stopped = any(record.get("session_id") == chosen and record.get("kind") == "evaluation_stop"
                      for _, record, _ in rows)
        if stopped or time.monotonic() >= deadline:
            break
        time.sleep(min(.025, max(0., deadline - time.monotonic())))
    if chosen is None:
        raise ValueError("specify --session-id; available sessions: " + ", ".join(sessions))
    selected = [(line, record, digest) for line, record, digest in rows
                if record.get("session_id") == chosen or record.get("kind") == "evaluation_parse_error"]
    own = [record for _, record, _ in selected if record.get("session_id") == chosen]
    if not own:
        raise ValueError("evaluation session not found: " + str(chosen))
    starts = [record for record in own if record.get("kind") == "evaluation_start"]
    stops = [record for record in own if record.get("kind") == "evaluation_stop"]
    start = starts[0] if starts else {"session_id": chosen}
    cycles = [record for record in own if record.get("kind") == "evaluation_cycle"]
    issues = []
    if len(starts) != 1:
        issues.append(f"expected one session start; found {len(starts)}")
    if len(stops) != 1:
        issues.append(f"expected one session stop; found {len(stops)}")
    if errors:
        issues.append(f"source contains {len(errors)} parse/truncation error(s); see parse_errors")
    if stops and stops[-1].get("cycles") != len(cycles):
        issues.append("stop cycle count does not match exported producer cycles")
    if [record.get("cycle") for record in cycles] != list(range(len(cycles))):
        issues.append("producer cycle sequence is missing, duplicate or out of order")
    if any(record.get("incomplete") for record in own):
        issues.append("raw session was explicitly marked incomplete")
    if starts and own[0].get("kind") != "evaluation_start":
        issues.append("session contains records preceding its start")
    if stops and own[-1].get("kind") != "evaluation_stop":
        issues.append("session contains records after its stop")
    metadata_path = output.with_suffix(".metadata.json")
    if output.resolve() == raw_path.resolve() or metadata_path.resolve() == raw_path.resolve():
        raise ValueError("export output must not replace the raw source")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("evaluation output already exists: " + str(output))
    metadata = {"schema": "urcal-evaluation-export-v1", "session_id": chosen,
                "trial_id": start.get("trial_id"), "raw_path": str(raw_path), "csv_path": str(output),
                "metadata_path": str(metadata_path), "source_bytes": size, "source_sha256": source_hash,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "incomplete": bool(issues),
                "issues": issues, "rows": len(selected), "cycles": len(cycles), "parse_errors": errors,
                "source_line_numbers": [line for line, _, _ in selected], "start": start, "stops": stops,
                "metric_definition": {"target": "session-start nominal relative flange target (immutable)",
                    "frame": "leader_flange", "position": "target.position - actual.position, metres",
                    "rotation": "rotvec(R_target @ R_actual.T), radians",
                    "command_sent": "this transaction SDK speedL/speedJ returned True; not physical execution confirmation"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for line, record, digest in selected:
            writer.writerow(_flatten(record, start, source_path=raw_path, line=line,
                                     digest=digest, incomplete=bool(issues)))
    with metadata_path.open("x", encoding="utf-8") as destination:
        json.dump(metadata, destination, ensure_ascii=False, allow_nan=False, indent=2)
        destination.write("\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export_session(args.raw, args.output, session_id=args.session_id)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, "Evaluation export failed: " + str(exc) + "\n")
    print(json.dumps({key: result[key] for key in ("session_id", "csv_path", "metadata_path", "cycles", "incomplete", "issues")},
                     ensure_ascii=False))
    return 2 if result["incomplete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
