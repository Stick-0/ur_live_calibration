"""Moving-base relative collaboration from Research_Implementation_Plan.txt.

Uses the unchanged pose sensor and the same speedL low-level RTDE interface.
The default is receive-only preview; --execute plus operator Arm is required.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import time
import uuid

import numpy as np

from urcal.control.runtime import BUNDLE, ROBOTS, MotionSession, StateReader, parse_args, put_latest, take_latest
from urcal.cooperative.controller import CooperativeController, ResearchFault
from urcal.cooperative.logging import ExperimentLogger
from urcal.control.policy import ControlFault


def finite_record(value):
    """Preserve rejected measurement structure, marking nonfinite values null."""
    if isinstance(value, np.ndarray):
        return finite_record(value.tolist())
    if isinstance(value, dict):
        return {key: finite_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_record(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(args):
    from rtde_receive import RTDEReceiveInterface
    from urcal.control.gui import gui_worker
    from urcal.control.vision import vision_worker

    if args.mode != "follow":
        raise ValueError("research collaboration uses follow mode; use run_dual_control.sh for camera-fixed hold")
    log_path = args.log or BUNDLE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                            + "-" + args.method + "-" + uuid.uuid4().hex[:8] + ".jsonl")
    logger = ExperimentLogger(log_path)
    session = None
    ctx = mp.get_context("spawn")
    frames, commands, statuses, gui_statuses = (ctx.Queue(maxsize=size) for size in (1, 8, 1, 1))
    stop = ctx.Event()
    vision = ctx.Process(target=vision_worker, args=(args, frames, commands, statuses, stop), daemon=True)
    gui = ctx.Process(target=gui_worker, args=(args, commands, gui_statuses, stop), daemon=True)
    engine = None
    final_reason = "operator_stop"
    try:
        plan = BUNDLE / "docs" / "Research_Implementation_Plan.txt"
        logger.log({"kind": "metadata", "at": time.monotonic(), "schema": 1,
                    "source_plan": str(plan), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                    "settings": vars(args), "method": args.method,
                    "clock": "host monotonic seconds; robot timestamps are retained separately",
                    "error_basis": "estimated camera bases plus encoder-based controller FK; independent reference required for real error",
                    "collision_model": "nominal link/tool capsules, optional camera keep-out spheres; no CAD/self-collision certification"})
        engine = CooperativeController(args)
        with ExitStack() as cleanup:
            readers = {}
            for name in ROBOTS:
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                cleanup.callback(receiver.disconnect)
                readers[name] = StateReader(receiver, args.max_robot_age, require_joint_velocity=True)
            cleanup.callback(lambda: session.close() if session is not None else None)
            gui.start()
            vision.start()
            print(f"[research] PREVIEW method={args.method}; leader={engine.leader}, follower={engine.follower}; log={log_path}", flush=True)
            if any(getattr(args, name + "_tcp_offset") is None for name in ROBOTS):
                print("[research] preview assumes zero unspecified TCP offsets; motion requires both actual offsets.", flush=True)
            period = 1. / args.control_hz
            last_tick, last_report, last_wait = time.monotonic(), 0., None
            while not stop.is_set():
                now = time.monotonic()
                dt, last_tick = now - last_tick, now
                ready, goals, proposal = False, {}, None
                category = ""
                try:
                    if not vision.is_alive() or not gui.is_alive():
                        raise ResearchFault("process_exit", "vision or GUI process exited")
                    if session is not None and dt > min(args.max_robot_age, 1. / args.watchdog_hz):
                        raise ResearchFault("control_deadline", "control loop missed its deadline")
                    arm, target_changed = False, False
                    while True:
                        try:
                            command = commands.get_nowait()
                        except Empty:
                            break
                        if isinstance(command, dict):
                            if session is not None:
                                raise ResearchFault("target_change", "stop before changing the relative task")
                            engine.apply_target(command)
                            target_changed = True
                            logger.log({"kind": "target", "at": now, "command": command})
                        elif command == "arm":
                            arm = True
                        elif command in ("stop", "quit"):
                            stop.set()
                        elif command in ("pause_task", "resume_task"):
                            engine.task_paused = command == "pause_task"
                            logger.log({"kind": "task_pause", "at": now, "paused": engine.task_paused})
                    if stop.is_set():
                        break
                    if target_changed:
                        arm = False
                    packet = take_latest(frames)
                    if packet is not None:
                        accepted = engine.ingest(packet, now)
                        logger.log(finite_record({
                            "kind": "measurement", "at": now, "sequence": packet.sequence,
                            "captured_at": packet.captured_at, "received_at": now,
                            "poses_camera_model": packet.poses, "raw_mean_px": packet.raw_mean_px,
                            "raw_max_px": packet.raw_max_px, "valid_counts": packet.valid_counts,
                            "accepted": accepted, "rejection": engine.last_rejection}))
                    states = {name: reader.read(now) for name, reader in readers.items()}
                    proposal = engine.prepare(now, states, min(max(dt, 1e-6), 2*period), active=session is not None)
                    proposal.record["command_sent"] = False
                    goals, ready = proposal.goals, session is None
                    message = f"{'ACTIVE' if session else 'READY (preview)'} {args.method}; progress={engine.progress:.1%}, rate={proposal.decision.task_scale:.2f}"
                    if arm and session is None:
                        if args.execute:
                            session = MotionSession(args, engine.commanded, engine.offsets, stop_event=stop)
                            ready = False
                            last_tick = time.monotonic()
                            logger.log({"kind": "armed", "at": last_tick, "path": engine.path.kind})
                            message = "ARMED; waiting for fresh state before movement"
                        else:
                            message = "PREVIEW ONLY: --execute and actual TCP offsets are required"
                    elif session is not None:
                        def check_fresh():
                            checked = time.monotonic()
                            if stop.is_set():
                                raise ResearchFault("operator_stop", "operator/vision stop requested")
                            engine.state(checked, active=True)
                            if checked - now > args.max_robot_age or any(
                                    checked - reader.last_advance > args.max_robot_age for reader in readers.values()):
                                raise ResearchFault("robot_state_expired", "state/command expired during controller checks")
                        logger.log({"kind": "command_proposal", "at": now, "tcp_goals": proposal.tcp_goals,
                                    "twists": proposal.twists, "next_progress": proposal.next_progress})
                        session.send(proposal.tcp_goals, proposal.twists, check_fresh=check_fresh,
                                     safety_targets=proposal.future_tcp_goals)
                        engine.commit(proposal)
                        proposal.record["command_sent"] = True
                        proposal.record["sent_at"] = time.monotonic()
                        proposal.record["committed_progress"] = engine.progress
                        proposal.record["committed_finished"] = engine.finished
                    else:
                        proposal.record["command_sent"] = False
                    logger.log(proposal.record)
                    last_wait = None
                except (ControlFault, ValueError, KeyError, TypeError) as exc:
                    category = getattr(exc, "category", "control_fault")
                    if session is not None:
                        raise
                    ready = False
                    message = "WAIT: " + str(exc)
                    if message != last_wait:
                        logger.log({"kind": "wait", "at": now, "category": category, "reason": str(exc)})
                        last_wait = message
                status = {"message": message, "ready": ready, "active": session is not None,
                          "goals": goals, "progress": engine.progress, "task_paused": engine.task_paused,
                          "task_scale": proposal.decision.task_scale if proposal is not None else 0.,
                          "measurement_age": proposal.record["measurement_age"] if proposal is not None else None,
                          "fault_category": category}
                put_latest(statuses, status)
                put_latest(gui_statuses, status)
                if now - last_report >= 1.:
                    print("[research] " + message, flush=True)
                    last_report = now
                stop.wait(max(0., period - (time.monotonic() - now)))
    except BaseException as exc:
        final_reason = getattr(exc, "category", "operator_stop" if isinstance(exc, KeyboardInterrupt) else "control_fault")
        try:
            logger.log({"kind": "fault", "at": time.monotonic(), "category": final_reason,
                        "reason": str(exc), "external_base_stop_required": session is not None})
        except Exception as logging_exc:
            print(f"[log STOP] {logging_exc}", flush=True)
        raise
    finally:
        if session is not None:
            session.close()
        stop.set()
        for child in (vision, gui):
            if child.pid is not None:
                child.join(timeout=3.)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=1.)
        for queue in (frames, commands, statuses, gui_statuses):
            queue.cancel_join_thread()
            queue.close()
        try:
            logger.log({"kind": "end", "at": time.monotonic(), "reason": final_reason,
                        "progress": engine.progress if engine is not None else None,
                        "external_base_stop_required": session is not None})
        finally:
            logger.close()
        if session is not None:
            print("[STOP] Robot control ended. Stop external table/base motion separately.", flush=True)


def main(argv=None):
    args = parse_args(["--mode", "follow", *(sys.argv[1:] if argv is None else argv)])
    try:
        run(args)
    except KeyboardInterrupt:
        print("[research] stopped by operator", flush=True)
    except Exception as exc:
        raise SystemExit(f"[STOP] {exc}. Stop external table motion; no automatic restart.") from exc


if __name__ == "__main__":
    main()
