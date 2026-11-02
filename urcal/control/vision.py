"""Non-actuating perception process for control_dual.py.

A dedicated capture thread drains the camera while neural inference runs.
The controller consumes only the latest paired estimate, tagged with capture
time. Camera/RTDE samples are approximately aligned, not hardware synchronized.
"""

import argparse
from contextlib import ExitStack
from queue import Full
import threading
import time

import numpy as np

from urcal.inference.display import import_opencv
from urcal.inference.reporting import InferenceReporter

cv2 = import_opencv()

from urcal.control.diagnostics import (
    calibration_debug_record, draw_calibration_debug, pose_input_diagnostics, save_calibration_debug,
)
from urcal.control.geometry import matrix_pose, pose_matrix
from urcal.control.policy import CalibrationFrame
from urcal.control.joint_history import JointAlignmentUnavailable, JointSampler


class LatestCapture:
    def __init__(self, stream, receivers, stop, fps, max_robot_age=.15, *, align_joints=False,
                 joint_history_hz=100., joint_alignment_wait=.03, joint_alignment_max_gap=.05):
        self.stream, self.receivers, self.stop_event = stream, receivers, stop
        self.period = 1. / fps
        self.max_robot_age = max_robot_age
        self.lock = threading.Lock()
        self.latest = None
        self.latest_rejection = None
        self.error = None
        self.align_joints = bool(align_joints)
        self.joint_alignment_wait, self.joint_alignment_max_gap = joint_alignment_wait, joint_alignment_max_gap
        self.joint_sampler = (JointSampler(receivers, stop, joint_history_hz, max_robot_age)
                              if self.align_joints else None)
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-capture")
        self.thread.start()

    def _run(self):
        offset = None
        previous_frame = None
        previous_sensor_time = None
        previous_timestamps = {name: None for name in self.receivers}
        last_progress = {name: time.monotonic() for name in self.receivers}
        warmed_frames = 0
        try:
            while not self.stop_event.is_set():
                frames = self.stream.pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                received = time.monotonic()
                wall_received = time.time()
                number = int(color.get_frame_number())
                sensor_time = float(color.get_timestamp()) / 1000.
                if not np.isfinite(sensor_time):
                    raise RuntimeError("invalid camera timestamp")
                if previous_frame is not None and (number <= previous_frame or sensor_time <= previous_sensor_time):
                    raise RuntimeError("camera timestamp/frame number stopped advancing")
                previous_frame, previous_sensor_time = number, sensor_time
                domain = color.get_frame_timestamp_domain()
                rs = self.stream.rs
                if domain in (rs.timestamp_domain.system_time, rs.timestamp_domain.global_time):
                    age = wall_received - sensor_time
                    if age < -.05:
                        raise RuntimeError("camera system clock is ahead of host")
                    # Interpolation needs the best timestamp estimate. An
                    # uncertainty allowance must not move the selected q a
                    # whole camera frame into the past. Keep the older-mode
                    # conservative age convention unchanged.
                    captured = received - max(age, 0. if self.align_joints else self.period)
                else:
                    # Device clock -> host monotonic lower-latency offset estimate.
                    # The arrival offset retains unknown constant transport
                    # latency. New-mode uncertainty is recorded separately;
                    # older modes retain their conservative exposure allowance.
                    candidate = received - sensor_time
                    offset = candidate if offset is None else min(offset, candidate)
                    captured = min(received, sensor_time + offset) - (0. if self.align_joints else self.period)
                if self.joint_sampler is not None:
                    image = np.asanyarray(color.get_data()).copy()
                    warmed_frames += 1
                    if warmed_frames < 10:
                        continue
                    alignment = {
                        "mode": "capture_time_interpolation", "aligned": False,
                        "hardware_synchronized": False, "sequence": number,
                        "captured_at": captured, "received_at": received,
                        "camera": {
                            "sensor_timestamp_s": sensor_time, "timestamp_domain": str(domain),
                            "frame_period_s": self.period,
                            "exposure_time_uncertainty_s": self.period,
                            "mapping": ("camera wall time to host monotonic"
                                        if domain in (rs.timestamp_domain.system_time, rs.timestamp_domain.global_time)
                                        else "minimum device-arrival offset; transport-biased time estimate"),
                            "clock_offset_s": (received - wall_received if domain in (
                                rs.timestamp_domain.system_time, rs.timestamp_domain.global_time) else offset),
                            "uncertainty_note": "separate one-frame exposure uncertainty, not a timestamp shift; unknown constant camera/RTDE transport latency",
                        },
                    }
                    interpolation_started = time.monotonic()
                    try:
                        joints, joint_metadata = self.joint_sampler.interpolate(
                            captured, wait=self.joint_alignment_wait, max_gap=self.joint_alignment_max_gap)
                    except JointAlignmentUnavailable as exc:
                        alignment.update(reason=str(exc), resolved_at=time.monotonic(),
                                         robots=getattr(exc, "diagnostics", {}),
                                         interpolation_wait_s=time.monotonic() - interpolation_started)
                        with self.lock:
                            self.latest_rejection = (number, captured, alignment)
                        continue
                    alignment.update(aligned=True, robots=joint_metadata, resolved_at=time.monotonic(),
                                     interpolation_wait_s=time.monotonic() - interpolation_started)
                    # This is an exposure/support uncertainty budget, not the
                    # time elapsed while inference or capture waited for data.
                    skew = self.period + max(
                        row["max_bracket_distance_s"] + row["sdk_read_half_interval_s"]
                        for row in joint_metadata.values())
                    alignment["effective_joint_skew_s"] = skew
                    with self.lock:
                        self.latest = (number, captured, image, joints, skew, alignment)
                    continue
                joints = []
                for name, receiver in self.receivers.items():
                    if not receiver.isConnected():
                        raise RuntimeError(f"{name}: capture RTDE disconnected")
                    stamp = float(receiver.getTimestamp())
                    if not np.isfinite(stamp) or (previous_timestamps[name] is not None and stamp < previous_timestamps[name]):
                        raise RuntimeError(f"{name}: invalid capture RTDE timestamp")
                    if previous_timestamps[name] is None or stamp > previous_timestamps[name]:
                        last_progress[name], previous_timestamps[name] = received, stamp
                    if received - last_progress[name] > self.max_robot_age:
                        raise RuntimeError(f"{name}: capture RTDE state is stale")
                    q = np.asarray(receiver.getActualQ(), dtype=float)
                    if q.shape != (6,) or not np.isfinite(q).all():
                        raise RuntimeError(f"{name}: invalid capture joints")
                    joints.append(q.tolist())
                skew = time.monotonic() - captured
                image = np.asanyarray(color.get_data()).copy()
                warmed_frames += 1
                if warmed_frames < 10:
                    continue
                with self.lock:
                    self.latest = (number, captured, image, joints, skew)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def read_after(self, sequence):
        if self.joint_sampler is not None and self.joint_sampler.error:
            raise RuntimeError(self.joint_sampler.error)
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            newer = (self.latest is not None and
                     (sequence is None or (self.latest[0] > sequence if self.align_joints
                                            else self.latest[0] != sequence)))
            if newer:
                return self.latest
            return None

    def read_rejection_after(self, sequence):
        with self.lock:
            if self.latest_rejection is not None and (sequence is None or self.latest_rejection[0] > sequence):
                return self.latest_rejection
            return None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.5)
        if self.joint_sampler is not None:
            self.joint_sampler.close()


def calibration_frame(sequence, captured, results, estimators, joints, K, dist, max_skew, joint_skew,
                      *, regions=None, joint_alignment=None):
    """Measure errors against ORIGINAL detections, not EM-corrected points."""
    poses, means, maxima, counts = {}, {}, {}, {}
    reason = ""
    if not np.isfinite(joint_skew) or joint_skew > max_skew:
        reason = f"camera/joint sampling skew {joint_skew:.3f}s exceeds {max_skew:.3f}s"
    for name, result, estimator, q in zip(("ur5e", "ur10e"), results, estimators, joints):
        valid = result.valid & np.isfinite(result.points).all(axis=1)
        counts[name] = int(valid.sum())
        if not result.ok:
            reason = reason or getattr(result, "constraint_error", "") or f"{name}: EMPnP failed"
            continue
        if counts[name] < 4:
            reason = reason or f"{name}: too few original detections"
            continue
        X = estimator.fk(q)[valid]
        T = pose_matrix(np.concatenate((np.asarray(result.tvec).reshape(3), np.asarray(result.rvec).reshape(3))))
        depths = (X @ T[:3, :3].T + T[:3, 3])[:, 2]
        if not np.isfinite(depths).all() or np.any(depths <= 0):
            reason = reason or f"{name}: points behind camera"
            continue
        points = result.points[valid]
        spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if len(spread) < 2 or spread[1] / np.sqrt(len(points)) < 5.:
            reason = reason or f"{name}: detections are too collinear/small"
            continue
        projected = cv2.projectPoints(X, result.rvec, result.tvec, K, dist)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - points, axis=1)
        poses[name], means[name], maxima[name] = T, float(errors.mean()), float(errors.max())
    diagnostics = (None if regions is None else
                   pose_input_diagnostics(results, estimators, joints, K, dist, regions))
    if joint_alignment is not None:
        diagnostics = {} if diagnostics is None else diagnostics
        diagnostics["joint_alignment"] = joint_alignment
    return CalibrationFrame(sequence, captured, poses, means, maxima, counts, reason, diagnostics)


def make_pose_args(args, robot):
    """Pass the control CLI's per-robot settings to the shared EMPnP wrapper."""
    default_lambda = {"ur5e": .5, "ur10e": .2}[robot]
    return argparse.Namespace(
        camera_w=args.camera_w, camera_h=args.camera_h,
        empnp_use_prev=getattr(args, "empnp_use_prev", False),
        empnp_sigma=3., empnp_prior=.9, empnp_max_iter=50, empnp_tol=1e-6,
        empnp_lambda=getattr(args, robot + "_empnp_lambda", default_lambda), seg_thr=.5,
    )


def vision_worker(args, frames, commands, statuses, stop):
    # Heavy CUDA/model imports are isolated from the timed controller process.
    import torch
    from rtde_receive import RTDEReceiveInterface
    import urcal.inference.infer_live_dual as dual
    from urcal.control.runtime import put_latest, take_latest

    window, seg_window = "Dual flange control", "Dual control segmentation"
    try:
        for path in (args.ur5e_checkpoint, args.ur10e_checkpoint,
                     dual.BUNDLE / "torch_cache/hub/checkpoints/resnet18-f37072fd.pth"):
            dual.require_weights(path)
        torch.hub.set_dir(str(dual.BUNDLE / "torch_cache/hub"))
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
        transform = dual.make_transform(*dual.MODEL_HW, normalize=True)
        vote_cfg = dual.PVNetVoteConfig(num_hypotheses=8)
        estimators = []
        for name, keys, fk in (
            ("ur5e", dual.UR5E_KEY_NAMES, dual.ur5e_keypoints_from_joints),
            ("ur10e", dual.UR10E_KEY_NAMES, dual.ur10e_keypoints_from_joints),
        ):
            print(f"[vision] loading {name} on {device}", flush=True)
            pose_args = make_pose_args(args, name)
            print(f"[vision] {name} EMPnP previous pose: {'ON' if pose_args.empnp_use_prev else 'OFF'}; "
                  f"lambda={pose_args.empnp_lambda:g}; max_iter={pose_args.empnp_max_iter}", flush=True)
            estimators.append(dual.RobotEstimator(
                name, dual.load_model(getattr(args, name + "_checkpoint"), device), keys, fk, pose_args,
            ))
        with ExitStack() as cleanup:
            cleanup.callback(cv2.destroyAllWindows)
            receivers = {}
            for name in ("ur5e", "ur10e"):
                receiver = RTDEReceiveInterface(getattr(args, name + "_ip"))
                receivers[name] = receiver
                cleanup.callback(receiver.disconnect)
            stream = dual.D435ColorStream(args.camera_w, args.camera_h, args.fps)
            cleanup.callback(stream.stop)
            align_joints = getattr(args, "base_motion_mode", "pose") == "relative_kalman"
            capture = LatestCapture(
                stream, receivers, stop, args.fps, args.max_robot_age, align_joints=align_joints,
                joint_history_hz=getattr(args, "joint_history_hz", 100.),
                joint_alignment_wait=getattr(args, "joint_alignment_wait", .03),
                joint_alignment_max_gap=getattr(args, "joint_alignment_max_gap", .05))
            cleanup.callback(capture.close)
            sequence = None
            last_frame = time.monotonic()
            status = {"message": "WAIT: collecting stable calibration", "goals": {}}
            show_seg, show_keypoints = args.show_seg, not args.hide_keypoints
            reporter = InferenceReporter(interval=1.)
            last_alignment_rejection, last_alignment_report = None, float("-inf")
            if align_joints:
                print("[vision] Joint alignment: exposure-time RTDE interpolation; "
                      "first-observation clock mapping, no extrapolation; not hardware synchronized.", flush=True)
            print("[vision] left=UR5e, right=UR10e; a: Arm | SPACE/q/ESC: stop+exit | "
                  "k: keypoints | s: segmentation | d: save diagnostic (preview only)", flush=True)
            print("[vision] Magenta crosses: pose reprojection; lines: detection residual. "
                  "Pose/target axes are shown without text; controller status is printed by [research].", flush=True)
            while not stop.is_set():
                sample = capture.read_after(sequence)
                if align_joints:
                    rejected = capture.read_rejection_after(last_alignment_rejection)
                    if rejected is not None:
                        last_alignment_rejection, rejected_at, alignment = rejected
                        if sample is None or last_alignment_rejection > sample[0]:
                            sequence = last_alignment_rejection
                            sample = None
                            reason = "camera/joint alignment: " + alignment["reason"]
                            put_latest(frames, CalibrationFrame(
                                sequence, rejected_at, {}, {}, {}, {name: 0 for name in receivers},
                                reason, {"joint_alignment": alignment}))
                            reported_at = time.monotonic()
                            if reported_at - last_alignment_report >= 1.:
                                print("[vision] WAIT: " + reason, flush=True)
                                last_alignment_report = reported_at
                if sample is None:
                    if time.monotonic() - last_frame > 2.:
                        raise RuntimeError("camera capture timed out")
                    stop.wait(.002)
                    continue
                start = time.monotonic()
                sequence, captured, bgr, joints, skew = sample[:5]
                joint_alignment = sample[5] if len(sample) > 5 else None
                last_frame = start
                regions, results = dual.process_frame(
                    bgr, joints, estimators, stream.K, stream.dist, args.split_ratio,
                    transform, vote_cfg, device, show_seg,
                    common_base_plane=getattr(args, "common_base_plane", False),
                )
                packet = calibration_frame(sequence, captured, results, estimators, joints,
                                           stream.K, stream.dist, args.max_joint_skew, skew, regions=regions,
                                           joint_alignment=joint_alignment)
                put_latest(frames, packet)
                update = take_latest(statuses)
                if update is not None:
                    status = update
                fps = 1. / max(time.monotonic() - start, 1e-6)
                reporter.emit(time.monotonic(), fps, estimators, results, packet=packet)
                vis, seg_vis = dual.draw_results(
                    bgr, regions, estimators, results, stream.K, stream.dist, .06, fps, show_seg, show_keypoints,
                )
                debug = calibration_debug_record(
                    packet, results, estimators, joints, stream.K, stream.dist, regions,
                    args.max_reproj_px, args.max_point_reproj_px, args.min_points,
                    check_reprojection=args.check_reprojection,
                )
                draw_calibration_debug(vis, debug, show_keypoints, show_text=False)
                for name, goal in status.get("goals", {}).items():
                    pose = matrix_pose(goal)
                    if pose[2] <= 0:
                        continue
                    vis = dual.draw_pose_axes(vis, stream.K, stream.dist, pose[3:], pose[:3], .06, 2)
                cv2.imshow(window, vis)
                if seg_vis is not None:
                    cv2.imshow(seg_window, seg_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27, ord(" ")):
                    # Shared event also works if the small command queue is full.
                    stop.set()
                elif key == ord("a"):
                    try:
                        commands.put_nowait(("arm", time.monotonic()))
                    except Full:
                        print("[vision] command queue busy; review target and press Arm again", flush=True)
                elif key == ord("k"):
                    show_keypoints = not show_keypoints
                elif key == ord("d"):
                    if args.execute:
                        print("[vision] diagnostic saving is available in preview without --execute", flush=True)
                    else:
                        try:
                            saved_overlay = draw_calibration_debug(vis.copy(), debug, show_keypoints, show_text=True)
                            path = save_calibration_debug(dual.BUNDLE / "runs/calibration-debug", bgr, saved_overlay,
                                                          debug, status.get("message", ""))
                            print(f"[vision] diagnostic snapshot: {path}", flush=True)
                        except (OSError, ValueError, cv2.error) as exc:
                            print(f"[vision] could not save diagnostic snapshot: {exc}", flush=True)
                elif key == ord("s"):
                    show_seg = not show_seg
                    if not show_seg:
                        cv2.destroyWindow(seg_window)
    except Exception as exc:
        print(f"[vision STOP] {exc}", flush=True)
    finally:
        stop.set()
