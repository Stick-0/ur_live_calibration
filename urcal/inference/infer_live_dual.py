"""Estimate UR5e (left) and UR10e (right) poses from one RealSense frame."""

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from urcal.inference.infer_live import (
    DEFAULT_KEY_NAMES as UR5E_KEY_NAMES,
    D435ColorStream,
    URJointReader,
    draw_keypoints,
    draw_pose_axes,
    make_segmentation_debug_bgr,
    make_transform,
    solve_pose_empnp,
    ur5e_keypoints_from_joints,
)
from urcal.inference.infer_live_ur10e import (
    DEFAULT_KEY_NAMES as UR10E_KEY_NAMES,
    ur10e_keypoints_from_joints,
)
from urcal.inference.network3 import RobotJointDetection
from urcal.inference.vote2 import PVNetVoteConfig, pvnet_vote


from urcal.paths import BUNDLE
MODEL_HW = (480, 640)


def split_regions(width, height, split_ratio=0.5):
    """Return non-overlapping (x0, y0, x1, y1) regions, left first."""
    if width < 2 or height < 1 or not 0.0 < split_ratio < 1.0:
        raise ValueError("Image dimensions and split_ratio must give two nonempty regions.")
    split_x = int(round(width * split_ratio))
    if not 0 < split_x < width:
        raise ValueError("split_ratio leaves an empty robot region at this resolution.")
    return (0, 0, split_x, height), (split_x, 0, width, height)


def letterbox_region(image_rgb, region, model_hw=MODEL_HW):
    """Resize a crop without stretching; return canvas and its content bounds."""
    x0, y0, x1, y1 = region
    crop = image_rgb[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    target_h, target_w = model_hw
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, min(target_w, int(round(width * scale))))
    resized_h = max(1, min(target_h, int(round(height * scale))))
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=image_rgb.dtype)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        crop, (resized_w, resized_h), interpolation=interpolation,
    )
    return canvas, (left, top, left + resized_w, top + resized_h)


def restore_keypoints(points, covariances, region, resized_hw):
    """Map votes in resized crop coordinates to the original camera image."""
    x0, y0, x1, y1 = region
    resized_h, resized_w = resized_hw
    scale = np.array([(x1 - x0) / resized_w, (y1 - y0) / resized_h])
    # OpenCV resize maps pixel centers, including a half-pixel translation.
    restored = (np.asarray(points, dtype=np.float64) + 0.5) * scale - 0.5 + [x0, y0]
    covariance = np.asarray(covariances, dtype=np.float64) * scale[None, :, None] * scale[None, None, :]
    return restored, covariance


@torch.no_grad()
def infer_region(model, image_rgb, region, transform, vote_cfg, device, show_seg=False):
    canvas, content = letterbox_region(image_rgb, region)
    x = transform(canvas).unsqueeze(0).to(device)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with amp:
        seg, vec = model(x)
        if tuple(seg.shape[-2:]) != MODEL_HW or tuple(vec.shape[-2:]) != MODEL_HW:
            raise ValueError("The dual runner requires 640x480 model outputs.")
        left, top, right, bottom = content
        # Exclude padding from voting. Translating pixel origins leaves direction
        # vectors unchanged; the votes now refer to the resized crop alone.
        seg = seg[..., top:bottom, left:right].contiguous()
        vec = vec[..., top:bottom, left:right].contiguous()
        mus, covs, extras = pvnet_vote(seg, vec, vote_cfg, return_debug=False)
    points, covariance = restore_keypoints(
        mus[0].cpu().numpy(), covs[0].cpu().numpy(), region, seg.shape[-2:],
    )
    valid = extras["valid"][0].cpu().numpy().copy()
    valid &= np.isfinite(points).all(axis=1) & np.isfinite(covariance).all(axis=(1, 2))
    valid &= (np.diagonal(covariance, axis1=1, axis2=2) >= 0).all(axis=1)
    x0, y0, x1, y1 = region
    valid &= (points[:, 0] >= x0) & (points[:, 0] < x1)
    valid &= (points[:, 1] >= y0) & (points[:, 1] < y1)
    # Exact intersections can have zero covariance; keep inverse-variance scores
    # finite without assigning fallback/invalid votes to the image center.
    covariance += np.eye(2)[None] * 1e-6
    probs = torch.sigmoid(seg[0]).cpu().float().numpy() if show_seg else None
    return points, covariance, valid, probs


@dataclass
class RobotResult:
    points: np.ndarray
    valid: np.ndarray
    seg_probs: Optional[np.ndarray]
    ok: bool
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    reproj_mean: float


@dataclass
class RobotEstimator:
    name: str
    model: torch.nn.Module
    key_names: list
    fk: Callable
    pose_args: argparse.Namespace
    prev_rvec: Optional[np.ndarray] = None
    prev_tvec: Optional[np.ndarray] = None

    def infer(self, image_rgb, region, joints, K, dist, transform, vote_cfg, device, show_seg=False):
        points, covariance, valid, probs = infer_region(
            self.model, image_rgb, region, transform, vote_cfg, device, show_seg,
        )
        points_3d = self.fk(joints)
        ok, rvec, tvec, reproj_mean = solve_pose_empnp(
            points_3d[valid], points[valid], covariance[valid], K, dist,
            self.pose_args, self.prev_rvec, self.prev_tvec, None,
        )
        if ok:
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
        return RobotResult(points, valid, probs, ok, rvec, tvec, reproj_mean)


def process_frame(bgr, joint_values, estimators, K, dist, split_ratio, transform, vote_cfg, device, show_seg=False):
    """Run both models sequentially on the SAME captured frame and joint samples."""
    regions = split_regions(bgr.shape[1], bgr.shape[0], split_ratio)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = [
        estimator.infer(rgb, region, joints, K, dist, transform, vote_cfg, device, show_seg)
        for estimator, region, joints in zip(estimators, regions, joint_values)
    ]
    return regions, results


def draw_results(bgr, regions, estimators, results, K, dist, axis_len, fps, show_seg, show_keypoints=True):
    vis = bgr.copy()
    seg_vis = bgr.copy() if show_seg else None
    for region, estimator, result in zip(regions, estimators, results):
        x0, y0, x1, y1 = region
        if show_keypoints:
            visible = result.points.copy()
            visible[~result.valid] = np.nan
            vis = draw_keypoints(vis, visible, estimator.key_names)
        if result.ok:
            vis = draw_pose_axes(vis, K, dist, result.rvec, result.tvec, axis_len, thickness=2)
        color = (40, 220, 40) if result.ok else (40, 40, 220)
        state = "OK" if result.ok else "FAIL"
        cv2.putText(vis, f"{estimator.name} EMPnP {state} ({result.valid.sum()}/7)",
                    (x0 + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if seg_vis is not None:
            seg_vis[y0:y1, x0:x1] = make_segmentation_debug_bgr(
                bgr[y0:y1, x0:x1], result.seg_probs, estimator.key_names, estimator.pose_args.seg_thr,
            )
    cv2.putText(vis, f"Dual FPS {fps:.1f}", (8, bgr.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis, seg_vis


def require_weights(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing weights: {path}. Download them with git lfs pull.")
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run git lfs pull before inference.")


def load_model(path, device):
    # Load on CPU so checkpoint contents do not occupy extra GPU memory.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RobotJointDetection()
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    del checkpoint
    model = model.eval().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def close_joint_reader(reader):
    if reader.receiver is not None:
        reader.receiver.disconnect()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="One D435 frame: left UR5e, right UR10e; two independent camera-from-base poses.")
    for robot, folder, default_lambda in (("ur5e", "v3m2a3", 0.5), ("ur10e", "ur10e_2", 0.2)):
        parser.add_argument(f"--{robot}_checkpoint", type=Path, default=BUNDLE / "trained_models" / folder / "best.pt")
        parser.add_argument(f"--{robot}_ip", help=f"{robot} RTDE controller IP")
        parser.add_argument(f"--{robot}_joints", type=float, nargs=6, help="Fixed/fallback joint angles in radians")
        parser.add_argument(f"--{robot}_empnp_lambda", type=float, default=default_lambda)
    parser.add_argument("--split_ratio", type=float, default=0.5, help="Left UR5e share of image width (default: 0.5)")
    parser.add_argument("--camera_w", type=int, default=640)
    parser.add_argument("--camera_h", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seg_thr", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=0.99)
    parser.add_argument("--max_pixels", type=int, default=4000)
    parser.add_argument("--num_hypotheses", type=int, default=8)
    parser.add_argument("--min_valid_pixels", type=int, default=50)
    parser.add_argument("--angle_bin_deg", type=int, default=3)
    parser.add_argument("--no_use_angle_binning", action="store_false", dest="use_angle_binning", default=True)
    parser.add_argument("--empnp_sigma", type=float, default=3.0)
    parser.add_argument("--empnp_prior", type=float, default=0.9)
    parser.add_argument("--empnp_max_iter", type=int, default=50)
    parser.add_argument("--empnp_tol", type=float, default=1e-6)
    parser.add_argument("--no_empnp_use_prev", action="store_false", dest="empnp_use_prev", default=True)
    parser.add_argument("--axis_len", type=float, default=0.1)
    parser.add_argument("--show_seg", action="store_true")
    parser.add_argument("--hide_keypoints", action="store_true", help="Start with keypoint dots and labels hidden; press k to toggle")
    parser.add_argument("--window", default="RobotPose Dual: UR5e | UR10e")
    parser.add_argument("--seg_window", default="RobotPose Dual Segmentation")
    args = parser.parse_args(argv)
    try:
        split_regions(args.camera_w, args.camera_h, args.split_ratio)
    except ValueError as exc:
        parser.error(str(exc))
    for robot in ("ur5e", "ur10e"):
        joints = getattr(args, f"{robot}_joints")
        if not getattr(args, f"{robot}_ip") and joints is None:
            parser.error(f"Set --{robot}_ip or --{robot}_joints for each robot.")
        if joints is not None and not np.isfinite(joints).all():
            parser.error(f"--{robot}_joints must contain finite angles in radians.")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Validate assets before allocating models or opening hardware connections.
    for path in (args.ur5e_checkpoint, args.ur10e_checkpoint,
                 BUNDLE / "torch_cache/hub/checkpoints/resnet18-f37072fd.pth"):
        require_weights(path)
    torch.hub.set_dir(str(BUNDLE / "torch_cache/hub"))
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was selected, but CUDA is not available.")
    transform = make_transform(*MODEL_HW, normalize=True)
    vote_cfg = PVNetVoteConfig(**{
        name: getattr(args, name) for name in (
            "seg_thr", "theta", "max_pixels", "num_hypotheses", "min_valid_pixels",
            "use_angle_binning", "angle_bin_deg",
        )
    })
    estimators = []
    for robot, names, fk in (("ur5e", UR5E_KEY_NAMES, ur5e_keypoints_from_joints),
                              ("ur10e", UR10E_KEY_NAMES, ur10e_keypoints_from_joints)):
        print(f"[info] loading {robot} on {device}", flush=True)
        pose_args = argparse.Namespace(**vars(args))
        pose_args.empnp_lambda = getattr(args, f"{robot}_empnp_lambda")
        estimators.append(RobotEstimator(
            robot.upper(), load_model(getattr(args, f"{robot}_checkpoint"), device), names, fk, pose_args,
        ))

    with ExitStack() as cleanup:
        cleanup.callback(cv2.destroyAllWindows)
        readers = []
        for robot in ("ur5e", "ur10e"):
            reader = URJointReader(getattr(args, f"{robot}_ip"), getattr(args, f"{robot}_joints"))
            cleanup.callback(close_joint_reader, reader)
            readers.append(reader)
        stream = D435ColorStream(args.camera_w, args.camera_h, args.fps)
        cleanup.callback(stream.stop)
        show_seg = args.show_seg
        show_keypoints = not args.hide_keypoints
        fps_ema = None
        print("[info] left=UR5e, right=UR10e; q/ESC quit, s toggles segmentation, k toggles keypoints")
        while True:
            start = time.perf_counter()
            bgr = stream.read()
            # Sample BOTH robots before either model runs. RTDE samples are not
            # hardware-synchronized with the camera exposure.
            joint_values = [reader.read() for reader in readers]
            regions, results = process_frame(
                bgr, joint_values, estimators, stream.K, stream.dist, args.split_ratio,
                transform, vote_cfg, device, show_seg,
            )
            vis, seg_vis = draw_results(
                bgr, regions, estimators, results, stream.K, stream.dist,
                args.axis_len, fps_ema or 0.0, show_seg, show_keypoints,
            )
            cv2.imshow(args.window, vis)
            if seg_vis is not None:
                cv2.imshow(args.seg_window, seg_vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                show_seg = not show_seg
                if not show_seg:
                    cv2.destroyWindow(args.seg_window)
            if key == ord("k"):
                show_keypoints = not show_keypoints
                print(f"[info] keypoints: {'ON' if show_keypoints else 'OFF'}")
            fps_now = 1.0 / max(time.perf_counter() - start, 1e-9)
            fps_ema = fps_now if fps_ema is None else 0.9 * fps_ema + 0.1 * fps_now


if __name__ == "__main__":
    main()
