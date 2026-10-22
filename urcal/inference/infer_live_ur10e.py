# infer_live_ur10e.py
import argparse
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

from urcal.inference.display import import_opencv

cv2 = import_opencv()

import numpy as np
import torch
from torchvision import transforms

from urcal.inference.my_pnp import EMPnP
from urcal.inference.network3 import RobotJointDetection
from urcal.inference.vote2 import PVNetVoteConfig, pvnet_vote


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_KEY_NAMES = [
    "ur10e_link0",
    "ur10e_link1",
    "ur10e_link2",
    "ur10e_link3",
    "ur10e_link4",
    "ur10e_link5",
    "ur10e_link6",
]
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

torch.backends.cudnn.benchmark = True


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


def make_transform(height: int, width: int, normalize: bool):
    ops = [transforms.ToTensor(), transforms.Resize((height, width), antialias=True)]
    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx


def transform_from_rpy_xyz(rpy: Tuple[float, float, float], xyz: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


UR10E_DEFAULT_CHECKPOINT = "trained_models/ur10e_2/best.pt"
UR10E_BASE_Z_OFFSET = 0.005


def compute_ur10e_link_transforms(joints: Dict[str, float]) -> Dict[str, np.ndarray]:
    # Matches robotpose/c_j_saver3_ur10e.py and mask_gen_non_occ_ur10e.py.
    # The saved UR10e keypoints use the Gazebo base_link height offset.
    link_tree = {
        "base_link_inertia": ("base_link", (0.0, 0.0, np.pi), (0.0, 0.0, 0.0), None),
        "shoulder_link": ("base_link_inertia", (0.0, 0.0, 0.0), (0.0, 0.0, 0.1807), "shoulder_pan_joint"),
        "upper_arm_link": ("shoulder_link", (np.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0), "shoulder_lift_joint"),
        "forearm_link": ("upper_arm_link", (0.0, 0.0, 0.0), (-0.6127, 0.0, 0.0), "elbow_joint"),
        "wrist_1_link": ("forearm_link", (0.0, 0.0, 0.0), (-0.57155, 0.0, 0.17415), "wrist_1_joint"),
        "wrist_2_link": ("wrist_1_link", (np.pi / 2, 0.0, 0.0), (0.0, -0.11985, 0.0), "wrist_2_joint"),
        "wrist_3_link": ("wrist_2_link", (np.pi / 2, np.pi, np.pi), (0.0, 0.11655, 0.0), "wrist_3_joint"),
    }
    T = {"base_link": transform_from_rpy_xyz((0.0, 0.0, 0.0), (0.0, 0.0, UR10E_BASE_Z_OFFSET))}
    for child, (parent, rpy, xyz, joint) in link_tree.items():
        T_joint = transform_from_rpy_xyz(rpy, xyz)
        T_rot = np.eye(4, dtype=np.float64)
        T_rot[:3, :3] = rpy_to_matrix(0.0, 0.0, joints.get(joint, 0.0) if joint else 0.0)
        T[child] = T[parent] @ T_joint @ T_rot
    return T


def ur10e_keypoints_from_joints(joint_values: List[float]) -> np.ndarray:
    joints = {name: float(value) for name, value in zip(JOINT_NAMES, joint_values)}
    T = compute_ur10e_link_transforms(joints)
    forearm_to_wrist_x = transform_from_rpy_xyz((0.0, 0.0, 0.0), (-0.57155, 0.0, 0.0))
    points = [
        T["base_link"][:3, 3],
        T["shoulder_link"][:3, 3],
        T["forearm_link"][:3, 3],
        (T["forearm_link"] @ forearm_to_wrist_x)[:3, 3],
        T["wrist_1_link"][:3, 3],
        T["wrist_2_link"][:3, 3],
        T["wrist_3_link"][:3, 3],
    ]
    return np.asarray(points, dtype=np.float64)


class URJointReader:
    def __init__(self, robot_ip: Optional[str], fallback_joints: Optional[List[float]]):
        self.fallback_joints = fallback_joints
        self.receiver = None
        if robot_ip:
            try:
                from rtde_receive import RTDEReceiveInterface

                self.receiver = RTDEReceiveInterface(robot_ip)
                print(f"[info] connected to UR RTDE: {robot_ip}")
            except Exception as exc:
                if fallback_joints is None:
                    raise RuntimeError(f"UR RTDE connection failed: {exc}") from exc
                print(f"[warn] UR RTDE connection failed, using --joints fallback: {exc}")

    def read(self) -> List[float]:
        if self.receiver is not None:
            return list(self.receiver.getActualQ())
        if self.fallback_joints is not None:
            return self.fallback_joints
        raise RuntimeError("No UR joint source. Set --robot_ip or --joints.")


class D435ColorStream:
    def __init__(self, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("pyrealsense2 is required for D435 streaming.") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist = np.asarray(intr.coeffs, dtype=np.float64)
        print(f"[info] D435 color stream: {width}x{height}@{fps}")
        print(f"[info] RealSense K: {self.K.reshape(-1).tolist()}")
        print(f"[info] RealSense dist: {self.dist.tolist()}")

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No D435 color frame received.")
        return np.asanyarray(color.get_data())

    def stop(self):
        self.pipeline.stop()


class RosPosePublisher:
    def __init__(self, topic: str, frame_id: str, reproj_topic: Optional[str] = None):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Float32
        except Exception as exc:
            raise RuntimeError("rclpy, geometry_msgs, and std_msgs are required for ROS pose publishing.") from exc

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.Float32 = Float32
        self.frame_id = frame_id

        if not rclpy.ok():
            rclpy.init(args=None)
            self.did_init = True
        else:
            self.did_init = False
        self.node = rclpy.create_node("robotpose_ur10e_pnp_publisher")
        self.pose_pub = self.node.create_publisher(PoseStamped, topic, 10)
        self.reproj_pub = self.node.create_publisher(Float32, reproj_topic, 10) if reproj_topic else None
        print(f"[info] ROS PoseStamped publisher: {topic} frame_id={frame_id}")
        if reproj_topic:
            print(f"[info] ROS reprojection error publisher: {reproj_topic}")

    def publish(self, rvec: np.ndarray, tvec: np.ndarray, reproj_mean: float):
        R_cam_obj, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
        qx, qy, qz, qw = rotation_matrix_to_quaternion_xyzw(R_cam_obj)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)

        msg = self.PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self.pose_pub.publish(msg)

        if self.reproj_pub is not None and np.isfinite(reproj_mean):
            err_msg = self.Float32()
            err_msg.data = float(reproj_mean)
            self.reproj_pub.publish(err_msg)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if self.did_init:
            self.rclpy.shutdown()


def color_table(count: int):
    base = [
        (71, 99, 255),
        (50, 205, 50),
        (255, 144, 30),
        (0, 215, 255),
        (211, 85, 186),
        (255, 255, 0),
        (180, 105, 255),
    ]
    return (base * ((count + len(base) - 1) // len(base)))[:count]


def draw_keypoints(bgr: np.ndarray, points_2d: np.ndarray, names: List[str]) -> np.ndarray:
    out = bgr.copy()
    colors = color_table(len(points_2d))
    for idx, (x, y) in enumerate(points_2d):
        if not np.isfinite(x + y):
            continue
        pt = (int(round(x)), int(round(y)))
        cv2.circle(out, pt, 5, colors[idx], -1, cv2.LINE_AA)
        cv2.putText(out, names[idx], (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[idx], 1, cv2.LINE_AA)
    return out


def make_segmentation_debug_bgr(bgr: np.ndarray, probs: np.ndarray, names: List[str], threshold: float) -> np.ndarray:
    colors = np.asarray(color_table(probs.shape[0]), dtype=np.uint8)
    best_idx = np.argmax(probs, axis=0)
    best_prob = np.max(probs, axis=0)
    color_mask = colors[best_idx]
    color_mask[best_prob < threshold] = 0

    color_mask = cv2.resize(color_mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize((best_prob >= threshold).astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = bgr.copy()
    # OpenCV returns None for empty inputs when no pixel passes the threshold.
    if np.any(mask):
        out[mask] = cv2.addWeighted(out[mask], 0.45, color_mask[mask], 0.55, 0)

    y = 24
    for idx, name in enumerate(names):
        cv2.rectangle(out, (12, y - 12), (28, y + 4), tuple(int(c) for c in colors[idx]), -1)
        cv2.putText(out, name, (36, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    cv2.putText(out, "seg debug (s toggle)", (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_pose_axes(
    bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_len: float,
    thickness: int,
) -> np.ndarray:
    out = bgr.copy()
    axis_3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    O, X, Y, Z = pts_2d
    cv2.line(out, tuple(O), tuple(X), (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.line(out, tuple(O), tuple(Y), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(out, tuple(O), tuple(Z), (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.circle(out, tuple(O), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def score_from_covariance(covs: np.ndarray) -> np.ndarray:
    scores = 1.0 / ((covs[:, 0, 0] + covs[:, 1, 1]) / 2.0)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if score_max - score_min < 1e-12:
        return np.full(scores.shape, 0.9, dtype=np.float64)
    scores = (scores - score_min) / (score_max - score_min)
    return 0.05 + 0.90 * scores


@torch.no_grad()
def infer_keypoints(model, image_rgb: np.ndarray, tf, vote_cfg: PVNetVoteConfig, device: torch.device, amp_ctx,
                    return_seg_probs: bool = True):
    H0, W0 = image_rgb.shape[:2]
    x = tf(image_rgb).unsqueeze(0).to(device)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    with amp_ctx:
        seg_pred, vec_pred = model(x)
        mus, covs, _ = pvnet_vote(seg_pred, vec_pred, vote_cfg, return_debug=False)
    points = mus[0].detach().cpu().numpy().astype(np.float64)
    covs_np = covs[0].detach().cpu().numpy().astype(np.float64)
    Ht, Wt = int(seg_pred.shape[-2]), int(seg_pred.shape[-1])
    sx = float(W0) / float(Wt)
    sy = float(H0) / float(Ht)
    points[:, 0] *= sx
    points[:, 1] *= sy
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
    covs_orig = np.einsum("ab,kbc,cd->kad", S, covs_np, S.T)
    # The live loop only needs this full-resolution CPU copy for the debug window.
    probs = torch.sigmoid(seg_pred[0]).detach().cpu().float().numpy() if return_seg_probs else None
    return points, covs_orig, probs


def solve_pose_empnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    covs_2d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    args,
    prev_rvec: Optional[np.ndarray],
    prev_tvec: Optional[np.ndarray],
    prev_points_2d: Optional[np.ndarray],
):
    valid = np.isfinite(points_2d).all(axis=1) & np.isfinite(points_3d).all(axis=1)
    if int(valid.sum()) < 4:
        return False, None, None, np.nan

    if args.empnp_use_prev and prev_rvec is not None and prev_tvec is not None:
        u_pre_full, _ = cv2.projectPoints(points_3d, prev_rvec, prev_tvec, K, dist)
        u_pre_full = u_pre_full.reshape(-1, 2)
    elif args.empnp_use_prev and prev_points_2d is not None:
        u_pre_full = prev_points_2d.copy()
    else:
        u_pre_full = points_2d.copy()

    X = points_3d[valid].astype(np.float64)
    u = points_2d[valid].astype(np.float64)
    u_pre = u_pre_full[valid].astype(np.float64)
    em_score = score_from_covariance(covs_2d)
    pi_prior = np.clip(em_score * (args.empnp_prior / 0.9), 1e-6, 1.0 - 1e-6)[valid]

    try:
        empnp = EMPnP(K, dist, sigma=args.empnp_sigma, image_size=(args.camera_w, args.camera_h))
        rvec, tvec, _, reproj_errs = empnp.fit(
            X,
            u,
            u_pre,
            pi_prior=pi_prior,
            max_iter=args.empnp_max_iter,
            tol=args.empnp_tol,
            use_ransac_init=True,
            lamda=args.empnp_lambda,
        )
        R_pred, _ = cv2.Rodrigues(rvec)
        ok = np.isfinite(tvec).all() and np.isfinite(R_pred).all()
        reproj_mean = float(np.mean(reproj_errs)) if reproj_errs is not None and len(reproj_errs) > 0 else np.nan
        return bool(ok), rvec, tvec, reproj_mean
    except Exception as exc:
        print(f"[warn] EMPnP failed: {exc}")
        return False, None, None, np.nan


def parse_args():
    ap = argparse.ArgumentParser(description="RealSense D435 RGB + UR10e joint live inference")
    ap.add_argument("--checkpoint", default=UR10E_DEFAULT_CHECKPOINT)
    ap.add_argument("--robot_ip", type=str, default=None, help="UR controller IP for RTDE getActualQ()")
    ap.add_argument("--joints", type=float, nargs=6, default=None, help="Fallback fixed joint values in radians")
    ap.add_argument("--image_h", type=int, default=480)
    ap.add_argument("--image_w", type=int, default=640)
    ap.add_argument("--camera_w", type=int, default=640)
    ap.add_argument("--camera_h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--seg_thr", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.99)
    ap.add_argument("--max_pixels", type=int, default=4000)
    ap.add_argument("--num_hypotheses", type=int, default=8)
    ap.add_argument("--weight_by_segprob", action="store_true", default=True)
    ap.add_argument("--min_valid_pixels", type=int, default=50)
    ap.add_argument("--use_angle_binning", action="store_true", default=True)
    ap.add_argument("--no_use_angle_binning", action="store_false", dest="use_angle_binning")
    ap.add_argument("--angle_bin_deg", type=float, default=3.0)
    ap.add_argument("--empnp_sigma", type=float, default=3.0)
    ap.add_argument("--empnp_lambda", type=float, default=0.2)
    ap.add_argument("--empnp_prior", type=float, default=0.9)
    ap.add_argument("--empnp_max_iter", type=int, default=50)
    ap.add_argument("--empnp_tol", type=float, default=1e-6)
    ap.add_argument("--empnp_use_prev", action="store_true", default=True)
    ap.add_argument("--no_empnp_use_prev", action="store_false", dest="empnp_use_prev")
    ap.add_argument("--axis_len", type=float, default=0.1)
    ap.add_argument("--ros_pose_topic", type=str, default="", help="Publish EMPnP pose as geometry_msgs/PoseStamped when set")
    ap.add_argument("--ros_frame_id", type=str, default="camera_color_optical_frame", help="frame_id for the published PnP pose")
    ap.add_argument("--ros_reproj_topic", type=str, default="", help="Optional std_msgs/Float32 topic for EMPnP reprojection mean")
    ap.add_argument("--show_seg", action="store_true", help="Start with the segmentation debug window open")
    ap.add_argument("--window", type=str, default="RobotPose D435 UR10e")
    ap.add_argument("--seg_window", type=str, default="RobotPose UR10e Segmentation")
    return ap.parse_args()


def main():
    args = parse_args()
    stream = D435ColorStream(args.camera_w, args.camera_h, args.fps)
    joint_reader = URJointReader(args.robot_ip, args.joints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotJointDetection().to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        amp_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    tf = make_transform(args.image_h, args.image_w, normalize=not args.no_normalize)
    vote_cfg = PVNetVoteConfig(
        seg_thr=args.seg_thr,
        theta=args.theta,
        max_pixels=args.max_pixels,
        num_hypotheses=args.num_hypotheses,
        weight_by_segprob=args.weight_by_segprob,
        min_valid_pixels=args.min_valid_pixels,
        use_angle_binning=args.use_angle_binning,
        angle_bin_deg=int(round(args.angle_bin_deg)),
    )
    K = stream.K.copy()
    dist = stream.dist.copy()
    prev_rvec = None
    prev_tvec = None
    prev_points_2d = None
    fps_ema = None
    last_time = time.perf_counter()
    show_seg = bool(args.show_seg)
    ros_pub = None

    try:
        if args.ros_pose_topic:
            ros_pub = RosPosePublisher(
                args.ros_pose_topic,
                args.ros_frame_id,
                reproj_topic=args.ros_reproj_topic or None,
            )

        print("[info] press q or ESC to quit, s to toggle segmentation window")
        while True:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_now = 1.0 / dt
                fps_ema = fps_now if fps_ema is None else (0.9 * fps_ema + 0.1 * fps_now)

            bgr = stream.read()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            joints = joint_reader.read()
            points_3d = ur10e_keypoints_from_joints(joints)
            points_2d, covs_2d, seg_probs = infer_keypoints(
                model, rgb, tf, vote_cfg, device, amp_ctx, return_seg_probs=show_seg,
            )

            ok, rvec, tvec, reproj_mean = solve_pose_empnp(
                points_3d,
                points_2d,
                covs_2d,
                K,
                dist,
                args,
                prev_rvec,
                prev_tvec,
                prev_points_2d,
            )
            vis = draw_keypoints(bgr, points_2d, DEFAULT_KEY_NAMES)
            if ok:
                vis = draw_pose_axes(vis, K, dist, rvec, tvec, args.axis_len, thickness=3)
                if ros_pub is not None:
                    ros_pub.publish(rvec, tvec, reproj_mean)
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
                prev_points_2d = points_2d.copy()
                cv2.putText(vis, "EMPnP OK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, "EMPnP FAIL", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2, cv2.LINE_AA)

            if fps_ema is not None:
                cv2.putText(vis, f"FPS {fps_ema:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(args.window, vis)
            if show_seg:
                seg_vis = make_segmentation_debug_bgr(bgr, seg_probs, DEFAULT_KEY_NAMES, args.seg_thr)
                cv2.imshow(args.seg_window, seg_vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                show_seg = not show_seg
                if not show_seg:
                    try:
                        cv2.destroyWindow(args.seg_window)
                    except cv2.error:
                        pass
    finally:
        stream.stop()
        if ros_pub is not None:
            ros_pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
