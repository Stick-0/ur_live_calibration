"""Open the dual-arm GUI with synthetic poses, without cameras or robots.

Run ``python3 -m urcal.tools.preview_gui --mode follow`` (or ``hold``).
The animated joints and camera/base transforms are invented display fixtures,
not a dynamics simulation or reachability/collision validation. Apply updates
only the displayed target; motion arming and task pause are always disabled.
Tk requires a graphical display. No model checkpoints are loaded.
"""

import argparse
from copy import deepcopy
import threading
import time

import numpy as np

from urcal.control.geometry import facing_follower, fixed_facing_targets, inverse, matrix_pose, pose_matrix


DEMO_NOTICE = "DEMO — synthetic poses only. No camera or robot connection. Apply changes display targets only."


class DemoState:
    """Small synthetic state source with no receiver or control interface."""

    def __init__(self, mode="follow", now=None):
        from urcal.control.runtime import parse_args
        from urcal.cooperative.kinematics import NominalKinematics

        if mode not in ("hold", "follow"):
            raise ValueError("mode must be hold or follow")
        self.args = parse_args(["--mode", mode, "--ur5e_ip", "offline-demo5", "--ur10e_ip", "offline-demo10"])
        self.args.execute = False
        self.started_at = time.monotonic() if now is None else float(now)
        self.models = {name: NominalKinematics(name) for name in ("ur5e", "ur10e")}
        self.joints = {
            "ur5e": np.array([-.61942049, .22685986, .50310870, -.00980895, .89066485, -.97300499]),
            "ur10e": np.array([-1.20260624, .19983087, .75013005, 1.30345049, -1.54067765, .96522864]),
        }
        self.offsets = {name: pose_matrix([0, 0, length, 0, 0, 0])
                        for name, length in (("ur5e", .035), ("ur10e", .055))}
        leader = pose_matrix([.15, 0, 1.4, 0, -np.pi/2, 0])
        flanges = {"ur10e": leader, "ur5e": facing_follower(leader, .30)}
        self.bases = {name: flanges[name] @ inverse(self.models[name].pose_and_jacobian(q, flange=True)[0])
                      for name, q in self.joints.items()}
        center = flanges["ur5e"].copy()
        center[:3, 3] = (flanges["ur5e"][:3, 3] + flanges["ur10e"][:3, 3]) / 2 + [0, -.04, 0]
        self.center_pose = matrix_pose(center).tolist()
        self.request = {"type": "target", "gap": .30}
        if mode == "hold":
            self.request["center_pose"] = self.center_pose
        else:
            self.request.update(path="hold", path_delta=[0, 0, -.05, 0, 0, 0],
                                path_radius=.02, path_duration=10.)
        self.applied = 0

    def apply(self, command):
        """Accept display target requests only; never forward a hardware command."""
        if not isinstance(command, dict) or command.get("type") != "target":
            raise ValueError("DEMO accepts display targets only; motion and pause are unavailable")
        gap = float(command["gap"])
        if not np.isfinite(gap) or gap < self.args.min_gap:
            raise ValueError("Invalid demo flange gap")
        request = deepcopy(command)
        if self.args.mode == "hold":
            center = matrix_pose(pose_matrix(request["center_pose"])).tolist()
            if center[2] <= 0:
                raise ValueError("Camera Z must be greater than zero")
            self.center_pose = center
            request["center_pose"] = center
        else:
            from urcal.cooperative.task import RelativePath
            RelativePath(facing_follower(np.eye(4), gap), kind=request.get("path", "hold"),
                         delta=request.get("path_delta", [0.]*6), radius=request.get("path_radius", .02),
                         duration=request.get("path_duration", 10.))
        self.request = request
        self.applied += 1

    def status(self, now=None):
        """Return a fresh paired display snapshot with slowly animated joints."""
        now = time.monotonic() if now is None else float(now)
        elapsed = now - self.started_at
        robots, flanges = {}, {}
        for index, (name, initial) in enumerate(self.joints.items()):
            q = initial.copy()
            q[1] += .025 * np.sin(.7 * elapsed + index * .6)
            q[3] += .035 * np.sin(.5 * elapsed + index)
            flange, _ = self.models[name].pose_and_jacobian(q, flange=True)
            robots[name] = {"q": q, "tcp": flange @ self.offsets[name],
                            "tcp_offset": self.offsets[name].copy()}
            flanges[name] = self.bases[name] @ flange
        if self.args.mode == "hold":
            goals = fixed_facing_targets(flanges["ur5e"], flanges["ur10e"], self.request["gap"],
                                        self.center_pose)
            task = None
        else:
            goals = {"ur5e": facing_follower(flanges["ur10e"], self.request["gap"])}
            task = {"leader": "ur10e", "kind": self.request.get("path", "hold"),
                    "gap": self.request["gap"], "delta": self.request.get("path_delta", [0.]*6),
                    "radius": self.request.get("path_radius", .02),
                    "duration": self.request.get("path_duration", 10.)}
        snapshot = {"at": now, "bases": {name: base.copy() for name, base in self.bases.items()},
                    "robots": robots, "goals": goals, "measurement_age": 0., "calibration_valid": True,
                    "reason": "Synthetic display fixture; no measured calibration", "task": task}
        return {"scene": snapshot, "active": False, "ready": False, "center_pose": self.center_pose,
                "message": DEMO_NOTICE + (f" Applied {self.applied} display target(s)." if self.applied else ""),
                "task_paused": False, "progress": 0., "task_scale": 0., "measurement_age": 0.}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("hold", "follow"), default="follow")
    options = parser.parse_args(argv)
    from tkinter import ttk
    from urcal.control.gui import TargetPanel

    demo = DemoState(options.mode)
    stop = threading.Event()
    panel = TargetPanel(demo.args, demo.apply, stop)
    panel.root.title("DEMO — synthetic dual-arm poses — no hardware")
    ttk.Label(panel.root, text=DEMO_NOTICE, foreground="#8a4100", padding=8,
              font=("TkDefaultFont", 11, "bold"), wraplength=950).grid(row=1, column=0, sticky="ew")
    panel._view._models = demo.models
    print(DEMO_NOTICE, flush=True)
    try:
        while not stop.is_set():
            panel.poll(demo.status())
            stop.wait(.02)
    except KeyboardInterrupt:
        stop.set()
    finally:
        panel.close()


if __name__ == "__main__":
    main()
