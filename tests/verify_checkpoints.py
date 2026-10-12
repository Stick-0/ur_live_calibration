"""Compare full trained-model outputs on a supplied image, without hardware I/O.

This is an implementation-equivalence check, not a pose accuracy evaluation or
an FPS benchmark. CUDA uses the live programs' FP16/channels-last settings.
"""
import argparse
from contextlib import nullcontext
import gc
import importlib.util
from pathlib import Path

import cv2
import torch
from torchvision import transforms


BUNDLE = Path(__file__).resolve().parents[1]


def load_network(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--robot", choices=("ur5e", "ur10e", "both"), default="both")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not accessible in this Python environment")

    torch.set_num_threads(2)
    torch.hub.set_dir(str(BUNDLE / "torch_cache" / "hub"))
    device = torch.device(args.device)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        parser.error(f"Cannot read image: {args.image}")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        x = x.contiguous(memory_format=torch.channels_last)

    networks = (
        load_network("checkpoint_reference", BUNDLE / "reference" / "network3.py"),
        load_network("checkpoint_optimized", BUNDLE / "network3.py"),
    )
    for robot, folder in (("ur5e", "v3m2a3"), ("ur10e", "ur10e_2")):
        if args.robot not in (robot, "both"):
            continue
        checkpoint = torch.load(
            BUNDLE / "trained_models" / folder / "best.pt",
            map_location="cpu", weights_only=True, mmap=True,
        )
        outputs = []
        for module in networks:
            model = module.RobotJointDetection().eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            model = model.to(device)
            if device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)
            amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with torch.no_grad(), amp:
                outputs.append(tuple(value.cpu() for value in model(x)))
            del model
            gc.collect()
        for name, expected, actual in zip(("seg", "vec"), *outputs):
            difference = (expected.float() - actual.float()).abs().max().item()
            print(f"{robot} {name}: max_abs_difference={difference}", flush=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del checkpoint, outputs
        gc.collect()
    print("PASS: all selected checkpoint outputs exactly match the reference.")


if __name__ == "__main__":
    main()
