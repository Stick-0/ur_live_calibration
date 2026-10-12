"""Live postprocessing and option compatibility without connecting to hardware."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


BUNDLE = Path(__file__).resolve().parents[1]


def load_live(directory, script, label):
    # Each live script uses absolute imports; isolate its dependency set so the
    # reference really uses the reference voter/PnP/network.
    names = ("vote2", "my_pnp", "network3")
    saved = {name: sys.modules.pop(name, None) for name in names}
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location(label, directory / script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


class FixedPrediction(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        generator = torch.Generator().manual_seed(731)
        self.seg = (torch.randn(1, 7, 48, 64, generator=generator) * 3).to(device)
        yy, xx = torch.meshgrid(torch.arange(48), torch.arange(64), indexing="ij")
        points = torch.tensor([[9, 8], [17, 13], [24, 19], [31, 23], [36, 29], [44, 34], [52, 40]])
        vectors = points[:, :, None, None] - torch.stack([xx, yy])[None]
        self.vec = torch.nn.functional.normalize(vectors.float(), dim=1)[None].to(device)

    def forward(self, image):
        return self.seg, self.vec


class LiveEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.modules = []
        for script in ("infer_live.py", "infer_live_ur10e.py"):
            cls.modules.append((
                load_live(BUNDLE / "reference", script, "reference_" + script[:-3]),
                load_live(BUNDLE, script, "optimized_" + script[:-3]),
            ))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_existing_cli_defaults_and_fk_unchanged(self):
        for original, optimized in self.modules:
            with self.subTest(robot=original.DEFAULT_KEY_NAMES[0]):
                with patch.object(sys, "argv", ["live", "--checkpoint", "test.pt"]):
                    self.assertEqual(vars(original.parse_args()), vars(optimized.parse_args()))
                fk_name = "ur10e_keypoints_from_joints" if "ur10e" in original.DEFAULT_KEY_NAMES[0] else "ur5e_keypoints_from_joints"
                joints = [0.2, -1.2, 1.1, -1.5, -1.2, 0.4]
                np.testing.assert_array_equal(getattr(original, fk_name)(joints), getattr(optimized, fk_name)(joints))

    def compare_inference(self, device):
        image = np.random.default_rng(43).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        model = FixedPrediction(device)
        for original, optimized in self.modules:
            tf = original.make_transform(48, 64, normalize=True)
            cfg = original.PVNetVoteConfig(num_hypotheses=8)
            torch.manual_seed(77)
            expected = original.infer_keypoints(model, image, tf, cfg, device, nullcontext())
            expected_rng = torch.get_rng_state()
            expected_cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            for debug in (True, False):
                with self.subTest(robot=original.DEFAULT_KEY_NAMES[0], debug=debug, device=str(device)):
                    torch.manual_seed(77)
                    actual = optimized.infer_keypoints(
                        model, image, tf, cfg, device, nullcontext(), return_seg_probs=debug,
                    )
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    if debug:
                        np.testing.assert_array_equal(actual[2], expected[2])
                    else:
                        self.assertIsNone(actual[2])
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    if expected_cuda_rng is not None:
                        self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), expected_cuda_rng))

    def test_cpu_debug_toggle_preserves_pose_inputs(self):
        self.compare_inference(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not accessible")
    def test_cuda_debug_toggle_preserves_pose_inputs(self):
        self.compare_inference(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
