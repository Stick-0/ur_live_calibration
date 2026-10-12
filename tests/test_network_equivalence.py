"""Compare optimized attention blocks against the preserved original code.

Run from ur_live_calibration:
  python -m unittest discover -s tests -v
  python tests/test_network_equivalence.py --benchmark
"""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import statistics
import sys
import time
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL = load_module("network_original", ROOT / "reference" / "network3.py")
OPTIMIZED = load_module("network_optimized", ROOT / "network3.py")
BLOCKS = ("TransformerEncoder", "TransformerEncoder_Split")


def pair(name, width, device="cpu", training=False):
    kwargs = {"embed_dim": width}
    if name == "TransformerEncoder":
        kwargs["num_heads"] = 12
    original = getattr(ORIGINAL, name)(**kwargs).to(device).train(training)
    optimized = getattr(OPTIMIZED, name)(**kwargs).to(device).train(training)
    optimized.load_state_dict(original.state_dict(), strict=True)
    return original, optimized


def outputs(value):
    return value if isinstance(value, tuple) else (value,)


class NetworkEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_outputs_equal(self, expected, actual):
        self.assertEqual(len(outputs(expected)), len(outputs(actual)))
        for left, right in zip(outputs(expected), outputs(actual)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_eval_fp32_exactly_matches_original(self):
        for name in BLOCKS:
            for width, batch in ((48, 2), (768, 1)):
                with self.subTest(block=name, width=width, batch=batch):
                    torch.manual_seed(17)
                    original, optimized = pair(name, width)
                    # Fifteen tokens is the actual live-model sequence length.
                    # Include a noncontiguous input, as feature layouts can vary.
                    for seed in (3, 29):
                        torch.manual_seed(seed)
                        x = torch.randn(batch, width, 15).transpose(1, 2)
                        with torch.no_grad():
                            self.assert_outputs_equal(original(x), optimized(x))

    def test_state_dict_is_unchanged(self):
        for name in BLOCKS:
            with self.subTest(block=name):
                original, optimized = pair(name, 48)
                self.assertEqual(list(original.state_dict()), list(optimized.state_dict()))
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, optimized.state_dict()[key], rtol=0, atol=0)

    def test_attention_input_identity_and_normalization_work(self):
        for name, old_count, new_count in (
            ("TransformerEncoder", 3, 1),
            ("TransformerEncoder_Split", 57, 19),
        ):
            with self.subTest(block=name):
                original, optimized = pair(name, 48)
                signatures = []
                counts = []
                for block in (original, optimized):
                    identities = []
                    normalizations = []
                    hooks = []

                    def record_identity(module, args):
                        q, k, v = args
                        identities.append((q is k, k is v, q is v))

                    def record_norm(module, args):
                        normalizations.append(1)

                    for module_name, module in block.named_modules():
                        if isinstance(module, torch.nn.MultiheadAttention):
                            hooks.append(module.register_forward_pre_hook(record_identity))
                        elif module_name.startswith("norm1"):
                            hooks.append(module.register_forward_pre_hook(record_norm))
                    try:
                        with torch.no_grad():
                            block(torch.randn(1, 15, 48))
                    finally:
                        for hook in hooks:
                            hook.remove()
                    signatures.append(identities)
                    counts.append(len(normalizations))
                self.assertEqual(signatures[0], signatures[1])
                self.assertTrue(all(flags == (False, False, False) for flags in signatures[1]))
                self.assertEqual(counts, [old_count, new_count])

    def test_training_output_and_gradients_are_unchanged(self):
        for name in BLOCKS:
            with self.subTest(block=name):
                torch.manual_seed(19)
                original, optimized = pair(name, 24, training=True)
                source = torch.randn(1, 15, 24)
                original_input = source.clone().requires_grad_()
                optimized_input = source.clone().requires_grad_()
                torch.manual_seed(23)
                expected = original(original_input)
                torch.manual_seed(23)
                actual = optimized(optimized_input)
                self.assert_outputs_equal(expected, actual)
                sum(value.square().mean() for value in outputs(expected)).backward()
                sum(value.square().mean() for value in outputs(actual)).backward()
                torch.testing.assert_close(original_input.grad, optimized_input.grad, rtol=0, atol=0)
                for left, right in zip(original.parameters(), optimized.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_autocast_exactly_matches_original(self):
        for name in BLOCKS:
            with self.subTest(block=name):
                torch.manual_seed(17)
                original, optimized = pair(name, 768, device="cuda")
                for seed in (3, 29):
                    torch.manual_seed(seed)
                    x = torch.randn(1, 15, 768, device="cuda")
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        self.assert_outputs_equal(original(x), optimized(x))
                del original, optimized


def benchmark():
    """Small paired block benchmark; timings are observations, not test assertions."""
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; batch=1; tokens=15; width=768; CPU threads=1")
    for name in BLOCKS:
        torch.manual_seed(17)
        blocks = pair(name, 768, device=device)
        x = torch.randn(1, 15, 768, device=device)
        timings = [[], []]
        amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            for block in blocks:
                for _ in range(3):
                    block(x)
            for repeat in range(20):
                # Alternate order to reduce bias from warmup and changing clocks.
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    blocks[index](x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    timings[index].append((time.perf_counter() - start) * 1000)
        old_ms, new_ms = (statistics.median(values) for values in timings)
        print(f"{name}: original={old_ms:.3f} ms; optimized={new_ms:.3f} ms; ratio={old_ms / new_ms:.3f}x")


if __name__ == "__main__":
    if sys.argv[1:] == ["--benchmark"]:
        benchmark()
    else:
        unittest.main()
