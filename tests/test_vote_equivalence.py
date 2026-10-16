"""Exact voting regressions against the unchanged reference implementation.

Run from the bundle: python -m unittest discover -s tests -v
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through this entry.
    spec.loader.exec_module(module)
    return module


REFERENCE = load_module("robotpose_vote_reference_test", BUNDLE / "reference" / "vote2.py")
from urcal.inference import vote2 as OPTIMIZED
DTYPES = (torch.float16, torch.float32, torch.float64)


def fixture(dtype, device, scenario="dense", batch=1, keypoints=7):
    generator = torch.Generator().manual_seed(3829)
    height, width = 16, 20
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xy = torch.stack((x, y), dim=0).to(torch.float64)
    targets = torch.stack(
        (torch.linspace(3.2, 16.1, keypoints), torch.linspace(3.4, 12.2, keypoints)), dim=1
    ).to(torch.float64)
    vec = targets[:, :, None, None] - xy[None]
    vec = vec[None].repeat(batch, 1, 1, 1, 1)
    seg = torch.randn(batch, keypoints, height, width, generator=generator) + 1.0

    if scenario == "ties":
        seg.zero_()
    elif scenario == "empty":
        seg.fill_(-8.0)
    elif scenario == "neighbors":
        seg.fill_(-8.0)
        # Channel 1 can donate, but has fewer than min_valid_pixels (50).
        # Its top-k fallback must not modify the cached 35-pixel mask.
        seg[:, 1, :5, :7] = 4.0
        seg[:, 4, :8, :8] = 3.0
        seg[:, 6, :2, :2] = 2.0
    elif scenario == "nonfinite":
        seg[:, :, 0, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
        vec[:, :, :, 1, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
        vec[:, :, :, 2, :3] = 0.0
    elif scenario == "zero_vectors":
        vec.zero_()

    return seg.to(device=device, dtype=dtype), vec.to(device=device, dtype=dtype)


class VoteEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_nested_exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assert_nested_exact(got, want)
        else:
            self.assertEqual(actual, expected)

    def compare_vote(self, dtype, device, scenario, seed, overrides=None, batch=1, debug=True):
        seg, vec = fixture(dtype, device, scenario, batch=batch)
        before = (seg.clone(), vec.clone())
        config = {"num_hypotheses": 8}
        config.update(overrides or {})
        results, rng_states = [], []
        for module in (REFERENCE, OPTIMIZED):
            torch.manual_seed(seed)
            results.append(module.pvnet_vote(seg, vec, module.PVNetVoteConfig(**config), return_debug=debug))
            state = [torch.random.get_rng_state()]
            if torch.device(device).type == "cuda":
                state.append(torch.cuda.get_rng_state(device))
            rng_states.append(state)
        self.assert_nested_exact(results[1], results[0])
        self.assert_nested_exact(rng_states[1], rng_states[0])
        self.assert_nested_exact((seg, vec), before)

    def compare_candidates(self, dtype, device, weight_by_prob, angle_bin_deg, scenario):
        seg, vec = fixture(dtype, device, "ties")
        vectors = vec[0, 0].clone()
        prob = torch.full_like(seg[0, 0], 0.5)
        # Mixed exact and fp16-rounded ties, with repeated directions.
        prob[::2] = 0.5001
        prob[:, ::3] = 0.75
        vectors[:, :, :3] = torch.tensor([1.0, 0.0], dtype=dtype, device=device)[:, None, None]
        mask = torch.ones_like(prob, dtype=torch.bool)
        if scenario == "empty":
            mask.zero_()
        elif scenario == "nonfinite_vectors":
            vectors[:, 0, :4] = torch.tensor(
                [torch.nan, torch.inf, -torch.inf, 0.0], dtype=dtype, device=device
            )
        elif scenario == "nonfinite_weights":
            prob[0, :6] = torch.tensor(
                [torch.nan, 0.9, torch.nan, torch.inf, -torch.inf, 0.0], dtype=dtype, device=device
            )
        elif scenario == "zero_vectors":
            vectors.zero_()
        args = (vectors, prob, mask, angle_bin_deg, weight_by_prob)
        expected = REFERENCE._reduce_pixels_by_angle(*args)
        actual = OPTIMIZED._reduce_pixels_by_angle(*args)
        self.assert_nested_exact(actual, expected)

    def test_candidate_order_ties_and_edge_cases_cpu(self):
        for dtype in DTYPES:
            for weighted in (False, True):
                for scenario in ("ties", "empty", "nonfinite_vectors", "nonfinite_weights", "zero_vectors"):
                    with self.subTest(dtype=dtype, weighted=weighted, scenario=scenario):
                        self.compare_candidates(dtype, "cpu", weighted, 3, scenario)

    def test_candidate_angle_sizes_cpu(self):
        for degrees in (None, 0, 1, 7, 360, 500):
            with self.subTest(degrees=degrees):
                self.compare_candidates(torch.float32, "cpu", True, degrees, "ties")

    def test_vote_exact_outputs_and_rng_cpu(self):
        for dtype in DTYPES:
            for scenario in ("dense", "ties", "empty", "neighbors", "nonfinite", "zero_vectors"):
                for seed in (0, 71):
                    with self.subTest(dtype=dtype, scenario=scenario, seed=seed):
                        self.compare_vote(dtype, "cpu", scenario, seed)

    def test_vote_configurations_and_batches_cpu(self):
        configurations = (
            {"use_neighbor_pixel_trigger": False},
            {"neighbor_max_hops": 1},
            {"neighbor_max_hops": 0},
            {"weight_by_segprob": False},
            {"use_angle_binning": False, "max_pixels": 20},
            {"num_hypotheses": 32, "angle_bin_deg": 90, "min_hypotheses": 1, "min_effective_hyps": 1},
        )
        for dtype in DTYPES:
            for config in configurations:
                with self.subTest(dtype=dtype, config=config):
                    self.compare_vote(dtype, "cpu", "neighbors", 19, config, batch=2)
        self.compare_vote(torch.float32, "cpu", "dense", 51, debug=False)

    def test_noncontiguous_inputs_cpu(self):
        for dtype in DTYPES:
            seg, vec = fixture(dtype, "cpu", "dense", batch=2)
            seg = seg.transpose(-1, -2)
            vec = vec.transpose(-1, -2)
            outputs = []
            for module in (REFERENCE, OPTIMIZED):
                torch.manual_seed(28)
                outputs.append(module.pvnet_vote(seg, vec, module.PVNetVoteConfig(num_hypotheses=8), True))
            self.assert_nested_exact(outputs[1], outputs[0])

    def test_channels_last_inputs_cpu(self):
        for dtype in DTYPES:
            seg, vec = fixture(dtype, "cpu", "neighbors", batch=2)
            seg = seg.contiguous(memory_format=torch.channels_last)
            outputs = []
            for module in (REFERENCE, OPTIMIZED):
                torch.manual_seed(28)
                outputs.append(module.pvnet_vote(seg, vec, module.PVNetVoteConfig(num_hypotheses=8), True))
            self.assert_nested_exact(outputs[1], outputs[0])

    def test_existing_sparse_transpose_error_preserved_cpu(self):
        seg, vec = fixture(torch.float32, "cpu", "neighbors")
        seg, vec = seg.transpose(-1, -2), vec.transpose(-1, -2)
        for module in (REFERENCE, OPTIMIZED):
            with self.assertRaisesRegex(RuntimeError, "view size is not compatible"):
                module.pvnet_vote(seg, vec, module.PVNetVoteConfig(num_hypotheses=8), True)

    def test_zero_keypoints_cpu(self):
        seg = torch.empty(1, 0, 8, 8)
        vec = torch.empty(1, 0, 2, 8, 8)
        self.assert_nested_exact(
            OPTIMIZED.pvnet_vote(seg, vec, OPTIMIZED.PVNetVoteConfig(), True),
            REFERENCE.pvnet_vote(seg, vec, REFERENCE.PVNetVoteConfig(), True),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_exact_candidates_outputs_and_rng(self):
        for dtype in DTYPES:
            for weighted in (False, True):
                for scenario in ("ties", "nonfinite_vectors", "nonfinite_weights"):
                    with self.subTest(dtype=dtype, weighted=weighted, candidate_scenario=scenario):
                        self.compare_candidates(dtype, "cuda", weighted, 3, scenario)
            for scenario in ("dense", "ties", "neighbors", "nonfinite"):
                for seed in (0, 71):
                    with self.subTest(dtype=dtype, scenario=scenario, seed=seed):
                        self.compare_vote(dtype, "cuda", scenario, seed)


if __name__ == "__main__":
    unittest.main()
