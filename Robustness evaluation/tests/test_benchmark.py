"""Focused regression tests for the three-dataset FRL/CMGRA benchmark."""

import unittest

import numpy as np
import torch

from benchmark import (
    ALL_METHODS,
    DATASETS,
    COMMON_ATTACKS,
    RANK_METHODS,
    attack_applies,
    method_family,
)
from benchmark.aggregators import bcpbfl, fedavg, rvpfl
from benchmark.manifest import build_tasks, validate_tasks
from benchmark.data import FixedBackdoorDataset, stamp_trigger
from benchmark.models import build_model, communicated_parameter_count
from benchmark.runner import (
    _rank_values,
    _vem_source_rank_values,
    parse_args,
    round_malicious_count,
    selected_clients,
)
from benchmark.vem_partition import vem_dirichlet_partition


class RegistryTests(unittest.TestCase):
    def test_only_requested_datasets_and_methods_are_exposed(self):
        self.assertEqual(DATASETS, ("mnist", "svhn", "cifar10"))
        self.assertEqual(
            ALL_METHODS, ("fedavg", "bcpbfl", "rvpfl", "frl", "cmgra")
        )
        for method in ("fedavg", "bcpbfl", "rvpfl"):
            self.assertEqual(method_family(method), "gradient")
        self.assertEqual(method_family("frl"), "rank")
        self.assertEqual(method_family("cmgra"), "rank")

    def test_rank_attacks_apply_to_both_methods(self):
        for method in ALL_METHODS:
            self.assertTrue(attack_applies(method, "clean"))
            self.assertTrue(attack_applies(method, "label_flip"))
            self.assertTrue(attack_applies(method, "grad_ascent"))
            self.assertTrue(attack_applies(method, "pixel_backdoor"))
        for method in RANK_METHODS:
            self.assertTrue(attack_applies(method, "vem"))
        for method in ("fedavg", "bcpbfl", "rvpfl"):
            self.assertFalse(attack_applies(method, "vem"))

    def test_manifest_contains_no_retired_dataset_or_method(self):
        tasks = build_tasks(rounds=3, seed=7)
        validate_tasks(tasks)
        self.assertTrue(tasks)
        self.assertEqual(len(tasks), 168)
        self.assertEqual({task["dataset"] for task in tasks}, set(DATASETS))
        self.assertEqual({task["method"] for task in tasks}, set(ALL_METHODS))
        self.assertEqual(
            {task["malicious_fraction"] for task in tasks if task["attack"] != "clean"},
            {0.1, 0.2, 0.3},
        )
        self.assertEqual(
            {task["attack"] for task in tasks},
            {"clean", *COMMON_ATTACKS, "vem"},
        )
        for task in tasks:
            config = parse_args(task["arguments"])
            self.assertEqual(config.n_clients, 1000)
            self.assertEqual(config.round_clients, 25)
            self.assertEqual(config.backdoor_target, 2)
            self.assertEqual(config.backdoor_examples, 9)

    def test_paper_f_trigger_is_top_left(self):
        images = torch.zeros(1, 1, 28, 28)
        stamped = stamp_trigger(images, "mnist", 5)
        self.assertGreater(float(stamped[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(stamped[0, 0, 2, 2]), 0.0)
        self.assertEqual(float(stamped[0, 0, -1, -1]), 0.0)

    def test_fixed_backdoor_labels_match_torchvision_integer_type(self):
        dataset = FixedBackdoorDataset(torch.zeros(3, 1, 28, 28), 2)
        image, label = dataset[1]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(label, 2)


class ModelTests(unittest.TestCase):
    def test_default_models_accept_each_dataset_shape(self):
        shapes = {
            "mnist": (2, 1, 28, 28),
            "svhn": (2, 3, 32, 32),
            "cifar10": (2, 3, 32, 32),
        }
        for dataset, shape in shapes.items():
            with self.subTest(dataset=dataset):
                model = build_model(dataset, rank_based=True)
                output = model(torch.randn(*shape))
                self.assertEqual(tuple(output.shape), (2, 10))
                self.assertGreater(communicated_parameter_count(model, True), 0)

    def test_paper_model_layer_counts_match_efficiency_profiles(self):
        expected = {
            "mnist": [576, 73728, 1605632, 2560],
            "svhn": [
                1728, 36864, 73728, 147456, 294912, 589824,
                1179648, 2359296, 524288, 65536, 2560,
            ],
            "cifar10": [
                1728, 36864, 36864, 36864, 36864,
                73728, 147456, 8192, 147456, 147456,
                294912, 589824, 32768, 589824, 589824,
                1179648, 2359296, 131072, 2359296, 2359296, 5120,
            ],
        }
        for dataset, counts in expected.items():
            with self.subTest(dataset=dataset):
                model = build_model(dataset, rank_based=True)
                actual = [
                    module.scores.numel()
                    for module in model.modules()
                    if hasattr(module, "scores")
                ]
                self.assertEqual(actual, counts)


class ProtocolTests(unittest.TestCase):
    def test_rank_conversion_is_a_permutation(self):
        scores = torch.tensor([0.2, -1.0, 4.0, 0.5])
        expected = torch.arange(scores.numel())
        self.assertTrue(torch.equal(torch.sort(_rank_values(scores)).values, expected))
        self.assertTrue(
            torch.equal(torch.sort(_vem_source_rank_values(scores)).values, expected)
        )

    def test_vem_partition_is_deterministic_and_complete(self):
        labels = np.repeat(np.arange(10), 20)
        first, _ = vem_dirichlet_partition(labels, 5, alpha=1.0, seed=9)
        second, _ = vem_dirichlet_partition(labels, 5, alpha=1.0, seed=9)
        self.assertEqual(len(first), 5)
        for client_id in range(5):
            np.testing.assert_array_equal(first[client_id], second[client_id])
        merged = np.concatenate([first[client_id] for client_id in range(5)])
        self.assertEqual(np.unique(merged).size, merged.size)
        self.assertGreaterEqual(merged.size, int(0.9 * labels.size))
        self.assertTrue(np.isin(merged, np.arange(labels.size)).all())

    def test_client_selection_is_reproducible(self):
        config = parse_args(
            [
                "--dataset", "mnist",
                "--method", "cmgra",
                "--attack", "vem",
                "--malicious-fraction", "0.2",
                "--n-clients", "1000",
                "--round-clients", "25",
                "--rounds", "5",
                "--seed", "3",
                "--device", "cpu",
            ]
        )
        selected_a, malicious_a = selected_clients(config, 2)
        selected_b, malicious_b = selected_clients(config, 2)
        np.testing.assert_array_equal(selected_a, selected_b)
        self.assertEqual(malicious_a, malicious_b)
        self.assertEqual(len(malicious_a), int(25 * 0.2))

    def test_cli_rejects_retired_dataset_and_method(self):
        base = ["--attack", "clean", "--malicious-fraction", "0"]
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "fashionmnist", "--method", "frl", *base])
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "mnist", "--method", "binary", *base])

    def test_pixel_backdoor_parameters_are_explicit(self):
        config = parse_args(
            [
                "--dataset", "mnist",
                "--method", "frl",
                "--attack", "pixel_backdoor",
                "--malicious-fraction", "0.2",
                "--backdoor-target", "2",
                "--backdoor-examples", "9",
                "--rounds", "1",
                "--device", "cpu",
            ]
        )
        self.assertEqual(config.backdoor_target, 2)
        self.assertEqual(config.backdoor_examples, 9)


class GradientAggregatorTests(unittest.TestCase):
    def test_fedavg(self):
        aggregate, diagnostics = fedavg(
            [[torch.tensor([1.0, 3.0])], [torch.tensor([3.0, 5.0])]]
        )
        self.assertTrue(torch.allclose(aggregate[0], torch.tensor([2.0, 4.0])))
        self.assertEqual(diagnostics.accepted, 2)

    def test_fedavg_rejects_a_nonfinite_update(self):
        updates = [
            [torch.tensor([1.0, 3.0])],
            [torch.tensor([float("nan"), 5.0])],
        ]
        aggregate, diagnostics = fedavg(updates)
        self.assertTrue(torch.equal(aggregate[0], updates[0][0]))
        self.assertEqual((diagnostics.accepted, diagnostics.rejected), (1, 1))

    def test_bcpbfl_rejects_negative_cosine(self):
        updates = [[torch.tensor([2.0, 0.0])], [torch.tensor([-7.0, 0.0])]]
        _, diagnostics = bcpbfl(updates, [torch.tensor([1.0, 0.0])], 1.0)
        self.assertEqual((diagnostics.accepted, diagnostics.rejected), (1, 1))

    def test_bcpbfl_rejects_a_nonfinite_update(self):
        updates = [
            [torch.tensor([2.0, 0.0])],
            [torch.tensor([float("nan"), 0.0])],
        ]
        aggregate, diagnostics = bcpbfl(
            updates, [torch.tensor([1.0, 0.0])], 1.0
        )
        self.assertTrue(torch.isfinite(aggregate[0]).all())
        self.assertEqual((diagnostics.accepted, diagnostics.rejected), (1, 1))

    def test_rvpfl_rejects_an_opposite_update(self):
        updates = [
            [torch.tensor([1.0, 2.0, 3.0, 4.0])],
            [torch.tensor([1.1, 2.1, 3.1, 4.1])],
            [torch.tensor([0.9, 1.9, 2.9, 3.9])],
            [torch.tensor([-9.0, -8.0, -7.0, -6.0])],
        ]
        _, diagnostics = rvpfl(updates)
        self.assertGreaterEqual(diagnostics.accepted, 3)
        self.assertLessEqual(diagnostics.similarities[-1], 0)

    def test_rvpfl_rejects_a_nonfinite_update(self):
        updates = [
            [torch.tensor([1.0, 2.0, 3.0, 4.0])],
            [torch.tensor([1.1, 2.1, 3.1, 4.1])],
            [torch.tensor([0.9, 1.9, 2.9, 3.9])],
            [torch.tensor([float("inf"), 0.0, 0.0, 0.0])],
        ]
        aggregate, diagnostics = rvpfl(updates)
        self.assertTrue(torch.isfinite(aggregate[0]).all())
        self.assertEqual(diagnostics.rejected, 1)


if __name__ == "__main__":
    unittest.main()
