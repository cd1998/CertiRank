"""Focused regression tests for the three-dataset FRL/CMGRA-PX benchmark."""

import unittest

import numpy as np
import torch

from benchmark import ALL_METHODS, DATASETS, RANK_METHODS, attack_applies, method_family
from benchmark.aggregators import bcpbfl, fedavg, rvpfl
from benchmark.manifest import build_tasks
from benchmark.finite_grad_ascent_campaign import (
    build_tasks as build_finite_gradient_ascent_tasks,
    validate_tasks as validate_finite_gradient_ascent_tasks,
)
from benchmark.supplement_campaign import build_tasks as build_supplement_tasks
from benchmark.supplement_campaign import validate_tasks
from benchmark.paper_backdoor_campaign import (
    build_tasks as build_paper_backdoor_tasks,
    validate_tasks as validate_paper_backdoor_tasks,
)
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
            ALL_METHODS, ("fedavg", "bcpbfl", "rvpfl", "frl", "cmgra-px")
        )
        for method in ("fedavg", "bcpbfl", "rvpfl"):
            self.assertEqual(method_family(method), "gradient")
        self.assertEqual(method_family("frl"), "rank")
        self.assertEqual(method_family("cmgra-px"), "rank")

    def test_rank_attacks_apply_to_both_methods(self):
        for method in ALL_METHODS:
            self.assertTrue(attack_applies(method, "clean"))
            self.assertTrue(attack_applies(method, "label_flip"))
            self.assertTrue(attack_applies(method, "label_flip_all_to_one"))
            self.assertTrue(attack_applies(method, "pixel_backdoor_low_data"))
        for method in RANK_METHODS:
            self.assertTrue(attack_applies(method, "vem"))
        for method in ("fedavg", "bcpbfl", "rvpfl"):
            self.assertFalse(attack_applies(method, "vem"))

    def test_manifest_contains_no_retired_dataset_or_method(self):
        tasks = build_tasks(rounds=3, seed=7)
        self.assertTrue(tasks)
        self.assertEqual({task["dataset"] for task in tasks}, set(DATASETS))
        self.assertEqual({task["method"] for task in tasks}, set(ALL_METHODS))

    def test_supplement_campaign_is_complete_and_unique(self):
        tasks = build_supplement_tasks(rounds=500, seed=0)
        validate_tasks(tasks)
        self.assertEqual(len(tasks), 246)
        vem = [task for task in tasks if task["attack"] == "vem"]
        common = [task for task in tasks if task["attack"] != "vem"]
        self.assertEqual(len(vem), 6)
        self.assertEqual(len(common), 240)
        self.assertEqual(
            {task["malicious_fraction"] for task in vem}, {0.1, 0.3, 0.4}
        )

    def test_paper_backdoor_campaign_is_complete_and_explicit(self):
        tasks = build_paper_backdoor_tasks(rounds=500, seed=0)
        validate_paper_backdoor_tasks(tasks)
        self.assertEqual(len(tasks), 60)
        for task in tasks:
            config = parse_args(task["arguments"])
            self.assertEqual(config.backdoor_target, 2)
            self.assertEqual(config.backdoor_examples, 9)

    def test_finite_gradient_ascent_campaign_is_complete(self):
        tasks = build_finite_gradient_ascent_tasks(rounds=500, seed=0)
        validate_finite_gradient_ascent_tasks(tasks)
        self.assertEqual(len(tasks), 9)
        self.assertEqual({task["dataset"] for task in tasks}, {"svhn"})
        self.assertEqual(
            {task["method"] for task in tasks},
            {"fedavg", "bcpbfl", "rvpfl"},
        )
        self.assertEqual(
            {task["malicious_fraction"] for task in tasks},
            {0.1, 0.2, 0.3},
        )

    def test_paper_f_trigger_is_top_left(self):
        images = torch.zeros(1, 1, 28, 28)
        stamped = stamp_trigger(images, "mnist", 5, pattern="paper-f")
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
                "--method", "cmgra-px",
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

    def test_all_to_one_label_flip_target_is_explicit(self):
        config = parse_args(
            [
                "--dataset", "mnist",
                "--method", "frl",
                "--attack", "label_flip_all_to_one",
                "--malicious-fraction", "0.2",
                "--label-flip-target", "3",
                "--rounds", "1",
                "--device", "cpu",
            ]
        )
        self.assertEqual(config.label_flip_target, 3)


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
