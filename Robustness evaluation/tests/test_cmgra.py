"""Regression tests for the retained CMGRA-PX aggregation path."""

import unittest

import torch

from cmgra import CMGRAAggregator, CMGRAUpdateBatch, membership_from_rank_values


class CMGRATest(unittest.TestCase):
    @staticmethod
    def batch(rows, k):
        rankings = torch.tensor(rows, dtype=torch.long)
        return CMGRAUpdateBatch(rankings, membership_from_rank_values(rankings, k))

    def test_certified_membership_gates_higher_borda_outsider(self):
        batch = self.batch(
            [
                [5, 3, 4, 0, 1, 2],
                [5, 3, 4, 0, 1, 2],
                [5, 3, 4, 0, 1, 2],
                [5, 3, 4, 0, 1, 2],
                [4, 5, 0, 1, 2, 3],
            ],
            2,
        )
        server = CMGRAAggregator(2 / 6, max_malicious=1, public_seed=7)
        result = server.aggregate_layer("layer", batch)
        self.assertTrue(result.certified)
        self.assertEqual(result.boundary_margin, 3)
        self.assertEqual(result.membership.tolist(), [True, False, True, False, False, False])
        self.assertGreater(result.borda_scores[1], result.borda_scores[2])
        self.assertTrue(torch.equal(result.global_ranking >= 4, result.membership))

    def test_failed_whole_set_certificate_uses_certified_pair_swap(self):
        first = self.batch([[5, 3, 4, 0, 1, 2]] * 4 + [[4, 5, 0, 1, 2, 3]], 2)
        second = self.batch(
            [[1, 0, 2, 5, 4, 3], [0, 1, 2, 5, 4, 3], [4, 3, 2, 5, 1, 0]],
            2,
        )
        server = CMGRAAggregator(2 / 6, max_malicious=1, public_seed=11)
        accepted = server.aggregate_layer("layer", first)
        updated = server.aggregate_layer("layer", second)
        self.assertFalse(updated.certified)
        self.assertEqual(updated.update_mode, "pairwise")
        self.assertEqual(updated.certified_swaps, 1)
        self.assertEqual(int(updated.membership.sum()), 2)
        added = updated.membership & ~accepted.membership
        removed = accepted.membership & ~updated.membership
        self.assertGreater(
            int(updated.membership_counts[added][0]),
            int(updated.membership_counts[removed][0]) + server.max_malicious,
        )

    def test_uncertified_change_freezes_without_a_safe_pair(self):
        batch = self.batch([[5, 4, 3, 2, 1, 0]], 2)
        server = CMGRAAggregator(2 / 6, max_malicious=2, public_seed=17)
        initial_membership = server.initial_ranking("layer", 6) >= 4
        result = server.aggregate_layer("layer", batch)
        self.assertEqual(result.update_mode, "freeze")
        self.assertTrue(torch.equal(result.membership, initial_membership))
        self.assertEqual(int(result.membership.sum()), 2)

    def test_invalid_client_pair_is_dropped(self):
        rankings = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        memberships = membership_from_rank_values(rankings, 2)
        memberships[1] = torch.tensor([1, 1, 0, 0])
        result = CMGRAAggregator(0.5, max_malicious=0).aggregate_layer(
            "layer", CMGRAUpdateBatch(rankings, memberships)
        )
        self.assertEqual(result.valid_clients, 1)
        self.assertEqual(result.rejected_clients, 1)

    def test_checkpoint_round_trip_preserves_membership(self):
        batch = self.batch([[3, 2, 1, 0], [3, 2, 1, 0]], 2)
        original = CMGRAAggregator(0.5, max_malicious=0, public_seed=3)
        original.aggregate_layer("layer", batch)
        restored = CMGRAAggregator(0.5, max_malicious=0, public_seed=3)
        restored.load_state_dict(original.state_dict())
        self.assertTrue(torch.equal(original.membership("layer"), restored.membership("layer")))

    def test_retired_membership_only_variants_are_rejected(self):
        for variant in ("cmgra-px-m", "binary"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                CMGRAAggregator(0.5, max_malicious=0, aggregation_variant=variant)


if __name__ == "__main__":
    unittest.main()
