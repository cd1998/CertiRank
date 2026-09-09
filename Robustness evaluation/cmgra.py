"""Certified Membership-Gated Ranking Aggregation (CMGRA).

The public API uses *rank values*: ``rankings[u, j]`` is the rank assigned
to edge ``j`` by client ``u`` and larger values mean more important edges.
The legacy VEM training code instead stores each row as edge ids from least to
most important.  ``CMGRAAggregator.aggregate_model`` explicitly adapts that
legacy representation at the server boundary.

Only the full-ranking CMGRA-PX protocol is exposed: every upload contains a
rank permutation and its consistent exact K-hot membership.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class CMGRAUpdateBatch:
    """One layer's client uploads in the paper's ``(r_u, b_u)`` format."""

    rankings: torch.Tensor
    memberships: torch.Tensor


@dataclass(frozen=True)
class CMGRAResult:
    """Full result for one layer (primarily useful for tests and auditing)."""

    global_ranking: torch.Tensor
    membership: torch.Tensor
    membership_counts: torch.Tensor
    borda_scores: torch.Tensor
    candidate_membership: torch.Tensor
    certified: bool
    update_mode: str
    certified_swaps: int
    changed_edges: int
    boundary_margin: Optional[float]
    valid_clients: int
    rejected_clients: int


@dataclass(frozen=True)
class CMGRALayerSummary:
    """Small per-layer result retained by the training loop."""

    certified: bool
    update_mode: str
    certified_swaps: int
    changed_edges: int
    boundary_margin: Optional[float]
    valid_clients: int
    rejected_clients: int
    selected_edges: int
    total_edges: int


UpdateInput = Union[torch.Tensor, CMGRAUpdateBatch, Tuple[torch.Tensor, torch.Tensor]]


def retained_edge_count(num_edges: int, retention_ratio: float) -> int:
    """Return K = ceil(k d), with conservative floating-point handling."""

    if num_edges <= 0:
        raise ValueError("num_edges must be positive")
    if not 0.0 <= retention_ratio <= 1.0:
        raise ValueError("retention_ratio must be in [0, 1]")
    return min(num_edges, int(math.ceil(retention_ratio * num_edges)))


def edge_order_to_rank_values(edge_order: torch.Tensor) -> torch.Tensor:
    """Convert legacy least-to-most edge orders to per-edge rank values."""

    if edge_order.ndim != 2:
        raise ValueError("edge_order must have shape [clients, edges]")
    order = edge_order.to(dtype=torch.long)
    clients, num_edges = order.shape
    rank_values = torch.empty_like(order)
    positions = torch.arange(num_edges, device=order.device, dtype=torch.long)
    rank_values.scatter_(1, order, positions.expand(clients, -1))
    return rank_values


def rank_values_to_edge_order(rank_values: torch.Tensor) -> torch.Tensor:
    """Convert per-edge rank values to legacy least-to-most edge orders."""

    if rank_values.ndim != 2:
        raise ValueError("rank_values must have shape [clients, edges]")
    return torch.argsort(rank_values, dim=1)


def membership_from_rank_values(rank_values: torch.Tensor, k: int) -> torch.Tensor:
    """Build the exact K-hot membership implied by a full ranking."""

    if rank_values.ndim != 2:
        raise ValueError("rank_values must have shape [clients, edges]")
    num_edges = rank_values.shape[1]
    if not 0 <= k <= num_edges:
        raise ValueError("k must be in [0, num_edges]")
    return rank_values >= (num_edges - k)


def _prepare_score_levels(scores: torch.Tensor) -> torch.Tensor:
    """Return strictly increasing non-negative values for realizing ranks."""

    levels = scores.detach().flatten().abs().sort().values
    if levels.numel() <= 1:
        return levels
    if torch.any(levels[1:] <= levels[:-1]):
        # Exact ties would let MaskConv's top-k implementation choose an edge
        # outside the certified membership. They are unusual for trained
        # floating scores, but a deterministic evenly-spaced fallback makes
        # the invariant unconditional for supported tensor sizes.
        if levels.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            exact_integer_limit = {
                torch.float16: 2**11,
                torch.bfloat16: 2**8,
                torch.float32: 2**24,
            }[levels.dtype]
            if levels.numel() > exact_integer_limit:
                raise ValueError(
                    "score dtype cannot represent enough distinct rank levels"
                )
        upper = max(float(levels[-1].item()), 1.0)
        levels = torch.linspace(
            0.0,
            upper,
            levels.numel(),
            dtype=levels.dtype,
            device=levels.device,
        )
        if torch.any(levels[1:] <= levels[:-1]):
            raise ValueError("failed to construct distinct score levels")
    return levels


class CMGRAAggregator:
    """Stateful, layer-wise full-ranking CMGRA-PX server aggregator.

    ``max_malicious`` is the certified upper bound ``m``.  Membership state is
    retained across rounds and is initialized from a deterministic public
    seed before any client update is processed. ``aggregation_variant`` is
    retained in the constructor/checkpoint schema for backward-compatible
    resumption, but its only accepted value is ``cmgra-px``.
    """

    def __init__(
        self,
        retention_ratio: float,
        max_malicious: int,
        public_seed: int = 0,
        invalid_policy: str = "drop",
        pairwise_fallback: bool = True,
        pairwise_matching: str = "extreme",
        aggregation_variant: str = "cmgra-px",
        preserve_group_order: bool = False,
        membership_group_order: bool = False,
        borda_group_order: bool = False,
        borda_final_order: bool = False,
        count_ema: float = 0.0,
        swap_cap: Optional[int] = None,
        persistence_rounds: int = 1,
        update_interval: int = 1,
        borda_mode: str = "sum",
        rank_buckets: int = 0,
        borda_ema: float = 0.0,
        prior_rank_weight: float = 0.0,
    ) -> None:
        if not 0.0 <= retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in [0, 1]")
        if max_malicious < 0:
            raise ValueError("max_malicious must be non-negative")
        if invalid_policy not in {"drop", "raise"}:
            raise ValueError("invalid_policy must be 'drop' or 'raise'")
        if pairwise_matching not in {"extreme", "max-cardinality"}:
            raise ValueError(
                "pairwise_matching must be 'extreme' or 'max-cardinality'"
            )
        if aggregation_variant != "cmgra-px":
            raise ValueError("aggregation_variant must be 'cmgra-px'")
        if not 0.0 <= count_ema < 1.0:
            raise ValueError("count_ema must be in [0, 1)")
        if sum(
            (
                bool(preserve_group_order),
                bool(membership_group_order),
                bool(borda_group_order),
                bool(borda_final_order),
            )
        ) > 1:
            raise ValueError(
                "preserve_group_order, membership_group_order, and "
                "Borda group-order modes are mutually exclusive"
            )
        if swap_cap is not None and swap_cap <= 0:
            raise ValueError("swap_cap must be positive or None")
        if persistence_rounds <= 0:
            raise ValueError("persistence_rounds must be positive")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if borda_mode not in {"sum", "lower-bound"}:
            raise ValueError("borda_mode must be 'sum' or 'lower-bound'")
        if rank_buckets == 1 or rank_buckets < 0:
            raise ValueError("rank_buckets must be 0 or at least 2")
        if not 0.0 <= borda_ema < 1.0:
            raise ValueError("borda_ema must be in [0, 1)")
        if not 0.0 <= prior_rank_weight < 1.0:
            raise ValueError("prior_rank_weight must be in [0, 1)")
        if (
            (borda_ema or prior_rank_weight)
            and not (borda_group_order or borda_final_order)
        ):
            raise ValueError(
                "temporal Borda requires borda_group_order or "
                "borda_final_order"
            )
        if rank_buckets and (borda_ema or prior_rank_weight):
            raise ValueError(
                "rank_buckets cannot be combined with temporal Borda"
            )
        self.retention_ratio = float(retention_ratio)
        self.max_malicious = int(max_malicious)
        self.public_seed = int(public_seed)
        self.invalid_policy = invalid_policy
        self.pairwise_fallback = bool(pairwise_fallback)
        self.pairwise_matching = pairwise_matching
        self.aggregation_variant = aggregation_variant
        self.preserve_group_order = bool(preserve_group_order)
        self.membership_group_order = bool(membership_group_order)
        self.borda_group_order = bool(borda_group_order)
        self.borda_final_order = bool(borda_final_order)
        self.count_ema = float(count_ema)
        self.swap_cap = int(swap_cap) if swap_cap is not None else None
        self.persistence_rounds = int(persistence_rounds)
        self.update_interval = int(update_interval)
        self.borda_mode = borda_mode
        self.rank_buckets = int(rank_buckets)
        self.borda_ema = float(borda_ema)
        self.prior_rank_weight = float(prior_rank_weight)
        self._memberships: Dict[str, torch.Tensor] = {}
        self._initial_rankings: Dict[str, torch.Tensor] = {}
        self._global_rankings: Dict[str, torch.Tensor] = {}
        self._count_ema_sums: Dict[str, torch.Tensor] = {}
        self._count_ema_weights: Dict[str, float] = {}
        self._candidate_memberships: Dict[str, torch.Tensor] = {}
        self._candidate_streaks: Dict[str, torch.Tensor] = {}
        self._borda_ema_scores: Dict[str, torch.Tensor] = {}
        self._layer_rounds: Dict[str, int] = {}

    def _layer_seed(self, layer_name: str, num_edges: int) -> int:
        material = f"CMGRA:{self.public_seed}:{layer_name}:{num_edges}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63 - 1)

    def initial_ranking(
        self, layer_name: str, num_edges: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Return the public-seed initial per-edge ranking for a layer."""

        cached = self._initial_rankings.get(layer_name)
        if cached is not None:
            if cached.numel() != num_edges:
                raise ValueError(f"layer {layer_name!r} changed size across rounds")
            return cached.to(device=device) if device is not None else cached.clone()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._layer_seed(layer_name, num_edges))
        edge_order = torch.randperm(num_edges, generator=generator)
        ranks = torch.empty(num_edges, dtype=torch.long)
        ranks[edge_order] = torch.arange(num_edges, dtype=torch.long)
        self._initial_rankings[layer_name] = ranks
        self._global_rankings[layer_name] = ranks.clone()
        k = retained_edge_count(num_edges, self.retention_ratio)
        self._memberships[layer_name] = ranks >= (num_edges - k)
        return ranks.to(device=device) if device is not None else ranks.clone()

    def initialize_model(
        self,
        model: torch.nn.Module,
        score_levels: Mapping[str, torch.Tensor],
    ) -> None:
        """Apply public initial rankings before clients receive the model."""

        with torch.no_grad():
            for name, module in model.named_modules():
                if not hasattr(module, "scores"):
                    continue
                levels = _prepare_score_levels(score_levels[str(name)])
                ranking = self.initial_ranking(str(name), levels.numel(), levels.device)
                module.scores.copy_(levels[ranking].reshape_as(module.scores))

    def adopt_global_rankings(
        self, rankings: Mapping[str, torch.Tensor]
    ) -> None:
        """Start a fresh CMGRA state from externally chosen global rankings.

        This is intended for an explicitly heuristic warm-start or rollback
        protocol.  The adopted boundary is not retrospectively certified, but
        every subsequent CMGRA boundary change remains subject to the normal
        membership certificate.
        """

        adopted_memberships: Dict[str, torch.Tensor] = {}
        adopted_rankings: Dict[str, torch.Tensor] = {}
        for layer_name, ranking in rankings.items():
            values = ranking.detach().long().flatten().cpu()
            valid = self._permutation_validity(values.unsqueeze(0))
            if not bool(valid.item()):
                raise ValueError(
                    f"adopted ranking for layer {layer_name!r} is not a "
                    "permutation"
                )
            k = retained_edge_count(values.numel(), self.retention_ratio)
            adopted_rankings[str(layer_name)] = values.clone()
            adopted_memberships[str(layer_name)] = (
                values >= values.numel() - k
            )
            self._initial_rankings.setdefault(
                str(layer_name), values.clone()
            )

        self._global_rankings = adopted_rankings
        self._memberships = adopted_memberships
        self._count_ema_sums.clear()
        self._count_ema_weights.clear()
        self._candidate_memberships.clear()
        self._candidate_streaks.clear()
        self._borda_ema_scores.clear()
        self._layer_rounds = {
            layer_name: 0 for layer_name in adopted_rankings
        }

    @staticmethod
    def _permutation_validity(rankings: torch.Tensor) -> torch.Tensor:
        if rankings.ndim != 2:
            raise ValueError("rankings must have shape [clients, edges]")
        if rankings.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=rankings.device)
        num_edges = rankings.shape[1]
        if rankings.dtype == torch.bool or torch.is_floating_point(rankings):
            integer_valued = rankings == rankings.round()
        else:
            integer_valued = torch.ones_like(rankings, dtype=torch.bool)
        in_range = (rankings >= 0) & (rankings < num_edges)
        prelim = (integer_valued & in_range).all(dim=1)

        # Sorting is exact (unlike sum/checksum shortcuts) and therefore also
        # rejects duplicated ranks whose aggregate moments happen to match.
        as_long = rankings.to(dtype=torch.long)
        expected = torch.arange(num_edges, device=rankings.device, dtype=torch.long)
        sorted_rows = torch.sort(as_long, dim=1).values
        return prelim & (sorted_rows == expected).all(dim=1)

    @classmethod
    def validate_updates(
        cls, rankings: torch.Tensor, memberships: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Return a Boolean validity flag for each uploaded ``(r_u, b_u)``."""

        if rankings.ndim != 2 or memberships.ndim != 2:
            raise ValueError("rankings and memberships must have shape [clients, edges]")
        if rankings.shape != memberships.shape:
            raise ValueError("rankings and memberships must have identical shapes")
        if rankings.device != memberships.device:
            memberships = memberships.to(rankings.device)
        num_edges = rankings.shape[1]
        if not 0 <= k <= num_edges:
            raise ValueError("k must be in [0, num_edges]")

        permutation_ok = cls._permutation_validity(rankings)
        binary = ((memberships == 0) | (memberships == 1)).all(dim=1)
        k_hot = memberships.to(dtype=torch.long).sum(dim=1) == k
        consistent = (
            memberships.to(dtype=torch.bool)
            == membership_from_rank_values(rankings.to(dtype=torch.long), k)
        ).all(dim=1)
        return permutation_ok & binary & k_hot & consistent

    @staticmethod
    def validate_memberships(memberships: torch.Tensor, k: int) -> torch.Tensor:
        """Return validity flags for membership-only K-hot uploads."""

        if memberships.ndim != 2:
            raise ValueError("memberships must have shape [clients, edges]")
        num_edges = memberships.shape[1]
        if not 0 <= k <= num_edges:
            raise ValueError("k must be in [0, num_edges]")
        binary = ((memberships == 0) | (memberships == 1)).all(dim=1)
        k_hot = memberships.to(dtype=torch.long).sum(dim=1) == k
        return binary & k_hot

    @staticmethod
    def _importance_order(
        membership_counts: torch.Tensor,
        borda_scores: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sort by count desc, Borda desc, edge id asc (stable lexicographic)."""

        # nonzero()/arange() supplies ascending edge ids. Stable secondary and
        # primary sorts preserve that deterministic tie break.
        ordered = edge_ids
        ordered = ordered[
            torch.argsort(borda_scores[ordered], descending=True, stable=True)
        ]
        ordered = ordered[
            torch.argsort(membership_counts[ordered], descending=True, stable=True)
        ]
        return ordered

    @staticmethod
    def _borda_order(
        borda_scores: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sort by Borda score descending, then edge id ascending."""

        return edge_ids[
            torch.argsort(
                borda_scores[edge_ids],
                descending=True,
                stable=True,
            )
        ]

    @staticmethod
    def _maximum_cardinality_pairs(
        membership_counts: torch.Tensor,
        weakest_incumbents: torch.Tensor,
        weakest_challengers: torch.Tensor,
        max_malicious: int,
        swap_cap: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the largest deterministic set of certified one-to-one swaps.

        Both edge lists must be ordered from weak to strong by membership
        count, with Borda score and edge id used only as deterministic tie
        breaks. For each incumbent, ``searchsorted`` locates the weakest
        challenger whose observed count exceeds the incumbent by more than
        ``max_malicious``. A cumulative maximum then enforces distinct,
        increasing challenger positions without a Python loop over edges.
        This is the standard greedy maximum-cardinality matching for a
        threshold-ordered bipartite graph.
        """

        empty = weakest_incumbents[:0]
        if not weakest_incumbents.numel() or not weakest_challengers.numel():
            return empty, weakest_challengers[:0]

        incumbent_counts = membership_counts[weakest_incumbents]
        challenger_counts = membership_counts[weakest_challengers]
        incumbent_positions = torch.arange(
            weakest_incumbents.numel(),
            device=weakest_incumbents.device,
            dtype=torch.long,
        )
        first_eligible = torch.searchsorted(
            challenger_counts.contiguous(),
            (incumbent_counts + max_malicious).contiguous(),
            right=True,
        )
        challenger_positions = incumbent_positions + torch.cummax(
            first_eligible - incumbent_positions,
            dim=0,
        ).values
        matched = int(
            (challenger_positions < weakest_challengers.numel()).sum().item()
        )
        if swap_cap is not None:
            matched = min(matched, swap_cap)
        if not matched:
            return empty, weakest_challengers[:0]
        return (
            weakest_incumbents[:matched],
            weakest_challengers[challenger_positions[:matched]],
        )

    def _aggregate_borda(
        self,
        rankings: Optional[torch.Tensor],
        membership_counts: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Compute an ordering score from additively aggregated statistics.

        The simulator receives validated client rows because it models both
        servers in one process. This method deliberately collapses those rows
        to aggregate Borda ``s`` before doing anything else. Both modes can
        therefore be implemented by a server that sees only ``s`` and
        aggregate membership counts ``c``.

        ``lower-bound`` subtracts the largest Borda contribution that at most
        ``m`` valid malicious clients could have made to each edge, subject to
        the edge's observed membership count. The result is a conservative
        lower bound on the honest clients' Borda score.
        """

        if rankings is None:
            return torch.zeros_like(membership_counts)
        aggregate_s = rankings.sum(dim=0)
        if self.borda_mode == "sum" or self.max_malicious == 0:
            return aggregate_s

        clients, num_edges = rankings.shape
        m = min(self.max_malicious, clients)
        # At most min(m, c_j) malicious clients can have placed edge j in
        # their top-K. A malicious member can contribute at most d-1 and a
        # malicious non-member at most d-K-1.
        malicious_members = torch.minimum(
            membership_counts,
            torch.full_like(membership_counts, m),
        )
        max_nonmember_rank = max(num_edges - k - 1, 0)
        malicious_max = (
            malicious_members * (num_edges - 1)
            + (m - malicious_members) * max_nonmember_rank
        )
        return aggregate_s - malicious_max

    @staticmethod
    def _aggregate_rank_buckets(
        rankings: torch.Tensor,
        buckets: int,
    ) -> torch.Tensor:
        """Aggregate fixed-cardinality quantized rank evidence.

        A valid permutation is mapped monotonically from ``[0, d-1]`` to
        ``[0, buckets-1]``. Consequently, one malicious client can alter the
        pairwise difference between two edges by at most ``buckets-1``,
        instead of the ``d-1`` budget of full Borda ranks.
        """

        if rankings.ndim != 2:
            raise ValueError("rankings must have shape [clients, edges]")
        if buckets < 2:
            raise ValueError("buckets must be at least 2")
        num_edges = rankings.shape[1]
        if num_edges == 0:
            return torch.zeros(
                0, dtype=torch.long, device=rankings.device
            )
        quantized = torch.div(
            rankings.to(dtype=torch.long) * buckets,
            num_edges,
            rounding_mode="floor",
        ).clamp_max(buckets - 1)
        return quantized.sum(dim=0)

    @staticmethod
    def _certified_bucket_group_order(
        bucket_scores: torch.Tensor,
        edge_ids: torch.Tensor,
        previous_ranking: torch.Tensor,
        max_malicious: int,
        buckets: int,
    ) -> torch.Tensor:
        """Refine one membership group through certified adjacent buckets.

        The previous fine-grained order is partitioned into ``Q`` fixed-size
        server buckets. Starting at the lowest boundary, a lower-bucket edge
        may replace an upper-bucket edge only when their aggregate quantized
        rank gap exceeds ``m * (Q - 1)``. Each operation is therefore the same
        K-preserving certified set exchange used at the main membership
        boundary. Processing boundaries bottom-up lets strongly supported
        edges climb multiple coarse levels, while uncertified fine order is
        inherited unchanged inside every resulting bucket.
        """

        previous_order = edge_ids[
            torch.argsort(
                previous_ranking[edge_ids],
                descending=True,
                stable=True,
            )
        ]
        size = previous_order.numel()
        if size < 2:
            return previous_order
        threshold = max_malicious * (buckets - 1)
        bucket_ids = torch.empty_like(previous_ranking)
        bucket_ids[previous_order] = torch.div(
            torch.arange(
                size, device=previous_order.device, dtype=torch.long
            )
            * buckets,
            size,
            rounding_mode="floor",
        ).clamp_max(buckets - 1)

        for upper_bucket in range(buckets - 2, -1, -1):
            upper_edges = edge_ids[
                bucket_ids[edge_ids] == upper_bucket
            ]
            lower_edges = edge_ids[
                bucket_ids[edge_ids] == upper_bucket + 1
            ]
            weakest_upper = torch.flip(
                CMGRAAggregator._borda_order(
                    bucket_scores, upper_edges
                ),
                dims=[0],
            )
            weakest_lower = torch.flip(
                CMGRAAggregator._borda_order(
                    bucket_scores, lower_edges
                ),
                dims=[0],
            )
            demoted, promoted = (
                CMGRAAggregator._maximum_cardinality_pairs(
                    bucket_scores,
                    weakest_upper,
                    weakest_lower,
                    threshold,
                    None,
                )
            )
            if demoted.numel():
                bucket_ids[demoted] = upper_bucket + 1
                bucket_ids[promoted] = upper_bucket

        ordered_buckets = []
        for bucket in range(buckets):
            members = edge_ids[bucket_ids[edge_ids] == bucket]
            ordered_buckets.append(
                members[
                    torch.argsort(
                        previous_ranking[members],
                        descending=True,
                        stable=True,
                    )
                ]
            )
        return torch.cat(ordered_buckets)

    def _temporal_borda_scores(
        self,
        layer_name: str,
        borda_scores: torch.Tensor,
        previous_ranking: torch.Tensor,
        valid_clients: int,
    ) -> torch.Tensor:
        """Blend normalized aggregate Borda with the prior global ranking.

        Only the aggregate statistic ``s`` is consumed.  The exponential
        moving average suppresses one-round ranking shocks, while the prior
        global ranking limits how quickly VEM can rewrite the within-group
        order that seeds the next local Edge-Popup round.
        """

        num_edges = borda_scores.numel()
        denominator = max(valid_clients * max(num_edges - 1, 1), 1)
        normalized = (
            borda_scores.to(dtype=torch.float32) / float(denominator)
        )
        previous_ema = self._borda_ema_scores.get(layer_name)
        if previous_ema is None:
            ema_scores = normalized
        else:
            previous_ema = previous_ema.to(
                device=borda_scores.device,
                dtype=torch.float32,
            )
            ema_scores = (
                self.borda_ema * previous_ema
                + (1.0 - self.borda_ema) * normalized
            )
        self._borda_ema_scores[layer_name] = ema_scores.detach().clone()

        prior = previous_ranking.to(dtype=torch.float32)
        if num_edges > 1:
            prior = prior / float(num_edges - 1)
        return (
            self.prior_rank_weight * prior
            + (1.0 - self.prior_rank_weight) * ema_scores
        )

    def _unpack(
        self, updates: UpdateInput, input_format: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(updates, CMGRAUpdateBatch):
            rankings, memberships = updates.rankings, updates.memberships
            input_format = "rank_values"
        elif isinstance(updates, tuple):
            if len(updates) != 2:
                raise ValueError("update tuple must contain (rankings, memberships)")
            rankings, memberships = updates
            input_format = "rank_values"
        else:
            rankings = updates
            memberships = None

        memberships_provided = memberships is not None

        if rankings.ndim != 2:
            raise ValueError("rankings must have shape [clients, edges]")
        if input_format not in {"rank_values", "edge_order"}:
            raise ValueError("input_format must be 'rank_values' or 'edge_order'")

        # Validate the legacy permutation before scatter so duplicate edge ids
        # cannot be hidden by an overwrite.
        if input_format == "edge_order":
            order_valid = self._permutation_validity(rankings)
            # Clamp only for the conversion itself. Out-of-range rows remain
            # invalid via ``order_valid`` and are subsequently rejected.
            safe_order = rankings.to(dtype=torch.long).clamp(0, rankings.shape[1] - 1)
            rank_values = edge_order_to_rank_values(safe_order)
        else:
            order_valid = torch.ones(
                rankings.shape[0], dtype=torch.bool, device=rankings.device
            )
            rank_values = rankings

        k = retained_edge_count(rank_values.shape[1], self.retention_ratio)
        if memberships is None:
            # Compatibility path for the original single-tensor simulator.
            memberships = membership_from_rank_values(rank_values.to(dtype=torch.long), k)
        else:
            memberships = memberships.to(rank_values.device)
        if input_format == "edge_order" and not memberships_provided:
            # A valid edge-order permutation converts bijectively to a valid
            # rank-value permutation, and the generated membership is exact.
            # Avoid sorting million-edge layers a second time.
            valid = order_valid
        else:
            valid = self.validate_updates(rank_values, memberships, k) & order_valid
        if not valid.all() and self.invalid_policy == "raise":
            rejected = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
            raise ValueError(f"invalid CMGRA client updates at rows {rejected}")
        if not valid.any():
            raise ValueError("no valid CMGRA client updates remain after validation")
        return rank_values[valid].to(dtype=torch.long), memberships[valid].to(dtype=torch.bool)

    def aggregate_layer(
        self,
        layer_name: str,
        updates: UpdateInput,
        input_format: str = "rank_values",
    ) -> CMGRAResult:
        """Validate uploads and aggregate one layer for one communication round."""

        if input_format == "membership":
            raise ValueError("cmgra-px requires full ranking uploads")
        if isinstance(updates, CMGRAUpdateBatch):
            submitted_clients = updates.rankings.shape[0]
        elif isinstance(updates, tuple):
            submitted_clients = updates[0].shape[0]
        else:
            submitted_clients = updates.shape[0]
        rankings, memberships = self._unpack(updates, input_format)
        valid_clients, num_edges = memberships.shape
        k = retained_edge_count(num_edges, self.retention_ratio)
        previous = self._memberships.get(layer_name)
        if previous is None:
            self.initial_ranking(layer_name, num_edges)
            previous = self._memberships[layer_name]
        if previous.numel() != num_edges:
            raise ValueError(f"layer {layer_name!r} changed size across rounds")
        previous = previous.to(memberships.device)

        membership_counts = memberships.to(dtype=torch.long).sum(dim=0)
        if self.count_ema:
            previous_sum = self._count_ema_sums.get(layer_name)
            previous_weight = self._count_ema_weights.get(layer_name, 0.0)
            if previous_sum is None:
                previous_sum = torch.zeros(
                    num_edges,
                    dtype=torch.float64,
                    device=memberships.device,
                )
            else:
                previous_sum = previous_sum.to(
                    memberships.device, dtype=torch.float64
                )
            ema_sum = (
                self.count_ema * previous_sum
                + membership_counts.to(dtype=torch.float64)
            )
            ema_weight = self.count_ema * previous_weight + 1.0
            decision_counts = ema_sum / ema_weight
            self._count_ema_sums[layer_name] = ema_sum.detach().clone()
            self._count_ema_weights[layer_name] = float(ema_weight)
        else:
            decision_counts = membership_counts
        borda_scores = self._aggregate_borda(rankings, membership_counts, k)
        ordering_scores = borda_scores
        if self.rank_buckets:
            if rankings is None:
                raise ValueError("rank_buckets requires full ranking uploads")
            ordering_scores = self._aggregate_rank_buckets(
                rankings, self.rank_buckets
            )
        edge_ids = torch.arange(
            num_edges, device=memberships.device, dtype=torch.long
        )
        candidate_order = edge_ids[
            torch.argsort(decision_counts, descending=True, stable=True)
        ]
        candidate = torch.zeros(
            num_edges, dtype=torch.bool, device=memberships.device
        )
        if k:
            candidate[candidate_order[:k]] = True

        previous_candidate = self._candidate_memberships.get(layer_name)
        previous_streak = self._candidate_streaks.get(layer_name)
        if previous_candidate is None or previous_streak is None:
            candidate_streak = torch.ones(
                num_edges,
                dtype=torch.long,
                device=memberships.device,
            )
        else:
            previous_candidate = previous_candidate.to(memberships.device)
            previous_streak = previous_streak.to(memberships.device)
            candidate_streak = torch.where(
                candidate == previous_candidate,
                previous_streak + 1,
                torch.ones_like(previous_streak),
            )
        self._candidate_memberships[layer_name] = candidate.detach().clone()
        self._candidate_streaks[layer_name] = candidate_streak.detach().clone()
        layer_round = self._layer_rounds.get(layer_name, 0) + 1
        self._layer_rounds[layer_name] = layer_round
        update_due = layer_round % self.update_interval == 0

        if k == 0 or k == num_edges:
            margin: Optional[float] = None
            certified = True
        else:
            kth = float(decision_counts[candidate_order[k - 1]].item())
            next_count = float(decision_counts[candidate_order[k]].item())
            margin = kth - next_count
            certified = margin > self.max_malicious
        if certified and self.persistence_rounds > 1:
            changed = candidate != previous
            certified = bool(
                not changed.any()
                or (
                    candidate_streak[changed] >= self.persistence_rounds
                ).all().item()
            )

        certified_swaps = 0
        if not update_due:
            global_membership = previous
            update_mode = "freeze"
        elif certified:
            global_membership = candidate
            update_mode = "full"
        elif self.pairwise_fallback:
            # Whole-set certification is often unavailable at an integer-vote
            # boundary. Preserve K exactly while accepting only paired changes
            # whose honest preference is itself certified:
            #   h_out >= c_out - m > c_in >= h_in.
            incumbents = torch.nonzero(previous, as_tuple=False).flatten()
            challengers = torch.nonzero(~previous, as_tuple=False).flatten()
            if self.pairwise_matching == "max-cardinality":
                # Maximum-cardinality matching needs both sides in ascending
                # count order. Borda and edge id retain deterministic tie
                # breaking but never relax the per-pair certificate.
                weakest_incumbents = torch.flip(
                    self._importance_order(
                        decision_counts, ordering_scores, incumbents
                    ),
                    dims=[0],
                )
                weakest_challengers = torch.flip(
                    self._importance_order(
                        decision_counts, ordering_scores, challengers
                    ),
                    dims=[0],
                )
                if self.persistence_rounds > 1:
                    weakest_incumbents = weakest_incumbents[
                        candidate_streak[weakest_incumbents]
                        >= self.persistence_rounds
                    ]
                    weakest_challengers = weakest_challengers[
                        candidate_streak[weakest_challengers]
                        >= self.persistence_rounds
                    ]
                paired_incumbents, paired_challengers = (
                    self._maximum_cardinality_pairs(
                        decision_counts,
                        weakest_incumbents,
                        weakest_challengers,
                        self.max_malicious,
                        self.swap_cap,
                    )
                )
                certified_swaps = int(paired_incumbents.numel())
            elif self.borda_group_order:
                weakest_incumbents = torch.flip(
                    self._borda_order(ordering_scores, incumbents),
                    dims=[0],
                )
                strongest_challengers = self._borda_order(
                    ordering_scores, challengers
                )
            else:
                weakest_incumbents = torch.flip(
                    self._importance_order(
                        decision_counts, ordering_scores, incumbents
                    ),
                    dims=[0],
                )
                strongest_challengers = self._importance_order(
                    decision_counts, ordering_scores, challengers
                )
            if self.pairwise_matching != "max-cardinality":
                pair_count = min(
                    weakest_incumbents.numel(), strongest_challengers.numel()
                )
                selected_pairs = torch.empty(
                    0, dtype=torch.long, device=memberships.device
                )
                if pair_count:
                    vote_gaps = (
                        decision_counts[strongest_challengers[:pair_count]]
                        - decision_counts[weakest_incumbents[:pair_count]]
                    )
                    eligible = vote_gaps > self.max_malicious
                    if self.persistence_rounds > 1:
                        eligible = eligible & (
                            candidate_streak[
                                strongest_challengers[:pair_count]
                            ]
                            >= self.persistence_rounds
                        ) & (
                            candidate_streak[
                                weakest_incumbents[:pair_count]
                            ]
                            >= self.persistence_rounds
                        )
                    selected_pairs = torch.nonzero(
                        eligible, as_tuple=False
                    ).flatten()
                    if self.swap_cap is not None:
                        selected_pairs = selected_pairs[: self.swap_cap]
                    certified_swaps = int(selected_pairs.numel())
                paired_incumbents = weakest_incumbents[selected_pairs]
                paired_challengers = strongest_challengers[selected_pairs]
            global_membership = previous.clone()
            if certified_swaps:
                global_membership[paired_incumbents] = False
                global_membership[paired_challengers] = True
                update_mode = "pairwise"
            else:
                update_mode = "freeze"
        else:
            global_membership = previous
            update_mode = "freeze"
        changed_edges = int((global_membership != previous).sum().item())
        if int(global_membership.sum().item()) != k:
            raise RuntimeError("CMGRA invariant violated: global membership is not K-hot")
        self._memberships[layer_name] = global_membership.detach().clone()

        selected = torch.nonzero(global_membership, as_tuple=False).flatten()
        unselected = torch.nonzero(~global_membership, as_tuple=False).flatten()
        previous_ranking = self._global_rankings.get(layer_name)
        if previous_ranking is None:
            previous_ranking = self._initial_rankings[layer_name]
        previous_ranking = previous_ranking.to(memberships.device)
        group_borda_scores = borda_scores
        if self.borda_ema or self.prior_rank_weight:
            group_borda_scores = self._temporal_borda_scores(
                layer_name,
                borda_scores,
                previous_ranking,
                valid_clients,
            )
        if self.rank_buckets:
            selected_order = self._certified_bucket_group_order(
                ordering_scores,
                selected,
                previous_ranking,
                self.max_malicious,
                self.rank_buckets,
            )
            unselected_order = self._certified_bucket_group_order(
                ordering_scores,
                unselected,
                previous_ranking,
                self.max_malicious,
                self.rank_buckets,
            )
        elif self.membership_group_order:
            # Use only additively aggregatable K-hot evidence to refine the
            # score order supplied to the next local Edge-Popup round.  The
            # current certified membership remains the highest-priority
            # boundary; previous global rank is a deterministic, poison-free
            # tie break.  This restores benign within-group learning without
            # reopening the full-ranking Borda channel targeted by VEM.
            selected_order = selected[
                torch.argsort(
                    previous_ranking[selected],
                    descending=True,
                    stable=True,
                )
            ]
            selected_order = selected_order[
                torch.argsort(
                    decision_counts[selected_order],
                    descending=True,
                    stable=True,
                )
            ]
            unselected_order = unselected[
                torch.argsort(
                    previous_ranking[unselected],
                    descending=True,
                    stable=True,
                )
            ]
            unselected_order = unselected_order[
                torch.argsort(
                    decision_counts[unselected_order],
                    descending=True,
                    stable=True,
                )
            ]
        elif self.preserve_group_order:
            selected_order = selected[
                torch.argsort(
                    previous_ranking[selected],
                    descending=True,
                    stable=True,
                )
            ]
            unselected_order = unselected[
                torch.argsort(
                    previous_ranking[unselected],
                    descending=True,
                    stable=True,
                )
            ]
        elif self.borda_group_order or self.borda_final_order:
            selected_order = self._borda_order(
                group_borda_scores, selected
            )
            unselected_order = self._borda_order(
                group_borda_scores, unselected
            )
        else:
            selected_order = self._importance_order(
                decision_counts, borda_scores, selected
            )
            unselected_order = self._importance_order(
                decision_counts, borda_scores, unselected
            )

        global_ranking = torch.empty(
            num_edges, dtype=torch.long, device=memberships.device
        )
        if unselected_order.numel():
            global_ranking[unselected_order] = torch.arange(
                unselected_order.numel() - 1,
                -1,
                -1,
                device=memberships.device,
                dtype=torch.long,
            )
        if selected_order.numel():
            global_ranking[selected_order] = (num_edges - k) + torch.arange(
                selected_order.numel() - 1,
                -1,
                -1,
                device=memberships.device,
                dtype=torch.long,
            )
        self._global_rankings[layer_name] = global_ranking.detach().clone()

        return CMGRAResult(
            global_ranking=global_ranking,
            membership=global_membership,
            membership_counts=membership_counts,
            borda_scores=borda_scores,
            candidate_membership=candidate,
            certified=certified,
            update_mode=update_mode,
            certified_swaps=certified_swaps,
            changed_edges=changed_edges,
            boundary_margin=margin,
            valid_clients=valid_clients,
            rejected_clients=submitted_clients - valid_clients,
        )

    def aggregate_model(
        self,
        model: torch.nn.Module,
        updates: Mapping[str, UpdateInput],
        score_levels: Mapping[str, torch.Tensor],
        input_format: str = "edge_order",
    ) -> Dict[str, CMGRALayerSummary]:
        """Aggregate all scored layers and write rankings back to the model."""

        summaries: Dict[str, CMGRALayerSummary] = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if not hasattr(module, "scores"):
                    continue
                key = str(name)
                if key not in updates:
                    raise KeyError(f"missing CMGRA updates for layer {key!r}")
                result = self.aggregate_layer(key, updates[key], input_format=input_format)
                levels = _prepare_score_levels(score_levels[key])
                if levels.numel() != result.global_ranking.numel():
                    raise ValueError(f"score level size mismatch for layer {key!r}")
                levels = levels.to(module.scores.device)
                ranking = result.global_ranking.to(module.scores.device)
                module.scores.copy_(levels[ranking].reshape_as(module.scores))
                summaries[key] = CMGRALayerSummary(
                    certified=result.certified,
                    update_mode=result.update_mode,
                    certified_swaps=result.certified_swaps,
                    changed_edges=result.changed_edges,
                    boundary_margin=result.boundary_margin,
                    valid_clients=result.valid_clients,
                    rejected_clients=result.rejected_clients,
                    selected_edges=int(result.membership.sum().item()),
                    total_edges=result.membership.numel(),
                )
        return summaries

    def membership(self, layer_name: str) -> torch.Tensor:
        """Return a copy of the current certified/frozen membership state."""

        if layer_name not in self._memberships:
            raise KeyError(layer_name)
        return self._memberships[layer_name].clone()

    def state_dict(self) -> Dict[str, object]:
        """Serialize the cross-round membership state for resumable runs."""

        return {
            "retention_ratio": self.retention_ratio,
            "max_malicious": self.max_malicious,
            "public_seed": self.public_seed,
            "invalid_policy": self.invalid_policy,
            "pairwise_fallback": self.pairwise_fallback,
            "pairwise_matching": self.pairwise_matching,
            "aggregation_variant": self.aggregation_variant,
            "preserve_group_order": self.preserve_group_order,
            "membership_group_order": self.membership_group_order,
            "borda_group_order": self.borda_group_order,
            "borda_final_order": self.borda_final_order,
            "count_ema": self.count_ema,
            "swap_cap": self.swap_cap,
            "persistence_rounds": self.persistence_rounds,
            "update_interval": self.update_interval,
            "borda_mode": self.borda_mode,
            "rank_buckets": self.rank_buckets,
            "borda_ema": self.borda_ema,
            "prior_rank_weight": self.prior_rank_weight,
            "memberships": {
                key: value.detach().cpu().clone()
                for key, value in self._memberships.items()
            },
            "initial_rankings": {
                key: value.detach().cpu().clone()
                for key, value in self._initial_rankings.items()
            },
            "global_rankings": {
                key: value.detach().cpu().clone()
                for key, value in self._global_rankings.items()
            },
            "count_ema_sums": {
                key: value.detach().cpu().clone()
                for key, value in self._count_ema_sums.items()
            },
            "count_ema_weights": dict(self._count_ema_weights),
            "candidate_memberships": {
                key: value.detach().cpu().clone()
                for key, value in self._candidate_memberships.items()
            },
            "candidate_streaks": {
                key: value.detach().cpu().clone()
                for key, value in self._candidate_streaks.items()
            },
            "borda_ema_scores": {
                key: value.detach().cpu().clone()
                for key, value in self._borda_ema_scores.items()
            },
            "layer_rounds": dict(self._layer_rounds),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        immutable = {
            "retention_ratio": self.retention_ratio,
            "public_seed": self.public_seed,
            "invalid_policy": self.invalid_policy,
            "pairwise_fallback": self.pairwise_fallback,
            "pairwise_matching": self.pairwise_matching,
            "aggregation_variant": self.aggregation_variant,
            "preserve_group_order": self.preserve_group_order,
            "membership_group_order": self.membership_group_order,
            "borda_group_order": self.borda_group_order,
            "borda_final_order": self.borda_final_order,
            "count_ema": self.count_ema,
            "swap_cap": self.swap_cap,
            "persistence_rounds": self.persistence_rounds,
            "update_interval": self.update_interval,
            "borda_mode": self.borda_mode,
            "rank_buckets": self.rank_buckets,
            "borda_ema": self.borda_ema,
            "prior_rank_weight": self.prior_rank_weight,
        }
        for key, expected in immutable.items():
            observed = state.get(key, expected)
            if observed != expected:
                raise ValueError(
                    f"CMGRA checkpoint mismatch for {key}: "
                    f"expected {expected!r}, got {observed!r}"
                )
        self.max_malicious = int(state.get("max_malicious", self.max_malicious))
        memberships = state.get("memberships", {})
        rankings = state.get("initial_rankings", {})
        if not isinstance(memberships, Mapping) or not isinstance(rankings, Mapping):
            raise ValueError("invalid CMGRA checkpoint tensors")
        self._memberships = {
            str(key): value.detach().cpu().clone()
            for key, value in memberships.items()
        }
        self._initial_rankings = {
            str(key): value.detach().cpu().clone()
            for key, value in rankings.items()
        }
        tensor_groups = {
            "_global_rankings": state.get("global_rankings", {}),
            "_count_ema_sums": state.get("count_ema_sums", {}),
            "_candidate_memberships": state.get(
                "candidate_memberships", {}
            ),
            "_candidate_streaks": state.get("candidate_streaks", {}),
            "_borda_ema_scores": state.get("borda_ema_scores", {}),
        }
        for attribute, values in tensor_groups.items():
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"invalid CMGRA checkpoint group {attribute}"
                )
            setattr(
                self,
                attribute,
                {
                    str(key): value.detach().cpu().clone()
                    for key, value in values.items()
                },
            )
        if not self._global_rankings:
            self._global_rankings = {
                key: value.clone()
                for key, value in self._initial_rankings.items()
            }
        self._count_ema_weights = {
            str(key): float(value)
            for key, value in state.get(
                "count_ema_weights", {}
            ).items()
        }
        self._layer_rounds = {
            str(key): int(value)
            for key, value in state.get("layer_rounds", {}).items()
        }
