"""Certified Membership-Gated Ranking Aggregation (CMGRA).

The public API uses *rank values*: ``rankings[u, j]`` is the rank assigned
to edge ``j`` by client ``u`` and larger values mean more important edges.
The legacy VEM training code instead stores each row as edge ids from least to
most important.  ``CMGRAAggregator.aggregate_model`` explicitly adapts that
legacy representation at the server boundary.

Only the paper's CMGRA rule is implemented: the membership vector is derived
from each valid rank permutation, membership counts certify boundary changes,
and aggregate Borda scores order edges within the two groups.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class CMGRAUpdateBatch:
    """One layer's ranks and the memberships derived from those ranks."""

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
    """Stateful, layer-wise full-ranking CMGRA server aggregator.

    ``max_malicious`` is the certified upper bound ``B``. Membership state is
    retained across rounds and initialized from a deterministic public seed
    before any client update is processed.
    """

    def __init__(
        self,
        retention_ratio: float,
        max_malicious: int,
        public_seed: int = 0,
        invalid_policy: str = "drop",
    ) -> None:
        if not 0.0 <= retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in [0, 1]")
        if max_malicious < 0:
            raise ValueError("max_malicious must be non-negative")
        if invalid_policy not in {"drop", "raise"}:
            raise ValueError("invalid_policy must be 'drop' or 'raise'")
        self.retention_ratio = float(retention_ratio)
        self.max_malicious = int(max_malicious)
        self.public_seed = int(public_seed)
        self.invalid_policy = invalid_policy
        self._memberships: Dict[str, torch.Tensor] = {}
        self._initial_rankings: Dict[str, torch.Tensor] = {}
        self._global_rankings: Dict[str, torch.Tensor] = {}

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
            raise ValueError("cmgra requires full ranking uploads")
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
        borda_scores = rankings.sum(dim=0)
        edge_ids = torch.arange(
            num_edges, device=memberships.device, dtype=torch.long
        )
        candidate_order = edge_ids[
            torch.argsort(membership_counts, descending=True, stable=True)
        ]
        candidate = torch.zeros(
            num_edges, dtype=torch.bool, device=memberships.device
        )
        if k:
            candidate[candidate_order[:k]] = True

        if k == 0 or k == num_edges:
            margin: Optional[float] = None
            certified = True
        else:
            kth = float(membership_counts[candidate_order[k - 1]].item())
            next_count = float(membership_counts[candidate_order[k]].item())
            margin = kth - next_count
            certified = margin > self.max_malicious

        certified_swaps = 0
        if certified:
            global_membership = candidate
            update_mode = "full"
        else:
            # Whole-set certification is often unavailable at an integer-vote
            # boundary. Preserve K exactly while accepting only paired changes
            # whose honest preference is itself certified:
            #   h_out >= c_out - m > c_in >= h_in.
            incumbents = torch.nonzero(previous, as_tuple=False).flatten()
            challengers = torch.nonzero(~previous, as_tuple=False).flatten()
            weakest_incumbents = torch.flip(
                self._importance_order(
                    membership_counts, borda_scores, incumbents
                ),
                dims=[0],
            )
            strongest_challengers = self._importance_order(
                membership_counts, borda_scores, challengers
            )
            pair_count = min(
                weakest_incumbents.numel(), strongest_challengers.numel()
            )
            selected_pairs = torch.empty(
                0, dtype=torch.long, device=memberships.device
            )
            if pair_count:
                vote_gaps = (
                    membership_counts[strongest_challengers[:pair_count]]
                    - membership_counts[weakest_incumbents[:pair_count]]
                )
                selected_pairs = torch.nonzero(
                    vote_gaps > self.max_malicious, as_tuple=False
                ).flatten()
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
        changed_edges = int((global_membership != previous).sum().item())
        if int(global_membership.sum().item()) != k:
            raise RuntimeError("CMGRA invariant violated: global membership is not K-hot")
        self._memberships[layer_name] = global_membership.detach().clone()

        selected = torch.nonzero(global_membership, as_tuple=False).flatten()
        unselected = torch.nonzero(~global_membership, as_tuple=False).flatten()
        selected_order = self._borda_order(borda_scores, selected)
        unselected_order = self._borda_order(borda_scores, unselected)

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
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        immutable = {
            "retention_ratio": self.retention_ratio,
            "public_seed": self.public_seed,
            "invalid_policy": self.invalid_policy,
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
        global_rankings = state.get("global_rankings", {})
        if not isinstance(global_rankings, Mapping):
            raise ValueError("invalid CMGRA global rankings")
        self._global_rankings = {
            str(key): value.detach().cpu().clone()
            for key, value in global_rankings.items()
        }
        if not self._global_rankings:
            self._global_rankings = {
                key: value.clone()
                for key, value in self._initial_rankings.items()
            }
