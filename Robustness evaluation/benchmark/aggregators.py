"""Plaintext FedAvg, BCPBFL, and RVPFL robust aggregation rules."""

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import torch

TensorList = List[torch.Tensor]


@dataclass
class AggregationDiagnostics:
    accepted: int
    rejected: int
    weights: List[float]
    similarities: List[float]
    fallback: str = ""


def _zeros_like(update: Sequence[torch.Tensor]) -> TensorList:
    return [torch.zeros_like(tensor) for tensor in update]


def _weighted_sum(updates, weights: torch.Tensor) -> TensorList:
    result = _zeros_like(updates[0])
    for weight, update in zip(weights, updates):
        for destination, tensor in zip(result, update):
            destination.add_(tensor, alpha=float(weight))
    return result


def _dot(left, right) -> torch.Tensor:
    value = torch.zeros((), device=left[0].device, dtype=torch.float64)
    for a, b in zip(left, right):
        value += torch.sum(a.double() * b.double())
    return value


def _norm(update) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_dot(update, update), min=0.0))


def _is_finite_update(update: Sequence[torch.Tensor]) -> bool:
    """Return whether every coordinate in a client update is finite.

    Robust aggregation rules are defined over real-valued vectors.  A NaN or
    infinity is therefore an invalid upload, rather than an adversarial value
    that should participate with weight zero: IEEE arithmetic would otherwise
    propagate it through ``0 * NaN`` and poison the global model.
    """

    return all(bool(torch.isfinite(tensor).all()) for tensor in update)


def _finite_updates(updates):
    return [update for update in updates if _is_finite_update(update)]


def fedavg(updates) -> Tuple[TensorList, AggregationDiagnostics]:
    finite = _finite_updates(updates)
    if not finite:
        return _zeros_like(updates[0]), AggregationDiagnostics(
            0, len(updates), [], [], "no_finite_updates"
        )
    weights = torch.full(
        (len(finite),), 1.0 / len(finite),
        device=finite[0][0].device, dtype=torch.float64,
    )
    return _weighted_sum(finite, weights), AggregationDiagnostics(
        len(finite), len(updates) - len(finite), weights.cpu().tolist(), []
    )


def bcpbfl(updates, root_update, server_lr: float, eps: float = 1e-12):
    """BCPBFL Eqs. (6), (9), and (12), without encryption."""
    if not _is_finite_update(root_update):
        return _zeros_like(root_update), AggregationDiagnostics(
            0, len(updates), [], [], "nonfinite_root_update"
        )
    root_norm = _norm(root_update)
    if not bool(torch.isfinite(root_norm)) or float(root_norm) <= eps:
        return _zeros_like(root_update), AggregationDiagnostics(
            0, len(updates), [], [], "zero_root_update"
        )
    normalized_root = [tensor / root_norm.to(tensor.dtype) for tensor in root_update]
    normalized, similarities = [], []
    for update in _finite_updates(updates):
        norm = _norm(update)
        if not bool(torch.isfinite(norm)) or float(norm) <= eps:
            normalized.append(_zeros_like(update))
            similarities.append(0.0)
            continue
        unit = [tensor / norm.to(tensor.dtype) for tensor in update]
        normalized.append(unit)
        similarity = float(_dot(normalized_root, unit).clamp(-1, 1))
        similarities.append(similarity if math.isfinite(similarity) else 0.0)
    if not normalized:
        return _zeros_like(root_update), AggregationDiagnostics(
            0, len(updates), [], [], "no_finite_updates"
        )
    scores = torch.tensor(
        similarities, device=normalized[0][0].device, dtype=torch.float64
    ).clamp_min_(0)
    if float(scores.sum()) <= eps:
        return _zeros_like(root_update), AggregationDiagnostics(
            0, len(updates), [], similarities, "no_positive_cosine"
        )
    weights = scores / scores.sum()
    aggregate = _weighted_sum(normalized, weights)
    for tensor in aggregate:
        tensor.mul_(server_lr)
    accepted = int((scores > 0).sum())
    return aggregate, AggregationDiagnostics(
        accepted, len(updates) - accepted, weights.cpu().tolist(), similarities
    )


def _coordinate_median(updates) -> TensorList:
    return [torch.median(torch.stack(tensors), dim=0).values for tensors in zip(*updates)]


def _joint_mean(update, benchmark) -> torch.Tensor:
    total = torch.zeros((), device=update[0].device, dtype=torch.float64)
    dimensions = 0
    for tensor, reference in zip(update, benchmark):
        total += tensor.double().sum() + reference.double().sum()
        dimensions += tensor.numel()
    return total / (2 * dimensions)


def _adjusted_cosine(update, benchmark, eps: float) -> float:
    mean = _joint_mean(update, benchmark)
    numerator = torch.zeros((), device=update[0].device, dtype=torch.float64)
    norm_update = torch.zeros_like(numerator)
    norm_benchmark = torch.zeros_like(numerator)
    for tensor, reference in zip(update, benchmark):
        centered_update = tensor.double() - mean
        centered_reference = reference.double() - mean
        numerator += torch.sum(centered_update * centered_reference)
        norm_update += torch.sum(centered_update.square())
        norm_benchmark += torch.sum(centered_reference.square())
    denominator = torch.sqrt(norm_update * norm_benchmark)
    if float(denominator) <= eps:
        return 0.0
    return float((numerator / denominator).clamp(-1, 1))


def rvpfl(updates, eps: float = 1e-12):
    """RVPFL Algorithm lines 9 and 16--18, without encryption."""
    finite = _finite_updates(updates)
    if not finite:
        return _zeros_like(updates[0]), AggregationDiagnostics(
            0, len(updates), [], [], "no_finite_updates"
        )
    benchmark = _coordinate_median(finite)
    similarities = []
    for update in finite:
        similarity = _adjusted_cosine(update, benchmark, eps)
        similarities.append(similarity if math.isfinite(similarity) else 0.0)
    scores = torch.tensor(
        similarities, device=finite[0][0].device, dtype=torch.float64
    ).clamp_min_(0)
    scores.square_()
    positive = scores > 0
    if not bool(positive.any()):
        return benchmark, AggregationDiagnostics(
            0, len(updates), [], similarities, "median_fallback"
        )
    weights = scores / scores.sum()
    aggregate = _weighted_sum(finite, weights)
    accepted = int(positive.sum())
    return aggregate, AggregationDiagnostics(
        accepted, len(updates) - accepted, weights.cpu().tolist(), similarities
    )
