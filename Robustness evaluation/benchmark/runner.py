"""One resumable 500-round experiment for the robustness benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import VEM
from cmgra import CMGRAAggregator, CMGRAUpdateBatch, retained_edge_count

from . import (
    ALL_ATTACKS,
    ALL_METHODS,
    DATASETS,
    MNIST_LIKE_DATASETS,
    attack_applies,
    method_family,
)
from .data import (
    FederatedData,
    client_loader,
    fixed_paper_backdoor_dataset,
    load_federated_data,
    root_loader,
    stamp_trigger,
    test_loader,
    validation_loader,
)

from .aggregators import TensorList, bcpbfl, fedavg, rvpfl
from .models import (
    build_model,
    communicated_parameter_count,
    scored_modules,
    trainable_parameters,
)


@dataclass
class RunConfig:
    dataset: str
    model: str
    method: str
    attack: str
    malicious_fraction: float
    rounds: int
    seed: int
    partition_seed: int
    data_partition: str
    partition_file: str
    n_clients: int
    round_clients: int
    local_epochs: int
    batch_size: int
    test_batch_size: int
    local_lr: float
    lr_decay: float
    global_cosine_start: int
    global_min_lr: float
    momentum: float
    weight_decay: float
    local_cosine: bool
    keep_ratio: float
    rank_weight_init: str
    svhn_batchnorm: bool
    bcp_server_lr: float
    root_size: int
    noise_scale: float
    # Retained only so checkpoints written before the attack was removed
    # remain resumable. It is not associated with any supported attack.
    shuffle_scale: float
    label_flip_target: int
    backdoor_fraction: float
    backdoor_target: int
    trigger_size: int
    backdoor_examples: int
    cmgra_seed: int
    cmgra_pairwise: bool
    cmgra_pair_matching: str
    cmgra_preserve_order: bool
    cmgra_membership_order: bool
    cmgra_borda_order: bool
    cmgra_borda_final_order: bool
    cmgra_count_ema: float
    cmgra_swap_cap: int
    cmgra_persistence_rounds: int
    cmgra_update_interval: int
    cmgra_borda_mode: str
    cmgra_rank_buckets: int
    cmgra_borda_ema: float
    cmgra_prior_rank_weight: float
    cmgra_hybrid_mode: str
    cmgra_hybrid_warmup_rounds: int
    cmgra_hybrid_val_window: int
    cmgra_hybrid_val_drop: float
    cmgra_hybrid_patience: int
    vem_lr: float
    vem_epochs: int
    vem_max_window: int
    vem_temperature: float
    vem_sinkhorn_iterations: int
    vem_noise: float
    vem_source_compatible: bool
    rvpfl_paper_extra_division: bool
    checkpoint_interval: int
    data_root: str
    output_root: str
    num_workers: int
    device: str


def _stable_seed(*values: object) -> int:
    text = ":".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**31)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _round_learning_rate(config: RunConfig, round_index: int) -> float:
    """Return the base LR for one communication round.

    The legacy schedule remains the default.  When ``global_cosine_start`` is
    non-negative, the same exponential schedule is used before that round and
    then decays smoothly from its value at the transition to
    ``global_min_lr`` at the final communication round.
    """

    if config.global_cosine_start < 0:
        return config.local_lr * (config.lr_decay**round_index)
    start = config.global_cosine_start
    if round_index < start:
        return config.local_lr * (config.lr_decay**round_index)
    start_lr = config.local_lr * (config.lr_decay**start)
    denominator = max(config.rounds - 1 - start, 1)
    progress = min(max((round_index - start) / denominator, 0.0), 1.0)
    return config.global_min_lr + 0.5 * (
        start_lr - config.global_min_lr
    ) * (1.0 + math.cos(math.pi * progress))


def _fraction_tag(value: float) -> str:
    return f"{int(round(value * 100)):02d}"


def run_directory(config: RunConfig) -> Path:
    return (
        Path(config.output_root)
        / config.dataset
        / config.method
        / config.attack
        / f"malicious_{_fraction_tag(config.malicious_fraction)}"
        / f"seed_{config.seed}"
    )


def round_malicious_count(
    round_index: int, round_clients: int, fraction: float
) -> int:
    """Alternate fractional counts so the multi-round mean is exact."""

    if fraction <= 0:
        return 0
    before = math.floor(round_index * round_clients * fraction + 1e-9)
    after = math.floor((round_index + 1) * round_clients * fraction + 1e-9)
    return after - before


def selected_clients(config: RunConfig, round_index: int) -> Tuple[np.ndarray, set]:
    rng = np.random.default_rng(
        _stable_seed("participants", config.seed, round_index)
    )
    users = rng.choice(config.n_clients, config.round_clients, replace=False)
    if config.attack == "vem" and config.vem_source_compatible:
        # VEM-master/FL_train.py fixes the number of malicious uploads in
        # every round with int(round_nclients * at_fractions).  Preserve that
        # behaviour, including floor rounding for fractions such as 10%.
        malicious_count = int(
            config.round_clients * config.malicious_fraction
        )
    else:
        malicious_count = round_malicious_count(
            round_index, config.round_clients, config.malicious_fraction
        )
    positions = rng.choice(
        config.round_clients, malicious_count, replace=False
    )
    malicious = {int(users[position]) for position in positions}
    return users.astype(np.int64), malicious


def _parameter_list(model: nn.Module) -> List[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _model_delta(local: nn.Module, global_model: nn.Module) -> TensorList:
    return [
        local_parameter.detach() - global_parameter.detach()
        for local_parameter, global_parameter in zip(
            _parameter_list(local), _parameter_list(global_model)
        )
    ]


def _append_backdoor_copies(
    images: torch.Tensor,
    labels: torch.Tensor,
    config: RunConfig,
    generator: torch.Generator,
    copies: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if copies <= 0:
        return images, labels
    candidates = torch.arange(labels.numel(), device=labels.device)
    order = torch.randperm(
        candidates.numel(), generator=generator, device=images.device
    )
    selected = candidates[order[: min(copies, candidates.numel())]]
    poisoned = stamp_trigger(
        images[selected], config.dataset, config.trigger_size
    )
    target = torch.full(
        (poisoned.shape[0],),
        config.backdoor_target,
        device=labels.device,
        dtype=labels.dtype,
    )
    return (
        torch.cat((images, poisoned), dim=0),
        torch.cat((labels, target), dim=0),
    )


def _train_local_model(
    global_model: nn.Module,
    loader: Iterable,
    config: RunConfig,
    round_index: int,
    client_id: int,
    malicious: bool,
) -> nn.Module:
    local = copy.deepcopy(global_model)
    local.train()
    lr = _round_learning_rate(config, round_index)
    optimizer = optim.SGD(
        trainable_parameters(local),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.local_epochs
        )
        if config.local_cosine
        else None
    )
    criterion = nn.CrossEntropyLoss()
    local_seed = _stable_seed(
        "local", config.seed, round_index, client_id
    )
    _set_seed(local_seed)
    device = next(local.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(local_seed + 17)

    for _ in range(config.local_epochs):
        backdoor_seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if malicious and config.attack == "label_flip":
                labels = config_num_classes(config) - labels - 1
            elif malicious and config.attack == "label_flip_all_to_one":
                labels = torch.full_like(labels, config.label_flip_target)
            elif malicious and config.attack == "pixel_backdoor":
                desired_before = math.floor(
                    backdoor_seen * config.backdoor_fraction + 1e-9
                )
                backdoor_seen += images.shape[0]
                desired_after = math.floor(
                    backdoor_seen * config.backdoor_fraction + 1e-9
                )
                images, labels = _append_backdoor_copies(
                    images,
                    labels,
                    config,
                    generator,
                    desired_after - desired_before,
                )
            optimizer.zero_grad(set_to_none=True)
            logits = local(images)
            loss = criterion(logits, labels)
            if malicious and config.attack == "grad_ascent":
                loss = -loss
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return local


def config_num_classes(config: RunConfig) -> int:
    del config
    return 10


def _add_gaussian_noise(
    tensors: Sequence[torch.Tensor], scale: float, seed: int
) -> TensorList:
    _set_seed(seed)
    squared = torch.zeros((), device=tensors[0].device, dtype=torch.float64)
    dimensions = 0
    for tensor in tensors:
        squared += tensor.double().square().sum()
        dimensions += tensor.numel()
    rms = torch.sqrt(squared / max(dimensions, 1)).to(tensors[0].dtype)
    return [
        tensor + torch.randn_like(tensor) * rms * scale for tensor in tensors
    ]


def _rank_values(scores: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores.detach().abs().flatten(), stable=True)
    ranks = torch.empty_like(order, dtype=torch.int32)
    ranks[order] = torch.arange(
        order.numel(), device=order.device, dtype=torch.int32
    )
    return ranks


def _vem_source_edge_order(scores: torch.Tensor) -> torch.Tensor:
    """Reproduce VEM-master.utils.Find_rank exactly.

    The published code sorts the raw score tensor, not ``abs(scores)``, and
    returns edge IDs from the smallest score to the largest score.
    """

    return torch.sort(scores.detach().flatten())[1].detach()


def _vem_source_rank_values(scores: torch.Tensor) -> torch.Tensor:
    """Return the benign row seen inside VEM-master.utils.FRL_Vote."""

    edge_order = _vem_source_edge_order(scores)
    return torch.sort(edge_order)[1].to(torch.int32)


def _rank_upload(
    local: nn.Module,
    config: RunConfig,
    round_index: int,
    client_id: int,
    malicious: bool,
) -> Dict[str, torch.Tensor]:
    upload = {}
    for layer_index, (name, module) in enumerate(scored_modules(local).items()):
        scores = module.scores.detach()
        if config.attack == "vem" and config.vem_source_compatible:
            # VEM-master creates r_s with Find_rank(raw_scores).  This
            # benchmark stores all server uploads canonically as per-edge
            # rank values, so materialize the exact row that the source
            # FRL_Vote obtains after its torch.sort(...)[1].
            upload[name] = _vem_source_rank_values(scores)
        elif malicious and config.attack == "gaussian_noise":
            noisy = _add_gaussian_noise(
                [scores],
                config.noise_scale,
                _stable_seed(
                    "rank-noise",
                    config.seed,
                    round_index,
                    client_id,
                    layer_index,
                ),
            )[0]
            upload[name] = _rank_values(noisy)
        else:
            upload[name] = _rank_values(scores)
    return upload


def _initial_score_levels(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: module.scores.detach().abs().flatten().sort().values
        for name, module in scored_modules(model).items()
    }


def _apply_rank_values(
    model: nn.Module,
    score_levels: Mapping[str, torch.Tensor],
    ranks: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, module in scored_modules(model).items():
            module.scores.copy_(
                score_levels[name][ranks[name].long()].reshape_as(module.scores)
            )


def _aggregate_rank_round(
    model: nn.Module,
    uploads: Sequence[Mapping[str, torch.Tensor]],
    malicious_rows: Sequence[int],
    score_levels: Mapping[str, torch.Tensor],
    config: RunConfig,
    round_index: int,
    cmgra: CMGRAAggregator | None,
    vem_state: MutableMapping[str, Dict[str, torch.Tensor]],
    force_frl: bool = False,
) -> Dict[str, object]:
    uploads = [dict(upload) for upload in uploads]
    vem_audit = ""
    if config.attack == "vem" and malicious_rows:
        for layer_index, name in enumerate(scored_modules(model)):
            source_rank_values = torch.stack(
                [uploads[row][name].long() for row in malicious_rows], dim=0
            )
            # The legacy VEM implementation accepts edge-order permutations
            # and internally converts them to per-edge rank values. The
            # benchmark upload representation is already per-edge rank
            # values, so convert it exactly once before calling VEM.
            source_edge_orders = torch.sort(
                source_rank_values, dim=1
            )[1]
            _set_seed(
                _stable_seed("vem", config.seed, round_index, layer_index)
            )
            try:
                has_history = (
                    config.vem_source_compatible
                    and round_index > 0
                    and name in vem_state["global_orders"]
                    and name in vem_state["malicious_rankings"]
                )
                common = (
                    config.keep_ratio,
                    len(malicious_rows),
                    source_rank_values.device,
                    config.vem_lr,
                    config.vem_epochs,
                    config.vem_max_window,
                    config.vem_temperature,
                    config.vem_sinkhorn_iterations,
                    config.vem_noise,
                )
                if has_history:
                    # Follows F:/VEM-master/FL_train.py: the published
                    # implementation uses the alternative estimate only in
                    # round zero and the historical estimate thereafter.
                    attacked = VEM.optimize_his(
                        config.round_clients,
                        source_edge_orders,
                        vem_state["global_orders"][name],
                        vem_state["malicious_rankings"][name],
                        *common,
                    )
                    attack_mode = "his"
                else:
                    attacked = VEM.optimize_alt(
                        config.round_clients,
                        source_edge_orders,
                        *common,
                    )
                    attack_mode = "alt"
            except (RuntimeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    f"VEM failed at layer {name!r}, round {round_index + 1}"
                ) from exc

            if config.vem_source_compatible:
                # The source VEM routine returns per-edge rank values, while
                # its FRL_Vote path applies torch.sort(...)[1] once more before
                # Borda summation.  Reproduce that server-visible permutation
                # exactly while keeping the benchmark's canonical rank-value
                # representation for every uploaded row.
                server_visible = torch.sort(attacked, dim=1)[1]
                vem_state["malicious_rankings"][name] = (
                    attacked.detach().clone()
                )
            else:
                server_visible = attacked

            changed = int((server_visible != source_rank_values).sum().item())
            total = int(source_rank_values.numel())
            dimensions = source_rank_values.shape[1]
            keep = retained_edge_count(dimensions, config.keep_ratio)
            before_membership = source_rank_values >= (dimensions - keep)
            after_membership = server_visible >= (dimensions - keep)
            topk_flips = int(
                (before_membership ^ after_membership).sum().item() // 2
            )
            audit_piece = (
                f"{name}:mode={attack_mode}:changed={changed}/{total}:"
                f"topk_flips={topk_flips}"
            )
            vem_audit = (
                audit_piece if not vem_audit else f"{vem_audit};{audit_piece}"
            )
            for source_row, upload_row in enumerate(malicious_rows):
                uploads[upload_row][name] = server_visible[source_row].to(
                    torch.int32
                )

    # CMGRA-PX is threat-budget adaptive.  With a certified adversary bound
    # m=0 there is no boundary attack to reject, so retain exact compatibility
    # with the original FRL Borda update.  Besides avoiding unnecessary
    # membership-boundary churn, this makes the clean arm measure the cost of
    # the defence without changing FRL's benign optimization trajectory.
    clean_borda_compat = (
        config.method == "cmgra-px"
        and config.attack == "clean"
        and not malicious_rows
    )
    if force_frl or config.method == "frl" or clean_borda_compat:
        global_ranks = {}
        for name in scored_modules(model):
            borda = torch.stack(
                [upload[name].long() for upload in uploads], dim=0
            ).sum(dim=0)
            order = torch.argsort(borda, stable=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(
                order.numel(), device=order.device, dtype=torch.long
            )
            global_ranks[name] = ranks
        _apply_rank_values(model, score_levels, global_ranks)
        if config.attack == "vem" and config.vem_source_compatible:
            for name, module in scored_modules(model).items():
                vem_state["global_orders"][name] = _vem_source_edge_order(
                    module.scores.detach()
                )
        return {
            "accepted": len(uploads),
            "rejected": 0,
            "certified_layers": "",
            "updated_layers": len(global_ranks),
            "pairwise_swaps": 0,
            "fallback": (
                "clean_borda_compat"
                if clean_borda_compat
                else (
                    ("hybrid_frl|" if force_frl else "")
                    + ("source_vem|" + vem_audit if vem_audit else "")
                )
            ),
        }

    if cmgra is None:
        raise RuntimeError("CMGRA state is missing")
    cmgra.max_malicious = len(malicious_rows)
    summaries = {}
    with torch.no_grad():
        for name, module in scored_modules(model).items():
            dimensions = uploads[0][name].numel()
            keep = retained_edge_count(dimensions, config.keep_ratio)
            rankings = torch.stack(
                [upload[name].long() for upload in uploads],
                dim=0,
            )
            memberships = rankings >= (dimensions - keep)
            result = cmgra.aggregate_layer(
                name,
                CMGRAUpdateBatch(rankings, memberships),
                input_format="rank_values",
            )
            module.scores.copy_(
                score_levels[name][result.global_ranking.long()].reshape_as(
                    module.scores
                )
            )
            summaries[name] = result
    if config.attack == "vem" and config.vem_source_compatible:
        for name, module in scored_modules(model).items():
            vem_state["global_orders"][name] = _vem_source_edge_order(
                module.scores.detach()
            )
    return {
        "accepted": min(result.valid_clients for result in summaries.values()),
        "rejected": max(result.rejected_clients for result in summaries.values()),
        "certified_layers": (
            f"{sum(result.certified for result in summaries.values())}/"
            f"{len(summaries)}"
        ),
        "updated_layers": sum(
            result.update_mode in {"full", "pairwise"}
            for result in summaries.values()
        ),
        "pairwise_swaps": sum(
            result.certified_swaps for result in summaries.values()
        ),
        "fallback": "source_vem|" + vem_audit if vem_audit else "",
    }


def _apply_gradient_update(model: nn.Module, update: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, delta in zip(_parameter_list(model), update):
            parameter.add_(delta)


def _root_update(
    model: nn.Module,
    data: FederatedData,
    config: RunConfig,
    round_index: int,
) -> TensorList:
    loader = root_loader(
        data,
        config.batch_size,
        _stable_seed("root-loader", config.seed, round_index),
        config.num_workers,
    )
    root = _train_local_model(
        model, loader, config, round_index, -1, malicious=False
    )
    update = _model_delta(root, model)
    del root
    return update


def _aggregate_gradient_round(
    model: nn.Module,
    updates: Sequence[Sequence[torch.Tensor]],
    data: FederatedData,
    config: RunConfig,
    round_index: int,
) -> Dict[str, object]:
    if config.method == "fedavg":
        aggregate, diagnostics = fedavg(updates)
    elif config.method == "bcpbfl":
        root = _root_update(model, data, config, round_index)
        aggregate, diagnostics = bcpbfl(updates, root, config.bcp_server_lr)
        del root
    elif config.method == "rvpfl":
        aggregate, diagnostics = rvpfl(
            updates, config.rvpfl_paper_extra_division
        )
    else:
        raise ValueError(config.method)
    _apply_gradient_update(model, aggregate)
    return {
        "accepted": diagnostics.accepted,
        "rejected": diagnostics.rejected,
        "certified_layers": "",
        "updated_layers": "",
        "pairwise_swaps": "",
        "fallback": diagnostics.fallback,
        "min_similarity": min(diagnostics.similarities) if diagnostics.similarities else "",
        "max_similarity": max(diagnostics.similarities) if diagnostics.similarities else "",
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    config: RunConfig,
) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    attack_success = 0
    attack_total = 0
    device = next(model.parameters()).device
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        logits = model(images)
        loss_sum += float(criterion(logits, labels))
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
        if config.attack in ("pixel_backdoor", "pixel_backdoor_low_data"):
            eligible = labels != config.backdoor_target
            if bool(eligible.any()):
                triggered = stamp_trigger(
                    images[eligible],
                    config.dataset,
                    config.trigger_size,
                    pattern=(
                        "paper-f"
                        if config.attack == "pixel_backdoor_low_data"
                        else "square"
                    ),
                )
                predictions = model(triggered).argmax(dim=1)
                attack_success += int(
                    (predictions == config.backdoor_target).sum()
                )
                attack_total += predictions.numel()
    asr = attack_success / attack_total if attack_total else float("nan")
    return correct / total, loss_sum / total, asr


METRIC_COLUMNS = (
    "round",
    "test_acc",
    "test_loss",
    "backdoor_asr",
    "best_acc",
    "val_acc",
    "val_loss",
    "hybrid_phase",
    "hybrid_switch",
    "hybrid_best_round",
    "malicious_clients",
    "accepted",
    "rejected",
    "certified_layers",
    "updated_layers",
    "pairwise_swaps",
    "fallback",
    "min_similarity",
    "max_similarity",
    "round_seconds",
    "elapsed_seconds",
)


def _write_metric(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def _last_logged_round(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["round"]) if rows else 0


def _truncate_metrics(path: Path, completed_rounds: int) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["round"]) <= completed_rounds
        ]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _capture_hybrid_snapshot(
    model: nn.Module,
    vem_state: Mapping[str, Mapping[str, torch.Tensor]],
    round_number: int,
) -> Dict[str, object]:
    return {
        "round": int(round_number),
        "model": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "vem_state": {
            group: {
                name: tensor.detach().cpu().clone()
                for name, tensor in layers.items()
            }
            for group, layers in vem_state.items()
        },
    }


def _restore_hybrid_snapshot(
    snapshot: Mapping[str, object],
    model: nn.Module,
    vem_state: MutableMapping[str, Dict[str, torch.Tensor]],
) -> None:
    device = next(model.parameters()).device
    model_state = snapshot.get("model")
    stored_vem = snapshot.get("vem_state")
    if not isinstance(model_state, Mapping) or not isinstance(
        stored_vem, Mapping
    ):
        raise ValueError("invalid hybrid rollback snapshot")
    model.load_state_dict(
        {
            str(name): tensor.to(device)
            for name, tensor in model_state.items()
        }
    )
    restored_vem: Dict[str, Dict[str, torch.Tensor]] = {}
    for group, layers in stored_vem.items():
        if not isinstance(layers, Mapping):
            raise ValueError("invalid hybrid VEM rollback state")
        restored_vem[str(group)] = {
            str(name): tensor.to(device)
            for name, tensor in layers.items()
        }
    vem_state.clear()
    vem_state.update(restored_vem)


def _hybrid_rankings_from_model(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    return {
        name: _rank_values(module.scores.detach()).long()
        for name, module in scored_modules(model).items()
    }


def _hybrid_triggered(
    mode: str,
    completed_round: int,
    warmup_rounds: int,
    window_ready: bool,
    signal: float,
    best_signal: float,
    drop: float,
    bad_streak: int,
    patience: int,
) -> bool:
    if mode == "fixed":
        return completed_round >= warmup_rounds
    return (
        mode == "adaptive"
        and completed_round >= warmup_rounds
        and window_ready
        and best_signal - signal >= drop
        and bad_streak >= patience
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    next_round: int,
    best_acc: float,
    elapsed_seconds: float,
    cmgra: CMGRAAggregator | None,
    vem_state: Mapping[str, Mapping[str, torch.Tensor]],
    hybrid_state: Mapping[str, object] | None = None,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "next_round": next_round,
            "best_acc": best_acc,
            "elapsed_seconds": elapsed_seconds,
            "cmgra": cmgra.state_dict() if cmgra is not None else None,
            "vem_state": {
                group: {
                    name: tensor.detach().clone()
                    for name, tensor in layers.items()
                }
                for group, layers in vem_state.items()
            },
            "hybrid_state": copy.deepcopy(hybrid_state),
        },
        temporary,
    )
    os.replace(temporary, path)


def run(config: RunConfig) -> Path:
    if config.method not in ALL_METHODS:
        raise ValueError(f"unknown method: {config.method}")
    if config.attack not in ALL_ATTACKS:
        raise ValueError(f"unknown attack: {config.attack}")
    if not attack_applies(config.method, config.attack):
        raise ValueError(
            f"attack {config.attack!r} does not apply to {config.method!r}"
        )
    if config.attack == "clean" and config.malicious_fraction != 0:
        raise ValueError("clean runs must use malicious_fraction=0")
    if config.attack != "clean" and not 0 < config.malicious_fraction < 0.5:
        raise ValueError("attacked runs require malicious_fraction in (0, 0.5)")
    hybrid_enabled = config.cmgra_hybrid_mode != "none"
    if hybrid_enabled and config.method != "cmgra-px":
        raise ValueError("CMGRA hybrid modes require method='cmgra-px'")
    if hybrid_enabled and config.root_size <= 0:
        raise ValueError("CMGRA hybrid modes require a non-empty root set")

    output = run_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    metrics_path = output / "metrics.csv"
    checkpoint_path = output / "checkpoint.pt"
    config_payload = asdict(config)
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        # Old campaigns may contain fields from retired datasets or ablations.
        # Compare only the active protocol schema when resuming target runs.
        previous = {
            key: value
            for key, value in previous.items()
            if key in config_payload or key == "rounds"
        }
        # A short-lived manifest revision omitted this now-inactive
        # compatibility field. Normalize both old checkpoint variants.
        previous.setdefault("shuffle_scale", 10000.0)
        previous.setdefault("label_flip_target", 0)
        previous.setdefault("vem_source_compatible", True)
        previous.setdefault("local_cosine", False)
        previous.setdefault("global_cosine_start", -1)
        previous.setdefault("global_min_lr", 0.02)
        previous.setdefault("data_partition", "iid")
        previous.setdefault("partition_file", "")
        previous.setdefault("model", "auto")
        previous.setdefault("rank_weight_init", "signed-constant")
        previous.setdefault("svhn_batchnorm", False)
        previous.setdefault("cmgra_preserve_order", False)
        previous.setdefault("cmgra_membership_order", False)
        previous.setdefault("cmgra_borda_order", False)
        previous.setdefault("cmgra_borda_final_order", False)
        previous.setdefault("cmgra_pair_matching", "extreme")
        previous.setdefault("cmgra_count_ema", 0.0)
        previous.setdefault("cmgra_swap_cap", 0)
        previous.setdefault("cmgra_persistence_rounds", 1)
        previous.setdefault("cmgra_update_interval", 1)
        previous.setdefault("cmgra_borda_mode", "sum")
        previous.setdefault("cmgra_rank_buckets", 0)
        previous.setdefault("cmgra_borda_ema", 0.0)
        previous.setdefault("cmgra_prior_rank_weight", 0.0)
        previous.setdefault("cmgra_hybrid_mode", "none")
        previous.setdefault("cmgra_hybrid_warmup_rounds", 20)
        previous.setdefault("cmgra_hybrid_val_window", 5)
        previous.setdefault("cmgra_hybrid_val_drop", 0.015)
        previous.setdefault("cmgra_hybrid_patience", 5)
        # Ignore retired fields written by the former multi-method and
        # FEMNIST benchmark.  They never affected FRL/CMGRA-PX updates.
        previous.setdefault("rvpfl_paper_extra_division", False)
        previous_rounds = int(previous.pop("rounds", config.rounds))
        current_without_rounds = dict(config_payload)
        current_rounds = int(current_without_rounds.pop("rounds"))
        if previous != current_without_rounds or current_rounds < previous_rounds:
            raise ValueError(f"existing run configuration differs: {output}")
        if current_rounds > previous_rounds:
            config_path.write_text(
                json.dumps(config_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    else:
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    _set_seed(config.seed)
    device = torch.device(config.device)
    family = method_family(config.method)
    rank_based = family == "rank"
    model = build_model(
        config.dataset,
        rank_based,
        keep_ratio=config.keep_ratio,
        rank_weight_init=config.rank_weight_init,
        svhn_batchnorm=config.svhn_batchnorm,
        model_name=config.model,
    ).to(device)
    parameter_count = communicated_parameter_count(model, rank_based)
    score_levels = _initial_score_levels(model) if rank_based else {}
    cmgra = None
    vem_state: MutableMapping[str, Dict[str, torch.Tensor]] = {
        "global_orders": {},
        "malicious_rankings": {},
    }
    if config.method == "cmgra-px":
        cmgra = CMGRAAggregator(
            retention_ratio=config.keep_ratio,
            max_malicious=0,
            public_seed=config.cmgra_seed,
            invalid_policy="raise",
            pairwise_fallback=config.cmgra_pairwise,
            pairwise_matching=config.cmgra_pair_matching,
            aggregation_variant="cmgra-px",
            preserve_group_order=config.cmgra_preserve_order,
            membership_group_order=config.cmgra_membership_order,
            borda_group_order=config.cmgra_borda_order,
            borda_final_order=config.cmgra_borda_final_order,
            count_ema=config.cmgra_count_ema,
            swap_cap=(
                config.cmgra_swap_cap
                if config.cmgra_swap_cap > 0
                else None
            ),
            persistence_rounds=config.cmgra_persistence_rounds,
            update_interval=config.cmgra_update_interval,
            borda_mode=config.cmgra_borda_mode,
            rank_buckets=config.cmgra_rank_buckets,
            borda_ema=config.cmgra_borda_ema,
            prior_rank_weight=config.cmgra_prior_rank_weight,
        )
        # Under the zero-threat clean protocol, CMGRA-PX deliberately reduces
        # to FRL.  Keep FRL's original public score initialization as well as
        # its Borda update so the two clean trajectories are directly paired.
        if config.attack != "clean":
            cmgra.initialize_model(model, score_levels)

    data = load_federated_data(
        config.dataset,
        config.data_root,
        config.n_clients,
        config.partition_seed,
        config.root_size,
        partition=config.data_partition,
        partition_file=config.partition_file or None,
        disjoint_root=hybrid_enabled,
    )
    evaluation_loader = test_loader(
        data, config.test_batch_size, config.num_workers
    )
    fixed_backdoor = (
        fixed_paper_backdoor_dataset(
            data,
            config.dataset,
            config.backdoor_target,
            config.backdoor_examples,
            config.trigger_size,
        )
        if config.attack == "pixel_backdoor_low_data"
        else None
    )
    heldout_loader = (
        validation_loader(
            data, config.test_batch_size, config.num_workers
        )
        if hybrid_enabled
        else None
    )
    start_round = 0
    best_acc = 0.0
    elapsed_before = 0.0
    hybrid_state: Dict[str, object] | None = (
        {
            "phase": "frl",
            "val_history": [],
            "best_signal": float("-inf"),
            "best_round": 0,
            "bad_streak": 0,
            "best_snapshot": None,
        }
        if hybrid_enabled
        else None
    )
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_round = int(checkpoint["next_round"])
        best_acc = float(checkpoint["best_acc"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        if cmgra is not None:
            cmgra.load_state_dict(checkpoint["cmgra"])
        stored_vem_state = checkpoint.get("vem_state")
        if stored_vem_state is not None:
            vem_state = {
                group: {
                    name: tensor.to(device)
                    for name, tensor in layers.items()
                }
                for group, layers in stored_vem_state.items()
            }
        elif (
            config.attack == "vem"
            and config.vem_source_compatible
            and start_round > 0
        ):
            raise RuntimeError(
                "source-compatible VEM checkpoint is missing attack history"
            )
        if hybrid_enabled:
            stored_hybrid = checkpoint.get("hybrid_state")
            if not isinstance(stored_hybrid, Mapping):
                raise RuntimeError(
                    "hybrid checkpoint is missing switching state"
                )
            hybrid_state = copy.deepcopy(dict(stored_hybrid))
        logged = _last_logged_round(metrics_path)
        if logged > start_round:
            # A process may stop after writing metrics but before the next
            # periodic checkpoint. Rewind the CSV to the last durable state.
            _truncate_metrics(metrics_path, start_round)
        elif logged < start_round:
            raise RuntimeError(
                f"checkpoint is ahead of metrics: checkpoint={start_round}, "
                f"metrics={logged}"
            )
    elif _last_logged_round(metrics_path):
        _truncate_metrics(metrics_path, 0)
    if start_round >= config.rounds:
        return output

    print(
        json.dumps(
            {
                "event": "start",
                "output": str(output),
                "dataset": config.dataset,
                "method": config.method,
                "attack": config.attack,
                "fraction": config.malicious_fraction,
                "round": start_round + 1,
                "parameters": parameter_count,
                "communication_mib": parameter_count * 4 / (1024**2),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    wall_start = time.time()
    for round_index in range(start_round, config.rounds):
        round_start = time.time()
        users, malicious_ids = selected_clients(config, round_index)
        if family == "gradient":
            updates = []
            for client_id in users:
                client_id = int(client_id)
                malicious = client_id in malicious_ids
                loader = client_loader(
                    data,
                    client_id,
                    config.batch_size,
                    _stable_seed("client-loader", config.seed, round_index, client_id),
                    config.num_workers,
                    fixed_backdoor if malicious else None,
                )
                # A finite gradient-ascent attack reverses the locally learned
                # descent delta.  Directly minimizing ``-loss`` for several
                # local epochs can overflow Conv8/BatchNorm and turn the
                # experiment into NaN injection, which is a malformed upload
                # rather than the intended adversarial-direction attack.
                reverse_delta = malicious and config.attack == "grad_ascent"
                local = _train_local_model(
                    model,
                    loader,
                    config,
                    round_index,
                    client_id,
                    malicious and not reverse_delta,
                )
                update = _model_delta(local, model)
                if reverse_delta:
                    update = [-tensor for tensor in update]
                if malicious and config.attack == "gaussian_noise":
                    update = _add_gaussian_noise(
                        update,
                        config.noise_scale,
                        _stable_seed("gradient-noise", config.seed, round_index, client_id),
                    )
                updates.append(update)
                del local
            diagnostics = _aggregate_gradient_round(
                model, updates, data, config, round_index
            )
            del updates
        else:
            uploads = []
            malicious_rows = []
            if (
                config.attack == "vem"
                and config.vem_source_compatible
                and malicious_ids
            ):
                # VEM-master does not train the selected malicious identities
                # to estimate r_s.  It samples the same number of source
                # clients from [0, int(nClients * at_fractions)) and uses
                # their locally trained rankings to construct all malicious
                # uploads.  Keep benign users first and malicious source
                # clients last, matching FL_train.py's user_updates layout.
                benign_users = [
                    int(client_id)
                    for client_id in users
                    if int(client_id) not in malicious_ids
                ]
                attacker_population = int(
                    config.n_clients * config.malicious_fraction
                )
                source_rng = np.random.default_rng(
                    _stable_seed(
                        "vem-source-clients",
                        config.seed,
                        round_index,
                    )
                )
                malicious_source_users = source_rng.choice(
                    attacker_population,
                    len(malicious_ids),
                    replace=False,
                ).astype(np.int64).tolist()
                rank_clients = benign_users + malicious_source_users
                malicious_start = len(benign_users)
            else:
                rank_clients = [int(client_id) for client_id in users]
                malicious_start = None

            for row, client_id in enumerate(rank_clients):
                client_id = int(client_id)
                malicious = (
                    row >= malicious_start
                    if malicious_start is not None
                    else client_id in malicious_ids
                )
                loader = client_loader(
                    data,
                    client_id,
                    config.batch_size,
                    _stable_seed(
                        "client-loader", config.seed, round_index, client_id
                    ),
                    config.num_workers,
                    fixed_backdoor if malicious else None,
                )
                local = _train_local_model(
                    model,
                    loader,
                    config,
                    round_index,
                    client_id,
                    malicious,
                )
                uploads.append(
                    _rank_upload(
                        local,
                        config,
                        round_index,
                        client_id,
                        malicious,
                    )
                )
                if malicious:
                    malicious_rows.append(row)
                del local
            diagnostics = _aggregate_rank_round(
                model,
                uploads,
                malicious_rows,
                score_levels,
                config,
                round_index,
                cmgra,
                vem_state,
                force_frl=(
                    hybrid_state is not None
                    and hybrid_state["phase"] == "frl"
                ),
            )
            del uploads

        val_acc = float("nan")
        val_loss = float("nan")
        hybrid_switch = ""
        if hybrid_state is not None:
            if heldout_loader is None or cmgra is None:
                raise RuntimeError("hybrid validation state is missing")
            val_acc, val_loss, _ = evaluate(
                model, heldout_loader, config
            )
            history = hybrid_state["val_history"]
            if not isinstance(history, list):
                raise RuntimeError("invalid hybrid validation history")
            history.append(float(val_acc))

            if hybrid_state["phase"] == "frl":
                window = config.cmgra_hybrid_val_window
                window_ready = len(history) >= window
                signal_values = history[-window:] if window_ready else history
                signal = sum(signal_values) / len(signal_values)
                best_signal = float(hybrid_state["best_signal"])
                if signal > best_signal:
                    hybrid_state["best_signal"] = signal
                    hybrid_state["best_round"] = round_index + 1
                    hybrid_state["bad_streak"] = 0
                    hybrid_state["best_snapshot"] = (
                        _capture_hybrid_snapshot(
                            model, vem_state, round_index + 1
                        )
                    )
                elif (
                    best_signal - signal
                    >= config.cmgra_hybrid_val_drop
                ):
                    hybrid_state["bad_streak"] = (
                        int(hybrid_state["bad_streak"]) + 1
                    )
                else:
                    hybrid_state["bad_streak"] = 0

                if _hybrid_triggered(
                    config.cmgra_hybrid_mode,
                    round_index + 1,
                    config.cmgra_hybrid_warmup_rounds,
                    window_ready,
                    signal,
                    float(hybrid_state["best_signal"]),
                    config.cmgra_hybrid_val_drop,
                    int(hybrid_state["bad_streak"]),
                    config.cmgra_hybrid_patience,
                ):
                    snapshot = hybrid_state.get("best_snapshot")
                    if not isinstance(snapshot, Mapping):
                        raise RuntimeError(
                            "hybrid trigger has no rollback snapshot"
                        )
                    rollback_round = int(snapshot["round"])
                    _restore_hybrid_snapshot(
                        snapshot, model, vem_state
                    )
                    cmgra.adopt_global_rankings(
                        _hybrid_rankings_from_model(model)
                    )
                    hybrid_state["phase"] = "cmgra"
                    hybrid_switch = (
                        f"{config.cmgra_hybrid_mode}:"
                        f"rollback={rollback_round}"
                    )
                    # Log validation for the state actually sent to the next
                    # communication round, not the discarded trigger state.
                    val_acc, val_loss, _ = evaluate(
                        model, heldout_loader, config
                    )

        test_acc, test_loss, asr = evaluate(
            model, evaluation_loader, config
        )
        best_acc = max(best_acc, test_acc)
        round_seconds = time.time() - round_start
        elapsed = elapsed_before + time.time() - wall_start
        metric = {
            "round": round_index + 1,
            "test_acc": f"{test_acc:.8f}",
            "test_loss": f"{test_loss:.8f}",
            "backdoor_asr": (
                f"{asr:.8f}" if not math.isnan(asr) else ""
            ),
            "best_acc": f"{best_acc:.8f}",
            "val_acc": (
                f"{val_acc:.8f}" if not math.isnan(val_acc) else ""
            ),
            "val_loss": (
                f"{val_loss:.8f}" if not math.isnan(val_loss) else ""
            ),
            "hybrid_phase": (
                hybrid_state["phase"] if hybrid_state is not None else ""
            ),
            "hybrid_switch": hybrid_switch,
            "hybrid_best_round": (
                hybrid_state["best_round"]
                if hybrid_state is not None
                else ""
            ),
            "malicious_clients": len(malicious_ids),
            "round_seconds": f"{round_seconds:.3f}",
            "elapsed_seconds": f"{elapsed:.3f}",
            **diagnostics,
        }
        _write_metric(metrics_path, metric)
        print(json.dumps(metric, sort_keys=True), flush=True)
        if (
            (round_index + 1) % config.checkpoint_interval == 0
            or round_index + 1 == config.rounds
        ):
            _save_checkpoint(
                checkpoint_path,
                model,
                round_index + 1,
                best_acc,
                elapsed,
                cmgra,
                vem_state,
                hybrid_state,
            )
        torch.cuda.empty_cache()
    return output


def _auto_local_lr(dataset: str, family: str) -> float:
    if family == "rank":
        return 0.4 if dataset in MNIST_LIKE_DATASETS else 1.0
    return 0.001 if dataset in MNIST_LIKE_DATASETS else 0.01


def _auto_keep_ratio(dataset: str) -> float:
    # The matched Conv2 ablation showed that 20% density is substantially more
    # stable than the original 10% setting across 20--40% source-faithful VEM.
    return 0.2 if dataset in MNIST_LIKE_DATASETS else 0.5


def _auto_momentum(dataset: str, family: str) -> float:
    if dataset in MNIST_LIKE_DATASETS and family == "rank":
        return 0.99
    return 0.9


def _auto_n_clients(dataset: str) -> int:
    # All three paper configurations use the same 1,000-client population.
    return 1000


def _auto_test_batch_size(dataset: str) -> int:
    return 128 if dataset in MNIST_LIKE_DATASETS else 512


def _auto_cmgra_seed(dataset: str) -> int:
    return 2026 if dataset in MNIST_LIKE_DATASETS else 0


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "auto",
            "conv2",
            "lenet",
            "lenet-wide-160",
            "conv4",
            "conv8",
            "resnet18",
            "resnet34",
            "wideresnet28x6",
            "wideresnet28x10",
        ),
        default="auto",
    )
    parser.add_argument("--method", choices=ALL_METHODS, required=True)
    parser.add_argument("--attack", choices=ALL_ATTACKS, required=True)
    parser.add_argument("--malicious-fraction", type=float, required=True)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument(
        "--data-partition",
        choices=("iid", "legacy-vem"),
        default="iid",
    )
    parser.add_argument(
        "--partition-file",
        default="",
        help="cached VEM partition used with --data-partition legacy-vem",
    )
    parser.add_argument("--n-clients", type=int)
    parser.add_argument("--round-clients", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int)
    parser.add_argument("--local-lr", type=float)
    parser.add_argument("--lr-decay", type=float, default=0.999)
    parser.add_argument(
        "--global-cosine-start",
        type=int,
        default=-1,
        help=(
            "communication round at which the global base LR switches from "
            "exponential decay to cosine decay; -1 disables it"
        ),
    )
    parser.add_argument(
        "--global-min-lr",
        type=float,
        default=0.02,
        help="final base LR used by global cosine decay",
    )
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    cosine = parser.add_mutually_exclusive_group()
    cosine.add_argument(
        "--local-cosine", dest="local_cosine", action="store_true"
    )
    cosine.add_argument(
        "--no-local-cosine", dest="local_cosine", action="store_false"
    )
    parser.set_defaults(local_cosine=None)
    parser.add_argument("--keep-ratio", type=float)
    parser.add_argument(
        "--rank-weight-init",
        choices=("signed-constant", "kaiming-uniform"),
        default="signed-constant",
        help="fixed-weight initialization used by rank-based models",
    )
    parser.add_argument(
        "--svhn-batchnorm",
        "--conv-batchnorm",
        dest="svhn_batchnorm",
        action="store_true",
        help="insert non-affine, stateless BatchNorm layers into Conv4/Conv8",
    )
    parser.add_argument("--bcp-server-lr", type=float, default=0.1)
    parser.add_argument("--root-size", type=int, default=200)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--label-flip-target", type=int, default=0)
    parser.add_argument(
        "--shuffle-scale",
        type=float,
        default=10000.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--backdoor-fraction", type=float, default=0.2)
    parser.add_argument("--backdoor-target", type=int, default=7)
    parser.add_argument("--trigger-size", type=int, default=5)
    parser.add_argument("--backdoor-examples", type=int, default=9)
    parser.add_argument("--cmgra-seed", type=int)
    parser.add_argument("--no-cmgra-pairwise", action="store_true")
    parser.add_argument(
        "--cmgra-pair-matching",
        choices=("extreme", "max-cardinality"),
        default="extreme",
        help=(
            "certified pair construction: strongest challenger versus weakest "
            "incumbent, or maximum-cardinality threshold matching"
        ),
    )
    group_order = parser.add_mutually_exclusive_group()
    group_order.add_argument("--cmgra-preserve-order", action="store_true")
    group_order.add_argument("--cmgra-membership-order", action="store_true")
    group_order.add_argument("--cmgra-borda-order", action="store_true")
    group_order.add_argument(
        "--cmgra-borda-final-order",
        action="store_true",
    )
    parser.add_argument("--cmgra-count-ema", type=float, default=0.0)
    parser.add_argument(
        "--cmgra-swap-cap",
        type=int,
        default=0,
        help="maximum certified pair swaps per layer and round; 0 is unlimited",
    )
    parser.add_argument(
        "--cmgra-persistence-rounds", type=int, default=1
    )
    parser.add_argument("--cmgra-update-interval", type=int, default=1)
    parser.add_argument(
        "--cmgra-borda-mode",
        choices=("sum", "lower-bound"),
        default="sum",
        help=(
            "within-group score computed from aggregate s and c only; "
            "certificates remain count-only"
        ),
    )
    parser.add_argument(
        "--cmgra-rank-buckets",
        type=int,
        default=0,
        help=(
            "quantize client ranks into Q fixed-cardinality buckets and use "
            "m*(Q-1)-certified within-group compare-swaps; 0 disables"
        ),
    )
    parser.add_argument(
        "--cmgra-borda-ema",
        type=float,
        default=0.0,
        help=(
            "EMA coefficient for normalized aggregate Borda used only for "
            "within-group ordering"
        ),
    )
    parser.add_argument(
        "--cmgra-prior-rank-weight",
        type=float,
        default=0.0,
        help=(
            "weight of the previous normalized global ranking in temporal "
            "within-group Borda ordering"
        ),
    )
    parser.add_argument(
        "--cmgra-hybrid-mode",
        choices=("none", "fixed", "adaptive"),
        default="none",
        help=(
            "use FRL during a held-out-data warm-up, then switch once to "
            "CMGRA-PX either at a fixed round or after a persistent "
            "validation decline"
        ),
    )
    parser.add_argument(
        "--cmgra-hybrid-warmup-rounds", type=int, default=20
    )
    parser.add_argument(
        "--cmgra-hybrid-val-window", type=int, default=5
    )
    parser.add_argument(
        "--cmgra-hybrid-val-drop", type=float, default=0.015
    )
    parser.add_argument(
        "--cmgra-hybrid-patience", type=int, default=5
    )
    parser.add_argument("--vem-lr", type=float, default=0.1)
    parser.add_argument("--vem-epochs", type=int, default=50)
    parser.add_argument("--vem-max-window", type=int, default=2500)
    parser.add_argument("--vem-temperature", type=float, default=0.0001)
    parser.add_argument("--vem-sinkhorn-iterations", type=int, default=50)
    parser.add_argument("--vem-noise", type=float, default=1.0)
    parser.add_argument(
        "--no-vem-source-compatible",
        action="store_true",
        help="disable the published VEM history and server-visible permutation path",
    )
    parser.add_argument("--rvpfl-paper-extra-division", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--data-root", default="benchmark_data")
    parser.add_argument("--output-root", default="BenchmarkRuns")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    values = parser.parse_args(argv)
    if values.global_cosine_start < -1:
        parser.error("--global-cosine-start must be -1 or non-negative")
    if values.global_cosine_start >= values.rounds:
        parser.error("--global-cosine-start must be smaller than --rounds")
    if values.global_min_lr < 0:
        parser.error("--global-min-lr must be non-negative")
    if not 0 <= values.label_flip_target < 10:
        parser.error("--label-flip-target must be in [0, 9]")
    if values.cmgra_hybrid_warmup_rounds < 1:
        parser.error("--cmgra-hybrid-warmup-rounds must be positive")
    if (
        values.cmgra_hybrid_mode != "none"
        and values.cmgra_hybrid_warmup_rounds >= values.rounds
    ):
        parser.error(
            "--cmgra-hybrid-warmup-rounds must be smaller than --rounds"
        )
    if values.cmgra_hybrid_val_window < 1:
        parser.error("--cmgra-hybrid-val-window must be positive")
    if values.cmgra_hybrid_val_drop < 0:
        parser.error("--cmgra-hybrid-val-drop must be non-negative")
    if values.cmgra_hybrid_patience < 1:
        parser.error("--cmgra-hybrid-patience must be positive")
    if values.data_partition == "legacy-vem" and not values.partition_file:
        parser.error(
            "--partition-file is required with --data-partition legacy-vem"
        )
    family = method_family(values.method)
    local_lr = (
        values.local_lr
        if values.local_lr is not None
        else _auto_local_lr(values.dataset, family)
    )
    keep_ratio = (
        values.keep_ratio
        if values.keep_ratio is not None
        else _auto_keep_ratio(values.dataset)
    )
    momentum = (
        values.momentum
        if values.momentum is not None
        else _auto_momentum(values.dataset, family)
    )
    local_cosine = (
        values.local_cosine
        if values.local_cosine is not None
        else family == "rank"
        and (
            values.dataset in MNIST_LIKE_DATASETS
            or (
                values.attack == "vem"
                and not values.no_vem_source_compatible
            )
        )
    )
    n_clients = (
        values.n_clients
        if values.n_clients is not None
        else _auto_n_clients(values.dataset)
    )
    test_batch_size = (
        values.test_batch_size
        if values.test_batch_size is not None
        else _auto_test_batch_size(values.dataset)
    )
    cmgra_seed = (
        values.cmgra_seed
        if values.cmgra_seed is not None
        else _auto_cmgra_seed(values.dataset)
    )
    return RunConfig(
        dataset=values.dataset,
        model=values.model,
        method=values.method,
        attack=values.attack,
        malicious_fraction=values.malicious_fraction,
        rounds=values.rounds,
        seed=values.seed,
        partition_seed=values.partition_seed,
        data_partition=values.data_partition,
        partition_file=values.partition_file,
        n_clients=n_clients,
        round_clients=values.round_clients,
        local_epochs=values.local_epochs,
        batch_size=values.batch_size,
        test_batch_size=test_batch_size,
        local_lr=local_lr,
        lr_decay=values.lr_decay,
        global_cosine_start=values.global_cosine_start,
        global_min_lr=values.global_min_lr,
        momentum=momentum,
        weight_decay=values.weight_decay,
        local_cosine=local_cosine,
        keep_ratio=keep_ratio,
        rank_weight_init=values.rank_weight_init,
        svhn_batchnorm=values.svhn_batchnorm,
        bcp_server_lr=values.bcp_server_lr,
        root_size=values.root_size,
        noise_scale=values.noise_scale,
        shuffle_scale=values.shuffle_scale,
        label_flip_target=values.label_flip_target,
        backdoor_fraction=values.backdoor_fraction,
        backdoor_target=values.backdoor_target,
        trigger_size=values.trigger_size,
        backdoor_examples=values.backdoor_examples,
        cmgra_seed=cmgra_seed,
        cmgra_pairwise=not values.no_cmgra_pairwise,
        cmgra_pair_matching=values.cmgra_pair_matching,
        cmgra_preserve_order=values.cmgra_preserve_order,
        cmgra_membership_order=values.cmgra_membership_order,
        cmgra_borda_order=values.cmgra_borda_order,
        cmgra_borda_final_order=values.cmgra_borda_final_order,
        cmgra_count_ema=values.cmgra_count_ema,
        cmgra_swap_cap=values.cmgra_swap_cap,
        cmgra_persistence_rounds=values.cmgra_persistence_rounds,
        cmgra_update_interval=values.cmgra_update_interval,
        cmgra_borda_mode=values.cmgra_borda_mode,
        cmgra_rank_buckets=values.cmgra_rank_buckets,
        cmgra_borda_ema=values.cmgra_borda_ema,
        cmgra_prior_rank_weight=values.cmgra_prior_rank_weight,
        cmgra_hybrid_mode=values.cmgra_hybrid_mode,
        cmgra_hybrid_warmup_rounds=(
            values.cmgra_hybrid_warmup_rounds
        ),
        cmgra_hybrid_val_window=values.cmgra_hybrid_val_window,
        cmgra_hybrid_val_drop=values.cmgra_hybrid_val_drop,
        cmgra_hybrid_patience=values.cmgra_hybrid_patience,
        vem_lr=values.vem_lr,
        vem_epochs=values.vem_epochs,
        vem_max_window=values.vem_max_window,
        vem_temperature=values.vem_temperature,
        vem_sinkhorn_iterations=values.vem_sinkhorn_iterations,
        vem_noise=values.vem_noise,
        vem_source_compatible=not values.no_vem_source_compatible,
        rvpfl_paper_extra_division=values.rvpfl_paper_extra_division,
        checkpoint_interval=values.checkpoint_interval,
        data_root=values.data_root,
        output_root=values.output_root,
        num_workers=values.num_workers,
        device=values.device,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    output = run(config)
    from .plotting import plot_run

    plot_run(output)


if __name__ == "__main__":
    main()
