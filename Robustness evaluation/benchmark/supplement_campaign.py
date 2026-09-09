"""Build the requested CIFAR10 VEM and common-attack supplement campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import ALL_METHODS, COMMON_ATTACKS, DATASETS, RANK_METHODS
from .runner import parse_args, run_directory


FRACTIONS = (0.1, 0.2, 0.3, 0.4)
VEM_FRACTIONS = (0.1, 0.3, 0.4)
ROUNDS = 500

PROFILES = {
    "mnist": {
        "model": "conv2",
        "partition": "VEM-master/MNIST_train_dirichlet_a_1.0_n1000.pkl",
        "local_epochs": 1,
        "test_batch_size": 128,
        "keep_ratio": 0.2,
        "rank_weight_init": "signed-constant",
        "cmgra_seed": 2026,
        "svhn_batchnorm": False,
    },
    "svhn": {
        "model": "conv8",
        "partition": "FinalPartitions/SVHN_train_dirichlet_a_1.0_n1000_seed0.pkl",
        "local_epochs": 5,
        "test_batch_size": 512,
        "keep_ratio": 0.5,
        "rank_weight_init": "kaiming-uniform",
        "cmgra_seed": 0,
        "svhn_batchnorm": True,
    },
    "cifar10": {
        "model": "resnet18",
        "partition": "VEM-master/CIFAR10_train_dirichlet_a_1.0_n1000.pkl",
        "local_epochs": 5,
        "test_batch_size": 512,
        "keep_ratio": 0.5,
        "rank_weight_init": "signed-constant",
        "cmgra_seed": 2026,
        "svhn_batchnorm": False,
    },
}


def _runner_arguments(
    dataset: str,
    method: str,
    attack: str,
    fraction: float,
    output_root: str,
    rounds: int = ROUNDS,
    seed: int = 0,
) -> list[str]:
    profile = PROFILES[dataset]
    rank_based = method in RANK_METHODS
    arguments = [
        "--dataset", dataset,
        "--model", str(profile["model"]),
        "--method", method,
        "--attack", attack,
        "--malicious-fraction", str(fraction),
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--partition-seed", "0",
        "--data-partition", "legacy-vem",
        "--partition-file", str(profile["partition"]),
        "--n-clients", "1000",
        "--round-clients", "25",
        "--local-epochs", str(profile["local_epochs"]),
        "--batch-size", "8",
        "--test-batch-size", str(profile["test_batch_size"]),
        "--lr-decay", "0.999",
        "--weight-decay", "0.0001",
        "--keep-ratio", str(profile["keep_ratio"]),
        "--rank-weight-init", str(profile["rank_weight_init"]),
        "--root-size", "200" if method == "bcpbfl" else "0",
        "--cmgra-seed", str(profile["cmgra_seed"]),
        "--checkpoint-interval", "10",
        "--data-root", "benchmark_data",
        "--output-root", output_root,
        "--num-workers", "0",
        "--device", "cuda",
    ]
    if rank_based:
        arguments.extend(("--local-lr", "0.4", "--momentum", "0.99", "--local-cosine"))
    else:
        gradient_lr = "0.001" if dataset == "mnist" else "0.01"
        arguments.extend(("--local-lr", gradient_lr, "--momentum", "0.9", "--no-local-cosine"))
    if profile["svhn_batchnorm"]:
        arguments.append("--svhn-batchnorm")
    if method == "cmgra-px":
        arguments.append("--cmgra-borda-final-order")
    return arguments


def build_tasks(rounds: int = ROUNDS, seed: int = 0) -> list[dict]:
    tasks: list[dict] = []

    # Put the explicitly requested missing CIFAR10 VEM cells first so the
    # queue starts them before the much larger common-attack matrix.
    for fraction in VEM_FRACTIONS:
        for method in RANK_METHODS:
            output_root = (
                "FinalRuns/LargeModel500/FRL"
                if method == "frl"
                else "FinalRuns/LargeModel500/CMGRAPX"
            )
            tasks.append(
                {
                    "campaign": "cifar10_vem_supplement",
                    "dataset": "cifar10",
                    "method": method,
                    "attack": "vem",
                    "malicious_fraction": fraction,
                    "rounds": rounds,
                    "seed": seed,
                    "arguments": _runner_arguments(
                        "cifar10", method, "vem", fraction, output_root, rounds, seed
                    ),
                }
            )

    for dataset in DATASETS:
        for attack in COMMON_ATTACKS:
            for fraction in FRACTIONS:
                for method in ALL_METHODS:
                    tasks.append(
                        {
                            "campaign": "common_attacks",
                            "dataset": dataset,
                            "method": method,
                            "attack": attack,
                            "malicious_fraction": fraction,
                            "rounds": rounds,
                            "seed": seed,
                            "arguments": _runner_arguments(
                                dataset,
                                method,
                                attack,
                                fraction,
                                "FinalRuns/CommonAttacks500",
                                rounds,
                                seed,
                            ),
                        }
                    )
    return [{"task_id": index, **task} for index, task in enumerate(tasks)]


def validate_tasks(tasks: Sequence[dict]) -> None:
    expected = 6 + 4 * 3 * 5 * 4
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} tasks, got {len(tasks)}")
    paths: set[Path] = set()
    for task in tasks:
        config = parse_args(task["arguments"])
        path = run_directory(config)
        if path in paths:
            raise ValueError(f"duplicate output directory: {path}")
        paths.add(path)
        if config.rounds != task["rounds"]:
            raise ValueError(f"round mismatch in task {task['task_id']}")


def write_manifest(path: str | Path, rounds: int = ROUNDS, seed: int = 0) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(rounds, seed)
    validate_tasks(tasks)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True) + "\n")
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="FinalRuns/supplement_campaign_500.jsonl"
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = write_manifest(args.output, args.rounds, args.seed)
    print(f"{destination} {len(build_tasks(args.rounds, args.seed))}")


if __name__ == "__main__":
    main()
