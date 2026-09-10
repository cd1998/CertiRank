"""Generate the exact experiment matrix reported in paper Table 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import ALL_METHODS, COMMON_ATTACKS, DATASETS, RANK_METHODS
from .runner import parse_args, run_directory


FRACTIONS = (0.1, 0.2, 0.3)
ROUNDS = 500
OUTPUT_ROOT = "PaperTable8Runs"

PROFILES = {
    "mnist": {
        "model": "conv2",
        "partition": "partitions/mnist_dirichlet_a1_seed0.pkl",
        "local_epochs": 1,
        "test_batch_size": 128,
        "keep_ratio": 0.2,
        "rank_weight_init": "signed-constant",
        "cmgra_seed": 2026,
        "conv_batchnorm": False,
    },
    "svhn": {
        "model": "conv8",
        "partition": "partitions/svhn_dirichlet_a1_seed0.pkl",
        "local_epochs": 5,
        "test_batch_size": 512,
        "keep_ratio": 0.5,
        "rank_weight_init": "kaiming-uniform",
        "cmgra_seed": 0,
        "conv_batchnorm": True,
    },
    "cifar10": {
        "model": "resnet18",
        "partition": "partitions/cifar10_dirichlet_a1_seed0.pkl",
        "local_epochs": 5,
        "test_batch_size": 512,
        "keep_ratio": 0.5,
        "rank_weight_init": "signed-constant",
        "cmgra_seed": 2026,
        "conv_batchnorm": False,
    },
}


def runner_arguments(
    dataset: str,
    method: str,
    attack: str,
    fraction: float,
    rounds: int = ROUNDS,
    seed: int = 0,
    output_root: str = OUTPUT_ROOT,
) -> list[str]:
    """Build the full runner argument list for one paper-table cell."""

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
        "--backdoor-target", "2",
        "--backdoor-examples", "9",
        "--trigger-size", "5",
        "--cmgra-seed", str(profile["cmgra_seed"]),
        "--checkpoint-interval", "10",
        "--data-root", "benchmark_data",
        "--output-root", output_root,
        "--num-workers", "0",
        "--device", "cuda",
    ]
    if rank_based:
        arguments.extend(
            ("--local-lr", "0.4", "--momentum", "0.99", "--local-cosine")
        )
    else:
        gradient_lr = "0.001" if dataset == "mnist" else "0.01"
        arguments.extend(
            ("--local-lr", gradient_lr, "--momentum", "0.9", "--no-local-cosine")
        )
    if profile["conv_batchnorm"]:
        arguments.append("--conv-batchnorm")
    return arguments


def _task(
    dataset: str,
    method: str,
    attack: str,
    fraction: float,
    rounds: int,
    seed: int,
) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "attack": attack,
        "malicious_fraction": fraction,
        "rounds": rounds,
        "seed": seed,
        "arguments": runner_arguments(
            dataset, method, attack, fraction, rounds, seed
        ),
    }


def build_tasks(rounds: int = ROUNDS, seed: int = 0) -> list[dict]:
    """Return all 168 Table-8 runs with explicit, paper-matched arguments."""

    tasks: list[dict] = []
    for dataset in DATASETS:
        for method in ALL_METHODS:
            tasks.append(_task(dataset, method, "clean", 0.0, rounds, seed))
        for attack in COMMON_ATTACKS:
            for fraction in FRACTIONS:
                for method in ALL_METHODS:
                    tasks.append(
                        _task(dataset, method, attack, fraction, rounds, seed)
                    )
        for fraction in FRACTIONS:
            for method in RANK_METHODS:
                tasks.append(
                    _task(dataset, method, "vem", fraction, rounds, seed)
                )
    return [{"task_id": index, **task} for index, task in enumerate(tasks)]


def validate_tasks(tasks: Sequence[dict]) -> None:
    expected = len(DATASETS) * (
        len(ALL_METHODS)
        + len(COMMON_ATTACKS) * len(FRACTIONS) * len(ALL_METHODS)
        + len(FRACTIONS) * len(RANK_METHODS)
    )
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} tasks, got {len(tasks)}")

    outputs: set[Path] = set()
    for task in tasks:
        config = parse_args(task["arguments"])
        output = run_directory(config)
        if output in outputs:
            raise ValueError(f"duplicate output directory: {output}")
        outputs.add(output)
        if config.rounds != task["rounds"]:
            raise ValueError(f"round mismatch in task {task['task_id']}")


def write_manifest(
    path: str | Path, rounds: int = ROUNDS, seed: int = 0
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(rounds, seed)
    validate_tasks(tasks)
    destination.write_text(
        "\n".join(json.dumps(task, sort_keys=True) for task in tasks) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper_table8_manifest.jsonl")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = write_manifest(args.output, args.rounds, args.seed)
    print(f"{destination} {len(build_tasks(args.rounds, args.seed))}")


if __name__ == "__main__":
    main()
