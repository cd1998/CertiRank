"""Build the corrected finite gradient-ascent SVHN rerun campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runner import parse_args, run_directory
from .supplement_campaign import _runner_arguments


METHODS = ("fedavg", "bcpbfl", "rvpfl")
FRACTIONS = (0.1, 0.2, 0.3)
ROUNDS = 500
OUTPUT_ROOT = "FinalRuns/SVHNFiniteGradientAscent500"


def build_tasks(rounds: int = ROUNDS, seed: int = 0) -> list[dict]:
    tasks = []
    for fraction in FRACTIONS:
        for method in METHODS:
            tasks.append(
                {
                    "campaign": "svhn_finite_gradient_ascent",
                    "dataset": "svhn",
                    "method": method,
                    "attack": "grad_ascent",
                    "malicious_fraction": fraction,
                    "rounds": rounds,
                    "seed": seed,
                    "arguments": _runner_arguments(
                        "svhn",
                        method,
                        "grad_ascent",
                        fraction,
                        OUTPUT_ROOT,
                        rounds,
                        seed,
                    ),
                }
            )
    return [{"task_id": index, **task} for index, task in enumerate(tasks)]


def validate_tasks(tasks: Sequence[dict]) -> None:
    if len(tasks) != len(METHODS) * len(FRACTIONS):
        raise ValueError(f"expected 9 tasks, got {len(tasks)}")
    paths = set()
    for task in tasks:
        config = parse_args(task["arguments"])
        path = run_directory(config)
        if path in paths:
            raise ValueError(f"duplicate output directory: {path}")
        paths.add(path)
        if config.dataset != "svhn" or config.attack != "grad_ascent":
            raise ValueError(f"invalid task {task['task_id']}")


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
        "--output",
        default="FinalRuns/svhn_finite_gradient_ascent_500.jsonl",
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(write_manifest(args.output, args.rounds, args.seed))


if __name__ == "__main__":
    main()
