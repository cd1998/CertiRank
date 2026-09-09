"""Generate the complete, protocol-aware experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import (
    ALL_METHODS,
    COMMON_ATTACKS,
    DATASETS,
    RANK_METHODS,
    RANK_ONLY_ATTACKS,
)


FRACTIONS = (0.1, 0.2)


def build_tasks(rounds: int = 500, seed: int = 0):
    tasks = []
    for dataset in DATASETS:
        for method in ALL_METHODS:
            tasks.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "attack": "clean",
                    "malicious_fraction": 0.0,
                    "rounds": rounds,
                    "seed": seed,
                }
            )
        for attack in COMMON_ATTACKS:
            for fraction in FRACTIONS:
                for method in ALL_METHODS:
                    tasks.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "attack": attack,
                            "malicious_fraction": fraction,
                            "rounds": rounds,
                            "seed": seed,
                        }
                    )
        for attack in RANK_ONLY_ATTACKS:
            for fraction in FRACTIONS:
                for method in RANK_METHODS:
                    tasks.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "attack": attack,
                            "malicious_fraction": fraction,
                            "rounds": rounds,
                            "seed": seed,
                        }
                    )
    return tasks


def write_manifest(path: str | Path, rounds: int = 500, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(rounds, seed)
    with path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            handle.write(json.dumps({"task_id": index, **task}, sort_keys=True) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_manifest.jsonl")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = write_manifest(args.output, args.rounds, args.seed)
    print(f"{path} {len(build_tasks(args.rounds, args.seed))}")


if __name__ == "__main__":
    main()
