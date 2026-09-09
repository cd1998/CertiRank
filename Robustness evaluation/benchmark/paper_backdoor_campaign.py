"""Build the low-data artificial pixel-backdoor campaign from FRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import ALL_METHODS, DATASETS
from .runner import parse_args, run_directory
from .supplement_campaign import _runner_arguments


ATTACK = "pixel_backdoor_low_data"
FRACTIONS = (0.1, 0.2, 0.3, 0.4)
ROUNDS = 500
TARGET_LABEL = 2
POISON_EXAMPLES = 9
TRIGGER_SIZE = 5
OUTPUT_ROOT = "FinalRuns/PaperPixelBackdoorLowData500"


def build_tasks(rounds: int = ROUNDS, seed: int = 0) -> list[dict]:
    tasks: list[dict] = []
    for dataset in DATASETS:
        for fraction in FRACTIONS:
            for method in ALL_METHODS:
                arguments = _runner_arguments(
                    dataset,
                    method,
                    ATTACK,
                    fraction,
                    OUTPUT_ROOT,
                    rounds,
                    seed,
                )
                arguments.extend(
                    (
                        "--backdoor-target",
                        str(TARGET_LABEL),
                        "--backdoor-examples",
                        str(POISON_EXAMPLES),
                        "--trigger-size",
                        str(TRIGGER_SIZE),
                    )
                )
                tasks.append(
                    {
                        "arguments": arguments,
                        "attack": ATTACK,
                        "campaign": "paper_pixel_backdoor_low_data",
                        "dataset": dataset,
                        "malicious_fraction": fraction,
                        "method": method,
                        "rounds": rounds,
                        "seed": seed,
                        "task_id": len(tasks),
                    }
                )
    return tasks


def validate_tasks(tasks: Sequence[dict]) -> None:
    expected = len(DATASETS) * len(FRACTIONS) * len(ALL_METHODS)
    if len(tasks) != expected:
        raise ValueError(f"expected {expected} tasks, found {len(tasks)}")
    outputs: set[str] = set()
    for task in tasks:
        config = parse_args(task["arguments"])
        if (
            config.attack != ATTACK
            or config.backdoor_target != TARGET_LABEL
            or config.backdoor_examples != POISON_EXAMPLES
        ):
            raise ValueError(f"invalid paper-backdoor task {task['task_id']}")
        output = str(run_directory(config))
        if output in outputs:
            raise ValueError(f"duplicate output directory: {output}")
        outputs.add(output)


def write_manifest(
    path: str | Path,
    rounds: int = ROUNDS,
    seed: int = 0,
) -> Path:
    tasks = build_tasks(rounds, seed)
    validate_tasks(tasks)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(task, sort_keys=True) for task in tasks) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="FinalRuns/paper_pixel_backdoor_low_data_500.jsonl",
    )
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = write_manifest(args.output, args.rounds, args.seed)
    print(destination)


if __name__ == "__main__":
    main()
