"""Per-run and comparison accuracy-curve plotting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rounds = [int(row["round"]) for row in rows]
    accuracy = [float(row["test_acc"]) for row in rows]
    asr = [
        float(row["backdoor_asr"])
        for row in rows
        if row.get("backdoor_asr")
    ]
    return rounds, accuracy, asr


def plot_run(run_directory: str | Path) -> Path:
    run_directory = Path(run_directory)
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    rounds, accuracy, asr = _read_metrics(run_directory / "metrics.csv")
    if not rounds:
        raise ValueError(f"no metrics in {run_directory}")
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(rounds, accuracy, color="#2563eb", linewidth=1.6, label="Clean accuracy")
    axis.set_xlabel("Global round")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.22)
    title = (
        f"{config['dataset'].upper()} | {config['method']} | "
        f"{config['attack']} | malicious={config['malicious_fraction']:.0%}"
    )
    axis.set_title(title)
    if asr and len(asr) == len(rounds):
        axis.plot(
            rounds,
            asr,
            color="#dc2626",
            linewidth=1.3,
            alpha=0.9,
            label="Backdoor ASR",
        )
        axis.legend(frameon=False)
    output = run_directory / "accuracy_curve.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_all(root: str | Path) -> int:
    root = Path(root)
    count = 0
    for metrics in root.rglob("metrics.csv"):
        plot_run(metrics.parent)
        count += 1
    return count


def plot_comparisons(root: str | Path) -> int:
    """Overlay methods for each dataset/attack/fraction/seed condition."""

    root = Path(root)
    groups = defaultdict(list)
    for metrics in root.rglob("metrics.csv"):
        config_path = metrics.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rounds, accuracy, _ = _read_metrics(metrics)
        if not rounds:
            continue
        key = (
            config["dataset"],
            config["attack"],
            float(config["malicious_fraction"]),
            int(config["seed"]),
        )
        groups[key].append((config["method"], rounds, accuracy))

    output_root = root / "comparison_curves"
    count = 0
    for (dataset, attack, fraction, seed), curves in groups.items():
        figure, axis = plt.subplots(
            figsize=(7.4, 4.7), constrained_layout=True
        )
        for method, rounds, accuracy in sorted(curves):
            axis.plot(rounds, accuracy, linewidth=1.45, label=method)
        axis.set_xlabel("Global round")
        axis.set_ylabel("Test accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.22)
        axis.set_title(
            f"{dataset.upper()} | {attack} | malicious={fraction:.0%}"
        )
        axis.legend(frameon=False, ncol=2)
        destination = (
            output_root
            / dataset
            / attack
            / f"malicious_{int(round(fraction * 100)):02d}_seed_{seed}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.tmp.png"
        )
        figure.savefig(temporary, dpi=180)
        plt.close(figure)
        os.replace(temporary, destination)
        count += 1
    return count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--comparisons", action="store_true")
    args = parser.parse_args(argv)
    if args.comparisons:
        print(plot_comparisons(args.path))
    elif args.all:
        print(plot_all(args.path))
    else:
        print(plot_run(args.path))


if __name__ == "__main__":
    main()
