"""Collect final/best accuracy and backdoor ASR from all completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


FIELDS = (
    "dataset",
    "method",
    "attack",
    "malicious_fraction",
    "seed",
    "completed_rounds",
    "final_accuracy",
    "best_accuracy",
    "final_backdoor_asr",
    "run_directory",
)


def collect(root: str | Path, output: str | Path) -> Path:
    root = Path(root)
    output = Path(output)
    records = []
    for metrics_path in root.rglob("metrics.csv"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        final = rows[-1]
        records.append(
            {
                "dataset": config["dataset"],
                "method": config["method"],
                "attack": config["attack"],
                "malicious_fraction": config["malicious_fraction"],
                "seed": config["seed"],
                "completed_rounds": final["round"],
                "final_accuracy": final["test_acc"],
                "best_accuracy": final["best_acc"],
                "final_backdoor_asr": final.get("backdoor_asr", ""),
                "run_directory": str(metrics_path.parent),
            }
        )
    records.sort(
        key=lambda row: (
            row["dataset"],
            row["attack"],
            float(row["malicious_fraction"]),
            row["method"],
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="BenchmarkRuns")
    parser.add_argument("--output", default="BenchmarkRuns/summary.csv")
    args = parser.parse_args(argv)
    print(collect(args.root, args.output))


if __name__ == "__main__":
    main()
