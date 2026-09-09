"""Claim and execute manifest tasks safely from one GPU worker."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Sequence

from .plotting import plot_run
from .runner import parse_args, run, run_directory


def _arguments(task: dict, common: argparse.Namespace):
    if "arguments" in task:
        # Campaign manifests may carry a complete runner command tail.  This
        # keeps heterogeneous dataset/method hyperparameters explicit and
        # makes the exact launched configuration auditable in the manifest.
        return [str(value) for value in task["arguments"]]
    values = [
        "--dataset",
        task["dataset"],
        "--method",
        task["method"],
        "--attack",
        task["attack"],
        "--malicious-fraction",
        str(task["malicious_fraction"]),
        "--rounds",
        str(task["rounds"]),
        "--seed",
        str(task["seed"]),
        "--partition-seed",
        str(common.partition_seed),
        "--data-root",
        common.data_root,
        "--output-root",
        common.output_root,
        "--num-workers",
        str(common.num_workers),
        "--device",
        common.device,
    ]
    if common.local_epochs is not None:
        values.extend(("--local-epochs", str(common.local_epochs)))
    if common.batch_size is not None:
        values.extend(("--batch-size", str(common.batch_size)))
    return values


def _completed(run_path: Path, expected_rounds: int) -> bool:
    metrics = run_path / "metrics.csv"
    if not metrics.exists():
        return False
    with metrics.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and int(rows[-1]["round"]) >= expected_rounds


def _claim(path: Path, task_id: int) -> int | None:
    path.mkdir(parents=True, exist_ok=True)
    claim = path / ".claim"
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            payload = json.loads(claim.read_text(encoding="utf-8"))
            same_host = payload.get("host") == socket.gethostname()
            pid = int(payload.get("pid", -1))
            alive = False
            if same_host and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
            if alive or not same_host:
                return None
            claim.unlink()
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    payload = json.dumps(
        {
            "task_id": task_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": time.time(),
        }
    ).encode("utf-8")
    os.write(descriptor, payload)
    os.close(descriptor)
    return task_id


def _wait_for_load(max_load: float, poll_seconds: float) -> None:
    if max_load <= 0 or not hasattr(os, "getloadavg"):
        return
    announced = False
    while True:
        one_minute = os.getloadavg()[0]
        if one_minute <= max_load:
            if announced:
                print(
                    json.dumps(
                        {"event": "load_ready", "load_1m": one_minute},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return
        if not announced:
            print(
                json.dumps(
                    {
                        "event": "load_wait",
                        "load_1m": one_minute,
                        "max_load": max_load,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            announced = True
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="benchmark_data")
    parser.add_argument("--output-root", default="BenchmarkRuns")
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--max-load",
        type=float,
        default=0.0,
        help="wait while the one-minute system load exceeds this value; 0 disables",
    )
    parser.add_argument("--load-poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    tasks = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(
        json.dumps(
            {
                "event": "worker_start",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "tasks": len(tasks),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for task in tasks:
        config = parse_args(_arguments(task, args))
        output = run_directory(config)
        if _completed(output, task["rounds"]):
            continue
        failed = output / "failure.json"
        if failed.exists() and not args.retry_failed:
            continue
        _wait_for_load(args.max_load, args.load_poll_seconds)
        if _claim(output, task["task_id"]) is None:
            continue
        claim = output / ".claim"
        try:
            if failed.exists():
                failed.unlink()
            completed_output = run(config)
            plot_run(completed_output)
        except Exception as exc:
            failed.write_text(
                json.dumps(
                    {
                        "task": task,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "time": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "task_failed",
                        "task_id": task["task_id"],
                        "output": str(output),
                        "error": repr(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            claim.unlink(missing_ok=True)
    print(json.dumps({"event": "worker_done", "pid": os.getpid()}), flush=True)


if __name__ == "__main__":
    main()
