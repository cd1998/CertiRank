"""Create deterministic VEM-style Dirichlet client-partition caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np
from torchvision import datasets

from . import DATASETS


def vem_dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, int]]]:
    """Follow VEM's per-class rounded Dirichlet allocation reproducibly."""

    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)
    partitions: Dict[int, List[int]] = defaultdict(list)
    histograms: Dict[int, Dict[int, int]] = defaultdict(dict)

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(indices)
        probabilities = rng.dirichlet(
            np.full(n_clients, alpha, dtype=np.float64)
        )
        requested = np.rint(probabilities * indices.size).astype(np.int64)
        cursor = 0
        for client_id, count in enumerate(requested.tolist()):
            stop = min(indices.size, cursor + count)
            assigned = indices[cursor:stop].tolist()
            partitions[client_id].extend(assigned)
            histograms[client_id][int(label)] = len(assigned)
            cursor = stop

    for client_id in range(n_clients):
        if not partitions[client_id]:
            raise ValueError(
                "Dirichlet draw produced an empty client; choose another seed"
            )
    return partitions, histograms


def dataset_labels(dataset: str, root: str, download: bool) -> np.ndarray:
    if dataset == "svhn":
        train = datasets.SVHN(
            root, split="train", download=download
        )
        return np.asarray(train.labels, dtype=np.int64)
    if dataset == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=download)
        return np.asarray(train.targets, dtype=np.int64)
    if dataset == "mnist":
        train = datasets.MNIST(root, train=True, download=download)
        values = train.targets
        return (
            values.cpu().numpy()
            if hasattr(values, "cpu")
            else np.asarray(values, dtype=np.int64)
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def write_partition(
    output: str,
    partitions: Dict[int, List[int]],
    histograms: Dict[int, Dict[int, int]],
    force: bool,
) -> None:
    path = Path(output)
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([partitions, histograms], handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=DATASETS, required=True
    )
    parser.add_argument("--root", default="benchmark_data")
    parser.add_argument("--n-clients", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    labels = dataset_labels(args.dataset, args.root, args.download)
    partitions, histograms = vem_dirichlet_partition(
        labels, args.n_clients, args.alpha, args.seed
    )
    write_partition(args.output, partitions, histograms, args.force)
    sizes = np.asarray(
        [len(partitions[index]) for index in range(args.n_clients)]
    )
    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(
        {
            "dataset": args.dataset,
            "clients": args.n_clients,
            "assigned": int(sizes.sum()),
            "unassigned": int(labels.size - sizes.sum()),
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
            "sha256": digest,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
