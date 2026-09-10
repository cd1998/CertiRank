"""Dataset loading, reproducible client partitioning, and trigger utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DATASET_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

MNIST_DATASETS = {"mnist": ("MNIST", {})}


@dataclass
class FederatedData:
    train: Dataset
    test: Dataset
    client_indices: List[np.ndarray]
    root_indices: np.ndarray
    train_eval: Dataset | None = None
    num_classes: int = 10


class FixedBackdoorDataset(Dataset):
    """Tensor images with Python-int labels compatible with torchvision.

    Torchvision datasets return labels as Python integers. Returning scalar
    tensors here would make the default DataLoader collator fail whenever a
    shuffled batch contains both benign and fixed backdoor samples.
    """

    def __init__(self, images: torch.Tensor, target: int):
        self.images = images
        self.target = int(target)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.target


def _targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return _targets(dataset.dataset)[
            np.asarray(dataset.indices, dtype=np.int64)
        ]
    values = getattr(dataset, "targets", None)
    if values is None:
        values = getattr(dataset, "labels", None)
    if values is None:
        samples = getattr(dataset, "_samples", None)
        if samples is not None:
            values = [label for _, label in samples]
    if values is None:
        raise ValueError("dataset does not expose class labels")
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def _balanced_root_indices(
    targets: np.ndarray, size: int, seed: int
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    classes = np.unique(targets)
    if size % len(classes):
        raise ValueError("root_size must be divisible by the number of classes")
    per_class = size // len(classes)
    rng = np.random.default_rng(seed + 1907)
    selected = []
    for label in classes:
        candidates = np.flatnonzero(targets == label)
        selected.append(rng.choice(candidates, per_class, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def _iid_partition(size: int, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed + 8101)
    shuffled = rng.permutation(size)
    return [part.astype(np.int64) for part in np.array_split(shuffled, n_clients)]


def _legacy_vem_partition(
    size: int, n_clients: int, partition_file: str
) -> List[np.ndarray]:
    """Load the exact cached partition distributed with the VEM source."""

    path = Path(partition_file)
    if not path.is_file():
        raise FileNotFoundError(f"VEM partition file does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    # VEM's Dirichlet cache stores [client_indices, label_histograms], whereas
    # its IID cache stores only client_indices.
    raw = payload[0] if isinstance(payload, (list, tuple)) else payload
    if not isinstance(raw, dict) or len(raw) != n_clients:
        raise ValueError(
            "VEM partition must contain exactly "
            f"{n_clients} client-index entries"
        )

    partitions = []
    for client_id in range(n_clients):
        if client_id not in raw:
            raise ValueError(f"VEM partition is missing client {client_id}")
        indices = np.asarray(raw[client_id], dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                f"VEM partition for client {client_id} must be non-empty and 1-D"
            )
        if int(indices.min()) < 0 or int(indices.max()) >= size:
            raise ValueError(
                f"VEM partition for client {client_id} contains invalid indices"
            )
        partitions.append(indices)

    flat = np.concatenate(partitions)
    if np.unique(flat).size != flat.size:
        raise ValueError("VEM partition assigns a training example more than once")
    return partitions


def load_federated_data(
    dataset: str,
    root: str,
    n_clients: int,
    partition_seed: int,
    root_size: int = 200,
    download: bool = True,
    partition: str = "iid",
    partition_file: str | None = None,
) -> FederatedData:
    dataset = dataset.lower()
    mean, std = DATASET_STATS[dataset]
    normalize = transforms.Normalize(mean, std)
    if dataset in MNIST_DATASETS:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = train_transform
        dataset_class_name, dataset_kwargs = MNIST_DATASETS[dataset]
        dataset_class = getattr(datasets, dataset_class_name)
        train = dataset_class(
            root,
            train=True,
            download=download,
            transform=train_transform,
            **dataset_kwargs,
        )
        test = dataset_class(
            root,
            train=False,
            download=download,
            transform=test_transform,
            **dataset_kwargs,
        )
        train_eval = dataset_class(
            root,
            train=True,
            download=False,
            transform=test_transform,
            **dataset_kwargs,
        )
    elif dataset == "svhn":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train = datasets.SVHN(root, split="train", download=download, transform=train_transform)
        test = datasets.SVHN(root, split="test", download=download, transform=test_transform)
        train_eval = datasets.SVHN(
            root, split="train", download=False, transform=test_transform
        )
    elif dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train = datasets.CIFAR10(
            root, train=True, download=download, transform=train_transform
        )
        test = datasets.CIFAR10(
            root, train=False, download=download, transform=test_transform
        )
        train_eval = datasets.CIFAR10(
            root, train=True, download=False, transform=test_transform
        )
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    targets = _targets(train)
    if partition == "iid":
        client_indices = _iid_partition(
            len(train), n_clients, partition_seed
        )
    elif partition == "legacy-vem":
        if not partition_file:
            raise ValueError(
                "partition_file is required for the legacy-vem partition"
            )
        client_indices = _legacy_vem_partition(
            len(train), n_clients, partition_file
        )
    else:
        raise ValueError(f"unknown client partition: {partition}")

    root_indices = _balanced_root_indices(targets, root_size, partition_seed)

    return FederatedData(
        train=train,
        test=test,
        client_indices=client_indices,
        root_indices=root_indices,
        train_eval=train_eval,
        num_classes=10,
    )


def client_loader(
    data: FederatedData,
    client_id: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    extra_dataset: Dataset | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset: Dataset = Subset(
        data.train, data.client_indices[client_id].tolist()
    )
    if extra_dataset is not None:
        dataset = ConcatDataset((dataset, extra_dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def root_loader(
    data: FederatedData,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        Subset(data.train, data.root_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def test_loader(
    data: FederatedData, batch_size: int, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        data.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def stamp_trigger(
    images: torch.Tensor,
    dataset: str,
    trigger_size: int = 3,
) -> torch.Tensor:
    """Stamp the paper's top-left white F trigger on normalized tensors."""

    images = images.clone()
    mean, std = DATASET_STATS[dataset.lower()]
    white = torch.tensor(
        [(1.0 - m) / s for m, s in zip(mean, std)],
        device=images.device,
        dtype=images.dtype,
    ).reshape(1, -1, 1, 1)
    height = min(trigger_size, images.shape[-2])
    width = min(max(2, trigger_size // 2 + 1), images.shape[-1])
    images[:, :, :height, :1] = white
    images[:, :, :1, :width] = white
    middle = min(height // 2, height - 1)
    images[:, :, middle : middle + 1, :width] = white
    return images


def fixed_paper_backdoor_dataset(
    data: FederatedData,
    dataset: str,
    target: int = 2,
    examples: int = 9,
    trigger_size: int = 5,
) -> FixedBackdoorDataset:
    """Return the fixed poison bank used by FRL's artificial backdoor.

    The first ``examples`` non-target training images are selected once and
    shared by every malicious client, matching the paper's statement that
    every malicious client has all nine backdoored examples.
    """

    if examples <= 0:
        raise ValueError("backdoor examples must be positive")
    source = data.train_eval if data.train_eval is not None else data.train
    targets = _targets(source)
    candidates = np.flatnonzero(targets != target)
    if candidates.size < examples:
        raise ValueError("not enough non-target examples for poison bank")
    images = torch.stack([source[int(index)][0] for index in candidates[:examples]])
    poisoned = stamp_trigger(images, dataset, trigger_size)
    return FixedBackdoorDataset(poisoned, target)
