"""Focused FRL/CMGRA benchmark for MNIST, SVHN, and CIFAR10."""

MNIST_LIKE_DATASETS = ("mnist",)
DATASETS = ("mnist", "svhn", "cifar10")

GRADIENT_METHODS = ("fedavg", "bcpbfl", "rvpfl")
RANK_METHODS = ("frl", "cmgra")
ALL_METHODS = GRADIENT_METHODS + RANK_METHODS

COMMON_ATTACKS = (
    "label_flip",
    "grad_ascent",
    "pixel_backdoor",
)
RANK_ONLY_ATTACKS = ("vem",)
ALL_ATTACKS = (
    ("clean",)
    + COMMON_ATTACKS
    + RANK_ONLY_ATTACKS
)


def method_family(method: str) -> str:
    if method in GRADIENT_METHODS:
        return "gradient"
    if method in RANK_METHODS:
        return "rank"
    raise ValueError(f"unknown method: {method}")


def attack_applies(method: str, attack: str) -> bool:
    method_family(method)
    if (
        attack == "clean"
        or attack in COMMON_ATTACKS
    ):
        return True
    if attack in RANK_ONLY_ATTACKS:
        return method in RANK_METHODS
    raise ValueError(f"unknown attack: {attack}")
