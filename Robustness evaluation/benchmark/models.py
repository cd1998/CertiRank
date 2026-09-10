"""The paper's matched Conv2, Conv8, and CIFAR-ResNet18 models.

The gradient methods optimize ordinary weights.  The rank methods optimize
Edge-Popup scores over fixed weights.  The paper uses signed-constant weights
for MNIST and CIFAR10 and Kaiming-uniform fixed weights for SVHN.  Each score
tensor has exactly the same shape as the corresponding trainable weight
tensor, keeping communicated parameter counts aligned across paradigms.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        out = torch.zeros_like(scores)
        keep = int(math.ceil(float(keep_ratio) * scores.numel()))
        if keep:
            selected = torch.topk(
                scores.detach().abs().flatten(), keep, sorted=False
            ).indices
            out.flatten()[selected] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, keep_ratio: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ratio = float(keep_ratio)
        self.scores = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        gain = nn.init.calculate_gain("relu")
        signed_level = gain / math.sqrt(fan)
        with torch.no_grad():
            self.weight.copy_(self.weight.sign() * signed_level)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subnet = _GetSubnet.apply(self.scores, self.keep_ratio)
        return F.conv2d(
            x,
            self.weight * subnet,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _conv(
    rank_based: bool,
    keep_ratio: float,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Conv2d:
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    if rank_based:
        return MaskedConv2d(**kwargs, keep_ratio=keep_ratio)
    layer = nn.Conv2d(**kwargs)
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    return layer


def _norm(rank_based: bool, channels: int) -> nn.Module:
    # Disabling running statistics avoids aggregating mutable client buffers.
    # Affine parameters are disabled for both paradigms so the communicated
    # ResNet18 size is the paper's matched 42.59 MiB in every method.
    return nn.BatchNorm2d(
        channels, affine=False, track_running_stats=False
    )


class Conv2(nn.Module):
    """The two-convolution MNIST model used by the published VEM code."""

    def __init__(self, rank_based: bool, keep_ratio: float):
        super().__init__()
        self.convs = nn.Sequential(
            _conv(rank_based, keep_ratio, 1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            _conv(rank_based, keep_ratio, 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            _conv(rank_based, keep_ratio, 128 * 7 * 7, 256, 1),
            nn.ReLU(inplace=True),
            _conv(rank_based, keep_ratio, 256, 10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = x.reshape(x.shape[0], 128 * 7 * 7, 1, 1)
        return self.linear(x).flatten(1)


class Conv8(nn.Module):
    def __init__(
        self,
        rank_based: bool,
        keep_ratio: float,
        batchnorm: bool = False,
    ):
        super().__init__()
        channels = (3, 64, 64, 128, 128, 256, 256, 512, 512)
        layers = []
        for block in range(4):
            i = block * 2
            layers.append(
                _conv(
                    rank_based,
                    keep_ratio,
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=1,
                )
            )
            if batchnorm:
                layers.append(_norm(rank_based, channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                _conv(
                    rank_based,
                    keep_ratio,
                    channels[i + 1],
                    channels[i + 2],
                    3,
                    padding=1,
                )
            )
            if batchnorm:
                layers.append(_norm(rank_based, channels[i + 2]))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.convs = nn.Sequential(*layers)
        # Do not normalize the 1x1 classifier head: some federated clients
        # yield a final batch of one example, for which BatchNorm at 1x1 has
        # no variance estimate.  Spatial Conv8 blocks remain normalized.
        self.linear = nn.Sequential(
            _conv(rank_based, keep_ratio, 512 * 2 * 2, 256, 1),
            nn.ReLU(inplace=True),
            _conv(rank_based, keep_ratio, 256, 256, 1),
            nn.ReLU(inplace=True),
            _conv(rank_based, keep_ratio, 256, 10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = x.reshape(x.shape[0], 512 * 2 * 2, 1, 1)
        return self.linear(x).flatten(1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        rank_based: bool,
        keep_ratio: float,
        in_planes: int,
        planes: int,
        stride: int,
    ):
        super().__init__()
        self.conv1 = _conv(
            rank_based, keep_ratio, in_planes, planes, 3, stride, 1
        )
        self.bn1 = _norm(rank_based, planes)
        self.conv2 = _conv(rank_based, keep_ratio, planes, planes, 3, 1, 1)
        self.bn2 = _norm(rank_based, planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                _conv(
                    rank_based, keep_ratio, in_planes, planes, 1, stride, 0
                ),
                _norm(rank_based, planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet(nn.Module):
    """Small-image ResNet with a 3x3 stem and no ImageNet max-pool."""

    def __init__(
        self,
        rank_based: bool,
        keep_ratio: float,
        in_channels: int,
        num_classes: int,
        blocks: tuple[int, int, int, int],
    ):
        super().__init__()
        self.rank_based = rank_based
        self.keep_ratio = keep_ratio
        self.in_planes = 64
        self.conv1 = _conv(
            rank_based, keep_ratio, in_channels, 64, 3, 1, 1
        )
        self.bn1 = _norm(rank_based, 64)
        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)
        self.linear = _conv(
            rank_based, keep_ratio, 512, num_classes, 1
        )

    def _make_layer(
        self, planes: int, blocks: int, first_stride: int
    ) -> nn.Sequential:
        strides = [first_stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                BasicBlock(
                    self.rank_based,
                    self.keep_ratio,
                    self.in_planes,
                    planes,
                    stride,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return self.linear(out).flatten(1)


MODEL_FOR_DATASET = {
    "mnist": "conv2",
    "svhn": "conv8",
    "cifar10": "resnet18",
}

DATASET_INPUT_CHANNELS = {
    "mnist": 1,
    "svhn": 3,
    "cifar10": 3,
}

DATASET_NUM_CLASSES = {
    "mnist": 10,
    "svhn": 10,
    "cifar10": 10,
}


def build_model(
    dataset: str,
    rank_based: bool,
    keep_ratio: float = 0.5,
    rank_weight_init: str = "signed-constant",
    conv_batchnorm: bool = False,
    model_name: str = "auto",
) -> nn.Module:
    dataset = dataset.lower()
    model_name = model_name.lower()
    if model_name == "auto":
        model_name = MODEL_FOR_DATASET[dataset]

    if model_name == "conv2":
        model = Conv2(rank_based, keep_ratio)
    elif model_name == "conv8":
        model = Conv8(rank_based, keep_ratio, batchnorm=conv_batchnorm)
    elif model_name == "resnet18":
        model = CIFARResNet(
            rank_based,
            keep_ratio,
            DATASET_INPUT_CHANNELS[dataset],
            DATASET_NUM_CLASSES[dataset],
            (2, 2, 2, 2),
        )
    else:
        raise ValueError(f"unknown model: {model_name}")

    if rank_weight_init not in ("signed-constant", "kaiming-uniform"):
        raise ValueError(
            f"unknown rank weight initialization: {rank_weight_init}"
        )
    if rank_based and rank_weight_init == "kaiming-uniform":
        # The SVHN/Conv8 profile uses this fixed-weight initialization.
        with torch.no_grad():
            for module in scored_modules(model).values():
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    return model


def scored_modules(model: nn.Module) -> Dict[str, MaskedConv2d]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, MaskedConv2d)
    }


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def communicated_parameter_count(model: nn.Module, rank_based: bool) -> int:
    if rank_based:
        return sum(module.scores.numel() for module in scored_modules(model).values())
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
