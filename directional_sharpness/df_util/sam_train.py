# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime as _datetime
import os
import random
import sys
from pathlib import Path as _Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import Dataset

_THIS_DIR = str(_Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

__all__ = [
    "get_datetime",
    "set_seed",
    "set_device",
    "mps_available",
    "SAM",
    "smooth_crossentropy",
    "enable_running_stats",
    "disable_running_stats",
    "prepare_dataset",
    "prepare_model",
]


# ----------------------------- misc helpers -----------------------------


def get_datetime():
    """Current timestamp as MMDD-HHMMSS."""
    return _datetime.datetime.now().strftime("%m%d-%H%M%S")


def set_seed(seed = 0):
    """Seed python/numpy/torch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mps_available():
    """True when the Apple-silicon (MPS) backend is usable."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def set_device(device):
    """Resolve a compute device: CUDA (by index) -> MPS -> raise."""
    if torch.cuda.is_available():
        if device >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA error, invalid device ordinal {device} "
                f"(visible devices={torch.cuda.device_count()})"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        return torch.device("cuda:" + str(device))
    if mps_available():
        # There is a single MPS device; the integer arg is irrelevant.
        return torch.device("mps")
    raise RuntimeError(
        "No CUDA or MPS device available. Enable a CPU fallback at the call site."
    )


# ----------------------------- SAM optimizer -----------------------------


class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "Sharpness Aware Minimization requires closure, but it was not provided"
        )
        closure = torch.enable_grad()(closure)  # full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def smooth_crossentropy(pred, gold, smoothing=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


# ----------------------------- CIFAR datasets -----------------------------


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
_CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left):right, max(0, top):bottom] = 0
        return image


def _cifar_load(
    *,
    which,
    root,
    normalize,
    randomize,
    cutout,
):
    from torchvision import datasets, transforms

    if which == "cifar10":
        mean, std, ds_cls = _CIFAR10_MEAN, _CIFAR10_STD, datasets.CIFAR10
    else:
        mean, std, ds_cls = _CIFAR100_MEAN, _CIFAR100_STD, datasets.CIFAR100

    transform_train = transforms.Compose([
        *([transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip()] if randomize else []),
        transforms.ToTensor(),
        *([transforms.Normalize(mean, std)] if normalize else []),
        *([Cutout()] if cutout else []),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        *([transforms.Normalize(mean, std)] if normalize else []),
    ])

    os.makedirs(root, exist_ok=True)
    trainset = ds_cls(root=root, train=True, download=True, transform=transform_train)
    testset = ds_cls(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset


def prepare_dataset(
    dataset,
    root = "../data/",
    normalize = True,
    randomize = True,
    cutout = False,
    **_kwargs,
):
    """Prepare (trainset, testset). Supports ``cifar10`` and ``cifar100``."""
    if dataset in ("cifar10", "cifar100"):
        return _cifar_load(
            which=dataset,
            root=root,
            normalize=normalize,
            randomize=randomize,
            cutout=cutout,
        )
    raise NotImplementedError(f"dataset {dataset} is not implemented.")



def prepare_model(model_name, dataset, ini_seed = 0):
    from sam_models import (
        cifar_resnet,
        cifar_vgg,
        cifar_vgg_plus,
        cifar_wide_resnet,
        cifar_wide_resnet_madry,
    )

    set_seed(ini_seed)
    if not dataset.startswith("cifar"):
        raise ValueError(f"{dataset} is not supported.")
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    else:
        raise ValueError(f"{dataset} is not supported.")

    if model_name.startswith("vgg"):
        try:
            model = cifar_vgg.__dict__[model_name](num_classes=num_classes)
        except Exception:
            model = cifar_vgg_plus.__dict__[model_name](num_classes=num_classes)
    elif model_name.startswith("ResNet") and dataset == "cifar10":
        model = cifar_resnet.__dict__[model_name]()
    elif model_name.startswith("WideResNet"):
        if model_name.endswith("madry"):
            model = cifar_wide_resnet_madry.__dict__[model_name](num_classes=num_classes)
        else:
            model = cifar_wide_resnet.__dict__[model_name](num_classes=num_classes)
    else:
        raise ValueError(f"unknown model name: {model_name} for dataset {dataset}")
    return model
