# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Batch norm stuff."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
# pylint: disable=g-complex-comprehension
# pylint: disable=using-constant-test
# pylint: disable=g-explicit-length-test
# pylint: disable=undefined-variable


import numpy as np
import torch
from torch import nn
import torchvision


def turnNormOff(model,):
  turnBNoff(model)
  turnGNoff(model)


def turnBNoff(model,):
  for m in model.modules():
    if (isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or
        isinstance(m, nn.BatchNorm3d)):
      m.eval()
    if isinstance(m, nn.SyncBatchNorm):
      m.eval()


def turnGNoff(model,):
  for m in model.modules():
    if (isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or
        isinstance(m, nn.BatchNorm3d)):
      m.eval()
