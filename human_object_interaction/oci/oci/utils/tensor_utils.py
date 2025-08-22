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

"""Tensor Utils."""
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
import numpy as np
import torch


def tensor_to_cuda(tensor, cuda, tensor_type=torch.FloatTensor):
  if torch.Tensor != type(tensor):
    tensor = tensor_type(tensor * 1)

  if cuda:
    tensor = tensor.cuda()
  else:
    tensor = tensor.cpu()
  return tensor


def tensor_to_numpy(tensor):
  if isinstance(tensor) == torch.Tensor:
    if tensor.device.type == "cuda":
      tensor = tensor.cpu()
    return tensor.data.numpy()
  elif isinstance(tensor) == np.ndarray:
    return tensor
  else:
    return tensor


copy2cpu = lambda tensor: tensor.detach().cpu().numpy()

numpy2torch = torch.Tensor
