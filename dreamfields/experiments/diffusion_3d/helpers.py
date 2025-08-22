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

"""Helpers methods."""

import random

import numpy as np
import torch
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd


def set_seed(seed):
  """Reproducibility."""
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def get_device():
  """Obtain the utilized device."""
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    has_cuda = True
  else:
    device = torch.device("cpu")
    has_cuda = False

  return device, has_cuda


def mean(tensor, start_dim=1):
  return torch.mean(tensor, dim=tuple(range(start_dim, tensor.ndim)))


def weighted_mean(tensor, weights):
  assert weights.ndim == 1
  assert weights.size(0) == tensor.size(0)
  with torch.no_grad():
    norm = torch.sum(weights).item()
    assert np.isclose(norm, 1.)

  return torch.sum(weights * mean(tensor, start_dim=1))


def psnr(squared_err):
  mean_err = mean(squared_err, start_dim=1)
  return -torch.mean(10 * torch.log10(mean_err))


def quantize(image):
  """Quantize a [0, 1] scaled float image to uint8 [0, 255]."""
  return (image * 255.).astype(np.uint8)


@torch.jit.script
def linspace(start, stop, num):
  """Linearly space values between multi-dimensional endpoints in PyTorch.

  Creates a tensor of shape [num, *start.shape] whose values are evenly spaced
  from start to end, inclusive. Replicates the multi-dimensional bahavior of
  numpy.linspace in PyTorch. Adds steps as a new, final dimension.
  Based on https://github.com/pytorch/pytorch/issues/61292.

  Args:
    start: Beginning values of range.
    stop: Ending values of range.
    num: Number of points in range, including endpoints.

  Returns:
    linspace
  """
  # create a tensor of "num" steps from 0 to 1.
  steps = torch.arange(num, dtype=start.dtype, device=start.device) / (num - 1.)
  # the output starts at "start" and increments until "stop" in each dimension,
  # adding a final dim.
  return start.unsqueeze(-1) + steps * (stop - start).unsqueeze(-1)


@torch.jit.script
def compute_tv_norm(values, losstype = "l2"):
  """Returns TV norm for input values.

  Source: regnerf/internal/math.py

  Args:
    values: [batch, H, W, *]. 3 or more dimensional tensor.
    losstype: l2 or l1

  Returns:
    loss: [batch, H-1, W-1, *]
  """
  v00 = values[:, :-1, :-1]
  v01 = values[:, :-1, 1:]
  v10 = values[:, 1:, :-1]

  if losstype == "l2":
    loss = ((v00 - v01)**2) + ((v00 - v10)**2)
  elif losstype == "l1":
    loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
  else:
    raise ValueError(f"Unsupported TV losstype {losstype}.")

  return loss


def grad_norm(model=None, parameters=None):
  """Compute parameter gradient norm."""
  assert parameters is not None or model is not None

  total_norm = 0
  if parameters is None:
    parameters = []
  if model is not None:
    parameters.extend(model.parameters())
  parameters = [p for p in parameters if p.grad is not None and p.requires_grad]
  for p in parameters:
    param_norm = p.grad.detach().data.norm(2)
    total_norm += param_norm.item()**2
  total_norm = total_norm**0.5
  return total_norm


def clamp_and_detach(a):
  return a.clamp(0.0, 1.0).cpu().detach().numpy()


# pylint: disable=redefined-builtin
class DifferentiableClamp(torch.autograd.Function):
  """Differentiable clamp operation based on a straight through estimator.

  In the forward pass this operation behaves like torch.clamp.
  But in the backward pass its gradient is 1 everywhere, as if instead of
  clamp one had used the identity function.
  Source: https://discuss.pytorch.org/t/exluding-torch-clamp-from-
  backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
  """

  @staticmethod
  @custom_fwd
  def forward(ctx, input, min, max):
    return input.clamp(min=min, max=max)

  @staticmethod
  @custom_bwd
  def backward(ctx, grad_output):
    return grad_output.clone(), None, None


def dclamp(input, min, max):
  """Like torch.clamp, but with a constant 1-gradient.

  Source: https://discuss.pytorch.org/t/exluding-torch-clamp-from-
  backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6

  Args:
    input: The input that is to be clamped.
    min: The minimum value of the output.
    max: The maximum value of the output.

  Returns:
    clamped input
  """
  return DifferentiableClamp.apply(input, min, max)
