# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Some utility numpy/pytorch functions to reshape and split variables."""

from __future__ import print_function

import math
import numpy as np
import torch

VAR_SHAPE = 'var_shape'
VAR_SPLITS = 'var_splits'


def merge_small_dims(var_shape, reshape_size):
  """Computes the shape of the variable for preconditioning.

  If the variable has several small dimensions, we can reshape it so
  that there are fewer big ones. e.g for a convolution (512, 5, 5, 1024)
  we could reshape it into (512, 25, 1024).

  Args:
    var_shape: the shape of the variable
    reshape_size: maximum size of a reshaped dimension
  Returns:
    shape: a list of integers. Product(shape) = number of elements in var.
  """
  if var_shape and np.all(np.array(var_shape) == 1):
    return [1]
  shape = []
  product = 1
  for d in var_shape:
    if product * d <= reshape_size:
      product *= d
    else:
      if product > 1: shape.append(product)
      product = d
  if product > 1:
    shape.append(product)
  return shape


def compute_splits(var_shape, block_size):
  """Splits larger dimensions into smaller ones, for preconditioning.

  For example, if a variable has shape (4096, 512), we might split the
  4096 into 4 blocks, so we effectively have 4 variables of size
  (1024, 512) each.

  Args:
    var_shape: list of integers, the shape to be split
    block_size: the maximum dimension of each block
  Returns:
    splits: set of tuples (i, split) if the i-th dimension should be split
    split_sizes: an array of tuples, one per dimension, each indicating how
                 to split that dimension.
  """
  splits = []
  split_sizes = []
  for i, d in enumerate(var_shape):
    if block_size > 0 and d > block_size:
      nsplit = math.ceil(d / block_size)
      sizes = np.ones(nsplit, dtype=np.int32) * block_size
      if d % block_size > 0:
        sizes[-1] = d % block_size
      splits.append((i, tuple(sizes)))
      split_sizes.append(sizes)
    else:
      split_sizes.append(np.array([d], dtype=np.int32))
  return splits, split_sizes


def split_grad(state, grad):
  """Split up the gradient according to the blocking strategy."""
  if len(state[VAR_SHAPE]) < len(list(grad.shape)):
    grad = torch.reshape(grad, state[VAR_SHAPE])
  grads = [grad]
  for i, split_sizes in state[VAR_SPLITS]:
    split_grads = []
    for grad in grads:
      split_grads.extend(torch.split(grad, split_sizes, dim=i))
    grads = split_grads
  return grads


def merge_grads(state, grads):
  """Merge the split gradients back into a single array."""
  for i, split_sizes in reversed(state[VAR_SPLITS]):
    n = len(split_sizes)
    conc_grads = []
    ind = 0
    while ind < len(grads):
      conc_grads.append(torch.cat(grads[ind:ind+n], axis=i))
      ind += n
    grads = conc_grads
  assert len(grads) == 1
  return grads[0]
