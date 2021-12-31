# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# pylint: skip-file
from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from math import ceil, log2
import copy

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def log_2(n):
    r = 0
    while n > 1:
        n = n >> 1
        r += 1
    return r

def log_2_ceil(n):
    return ceil(log2(n))

def cal_max_idx(n):
    max_idx = 0
    for i in range(log_2_ceil(n)):
        assert n != 0
        max_idx += n
        n = n >> 1
    return max_idx

def shift_right(x, axis=1):
    """Shift the input to the right by padding and slicing on axis."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
    )
    return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_left(x, axis=1):
    """Shift the input to the right by padding and slicing on axis."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, 1)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
    )
    # print(padded)
    return lax.dynamic_slice_in_dim(padded, 1, padded.shape[axis] - 1, axis)


def make_causal_mask(x, length_axis, extra_batch_dims=0, strict=False):
    idxs = jnp.broadcast_to(jnp.arange(x.shape[length_axis], dtype=jnp.int32),
                            x.shape[:length_axis + 1])
    mask = nn.make_attention_mask(idxs, idxs, jnp.greater_equal if not strict else jnp.greater,
                             extra_batch_dims=extra_batch_dims, dtype=jnp.float32)
    return mask

# Util functions
def _create_expr(symbols, prefix='B', suffix='NK'):
  return prefix + ''.join(symbols) + suffix


def _insert(l, index, elem):
  l = copy.deepcopy(l)
  l.insert(index, elem)
  return l

if __name__ == '__main__':

    x = jnp.ones((2, 3, 4))
    mask = make_causal_mask(x, length_axis=1)
    print(mask)
    print(shift_left(x, axis=1))
