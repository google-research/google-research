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

"""Utility functions."""
import functools
from typing import Callable, Union

from optax import losses


def masked_loss(output, target, mask, loss_fn):
  """Compute the value of a masked loss.

  Args:
    output: jnp.ndarray-like [bs, t, h, w, num_features, ...]
    target: jnp.ndarray-like [bs, t, h, w, num_features]
    mask: jnp.ndarray-like [bs, t, h, w, num_features]
    loss_fn: loss function returning per pixel losses

  Returns:
    The masked loss.
  """
  if output.shape[: target.ndim] != target.shape or target.shape != mask.shape:
    raise ValueError(
        f'Output shape is {output.shape}, target shape is {target.shape} and'
        f' mask shape is {mask.shape}.'
    )
  if target.ndim != 5:  # batch, t, h, w, features
    raise ValueError(f'Target should be 5 dimensional but was {target.ndim}.')

  axis = (0, 1, 2, 3, 4)  # batch, t, h, w, channel

  loss = loss_fn(output, target)
  assert loss.shape == mask.shape, f'{loss.shape} != {mask.shape}'
  loss *= mask

  return loss.sum(axis=axis) / (mask.sum(axis=axis) + 1e-6)


masked_cross_entropy_loss = functools.partial(
    masked_loss, loss_fn=losses.softmax_cross_entropy_with_integer_labels
)


def get_tuple(val):
  if isinstance(val, tuple):
    return val
  return (val, val)


def satisfies_op(
    a,
    b,
    op,
):
  if isinstance(a, int) and isinstance(b, int):
    return op(a, b)
  a = get_tuple(a)
  b = get_tuple(b)
  return op(a[0], b[0]) and op(a[1], b[1])
