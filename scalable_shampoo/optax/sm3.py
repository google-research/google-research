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

# An implementation of SM3 from:
#
# Memory-Efficient Adaptive Optimization, https://arxiv.org/pdf/1901.11150.pdf
# Rohan Anil, Vineet Gupta, Tomer Koren, Yoram Singer
#
# Author: Rohan Anil (rohananil at google dot com)
#

"""SM3 Implementation."""

import functools
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax

from scalable_shampoo.optax.quantization_utils import QuantizedValue


class SM3State(NamedTuple):
  count: chex.Array
  stats: Any


# Per parameter optimizer state used in data-parallel training.
class ParameterStats(NamedTuple):
  """State associated to each parameter of the model being trained."""
  diagonal_statistics: chex.Array  # Accumulator for diagonal preconditioner
  diagonal_momentum: QuantizedValue  # Momentum for the diagonal preconditioner


def sm3(
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    diagonal_epsilon=1e-10,
    normalize_grads=False):
  """SM3 optimizer.

  Memory-Efficient Adaptive Optimization, Rohan Anil, Vineet Gupta, Tomer Koren,
    Yoram Singer

  https://arxiv.org/abs/1901.11150

  Args:
    learning_rate: the step size used to update the parameters.
    beta1: momentum parameter.
    beta2: second moment averaging parameter.
    diagonal_epsilon: epsilon for sm3
    normalize_grads: Whether to normalize grads. Author finds it useful when
      grads are high variance.

  Returns:
    a GradientTransformation.
  """

  def _quantize_momentum(momentum_statistics):
    return QuantizedValue.from_float_value(momentum_statistics, jnp.int8)

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      accumulators = [jnp.zeros([s]) for s in param.shape]
      momentum = _quantize_momentum(jnp.zeros_like(param))
      return ParameterStats(accumulators, momentum)

    return SM3State(
        count=jnp.zeros([], jnp.int32), stats=jax.tree_map(_init, params))

  def _get_expanded_shape(shape, i):
    rank = len(shape)
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    return [1] * i + [shape[i]] + [1] * (rank - i - 1)

  def _moving_averages(grad, accumulators):
    w = (1.0 - beta2) if beta2 != 1.0 else 1.0
    if grad.ndim < 2:
      return beta2 * accumulators[0] + w * grad**2
    else:
      min_accumulator = functools.reduce(jnp.minimum, accumulators)
      return beta2 * min_accumulator + w * grad**2

  def _moving_averages_momentum(grad, momentum):
    w = (1.0 - beta1) if beta1 != 1.0 else 1.0
    return beta1 * momentum.to_float() + w * grad

  def _sketch_diagonal_statistics(grad, updated_diagonal_statistics):
    all_diagonal_statistics = []
    for i in range(grad.ndim):
      axes = list(range(i)) + list(range(i + 1, grad.ndim))
      dim_diagonal_statistics = jnp.max(updated_diagonal_statistics, axis=axes)
      all_diagonal_statistics.append(dim_diagonal_statistics)
    if grad.ndim == 1:
      all_diagonal_statistics[0] = updated_diagonal_statistics
    return all_diagonal_statistics

  def update_fn(updates, state, params=None):
    del params
    stats = state.stats
    if normalize_grads:
      updates = jax.tree_map(
          lambda g: g / (jnp.linalg.norm(g) + 1e-16), updates)
    # Reshape all vectors into N-d tensors to compute min over them.
    # [n], [m] -> [n, 1], [1, m]
    expanded_diagonal_statistics = jax.tree_multimap(
        lambda grad, state:  # pylint:disable=g-long-lambda
        [
            jnp.reshape(state.diagonal_statistics[i],
                        _get_expanded_shape(grad.shape, i))
            for i in range(grad.ndim)
        ],
        updates,
        stats)

    # Compute new diagonal statistics
    new_diagonal_statistics = jax.tree_multimap(_moving_averages, updates,
                                                expanded_diagonal_statistics)

    # Compute preconditioners (1/sqrt(s)) where s is the statistics.
    new_preconditioners = jax.tree_map(
        lambda t: 1.0 / jnp.sqrt(t + diagonal_epsilon), new_diagonal_statistics)
    preconditioned_grads = jax.tree_multimap(lambda g, p: g * p, updates,
                                             new_preconditioners)

    # Compute updated momentum (also handle quantization)
    updated_momentum = jax.tree_multimap(
        lambda preconditioned_grad, state:  # pylint:disable=g-long-lambda
        _moving_averages_momentum(preconditioned_grad, state.diagonal_momentum),
        preconditioned_grads,
        stats)

    # Update diagonal statistics.
    updated_diagonal_statistics = jax.tree_multimap(
        _sketch_diagonal_statistics,
        updates,
        new_diagonal_statistics)

    # Update momentum.
    new_sm3_stats = jax.tree_multimap(
        lambda momentum, diagonal_stats:  # pylint:disable=g-long-lambda
        ParameterStats(diagonal_stats, _quantize_momentum(momentum)),
        updated_momentum,
        updated_diagonal_statistics)

    lr = learning_rate
    if callable(learning_rate):
      lr = learning_rate(state.count)

    new_updates = jax.tree_map(lambda pg: -lr * pg, updated_momentum)
    return new_updates, SM3State(count=state.count+1, stats=new_sm3_stats)

  return optax.GradientTransformation(init_fn, update_fn)
