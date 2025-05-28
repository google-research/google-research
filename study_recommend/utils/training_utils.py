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

"""Utility functions for training STUDY recommender models."""

from collections.abc import Callable
from typing import Optional, Union

import jax
from jax import lax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import optax


def _rsqrt_schedule(
    init_value,
    shift = 0,
):
  """Applies a reverse square-root schedule.

  The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

  Args:
    init_value: Base learning rate (before applying the rsqrt schedule).
    shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
      schedule makes it less steep in the beginning (close to 0).

  Returns:
    A schedule that applies the reverse square root.
  """

  epsilon = 0.01

  def schedule(count):
    return init_value * (count + shift + epsilon) ** -0.5 * shift**0.5

  return schedule


def create_learning_rate_schedule(
    learning_rate, warmup_steps
):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules(
      [
          optax.linear_schedule(
              init_value=0,
              end_value=learning_rate,
              transition_steps=warmup_steps,
          ),
          _rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
      ],
      boundaries=[warmup_steps],
  )


def compute_weighted_cross_entropy(
    logits,
    targets,
    weights = None,
):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array. Output logits from model.
   targets: [batch, length] int array. Ground truth categorical labels.
   weights: None or array of shape [batch, length]. Used as a multiplicative
     weighting for the loss at each token.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )

  loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(
    logits,
    targets,
    oov_value = None,
    weights = None,
):
  """Compute weighted accuracy for log probs and targets.

  Args:
    logits: [batch, length, num_classes] float array. Output logits from the
      model.
    targets: [batch, length] int array. Categorical target labels.
    oov_value: The value assigned to out-of-vocabulary tokens.
    weights: None or array of shape [batch, length]. Used as multiplicative
      weight for accuracy at each token.

  Returns:
    oov_adjusted_correct_preds: The number of predictions the model made
      correctly. OOV correction is applied. When the model correctly
      predicts that the next token will be OOV we consider this to be a trivial
      prediction and do not count it as correct. This will be an a jax.Array
      iff the inputs passed to the function are sharded across multiple
      accelerators.
    correct_preds: Raw number of predictions the model made correctly. This will
      be an a jax.Array iff the inputs passed to the function are sharded
      across multiple accelerators.
    normalizing_factor: A normalizing factor computed from weights if provided
      else computed from the total number of tokens provided.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets'
        % (str(logits.shape), str(targets.shape))
    )
  correct_preds = jnp.equal(jnp.argmax(logits, axis=-1), targets)

  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    correct_preds = correct_preds * weights
    normalizing_factor = weights.sum()

  if oov_value is not None:
    oov_adjusted_correct_preds = correct_preds * (targets != oov_value)
  else:
    oov_adjusted_correct_preds = correct_preds

  return (
      oov_adjusted_correct_preds.sum(),
      correct_preds.sum(),
      normalizing_factor,
  )


def compute_metrics(
    logits,
    targets,
    weights = None,
    oov_value = None,
):
  """Compute summary metrics.

  Args:
    logits: [batch, length, num_classes] float array. Output logits from the
      model.
    targets: [batch, length] int array. Categorical target labels.
    weights: None or array of shape [batch, length]. Used as multiplicative
      weight for the metrics at each token.
    oov_value: The value assigned to out-of-vocabulary tokens.

  Returns:
    A dictionary of metrics computed from given logits. Metrics returned are
    aggregate sums. A denominator to normalize is also provided for users to
    compute averages downstream.
  """
  loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
  real_acc, acc, _ = compute_weighted_accuracy(
      logits, targets, oov_value, weights
  )
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
      'oov_corrected_accuracy': real_acc,
  }
  metrics = lax.psum(metrics, axis_name='batch')
  return metrics


def compute_weight_matrix(
    titles, separator_token = None
):
  """Return weight matrix that assigns zero weight to non-title positions .

  The weight matrix is used to zero out losses and metrics computed on padding
  or separator token inputs.

  Args:
    titles: An array of tokenized titles
    separator_token: The value assigned to separator tokens.

  Returns:
    A float jnp.ndarray with with binary values.
  """

  # Assign 0 weight to padding, 1 elsewhere
  weights = jnp.where(titles > 0, True, False)

  if separator_token is not None:
    # Assign zero weight to separator_tokens
    weights = jnp.logical_and(weights, titles != separator_token)
  return weights


def normalize_metrics(metrics):
  """Normalize the metric aggregates to get the mean values."""
  metrics = jax.tree.map(jnp.sum, metrics)
  denominator = metrics.pop('denominator')
  metrics = jax.tree_util.tree_map(lambda x: (x / denominator).item(), metrics)
  return metrics


def convert_host_local_array_to_global_array(arr):
  """Converts a host local array from pmap to global jax.Array.

  This is a utiltiy function for saving pytrees with orbax.

  Args:
    arr: Input host local array produced by pmap.

  Returns:
    A global array that can be checkpointed with orbax.
  """
  # input `arr` is fully replicated, so it's shape is the global shape.
  global_shape = arr.device_buffers[0].shape
  # Create a 1D mesh to create fully replicated global jax.Array.
  mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
  partition_spec = jax.sharding.PartitionSpec(None)
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(arr.device_buffers, key=lambda x: x.device().id)
  return jax.make_array_from_single_device_arrays(
      global_shape, jax.sharding.NamedSharding(mesh, partition_spec), dbs
  )
