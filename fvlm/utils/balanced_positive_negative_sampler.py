# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Class to subsample minibatches by balancing positives and negatives.

This is a JAX reimplementation of the following files:
third_party/tensorflow_models/object_detection/core/balanced_positive_negative_sampler.py
third_party/tensorflow_models/object_detection/core/minibatch_sampler.py

Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired batch_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired batch_size, it returns fewer examples.

The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.
"""
import abc
from typing import Tuple, Optional, Union

import jax
from jax import random
import jax.numpy as jnp


Array = jnp.ndarray


def indices_to_dense_vector(indices,
                            size,
                            indices_value = 1.0,
                            default_value = 0.0,
                            dtype = jnp.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  The TF implementation depends on jnp.dynamic_stitch which does not exist in
  JAX. We substitute it with jax.ops.index_update.

  Args:
    indices: 1d array with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output array.
    indices_value: values of elements specified by indices in the output
        vector.
    default_value: values of other elements in the output vector.
    dtype: output data type.

  Returns:
    output: dense 1D array of shape [size] with indices set to indices_values
        and the rest set to default_value.
  """
  indices = indices.astype(jnp.int32)
  output = jnp.ones([size], dtype=dtype) * default_value
  output = output.at[indices].set(
      jnp.ones_like(indices, dtype=dtype) * indices_value)
  return output.astype(dtype)


class MinibatchSampler(metaclass=abc.ABCMeta):
  """Abstract base class for subsampling minibatches."""

  @abc.abstractmethod
  def subsample(self, indicator, batch_size, **params):
    """Returns subsample of entries in indicator.

    Args:
      indicator: boolean array of shape [N] whose True entries can be sampled.
      batch_size: desired batch size.
      **params: additional keyword arguments for specific implementations of
          the MinibatchSampler.

    Returns:
      sample_indicator: boolean array of shape [N] whose True entries have been
      sampled. If sum(indicator) >= batch_size, sum(is_sampled) = batch_size
    """
    pass

  @staticmethod
  def subsample_indicator(indicator,
                          num_samples,
                          key = None):
    """Subsample indicator vector in a non-static fashion (no XLA support).

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Args:
      indicator: a 1-dimensional boolean array indicating which elements
        are allowed to be sampled and which are not.
      num_samples: int32 scalar array
      key: a key representing the state of JAX random function.

    Returns:
      a boolean array with the same shape as input (indicator) array
    """
    if indicator.ndim != 1:
      raise ValueError(f'Input shape {indicator.shape} must be 1-d!')

    if indicator.dtype != jnp.bool_:
      raise ValueError(
          f'indicator should be of type bool. Received: {indicator.dtype}')

    if indicator.size == 0 or jnp.sum(indicator) < num_samples:
      return indicator.astype(jnp.bool_)

    if key is None:
      key = random.PRNGKey(0)

    indices = jnp.where(indicator)[0]
    indices = random.permutation(key, indices)
    selected_indices = indices[0:jnp.minimum(indices.size, num_samples)]
    selected_indicator = indices_to_dense_vector(
        selected_indices, jnp.shape(indicator)[0]) == 1.
    return selected_indicator


class BalancedPositiveNegativeSampler(MinibatchSampler):
  """Subsamples minibatches to a desired balance of positives and negatives."""

  def __init__(self, positive_fraction = 0.5, is_static = False):
    """Constructs a minibatch sampler.

    Args:
      positive_fraction: desired fraction of positive examples (scalar in [0,1])
        in the batch.
      is_static: If True, uses an implementation with static shape guarantees.

    Raises:
      ValueError: if positive_fraction < 0, or positive_fraction > 1
    """
    if positive_fraction < 0 or positive_fraction > 1:
      raise ValueError(f'positive_fraction should be in range [0,1]. '
                       f'Received: {positive_fraction}.')
    self._positive_fraction = positive_fraction
    self._is_static = is_static

  def subsample(
      self,
      indicator,
      batch_size,
      labels,
      key,
  ):
    """Returns subsampled minibatch.

    Args:
      indicator: boolean array of shape [N] whose True entries can be sampled.
      batch_size: desired batch size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches self._positive_fraction.
      labels: boolean array of shape [N] denoting positive(=True) and negative
          (=False) examples.
      key: a key representing the state of JAX random function.

    Returns:
      sampled_idx_indicator: boolean array of shape [N], True for entries which
        are sampled.

    Raises:
      ValueError: if labels and indicator are not 1D boolean arrays.
    """
    if len(indicator.shape) != 1:
      raise ValueError(f'indicator must be 1 dimensional, got a array of '
                       f'shape {indicator.shape}')
    if len(labels.shape) != 1:
      raise ValueError(f'labels must be 1 dimensional, got a array of '
                       f'shape {labels.shape}')
    if labels.dtype != jnp.bool_:
      raise ValueError(
          f'labels should be of type bool. Received: {labels.dtype}')
    if indicator.dtype != jnp.bool_:
      raise ValueError(
          f'indicator should be of type bool. Received: {indicator.dtype}.')

    if self._is_static:
      return self._static_subsample(indicator, batch_size, labels, key)

    # Only sample from indicated samples
    negative_idx = jnp.logical_not(labels)
    positive_idx = jnp.logical_and(labels, indicator)
    negative_idx = jnp.logical_and(negative_idx, indicator)

    # Sample positive and negative samples separately
    if batch_size is None:
      max_num_pos = jnp.sum(positive_idx.astype(dtype=jnp.int32))
    else:
      max_num_pos = int(self._positive_fraction * batch_size)
    sampled_pos_idx = self.subsample_indicator(positive_idx, max_num_pos, key)
    num_sampled_pos = jnp.sum(sampled_pos_idx.astype(jnp.int32))
    if batch_size is None:
      negative_positive_ratio = (
          1.0 - self._positive_fraction) / self._positive_fraction
      max_num_neg = (negative_positive_ratio * num_sampled_pos).astype(
          dtype=jnp.int32)
    else:
      max_num_neg = batch_size - num_sampled_pos
    sampled_neg_idx = self.subsample_indicator(negative_idx, max_num_neg, key)

    return jnp.logical_or(sampled_pos_idx, sampled_neg_idx)

  def _get_num_pos_neg_samples(self, sorted_indices_tensor,
                               sample_size):
    """Counts the number of positives and negatives numbers to be sampled.

    Args:
      sorted_indices_tensor: A sorted int32 tensor of shape [N] which contains
        the signed indices of the examples where the sign is based on the label
        value. The examples that cannot be sampled are set to 0. It samples
        at most sample_size*positive_fraction positive examples and remaining
        from negative examples.
      sample_size: Size of subsamples.

    Returns:
      A tuple containing the number of positive and negative labels in the
      subsample.
    """
    valid_positive_index = sorted_indices_tensor > 0
    num_sampled_pos = jnp.sum(valid_positive_index.astype(jnp.int32))
    max_num_positive_samples = int(sample_size * self._positive_fraction)
    num_positive_samples = jnp.minimum(max_num_positive_samples,
                                       num_sampled_pos)
    num_negative_samples = sample_size - num_positive_samples
    return num_positive_samples, num_negative_samples

  def _get_values_from_start_and_end(self, input_tensor,
                                     num_start_samples,
                                     num_end_samples,
                                     total_num_samples):
    """slices num_start_samples and last num_end_samples from input_tensor.

    Args:
      input_tensor: An int32 tensor of shape [N] to be sliced.
      num_start_samples: Number of examples to be sliced from the beginning
        of the input tensor.
      num_end_samples: Number of examples to be sliced from the end of the
        input tensor.
      total_num_samples: Sum of is num_start_samples and num_end_samples. This
        should be a scalar.

    Returns:
      A tensor containing the first num_start_samples and last num_end_samples
      from input_tensor.

    """
    input_length = input_tensor.shape[0]
    start_positions = jnp.arange(input_length) < num_start_samples
    end_positions = jnp.arange(input_length) >= (input_length - num_end_samples)
    selected_positions = jnp.logical_or(start_positions, end_positions)
    selected_positions = selected_positions.astype(jnp.float32)
    indexed_positions = jnp.multiply(jnp.cumsum(selected_positions),
                                     selected_positions)
    one_hot_selector = jax.nn.one_hot(indexed_positions.astype(jnp.int32) - 1,
                                      total_num_samples,
                                      dtype=jnp.float32)
    return jnp.tensordot(input_tensor.astype(jnp.float32),
                         one_hot_selector, axes=[0, 0]).astype(jnp.int32)

  def _static_subsample(self, indicator, batch_size, labels,
                        key):
    """Returns subsampled minibatch.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
        N should be a complie time constant.
      batch_size: desired batch size. This scalar cannot be None.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
        (=False) examples. N should be a complie time constant.
      key: a key representing the state of JAX random function.

    Returns:
      sampled_idx_indicator: boolean tensor of shape [N], True for entries which
        are sampled. It ensures the length of output of the subsample is always
        batch_size, even when number of examples set to True in indicator is
        less than batch_size.

    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
    input_length = indicator.shape[0]

    # Set the number of examples set True in indicator to be at least
    # batch_size.
    num_true_sampled = jnp.sum(indicator.astype(jnp.float32))
    additional_false_sample = jnp.cumsum(
        jnp.logical_not(indicator).astype(jnp.float32)) <= (
            batch_size - num_true_sampled)
    indicator = jnp.logical_or(indicator, additional_false_sample)

    # Shuffle indicator and label. Need to store the permutation to restore the
    # order post sampling.
    permutation = random.permutation(key, input_length)
    indicator = indicator[permutation]
    labels = labels[permutation]

    # index (starting from 1) when indicator is True, 0 when False
    indicator_idx = jnp.where(
        indicator, jnp.arange(1, input_length + 1),
        jnp.zeros(input_length, jnp.int32))

    # Replace -1 for negative, +1 for positive labels
    signed_label = jnp.where(
        labels,
        jnp.ones(input_length, jnp.int32),
        -jnp.ones(input_length, jnp.int32))
    # negative of index for negative label, positive index for positive label,
    # 0 when indicator is False.
    signed_indicator_idx = jnp.multiply(indicator_idx, signed_label)
    sorted_signed_indicator_idx = jnp.sort(signed_indicator_idx)[::-1]

    [num_positive_samples,
     num_negative_samples] = self._get_num_pos_neg_samples(
         sorted_signed_indicator_idx, batch_size)

    sampled_idx = self._get_values_from_start_and_end(
        sorted_signed_indicator_idx, num_positive_samples,
        num_negative_samples, batch_size)

    # Shift the indices to start from 0 and remove any samples that are set as
    # False.
    sampled_idx = jnp.absolute(sampled_idx) - jnp.ones(batch_size, jnp.int32)
    sampled_idx = (sampled_idx >= 0) * sampled_idx
    sampled_idx_indicator = jnp.sum(jax.nn.one_hot(
        sampled_idx, num_classes=input_length), axis=0)

    # project back the order based on stored permutations
    idx_indicator = jnp.zeros((input_length,))
    idx_indicator = idx_indicator.at[permutation].set(sampled_idx_indicator)
    return idx_indicator.astype(jnp.bool_)
