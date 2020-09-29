# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Data processing utility functions."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def flatten_last_dims(x, num_last_dims):
  """Flattens the last dimensions of a tensor.

  For example:
    x.shape.as_list() == [1, 2, 3, 4].
    flatten_last_dims(x, num_last_dims=2).shape.as_list() == [1, 2, 12].

  Args:
    x: A tensor to flatten.
    num_last_dims: An integer for the number of last dimensions to flatten.

  Returns:
    A flattened tensor.

  Raises:
    ValueError: If `num_last_dims` is greater than the total dimensionality of
      `x`.
    ValueError: If any of the `num_last_dims` dimensions of `x` is None.
  """
  shape_list = x.shape.as_list()
  if num_last_dims > len(shape_list):
    raise ValueError(
        'Number of last dimensions must not be greater than the total '
        'dimensionality of input tensor: %d vs. %d.' %
        (num_last_dims, len(shape_list)))
  for i in range(num_last_dims):
    if shape_list[-i - 1] is None:
      raise ValueError(
          'Shape of `x` has None dimension within `num_last_dims`: `%s`.' %
          shape_list)

  shape_list = [-1 if d is None else d for d in shape_list]
  last_dim = np.prod(shape_list[-num_last_dims:])
  shape_list = shape_list[:-num_last_dims] + [last_dim]
  return tf.reshape(x, shape_list)


def flatten_first_dims(x, num_last_dims_to_keep):
  """Flattens the first dimensions of a tensor.

  For example:
    x.shape.as_list() == [1, 2, 3, 4, 5].
    flatten_first_dims(x, num_last_dims_to_keep=2).shape.as_list() == [6, 4, 5].

  Returns:
    A flattened tensor.

  Args:
    x: A tensor to flatten.
    num_last_dims_to_keep: An integer for the number of last dimensions to keep.
      The rest of the proceeding dimensions will be flattened.

  Raises:
    ValueError: If `num_last_dims_to_keep` is greater than the total
      dimensionality of `x`.
    ValueError: If any of the `num_last_dims_to_keep` dimensions of `x` is None.
  """
  shape_list = x.shape.as_list()
  if num_last_dims_to_keep > len(shape_list):
    raise ValueError(
        'Number of last dimensions must not be greater than the total '
        'dimensionality of input tensor: %d vs. %d.' %
        (num_last_dims_to_keep, len(shape_list)))

  new_shape_list = []
  for i in range(num_last_dims_to_keep):
    last_dim = shape_list[-i - 1]
    if last_dim is None:
      raise ValueError('Shape of `x` has None dimension within '
                       '`num_last_dims_to_keep`: `%s`.' % shape_list)
    new_shape_list.append(last_dim)

  new_shape_list.append(-1)
  new_shape_list.reverse()
  return tf.reshape(x, new_shape_list)


def tile_first_dims(x, first_dim_multiples):
  """Tiles the first dimensions of a tensor.

  For example:
    x.shape.as_list() = [2, 3, 4].
    tile_first_dims(x, first_dim_multiples=[2, 3]).shape_as_list() == [4, 9, 4].

  Args:
    x: A tensor to tile.
    first_dim_multiples: A list of integers for the multiples to tile the first
      dimensions.

  Returns:
    A tiled tensor.

  Raises:
    ValueError: If length of `first_dim_multiples` is greater than the total
      dimensionality of `x`.
  """
  shape_list = x.shape.as_list()
  num_first_dims = len(first_dim_multiples)
  if num_first_dims > len(shape_list):
    raise ValueError(
        'Number of first dimensions must not be greater than the total '
        'dimensionality of input tensor: %d vs. %d.' %
        (num_first_dims, len(shape_list)))
  num_other_dims = len(shape_list) - num_first_dims
  multiples = list(first_dim_multiples) + [1] * num_other_dims
  return tf.tile(x, multiples=multiples)


def tile_last_dims(x, last_dim_multiples):
  """Tiles the last dimensions of a tensor.

  For example:
    x.shape.as_list() = [2, 3, 4].
    tile_last_dims(x, last_dim_multiples=[2, 3]).shape_as_list() == [2, 6, 12].

  Args:
    x: A tensor to tile.
    last_dim_multiples: A list of integers for the multiples to tile the last
      dimensions.

  Returns:
    A tiled tensor.

  Raises:
    ValueError: If length of `last_dim_multiples` is greater than the total
      dimensionality of `x`.
  """
  shape_list = x.shape.as_list()
  num_last_dims = len(last_dim_multiples)
  if num_last_dims > len(shape_list):
    raise ValueError(
        'Number of last dimensions must not be greater than the total '
        'dimensionality of input tensor: %d vs. %d.' %
        (num_last_dims, len(shape_list)))
  num_other_dims = len(shape_list) - num_last_dims
  multiples = [1] * num_other_dims + list(last_dim_multiples)
  return tf.tile(x, multiples=multiples)


def recursively_expand_dims(x, axes):
  """Recursively applies a sequence of axis expansion on a tensor.

  For example:
    x.shape.as_list() == [2, 3]
    recursively_expand_dims(x, axes=[-1, -1]).shape.as_list() == [2, 3, 1, 1].

  Args:
    x: A tensor to expand axes.
    axes: A list of integers for axes to expand in sequence.

  Returns:
    An expanded tensor.
  """
  for axis in axes:
    x = tf.expand_dims(x, axis)
  return x


def reshape_by_last_dims(x, last_dim_shape):
  """Reshapes a tensor by the last dimensions.

  Note that the last dimensions to reshape must have the same total number of
  elements as in the output shape. Other dimensions will not be changed:
    Input shape = [M_1, M_2, .., M_{d-k}, M_{d-k+1}, M_{d-k+2}, ..., M_d].
    Last_dim_shape = [N_1, N_2, ..., N_k].
    Output shape = [M_1, M_2, ..., M_{d-k}, N_1, N_2, ..., N_k].

  For example:
    x.shape.as_list() == [2, 4, 6].
    reshape_by_last_dims(x, [8, 3]).shape.as_list() == [2, 8, 3].

  Args:
    x: A tensor to reshape.
    last_dim_shape: A list of integers for the last dimensions to reshape to.

  Returns:
    A reshaped tensor.
  """
  new_shape = []
  for d in range(len(x.shape.as_list()) - len(last_dim_shape)):
    new_shape.append(tf.shape(x)[d])
  new_shape.extend(last_dim_shape)
  return tf.reshape(x, new_shape)


def reduce_weighted_mean(tensor, weights, axis=None, keepdims=False):
  """Reduces weighted means.

  Args:
    tensor: A tensor to reduce.
    weights: A tensor for weights. Must be non-negative and multiplicable to
      `tensor`. If None, this function falls back to tf.math.reduce_mean.
    axis: A list of integer for axes to reduce along.
    keepdims: A boolean for whether to keep the orignial dimension for results.

  Returns:
    A reduced tensor.
  """
  if weights is None:
    return tf.math.reduce_mean(tensor, axis=axis, keepdims=keepdims)
  return (tf.math.reduce_sum(tensor * weights, axis=axis, keepdims=keepdims) /
          tf.math.maximum(
              1e-12, tf.math.reduce_sum(weights, axis=axis, keepdims=keepdims)))


def sample_gaussians(means, stddevs, num_samples, seed=None):
  """Samples from multivariate Gaussian distributions.

  Uses reparameterization trick:
    epsilon ~ N(0, I),
    sample = epsilon * stddev + mean.

  Args:
    means: A tensor for Gaussin distribution means. Shape = [..., dim].
    stddevs: A tensor for Gaussian distribution standard deviations. Shape =
      [..., dim].
    num_samples: An integer for the number of samples to draw.
    seed: An integer for random seed.

  Returns:
    A tensor for samples. Shape = [..., num_samples, dim].
  """
  means = tile_last_dims(
      tf.expand_dims(means, axis=-2), last_dim_multiples=[num_samples, 1])
  stddevs = tile_last_dims(
      tf.expand_dims(stddevs, axis=-2), last_dim_multiples=[num_samples, 1])
  epsilons = tf.random.normal(tf.shape(means), seed=seed)
  return epsilons * stddevs + means


def compute_lower_percentile_means(x, axis, q=50):
  """Computes means of elements less or equal to q percentile along axes.

  Args:
    x: A tensor for input values. Shape = [..., dim_1, ..., dim_len(axis)].
    axis: An integer or a list of integers for percentile reduction axis.
    q: A scalar for percentile.

  Returns:
    A tensor for means of elements less or equal to the 1 percentiles. Shape =
      [...].
  """
  percentiles = tfp.stats.percentile(x, q=q, axis=axis, keep_dims=True)
  weights = tf.cast(x <= percentiles, dtype=tf.float32)
  return (tf.math.reduce_sum(x * weights, axis=axis) /
          tf.math.reduce_sum(weights, axis=axis))


def mix_pair_batch(lhs_pairs,
                   rhs_pairs,
                   axis,
                   sub_batch_ratios=(1.0, 1.0, 1.0, 1.0)):
  """Slices and mixes pair batches.

  Assumes batch axis is 0. An example for the batch mixing is:

  # Shape = [8, 2, 1].
  lhs_pairs = [
    [[1.0], [2.0]],
    [[1.1], [2.1]],
    [[3.0], [4.0]],
    [[3.1], [4.1]],
    [[5.0], [6.0]],
    [[5.1], [6.1]],
    [[7.0], [8.0]],
    [[7.1], [8.1]],
  ]
  rhs_pairs = [
    [[11.0], [12.0]],
    [[11.1], [12.1]],
    [[13.0], [14.0]],
    [[13.1], [14.1]],
    [[15.0], [16.0]],
    [[15.1], [16.1]],
    [[17.0], [18.0]],
    [[17.1], [18.1]],
  ]
  mixed_batch = mix_pair_batch(lhs_pairs, rhs_pairs, axis=1)

  mixed_batch == [
    [[1.0], [2.0]],
    [[1.1], [2.1]],
    [[13.0], [14.0]],
    [[13.1], [14.1]],
    [[5.0], [16.0]],
    [[5.1], [16.1]],
    [[17.0], [8.0]],
    [[17.1], [8.1]],
  ]

  Args:
    lhs_pairs: A tensor for LHS pair data. Shape = [batch_size, ..., 2, ...].
    rhs_pairs: A tensor for RHS pair data. Shape = [batch_size, ..., 2, ...].
    axis: An integer for the pair axis.
    sub_batch_ratios: A tuple of four floats for the ratios between each
      sub-batch. The first three sub-batch size will be calculated and rounded
      to closest integers, and the last sub-batch size will be the remaining.

  Returns:
    A tensor for the mixed pair batch.

  Raises:
    ValueError: Size of `sub_batch_ratios` is not 4.
  """
  if len(sub_batch_ratios) != 4:
    raise ValueError('Sub-batch ratios must be of size 4: %s.' %
                     str(sub_batch_ratios))
  total_ratios = np.sum(sub_batch_ratios)
  sub_batch_ratios = [x / total_ratios for x in sub_batch_ratios]
  batch_size = tf.shape(lhs_pairs)[0]
  float_batch_size = tf.cast(batch_size, dtype=tf.float32)
  sub_batch_sizes = [
      tf.cast(
          tf.math.round(float_batch_size * sub_batch_ratios[i]), dtype=tf.int32)
      for i in range(3)
  ]
  sub_batch_sizes.append(batch_size - sub_batch_sizes[0] - sub_batch_sizes[1] -
                         sub_batch_sizes[2])
  lhs_pairs_sub_batches = tf.split(
      lhs_pairs, num_or_size_splits=sub_batch_sizes)
  rhs_pairs_sub_batches = tf.split(
      rhs_pairs, num_or_size_splits=sub_batch_sizes)
  rhs_first_data, rhs_second_data = tf.unstack(rhs_pairs, axis=axis)
  lhs_first_data, lhs_second_data = tf.unstack(lhs_pairs, axis=axis)
  lhs_first_sub_batches = tf.split(
      lhs_first_data, num_or_size_splits=sub_batch_sizes)
  lhs_second_sub_batches = tf.split(
      lhs_second_data, num_or_size_splits=sub_batch_sizes)
  rhs_first_sub_batches = tf.split(
      rhs_first_data, num_or_size_splits=sub_batch_sizes)
  rhs_second_sub_batches = tf.split(
      rhs_second_data, num_or_size_splits=sub_batch_sizes)
  lhs_first_rhs_second_pairs = tf.stack(
      [lhs_first_sub_batches[2], rhs_second_sub_batches[2]], axis=axis)
  rhs_first_lhs_second_pairs = tf.stack(
      [rhs_first_sub_batches[3], lhs_second_sub_batches[3]], axis=axis)
  return tf.concat([
      lhs_pairs_sub_batches[0],
      rhs_pairs_sub_batches[1],
      lhs_first_rhs_second_pairs,
      rhs_first_lhs_second_pairs,
  ],
                   axis=0)


def shuffle_batches(tensors, seed=None):
  """Consistently shuffles tensors by the batch dimension.

  Args:
    tensors: A list of tensors to shuffle. They are assumed to have the same
      batch size.
    seed: An integer for random seed.

  Returns:
    A list of shuffled tensors.
  """
  if not tensors:
    raise ValueError('Tensor list is empty.')
  batch_indices = tf.range(tf.shape(tensors[0])[0])
  batch_indices = tf.random.shuffle(batch_indices, seed=seed)
  return [tf.gather(tensor, indices=batch_indices) for tensor in tensors]


def update_sub_tensor(x, indices, axis, update_func):
  """Applies `update_func` to sub-tensor.

  Only updates the specified sub-tensor. The rest of the tensor remains
  unchanged. Currently only supports integer indices, and does not support index
  tensors.

  Args:
    x: A tensor to update sub-tensor from.
    indices: A list of integers for slicing sub-tensor.
    axis: An integer for the axis to slice the sub-tensor along.
    update_func: A Python function handle for computing the updated sub-tensor.

  Returns:
    A tensor with only the specified sub-tensor updated.

  Raises:
    ValueError: If any element in `indices` is not an integer.
  """
  for i in indices:
    if not isinstance(i, int):
      raise ValueError('Only supports integer indices: `%s`.' % indices)

  sub_tensor = tf.gather(x, indices=indices, axis=axis)
  updated_sub_tensor = update_func(sub_tensor)
  dim = x.shape.as_list()[axis]
  complement_indices = [i for i in range(dim) if i not in indices]
  complement_tensor = tf.gather(x, indices=complement_indices, axis=axis)
  combined_tensor = tf.concat([updated_sub_tensor, complement_tensor],
                              axis=axis)
  combined_indices = indices + complement_indices
  combined_indices = [combined_indices.index(i) for i in range(dim)]
  return tf.gather(combined_tensor, indices=combined_indices, axis=axis)


def merge_dict(source_dict, target_dict):
  """Merges source dictionary into target dictionary without overriding.

  `Target_dict` will be updated in place.

  Args:
    source_dict: A dictionary for source elements.
    target_dict: A dictionary for target elements.

  Raises:
    ValueError: If `source_dict` and `target_dict` have key conflict.
  """
  for key, value in source_dict.items():
    if key in target_dict:
      raise ValueError('Key conflict: `%s`.' % key)
    target_dict[key] = value
