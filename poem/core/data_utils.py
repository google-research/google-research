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


def unflatten_first_dim(x, shape_to_unflatten):
  """Unflattens the first dimension of a tensor.

  For example:
    x.shape.as_list() == [6, 2].
    unflatten_first_dim(x, [2, 3]).shape.as_list() == [2, 3, 2].

  Args:
    x: A tensor to unflatten.
    shape_to_unflatten: A list of integers to reshape the first dimension of `x`
      into.

  Returns:
    A unflattened tensor.
  """
  new_shape = list(shape_to_unflatten) + x.shape.as_list()[1:]
  return tf.reshape(x, new_shape)


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


def get_shape_by_last_dims(x, num_last_dims):
  """Gets tensor shape by the last dimensions.

  For example:
    x.shape.as_list() == [1, 2, 3, 4, 5]
    get_shape_by_last_dims(x, num_last_dims=2) == [4, 5].

  Args:
    x: A tensor to get shape of.
    num_last_dims: An integer for the number of last dimensions to get shape of.

  Returns:
    A list for tensor shape.
  """
  shape = tf.shape(x)
  output_shape = []
  for i in range(x.shape.ndims - num_last_dims, x.shape.ndims):
    output_shape.append(shape[i])
  return output_shape


def get_shape_by_first_dims(x, num_last_dims):
  """Gets tensor shape by the first dimensions.

  For example:
    x.shape.as_list() == [1, 2, 3, 4, 5]
    get_shape_by_first_dims(x, num_last_dims=2) == [1, 2, 3].

  Args:
    x: A tensor to get shape of.
    num_last_dims: An integer for the number of last dimensions not to get shape
      of.

  Returns:
    A list for tensor shape.
  """
  shape = tf.shape(x)
  output_shape = []
  for i in range(0, x.shape.ndims - num_last_dims):
    output_shape.append(shape[i])
  return output_shape


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
  for i in range(len(x.shape.as_list()) - len(last_dim_shape)):
    d = x.shape.as_list()[i]
    new_shape.append(-1 if d is None else d)
  new_shape.extend(last_dim_shape)
  return tf.reshape(x, new_shape)


def swap_axes(x, lhs_axis, rhs_axis):
  """Permutes a tensor by swapping two axes.

  Args:
    x: A tensor to permute.
    lhs_axis: An integer for one of the axes to swap.
    rhs_axis: An integer for one of the axes to swap.

  Returns:
    A permuted tensor.
  """
  permutation = list(range(x.shape.ndims))
  permutation[lhs_axis], permutation[rhs_axis] = (permutation[rhs_axis],
                                                  permutation[lhs_axis])
  return tf.transpose(x, permutation)


def move_axis(x, input_axis, output_axis):
  """Permutes a tensor such that an axis is moved to the destination axis.

  Example:
    x.shape.as_list() == [1, 2, 3, 4, 5].
    x = move_axis(x, input_axis=1, output_axis=-2)
    x.shape.as_list() == [1, 3, 4, 2, 5].

  Args:
    x: A tensor to permute.
    input_axis: An integer for the axis to move.
    output_axis: An integer for the destination to move the input axis to.

  Returns:
    A permuted tensor.
  """
  permutation = list(range(x.shape.ndims))
  if output_axis < 0:
    output_axis += len(permutation)
  axis_index = permutation.pop(input_axis)
  permutation.insert(output_axis, axis_index)
  return tf.transpose(x, permutation)


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
  percentiles = tfp.stats.percentile(x, q=q, axis=axis, keepdims=True)
  weights = tf.cast(x <= percentiles, dtype=tf.float32)
  return (tf.math.reduce_sum(x * weights, axis=axis) /
          tf.math.reduce_sum(weights, axis=axis))


def mix_batch(lhs_batches, rhs_batches, axis, assignment=None,
              keep_lhs_prob=0.5, seed=None):
  """Mixes batches.

  A pair of tensors from the same location in each list are assumed to have the
  same shape.

  Example:
    # Shape = [4, 3, 2, 1].
    lhs_batches[0] = [[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                      [[[2.0], [2.1]], [[2.2], [2.3]], [[2.4], [2.5]]],
                      [[[3.0], [3.1]], [[3.2], [3.3]], [[3.4], [3.5]]],
                      [[[4.0], [4.1]], [[4.2], [4.3]], [[4.4], [4.5]]]]
    rhs_batches[0] = [[[[11.0], [11.1]], [[11.2], [11.3]], [[11.4], [11.5]]],
                      [[[12.0], [12.1]], [[12.2], [12.3]], [[12.4], [12.5]]],
                      [[[13.0], [13.1]], [[13.2], [13.3]], [[13.4], [13.5]]],
                      [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]]

    # Shape = [4, 3, 2, 2, 1].
    lhs_batches[1] = [[[[[1.0], [10.0]], [[1.1], [10.1]]],
                       [[[1.2], [10.2]], [[1.3], [10.3]]],
                       [[[1.4], [10.4]], [[1.5], [10.5]]]],
                      [[[[2.0], [20.0]], [[2.1], [20.1]]],
                       [[[2.2], [20.2]], [[2.3], [20.3]]],
                       [[[2.4], [20.4]], [[2.5], [20.5]]]],
                      [[[[3.0], [30.0]], [[3.1], [30.1]]],
                       [[[3.2], [30.2]], [[3.3], [30.3]]],
                       [[[3.4], [30.4]], [[3.5], [30.5]]]],
                      [[[[4.0], [40.0]], [[4.1], [40.1]]],
                       [[[4.2], [40.2]], [[4.3], [40.3]]],
                       [[[4.4], [40.4]], [[4.5], [40.5]]]]]
    rhs_batches[1] = [[[[[11.0], [110.0]], [[11.1], [110.1]]],
                       [[[11.2], [110.2]], [[11.3], [110.3]]],
                       [[[11.4], [110.4]], [[11.5], [110.5]]]],
                      [[[[12.0], [120.0]], [[12.1], [120.1]]],
                       [[[12.2], [120.2]], [[12.3], [120.3]]],
                       [[[12.4], [120.4]], [[12.5], [120.5]]]],
                      [[[[13.0], [130.0]], [[13.1], [130.1]]],
                       [[[13.2], [130.2]], [[13.3], [130.3]]],
                       [[[13.4], [130.4]], [[13.5], [130.5]]]],
                      [[[[14.0], [140.0]], [[14.1], [140.1]]],
                       [[[14.2], [140.2]], [[14.3], [140.3]]],
                       [[[14.4], [140.4]], [[14.5], [140.5]]]]]

    # Shape = [4, 1, 2].
    assignment = [[[True, True]], [[True, False]],
                  [[False, True]], [[False, False]]]
    axis = 2
    -->
    # Shape = [4, 3, 2, 1].
    mixed_batches[0] = [[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                        [[[2.0], [12.1]], [[2.2], [12.3]], [[2.4], [12.5]]],
                        [[[13.0], [3.1]], [[13.2], [3.3]], [[13.4], [3.5]]],
                        [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]]

    # Shape = [4, 3, 2, 2, 1].
    mixed_batches[1] = [[[[[1.0], [10.0]], [[1.1], [10.1]]],
                         [[[1.2], [10.2]], [[1.3], [10.3]]],
                         [[[1.4], [10.4]], [[1.5], [10.5]]]],
                        [[[[2.0], [20.0]], [[12.1], [120.1]]],
                         [[[2.2], [20.2]], [[12.3], [120.3]]],
                         [[[2.4], [20.4]], [[12.5], [120.5]]]],
                        [[[[13.0], [130.0]], [[3.1], [30.1]]],
                         [[[13.2], [130.2]], [[3.3], [30.3]]],
                         [[[13.4], [130.4]], [[3.5], [30.5]]]],
                        [[[[14.0], [140.0]], [[14.1], [140.1]]],
                         [[[14.2], [140.2]], [[14.3], [140.3]]],
                         [[[14.4], [140.4]], [[14.5], [140.5]]]]]

  Args:
    lhs_batches: A list of tensors for LHS batches. Each tensor shape =
      [batch_size, ..., num_instances, ...].
    rhs_batches: A list of tensors for RHS batches. Each tensor shape =
      [batch_size, ..., num_instances, ...].
    axis: An integer for the mixing axis (the `num_instances` dimension).
    assignment: A tensor for assignment indicator matrix. Shape = [batch_size,
      ..., num_instances]. A True/False value indicates element from the LHS/RHS
      tensor will be kept at the corresponding location. For the "idle
      dimensions" between the batch dimension (0) and the mixing axis dimension,
      size 1 can be used to take advantage of broadcasting. If None, A uniformly
      random assignment matrix will be created.
    keep_lhs_prob: A float indicates the probability to randomly keep
      lhs_batches along axis. This is only useful when `assignment` is None.
    seed: An integer for random seed.

  Returns:
    mixed_batches: A list of tensors for mixed batches. Each tensor shape =
      [batch_size, ..., num_instances, ...].

  Raises:
    ValueError: If `lhs_batches` and `rhs_batches` have different sizes.
    ValueError: If `lhs_batches` or `rhs_batches` is empty.
    ValueError: If `axis` is out of range or incompatible with `assignment`.

  """
  if len(lhs_batches) != len(rhs_batches):
    raise ValueError(
        '`Lhs_batches` and `rhs_batches` size disagree: %d vs. %d.' %
        (len(lhs_batches), len(rhs_batches)))
  if not lhs_batches:
    raise ValueError('Tensor lists are empty.')

  def get_random_assignment(batch_shape):
    """Gets batch-compatible assignment."""
    assignment_shape = [1] * (axis + 1)
    assignment_shape[0] = batch_shape[0]
    assignment_shape[axis] = batch_shape[axis]
    assignment = tf.random.uniform(
        assignment_shape, minval=0.0, maxval=1.0, seed=seed)
    assignment = tf.math.greater_equal(assignment, keep_lhs_prob)
    return assignment

  if assignment is None:
    assignment = get_random_assignment(tf.shape(lhs_batches[0]))
  else:
    assignment_rank = len(assignment.shape.as_list())
    if assignment_rank != axis + 1:
      raise ValueError('`Assignment` and `axis` are incompatible: %d vs. %d.' %
                       (assignment_rank, axis))

  mixed_batches = []
  for i, batch_pair in enumerate(zip(lhs_batches, rhs_batches)):
    lhs_batch, rhs_batch = batch_pair
    batch_rank = len(lhs_batch.shape.as_list())
    if axis < 0 or batch_rank <= axis:
      raise ValueError('Axis out of range for the %d-th tensor: %d.' %
                       (i, axis))
    if batch_rank != len(rhs_batch.shape.as_list()):
      raise ValueError(
          'The %d-th LHS/RHS tensor have different ranks: %d vs. %d.' %
          (i, batch_rank, len(rhs_batch.shape.as_list())))

    assignment_rank = axis + 1
    if len(lhs_batch.shape.as_list()) > assignment_rank:
      batch_assignment = recursively_expand_dims(
          assignment, axes=[-1] * (batch_rank - assignment_rank))
    else:
      batch_assignment = assignment

    mixed_batches.append(tf.where(batch_assignment, lhs_batch, rhs_batch))

  return mixed_batches


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
