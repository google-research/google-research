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

"""Utilities for transforming Tensors."""

import string
from typing import Optional, Text, Union

import numpy as np
import tensorflow as tf


def flatten_dims(tensor: tf.Tensor,
                 first_dim: Optional[int] = 0,
                 last_dim: Optional[int] = -1,
                 name: Optional[Text] = None) -> tf.Tensor:
  """Flattens the given span of dimensions in `tensor`.

  Args:
    tensor: [..., first_dim_size, ...middle_dims..., last_dim_size, ...] shaped
      Tensor.
    first_dim: The first dimension to flatten (inclusive). Must be a valid index
      for the rank of `tensor`. Default is 0.
    last_dim: The last dimension to flatten (inclusive). Must be a valid index
      for the rank of `tensor`. Default is -1.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., flattened_dim_size, ...] where
    flattened_dim_size = first_dim_size * ...middle_dims... * last_dim_size.
  """
  with tf.name_scope(name or 'flatten_dims'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if first_dim < 0:
      first_dim += rank
    if first_dim < 0 or first_dim >= rank:
      raise ValueError('`first_dim` out of bounds for `tensor` rank.')
    if last_dim < 0:
      last_dim += rank
    if last_dim < 0 or last_dim >= rank:
      raise ValueError('`last_dim` out of bounds for `tensor` rank.')
    if first_dim > last_dim:
      raise ValueError('`first_dim` must not be larger than `last_dim`.')

    # Try to calculate static flattened dim size if all input sizes to flatten
    # are statically known. Otherwise, just use -1.
    flat_dims_shape = tensor.shape[first_dim:(last_dim + 1)].as_list()
    flattened_dim_size = 1
    for size in flat_dims_shape:
      if size is None:
        flattened_dim_size = -1
        break
      flattened_dim_size *= size

    old_shape = tf.shape(tensor)
    output_shape = tf.concat([
        old_shape[:first_dim], [flattened_dim_size], old_shape[(last_dim + 1):]
    ], 0)
    return tf.reshape(tensor, output_shape)


def gather_by_one_hot(params: tf.Tensor,
                      indices: tf.Tensor,
                      name: Optional[Text] = None) -> tf.Tensor:
  """Performs a gather operation using tf.one_hot multiplication.

  This is intended for TPU friendliness but comes with additional complexity
  costs. In particular, the materialized one-hot tensor has
  `lookup_size * indices.shape.num_elements()` elements.
  The time complexity is higher by a factor of `lookup_size` also.

  Unlike `tf.gather`, the axis to gather along is always the first one from
  `params`.

  Args:
    params: <float32>[lookup_size, ...] Tensor of rank >= 1 to gather values
      from.
    indices: <int>[...] Tensor of ids to index into `params`. Any ids outside
      the range [0, lookup_size) will translate to 0 values in the output.
    name: A name for the operation (optional).

  Returns:
    [indices.shape, params.shape[1:]] Tensor.
  """
  with tf.name_scope(name or 'gather_by_one_hot'):
    params = tf.convert_to_tensor(params)
    indices = tf.convert_to_tensor(indices)

    lookup_size = tf.shape(params)[0]
    flat_indices = tf.reshape(indices, [-1])
    one_hot_matrix = tf.one_hot(flat_indices, lookup_size, dtype=params.dtype)
    flat_result = einsum_wrap_ellipses('ij,j...->i...', one_hot_matrix, params)
    output_shape = tf.concat([tf.shape(indices), tf.shape(params)[1:]], 0)
    return tf.reshape(flat_result, output_shape)


def batch_gather_by_one_hot(params: tf.Tensor,
                            indices: tf.Tensor,
                            batch_dims: Optional[int] = None,
                            name: Optional[Text] = None) -> tf.Tensor:
  """Performs a batched version of gather using tf.one_hot multiplication.

  The first `batch_dims` dimensions of `params` and `indices` must match in
  shape.

  This is intended for TPU friendliness but comes with additional complexity
  costs. In particular, the materialized one-hot tensor has
  `lookup_size * indices.shape.num_elements()` elements.
  The time complexity is higher by a factor of `lookup_size` also.

  Args:
    params: <float32>[...some_batch_dims, lookup_size, ...] Tensor of values to
      gather from.
    indices: <int>[...some_batch_dims, ...index_dims...] Tensor of ids to index
      into `params`. Any values outside the range [0, lookup_size) will
      translate to 0 values in the output.
    batch_dims: Number of batched dimensions. Must be positive. Defaults to
      len(indices.shape) - 1.
    name: A name for the operation (optional).

  Returns:
    [indices.shape, params.shape[(batch_dims+1):]] Tensor.
  """
  # We rename `batch_dims` to `num_batch_dims` since it refers to a single
  # integer rather than a list of the dimensions themselves. The argument
  # name is kept to match `tf.gather`.
  num_batch_dims = batch_dims
  del batch_dims

  with tf.name_scope(name or 'batch_gather_by_one_hot'):
    params = tf.convert_to_tensor(params)
    indices = tf.convert_to_tensor(indices)

    if num_batch_dims is None:
      num_batch_dims = len(indices.shape) - 1
    if num_batch_dims <= 0:
      raise ValueError('`num_batch_dims` must be positive.')
    if len(params.shape) <= num_batch_dims:
      raise ValueError('`params` has too few dimensions.')
    if len(indices.shape) < num_batch_dims:
      raise ValueError('`indices` has too few dimensions.')
    if not params.shape[:num_batch_dims].is_compatible_with(
        indices.shape[:num_batch_dims]):
      raise ValueError('`params` and `indices` must have compatible batch '
                       'dimensions.')

    lookup_size = tf.shape(params)[num_batch_dims]

    # Flatten all "index_dims" in `indices` into a single dimension.
    flat_indices_shape = tf.concat([tf.shape(indices)[:num_batch_dims], [-1]],
                                   0)
    flat_indices = tf.reshape(indices, flat_indices_shape)
    one_hot_matrices = tf.one_hot(flat_indices, lookup_size, dtype=params.dtype)

    # Flatten all `params` dims after the "lookup_size" dimension. (If there
    # aren't any, then expand a final dimension.)
    flat_params_shape = tf.concat(
        [tf.shape(params)[:(num_batch_dims + 1)], [-1]], 0)
    flat_params = tf.reshape(params, flat_params_shape)

    flat_result = tf.matmul(one_hot_matrices, flat_params)
    output_shape = tf.concat(
        [tf.shape(indices),
         tf.shape(params)[(num_batch_dims + 1):]], 0)
    return tf.reshape(flat_result, output_shape)


def pad_to_multiple(tensor: tf.Tensor,
                    factor: Union[int, tf.Tensor],
                    axis: int,
                    mode: Optional[Text] = 'CONSTANT',
                    constant_values=0,
                    name: Optional[Text] = None) -> tf.Tensor:
  """Pads `tensor` on a given `axis` to be a multiple of `factor`.

  Padding will be concatenated to the end of the axis only, not the beginning.
  If the length along `axis` is already a multiple of `factor`, this is
  effectively a no-op.

  Args:
    tensor: A Tensor with rank >= 1 to pad.
    factor: Positive integer factor to pad for. If a Tensor, must be a scalar
      int.
    axis: A valid axis in `tensor` to pad.
    mode: The padding mode to use according to `tf.pad`. Defaults to 'CONSTANT'.
    constant_values: For 'CONSTANT' mode, the scalar pad value to use within
      `tf.pad`. Defaults to 0. Must be same type as `tensor`.
    name: A name for the operation (optional).

  Returns:
    The padded Tensor result.
  """
  with tf.name_scope(name or 'pad_to_multiple'):
    tensor = tf.convert_to_tensor(tensor)

    if isinstance(factor, int) and factor < 1:
      raise ValueError('`factor` must be positive.')
    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis < 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    axis_len = get_shape_list(tensor)[axis]
    pad_len = -axis_len % factor
    paddings = pad_len * tf.one_hot([-1, axis], rank, axis=0, dtype=tf.int32)
    return tf.pad(
        tensor=tensor,
        paddings=paddings,
        mode=mode,
        constant_values=constant_values)


def split_into_blocks(tensor: tf.Tensor,
                      block_len: int,
                      axis: int,
                      pad_value=0,
                      name: Optional[Text] = None) -> tf.Tensor:
  """Splits a tensor into blocks along the given `axis`.

  If the axis length isn't a multiple of `block_len`, it'll be padded via
  `pad_to_multiple` first.

  Args:
    tensor: Tensor of shape [..., axis_len, ...].
    block_len: Positive integer length of each block.
    axis: A valid axis in `tensor` to split along.
    pad_value: The scalar pad value to use. Defaults to 0. Must be same type as
      `tensor`.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., num_blocks, block_len, ...], where
    num_blocks = ceiling(axis_len / block_len).
  """
  with tf.name_scope(name or 'split_into_blocks'):
    tensor = tf.convert_to_tensor(tensor)

    if block_len < 1:
      raise ValueError('`block_len` must be positive.')
    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis < 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    padded_tensor = pad_to_multiple(
        tensor, factor=block_len, axis=axis, constant_values=pad_value)
    padded_len = get_shape_list(padded_tensor)[axis]
    num_blocks = padded_len // block_len
    output_shape = tf.concat([
        tf.shape(tensor)[:axis], [num_blocks, block_len],
        tf.shape(tensor)[(axis + 1):]
    ], 0)
    return tf.reshape(padded_tensor, output_shape)


def concat_3_blocks(blocked_seq: tf.Tensor,
                    name: Optional[Text] = None) -> tf.Tensor:
  """Concatenates 3 consecutive blocks for each input block.

  This is meant to be called on a blocked sequence as returned by
  `split_into_blocks` for example. This function augments each block with its
  adjacent left and right blocks so that every token from the original block
  can access all other tokens `block_len` away from it. The first and last input
  blocks will have 0-padded blocks to their left and right, respectively.

  Args:
    blocked_seq: [batch_size, num_blocks, block_len, ...] shaped Tensor.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape [batch_size, num_blocks, 3*block_len, ...].
  """
  with tf.name_scope(name or 'concat_3_blocks'):
    blocked_seq = tf.convert_to_tensor(blocked_seq)

    num_blocks = tf.shape(blocked_seq)[1]

    paddings = tf.one_hot([1, 1],
                          blocked_seq.shape.rank,
                          axis=0,
                          dtype=tf.int32)

    # [batch_size, num_blocks + 2, block_len, ...]
    padded_blocked_seq = tf.pad(blocked_seq, paddings)

    blocks_list = []
    for i in range(3):
      blocks_list.append(padded_blocked_seq[:, i:(i + num_blocks), ...])
    return tf.concat(blocks_list, 2)


def shift_elements_right(tensor: tf.Tensor,
                         axis: int = -1,
                         amount: int = 1,
                         pad_value=0,
                         name: Optional[Text] = None) -> tf.Tensor:
  """Shifts elements right (towards higher indices) along the given `axis`.

  This changes an input like
  [5, 4, 3, 2, 1]
  into the following (for amount=1):
  [0, 5, 4, 3, 2]

  New elements resulting from the shift are populated using `pad_value`.

  Args:
    tensor: Tensor with rank at least 1.
    axis: A valid axis in `tensor` to shift elements along.
    amount: Integer number of positions to shift element. Use negative numbers
      to shift left instead of right.
    pad_value: The scalar pad value to use. Defaults to 0. Must be the same type
      as `tensor`.
    name: A name for the operation (optional).

  Returns:
    Shifted tensor with the same shape as the input `tensor`.
  """
  with tf.name_scope(name or 'shift_elements_right'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis < 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    if amount == 0:
      return tensor

    paddings = abs(amount) * tf.one_hot(
        [axis, -1] if amount > 0 else [-1, axis], rank, axis=0, dtype=tf.int32)

    # [..., axis_len + abs(amount), ...]
    padded_tensor = tf.pad(tensor, paddings, constant_values=pad_value)

    if amount > 0:
      slice_begin = tf.zeros([rank], dtype=tf.int32)
    else:
      slice_begin = abs(amount) * tf.one_hot(axis, rank, dtype=tf.int32)

    return tf.slice(padded_tensor, begin=slice_begin, size=tf.shape(tensor))


def skew_elements_right(tensor: tf.Tensor,
                        axis: int,
                        pad_value=0,
                        name: Optional[Text] = None) -> tf.Tensor:
  """Skews successive elements right along the given `axis`.

  This changes an input like
  [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ]
  into the following:
  [
    [1, 2, 3, 0, 0],
    [0, 4, 5, 6, 0],
    [0, 0, 7, 8, 9]
  ]

  Args:
    tensor: Tensor of shape [..., num_rows, axis_len, ...].
    axis: A valid axis in `tensor` to skew along. It must not be the first axis
      in `tensor`.
    pad_value: The scalar pad value to use. Defaults to 0. Must be the same type
      as `tensor`.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., num_rows, axis_len + num_rows - 1, ...].
  """
  with tf.name_scope(name or 'skew_elements_right'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    num_rows = get_shape_list(tensor)[axis - 1]
    axis_len = get_shape_list(tensor)[axis]

    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis <= 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    output_len = axis_len + num_rows - 1

    paddings = num_rows * tf.one_hot([-1, axis], rank, axis=0, dtype=tf.int32)

    # [..., num_rows, axis_len + num_rows, ...]
    padded_tensor = tf.pad(tensor, paddings, constant_values=pad_value)

    # [..., num_rows * (axis_len + num_rows), ...]
    flat_tensor = flatten_dims(padded_tensor, first_dim=axis - 1, last_dim=axis)

    padded_tensor2 = pad_to_multiple(
        flat_tensor,
        factor=output_len,
        axis=axis - 1,
        constant_values=pad_value)

    # [..., num_rows + 1, output_len, ...]
    new_shape = tf.concat([
        tf.shape(tensor)[:(axis - 1)], [num_rows + 1, output_len],
        tf.shape(tensor)[(axis + 1):]
    ], 0)
    reshaped_tensor = tf.reshape(padded_tensor2, new_shape)

    # [..., num_rows, output_len, ...]
    output_shape = new_shape - tf.one_hot(axis - 1, depth=rank, dtype=tf.int32)
    return tf.slice(
        reshaped_tensor, begin=tf.zeros_like(output_shape), size=output_shape)


def unskew_elements_right(tensor: tf.Tensor,
                          axis: int,
                          name: Optional[Text] = None) -> tf.Tensor:
  """Unskews elements that were "skewed right" along the given `axis`.

  This operation is the inverse of `skew_elements_right`. It changes an input
  like
  [
    [1, 2, 3, 0, 0],
    [0, 4, 5, 6, 0],
    [0, 0, 7, 8, 9]
  ]
  into the following:
  [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ]

  Args:
    tensor: Tensor of shape [..., num_rows, axis_len, ...], where `axis_len`
      must be at least `num_rows`.
    axis: A valid axis in `tensor` to unskew along. It must not be the first
      axis in `tensor`.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., num_rows, axis_len - num_rows + 1, ...]
  """
  with tf.name_scope(name or 'unskew_elements_right'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    num_rows = tensor.shape.as_list()[axis - 1]
    axis_len = tensor.shape.as_list()[axis]

    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis <= 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')
    if num_rows is None:
      raise ValueError('Static size `num_rows` must be known.')
    if axis_len is None:
      raise ValueError('Static size `axis_len` must be known.')
    if axis_len < num_rows:
      raise ValueError('`axis_len` ({}) is less than `num_rows` ({}).'.format(
          axis_len, num_rows))

    output_len = axis_len - num_rows + 1

    # [..., num_rows * axis_len, ...]
    flat_tensor = flatten_dims(tensor, first_dim=axis - 1, last_dim=axis)

    padded_tensor = pad_to_multiple(
        flat_tensor, factor=axis_len + 1, axis=axis - 1)

    # [..., num_rows, axis_len + 1, ...]
    unskewing_shape = tf.concat([
        tf.shape(tensor)[:axis], [axis_len + 1],
        tf.shape(tensor)[(axis + 1):]
    ], 0)
    reshaped_tensor = tf.reshape(padded_tensor, unskewing_shape)

    # [..., num_rows, output_len, ...]
    output_shape = tf.concat(
        [tf.shape(tensor)[:axis], [output_len],
         tf.shape(tensor)[(axis + 1):]], 0)
    return tf.slice(
        reshaped_tensor, begin=tf.zeros_like(output_shape), size=output_shape)


def einsum_wrap_ellipses(equation: Text, *inputs, **kwargs) -> tf.Tensor:
  """Wrapper over `tf.einsum` that rewrites ellipses for efficiency.

  `tf.einsum` equations with ellipses (e.g. '...ab,abc->...ac') seem to
  result in slow backward pass computations on TPU. As an optimization, this
  wrapper replaces the ellipses with explicit letters unused by the equation
  (e.g. 'ABab,abc->ABac' in the previous example) before calling `tf.einsum`.

  This wrapper is more restrictive than `tf.einsum` in that it requires all
  input tensor ranks to be statically known, and '...' must represent the same
  number of dimensions in each input. It will not broadcast a mismatch in
  number of dimensions automatically.

  The result should be the same as calling `tf.einsum` directly, and equations
  without ellipses are passed to `tf.einsum` untouched.

  Args:
    equation: a `str` describing the contraction, in the same format as
      `tf.einsum`.
    *inputs: the inputs to contract (each one a `Tensor`), whose shapes should
      be consistent with `equation`.
    **kwargs: Additional keyword arguments to pass to `tf.einsum`.

  Returns:
    The result of calling `tf.einsum`.

  Raises:
    ValueError: if required input tensor ranks aren't statically known or usage
      of '...' implies different numbers of dimensions for different inputs.
  """
  if '...' not in equation:
    return tf.einsum(equation, *inputs, **kwargs)

  if '->' in equation:
    equation_inputs, equation_output = equation.split('->')
  else:
    equation_inputs = equation
    equation_output = None

  equation_inputs = equation_inputs.split(',')

  # Determine how many dimensions to replace '...' with.
  num_dims = None
  for i, input_str in enumerate(equation_inputs):
    if '...' not in input_str:
      continue
    rank = inputs[i].shape.rank
    if not isinstance(rank, int):
      raise ValueError(
          'Unknown static rank for input tensor at position {}.'.format(i))
    input_str = input_str.replace('...', '')
    input_str = input_str.replace(' ', '')
    if num_dims is None:
      num_dims = rank - len(input_str)
      if num_dims < 0:
        raise ValueError(
            'Not enough dimensions for input tensor at position {}.'.format(i))
    elif num_dims != rank - len(input_str):
      raise ValueError(
          'Mismatch for implied ellipses number of dimensions: {} vs. {}.'
          .format(num_dims, rank - len(input_str)))

  # Find letters to use in place of '...'
  used_letters = set() if equation_output is None else set(equation_output)
  for input_str in equation_inputs:
    used_letters.update(input_str)
  unused_letters = sorted(set(string.ascii_letters) - used_letters)
  ellipsis_letters = ''.join(unused_letters[:num_dims])

  new_equation_inputs = []
  for input_str in equation_inputs:
    new_equation_inputs.append(input_str.replace('...', ellipsis_letters))

  new_equation = ','.join(new_equation_inputs)
  if equation_output is not None:
    new_equation += '->{}'.format(
        equation_output.replace('...', ellipsis_letters))

  return tf.einsum(new_equation, *inputs, **kwargs)


# The following functions are from the original BERT code:


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    # Tensor.name is not supported in Eager mode.
    if tf.executing_eagerly():
      name = 'get_shape_list'
    else:
      name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, int):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        'For the tensor `%s` in scope `%s`, the actual rank '
        '`%d` (shape = %s) is not equal to the expected rank `%s`' %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that's not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, str):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == 'linear':
    return None
  elif act == 'relu':
    return tf.nn.relu
  elif act == 'gelu':
    return gelu
  elif act == 'tanh':
    return tf.tanh
  else:
    raise ValueError('Unsupported activation: %s' % act)
