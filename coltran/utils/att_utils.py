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

"""Some utils for attention layers."""

import functools
import itertools
import operator
import numpy as np
import tensorflow.compat.v2 as tf


def index_to_step(index, shape):
  """Compute step for a given nd index if we were enumerating to shape."""
  step = index[0]
  for i, s in enumerate(shape[1:]):
    step = step * s + index[i + 1]
  return step


def pad_to_multiple_nd(x, shape):
  """Pads x such that nd-shape axes are multiples of shape axes.

  Args:
    x: Tensor of shape [B] + nd_shape + [...].
    shape: Shape tuple of same length as nd_shape.

  Returns:
    x padded to make each axis in nd_shape divisible by the same shape axis.
  """
  x_shape = x.shape.as_list()
  num_feat_dim = len(x_shape) - len(shape) - 1
  if all(s for s in x_shape[1:len(shape) + 1]):
    pad_amount = np.mod(-np.asarray(x_shape[1:len(shape) + 1]), shape)
    paddings = [[0, 0]] + [[0, p] for p in pad_amount] + [[0, 0]] * num_feat_dim

    return tf.pad(x, paddings) if any(any(p) for p in paddings) else x
  else:
    # If shape is not fully defined.
    tf_shape = tf.shape(x)
    last = x_shape[-num_feat_dim:]
    paddings = [[0, -(x_shape[i + 1] or tf_shape[i + 1]) % s]
                for i, s in enumerate(shape)]
    paddings = [[0, 0]] + paddings + [[0, 0]] * num_feat_dim
    padded_x = tf.pad(x, paddings)
    padded_shape = padded_x.shape.as_list()
    padded_shape = padded_shape[:-1] + last
    return padded_x


def divide_nd_blocks(inputs, nd_block_size, collapse=False):
  """Divides input into non-overlapping n-dimensional blocks.

  Args:
    inputs: [B, D1, D2, ..., Dk, ...] tensor.
    nd_block_size: Shape tuple of length k.
    collapse: collapse.

  Returns:
    A [B, D1 // S1, D2 // S2, ..., Dk // Sk, S1 , S2 , ... , Sk, ...] tensor.
  """
  nd_block_size = list(nd_block_size)
  inputs = pad_to_multiple_nd(inputs, nd_block_size)

  shape = list(inputs.shape)
  for i, s in enumerate(shape):
    if s is None:
      shape[i] = tf.shape(inputs)[i]

  block_axes = shape[1:len(nd_block_size) + 1]
  num_blocks = [l // s for l, s in zip(block_axes, nd_block_size)]
  num_nd_axes = len(nd_block_size)
  num_feat_axes = len(shape) - num_nd_axes - 1
  features_shape = shape[-num_feat_axes:]

  # Reshape into [B, D1 // S1, S1, D2 // S2, S2, ..., Dk // Sk, Sk, ...].
  mid_shape = list(itertools.chain(*zip(num_blocks, nd_block_size)))
  cut_shape = shape[:1] + mid_shape + features_shape
  cut_inputs = tf.reshape(inputs, cut_shape)

  # Permute into [B, D1 // S1, D2 // S2, ..., Dk // Sk, S1, S2, ..., Sk, ...].
  num_mid_axes = num_nd_axes * 2
  num_feat_axes = len(shape) - num_nd_axes - 1
  mid_permute = itertools.chain(
      range(1, num_mid_axes, 2), range(2, num_mid_axes + 1, 2))
  post_permute = range(num_mid_axes + 1, num_mid_axes + num_feat_axes + 1)
  permutation = [0] + list(mid_permute) + list(post_permute)
  permuted_inputs = tf.transpose(cut_inputs, permutation)

  if not collapse:
    return permuted_inputs
  # Collapse to [B * D1 // S1 * D2 // S2 * ... * Dk // Sk, S1 * S2 * Sk, ...]
  block_length = functools.reduce(operator.mul, nd_block_size, 1)
  collapsed_inputs = tf.reshape(permuted_inputs, [-1, block_length] +
                                features_shape)

  return collapsed_inputs


def relative_attn_bias(rel_bias, num_heads, decode_step=None):
  """Computes attention bias based on relative positions.

  Content-based relative position attention bias was used in:
    https://arxiv.org/pdf/1803.02155.
  Non-content-based relative position attention bias was used in:
    https://arxiv.org/abs/1606.01933.

  Args:
    rel_bias: Relative bias variable of shape [num_heads, 2 * length].
    num_heads: Number of attention heads.
    decode_step: Optional decode step, used for slicing during decoding.

  Returns:
    A [..., length, num_heads, length] tensor with queries.
  """
  num_rel_pos = rel_bias.shape[-1]
  length = num_rel_pos // 2

  if tf.is_tensor(decode_step):
    # This is decoding so we need to select the current slice within rel_bias.
    # E.g.: len_k = 3, decode_step = 1
    # We have: rel_bias = [-2, -1, 0, 1, 2, 3]
    # We want: [-1, 0, 1]
    # We slice at len_k - decode_step - 1 = 1
    rel_bias = tf.reshape(rel_bias, [1, num_heads, num_rel_pos])
    start = ((length - 1) - decode_step)
    rel_bias = tf.slice(rel_bias, [0, 0, start], [1, num_heads, length])
    return rel_bias

  # Now we have to shift in order to compute relative biases.
  # Example: length = 3
  # Say we want:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
  # Start: [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
  # We linearize: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3]
  # We slice: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0]
  # We reshape: [[-2, -1, 0, 1, 2], [3, -2, -1, 0, 1], [2, 3, -2, -1, 0]]
  # We slice: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
  # Tadaaa!

  # [heads, len_q * num_rel_pos]
  rel_bias = tf.tile(rel_bias, [1, length])

  # [heads, len_q * (num_rel_pos - 1)]
  num_rel_pos -= 1
  rel_bias = rel_bias[Ellipsis, :length * num_rel_pos]

  # [heads, len_q, num_rel_pos - 1]
  # Now every row is shifted by 1 to the right.
  rel_bias = tf.reshape(rel_bias, [num_heads, length, num_rel_pos])

  # [heads, len_q, len_k]
  # Slice the overlapping elements from start.
  rel_bias = rel_bias[Ellipsis, num_rel_pos - length:]
  # [len_q, heads, len_k]
  rel_bias = tf.transpose(rel_bias, [1, 0, 2])

  return rel_bias
