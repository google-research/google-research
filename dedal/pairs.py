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

"""Contains auxiliary functions to operator over "paired" tensors."""


from typing import Tuple

import tensorflow as tf


def pair_masks(mask_x, mask_y):
  """Combines a pair of 2D masks into a single 3D mask.

  Args:
    mask_x: A tf.Tensor<float>[batch, len_x] with binary entries.
    mask_y: A tf.Tensor<float>[batch, len_y] with binary entries.

  Returns:
    A tf.Tensor<float>[batch, len_x, len_y] with binary entries, defined as
      out[n][i][j] := mask_x[n][i] * mask_y[n][j].
  """
  mask1, mask2 = tf.cast(mask_x, tf.float32), tf.cast(mask_y, tf.float32)
  return tf.cast(tf.einsum('ij,ik->ijk', mask1, mask2), tf.bool)


def build(indices, *args):
  """Builds the pairs of whatever is passed as args for the given indices.

  Args:
    indices: a tf.Tensor<int32>[batch, 2]
    *args: a sequence of tf.Tensor[2 * batch, ...].

  Returns:
    A tuple of tf.Tensor[batch, 2, ...]
  """
  return tuple(tf.gather(arg, indices) for arg in args)


def consecutive_indices(batch):
  """Builds a batch of consecutive indices of size N from a batch of size 2N.

  Args:
    batch: tf.Tensor<float>[2N, ...].

  Returns:
    A tf.Tensor<int32>[N, 2] of consecutive indices.
  """
  batch_size = tf.shape(batch)[0]
  return tf.reshape(tf.range(batch_size), (-1, 2))


def roll_indices(indices, shift = 1):
  """Build a batch of non matching indices by shifting the batch of indices.

  Args:
    indices: a tf.Tensor<int32>[N, 2] of indices.
    shift: how much to shift the second column.

  Returns:
    A tf.Tensor<int32>[N, 2] of indices where the second columns has been
    rolled.
  """
  return tf.stack([indices[:, 0], tf.roll(indices[:, 1], shift=shift, axis=0)],
                  axis=1)


def square_distances(embs_1, embs_2):
  """Returns the matrix of square distances.

  Args:
    embs_1: tf.Tensor<float>[batch, len, dim].
    embs_2: tf.Tensor<float>[batch, len, dim].

  Returns:
    A tf.Tensor<float>[batch, len, len] containing the square distances.
  """
  gram_embs = tf.matmul(embs_1, embs_2, transpose_b=True)
  sq_norm_embs_1 = tf.linalg.norm(embs_1, axis=-1, keepdims=True)**2
  sq_norm_embs_2 = tf.linalg.norm(embs_2, axis=-1)**2
  return sq_norm_embs_1 + sq_norm_embs_2[:, tf.newaxis, :] - 2 * gram_embs
