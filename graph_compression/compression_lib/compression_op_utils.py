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

"""Util methods for compression_op library."""

from __future__ import division

import math
import numpy as np
import tensorflow.compat.v1 as tf


def compute_compressed_rank_from_matrix_shape(matrix_shape, rank_factor):
  """Compute rank for compression from input matrix shape and rank factor.

  Args:
    matrix_shape: tuple corresponding to the shape of the matrix that will be
      compressed.
    rank_factor: int corresponding to the amount to compress by. New rank will
      be (old_rank * 100 / rank_factor).

  Returns:
    The new rank. Note, if the computed rank is greater than the current rank,
    the current rank will be returned. This prevents "compressing" to a rank
    larger than the current rank.
  """
  rank = math.ceil(int(np.min(matrix_shape)) * 100 / rank_factor)
  # If matrix dimension is smaller than rank specified then adjust rank
  rank = np.min(matrix_shape + (rank,))
  return int(rank)


def flatten_last_dims(x, ndims=1):
  """Flatten the last `ndims` dimension of the input tensor.

  Args:
    x: a tf.Tensor.
    ndims: number of dimensions to flatten.

  Returns:
    x with last `ndims` dimensions flattened.
  """
  newshape = tf.concat([tf.shape(x)[:-ndims], [-1]], axis=0)
  return tf.reshape(x, shape=newshape)
