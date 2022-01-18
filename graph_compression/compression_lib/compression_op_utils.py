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

"""Util methods and enums for compression_op library."""

from __future__ import division

import enum
import math
import numpy as np
import tensorflow.compat.v1 as tf


class CompressionOptions(enum.IntEnum):
  """Options for specifying the method of matrix compression."""
  NO_MATRIX_COMPRESSION = 0
  LOWRANK_MATRIX_COMPRESSION = 1
  SIMHASH_MATRIX_COMPRESSION = 2
  DL_MATRIX_COMPRESSION = 3
  KMEANS_MATRIX_COMPRESSION = 4
  KMEANS_AND_PRUNING_MATRIX_COMPRESSION = 8
  INPUTOUTPUT_COMPRESSION = 9
  BLOCK_COMPRESSION = 10
  MIXED_BLOCK_COMPRESSION = 12


class UpdateOptions(enum.IntEnum):
  """Options for implementing alpha update logic.

  update_option: integer
        indicates how the update logic is being run. More specifically:
        0: TF_UPDATE - run the update logic in TF; needed when using GPU/TPU
        1: PYTHON_UPDATE - run the update logic in regular
                           python as opposed to TF.
        2: TF_AND_PYTHON_UPDATE - run the update logic in TF
                                  and in regular python.
        3: NO_UPDATE - no alpha update as alpha not used
                       in some compression options.

  """
  TF_UPDATE = 0
  PYTHON_UPDATE = 1
  TF_AND_PYTHON_UPDATE = 2
  NO_UPDATE = 3


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
