# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Matrix decomposition methods."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf


def _strict_sign(x):
  if x < 0:
    return -1
  else:
    return 1


_np_strict_sign = np.vectorize(_strict_sign)


def np_simhash_decompose(matrix, inner_dimension, seed=0):
  """Approximately decompose matrix as a product R S D.

  Args:
    matrix: the matrix to be decomposed, given as a Numpy array.
    inner_dimension: the number of rows in S.
    seed: a seed for the pseudorandom matrix R.

  Returns:
    Matrices (Numpy arrays) R, S, and D, where:
    R is iid normal distributed with inner_dimension columns.
    S is a +/-1 sign matrix, and
    D is a diagonal matrix,
  """
  rows, _ = np.shape(matrix)
  np.random.seed(seed=seed)
  r = np.random.normal(size=(rows, inner_dimension))
  s = _np_strict_sign(np.dot(np.transpose(r), matrix))
  rs_column_norms = np.apply_along_axis(np.linalg.norm, 0, np.dot(r, s))
  matrix_column_norms = np.apply_along_axis(np.linalg.norm, 0, matrix)
  d = np.diagflat(np.divide(matrix_column_norms, rs_column_norms))
  return r, s, d


def tf_simhash_decompose(matrix, inner_dimension, seed=0):
  """Approximately decompose matrix as a product R S D.

  Args:
    matrix: the matrix to be decomposed, given as a tensorflow matrix.
    inner_dimension: the number of rows in S.
    seed: a seed for the pseudorandom matrix R.

  Returns:
    Tensorflow matrices R, S, and D, where:
    R is iid normal distributed with inner_dimension columns.
    S is a +/-1 sign matrix, and
    D is a diagonal matrix,
  """
  rows, _ = matrix.get_shape().as_list()
  np.random.seed(seed=seed)
  r = tf.convert_to_tensor(
      np.random.normal(size=(rows, inner_dimension)), dtype=tf.float32)
  s_with_zeros = tf.math.sign(tf.linalg.matmul(r, matrix, transpose_a=True))
  s = tf.where(
      tf.math.equal(s_with_zeros, tf.constant(0.)),
      tf.ones(tf.shape(s_with_zeros)), s_with_zeros)
  rs_column_norms = tf.norm(tf.matmul(r, s), axis=0)
  matrix_column_norms = tf.norm(matrix, axis=0)
  d = tf.linalg.tensor_diag(
      tf.math.divide(matrix_column_norms, rs_column_norms))
  return r, s, d
