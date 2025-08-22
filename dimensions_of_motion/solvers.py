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

# -*- coding: utf-8 -*-
"""Solvers for projecting into column spaces."""

import tensorflow as tf


def project_svd(basis, vector, singular_threshold):
  """Project `vector` into the colspace of `basis' using SVD.

  Args:
    basis: a column space into which we will project, shape [B, N, M]
    vector: a vector to be projected, shape [B, N, 1]
    singular_threshold: singular values below this threshold are treated as 0

  Returns:
    projected_vector: the same shape as vector
    weights: [B, M]: weights so that basis*vector=projected_vector. If basis
                     is not full-rank, this is non-unique and no guarantees are
                     made about which set is returned.
  """
  s, u, v = tf.linalg.svd(basis)
  # The columns of U form an orthonormal basis and the matrix 'subspace'
  # indicates a subset of those which form an orthonormal basis for the column
  # space of M.
  subspace = tf.abs(s) >= singular_threshold
  subspace = tf.cast(subspace, tf.float32)

  projected_vector = tf.matmul(
      u,
      tf.matmul(
          tf.linalg.diag(subspace),
          tf.matmul(u, vector, transpose_a=True)))

  # We can find weights that give this projected vector by solving the normal
  # equations (A^TA)^(-1)A^Tx.

  # 1/s where s > singular_threshold, 0 elsewhere
  s_inverse = tf.math.divide_no_nan(subspace, s)
  weights = tf.matmul(
      v,
      tf.matmul(
          tf.linalg.diag(s_inverse),
          tf.matmul(u, vector, transpose_a=True)))
  return projected_vector, tf.squeeze(weights, -1)

