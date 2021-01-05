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

"""Contains all functions to construct affinity graph for CNC and siamese nets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K


def squared_distance(input_x, input_y=None, weight=None):
  """Calculates the pairwise distance between points in X and Y.

  Args:
    input_x: n x d matrix
    input_y: m x d matrix
    weight: affinity n x m -- if provided, we normalize the distance

  Returns:
    n x m matrix of all pairwise squared Euclidean distances
  """
  if input_y is None:
    input_y = input_x
  sum_dimensions = list(range(2, K.ndim(input_x) + 1))
  input_x = K.expand_dims(input_x, axis=1)
  if weight is not None:
    # if weight provided, we normalize input_x and input_y by weight
    d_diag = K.expand_dims(K.sqrt(K.sum(weight, axis=1)), axis=1)
    input_x /= d_diag
    input_y /= d_diag
  squared_difference = K.square(input_x - input_y)
  distance = K.sum(squared_difference, axis=sum_dimensions)
  return distance


def knn_affinity(input_x,
                 n_nbrs,
                 scale=None,
                 scale_nbr=None,
                 local_scale=None,
                 verbose=False):
  """Calculates Gaussian affinity matrix.

  Calculates the symmetrized Gaussian affinity matrix with k1 nonzero
  affinities for each point, scaled by
  1) a provided scale,
  2) the median distance of the k2-th neighbor of each point in X, or
  3) a covariance matrix S where S_ii is the distance of the k2-th
  neighbor of each point i, and S_ij = 0 for all i != j
  Here, k1 = n_nbrs, k2 = scale_nbr

  Args:
    input_x: input dataset of size n
    n_nbrs: k1
    scale: provided scale
    scale_nbr: k2, used if scale not provided
    local_scale: if True, then we use the aforementioned option 3), else we
      use option 2)
    verbose: extra printouts

  Returns:
    n x n affinity matrix
  """
  if isinstance(n_nbrs, np.float):
    n_nbrs = int(n_nbrs)
  elif isinstance(n_nbrs,
                  tf.Variable) and n_nbrs.dtype.as_numpy_dtype != np.int32:
    n_nbrs = tf.cast(n_nbrs, np.int32)
  # get squared distance
  dist_x = squared_distance(input_x)
  # calculate the top k losest neighbors
  nn = tf.nn.top_k(-dist_x, n_nbrs, sorted=True)

  vals = nn[0]
  # apply scale
  if scale is None:
    # if scale not provided, use local scale
    if scale_nbr is None:
      scale_nbr = 0
    else:
      assert scale_nbr > 0 and scale_nbr <= n_nbrs
    if local_scale:
      scale = -nn[0][:, scale_nbr - 1]
      scale = tf.reshape(scale, [-1, 1])
      scale = tf.tile(scale, [1, n_nbrs])
      scale = tf.reshape(scale, [-1, 1])
      vals = tf.reshape(vals, [-1, 1])
      if verbose:
        vals = tf.Print(vals, [tf.shape(vals), tf.shape(scale)],
                        'vals, scale shape')
      vals = vals / (2 * scale)
      vals = tf.reshape(vals, [-1, n_nbrs])
    else:

      def get_median(scales, m):
        with tf.device('/cpu:0'):
          scales = tf.nn.top_k(scales, m)[0]
        scale = scales[m - 1]
        return scale, scales

      scales = -vals[:, scale_nbr - 1]
      const = tf.shape(input_x)[0] // 2
      scale, scales = get_median(scales, const)
      vals = vals / (2 * scale)
  else:
    # otherwise, use provided value for global scale
    vals = vals / (2 * scale**2)

  # get the affinity
  aff_vals = tf.exp(vals)
  # flatten this into a single vector of values to shove in a sparse matrix
  aff_vals = tf.reshape(aff_vals, [-1])
  # get the matrix of indices corresponding to each rank
  # with 1 in the first column and k in the kth column
  nn_ind = nn[1]
  # get the j index for the sparse matrix
  j_index = tf.reshape(nn_ind, [-1, 1])
  # the i index is just sequential to the j matrix
  i_index = tf.range(tf.shape(nn_ind)[0])
  i_index = tf.reshape(i_index, [-1, 1])
  i_index = tf.tile(i_index, [1, tf.shape(nn_ind)[1]])
  i_index = tf.reshape(i_index, [-1, 1])
  # concatenate the indices to build the sparse matrix
  indices = tf.concat((i_index, j_index), axis=1)
  # assemble the sparse weight matrix
  weight_mat = tf.SparseTensor(
      indices=tf.cast(indices, dtype='int64'),
      values=aff_vals,
      dense_shape=tf.cast(tf.shape(dist_x), dtype='int64'))
  # fix the ordering of the indices
  weight_mat = tf.sparse_reorder(weight_mat)
  # convert to dense tensor
  weight_mat = tf.sparse_tensor_to_dense(weight_mat)
  # symmetrize
  weight_mat = (weight_mat + tf.transpose(weight_mat)) / 2.0

  return weight_mat


def full_affinity(input_x, scale):
  """Calculates the symmetrized full Gaussian affinity matrix, scaled by a provided scale.

  Args:
    input_x: input dataset of size n x d
    scale: provided scale

  Returns:
    n x n affinity matrix
  """
  sigma = K.variable(scale)
  dist_x = squared_distance(input_x)
  sigma_squared = K.expand_dims(K.pow(sigma, 2), -1)
  weight_mat = K.exp(-dist_x / (2 * sigma_squared))
  return weight_mat


def get_contrastive_loss(m_neg=1, m_pos=.2):
  """Contrastive loss from Hadsell-et-al.'06.

  http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.

  Args:
    m_neg: negativeness.
    m_pos: possitiveness.

  Returns:
    Contrastive loss
  """

  def contrastive_loss(y_true, y_pred):
    return K.mean(y_true * K.square(K.maximum(y_pred - m_pos, 0)) +
                  (1 - y_true) * K.square(K.maximum(m_neg - y_pred, 0)))

  return contrastive_loss


def euclidean_distance(vects):
  """Computes the euclidean distances between vects[0] and vects[1]."""
  x, y = vects
  return K.sqrt(
      K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
  """Provides the output shape of the above computation."""
  s_1, _ = shapes
  return (s_1[0], 1)
