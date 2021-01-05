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

"""Distance utility functions."""

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from poem.core import data_utils


def compute_l2_distances(lhs, rhs, squared=False, keepdims=False):
  """Computes (optionally squared) L2 distances between points.

  Args:
    lhs: A tensor for LHS points. Shape = [..., point_dim].
    rhs: A tensor for RHS points. Shape = [..., point_dim].
    squared: A boolean for whether to compute squared L2 distance instead.
    keepdims: A boolean for whether to keep the reduced `point_dim` dimension
      (of length 1) in the result distance tensor.

  Returns:
    A tensor for L2 distances. Shape = [..., 1] if `keepdims` is True, or [...]
      otherwise.
  """
  squared_l2_distances = tf.math.reduce_sum(
      tf.math.subtract(lhs, rhs)**2, axis=-1, keepdims=keepdims)
  return squared_l2_distances if squared else tf.math.sqrt(squared_l2_distances)


def compute_sigmoid_matching_probabilities(inner_distances,
                                           a_initializer=None,
                                           b_initializer=None,
                                           smoothing=0.1,
                                           name='MatchingSigmoid'):
  """Computes sigmoid matching probabilities.

  We define sigmoid matching probability as:
    P(x_1, x_2) = (1 - s) * sigmoid(-a * d(x_1, x_2) + b) + s / 2
                = (1 - s) / (1 + exp(a * d(x_1, x_2) - b)) + s / 2,

  in which d(x_1, x_2) is inner distance between x_1 and x_2, and a and b are
  trainable parameters, with s being the smoothing constant.

  Args:
    inner_distances: A tensor for inner distances. Shape = [...].
    a_initializer: A function handle for initializer of `a` parameter. Use None
      for default initializer.
    b_initializer: A function handle for initializer of `b` parameter. Use None
      for default initializer.
    smoothing: A float for label smoothing constant.
    name: A string for the variable scope name.

  Returns:
    A tensor for matching probabilities. Shape = [...].
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    a = tf.get_variable(
        'a', shape=[], dtype=tf.float32, initializer=a_initializer)
    a = tf.nn.elu(a) + 1.0
    b = tf.get_variable(
        'b', shape=[], dtype=tf.float32, initializer=b_initializer)
  p = tf.math.sigmoid(-a * inner_distances + b)
  return (1.0 - smoothing) * p + smoothing / 2.0


def compute_sigmoid_matching_distances(inner_distances,
                                       a_initializer=None,
                                       b_initializer=None,
                                       smoothing=0.1,
                                       name='MatchingSigmoid'):
  """Computes sigmoid matching distances.

  Given sigmoid matching probability P(x_1, x_2), we define sigmoid matching
  distance as:
    d_m(x_1, x_2) = -log P(x_1, x_2)

  Note that setting smoothing to 0 may cause numerical instability.

  Args:
    inner_distances: A tensor for inner distances. Shape = [...].
    a_initializer: A function handle for initializer of `a` parameter. Use None
      for default initializer.
    b_initializer: A function handle for initializer of `b` parameter. Use None
      for default initializer.
    smoothing: A float for label smoothing constant.
    name: A string for the variable scope name.

  Returns:
    A tensor for matching distances. Shape = [...].
  """
  p = compute_sigmoid_matching_probabilities(
      inner_distances,
      a_initializer=a_initializer,
      b_initializer=b_initializer,
      smoothing=smoothing,
      name=name)
  return -tf.math.log(p)


def compute_all_pair_l2_distances(lhs, rhs, squared=False):
  """Computes all-pair (squared) L2 distances.

  Args:
    lhs: A tensor for LHS point groups. Shape = [..., num_lhs_points,
      point_dim].
    rhs: A tensor for RHS point groups. Shape = [..., num_rhs_points,
      point_dim].
    squared: A boolean for whether to use squared L2 distance instead.

  Returns:
    distances: A tensor for (squared) L2 distances. Shape = [...,
      num_lhs_points, num_rhs_points].
  """
  lhs_squared_norms = tf.math.reduce_sum(
      tf.math.square(lhs), axis=-1, keepdims=True)
  rhs_squared_norms = tf.expand_dims(
      tf.math.reduce_sum(tf.math.square(rhs), axis=-1), axis=-2)
  dot_products = tf.linalg.matmul(lhs, rhs, transpose_b=True)
  distances = -2.0 * dot_products + lhs_squared_norms + rhs_squared_norms

  if not squared:
    distances = tf.math.sqrt(tf.math.maximum(0.0, distances))

  return distances


def compute_corresponding_pair_l2_distances(lhs, rhs, squared=False):
  """Computes corresponding-pair (squared) L2 distances.

  Args:
    lhs: A tensor for LHS point groups. Shape = [..., num_points, point_dim].
    rhs: A tensor for RHS point groups. Shape = [..., num_points, point_dim].
    squared: A boolean for whether to use squared L2 distance instead.

  Returns:
    distances: A tensor for (squared) L2 distances. Shape = [..., num_points,
      1].
  """
  return compute_l2_distances(lhs, rhs, squared=squared, keepdims=True)


def compute_gaussian_likelihoods(
    means,
    stddevs,
    samples,
    l2_distance_computer=compute_all_pair_l2_distances,
    min_stddev=0.0,
    max_squared_mahalanobis_distance=0.0,
    smoothing=0.0):
  """Computes sample likelihoods with respect to Gaussian distributions.

  Args:
    means: A tensor for Gaussian means. Shape = [..., 1, sample_dim].
    stddevs: A tensor for Gaussian stddevs. Shape = [..., 1, sample_dim].
    samples: A tensor for samples. Shape = [..., num_samples, sample_dim].
    l2_distance_computer: A function handle for L2 distance computer to use.
    min_stddev: A float for minimum standard deviation to use. Ignored if
      non-positive.
    max_squared_mahalanobis_distance: A float for maximum inner squared
      mahalanobis distance to use. Larger distances will be clipped. Ignored if
      non-positive.
    smoothing: A float for label smoothing constant. Ignored if non-positive.

  Returns:
    A tensor for sample likelihoods. Shape = [..., num_samples].
  """
  if min_stddev > 0.0:
    stddevs = tf.math.maximum(min_stddev, stddevs)

  samples *= tf.math.reciprocal(stddevs)
  means *= tf.math.reciprocal(stddevs)
  squared_mahalanobis_distances = l2_distance_computer(
      samples, means, squared=True)

  if max_squared_mahalanobis_distance > 0.0:
    squared_mahalanobis_distances = tf.clip_by_value(
        squared_mahalanobis_distances,
        clip_value_min=0.0,
        clip_value_max=max_squared_mahalanobis_distance)

  chi2 = tfp.distributions.Chi2(
      df=means.shape.as_list()[-1], allow_nan_stats=False)
  p = 1.0 - chi2.cdf(squared_mahalanobis_distances)

  if smoothing > 0.0:
    p = (1.0 - smoothing) * p + smoothing / 2.0
  return p


def compute_distance_matrix(start_points,
                            end_points,
                            distance_fn,
                            start_point_masks=None,
                            end_point_masks=None):
  """Computes all-pair distance matrix.

  Note if either point mask tensor is specified, `distance_fn` must support a
  third argument as point masks. If both masks are specified, they will be
  multiplied. Otherwise if either is specified, it will be used for both points.

  Computes distance matrix as:
    [d(s_1, e_1),  d(s_1, e_2),  ...,  d(s_1, e_N)]
    [d(s_2, e_1),  d(s_2, e_2),  ...,  d(s_2, e_N)]
    [...,          ...,          ...,  ...        ]
    [d(s_M, e_1),  d(s_2, e_2),  ...,  d(s_2, e_N)]

  Args:
    start_points: A tensor for start points. Shape = [num_start_points, ...,
      point_dim].
    end_points: A tensor for end_points. Shape = [num_end_points, ...,
      point_dim].
    distance_fn: A function handle for computing distance matrix, which takes
      two matrix point tensors and a mask matrix tensor, and returns an
      element-wise distance matrix tensor.
    start_point_masks: A tensor for start point masks. Shape =
      [num_start_points, ...].
    end_point_masks: A tensor for end point masks. Shape = [num_end_points,
      ...].

  Returns:
    A tensor for distance matrix. Shape = [num_start_points, num_end_points,
      ...].
  """

  def expand_and_tile_axis_01(x, target_axis, target_dim):
    """Expands and tiles tensor along target axis 0 or 1."""
    if target_axis not in [0, 1]:
      raise ValueError('Only supports 0 or 1 as target axis: %s.' %
                       str(target_axis))
    x = tf.expand_dims(x, axis=target_axis)
    first_dim_multiples = [1, 1]
    first_dim_multiples[target_axis] = target_dim
    return data_utils.tile_first_dims(
        x, first_dim_multiples=first_dim_multiples)

  num_start_points = tf.shape(start_points)[0]
  num_end_points = tf.shape(end_points)[0]
  start_points = expand_and_tile_axis_01(
      start_points, target_axis=1, target_dim=num_end_points)
  end_points = expand_and_tile_axis_01(
      end_points, target_axis=0, target_dim=num_start_points)

  if start_point_masks is None and end_point_masks is None:
    return distance_fn(start_points, end_points)

  point_masks = None
  if start_point_masks is not None and end_point_masks is not None:
    start_point_masks = expand_and_tile_axis_01(
        start_point_masks, target_axis=1, target_dim=num_end_points)
    end_point_masks = expand_and_tile_axis_01(
        end_point_masks, target_axis=0, target_dim=num_start_points)
    point_masks = start_point_masks * end_point_masks
  elif start_point_masks is not None:
    start_point_masks = expand_and_tile_axis_01(
        start_point_masks, target_axis=1, target_dim=num_end_points)
    point_masks = start_point_masks
  else:  # End_point_masks is not None.
    end_point_masks = expand_and_tile_axis_01(
        end_point_masks, target_axis=0, target_dim=num_start_points)
    point_masks = end_point_masks

  return distance_fn(start_points, end_points, point_masks)


def compute_gaussian_kl_divergence(lhs_means,
                                   lhs_stddevs,
                                   rhs_means=0.0,
                                   rhs_stddevs=1.0):
  """Computes Kullback-Leibler divergence between two multivariate Gaussians.

  Only supports Gaussians with diagonal covariance matrix.

  Args:
    lhs_means: A tensor for LHS Gaussian means. Shape = [..., dim].
    lhs_stddevs: A tensor for LHS Gaussian standard deviations. Shape = [...,
      dim].
    rhs_means: A tensor or a float for LHS Gaussian means. Shape = [..., dim].
    rhs_stddevs: A tensor or a float for LHS Gaussian standard deviations. Shape
      = [..., dim].

  Returns:
    A tensor for KL divergence. Shape = [].
  """
  return 0.5 * tf.math.reduce_sum(
      (tf.math.square(lhs_stddevs) + tf.math.subtract(rhs_means, lhs_means)**2)
      / tf.math.maximum(1e-12, tf.math.square(rhs_stddevs)) - 1.0 +
      2.0 * tf.math.log(tf.math.maximum(1e-12, rhs_stddevs)) -
      2.0 * tf.math.log(tf.math.maximum(1e-12, lhs_stddevs)),
      axis=-1)
