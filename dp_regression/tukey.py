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

"""Tukey mechanism for differentially private linear regression.

This file implements the Tukey mechanism for (epsilon, delta)-differentially
private linear regression.
"""

import numpy as np
import sklearn


def perturb_and_sort_matrix(input_matrix):
  """Perturbs and sorts input_matrix.

  Args:
    input_matrix: Matrix in which each row is a vector to be perturbed and
      sorted.

  Returns:
    Adds a small amount of noise to each entry in input_matrix and then sorts
    each row in increasing order.
  """
  d, m = input_matrix.shape
  perturbation_matrix = 1e-10 * np.random.rand(d, m)
  perturbed_matrix = input_matrix + perturbation_matrix
  return np.sort(perturbed_matrix, axis=1)


def log_measure_geq_all_depths(projections):
  """Computes log(volume) of region of at least depth, for all possible depths.

  Args:
    projections: Matrix where each row is a projection of the data sorted in
      increasing order.

  Returns:
    Array A where A[i] is the (natural) logarithm of the volume in projections
    of the region of depth > i.
  """
  max_depth = int(np.ceil(len(projections[0]) / 2))
  diff = np.flip(projections, axis=1) - projections
  return np.sum(np.log(diff[:, :max_depth]), axis=0)


def racing_sample(log_terms):
  """Numerically stable method for sampling from an exponential distribution.

  Args:
    log_terms: Array of terms of form log(coefficient) - (exponent term).

  Returns:
    A sample from the distribution over 0, 1, ..., len(log_terms) - 1 where
    integer k has probability proportional to exp(log_terms[k]). For details,
    see the "racing sampling" described in https://arxiv.org/abs/2201.12333.
  """
  return np.argmin(
      np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def restricted_racing_sample_depth(projections, epsilon, restricted_depth):
  """Executes epsilon-DP Tukey depth exponential mechanism on projections.

  Args:
    projections: Matrix where each row is a projection of the data sorted in
      increasing order.
    epsilon: The output will be (epsilon, delta)-DP, where delta is determined
      by the preceding call to distance_to_unsafety.
    restricted_depth: Sampling will be restricted to points in projections of
      depth at least restricted_depth.

  Returns:
    A sample from the epsilon-DP Tukey depth exponential mechanism on
    projections (line 3 in Algorithm 1 in https://arxiv.org/pdf/2106.13329.pdf).
  """
  projections = projections[:, (restricted_depth-1):-(restricted_depth-1)]
  atleast_volumes = np.exp(log_measure_geq_all_depths(projections))
  measure_exact_all = atleast_volumes
  measure_exact_all[:-1] = atleast_volumes[:-1] - atleast_volumes[1:]
  depths = np.arange(restricted_depth,
                     restricted_depth + len(atleast_volumes))
  log_terms = np.log(measure_exact_all) + epsilon * depths
  # add 1 because returned depth is 0-indexed
  return 1 + racing_sample(log_terms)


def distance_to_unsafety(log_volumes, epsilon, delta, t, k_low, k_high):
  """Returns Hamming distance lower bound computed by PTR check.

  Args:
    log_volumes: Array of logarithmic volumes of regions of different depths;
      log_volumes[i] = log(volume of region of depth > i).
    epsilon: The overall check is (epsilon, delta)-DP.
    delta: The overall check is (epsilon, delta)-DP.
    t: Fixed depth around which neighboring depth volumes are computed.
    k_low: Lower bound for neighborhood size k. First call of this function
      should use k_low=-1
    k_high: Upper bound for neighborhood size k.

  Returns:
    Hamming distance lower bound computed by PTR check.
  """
  k = int((k_low + k_high) / 2)
  log_vol_ytk = log_volumes[t-k-2]
  log_vols_ytk_gs = log_volumes[t+k+1:len(log_volumes)-1]
  log_epsilon_terms = (epsilon / 2) * np.arange(1, len(log_vols_ytk_gs) + 1)
  log_threshold = np.log(delta / (8 * np.exp(epsilon)))
  if np.min((log_vol_ytk - log_vols_ytk_gs) -
            log_epsilon_terms) <= log_threshold:
    if k_low >= k_high - 1:
      return k_high
    else:
      new_k_low = k_low + int((k_high - k_low) / 2)
      return distance_to_unsafety(log_volumes, epsilon, delta, t, new_k_low,
                                  k_high)
  if k_high > k_low + 1:
    new_k_high = k_high - int((k_high - k_low) / 2)
    return distance_to_unsafety(log_volumes, epsilon, delta, t, k_low,
                                new_k_high)
  return k_low


def log_measure_geq_all_dims(depth, projections):
  """Computes log(length) of region of at least depth, for each dimension.

  Args:
    depth: Desired depth in projections. Assumes
      1 <= depth < len(projections[0]) / 2.
    projections: Matrix where each row is a projection of the data sorted in
      increasing order.

  Returns:
    Array A where A[j] is the (natural) logarithm of the length in projections
    of the region of depth >= i in dimension j+1.
  """
  return np.log(projections[:, -depth] - projections[:, depth - 1])


def sample_geq_1d(depth, projection):
  """Samples a point of at least given depth from projection.

  Args:
    depth: Lower bound on depth of point to sample. Assumes
      1 <= depth <= len(proj) / 2.
    projection: Increasing array of 1-dimensional points.

  Returns:
    Point sampled uniformly at random from the region of at least the given
    depth in projection.
  """
  low = projection[depth-1]
  high = projection[-depth]
  return np.random.uniform(low, high)


def sample_exact_1d(depth, projection):
  """Samples a point of exactly given depth from projection.

  Args:
    depth: Depth of point to sample. Assumes
      1 <= depth <= len(proj) / 2.
    projection: Increasing array of 1-dimensional points.

  Returns:
    Point sampled uniformly at random from the region of given depth in
    projection.
  """
  left_low = projection[depth-1]
  left_high = projection[depth]
  right_low = projection[-(depth+1)]
  right_high = projection[-depth]
  measure_left = left_high - left_low
  measure_right = right_high - right_low
  if np.random.uniform() < measure_left / (measure_left + measure_right):
    return left_low + np.random.uniform() * measure_left
  else:
    return right_low + np.random.uniform() * measure_right


def sample_exact(depth, projections):
  """Samples a point of exactly given depth from projections.

  Args:
    depth: Minimum depth of point to sample. Assumes
      1 <= depth < len(proj) / 2.
    projections: Matrix where each row is a projection of the data sorted in
      increasing order.

  Returns:
    Point sampled uniformly at random from the region of exactly given depth in
    projections.
  """
  d, _ = projections.shape
  log_measures_greater_than_depth = log_measure_geq_all_dims(
      depth + 1, projections)
  log_measures_geq_depth = log_measure_geq_all_dims(depth, projections)
  # exact_lengths[j] = W_{j+1, depth} in the paper's notation
  log_exact_lengths = np.log(
      np.exp(log_measures_geq_depth) -
      np.exp(log_measures_greater_than_depth))
  # exp(log_volume_greater_than_depth_left[j]) = V_{<j+1, depth+1}, in the
  # paper's notation. log_volume_greater_than_depth_left[j] is the volume
  # measured along the first j dimensions of the region of depth greater than
  # the depth argument to this function
  log_volume_greater_than_depth_left = np.zeros(d)
  log_volume_greater_than_depth_left[1:] = np.cumsum(
      log_measures_greater_than_depth)[:-1]
  # exp(right_dims_geq_than_depth[j]) = V_{>j, depth}, in the paper's notation
  log_right_dims_geq_depth = np.zeros(d)
  log_right_dims_geq_depth[:-1] = (np.cumsum(
      log_measures_greater_than_depth[::-1])[::-1])[1:]
  log_volumes = log_exact_lengths + log_volume_greater_than_depth_left + log_right_dims_geq_depth
  sampled_volume_idx = racing_sample(log_volumes)
  # sampled point will be exactly depth in dimension sampled_volume_idx and
  # >= depth elsewhere
  sample = np.zeros(d)
  for j in range(sampled_volume_idx):
    sample[j] = sample_geq_1d(depth + 1, projections[j, :])
  sample[sampled_volume_idx] = sample_exact_1d(
      depth, projections[sampled_volume_idx, :])
  for j in range(sampled_volume_idx+1, d):
    sample[j] = sample_geq_1d(depth, projections[j, :])
  return sample


def multiple_regressions(features, labels, num_models):
  """Computes num_models regression models on random partition of given data.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    num_models: Number of models to train.

  Returns:
    (num_models x d) matrix of models, where features is (n x d).

  Raises:
    RuntimeError: [num_models] models requires [num_models * d] points, but
    given features only has [len(features)] points.
  """
  (n, d) = features.shape
  batch_size = int(n / num_models)
  if batch_size < d:
    raise RuntimeError(
        str(num_models) + " models requires " + str(num_models * d) +
        " points, but given features only has " + str(n) + " points.")
  # shuffle samples
  order = np.arange(n)
  np.random.shuffle(order)
  shuffled_features = features[order]
  shuffled_labels = labels[order]
  models = np.zeros((num_models, d))
  for i in range(num_models):
    shuffled_features_batch = shuffled_features[batch_size * i:batch_size * i +
                                                batch_size, :]
    shuffled_labels_batch = shuffled_labels[batch_size * i:batch_size * i +
                                            batch_size]
    model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(
        shuffled_features_batch, shuffled_labels_batch)
    models[i, :] = model.coef_
  return models


def dp_tukey(models, epsilon, delta):
  """Runs (epsilon, delta)-DP Tukey mechanism using models.

  Args:
    models: Feature vectors of non-private models. Assumes that each user
      contributes to a single model.
    epsilon: Computed model satisfies (epsilon, delta)-DP.
    delta: Computed model satisfies (epsilon, delta)-DP.

  Returns:
    Regression model computed using Tukey mechanism, or 0 if PTR fails.
  """
  projections = perturb_and_sort_matrix(models.T)
  max_depth = int(len(models) / 2)
  # compute log(volume_i)_{i=1}^max_depth where volume_i is the volume of
  # the region of depth >= i, according to projections
  log_volumes = log_measure_geq_all_depths(projections)
  t = int(max_depth / 2)
  # do ptr check
  split_epsilon = epsilon / 2
  distance = distance_to_unsafety(log_volumes, split_epsilon, delta, t, -1, t-1)
  threshold = np.log(1 / (2 * delta)) / split_epsilon
  if not distance + np.random.laplace(scale=split_epsilon) > threshold:
    print("ptr fail")
    return np.zeros(len(models[0]))
  # sample a depth using the restricted exponential mechanism
  depth = restricted_racing_sample_depth(projections, split_epsilon, t)
  # sample uniformly from the region of given depth
  return sample_exact(depth, projections[:, t:-t])
