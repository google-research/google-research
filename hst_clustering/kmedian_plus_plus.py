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

"""Implements kmedians++ algorithm."""

import numpy as np

from scipy import optimize
from scipy import spatial

from hst_clustering import dp_one_median as dp_med


def find_distances(data, centers):
  """Finds the distance from dataset to centers.

  For each data point in data find the distance to its closest center.

  Args:
    data: Two dimensional np array. One data point per row.
    centers: Two dimensional np array. One center per row.

  Returns:
    Numpy array where i-th entry is the distance of point i to centers.
  """
  distances = spatial.distance_matrix(data, centers)
  return np.min(distances, 1)


def initialize_kmed(data, k, weights=None):
  """Initializes k-medians using k-medians++.

  Args:
    data: Two dimensional np array with one row per data point. k : Number of
      centers to return.
    k: Number of centers.
    weights: Optional, weight to give every data point.

  Returns:
    K centers from the data using kmedian++ algorithm.
  """
  n_rows, _ = data.shape
  # pylint: disable=g-long-lambda
  center_selector = lambda probabilities: np.random.choice(
      range(n_rows), 1, p=probabilities
  )
  return _initialize_kmed(data, k, weights, center_selector)


def _initialize_kmed(data, k, weights, center_selector):
  """Initializes k-medians using k-medians++.

  Args:
    data: Two dimensional np array with one row per data point.
    k: Number of centers to return.
    weights: Optional, weight to give every data point.
    center_selector: Function that takes as input a set of probabilities
      and returns an index less than data.shape[0].

  Returns:
    K centers from the data using kmedian++ algorithm.
  """
  n_rows, _ = data.shape
  if weights is None:
    weights = np.ones(n_rows)
  probabilities = weights / np.sum(weights)

  ix = center_selector(probabilities)
  centers = np.array([data[ix, :]]).reshape(1, -1)
  for _ in range(1, k):
    distances = find_distances(data, centers)
    probabilities = distances.reshape(-1) * weights / np.sum(
        distances.reshape(-1) * weights)
    new_ix = center_selector(probabilities)
    centers = np.concatenate([centers, data[new_ix, :].reshape(1, -1)], 0)
  return centers


def objective(data, mu, weights=None):
  """K medians objective.

  Args:
    data: Dataset. Each row is a datapoint
    mu: Center to evaluate.
    weights: Optional weights to create a weighted average of the distances.

  Returns:
    Sum(w_i||data_i - mu||)
  """
  n, _ = data.shape
  if weights is None:
    weights = np.ones(n)
  distances = spatial.distance_matrix(data, mu.reshape(1, -1))
  return np.sum(distances.reshape(-1) * weights)


def optimize_obj(data, weights=None, max_num_points=None):
  """Finds the point that minimizes the kmedians objective.

  Args:
    data: Two dimensional array with one row per point.
    weights: Weight for each data point.
    max_num_points: Max number of points to be used in the objective. If not
      None it will randomly sample the data to this many points. Otherwise, it
      uses all points in the dataset.

  Returns:
    Point that minimizes the objective as a np array and the objective value.
  """
  data_shape = data.shape
  if max_num_points is None:
    max_num_points = data_shape[0]
  if data_shape[0] == 0:
    return np.random.uniform(0, 2, data_shape[1]), 0.0
  sampling_fraction = np.minimum(max_num_points / data_shape[0], 1.0)
  if sampling_fraction < 1.0:
    ix = np.random.binomial(1, sampling_fraction, data_shape[0]) == 1
    data = data[ix, :]
  x0 = np.mean(data, 0)
  sol = optimize.minimize(
      lambda c: objective(data, c, weights), x0, method="BFGS"
  )
  return sol.x, sol.fun


def _private_optimize_obj(
    data, weights=None, max_num_points=None, privacy_params=None
):
  """Privately optimizes the k median objective.

  Args:
    data: Data to find the optimal center.
    weights: Optional weights for each point in the dataset.
    max_num_points: Optional max number of points to use from the data.
    privacy_params: PrivacyPrams object.

  Returns:
    A center and the objective value of the function.
  """
  _, center = dp_med.get_private_kmed_center(
      data,
      privacy_params.lambd,
      privacy_params.epsilon,
      privacy_params.delta,
      privacy_params.gamma,
      max_num_points,
  )
  obj = objective(data, center, weights)
  return center, obj


def get_assignments(data, centers, weights=None):
  """Assigns all points in data to its nearest center.

  Args:
    data: Dataset as np array with one row per data point.
    centers: Two dimensional dataset with one row per center.
    weights: Weight for each point.

  Returns:
    Dictionary mapping cluster id to np array of points in the cluster.
    Dictionary mapping cluster id to weights of the points in the cluster.
    If weights is None then the dictionary maps cluster id to None.
  """
  k = centers.shape[0]
  assignment = np.argmin(spatial.distance_matrix(data, centers), 1)

  clusters = {i: data[assignment == i, :] for i in range(k)}
  weight_assignment = {}
  if weights is None:
    weight_assignment = {i: None for i in range(k)}
  else:
    weight_assignment = {i: weights[assignment == i] for i in range(k)}
  return clusters, weight_assignment


def lloyd_iter(
    data,
    centers,
    iters=10,
    weights=None,
    max_num_points=None,
    privacy_params=None,
):
  """LLoyd's algorithm for kmedians.

  Args:
    data: Dataset.
    centers: Initial centers.
    iters: Number of iterations.
    weights: Weights for each datapoint.
    max_num_points: Max number of pointsto use in the iteration.
    privacy_params: If passed, calculates the median in a private way.

  Returns:
    Centers after iters iterations of LLoyd.
    Objective function at the solution.
  """
  assigner = get_assignments
  optimizer = optimize_obj
  if privacy_params is not None:

    def optimizer_fn(data, weights, max_num_points):
      return _private_optimize_obj(
          data, weights, max_num_points, privacy_params
      )

    optimizer = optimizer_fn
  return _lloyd_iter(
      data, centers, iters, weights, assigner, optimizer, max_num_points
  )


def _lloyd_iter(
    data, centers, iters, weights, assigner, optimizer, max_num_points=None
):
  """LLoyd's algorithm for kmedians.

  Args:
    data: Dataset.
    centers: Initial centers.
    iters: Number of iterations.
    weights: Weights for each datapoint.
    assigner: Function that takes as input X, centers, weights and returns a
      dictionary one for cluster ids to points (as np array) the other one from
      cluster ids to weights.
    optimizer: Function that takes as input a dataset and weights and returns
      the minimizer of the kmedian objective and the objective.
    max_num_points: Maximum number of points in a cluster used to find a new
      center. If not None, it samples points in the cluster.

  Returns:
    Centers after iters iterations of LLoyd.
    Objective function at the solution.
  """
  new_centers = np.copy(centers)
  for i in range(iters):
    clusters, weight_assignment = assigner(data, new_centers, weights)
    objective_value = 0
    new_centers = []
    for j, cluster in clusters.items():
      new_center, val = optimizer(
          cluster, weight_assignment[j], max_num_points=max_num_points
      )
      objective_value += val
      new_centers.append(new_center.reshape(1, -1))
    new_centers = np.concatenate(new_centers, 0)
    print("Iteration %d obj: %f" % (i, objective_value))
  return new_centers, objective


def kmed_plus_plus(data, k, iters=10, weights=None):
  """Runs kmedians++.

  Args:
    data: Numpy dataset 1 row per example.
    k: Number of centers to return.
    iters: Number of lloyd iterations.
    weights: If passed, it weights each point by this.

  Returns:
    Centers of kmed++ as a numpy array and the objective. One row per center.
  """
  centers = initialize_kmed(data, k, weights)
  return lloyd_iter(data, centers, iters, weights)


def private_centers(data, norm, epsilon, delta):
  sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
  n, d = data.shape
  print("Points in cluster %d Noise added %f " % (n, norm / n * sigma))
  return np.mean(data, 0) + norm / n * np.random.normal(0, sigma, (d))


def private_center_recovery(raw_data, cluster_assignment, norm, epsilon, delta):
  """Recovers centers of raw data in a private way using the Gaussian mechanism.

  Args:
    raw_data: Numpy array with one row per data point.
    cluster_assignment: Numpy array mapping points to a cluster id.
      cluster_assignment[i] corresponds to the cluster id of raw_data[i,:].
    norm: Max norm of an element in raw_data.
    epsilon: Epsilon of differential privacy.
    delta: Delta of differential privacy.

  Returns:
    A collection of k centers found privately.
  """
  cluster_ids = np.unique(cluster_assignment)
  print(cluster_ids)
  _, d = raw_data.shape
  centers = []
  print("Max cluster id %d " + str(max(cluster_ids)))
  for _ in range(max(cluster_ids) + 1):
    centers.append(np.zeros(d))

  for key in cluster_ids:
    points = raw_data[key == cluster_assignment]
    if not points:
      centers[key] = np.random.uniform(-norm, norm)
    centers[key] = private_centers(points, norm, epsilon, delta)
  return np.vstack(centers)


def eval_projected_clustering(
    projected_data, data, centers, norm, epsilon, delta
):
  assignment = np.argmin(spatial.distance_matrix(projected_data, centers), 1)
  recovered_centers = private_center_recovery(
      data, assignment, norm, epsilon, delta
  )
  return np.sum(find_distances(data, recovered_centers))
