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

"""Utility functions used in fair clustering algorithms."""

import collections
import math
import random
from typing import List, Sequence, Set
import numpy as np
import sklearn.cluster


def ReadData(file_path):
  """Read the data from the file.

  Args:
    file_path: path to the input file in tsv format.

  Returns:
    The dataset as a np.array of points each as np.array vector.
  """
  with open(file_path, "r") as f:
    dataset = []
    for line in f:
      x = [float(x) for x in line.split("\t")]
      dataset.append(x)
  return np.array(dataset)


def DistanceToCenters(
    x, centers, p
):
  """Distance of a point to nearest center elevanted to p-th power.

  Args:
    x: the point.
    centers: the centers.
    p: power.

  Returns:
    The distance of the point to the nearest center to the p-th power.
  """
  min_cost = math.inf
  for c in centers:
    assert len(c) == len(x)
    cost_p = np.linalg.norm(x - c) ** p
    if cost_p < min_cost:
      min_cost = cost_p
  return min_cost


def FurthestPointPosition(
    dataset, centers
):
  """Returns the position of the furthest point in the dataset from the centers.

  Args:
    dataset: the dataset.
    centers: the centers.

  Returns:
    The furthest point position.
  """

  max_cost_position = -1
  max_cost = -1
  for pos, x in enumerate(dataset):
    d = DistanceToCenters(x, centers, 1)
    if d > max_cost:
      max_cost = d
      max_cost_position = pos
  assert max_cost_position >= 0
  return max_cost_position


def KMeansCost(dataset, centers):
  """Returns the k-means cost a solution.

  Args:
    dataset: the dataset.
    centers: the centers.

  Returns:
    The kmeans cost of the solution.
  """
  tot = 0.0
  for x in dataset:
    tot += DistanceToCenters(x, centers, 2)
  return tot


def MaxFairnessCost(
    dataset,
    centers,
    dist_threshold_vec,
):
  """Computes the max bound ratio on the dataset for a given solution.

  Args:
    dataset: the dataset.
    centers: the centers.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.

  Returns:
    The max ratio of the distance of a point to the closest center over the
    threshold.
  """
  tot = 0.0
  for i, x in enumerate(dataset):
    d = 1.0 * DistanceToCenters(x, centers, 1) / dist_threshold_vec[i]
    if d > tot:
      tot = d
  return tot


def ComputeDistanceThreshold(
    dataset,
    sampled_points,
    rank_sampled,
    multiplier,
):
  """Computes a target distance for the individual fairness requirement.

  In order to allow the efficient definition of a the fairness distance bound
  for each point we do not compute all pairs distances of points.
  Instead we use a sample of points. For each point p we define the threshold
  d(p) of the maximum distance that is allowed for a center near p to be the
  distance of the rank_sampled-th point closest to be among
  sampled_points sampled points times multiplier.

  Args:
    dataset: the dataset.
    sampled_points: number of points sampled.
    rank_sampled: rank of the distance to the sampled points used in the
      definition of the threshold.
    multiplier: multiplier used.

  Returns:
    The max ratio of the distance of a point to the closest center over the
    threshold.
  """
  ret = np.zeros(len(dataset))
  # Set the seeds to ensure multiple runs use the same thresholds
  random.seed(100)
  sample = random.sample(list(dataset), sampled_points)
  # reset the seed to time
  random.seed(None)
  for i, x in enumerate(dataset):
    distances = [np.linalg.norm(x - s) for s in sample]
    distances.sort()
    ret[i] = multiplier * distances[rank_sampled - 1]
  return ret


# Lloyds improvement algorithm
def IsFeasibleSolution(
    dataset,
    anchor_points_pos,
    candidate_centers_vec,
    dist_threshold_vec,
):
  """Check if candidate centers set is feasible.

  Args:
    dataset: the dataset.
    anchor_points_pos: position of the archor points.
    candidate_centers_vec: vector of candidate centers.
    dist_threshold_vec: distance thresholds.

  Returns:
    If the solution is feasible.
  """
  for s in anchor_points_pos:
    if (
        DistanceToCenters(dataset[s], candidate_centers_vec, 1)
        > dist_threshold_vec[s]
    ):
      return False
  return True


def Mean(dataset, positions):
  """Average the points in 'positions' in the dataset.

  Args:
    dataset: the dataset.
    positions: position in dataset of the points to average.

  Returns:
    Average of the points.
  """
  assert positions
  mean = np.zeros(len(dataset[0]))
  for i in positions:
    mean += dataset[i]
  mean /= len(positions)
  return mean


def LloydImprovementStepOneCluster(
    dataset,
    anchor_points_pos,
    curr_centers_vec,
    dist_threshold_vec,
    cluster_position,
    cluster_points_pos,
    approx_error = 0.01,
):
  """Improve the current center respecting feasibility of the solution.

  Given a cluster of points and a center centers_vec[cluster_position] is the
  center that will be updated.  The current centers must be a list of np.array.

  Args:
    dataset: the set of points.
    anchor_points_pos: the positions in dataset for the anchor points.
    curr_centers_vec: the current centers as a list of np.array vectors.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    cluster_position: the cluster being improved.
    cluster_points_pos: the points in the cluster.
    approx_error: approximation error tollerated in the binary search.

  Returns:
    An improved center.
  """

  def _IsValidSwap(vec_in):
    new_centers_vec = curr_centers_vec[:]
    new_centers_vec[cluster_position] = vec_in
    return IsFeasibleSolution(
        dataset, anchor_points_pos, new_centers_vec, dist_threshold_vec
    )

  def _Interpolate(curr_vec, new_vec, mult_new_vec):
    return curr_vec + (new_vec - curr_vec) * mult_new_vec

  assert IsFeasibleSolution(
      dataset, anchor_points_pos, curr_centers_vec, dist_threshold_vec
  )
  curr_center_vec = np.array(curr_centers_vec[cluster_position])
  mean = Mean(dataset, cluster_points_pos)

  if _IsValidSwap(mean):
    return mean
  highest_valid_mult = 0.0
  lowest_invalid_mult = 1.0
  while highest_valid_mult - lowest_invalid_mult >= approx_error:
    m = (lowest_invalid_mult + highest_valid_mult) / 2
    if _IsValidSwap(_Interpolate(curr_center_vec, mean, m)):
      highest_valid_mult = m
    else:
      lowest_invalid_mult = m
  return _Interpolate(curr_center_vec, mean, highest_valid_mult)


def LloydImprovement(
    dataset,
    anchor_points_pos,
    inital_centers_vec,
    dist_threshold_vec,
    num_iter = 20,
):
  """Runs the LloydImprovement algorithm respecting feasibility.

    Given the current centers improves the solution respecting the feasibility.

  Args:
    dataset: the set of points.
    anchor_points_pos: the positions in dataset for the anchor points.
    inital_centers_vec: the current centers.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    num_iter: number of iterations for the algorithm.

  Returns:
    An improved solution.
  """

  def _ClusterAssignment(pos_point, curr_centers):
    pos_center = 0
    min_cost = math.inf
    for i, c in enumerate(curr_centers):
      cost_p = np.linalg.norm(dataset[pos_point] - c)
      if cost_p < min_cost:
        min_cost = cost_p
        pos_center = i
    return pos_center

  curr_center_vec = [np.array(x) for x in inital_centers_vec]

  for _ in range(num_iter):
    cluster_elements = collections.defaultdict(list)
    for i in range(len(dataset)):
      cluster_elements[_ClusterAssignment(i, curr_center_vec)].append(i)
    for cluster_position in range(len(curr_center_vec)):
      if not cluster_elements[cluster_position]:
        continue
      curr_center_vec[cluster_position] = LloydImprovementStepOneCluster(
          dataset,
          anchor_points_pos,
          curr_center_vec,
          dist_threshold_vec,
          cluster_position,
          cluster_elements[cluster_position],
      )
  return curr_center_vec


# Bookkeeping class for local search
class TopTwoClosestToCenters:
  """Bookkeeping class used in local search.

  The class stores and updates efficiently the 2 closest centers for each point.
  """

  def __init__(self, dataset, centers_ids):
    """Constructor.

    Args:
      dataset: the dataset.
      centers_ids: the positions of the centers.
    """
    assert len(dataset) > 2
    assert len(centers_ids) >= 2

    self.dataset = dataset
    # all these fields use the position of the center in dataset not the center.
    self.centers = set(centers_ids)  # id of the centers
    self.center_to_min_dist_cluster = collections.defaultdict(set)
    # mapping from center pos to list of pos of min distance points
    self.center_to_second_dist_cluster = collections.defaultdict(set)
    # mapping from center pos to list of pos of second min distance squared
    # points.
    self.point_to_min_dist_center_and_distance = {}
    # mapping of points to min distance center pos, and distance.
    self.point_to_second_dist_center_and_distance = {}
    # mapping of points to second min distance center pos, and distance squared.
    for point_pos, _ in enumerate(dataset):
      self.InitializeDatastructureForPoint(point_pos)

  def InitializeDatastructureForPoint(self, point_pos):
    """Initialize the datastructure for a point."""
    if point_pos in self.point_to_min_dist_center_and_distance:
      del self.point_to_min_dist_center_and_distance[point_pos]
    if point_pos in self.point_to_second_dist_center_and_distance:
      del self.point_to_second_dist_center_and_distance[point_pos]
    for center_pos in self.centers:
      self.ProposeAsCenter(point_pos, center_pos)

  def ProposeAsCenter(self, pos_point, pos_center_to_add):
    """Updates the datastructure proposing a point as a new center.

    Args:
      pos_point: the position of the point.
      pos_center_to_add: the position of the center to be added.
    """
    d = (
        np.linalg.norm(
            self.dataset[pos_point] - self.dataset[pos_center_to_add]
        )
        ** 2
    )
    # never initialized point
    if pos_point not in self.point_to_min_dist_center_and_distance:
      assert pos_point not in self.point_to_second_dist_center_and_distance
      self.point_to_min_dist_center_and_distance[pos_point] = (
          pos_center_to_add,
          d,
      )
      self.center_to_min_dist_cluster[pos_center_to_add].add(pos_point)
      return
    if (
        self.point_to_min_dist_center_and_distance[pos_point][0]
        == pos_center_to_add
    ):
      return

    if d < self.point_to_min_dist_center_and_distance[pos_point][1]:
      # New first center. Move first to second.
      old_first_center = self.point_to_min_dist_center_and_distance[pos_point][
          0
      ]
      self.center_to_min_dist_cluster[old_first_center].remove(pos_point)

      if pos_point in self.point_to_second_dist_center_and_distance:
        self.center_to_second_dist_cluster[
            self.point_to_second_dist_center_and_distance[pos_point][0]
        ].remove(pos_point)
      self.point_to_second_dist_center_and_distance[pos_point] = (
          self.point_to_min_dist_center_and_distance[pos_point]
      )
      self.center_to_second_dist_cluster[old_first_center].add(pos_point)

      self.point_to_min_dist_center_and_distance[pos_point] = (
          pos_center_to_add,
          d,
      )
      self.center_to_min_dist_cluster[pos_center_to_add].add(pos_point)
    else:  # not first
      # not initialized second.
      if pos_point not in self.point_to_second_dist_center_and_distance:
        self.point_to_second_dist_center_and_distance[pos_point] = (
            pos_center_to_add,
            d,
        )
        self.center_to_second_dist_cluster[pos_center_to_add].add(pos_point)
        return
      if (
          self.point_to_second_dist_center_and_distance[pos_point][0]
          == pos_center_to_add
      ):
        return

      if d < self.point_to_second_dist_center_and_distance[pos_point][1]:
        self.center_to_second_dist_cluster[
            self.point_to_second_dist_center_and_distance[pos_point][0]
        ].remove(pos_point)
        self.point_to_second_dist_center_and_distance[pos_point] = (
            pos_center_to_add,
            d,
        )
        self.center_to_second_dist_cluster[pos_center_to_add].add(pos_point)

  def CostAfterSwap(
      self, pos_center_to_remove, pos_center_to_add
  ):
    """Computes the cost of a proposed swap.

    This function does not change the data structure. It runs in O(n) time.

    Args:
      pos_center_to_remove: proposed center to be removed.
      pos_center_to_add: proposed center to be added.

    Returns:
      The cost after the swap.
    """
    center_to_add = self.dataset[pos_center_to_add]
    total_cost = 0
    for point_pos, point in enumerate(self.dataset):
      cost_point = np.linalg.norm(point - center_to_add) ** 2
      if (
          self.point_to_min_dist_center_and_distance[point_pos][0]
          != pos_center_to_remove
      ):
        cost_point = min(
            cost_point, self.point_to_min_dist_center_and_distance[point_pos][1]
        )
      else:
        cost_point = min(
            cost_point,
            self.point_to_second_dist_center_and_distance[point_pos][1],
        )
      total_cost += cost_point
    return total_cost

  def SwapCenters(
      self, pos_center_to_remove, pos_center_to_add
  ):
    """Updates the data structure swapping two centers.

    Args:
      pos_center_to_remove: center to remove.
      pos_center_to_add: center to add.
    """
    invalidated_points = (
        self.center_to_min_dist_cluster[pos_center_to_remove]
        | self.center_to_second_dist_cluster[pos_center_to_remove]
    )
    for point in invalidated_points:
      min_c = self.point_to_min_dist_center_and_distance[point][0]
      self.center_to_min_dist_cluster[min_c].remove(point)
      second_c = self.point_to_second_dist_center_and_distance[point][0]
      self.center_to_second_dist_cluster[second_c].remove(point)

    self.centers.remove(pos_center_to_remove)
    del self.center_to_min_dist_cluster[pos_center_to_remove]
    del self.center_to_second_dist_cluster[pos_center_to_remove]
    self.centers.add(pos_center_to_add)
    for pos in invalidated_points:
      self.InitializeDatastructureForPoint(pos)
    for pos in range(len(self.dataset)):
      self.ProposeAsCenter(pos, pos_center_to_add)

  def SampleWithD2Distribution(self):
    """Sample a random point with prob. proportional to distance squared.

    Returns:
      The sampled point.
    """
    sum_cost = 0
    for i in range(len(self.dataset)):
      sum_cost += self.point_to_min_dist_center_and_distance[i][1]
    sampled_random = random.random() * sum_cost
    pos = 0
    while True:
      sampled_random -= self.point_to_min_dist_center_and_distance[pos][1]
      if sampled_random <= 0:
        break
      pos += 1
    return pos


def VanillaKMeans(dataset, k):
  """Vanilla (not fair) KMeans baseline.

  Args:
    dataset: the set of points.
    k: the number of clusters.

  Returns:
    The cluster centers.
  """
  kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(dataset)
  return kmeans.cluster_centers_
