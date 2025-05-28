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

"""Implementation of the fair clustering algorithms."""

import hashlib
from typing import Optional, Sequence, Set, List

import numpy as np

from individually_fair_clustering import fair_clustering_utils


# Main algorithm Implementation
def AnchorPoints(
    dataset, k, strinct_dist_threshold_vec
):
  """Compute a set of anchor points.

  Args:
    dataset: the set of points.
    k: the number of clusters.
    strinct_dist_threshold_vec: the individual fairness distance thresholds of
      the points multiplied by the coefficent (e.g., 6).

  Returns:
    The anchor points positions or None if the problem is not feasible.
  """
  # sequence of elements by dist_threshold and then random in increasing order
  order_of_selection = list(range(len(dataset)))
  order_of_selection.sort(
      key=lambda i: (
          strinct_dist_threshold_vec[i],
          hashlib.md5(str(i).encode()).hexdigest(),
      )
  )

  # positions of centers
  centers_pos = []
  centers = []

  for i in order_of_selection:
    if not centers:
      centers_pos.append(i)
      centers.append(dataset[i])
      continue
    if (
        fair_clustering_utils.DistanceToCenters(dataset[i], centers, 1)
        > strinct_dist_threshold_vec[i]
    ):
      centers_pos.append(i)
      centers.append(dataset[i])
    if len(centers_pos) > k:
      return None
  return centers_pos


def IsValidCenterSwap(
    dataset,
    anchor_points,
    curr_centers,
    dist_threshold_vec,
    to_remove,
    to_add,
):
  """Check if the solution is feasible after a swap.

  Args:
    dataset: the set of points.
    anchor_points: the anchor points.
    curr_centers: the current solution,
    dist_threshold_vec: the individual fairness distance thresholds of the
      points .
    to_remove: the center to remove.
    to_add: the center to add.

  Returns:
    If the new solution is valid.
  """
  new_centers = set(curr_centers)
  new_centers.remove(to_remove)
  new_centers.add(to_add)
  new_centers_vec = [dataset[i] for i in new_centers]
  return fair_clustering_utils.IsFeasibleSolution(
      dataset, anchor_points, new_centers_vec, dist_threshold_vec
  )


def AnchorPointInitialization(
    dataset,
    k,
    dist_threshold_vec,
    coeff_anchor = 6.0,
):
  """Outputs å set of anchor points, filled to have exactly k points if needed.

  Args:
    dataset: the set of points.
    k: the number of clusters.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    coeff_anchor: coefficenet used for the anchor point

  Returns:
    The anchor points positions or None if the problem is not feasible
      and the centers.

  Raises:
    RuntimeError: if the solution is not feasible.
  """
  dist_threshold_vec = np.array(dist_threshold_vec)

  anchor_points = AnchorPoints(dataset, k, dist_threshold_vec * coeff_anchor)
  if anchor_points is None:
    raise RuntimeError("infeasible solution")
  assert anchor_points is not None
  centers = set(anchor_points)
  while len(centers) < k:
    c = fair_clustering_utils.FurthestPointPosition(
        dataset, [dataset[c] for c in centers]
    )
    centers.add(c)
  assert len(centers) == k
  return anchor_points, centers


def LocalSearchPlusPlus(
    dataset,
    *,
    k,
    dist_threshold_vec,
    coeff_anchor = 3.0,
    coeff_search = 1.0,
    number_of_iterations = 500,
    use_lloyd = True,
):
  """Our maim local search-based algorithm for individually-fair clustering.

  The algorithm outputs an approximate bicreteria k-means solution under
  individual-fairness constraints. The algorithm is based on efficient local
  search and improves over the speed of prior work.

  Args:
    dataset: the set of points.
    k: the number of clusters.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    coeff_anchor: coefficent used for the anchor points distance threshold.
    coeff_search: coefficent used for the local search distance threshold.
    number_of_iterations: number of local search iterations executed.
    use_lloyd: whether to use the fair Lloyd improvement steps.

  Returns:
    The cluster centers.

  Raises:
    RuntimeError: if the solution is not feasible.
  """
  anchor_points, centers = AnchorPointInitialization(
      dataset, k, dist_threshold_vec, coeff_anchor=coeff_anchor
  )

  curr_cost = fair_clustering_utils.KMeansCost(
      dataset, [dataset[c] for c in centers]
  )

  t2cc = fair_clustering_utils.TopTwoClosestToCenters(dataset, centers)
  for _ in range(number_of_iterations):
    proposed_new_center = t2cc.SampleWithD2Distribution()
    if proposed_new_center in centers:
      break  # zero distance to all points, we reached the optimum solution.
    best_swap_out = None
    best_swap_out_cost = (
        curr_cost  # consider a swap only if better than current.
    )
    for proposed_to_remove in centers:
      if not IsValidCenterSwap(
          dataset,
          anchor_points,
          centers,
          coeff_search * dist_threshold_vec,
          proposed_to_remove,
          proposed_new_center,
      ):
        break
      c = t2cc.CostAfterSwap(proposed_to_remove, proposed_new_center)
      if c < best_swap_out_cost:
        best_swap_out = proposed_to_remove
        best_swap_out_cost = c
    if best_swap_out is not None:
      centers.remove(best_swap_out)
      centers.add(proposed_new_center)
      t2cc.SwapCenters(best_swap_out, proposed_new_center)
      curr_cost = best_swap_out_cost
    else:
      pass

  if use_lloyd:
    # Lloyd's improvemetns
    return fair_clustering_utils.LloydImprovement(
        dataset,
        anchor_points_pos=anchor_points,
        inital_centers_vec=[dataset[c] for c in centers],
        dist_threshold_vec=dist_threshold_vec * 3,
        num_iter=20,
    )
  else:
    return [dataset[c] for c in centers]


def Greedy(
    dataset,
    *,
    k,
    dist_threshold_vec,
    coeff_anchor = 3.0,
):
  """Greedy fair baseline based on using the anchor points directly solution.

  Args:
    dataset: the set of points.
    k: the number of clusters.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    coeff_anchor: coefficent used for the anchor points distance threshold.

  Returns:
    The cluster centers.
  """

  _, init_centers = AnchorPointInitialization(
      dataset, k, dist_threshold_vec, coeff_anchor=coeff_anchor
  )
  return [dataset[c] for c in init_centers]


def LocalSearchICML2020(
    dataset,
    *,
    k,
    dist_threshold_vec,
    coeff_anchor = 3.0,
    coeff_search = 1.0,
    epsilon = 0.01,
    use_lloyd = False,
):
  """Implementation of the baseline individually-fair clustering algorithm.

  The algorithm outputs an approximate bicreteria k-means solution under
  individual-fairness constraints. The algorithm was presented in
  "Individual fairness for k-clustering", Mahabadi, Vakilian, ICML 2020.

  Args:
    dataset: the set of points.
    k: the number of clusters.
    dist_threshold_vec: the individual fairness distance thresholds of the
      points.
    coeff_anchor: coefficent used for the anchor points distance threshold.
    coeff_search: coefficent used for the local search distance threshold.
    epsilon: improvement required for a swap.
    use_lloyd: whether to use the fair Lloyd improvement steps. Note: Lloyd was
      not part of the ICML2020 paper.

  Returns:
    The cluster centers.

  Raises:
    RuntimeError: if the solution is not feasible.
  """
  anchor_points, centers = AnchorPointInitialization(
      dataset, k, dist_threshold_vec, coeff_anchor=coeff_anchor
  )

  curr_cost = fair_clustering_utils.KMeansCost(
      dataset, [dataset[c] for c in centers]
  )
  t2cc = fair_clustering_utils.TopTwoClosestToCenters(dataset, centers)

  iteration = 0
  while True:
    iteration += 1
    found = False
    proposed_new_center = 0
    while not found and proposed_new_center < len(dataset):
      if proposed_new_center in centers:
        proposed_new_center += 1
        continue
      for proposed_to_remove in centers:
        if IsValidCenterSwap(
            dataset,
            anchor_points,
            centers,
            coeff_search * dist_threshold_vec,
            proposed_to_remove,
            proposed_new_center,
        ):
          c = t2cc.CostAfterSwap(proposed_to_remove, proposed_new_center)
          if c < curr_cost * (1 - epsilon):
            centers.remove(proposed_to_remove)
            centers.add(proposed_new_center)
            t2cc.SwapCenters(proposed_to_remove, proposed_new_center)
            curr_cost = c
            found = True
            break
      proposed_new_center += 1
    if not found:
      break
  assert len(centers) == k
  centers = [dataset[c] for c in centers]

  if use_lloyd:
    # Lloyd's improvemetns
    return fair_clustering_utils.LloydImprovement(
        dataset,
        anchor_points_pos=anchor_points,
        inital_centers_vec=centers,
        dist_threshold_vec=dist_threshold_vec * 3,
        num_iter=20,
    )
  else:
    return centers
