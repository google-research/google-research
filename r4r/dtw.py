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

"""Dynamic Time Warping based evaluation metrics for VLN."""

from __future__ import print_function

import networkx as nx
import numpy as np


class DTW(object):
  """Dynamic Time Warping (DTW) evaluation metrics.

  Python doctest:

  >>> graph = nx.grid_graph([3, 4])
  >>> prediction = [(0, 0), (1, 0), (2, 0), (3, 0)]
  >>> reference = [(0, 0), (1, 0), (2, 1), (3, 2)]
  >>> dtw = DTW(graph)
  >>> assert np.isclose(dtw(prediction, reference, 'dtw'), 3.0)
  >>> assert np.isclose(dtw(prediction, reference, 'ndtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction, reference, 'sdtw'), 0.77880078307140488)
  >>> assert np.isclose(dtw(prediction[:2], reference, 'sdtw'), 0.0)
  """

  def __init__(self, graph, weight='weight', threshold=3.0):
    """Initializes a DTW object.

    Args:
      graph: networkx graph for the environment.
      weight: networkx edge weight key (str).
      threshold: distance threshold $d_{th}$ (float).
    """
    self.graph = graph
    self.weight = weight
    self.threshold = threshold
    self.distance = dict(
        nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

  def __call__(self, prediction, reference, metric='sdtw'):
    """Computes DTW metrics.

    Args:
      prediction: list of nodes (str), path predicted by agent.
      reference: list of nodes (str), the ground truth path.
      metric: one of ['ndtw', 'sdtw', 'dtw'].

    Returns:
      the DTW between the prediction and reference path (float).
    """
    assert metric in ['ndtw', 'sdtw', 'dtw']

    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
      for j in range(1, len(reference)+1):
        best_previous_cost = min(
            dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
        cost = self.distance[prediction[i-1]][reference[j-1]]
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]

    if metric == 'dtw':
      return dtw

    ndtw = np.exp(-dtw/(self.threshold * len(reference)))
    if metric == 'ndtw':
      return ndtw

    success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
    return success * ndtw
