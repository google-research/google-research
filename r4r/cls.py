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

"""Coverage weighted by length score (CLS).

Link to the original paper:
  https://arxiv.org/abs/1905.12255
"""

from __future__ import print_function

import networkx as nx
import numpy as np


class CLS(object):
  """Coverage weighted by length score (CLS).

  Python doctest:

  >>> cls = CLS(nx.grid_graph([3, 4]))
  >>> reference = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
  >>> assert np.isclose(cls(reference, reference), 1.0)
  >>> prediction = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)]
  >>> assert np.isclose(cls(reference, prediction), 0.81994915125863865)
  >>> prediction = [(0, 1), (1, 1), (2, 1), (3, 1)]
  >>> assert np.isclose(cls(reference, prediction), 0.44197196102702557)

  Link to the original paper:
    https://arxiv.org/abs/1905.12255
  """

  def __init__(self, graph, weight='weight', threshold=3.0):
    """Initializes a CLS object.

    Args:
      graph: networkx graph for the environment.
      weight: networkx edge weight key (str).
      threshold: distance threshold $d_{th}$ (float).
    """
    self.graph = graph
    self.weight = weight
    self.threshold = threshold
    self.distance = dict(
        nx.all_pairs_dijkstra_path_length(
            self.graph, weight=self.weight))

  def __call__(self, prediction, reference):
    """Computes the CLS metric.

    Args:
      prediction: list of nodes (str), path predicted by agent.
      reference: list of nodes (str), the ground truth path.

    Returns:
      the CLS between the prediction and reference path (float).
    """

    def length(nodes):
      return np.sum([
          self.graph.edges[edge].get(self.weight, 1.0)
          for edge in zip(nodes[:-1], nodes[1:])
      ])

    coverage = np.mean([
        np.exp(-np.min([  # pylint: disable=g-complex-comprehension
            self.distance[u][v] for v in prediction
        ]) / self.threshold) for u in reference
    ])
    expected = coverage * length(reference)
    score = expected / (expected + np.abs(expected - length(prediction)))
    return coverage * score
