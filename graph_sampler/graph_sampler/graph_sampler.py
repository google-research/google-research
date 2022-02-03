# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""The core algorithm: A sampler for graphs with given degree vector.

Multiple edges are allowed, but self-edges are not.
"""

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import runstats

EdgeList = List[Tuple[int, int]]


def can_be_connected(degrees):
  # Make a spanning tree on all nodes with valence > 1
  big_degrees = [d for d in degrees if d > 1]
  spanning_tree_valence = sum(big_degrees) - 2 * (len(big_degrees) - 1)
  # The graph can be connected if and only if there's enough room for all
  # the stuff of degree 1 to attach to this spanning tree.
  return degrees.count(1) <= spanning_tree_valence


def valid_degrees(degrees):
  """True if there exists a graph with given degree vector."""
  twice_num_edges = sum(degrees)
  if twice_num_edges % 2 != 0:
    return False
  # We exclude self-edges. The only way self-edges can be forced on us is if the
  # most popular vertex wants to have more edges than all other vertices
  # combined.
  return 2 * max(degrees, default=0) <= twice_num_edges


def valid_edge(i, j, degrees):
  """True if there is a graph containing an edge between i and j."""
  if i == j:
    # No self-edges allowed.
    return False
  if degrees[i] <= 0 or degrees[j] <= 0:
    return False
  degrees[i] -= 1
  degrees[j] -= 1
  valid = valid_degrees(degrees)
  # Restore the list of degrees before returning.
  degrees[i] += 1
  degrees[j] += 1
  return valid


def sample_graph(
    degrees,
    prune_disconnected = False,
    *,
    rng = None,
):
  """Produces an importance-sampled graph with the given degree vector.

  This is a modification of the algorithm in [1] which allows graphs to have
  multiple edges connecting two nodes. Note that vertices are labelled, but
  edges are not.

  [1] Joseph Blitzstein and Persi Diaconis, A Sequential Importance Sampling
  Algorithm for Generation Random Graphs with Prescribed Degrees.

  Args:
    degrees: a list of degrees.
    prune_disconnected: if True, uses an algorithm which avoids creating
      obviously disconnected graphs. Some still slip through though.
    rng: a np.random.Generator.

  Returns:
    A pair where the first element is a list of edges representing a graph, and
        the second element is an importance weight. The importance weight has
        the following property. If P is any probability distribution on graphs
        with this degree vector, f is any function, and we run this sampler N
        times, then
          sum([P(graph[i]) * importance[i] * f(graph[i]) for i in range(N)]) / N
        is an unbiased estimator of the expected value of f under P.
  """
  if rng is None:
    rng = np.random.default_rng()

  edges = []
  c_y = 1  # the number of equivalent edge sequences
  sigma_y = 1.0  # the probability of this edge sequence
  degrees = list(degrees)  # make a copy
  assert valid_degrees(degrees)

  connectedness_is_easy = True
  while max(degrees) > 0:
    node = degrees.index(min([d for d in degrees if d > 0]))
    c_y *= math.factorial(degrees[node])
    if degrees[node] > 1:
      # Once the minimal degrees are bigger than 1, the partially-built chunks
      # of our graph have could have more than one vertex with incomplete
      # bonding, so connectedness gets harder to track.
      connectedness_is_easy = False
    while degrees[node] > 0:
      # For every other vertex which node could connect to, connect to it with
      # weight proportional to its remaining degree. This choice of weight is
      # somewhat arbitrary. Its geneneral goal is to make the final result of
      # the sampling more uniform. For graphs without multiple edges, this
      # choice makes the sampling *exactly* uniform!
      weights = np.array(degrees)
      prune_disconnections = (
          connectedness_is_easy and prune_disconnected and sum(degrees) > 2)
      for other_node in range(len(weights)):
        if prune_disconnections and degrees[other_node] == 1:
          weights[other_node] = 0
        elif not valid_edge(node, other_node, degrees):
          weights[other_node] = 0

      weights = weights / np.sum(weights)
      partner = rng.choice(len(weights), p=weights)
      sigma_y *= weights[partner]
      edges.append(tuple(sorted([node, partner])))
      degrees[node] -= 1
      degrees[partner] -= 1

  edges = sorted(edges)

  # Graphs with multiple edges have been over-counted, with each k-fold edge
  # contributing an over-counting by factorial(k). Adjust c_y accordingly.
  for edge in set(edges):
    c_y /= math.factorial(edges.count(edge))

  return edges, 1.0 / (c_y * sigma_y)


def rng_seed_int32():
  # Why does this exist? Because integers larger than this get converted to
  # floats by igraph when writing to files.
  return np.random.default_rng().bit_generator.random_raw() % 2**31


@dataclasses.dataclass
class GraphSampler:
  """Yields graph samples, tracking various stats and stopping conditions.

  This class is will generate enough samples to
  (1) estimate the total number of graphs to specified confidence, and
  (2) uniformly sample a specified proportion of those graphs with rejection
      sampling.

  We achieve (1) by using our graph sampler to compute the expected value of the
  constant function f(graph) = num_graphs under the normalized probability
  distribution P(graph) = weight_func(graph) / num_graphs. We get the
  estimate
    sum_i(P(graph[i]) * importance[i] * f(graph[i])) / N
      = mean_i(weight_func(graph[i]) * importance[i])

  Attributes:
    degrees: a degree vector.
    weight_func: a function specifying how much to count each graph. This should
      return 0 for graphs you don't want to count and 1/num_automorphisms
      (however you define automorphisms) for graphs you do want to count.
    prune_disconnected: if True, use a sampling algorithm which prunes many
      disconnected graphs. Set this to True if your weight_func is zero on
      disconnected graphs.
    min_samples: minimum number of samples to generate. Note that this includes
      samples of weight zero, but samples of weight zero are internally dropped.
    rng_seed: an integer to seed random number generation.
    absolute_precision: If not None, keep sampling until the standard error is
      less than this number.
    relative_precision: If not None, keep sampling until (std_err /
      estimated_num_graphs) is less than this number.
    min_uniform_proportion: If not None, keep sampling until this this set of
      samples can be rejected down to a uniform sample containing at least this
      proportion of the estimated number of graphs.

  Yields:
    Triples (graph, importance, weight), where graph is a graph with importance
    weight importance*weight.
  """
  degrees: List[int]
  prune_disconnected: bool = False
  weight_func: Callable[[EdgeList], float] = lambda graph: 1.0
  min_samples: int = 500
  rng_seed: Optional[int] = None

  # stopping criteria
  absolute_precision: Optional[float] = None
  relative_precision: Optional[float] = None
  min_uniform_proportion: Optional[float] = None

  def __post_init__(self):
    if self.rng_seed is None:
      # Record an explicit seed to save out for reproducibility.
      self.rng_seed = rng_seed_int32()
    self._rng = np.random.default_rng(self.rng_seed)

    # initialize stats
    self._num_graphs_stats = runstats.Statistics()
    self._importance_stats = runstats.Statistics()
    self._weight_stats = runstats.Statistics()
    self._num_weight_zero_samples = 0

  def sample(self):
    """A single sample, with importance and weight."""
    graph, importance = sample_graph(
        self.degrees, self.prune_disconnected, rng=self._rng)
    weight = self.weight_func(graph)

    self._num_graphs_stats.push(importance * weight)
    self._importance_stats.push(importance)
    self._weight_stats.push(weight)
    if weight == 0:
      self._num_weight_zero_samples += 1
    return graph, importance, weight

  def should_stop(self):
    """True if enough samples have been generated to meet stopping criteria."""
    num_samples = len(self._num_graphs_stats)
    if num_samples < self.min_samples:
      return False

    num_graphs = self._num_graphs_stats.mean()
    std_err = self._num_graphs_stats.stddev(ddof=0) / np.sqrt(num_samples)
    if self._num_graphs_stats.maximum() == 0:
      num_after_rejection = 0
    else:
      acceptance_ratio = num_graphs / self._num_graphs_stats.maximum()
      num_after_rejection = num_samples * acceptance_ratio

    if (self.min_uniform_proportion and
        num_after_rejection / num_graphs < self.min_uniform_proportion):
      return False
    if self.absolute_precision and self.absolute_precision < std_err:
      return False
    if (self.relative_precision and std_err > 10.0 and self.relative_precision <
        (std_err / num_graphs)):
      return False

    return True

  def __iter__(self):
    while not self.should_stop():
      # Only call self.should_stop every once in a while.
      for _ in range(min(self.min_samples, 10000)):
        graph, importance, weight = self.sample()
        if weight > 0:
          yield graph, importance, weight

  def stats(self):
    """Dictionary of statistics about samples so far."""
    num_samples = len(self._num_graphs_stats)
    num_graphs = self._num_graphs_stats.mean()
    std_err = self._num_graphs_stats.stddev(ddof=0) / np.sqrt(num_samples)
    if self._num_graphs_stats.maximum() == 0:
      num_after_rejection = 0
    else:
      acceptance_ratio = num_graphs / self._num_graphs_stats.maximum()
      num_after_rejection = num_samples * acceptance_ratio

    stats = dict(
        rng_seed=self.rng_seed,
        estimated_num_graphs=num_graphs,
        num_graphs_std_err=std_err,
        num_after_rejection=num_after_rejection,
        num_samples=num_samples,
        num_weight_zero_samples=self._num_weight_zero_samples,
    )
    for name, stat in [('importance', self._importance_stats),
                       ('weight', self._weight_stats),
                       ('final_importance', self._num_graphs_stats)]:
      stats[f'min_{name}'] = stat.minimum()
      stats[f'max_{name}'] = stat.maximum()
    return stats


def estimate_number_of_graphs(degrees,
                              **kwargs):
  """Estimates the of number of graphs with given degree vector.

  Args:
    degrees: a degree vector.
    **kwargs: keyword arguments to pass along to GraphSampler.

  Returns:
    A pair (num_graphs, std_err), where num_graphs is an estimate of the
    (weighted) number of graphs with given degree vector and std_err is the
    standard error of the estimate.
  """
  sampler = GraphSampler(degrees, **kwargs)
  for _ in sampler:
    # Ignore the samples, we'll just return the estimated size.
    pass
  stats = sampler.stats()
  return stats['estimated_num_graphs'], stats['num_graphs_std_err']


def estimate_expected_value(func,
                            degrees,
                            unnormalized_probability = lambda graph: 1.0,
                            prune_disconnected = False,
                            num_samples = 500,
                            rng = None):
  """Estimates the expected value of func on graphs with given degree vector.

  Taking f(graph) = 1 for all graphs, we see that the expected value of
  P(graph)*importance is 1. Therefore 1/N can be estimated by
  1/sum(P(graph[i])*importance[i]) in the formula. This formulation allows us to
  use an unnormalized version of P, since it appears in both the numerator and
  denominator.

  Args:
    func: a function on graphs with given degree vector.
    degrees: a degree vector.
    unnormalized_probability: an unnormalized probability distribution.
    prune_disconnected: if True, use a sampling algorithm which prunes many
      disconnected graphs. Set this to True if unnormalized_probability is zero
      on disconnected graphs.
    num_samples: how many samples to use.
    rng: a np.random.Generator

  Returns:
    An estimate of the expected value of func under the normalization of
    unnormalized_probability.
  """
  total_weight = 0.0
  total_value = 0.0
  for _ in range(num_samples):
    graph, importance = sample_graph(degrees, prune_disconnected, rng=rng)
    total_weight += unnormalized_probability(graph) * importance
    total_value += unnormalized_probability(graph) * importance * func(graph)
  return total_value / total_weight
