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

"""Models and baselines for clustering with weak and strong signals."""

import random
import gin
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering

from kwikbucks import cluster_utils
from kwikbucks import data_utils


@gin.configurable
def create_model(name):
  return globals()[name]()


@gin.configurable()
def set_random_seed(seed):
  """Sets random seed."""
  random.seed(seed)
  np.random.seed(seed)


@gin.configurable
class Clustering:
  """Abstract Clustering class from which models inherit."""

  def __init__(self, num_queries_multiplier=gin.REQUIRED):
    self.num_queries_multiplier = num_queries_multiplier

  def affinity_cluster(self, similarities):
    ap = AffinityPropagation(affinity='precomputed').fit(similarities)
    return cluster_utils.Clusters(
        {ex_id: c_id for ex_id, c_id in enumerate(list(ap.labels_))}
    )


class Singleton(Clustering):
  """Assigns each example to its own cluster."""

  def __init__(self):
    super().__init__()

  def cluster(self, dataset):
    return cluster_utils.Clusters({i: i for i in range(dataset.num_examples)})


class OnlyOneCluster(Clustering):
  """Assigns all examples to one cluster."""

  def __init__(self):
    super().__init__()

  def cluster(self, dataset):
    return cluster_utils.Clusters({i: 0 for i in range(dataset.num_examples)})


class Random(Clustering):
  """Randomly selects and queries pairs of nodes."""

  def __init__(self):
    super().__init__()
    self.cluster_pairs = cluster_utils.ClusterPairs()
    self.selected_pairs = set()

  def sample_pair(self, num_examples):
    while True:
      ex_id1 = random.randint(0, num_examples - 1)
      ex_id2 = random.randint(0, num_examples - 1)
      if ex_id1 != ex_id2 and (ex_id1, ex_id2) not in self.selected_pairs:
        self.selected_pairs.add((ex_id1, ex_id2))
        self.selected_pairs.add((ex_id2, ex_id1))
        return ex_id1, ex_id2

  def cluster(self, dataset):
    num_queries = dataset.num_examples * self.num_queries_multiplier
    for _ in range(num_queries):
      ex_id1, ex_id2 = self.sample_pair(dataset.num_examples)
      if dataset.same_cluster(ex_id1, ex_id2):
        self.cluster_pairs.add(ex_id1, ex_id2)
    return self.cluster_pairs.convert_cluster_pairs_to_clusters(
        dataset.num_examples
    )


class MostSimilarPairs(Clustering):
  """Runs expensive oracle on the most similar pairs."""

  def __init__(self):
    super().__init__()
    self.cluster_pairs = cluster_utils.ClusterPairs()

  def cluster(self, dataset):
    num_queries = dataset.num_examples * self.num_queries_multiplier
    for ex_id1, ex_id2 in dataset.most_similar_pairs(num_queries):
      if dataset.same_cluster(ex_id1, ex_id2):
        self.cluster_pairs.add(ex_id1, ex_id2)
    return self.cluster_pairs.convert_cluster_pairs_to_clusters(
        dataset.num_examples
    )


class WeakSignalAP(Clustering):
  """Runs Affinity Propoagation on the weak signal only."""

  def __init__(self):
    super().__init__()

  def cluster(self, dataset):
    return self.affinity_cluster(dataset.weak_signal)


class StrongSignalAP(Clustering):
  """Runs Affinity Propoagation on the strong signal (AP upper-bound)."""

  def __init__(self):
    super().__init__()

  def cluster(self, dataset):
    return self.affinity_cluster(dataset.strong_signal)


@gin.configurable
class CorrelationClustering(Clustering):
  """Class for corrleation clustering algorithms."""

  def __init__(self):
    """Constructor for CorrelationClustering class."""
    super().__init__()
    self.cluster_pairs = cluster_utils.ClusterPairs()

  @gin.configurable()
  def set_merge_post_processing_pairs(self, num_pairs_to_query=gin.REQUIRED):
    """Sets the number of pairs to query.

    Weak signal value above threshold is considered possible + neighbor.
    Requires threshold to be between 0 and 1.

    Args:
      num_pairs_to_query: an integer > 0. the number of random edges that will
        be queried when two clusters are possibly merged in merge_post_process

    Returns:
      None.
    """
    if num_pairs_to_query < 0:
      print('num_pairs_to_query must be an int > 0.')
      exit()
    self.num_pairs_to_query = num_pairs_to_query

  @gin.configurable()
  def set_num_pivots(self, num_examples, num_pivots=gin.REQUIRED):
    """Sets number of pivots to be used in qwick_cluster_using_ordering.

    Weak signal value above threshold is considered possible + neighbor.
    num_pivots is ideally a small positive integer, smaller than dataset size.

    Args:
      num_examples: size of dataset.
      num_pivots: number of pivots to set in correlation qwickcluster_ordered.

    Returns:
      None.
    """
    if num_pivots < 0 or num_pivots > num_examples:
      print('Number of pivots must be between 0 and the total # of points.')
    self.num_pivots = num_pivots

  def get_weak_signal_similarities(
      self, dataset, vertex_indices, list_of_elements
  ):
    """Calculates weak signal value between vertex_indices to list_of_elements."""
    if dataset.is_graph:
      return (
          dataset.features[list_of_elements, :]
          .dot(dataset.features[vertex_indices, :].T)
          .T
      )
    elif dataset.is_sparse:
      return dataset.weak_signal[vertex_indices, :][
          :, list_of_elements
      ].toarray()
    else:
      return dataset.weak_signal[vertex_indices, :][:, list_of_elements]

  def batch_weak_signal_ordering(
      self,
      dataset,
      vertex_indices,
      list_of_elements,
      ordering_style='weak_signal',
  ):
    """Calculates weak signal ordering from vertex_indices to list_of_elements."""
    if ordering_style == 'weak_signal':
      weak_signal_similarities = self.get_weak_signal_similarities(
          dataset, vertex_indices, list_of_elements
      )
      return np.argsort(weak_signal_similarities)[:, ::-1]
    else:
      return np.tile(range(len(list_of_elements)), (len(vertex_indices), 1))

  def batch_weak_signal_ordering_boosted_by_nearest_neighbor_assignments(
      self,
      dataset,
      vertex_indices,
      pivots,
      num_neighbors_per_pivot_of_vertex,
      gamma=0.1,
  ):
    """Calculates weak signal ordering incorporating neighborhood pivot info.

    The final similarity to a pivot is computed as cheap signal similarity +
    gamma*(# of nearest neighbors in weak signal which connect to the pivot).

    See the function update_neighborhood_pivot_stats which sets how many nearest
    neighbors are used.

    Args:
      dataset: input dataset (see Dataset class in data_utils).
      vertex_indices: the vertices that we calculate the weak signal ordering
        with respect to.
      pivots: pivots which will be sorted by the function according to the above
        formula.
      num_neighbors_per_pivot_of_vertex: for each vertex, the count of (already
        assigned) nearest neighbors (of the current vertex) to pivots.
      gamma: how to weight the value of number of neighbors which have already
        connected to some pivot.

    Returns:
      ordering of pivots, where each row corresponds to a vertex in
      vertex_indices.
    """
    weak_signal_similarities = self.get_weak_signal_similarities(
        dataset, vertex_indices, pivots
    )
    # If a pivot has a near neighbor already connected to it, boost its
    # weak signal similarity.
    for i, index in enumerate(vertex_indices):
      if index not in num_neighbors_per_pivot_of_vertex:
        continue
      for j, pivot in enumerate(pivots):
        if pivot not in num_neighbors_per_pivot_of_vertex[index]:
          continue
        increase_score = num_neighbors_per_pivot_of_vertex[index][pivot] * gamma
        weak_signal_similarities[i, j] += increase_score
    return np.argsort(weak_signal_similarities)[:, ::-1]

  def compute_non_pivots_ordering(
      self, dataset, non_pivots, pivots, batch_size
  ):
    """Orders non pivots based on weak signal similarities to pivots."""
    len_non_pivots = len(non_pivots)
    max_weak_signal_similarities = np.zeros(len_non_pivots)
    # Process non pivots in groups of batch_size
    for i in range(0, len_non_pivots, batch_size):
      end_index = min(len_non_pivots, i + batch_size)
      current_non_pivots = non_pivots[i:end_index]
      # Get weak signal values from current_non_pivots to the list of pivots.
      current_weak_signal_similarities = self.get_weak_signal_similarities(
          dataset, current_non_pivots, pivots
      )
      # Compute and record maximum weak signal similarities to pivots.
      max_weak_signal_similarities[i:end_index] = np.max(
          current_weak_signal_similarities, axis=1
      )
    # Sort non pivots by their maximum weak signal value to some pivot.
    sorted_by_max_weak_signal = np.argsort(max_weak_signal_similarities)
    return non_pivots[sorted_by_max_weak_signal][::-1]

  def randomly_sample_indices(self, max_value, num_of_samples):
    """Randomly samples num_of_samples random integers in [0, max_value)."""
    return np.random.randint(low=0, high=max_value, size=num_of_samples)

  def independent_set(
      self, dataset, pivots, ordering_style, max_neighbors_to_query=100
  ):
    """Calculates an independent set (IS) given list of sampled pivots.

    Algorithm idea: the first index is always in IS.
    For every other pivot in pivots, check if it has a positive edge to prior
    pivots. If no edge, add the current pivot to IS and continue. Check is done
    by sorting prior pivots using noisy oracle signal. Then we use strong oracle
    to validate that an edge exists.

    Args:
      dataset: input dataset (see Dataset class in data_utils).
      pivots: list of indices which are possible pivots to choose independent
        set from.
      ordering_style: how to order the pivots when checking for positive edges.
      max_neighbors_to_query: maximum number of weak signal nearest neighbors to
        check using strong signal query.

    Returns:
      An independent set according to the algorithm described
      and number of calls to expensive oracle.
    """
    # Keep track of the number of expensive queries used for IS part.
    expensive_queries = 0
    # List of final pivots, 1st one is always in.
    final_pivots = [pivots[0]]
    for possible_pivot in pivots[1:]:
      # Order previously chosen pivots by weak signal, from highest to lowest.
      ordering_to_use = self.batch_weak_signal_ordering(
          dataset, [possible_pivot], final_pivots, ordering_style
      )[0, :]
      is_new_pivot = True
      # Loop over already chosen pivots in weak signal order.
      for already_pivot_index in ordering_to_use[:max_neighbors_to_query]:
        already_pivot = final_pivots[already_pivot_index]
        expensive_queries += 1
        # Use expensive query to verify if a positive edge exists.
        if dataset.strong_signal[already_pivot, possible_pivot] == 1:
          # If edge exists with something before, can't be a pivot.
          is_new_pivot = False
          self.cluster_pairs.add(already_pivot, possible_pivot)
          break
      # No edge exists to something prior so declare new pivot.
      if is_new_pivot:
        final_pivots.append(possible_pivot)
    return final_pivots, expensive_queries

  def update_num_neighbors_per_pivot_of_vertex(
      self,
      dataset,
      current_vertices,
      vertices_to_pivots,
      num_neighbors_per_pivot_of_vertex,
      num_neighbors=10,
  ):
    """Updates the size of partitions of weak signal neighbors of vertices wrt their pivot assignments.

    Args:
      dataset: input dataset (see Dataset class in data_utils).
      current_vertices: vertices that we will update neighborhood statistics
        for.
      vertices_to_pivots: dictionary where key is an index and value is the
        pivot it is connected to.
      num_neighbors_per_pivot_of_vertex: for each index, records which pivots
        the index's nearest neighbors (in weak signal) are connected to. In
        particular, it is a dictionary where key is an index i and the value is
        another dictionary D_i. The keys of D_i are pivots p and its values are
        numbers (from 1 to num_neighbors) which indicates the number of nearest
        neighbors of the index i that are assigned to p.
      num_neighbors: number of nearest neighbors to be considered when recording
        pivot information.

    Returns:
      updated num_neighbors_per_pivot_of_vertex.
    """
    # Get weak signal nearest neighbors of current indices.
    nearest_neighbors_of_current_vertices = (
        dataset.k_nearest_neighbors_weak_signal(current_vertices, num_neighbors)
    )

    for j, vertex_index in enumerate(current_vertices):
      # Loop over nearest neighbors.
      for neighbor in nearest_neighbors_of_current_vertices[j, :]:
        # If neighbor has already been assigned to a pivot,
        # need to update neighborhood pivot stats.
        if neighbor == vertex_index or neighbor not in vertices_to_pivots:
          continue
        pivot_of_this_neighbor = vertices_to_pivots[neighbor]
        # Increment counter which records that the current
        # index has a neighbor assigned to pivot_of_this_neighbor.
        if vertex_index not in num_neighbors_per_pivot_of_vertex:
          num_neighbors_per_pivot_of_vertex[vertex_index] = {}
        if (
            pivot_of_this_neighbor
            not in num_neighbors_per_pivot_of_vertex[vertex_index]
        ):
          num_neighbors_per_pivot_of_vertex[vertex_index][
              pivot_of_this_neighbor
          ] = 1
        else:
          num_neighbors_per_pivot_of_vertex[vertex_index][
              pivot_of_this_neighbor
          ] += 1
    return num_neighbors_per_pivot_of_vertex

  @gin.configurable()
  def assign_to_pivots_given_ordering(
      self,
      dataset,
      points,
      pivots,
      budget,
      batch_size,
      max_neighbors_to_query,
      ordering_style,
      boost_by_neighbors_assignment=gin.REQUIRED,
  ):
    """Assigns points to pivots to form clusters.

    For every point, find the first pivot which it connects to.
    This is done by sorting the set of pivots using the ordering style
    (default is use ordering from weak signal). Then checking over this
    order of the pivots until a positive edge is found.

    Args:
      dataset: input dataset (see Dataset class from data_utils).
      points: the points which are assigned to pivots
      pivots: the pivots of the clusters
      budget: max # of strong signal calls that we can make.
      batch_size: number of points grouped together while computing weak signal
        ordering.
      max_neighbors_to_query: maximum number of weak signal nearest neighbors to
        check using strong signal.
      ordering_style: how to order the pivots when matching points to pivots.
      boost_by_neighbors_assignment: boolean to determine if we want to boost
        weak signal values of pivots using the assignment of nearest neighbors
        to the pivots.

    Returns:
      a Clusters class from cluster_utils and total number of
      strong signal queries used.
    """

    queries_used = 0
    vertices_to_pivots = {}
    num_neighbors_per_pivot_of_vertex = {}
    # Loop over points.
    for i in range(0, len(points), batch_size):
      end_index = min(i + batch_size, len(points))
      batch_node_indices = points[i:end_index]
      self.update_num_neighbors_per_pivot_of_vertex(
          dataset,
          batch_node_indices,
          vertices_to_pivots,
          num_neighbors_per_pivot_of_vertex,
      )
      # Calculate ordering of pivots from weak_signal.
      # Calculating weak signal ordering of pivots for many at once is faster.
      if boost_by_neighbors_assignment:
        ordering_to_use = self.batch_weak_signal_ordering_boosted_by_nearest_neighbor_assignments(
            dataset,
            batch_node_indices,
            pivots,
            num_neighbors_per_pivot_of_vertex,
        )
      else:
        ordering_to_use = self.batch_weak_signal_ordering(
            dataset, batch_node_indices, pivots, ordering_style
        )
      # Now loop over the indices that were grouped together.
      for j, index in enumerate(batch_node_indices):
        for pivot_index_order in ordering_to_use[j, :max_neighbors_to_query]:
          # If budget is exceeded, return clusters and queries used.
          if queries_used >= budget:
            return (
                self.cluster_pairs.convert_cluster_pairs_to_clusters(
                    dataset.num_examples
                ),
                queries_used,
            )
          current_pivot = pivots[pivot_index_order]
          queries_used += 1
          # Add current index to the current pivot's cluster
          # if strong signal says positive edge.
          if dataset.strong_signal[index, current_pivot] == 1:
            self.cluster_pairs.add(current_pivot, index)
            vertices_to_pivots[index] = current_pivot
            break
    return (
        self.cluster_pairs.convert_cluster_pairs_to_clusters(
            dataset.num_examples
        ),
        queries_used,
    )

  @gin.configurable()
  def qwick_cluster_using_ordering(
      self,
      dataset,
      budget,
      ordering_style=gin.REQUIRED,
      order_non_pivots=gin.REQUIRED,
      batch_size=gin.REQUIRED,
      max_neighbors_to_query=gin.REQUIRED,
  ):
    """Runs a variant of qwick cluster algorithm and returns clusters found.

    First sample t indices for pivots.
    Then extract independent set using IS algorithm.
    For every other point, find the first pivot which it connects to.
    This is done by sorting the set of pivots using the ordering specified
    (default is use ordering from weak signal).
    Then checking over this order of the pivots until a
    positive edge is found.

    Args:
      dataset: input dataset (see Dataset class from data_utils).
      budget: max number of strong signal calls that we can make.
      ordering_style: how to order the pivots when matching points to pivots,
        see assign_to_pivots_given_ordering for more information.
      order_non_pivots: boolean to determine if we want to assing points to
        pivots in weak signal ordering.
      batch_size: number of points grouped together while computing weak signal
        ordering.
      max_neighbors_to_query: maximum number of weak signal nearest neighbors to
        check using strong signal.

    Returns:
      a Clusters class from cluster_utils and total number of
      strong signal queries used.
    """

    # Check if number of pivots is set.
    try:
      self.num_pivots
    except AttributeError:
      print('num_pivots must be set in set_num_pivots')
      exit()

    # Randomly sample t points to be pivots.
    pivot_indices = np.random.choice(
        dataset.num_examples, replace=False, size=self.num_pivots
    )
    # Get independent set from pivots sampled.
    independent_pivots, expensive_queries_for_is = self.independent_set(
        dataset, pivot_indices, ordering_style, max_neighbors_to_query
    )
    # These are points that are yet to be clustered.
    indices_not_clustered_bool = np.ones(dataset.num_examples, dtype=bool)
    indices_not_clustered_bool[pivot_indices] = False
    indices_not_clustered = np.arange(dataset.num_examples)[
        indices_not_clustered_bool
    ]

    if order_non_pivots:
      indices_not_clustered = self.compute_non_pivots_ordering(
          dataset, indices_not_clustered, pivot_indices, batch_size
      )

    # Loop over all unclustered points.
    clusters, expensive_queries_for_clustering = (
        self.assign_to_pivots_given_ordering(
            dataset,
            indices_not_clustered,
            independent_pivots,
            budget - expensive_queries_for_is,
            batch_size,
            max_neighbors_to_query,
            ordering_style,
        )
    )
    total_queries_used = (
        expensive_queries_for_clustering + expensive_queries_for_is
    )
    return clusters, total_queries_used

  def approximate_strong_signal_between_clusters(
      self,
      dataset,
      cluster_ids_to_indices,
      len_cluster_ids,
      cluster_id_1,
      cluster_id_2,
  ):
    """Approximates the strong signal avg. score across two given clusters."""
    current_score = 0.0
    random_edges_head = self.randomly_sample_indices(
        len_cluster_ids[cluster_id_1], self.num_pairs_to_query
    )
    random_edges_tail = self.randomly_sample_indices(
        len_cluster_ids[cluster_id_2], self.num_pairs_to_query
    )
    current_score = 0.0
    # Query random pairs between the two clusters
    # and compute strong signal average.
    for t in range(self.num_pairs_to_query):
      current_score += dataset.strong_signal[
          cluster_ids_to_indices[cluster_id_1][random_edges_head[t]],
          cluster_ids_to_indices[cluster_id_2][random_edges_tail[t]],
      ]
    current_score /= self.num_pairs_to_query
    return current_score

  def compute_merge_score_for_two_clusters(
      self,
      dataset,
      len_cluster_ids,
      cluster_ids_to_indices,
      cluster_id_1,
      cluster_id_2,
  ):
    """Computes the suitability of merging two particular clusters."""
    # Compute the 'score' of pair where score is
    # the expected decrease in the correlation clustering objective.
    # First get the number of edges across two clusters.
    num_crossing_edges = (
        len_cluster_ids[cluster_id_1] * len_cluster_ids[cluster_id_2]
    )
    if dataset.is_graph:
      current_weak_signal_avg = np.average(
          dataset.features[cluster_ids_to_indices[cluster_id_1], :].dot(
              dataset.features[cluster_ids_to_indices[cluster_id_2], :].T
          )
      )
    else:
      current_weak_signal_avg = (
          dataset.weak_signal[cluster_ids_to_indices[cluster_id_1], :][
              :, cluster_ids_to_indices[cluster_id_2]
          ]
      ).mean()
    # Estimate the number of possible positive edges in between.
    # Note we transaform the weak signal average by f(x) = (x+1)/2
    # to map the interval [-1, 1] to [0, 1].
    expected_num_positive_edges = (
        0.5 * (current_weak_signal_avg + 1) * num_crossing_edges
    )
    # Get expected number of negative edges between the two clusters.
    expected_num_negative_edges = (
        num_crossing_edges - expected_num_positive_edges
    )
    # The score of each pair is the expected objective function decrease.
    expected_objective_cost_decrease = (
        expected_num_positive_edges - expected_num_negative_edges
    )
    return expected_objective_cost_decrease

  def compute_merge_scores_for_cluster_pairs(self, dataset, clusters, k=20):
    """Computes the suitability of merging for many pairs of clusters."""
    cluster_ids_to_indices = clusters.c_id_to_ex_id()
    cluster_ids = [
        c_id for c_id in cluster_ids_to_indices if cluster_ids_to_indices[c_id]
    ]
    len_cluster_ids = {
        c_id: len(cluster_ids_to_indices[c_id]) for c_id in cluster_ids
    }
    # Keep track of the scores computed for each cluster pair and which pairs
    # have already been processed so far.
    score_per_cluster_pair = []
    cluster_pairs_processed = set()
    # Only cluster pairs arising from edges of weak signal knn graph are
    # considered for merging. This is more efficient than looping over
    # all pairs.
    weak_signal_knn_graph = dataset.construct_weak_signal_knn_graph(k)
    rows, cols = weak_signal_knn_graph.nonzero()
    for edge_head, edge_tail in zip(rows, cols):
      cluster_id_1 = clusters.assignments[edge_head]
      cluster_id_2 = clusters.assignments[edge_tail]
      if cluster_id_2 < cluster_id_1:
        cluster_id_1, cluster_id_2 = cluster_id_2, cluster_id_1
      condition_1 = (cluster_id_1, cluster_id_2) not in cluster_pairs_processed
      condition_2 = cluster_id_1 != cluster_id_2
      if condition_1 and condition_2:
        cluster_pairs_processed.add((cluster_id_1, cluster_id_2))
        expected_objective_cost_decrease = (
            self.compute_merge_score_for_two_clusters(
                dataset,
                len_cluster_ids,
                cluster_ids_to_indices,
                cluster_id_1,
                cluster_id_2,
            )
        )
        score_per_cluster_pair.append(
            ((cluster_id_1, cluster_id_2), expected_objective_cost_decrease)
        )
    # Sort pairs by score.
    score_per_cluster_pair.sort(key=lambda x: x[1], reverse=True)
    return score_per_cluster_pair, len_cluster_ids

  @gin.configurable()
  def merge_post_process(
      self, dataset, clusters, budget, merge_threshold=gin.REQUIRED
  ):
    """Post proceesing step of possibly merging clusters.

    Algorithm: Order all pairs of two different clusters by score where
    score measures how 'suitable' the two clusters are for merging
    by computing the expected decrease in CC cost.
    Score = (# of edges across clusters)*(average
    weak signal value across clusters). We rank possible clusters to merge
    and only pursue a few number of possible merges as allowed by the budget.
    For each possible pair that we are merging, we verify that the merge
    helps by examining num_pair_to_query randomly cross cluster edges.
    We only merge if the strong signal average across these sampled edges
    is higher than the merge_threshold (ideally set to 0.5).
    This represents that the majority of the eges in between were positive
    edges, meaning the clusters should be merged.

    Args:
      dataset: input dataset (see Dataset class from data_utils).
      clusters: Clusters class from cluster_utils
      budget: how much budget to use in merging post processing
      merge_threshold: value such that we merge two clusters if the avg. strong
        signal value is above this threshold

    Returns:
      # of queries used and a dictionary where key = pivot indices,
      values are lists of points belonging to the pivot.
    """

    queries_used = 0
    cluster_ids_to_indices = clusters.c_id_to_ex_id()
    # Get the scores for each pair of clusters, sorted from highest to lowest
    # and also the number of points in each cluster.
    score_per_cluster_pair, len_cluster_ids = (
        self.compute_merge_scores_for_cluster_pairs(dataset, clusters)
    )
    alive_clusters = [True] * len(len_cluster_ids)

    for potential_merge_pair in score_per_cluster_pair:
      if queries_used >= budget:
        return (
            self.cluster_pairs.convert_cluster_pairs_to_clusters(
                dataset.num_examples
            ),
            queries_used,
        )

      cluster_id_1 = potential_merge_pair[0][0]
      cluster_id_2 = potential_merge_pair[0][1]
      # Check if the clusters of both pivots selected can still be merged.
      if (
          alive_clusters[potential_merge_pair[0][0]]
          and alive_clusters[potential_merge_pair[0][1]]
      ):
        current_score = self.approximate_strong_signal_between_clusters(
            dataset,
            cluster_ids_to_indices,
            len_cluster_ids,
            cluster_id_1,
            cluster_id_2,
        )
        queries_used += self.num_pairs_to_query
        # If strong signal avg larger than merge_threshold, merge clusters.
        if current_score >= merge_threshold:
          self.cluster_pairs.add(
              cluster_ids_to_indices[cluster_id_1][0],
              cluster_ids_to_indices[cluster_id_2][0],
          )
          alive_clusters[potential_merge_pair[0][1]] = False
    return (
        self.cluster_pairs.convert_cluster_pairs_to_clusters(
            dataset.num_examples
        ),
        queries_used,
    )

  def get_positive_neighborhood(self, dataset, index, list_of_vertices):
    """Get positive neighborhood of index among list_of_vertices (strong signal)."""
    if dataset.is_graph or dataset.is_sparse:
      _, positive_neighborhood, _ = sp.find(
          dataset.strong_signal[index, list_of_vertices] == 1.0
      )
    else:
      positive_neighborhood = np.where(
          dataset.strong_signal[index, list_of_vertices] == 1.0
      )[0]
    return positive_neighborhood

  def qwick_cluster_original(self, dataset, budget):
    """Runs the original qwick cluster algorithm and returns clusters found.

    Algorithm: See the description in https://arxiv.org/pdf/2002.11557.pdf.
    Pick a random vertex (pivot) and declare its positive neighborhood
    as a cluster. Remove these vertices and recurse on the remaining graph.
    Only uses the strong signal. Stops and returns all remaining vertices
    as singleton clusters if budget has been exceeded.

    Args:
      dataset: input dataset (see Dataset class from data_utils).
      budget: integer for max # of queries to strong oracle to use.

    Returns:
      # of queries used and a dictionary where key = pivot indices,
      values are lists of points belonging to the pivot.
    """
    queries_used = 0
    vertices_left_to_cluster = np.arange(dataset.num_examples)
    # Randomly shuffle array in the beginning which is equivalent
    # to randomly picking pivots every time.
    np.random.shuffle(vertices_left_to_cluster)
    # Loop while there are still vertices left to cluster.
    while vertices_left_to_cluster.size:
      if queries_used >= budget:
        return (
            self.cluster_pairs.convert_cluster_pairs_to_clusters(
                dataset.num_examples
            ),
            queries_used,
        )
      # Pick pivot.
      pivot, *vertices_left_to_cluster = vertices_left_to_cluster
      # If we will run out of queries, only query up to allocated budget.
      # We will immediately return on the next iteration since
      # queries_used >= budget.
      if queries_used + len(vertices_left_to_cluster) >= budget:
        leftover_queries = int(budget - queries_used)
        vertices_left_to_cluster = vertices_left_to_cluster[:leftover_queries]
      queries_used += len(vertices_left_to_cluster)
      # Find the positive neighborhood and add to pivot's cluster.
      current_cluster_members = self.get_positive_neighborhood(
          dataset, pivot, vertices_left_to_cluster
      )
      for cluster_member in current_cluster_members:
        self.cluster_pairs.add(vertices_left_to_cluster[cluster_member], pivot)
      # Remove positive neighborhood of chosen pivot
      # as they have now been clustered.
      vertices_left_to_cluster = np.delete(
          vertices_left_to_cluster, current_cluster_members
      )
    return (
        self.cluster_pairs.convert_cluster_pairs_to_clusters(
            dataset.num_examples
        ),
        queries_used,
    )

  @gin.configurable(module='correlation_clustering')
  def cluster(
      self,
      dataset,
      algorithm=gin.REQUIRED,
      budget_algorithm=gin.REQUIRED,
      budget_merge=gin.REQUIRED,
      use_merge_post_processing=gin.REQUIRED,
  ):
    """Function for calling various correlation clustering algorithms.

    Args:
      dataset: dataset class from data_utils.
      algorithm: choice of algorithm to use.
      budget_algorithm: budget to use for the clustering algorithm.
      budget_merge: budget to use for merge post processing.
      use_merge_post_processing: if post processing merge should be used or not.

    Returns:
        Clusters class.
    """
    if algorithm == 'qwick_cluster_using_ordering':
      # Set number of pivots. Note the number of pivots is
      # specified in gin config file. We pass along size of dataset
      # to check if the number of pivots set is larger than dataset size.
      self.set_num_pivots(dataset.num_examples)
      clusters, queries_used = self.qwick_cluster_using_ordering(
          dataset, budget_algorithm
      )
    elif algorithm == 'qwick_cluster':
      clusters, queries_used = self.qwick_cluster_original(
          dataset, budget_algorithm
      )
    else:
      clusters = None
      queries_used = 0
      use_merge_post_processing = False
    # Possibly call merge post processing.hg
    if use_merge_post_processing:
      # Set number of pairs to query when merging.
      self.set_merge_post_processing_pairs()
      # If algorithm hasn't used all of it's budget, we can increase budget for
      # merge post processing.
      budget_merge += max(0, budget_algorithm - queries_used)
      clusters, queries_used_merging = self.merge_post_process(
          dataset, clusters, budget_merge
      )
    else:
      queries_used_merging = 0
    print('Queries used:', queries_used + queries_used_merging)
    return clusters


class QwickClusterSampled(CorrelationClustering):
  """Run vanilla Qwick Cluster on strong signal graph restricted to sampled nodes."""

  def construct_neighborhood_graph(self, dataset, budget):
    """Construct the full neighborhood graph of selected vertices."""
    # Sample number of vertices allowed by budget.
    m = int(budget / dataset.num_examples)
    vertices_chosen = np.random.choice(
        dataset.num_examples, size=m, replace=False
    )
    possible_edges = []
    for v in vertices_chosen:
      for u in range(dataset.num_examples):
        possible_edges.append((v, u))
    neighborhood_graph = dataset.reweight_graph_using_strong_signal(
        possible_edges
    )
    m = int(budget / dataset.num_examples)
    return neighborhood_graph

  @gin.configurable(module='qwick_cluster_sampled')
  def cluster(self, dataset, budget_algorithm=gin.REQUIRED):
    # Get sampled subgraph and create dataset class.
    neighborhood_graph = self.construct_neighborhood_graph(
        dataset, budget_algorithm
    )
    neighborhood_graph_dataset = data_utils.AdhocDataset(neighborhood_graph, [])
    # Run vanilla Qwick Cluster with no budget restrictions
    # (budget already spent on creating graph).
    clusters, _ = self.qwick_cluster_original(
        neighborhood_graph_dataset, float('inf')
    )
    print('Queries used: %s', budget_algorithm)
    return clusters


class QwickClusterKnn(CorrelationClustering):
  """Run vanilla Qwick Cluster on knn graph from weak signal weighted by strong signal."""

  @gin.configurable(module='qwick_cluster_knn')
  def cluster(self, dataset, budget_algorithm=gin.REQUIRED):
    k = int(budget_algorithm / dataset.num_examples)
    strong_signal_weighted_knn_graph = dataset.construct_weighted_knn_graph(k)
    strong_signal_weighted_knn_graph_dataset = data_utils.AdhocDataset(
        strong_signal_weighted_knn_graph, []
    )
    clusters, _ = self.qwick_cluster_original(
        strong_signal_weighted_knn_graph_dataset, float('inf')
    )
    queries_used = k * dataset.num_examples
    print('Queries used: %s', queries_used)
    return clusters


class SpectralClusteringKnn(QwickClusterKnn):
  """Run spectral clustering on weak signal knn graph re-weighted by strong signal."""

  def spectral_clustering_on_knn_graph(self, dataset, k, n_clusters):
    """Call scikit-learn spectral clustering on weighted knn graph."""
    weighted_knn_graph = dataset.construct_weighted_knn_graph(k)
    clustering = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed'
    )
    clustering.fit(weighted_knn_graph)
    return clustering

  def turn_labels_to_clusters(self, cluster_labels):
    """Output clusters given spectral clustering labels."""
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
      current_cluster = np.where(cluster_labels == label)[0]
      for cluster_member in current_cluster[1:]:
        self.cluster_pairs.add(current_cluster[0], cluster_member)

  @gin.configurable(module='spectral_clustering_knn')
  def cluster(
      self, dataset, budget_algorithm=gin.REQUIRED, n_clusters=gin.REQUIRED
  ):
    k = int(budget_algorithm / dataset.num_examples)
    if k < 1:
      self.turn_labels_to_clusters(np.arange(dataset.num_examples))
    else:
      clustering = self.spectral_clustering_on_knn_graph(dataset, k, n_clusters)
      self.turn_labels_to_clusters(clustering.labels_)
    return self.cluster_pairs.convert_cluster_pairs_to_clusters(
        dataset.num_examples
    )
