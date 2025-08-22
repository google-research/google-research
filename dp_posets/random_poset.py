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

"""Functions for random poset experiments.

This module introduces a sampler to uniformly sample a random DAG on a fixed
number of uniquely labeled nodes. This is based on the following paper:
https://www.sciencedirect.com/science/article/abs/pii/S1571065304003944. It also
includes functions for running and plotting relevant experiments.
"""

import random
import numpy as np
from dp_posets import sensitivity_space_sampler
from dp_posets import utils


def is_cyclic_aux(
    adj,
    current_node,
    visited,
    rec_stack,
):
  """Auxiliary function to recursively detect cycle via depth first search.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.
    current_node: int representing the current node from which to explore.
    visited: list of bool representing the nodes that have ever been visited.
    rec_stack: list of bool keeping track of the nodes currently on the
      recursion stack of is_cyclic_aux.

  Returns:
    bool representing the existence of a cycle detected by exploring the graph
    starting from `current_node`.
  """
  if not visited[current_node]:
    # Mark the current node as visited and part of recursion stack
    visited[current_node] = True
    rec_stack[current_node] = True

    # Recur for all the nodes adjacent to this node
    for x in adj[current_node]:
      if not visited[x] and is_cyclic_aux(adj, x, visited, rec_stack):
        return True
      elif rec_stack[x]:
        return True

  # Remove the node from recursion stack
  rec_stack[current_node] = False
  return False


def is_cyclic(adj):
  """Function to detect whether the `adj` graph contains a directed cycle.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.

  Returns:
    bool representing the existence of a cycle in the graph.
  """
  num_nodes = len(adj)
  visited = [False] * num_nodes
  rec_stack = [False] * num_nodes

  # Call the recursive helper function to
  # detect cycle in different DFS trees
  for i in range(num_nodes):
    if not visited[i] and is_cyclic_aux(adj, i, visited, rec_stack):
      return True

  return False


def walk_one_step(adj, d):
  """Takes one step on the markov chain described in `generate_random_dag`.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.
    d: int representing the number of nodes in the graph.
  """
  start_node = random.randint(0, d - 1)
  end_node = random.randint(0, d - 1)
  if end_node in adj[start_node]:
    adj[start_node].remove(end_node)
  else:
    adj[start_node].add(end_node)
    if is_cyclic(adj):
      adj[start_node].remove(end_node)
  return


def generate_random_dag(d):
  """Generates a uniformly random DAG via the markov chain method by Melacon.

  The states of the markov chain are the set of node labeled DAGs. Let X_0 be
  d nodes with no edges. Let X_t be the state at time t. At time t, sample
  (i,j) uniformly at random where i,j are in {1,...,d}. If i->j is an edge of
  X_t, then remove it to form X_{t+1}. If i->j is not an edge of X_t, then add
  it to form X_{t+1} if it does not create a cycle or else X_{t+1} = X_t.
  See "Random Generation of Directed Acyclic Graphs" by Melacon. Also,
  see https://oeis.org/A003024.

  Args:
    d: int representing the number of nodes in the graph.

  Returns:
    Uniformly sampled DAG represented by `adj` which is a list of sets such that
    adj[v] is the set of out-going neighbors of v.
  """
  adj = [set([]) for _ in range(d)]
  # The markov chain has O(d^2) mixing time.
  for _ in range(10 * (d**2)):
    walk_one_step(adj, d)
  return adj


def get_matrix_from_adj(adj):
  """Get adjacency matrix from adjacency list representation of graph.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.

  Returns:
    The adjacency matrix representation of the input graph.
  """
  d = len(adj)
  adj_matrix = np.zeros((d, d))
  for i in range(d):
    for val in adj[i]:
      adj_matrix[i][val] = 1
  adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
  return adj_matrix


def get_transitive_closure_from_adj_matrix(
    adj_matrix,
):
  """Compute the order matrix using Floyd Warshall algorithm.

  The order matrix defines the poset order in sensitivity_space_sampler.py.
  Args:
    adj_matrix: adjacency matrix representation of a graph.

  Returns:
    The transitive closure of `adj_matrix`.
  """
  order = adj_matrix.copy()
  d = len(adj_matrix)
  for middle in range(d):
    # Pick all nodes as source one by one
    for start in range(d):
      # Pick all nodes as destination for the above picked source
      for end in range(d):
        # If node middle is on a path from start to end,
        # then make sure that the value of order[start][end] is 1
        order[start][end] = order[start][end] or (
            order[start][middle] and order[middle][end]
        )

  return np.array(order)


def get_order_from_adj(adj):
  """Compute order matrix from adjacency list.

  The order matrix defines the poset order in sensitivity_space_sampler.py
  Args:
   adj: list of sets such that adj[v] is the set of out-going neighbors of v.

  Returns:
    The order matrix corresponding to the input adjacency list.
  """
  # We have to transpose since order[i][j] means that i is a descendant of j
  # whereas adj_matrix[i][j] means that j is a descendant of i.
  return np.transpose(
      get_transitive_closure_from_adj_matrix(get_matrix_from_adj(adj))
  )


def find_depth_from_node(
    adj,
    start_node,
    depth_of_graph_from_node,
):
  """Computes the length of longest directed path from `start_node`.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.
    start_node: int representing the starting node from which to recursively
      compute the depth.
    depth_of_graph_from_node: dynamic programming dict where entry (v, d) states
      that the length of the longest directed path from v is d.

  Returns:
    Number of nodes in the longest directed path starting from `start_node`.
  """
  if start_node in depth_of_graph_from_node:
    return depth_of_graph_from_node[start_node]
  if len(adj[start_node]) == 0:
    return 1
  depth_of_neighbors = []
  for neighbor in adj[start_node]:
    depth_of_neighbors.append(
        find_depth_from_node(adj, neighbor, depth_of_graph_from_node)
    )
  depth_of_graph_from_node[start_node] = 1 + max(depth_of_neighbors)
  return 1 + max(depth_of_neighbors)


def find_depth_of_graph(adj):
  """Finds the number of nodes in the longest directed path in the graph.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.

  Returns:
    int representing the number of nodes in the longest directed path in the
    graph.
  """
  depth_of_graph_dp = {}
  all_depths = []
  for start_node in range(len(adj)):
    all_depths.append(find_depth_from_node(adj, start_node, depth_of_graph_dp))
  return max(all_depths)


def find_number_of_edges_of_graph(adj):
  """Compute number of edges of given graph.

  Args:
    adj: list of sets such that adj[v] is the set of out-going neighbors of v.

  Returns:
    int representing the number of edges of the graph.
  """
  edges = 0
  for v in range(len(adj)):
    edges += len(adj[v])
  return edges


def generate_samples_from_poset_ball_and_get_parameter(
    num_points, d, parameter_str
):
  """Generate a single DAG and generate samples from its poset ball.

  Args:
    num_points: int representing the number of samples to draw from the poset
      ball.
    d: int representing the number of nodes in the graph.
    parameter_str: string for the name of the parameter that we are varying. It
      is either "depth" or "num_edges".

  Returns:
    List of samples from poset ball, int representing the parameter value of the
    random graph.
  """
  adj = generate_random_dag(d)
  parameter = 0
  if parameter_str == "depth":
    parameter = find_depth_of_graph(adj)
  if parameter_str == "num_edges":
    parameter = find_number_of_edges_of_graph(adj)
  order = get_order_from_adj(adj)
  ground_set = list(np.arange(order.shape[0]))
  samples = []
  for _ in range(num_points):
    sample = sensitivity_space_sampler.sample_poset_ball(
        ground_set=ground_set, order=order
    )
    # Remove the root dimension from the sample
    samples.append(sample[1:])
  return samples, parameter


def compute_expected_norm_comparison(
    highest_dimension, num_graphs, num_samples
):
  """Compares poset_ball vs. unit l_inf ball expected squared l_{2} norm.

  We uniformly at random generate `num_graphs` DAGs and `num_samples` samples
  from the poset ball of each DAG.

  Args:
    highest_dimension: int representing the maximum number of nodes in a
      randomly generated DAG.
    num_graphs: int representing the number of randomly generated graphs.
    num_samples: int representing the number of samples from each poset ball.

  Returns:
    List of tuples of the form (d, e) where d is dimension and e is the
    expected norm ratio between the poset_ball and unit l_inf ball over
    random DAGs on d nodes.
  """
  expected_norm_ratios_per_dimension = {}
  for d in range(1, highest_dimension):
    expected_norm_ratios_for_all_graphs = []
    for _ in range(num_graphs):
      samples, _ = generate_samples_from_poset_ball_and_get_parameter(
          num_samples, d, ""
      )
      expected_norm_for_poset_ball = utils.compute_average_squared_l2_norm(
          samples
      )
      expected_norm_ratios_for_all_graphs.append(
          expected_norm_for_poset_ball
          / utils.compute_linf_average_squared_l2_norm(d)
      )
    expected_norm_ratios_per_dimension[d] = np.mean(
        expected_norm_ratios_for_all_graphs
    )

  expected_norm_ratio_pairs = []
  for key, val in expected_norm_ratios_per_dimension.items():
    expected_norm_ratio_pairs.append((key + 1, val))
  return expected_norm_ratio_pairs


def compute_expected_norm_comparison_for_fixed_dimension_over_parameter(
    d, num_graphs, parameter_str
):
  """Compares poset_ball vs. unit l_inf ball expected squared l_{2} norm.

  We uniformly at random generate `num_graphs` DAGs and 100 samples from the
  poset ball of each DAG.

  Args:
    d: int representing the number of nodes in the graph.
    num_graphs: int representing the number of randomly generated graphs.
    parameter_str: string for the name of the parameter that we are varying.

  Returns:
    List of tuples of the form (p, e) where p is the parameter value and e is
    the expected norm ratio between the poset_ball and unit l_inf ball over
    random DAGs with parameter value p.
  """
  parameter_lower_bound = None
  parameter_upper_bound = None
  if parameter_str == "depth":
    parameter_lower_bound = 1
    parameter_upper_bound = d
  if parameter_str == "num_edges":
    parameter_lower_bound = 0
    parameter_upper_bound = int(d * (d - 1) / 2)
  expected_norm_ratios_for_all_graphs_per_parameter = [
      [] for _ in range(parameter_upper_bound + 1)
  ]
  for _ in range(num_graphs):
    samples, parameter = generate_samples_from_poset_ball_and_get_parameter(
        100, d, parameter_str
    )
    expected_norm_for_poset_ball = utils.compute_average_squared_l2_norm(
        samples
    )
    expected_norm_for_linf_ball = utils.compute_linf_average_squared_l2_norm(d)
    expected_norm_ratios_for_all_graphs_per_parameter[parameter].append(
        expected_norm_for_poset_ball / expected_norm_for_linf_ball
    )
  expected_norm_ratios_per_parameter = []
  for parameter in range(parameter_lower_bound, parameter_upper_bound + 1):
    expected_norm_ratios_per_parameter.append((
        parameter,
        np.mean(expected_norm_ratios_for_all_graphs_per_parameter[parameter]),
    ))
  return expected_norm_ratios_per_parameter
