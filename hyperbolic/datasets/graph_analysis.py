# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Curvature estimates of interaction graphs functions."""

import random

from absl import app
from absl import flags
import networkx as nx
import numpy as np

from scipy.sparse import csr_matrix
from tqdm import tqdm

from hyperbolic.datasets.datasets import DatasetClass

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path',
    default='data/ml-1m/',
    help='Path to dataset')
flags.DEFINE_integer(
    'num_of_triangles', default=20, help='number of triangles to sample')


def pairs_to_adj(dataset):
  """Creates the adjacency matrix of the user-item bipartite graph."""
  as_matrix = np.zeros((dataset.n_users, dataset.n_items))
  for pair in dataset.data['train']:
    user, item = pair
    as_matrix[user, item] = 1.0
  return as_matrix


def interaction(adj_matrix, users_as_nodes=True):
  """Creates interaction matrix.

  Args:
    adj_matrix: Numpy array representing the adjacency matrix of the
      user-item bipartite graph.
    users_as_nodes: Bool indicating which interaction matrix to generate. If
      True (False), generates a user-user (item-item) interaction matrix.

  Returns:
    Numpy array of size n_users x n_users (n_items x n_items) with zeros
    on the diagonal and number of shared items (users) elsewhere, if user=True
    (False).
  """
  sp_adj = csr_matrix(adj_matrix)
  if users_as_nodes:
    ret_matrix = (sp_adj * sp_adj.transpose()).todense()
  else:
    ret_matrix = (sp_adj.transpose() * sp_adj).todense()
  np.fill_diagonal(ret_matrix, 0.0)
  return ret_matrix


def weight_with_degree(interaction_matrix, adj_matrix, users_as_nodes=True):
  """Includes the bipartite nodes degree in the interaction graph edges weights."""
  if users_as_nodes:
    degrees = np.sum(adj_matrix, axis=1).reshape(-1, 1)
  else:
    degrees = np.sum(adj_matrix, axis=0).reshape(-1, 1)
  sum_degrees = np.maximum(1, degrees + degrees.transpose())
  total_weight = 2 * interaction_matrix / sum_degrees
  return total_weight


def weight_to_dist(weight, exp=False):
  """Turns the weights to distances, if needed uses exponential scale."""
  if exp:
    return (weight != 0) * np.exp(-10* weight)
  return np.divide(
      1.0, weight, out=np.zeros_like(weight), where=weight != 0)


def xi_stats(graph, n_iter=20):
  """Calculates curvature estimates for a given graph.

  Args:
    graph: NetworkX Graph class, representing undirected graph with positive
      edge weights (if no weights exist, assumes edges weights are 1).
    n_iter: Int indicating how many triangles to sample in the graph.

  Returns:
    Tuple of size 3 containng the mean of the curvatures of the triangles,
    the standard deviation of the curvatures of the triangles and the total
    number of legally sampled triangles.
  """
  xis = []
  if not nx.is_connected(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
  nodes_list = list(graph.nodes())
  for _ in tqdm(range(n_iter), ascii=True, desc='Sample triangles'):
    # sample a triangle
    a, b, c = random.sample(nodes_list, 3)
    # find the middle node m between node b and c
    d_b_c, path_b_c = nx.single_source_dijkstra(graph, b, c)
    if len(path_b_c) <= 2:
      continue
    m = path_b_c[len(path_b_c) // 2]
    if m == a:
      continue
    # calculate xi for the sampled triangle, following section 3.2 in
    # Gu et al, Learning Mixed Curvature..., 2019
    all_len = nx.single_source_dijkstra_path_length(graph, a)
    d_a_m, d_a_b, d_a_c = all_len[m], all_len[b], all_len[c]
    xi = (d_a_m**2 + 0.25*d_b_c**2 -0.5*(d_a_b**2+d_a_c**2))/(2*d_a_m)
    xis.append(xi)
  return np.mean(xis), np.std(xis), len(xis)


def format_xi_stats(users_as_nodes, exp, xi_mean, xi_std, tot):
  """Formats the curvature estimates for logging.

  Args:
    users_as_nodes: Bool indicating which interaction graph was generated. If
      True (False), a user-user (item-item) interaction graph was generated.
    exp: Boolean indicating if the interaction graph distances are on
      an exponential scale.
    xi_mean: Float containng the mean of the curvatures of the sampled
      triangles.
    xi_std: Float containng the standard deviation of the curvatures of the
      sampled triangles.
    tot: Int containing the total number of legal sampled triangles.

  Returns:
    String storing the input information in a readable format.
  """
  stats = 'User-user stats:' if users_as_nodes else 'Item-item stats:'
  if exp:
    stats += ' (using exp)'
  stats += '\n'
  stats += '{:.3f} +/- {:.3f} \n'.format(xi_mean, xi_std)
  stats += 'out of {} samples.'.format(tot)
  return stats


def all_stats(dataset, n_iter=20):
  """Estimates curvature for all interaction graphs, returns a string summery."""
  summary = '\n'
  adj_matrix = pairs_to_adj(dataset)
  for users_as_nodes in [True, False]:
    one_side_interaction = interaction(adj_matrix, users_as_nodes)
    weights = weight_with_degree(one_side_interaction, adj_matrix,
                                 users_as_nodes)
    for exp in [True, False]:
      dist_matrix = weight_to_dist(weights, exp)
      graph = nx.from_numpy_matrix(dist_matrix)
      xi_mean, xi_std, tot = xi_stats(graph, n_iter)
      summary += format_xi_stats(users_as_nodes, exp, xi_mean, xi_std, tot)
      summary += '\n \n'
  return summary


def main(_):
  dataset_path = FLAGS.dataset_path
  data = DatasetClass(dataset_path, debug=False)
  print(all_stats(data, FLAGS.num_of_triangles))


if __name__ == '__main__':
  app.run(main)
