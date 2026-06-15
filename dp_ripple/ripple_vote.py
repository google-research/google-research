# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Ripple vote mechanism."""

import numpy as np
from scipy import special

comb = special.comb

# pylint: disable=g-docstring-has-escape
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=invalid-name


def _sample_from_log_weights(size, log_weights):
  """Samples an index from unnormalized log weights using Gumbel-max trick.

  See
  https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparameterization_tricks

  Args:
    size: Integer size of the sample space.
    log_weights: Array of unnormalized log weights.
  """
  gumbel_samples = np.random.gumbel(size=size)
  return np.argmax(gumbel_samples + log_weights)


def _compute_labeled_forest_table(d):
  """Returns the F[d][m] table for 1 <= m <= d from Lemma 12.19.

  F[d][m] is the number of forests on d labeled vertices with m trees. See
  Lemma 12.19 for recurrence relation.

  Args:
    d: Integer number of vertices.
  """

  F = np.zeros((d + 1, d + 1))
  F[0][0] = 1
  for dim in range(1, d + 1):
    for num_trees in range(1, dim + 1):
      result = 0
      for k in range(1, d - num_trees + 2):
        result += (
            comb(dim - 1, k - 1) * (k ** (k - 2)) * F[dim - k][num_trees - 1]
        )
      F[dim][num_trees] = result
  return F


def _prufer_to_tree(prufer_sequence):
  """Returns a tree from a Prufer sequence.

  Returns a List of tuples representing the ordered edges (i,j) of the tree
  where i < j. See https://en.wikipedia.org/wiki/Prüfer_sequence#Algorithm_to_
  convert_a_Prüfer_sequence_into_a_tree.

  Args:
      prufer_sequence: A List of integers representing the Prufer sequence.
  """
  n = len(prufer_sequence) + 2
  degrees = [1] * n
  for node in prufer_sequence:
    degrees[node] += 1

  edges = []
  for node in prufer_sequence:
    for i in range(n):
      if degrees[i] == 1:
        edges.append((min(i, node), max(i, node)))
        degrees[i] -= 1
        degrees[node] -= 1
        break

  # Add the last edge
  u = -1
  v = -1
  for i in range(n):
    if degrees[i] == 1:
      if u == -1:
        u = i
      else:
        v = i
        break
  edges.append((min(u, v), max(u, v)))
  return edges


def _relabel_tree_edges(tree_edges, new_vertex_labels):
  """Returns a new List of tree edges with new vertex labels.

  Args:
    tree_edges: A List of tuples where (i,j) represents an edge between vertices
      i and j.
    new_vertex_labels: A dictionary mapping old vertex labels to new vertex
      labels.
  """
  relabeled_tree_edges = []
  for edge in tree_edges:
    relabeled_tree_edges.append(
        (new_vertex_labels[edge[0]], new_vertex_labels[edge[1]])
    )
  return relabeled_tree_edges


def _sample_random_forest(vertices, m, F):
  """Returns a uniformly sampled random forest with m trees (see Lemma 12.20).

  Returns a List of form [T_1,...,T_k] where each T_i is a tuple (vertices,
  edges) where vertices is a List of ints and edges is a List of ordered tuples
  (i,j) where i < j

  Args:
    vertices: is a List of Ints.
    m: Integer number of trees.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  d = len(vertices)
  if d == 1:
    assert m == 1
    return [(vertices, [])]
  if m == 1:
    prufer_sequence = np.random.choice(np.arange(d), size=d - 2)
    return [(vertices, _prufer_to_tree(prufer_sequence))]

  weights_for_picking_tree_0_size = [
      comb(d - 1, k - 1) * (k ** (k - 2)) * F[d - k][m - 1]
      for k in range(1, d - m + 2)
  ]
  tree_0_size = np.random.choice(
      np.arange(1, d - m + 2),
      p=weights_for_picking_tree_0_size / sum(weights_for_picking_tree_0_size),
  )
  tree_0 = None
  remaining_vertices = None
  if tree_0_size == 1:
    tree_0 = ([0], [])
    remaining_vertices = np.arange(1, d)
  else:
    random_permutation = np.random.permutation(np.arange(1, d))
    tree_0_vertices = [0] + list(random_permutation[: tree_0_size - 1])
    prufer_sequence = np.random.choice(
        np.arange(tree_0_size), size=tree_0_size - 2
    )
    prufer_tree_edges = _prufer_to_tree(prufer_sequence)
    tree_0_edges = _relabel_tree_edges(prufer_tree_edges, tree_0_vertices)
    tree_0 = (tree_0_vertices, tree_0_edges)
    remaining_vertices = random_permutation[tree_0_size - 1 :]
  subproblem_forest = _sample_random_forest(remaining_vertices, m - 1, F)
  return [tree_0] + subproblem_forest


def _compute_P_n_leq_size(d, n, F):
  """Returns |P_n_leq| from Lemma 6.7.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  if n == 0:
    return 1

  result = 0
  for i in range(d):
    result += F[d][d - i] * (n**i)
  return (n + 1) * result


def _compute_P_n_size(d, n, F):
  """Returns |P_n| from Lemma 6.8.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  if n == 0:
    return 1
  result = 0
  for i in range(d):
    result += F[d][d - i] * ((n + 1) * (n**i) - (n - 1) * ((n - 2) ** i))
  return result


def _random_forest_to_columns_of_submatrix_of_A(forest, num_edges):
  """Returns the submatrix of A (Definition 6.3) corresponding to a random forest.

  Args:
    forest: A List of tuples (vertices, edges) where vertices is a List of ints
      and edges is a List of ordered tuples (i,j) where i < j. Each such tuple
      represents a tree in the forest.
    num_edges: Integer number of edges in forest.
  """
  num_connected_components = len(forest)
  d = np.sum(
      np.array([len(forest[i][0]) for i in range(num_connected_components)])
  )
  submatrix = np.zeros((d, num_edges))
  column_index = 0
  for _, edges in forest:
    for edge in edges:
      submatrix[edge[0]][column_index] = 1
      submatrix[edge[1]][column_index] = -1
      column_index += 1
  return submatrix


def _sample_Y_star(d, n, F):
  """Returns Y_star sampled with the appropriate weights. See Claim 12.21.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  Q_index_weights = np.array([
      F[d][d - i] * ((n + 1) * (n**i) - ((n - 1) * (n - 2) ** (i)))
      for i in range(d)
  ])
  Q_index = np.random.choice(
      np.arange(d), p=Q_index_weights / np.sum(Q_index_weights)
  )
  # Q_index is the number of edges in the forest, so d - Q_index in the
  # number of trees in the forest
  Y_star = _sample_random_forest(np.arange(d), d - Q_index, F)
  return Y_star


def _sample_i_star(n, num_edges_Y_star):
  """Returns i_star sampled with the appropriate weights. See Claim 12.22.

  Args:
    n: Integer layer index.
    num_edges_Y_star: Integer number of edges in Y_star.
  """
  # if i = 0 or i = n, then log weight is log(n^edges_Y_star) =
  # edges_Y_star * log(n). Otherwise,
  # log(n^edges_Y_star - (n-2)^edges_Y_star)
  # = log(n^edges_Y_star[1 - ([n-2]/n)^edges_Y_star])
  # = edges_Y_star * log(n) + log(1 - [(n-2)/n]^edges_Y_star)
  log_i_index_weights = np.array(
      [num_edges_Y_star * np.log(n)]
      + [
          num_edges_Y_star * np.log(n)
          + np.log(1 - ((n - 2) / n) ** num_edges_Y_star)
          for _ in range(1, n)
      ]
      + [num_edges_Y_star * np.log(n)]
  )
  i_star = _sample_from_log_weights(n + 1, log_i_index_weights)
  return i_star


def _sample_S_star_S_star_complement_and_j_star(n, i_star, num_edges_Y_star):
  """Returns S_star, S_star_complement, and j_star sampled per Claim 12.23.

  See Claim 12.23.

  Args:
    n: Integer layer index.
    i_star: Integer index output of sample_i_star.
    num_edges_Y_star: Integer number of edges in Y_star.
  """

  j_star = None
  if i_star == 0 or i_star == n:
    j_star = i_star
  else:
    # Derivation for log_M_weights:
    # Let y = |Y^*| and i = i^*. Then
    # M_j = \binom{y}{j} * i^j * (n - i)^{y-j}
    #       - \binom{y}{j} * (i - 1)^j * (n - i - 1)^{y-j}
    # Let A = \binom{y}{j} * i^j * (n - i)^{y-j}
    # Let B = \binom{y}{j} * (i - 1)^j * (n - i - 1)^{y-j}
    # Using A - B = A * (1 - B/A), we get:
    # M_j = \binom{y}{j} * i^j * (n - i)^{y-j}
    #       * [1 - ((i - 1)/i)^j * ((n - i - 1)/(n - i))^{y-j}]
    # Taking the log of M_j:
    # \log(M_j) = \log\binom{y}{j} + j\log(i) + (y-j)\log(n-i)
    #             + \log[1 - ((i-1)/i)^j * ((n - i - 1)/(n-i))^{y-j}]
    log_M_weights = np.array([
        np.log(comb(num_edges_Y_star, j))
        + j * np.log(i_star)
        + (num_edges_Y_star - j) * np.log(n - i_star)
        + np.log(
            1
            - ((i_star - 1) / i_star) ** j
            * ((n - 1 - i_star) / (n - i_star)) ** (num_edges_Y_star - j)
        )
        for j in range(num_edges_Y_star + 1)
    ])
    j_star = _sample_from_log_weights(num_edges_Y_star + 1, log_M_weights)

  random_permutation = np.random.permutation(np.arange(num_edges_Y_star))
  S_star = random_permutation[:j_star]
  S_star_complement = random_permutation[j_star:]
  return S_star, S_star_complement, j_star


def _sample_lattice_point(
    d, n, i_star, S_star, S_star_complement, j_star, Y_star
):
  """Returns a lattice point uniformly sampled from P_n. See Claim 12.24.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    i_star: Integer index output of sample_i_star.
    S_star: Integer array of indices of vertices that is a subset of Y_star.
      Output of sample_S_star_S_star_complement_and_j_star.
    S_star_complement: Integer array of indices of vertices that is the
      complement of S_star in Y_star. Output of
      sample_S_star_S_star_complement_and_j_star.
    j_star: Integer index such that |S_star| = j_star.
    Y_star: A List of tuples (vertices, edges) where vertices is a List of ints
      and edges is a List of ordered tuples (i,j) where i < j. Each such tuple
      represents a tree in the forest. Output of sample_Y_star.
  """
  num_edges_Y_star = d - len(Y_star)
  # lattice_vector corresponds to the {c_{v_1}, ..., c_{v_{|Y^*|}}}
  # coefficients in the proof of Claim 12.24
  lattice_vector = None
  if i_star == 0:
    lattice_vector = np.random.choice(np.arange(-n, 0), size=num_edges_Y_star)
  elif i_star == n:
    lattice_vector = np.random.choice(
        np.arange(1, n + 1), size=num_edges_Y_star
    )
  else:
    lattice_vector = np.zeros(num_edges_Y_star)
    if i_star == 1 and (len(S_star) > 0):
      lattice_vector[S_star] = 1
      lattice_vector[S_star_complement] = np.random.choice(
          np.arange(-(n - i_star), 0), size=num_edges_Y_star - j_star
      )
    elif i_star == n - 1 and (len(S_star_complement) > 0):
      lattice_vector[S_star_complement] = -1
      lattice_vector[S_star] = np.random.choice(
          np.arange(1, i_star + 1), size=j_star
      )
    else:
      # one of the following holds:
      # case 1: 2 \leq i_star \leq n-2
      # case 2: i_star == 1 and len(S_star) == 0
      # case 3: i_star == n - 1 and len(S_star_complement) == 0
      log_P = np.zeros((j_star + 1, num_edges_Y_star - j_star + 1))
      log_P[0][0] = -np.inf  # Since P_{0,0} = 0
      for x in range(j_star + 1):
        for y in range(num_edges_Y_star - j_star + 1):
          if (x, y) != (0, 0):
            # Conditional logic avoids case where 0 * -inf = nan when there are
            # 0 coordinates where c_{v_k} = i_star, (j_star - x = 0), and
            # i_star - 1 = 0 (meaning log(i_star - 1) = -inf).
            pos_inner_term = (
                (j_star - x) * np.log(i_star - 1) if (j_star - x) > 0 else 0.0
            )
            # Conditional logic avoids case where avoids case where
            # 0 * -inf = nan when there are 0 coordinates where
            # c_{v_k} = -n + i_star, (num_edges_Y_star - j_star - y = 0), and
            # n - i_star - 1 = 0 (meaning log(n - i_star - 1) = -inf).
            neg_inner_term = (
                (num_edges_Y_star - j_star - y) * np.log(n - i_star - 1)
                if (num_edges_Y_star - j_star - y) > 0
                else 0.0
            )
            log_P[x][y] = (
                np.log(comb(j_star, x))
                + np.log(comb(num_edges_Y_star - j_star, y))
                + pos_inner_term
                + neg_inner_term
            )

      flattened_log_probabilities = log_P.flatten()
      chosen_flattened_index = _sample_from_log_weights(
          len(flattened_log_probabilities), flattened_log_probabilities
      )

      x_star, y_star = np.unravel_index(chosen_flattened_index, log_P.shape)

      lattice_vector[S_star[:x_star]] = i_star
      lattice_vector[S_star[x_star:]] = np.random.choice(
          np.arange(1, i_star), size=j_star - x_star
      )
      lattice_vector[S_star_complement[:y_star]] = -1 * (n - i_star)
      lattice_vector[S_star_complement[y_star:]] = np.random.choice(
          np.arange(-(n - i_star - 1), 0),
          size=num_edges_Y_star - j_star - y_star,
      )

  submatrix_of_Y_star = _random_forest_to_columns_of_submatrix_of_A(
      Y_star, num_edges_Y_star
  )
  result = np.zeros(d)

  for i in range(num_edges_Y_star):
    result += lattice_vector[i] * submatrix_of_Y_star[:, i]
  v = np.arange(0, d)
  return result + (-n + 2 * i_star) * v


def _sample_point_from_P_n(d, n, F):
  """Returns a point sampled uniformly from P_n (see Lemma 6.11).

  Args:
    d: Integer dimension.
    n: Integer layer index.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  if d == 1 or n == 0:
    return np.zeros(d)

  # Claim 12.21
  Y_star = _sample_Y_star(d, n, F)
  num_edges_Y_star = d - len(Y_star)

  # If Y* is empty, then return {-nv, nv} uniformly at random.
  if num_edges_Y_star == 0:
    random_sign = np.random.choice([-1, 1])
    return random_sign * n * np.arange(d)

  # Claim 12.22
  i_star = _sample_i_star(n, d - len(Y_star))

  # Claim 12.23
  S_star, S_star_complement, j_star = (
      _sample_S_star_S_star_complement_and_j_star(n, i_star, d - len(Y_star))
  )

  # Claim 12.24
  return _sample_lattice_point(
      d, n, i_star, S_star, S_star_complement, j_star, Y_star
  )


def _compute_normalizing_constant(d, eps, F, eulerian_numbers):
  """Returns Z(P_{n}(e^{-eps})). See Lemma 6.9.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
    eulerian_numbers: Numpy array of shape (d+1, d+1) such that
      eulerian_numbers[i][j] is the number of permutations of {1,...i} with j
      ascents.
  """
  z = np.exp(-eps)
  result = 1 + z / (1 - z) * (1 + 1 / (1 - z))
  for i in range(1, d):
    sub_result_1 = eulerian_numbers[i + 1][i] * z ** (
        i + 1
    )  # first summation's k=i term
    sub_result_2 = 0
    for k in range(i):
      sub_result_1 += eulerian_numbers[i + 1][k] * (z ** (k + 1))
      sub_result_2 += eulerian_numbers[i][k] * (z ** (k + 1))
    sub_result = (F[d][d - i] * (1 - z) ** (-i - 2)) * (
        sub_result_1 + (1 - z) * sub_result_2
    )
    result += sub_result
  return (1 - z**2) * result


def _compute_w_n(d, n, eps, normalizing_constant, F):
  """Returns w_n from Lemma 5.23.

  Args:
    d: Integer dimension.
    n: Integer layer index.
    eps: Float privacy parameter.
    normalizing_constant: Float normalizing constant for the ripple vote.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
  """
  P_n = _compute_P_n_size(d, n, F)
  return P_n * np.exp(-n * eps) / normalizing_constant


def _reservoir_sample_layer_index_n(d, eps, F, eulerian_numbers):
  """Returns a layer index from reservoir sampling using the w_n weights.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter.
    F: Integer array where F[d][m] is the number of forests on d labeled
      vertices with m trees (as computed by compute_labeled_forest_table).
    eulerian_numbers: Numpy array of shape (d+1, d+1) such that
      eulerian_numbers[i][j] is the number of permutations of {1,...i} with j
      ascents.
  """
  normalizing_constant = _compute_normalizing_constant(
      d, eps, F, eulerian_numbers
  )
  w_prefix_sum = [_compute_w_n(d, 0, eps, normalizing_constant, F)]
  w_index = 0
  success_probability = w_prefix_sum[0]
  while True:
    if np.random.binomial(1, success_probability) == 1:
      return w_index
    w_index += 1
    next_w_n = _compute_w_n(d, w_index, eps, normalizing_constant, F)
    w_prefix_sum.append(w_prefix_sum[-1] + next_w_n)
    success_probability = next_w_n / (1 - w_prefix_sum[w_index - 1])


def _compute_eulerian_numbers(d):
  """Returns A where A[i][j] is the Eulerian number A(i, j).

  A(i, j) is the number of permutations of {1,...i} with j ascents. Code is
  taken from
  https://github.com/google-research/google-research/blob/master/
  k_norm/sum_mechanism.py.


  Args:
    d: Integer such that the returned matrix has d+1 rows and columns.
  """
  eulerian_numbers = np.zeros((d + 1, d + 1))
  eulerian_numbers[:, 0] = np.ones(d + 1)
  for row in range(2, d + 1):
    for k in range(1, row + 1):
      eulerian_numbers[row, k] = (row - k) * eulerian_numbers[
          row - 1, k - 1
      ] + (k + 1) * eulerian_numbers[row - 1, k]
  return eulerian_numbers


def sample_ripple_vote_point(d, eps, num_samples=1):
  """Returns points sampled uniformly from the ripple vote mechanism.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter.
    num_samples: Integer number of samples.
  """
  samples = []
  F = _compute_labeled_forest_table(d)
  eulerian_numbers = _compute_eulerian_numbers(d)
  for _ in range(num_samples):
    n = _reservoir_sample_layer_index_n(d, eps, F, eulerian_numbers)
    samples.append(_sample_point_from_P_n(d, n, F))
  return samples
