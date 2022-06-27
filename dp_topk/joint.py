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

"""Joint exponential mechanism for differentially private top-k selection.

This file implements a joint exponential mechanism for eps-differentially
private top-k selection.
"""

import itertools
import numpy as np
from dp_topk.differential_privacy import NeighborType


def make_diff_matrix(item_counts, k):
  """Makes diff matrix where diff_matrix[i,j] = c_i - c_j + uniquifying term.

  Args:
    item_counts: Array of item counts, sorted in decreasing order.
    k: Number of top counts desired.

  Returns:
    k x d matrix diff_matrix where diff_matrix[i,j] = c_i - c_j + (d(k-i-1) + j
    + 1) / (2dk) and c_1 <= c_2 <= ... <= c_d. diff_matrix is therefore strictly
    increasing along rows and strictly decreasing down columns. Note that the
    added uniquifying term is determined entirely by d, i, j, k and is therefore
    data-independent. Moreover, since the counts are integers, their differences
    are integers as well; the uniquifying terms, which have maximum
    dk / (2dk) = 0.5, therefore do not change the relative order of any
    non-identical count differences.
  """
  d = len(item_counts)
  base_along_row = np.arange(1, d + 1)
  base_down_col = np.arange(k - 1, -1, -1) * d
  uniquifying_terms = (base_along_row[np.newaxis, :] +
                       base_down_col[:, np.newaxis]) / (2 * d * k)
  return (item_counts[:k, np.newaxis] -
          item_counts[np.newaxis, :]) + uniquifying_terms


def get_diffs_to_positions(diff_matrix):
  """Computes array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i].

  Args:
    diff_matrix: Matrix of distinct count differences.

  Returns:
    Array a where diff_matrix[a[0][i], a[1][i]] = sorted_diffs[i], where
    sorted_diffs contains all entries of diff_matrix in increasing order.
  """
  # The below line of the code runs in time O(d * k * log(dk)). This could be
  # implemented more efficiently, leveraging the fact that diff_matrix is
  # strictly increasing along rows. This property allows us to use k-way merging
  # (https://en.wikipedia.org/wiki/K-way_merge_algorithm), which can bring the
  # runtime down to O(d * k * log(k)). However, since this function is not a
  # practical bottleneck in the code, we leave it as-is for now.
  return np.unravel_index(np.argsort(diff_matrix, axis=None), diff_matrix.shape)


def brute_compute_log_diff_counts(diff_matrix, sorted_diffs):
  """Computes array of log(# sequences w/ diff) for diff in diff_matrix.

  Args:
    diff_matrix: Matrix of distinct count differences.
    sorted_diffs: Diffs from diff_matrix, in increasing order.

  Returns:
    Array log_counts where, for array sorted_diffs of diffs from diff_matrix
    sorted in increasing order,
    log_counts[i] = log(# of sequences where largest count difference is diff
    from sorted_diffs[i]), computed using brute force.
  """
  k, d = diff_matrix.shape
  possible_sequences = itertools.permutations(np.arange(d), k)
  diffs_to_counts = np.zeros(d * k)
  for sequence in possible_sequences:
    diff = np.amax(
        [diff_matrix[row_idx, sequence[row_idx]] for row_idx in np.arange(k)])
    diff_idx = np.searchsorted(sorted_diffs, diff)
    diffs_to_counts[diff_idx] += 1
  # Ignore warnings from taking log(0). This produces -np.inf as intended.
  with np.errstate(divide='ignore'):
    return np.log(diffs_to_counts)


def compute_log_diff_counts(diff_matrix, diffs_to_positions):
  """Computes array of log(sequence count) for each diff in diff_matrix.

  Args:
    diff_matrix: Matrix of distinct count differences.
    diffs_to_positions: Dictionary mapping diffs to positions in diff_matrix.

  Returns:
    Array log_counts where, for array sorted_diffs of diffs from diff_matrix
    sorted in decreasing order, log_counts[i] = log(# of sequences where largest
    count difference is diff from sorted_diffs[i]), computed using Lemma 3.7
    (the definition of \tilde{m}) from the paper.

  Raises:
    RuntimeError: ns vector never filled.
  """
  k, d = diff_matrix.shape
  num_diffs = d * k
  log_diff_counts = np.empty(num_diffs)
  log_ns = np.empty(k)
  indices_filled = set()
  last_diff_idx_processed = -1
  # Ignore warnings from, respectively, taking logs of 0 or negative numbers.
  # log(0) becomes -np.inf as intended, and log(<0) becomes nan and is ignored.
  with np.errstate(divide='ignore'):
    with np.errstate(invalid='ignore'):
      updates = np.log((diffs_to_positions[1] + 1) - diffs_to_positions[0])
  for (diff_idx, i, u) in zip(range(num_diffs), diffs_to_positions[0], updates):
    if np.isnan(u):
      continue
    log_ns[i] = u
    indices_filled.add(i)
    if len(indices_filled) == k:
      last_diff_idx_processed = diff_idx
      break
  if last_diff_idx_processed == -1:
    raise RuntimeError('ns vector never filled')
  log_diff_counts[:last_diff_idx_processed] = -np.inf
  log_ns_sum = np.sum(log_ns)
  for (diff_idx, i, u) in zip(
      range(last_diff_idx_processed, num_diffs),
      diffs_to_positions[0][diff_idx:], updates[diff_idx:]):
    log_ns_sum -= log_ns[i]
    log_diff_counts[diff_idx] = log_ns_sum
    log_ns[i] = u
    log_ns_sum += log_ns[i]
  return log_diff_counts


def racing_sample(log_terms):
  """Numerically stable method for sampling from an exponential distribution.

  Args:
    log_terms: Array of terms of form log(coefficient) - (exponent term).

  Returns:
    A sample from the exponential distribution determined by terms. See
    Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
    Utility Function Family for the Exponential Mechanism"
    (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
    analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithm.
  """
  return np.argmin(
      np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def sample_diff_idx(log_diff_counts, sorted_diffs, epsilon, neighbor_type):
  """Racing samples a diff index from the exponential mechanism.

  Args:
    log_diff_counts: Array of log(# sequences with diff) for each diff in
      sorted_diffs.
    sorted_diffs: Increasing array of possible diffs.
    epsilon: Privacy parameter epsilon.
    neighbor_type: Available neighbor types are defined in the NeighborType
      enum.

  Returns:
    Index idx sampled from, for diff = sorted_diffs[idx], distribution
    P(diff) ~ count[diff] * exp(-epsilon * floor(diff) / 2).
  """
  if neighbor_type is NeighborType.SWAP:
    sensitivity = 2
  else:
    sensitivity = 1
  return racing_sample(log_diff_counts - (epsilon * np.floor(sorted_diffs) /
                                          (2 * sensitivity)))


def sequence_from_diff(diff,
                       diff_row,
                       diff_col,
                       diff_matrix,
                       sampler=lambda ary: np.random.choice(ary, 1)):
  """Samples a sequence with given diff uniformly at random.

  Args:
    diff: Diff (negative utility) of sequence to sample.
    diff_row: diff_matrix[diff_row, diff_col] = diff.
    diff_col: diff_matrix[diff_row, diff_col] = diff.
    diff_matrix: Matrix of distinct count differences.
    sampler: Function that selects an item from an array. Default value is
      uniform random choice.

  Returns:
    Array of item indices forming a sequence with utility -diff.
  """
  k = len(diff_matrix)
  sequence = np.full(k, diff_col)
  ts = [
      np.searchsorted(diff_matrix[row, :], diff, side='right')
      for row in range(k)
  ]
  for row in range(k):
    if row != diff_row:
      # The below line of the code runs in time O(dk), which technically makes
      # the runtime of the sequence_from_diff function O(dk^2). This could be
      # implemented more efficiently, bringing the runtime of the function down
      # to O(dk). However, this is not a practical bottleneck in the code, so we
      # leave it as-is for now.
      to_sample = [i for i in range(ts[row]) if i not in sequence]
      sequence[row] = sampler(to_sample)
  return sequence


def joint(item_counts, k, epsilon, neighbor_type):
  """Applies joint exponential mechanism to return sequence of top-k items.

  Args:
    item_counts: Array of item counts.
    k: Number of items with top counts to return.
    epsilon: Privacy parameter epsilon.
    neighbor_type: Available neighbor types are defined in the NeighborType
      enum.

  Returns:
    Array of k item indices as estimated by the joint exponential mechanism.
  """
  # Sort the counts in non-increasing order.
  sort_indices = np.argsort(item_counts)[::-1]
  item_counts = item_counts[sort_indices]

  # Note that the diff_matrix here is the negative of the \tilde{U} matrix from
  # the paper.
  diff_matrix = make_diff_matrix(item_counts, k)
  diffs_to_positions = get_diffs_to_positions(diff_matrix)
  log_diff_counts = compute_log_diff_counts(diff_matrix, diffs_to_positions)
  sorted_diffs = diff_matrix[diffs_to_positions]
  diff_idx = sample_diff_idx(log_diff_counts, sorted_diffs, epsilon,
                             neighbor_type)
  diff_row, diff_col = diffs_to_positions[0][diff_idx], diffs_to_positions[1][
      diff_idx]
  sequence = sequence_from_diff(sorted_diffs[diff_idx], diff_row, diff_col,
                                diff_matrix)
  # Convert the indices returned by sequence_from_diff to the original item ids.
  return sort_indices[sequence]
