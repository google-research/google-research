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

"""Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence
"""

import contextlib
import sys


@contextlib.contextmanager
def _recursion_limit(new_limit):
  original_limit = sys.getrecursionlimit()
  sys.setrecursionlimit(new_limit)
  try:
    yield
  finally:
    sys.setrecursionlimit(original_limit)


def compute_lcs(sequence_1, sequence_2, max_recursion_depth=10000):
  """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    sequence_1: First of the two sequences to be aligned.
    sequence_2: Second of the two sequences to be aligned.
    max_recursion_depth: Maximum recursion depth for the LCS backtracking.

  Returns:
    Sequence of items in the LCS.

  Raises:
    RecursionError: If computing LCS requires too many recursive calls.
      This can be avoided by setting a higher max_recursion_depth.
  """
  table = _lcs_table(sequence_1, sequence_2)
  with _recursion_limit(max_recursion_depth):
    return _backtrack(table, sequence_1, sequence_2, len(sequence_1),
                      len(sequence_2))


def _lcs_table(sequence_1, sequence_2):
  """Returns the Longest Common Subsequence dynamic programming table."""
  rows = len(sequence_1)
  cols = len(sequence_2)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if sequence_1[i - 1] == sequence_2[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack(table, sequence_1, sequence_2, i, j):
  """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    sequence_1: First of the two sequences to be aligned.
    sequence_2: Second of the two sequences to be aligned.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
  if i == 0 or j == 0:
    return []
  if sequence_1[i - 1] == sequence_2[j - 1]:
    # Append the aligned token to output.
    return _backtrack(table, sequence_1, sequence_2, i - 1,
                      j - 1) + [sequence_2[j - 1]]
  if table[i][j - 1] > table[i - 1][j]:
    return _backtrack(table, sequence_1, sequence_2, i, j - 1)
  else:
    return _backtrack(table, sequence_1, sequence_2, i - 1, j)
