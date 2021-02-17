# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilities for optimizing phrase vocabulary for Felix.

The goal is to find a fixed-size set of phrases that cover as many training
examples as possible. Based on the phrases, saves a file containing all possible
tags to be predicted and another file reporting the percentage of covered
training examples with different vocabulary sizes.
"""

import collections
import contextlib
import sys
from typing import Sequence, Text


from absl import logging
import numpy as np
import scipy.sparse

from felix import tokenization


def compute_lcs(source, target):
  """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    source: List of source tokens.
    target: List of target tokens.

  Returns:
    List of tokens in the LCS.
  """
  table = _lcs_table(source, target)
  return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
  """Returns the Longest Common Subsequence dynamic programming table."""
  rows = len(source)
  cols = len(target)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if source[i - 1] == target[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack(table, source, target, i, j):
  """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    source: List of source tokens.
    target: List of target tokens.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
  if i == 0 or j == 0:
    return []
  if source[i - 1] == target[j - 1]:
    # Append the aligned token to output.
    return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
  if table[i][j - 1] > table[i - 1][j]:
    return _backtrack(table, source, target, i, j - 1)
  else:
    return _backtrack(table, source, target, i - 1, j)


def _get_added_phrases(source_tokens,
                       target_tokens):
  """Computes the phrases that need to be added to the source to get the target.

  This is done by aligning each token in the LCS to the first match in the
  target and checking which phrases in the target remain unaligned.

  TODO: The LCS tokens should ideally be aligned to consecutive
  target tokens whenever possible, instead of aligning them always to the first
  match. This should result in a more meaningful phrase vocabulary with a higher
  coverage.

  Note that the algorithm is case-insensitive and the resulting phrases are
  always lowercase.

  Args:
    source_tokens: Source text tokens.
    target_tokens: Target text tokens.

  Returns:
    List of added phrases.

  Raises:
    RecursionError: If computing LCS between source and target tokens requires
      too many recursive calls. This can be avoided by calling
      `_recursion_limit` with a higher value.
  """
  kept_tokens = compute_lcs(source_tokens, target_tokens)
  added_phrases = []
  # Index of the `kept_tokens` element that we are currently looking for.
  kept_idx = 0
  phrase = []
  for token in target_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      kept_idx += 1
      if phrase:
        added_phrases.append(' '.join(phrase))
        phrase = []
    else:
      phrase.append(token)
  if phrase:
    added_phrases.append(' '.join(phrase))
  return added_phrases


@contextlib.contextmanager
def _recursion_limit(new_limit):
  original_limit = sys.getrecursionlimit()
  sys.setrecursionlimit(new_limit)
  try:
    yield
  finally:
    sys.setrecursionlimit(original_limit)


def added_token_counts(data_iterator,
                       try_swapping,
                       tokenizer,
                       max_input_examples=10000,
                       max_recursion_depth=10000):
  """Computes how many times different phrases have to be added.

  Args:
    data_iterator: Iterator to yield source lists and targets. See function
      yield_sources_and_targets in utils.py for the available iterators. The
      strings in the source list will be concatenated, possibly after swapping
      their order if swapping is enabled.
    try_swapping: Whether to try if swapping sources results in less added text.
    tokenizer: Text tokenizer (derived from tokenization.FullTokenizer).
    max_input_examples: Maximum number of examples to be read from the iterator.
    max_recursion_depth: Maximum recursion depth for LCS. If a long example
      surpasses this recursion depth, the given example is skipped and a warning
      is logged.

  Returns:
    Tuple (collections.Counter for phrases, added phrases for each example).
  """
  phrase_counter = collections.Counter()
  num_examples = 0
  all_added_phrases = []
  for sources, target in data_iterator:
    if num_examples >= max_input_examples:
      break
    logging.log_every_n(logging.INFO, f'{num_examples} examples processed.',
                        1000)
    source_tokens = [t.lower() for t in tokenizer.tokenize(' '.join(sources))]
    target_tokens = [t.lower() for t in tokenizer.tokenize(target)]
    with _recursion_limit(max_recursion_depth):
      try:
        added_phrases = _get_added_phrases(source_tokens, target_tokens)
        if try_swapping and len(sources) == 2:
          source_tokens_swap = [
              t.lower() for t in tokenizer.tokenize(' '.join(sources[::-1]))
          ]
          added_phrases_swap = _get_added_phrases(source_tokens_swap,
                                                  target_tokens)
          # If we can align more and have to add less after swapping, we assume
          # that the sources would be swapped during conversion.
          if len(''.join(added_phrases_swap)) < len(''.join(added_phrases)):
            added_phrases = added_phrases_swap
      except RecursionError:
        logging.log_first_n(
            logging.WARNING, 'Skipping a too long source. Consider increasing '
            '`max_recursion_depth` argument of the `added_token_counts` '
            'function in phrase_vocabulary_optimization_utils.py to keep this '
            f'source: {" ".join(source_tokens)}', 100)
        continue
    for phrase in added_phrases:
      phrase_counter[phrase] += 1
    all_added_phrases.append(added_phrases)
    num_examples += 1
  logging.info('%d examples processed.\n', num_examples)
  return phrase_counter, all_added_phrases


def construct_added_phrases_matrix(all_added_phrases, phrase_counter):
  """Constructs a sparse phrase occurrence matrix.

  Examples are on rows and phrases on columns.

  Args:
    all_added_phrases: List of lists of added phrases (one list per example).
    phrase_counter: Frequence of each unique added phrase.

  Returns:
    Sparse boolean matrix whose element (i, j) indicates whether example i
    contains the added phrase j. Columns start from the most frequent phrase.
  """
  phrase_2_idx = {
      tup[0]: i for i, tup in enumerate(phrase_counter.most_common())
  }
  matrix = scipy.sparse.dok_matrix((len(all_added_phrases), len(phrase_2_idx)),
                                   dtype=np.bool)
  for i, added_phrases in enumerate(all_added_phrases):
    for phrase in added_phrases:
      phrase_idx = phrase_2_idx[phrase]
      matrix[i, phrase_idx] = True
  # Convert to CSC format to support more efficient column slicing.
  return matrix.tocsc()


def count_covered_examples(matrix, vocabulary_size):
  """Returns the number of examples whose added phrases are in the vocabulary.

  This assumes the vocabulary is created simply by selecting the
  `vocabulary_size` most frequent phrases.

  Args:
    matrix: Phrase occurrence matrix with the most frequent phrases on the
      left-most columns.
    vocabulary_size: Number of most frequent phrases to include in the
      vocabulary.
  """
  # Ignore the `vocabulary_size` most frequent (i.e. leftmost) phrases (i.e.
  # columns) and count the rows with zero added phrases.
  return (matrix[:, vocabulary_size:].sum(axis=1) == 0).sum()
