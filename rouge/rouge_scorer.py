# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Computes rouge scores between two text blobs.

Implementation replicates the functionality in the original ROUGE package. See:

Lin, Chin-Yew. ROUGE: a Package for Automatic Evaluation of Summaries. In
Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004),
Barcelona, Spain, July 25 - 26, 2004.

Default options are equivalent to running:
ROUGE-1.5.5.pl -e data -n 2 -a settings.xml

Or with use_stemmer=True:
ROUGE-1.5.5.pl -m -e data -n 2 -a settings.xml

In these examples settings.xml lists input files and formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from nltk.stem import porter
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from rouge import scoring
from rouge import tokenize


class RougeScorer(scoring.BaseScorer):
  """Calculate rouges scores between two blobs of text.

  Sample usage:
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
  """

  def __init__(self, rouge_types, use_stemmer=False):
    """Initializes a new RougeScorer.

    Valid rouge types that can be computed are:
      rougen (e.g. rouge1, rouge2): n-gram based scoring.
      rougeL: Longest common subsequence based scoring.

    Args:
      rouge_types: A list of rouge types to calculate.
      use_stemmer: Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
    Returns:
      A dict mapping rouge types to Score tuples.
    """

    self.rouge_types = rouge_types
    self._stemmer = porter.PorterStemmer() if use_stemmer else None

  def score(self, target, prediction):
    """Calculates rouge scores between the target and prediction.

    Args:
      target: Text containing the target (ground truth) text.
      prediction: Text containing the predicted text.
    Returns:
      A dict mapping each rouge type to a Score object.
    Raises:
      ValueError: If an invalid rouge type is encountered.
    """

    target_tokens = tokenize.tokenize(target, self._stemmer)
    prediction_tokens = tokenize.tokenize(prediction, self._stemmer)
    result = {}

    for rouge_type in self.rouge_types:
      if rouge_type == "rougeL":
        # Rouge from longest common subsequences.
        scores = _score_lcs(target_tokens, prediction_tokens)
      elif re.match(r"rouge[0-9]$", rouge_type):
        # Rouge from n-grams.
        n = int(rouge_type[5:])
        if n <= 0:
          raise ValueError("rougen requires positive n: %s" % rouge_type)
        target_ngrams = _create_ngrams(target_tokens, n)
        prediction_ngrams = _create_ngrams(prediction_tokens, n)
        scores = _score_ngrams(target_ngrams, prediction_ngrams)
      else:
        raise ValueError("Invalid rouge type: %s" % rouge_type)
      result[rouge_type] = scores

    return result


def _create_ngrams(tokens, n):
  """Creates ngrams from the given list of tokens.

  Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.
  Returns:
    A dictionary mapping each bigram to the number of occurrences.
  """

  ngrams = collections.Counter()
  for ngram in (tuple(tokens[i:i + n]) for i in xrange(len(tokens) - n + 1)):
    ngrams[ngram] += 1
  return ngrams


def _score_lcs(target_tokens, prediction_tokens):
  """Computes LCS (Longest Common Subsequence) rouge scores.

  Args:
    target_tokens: Tokens from the target text.
    prediction_tokens: Tokens from the predicted text.
  Returns:
    A Score object containing computed scores.
  """

  if not target_tokens or not prediction_tokens:
    return scoring.Score(precision=0, recall=0, fmeasure=0)

  # Compute length of LCS from the bottom up in a table (DP appproach).
  cols = len(prediction_tokens) + 1
  rows = len(target_tokens) + 1
  lcs_table = np.zeros((rows, cols))
  for i in xrange(1, rows):
    for j in xrange(1, cols):
      if target_tokens[i - 1] == prediction_tokens[j - 1]:
        lcs_table[i, j] = lcs_table[i - 1, j - 1] + 1
      else:
        lcs_table[i, j] = max(lcs_table[i - 1, j], lcs_table[i, j - 1])
  lcs_length = lcs_table[-1, -1]

  precision = lcs_length / len(prediction_tokens)
  recall = lcs_length / len(target_tokens)
  fmeasure = scoring.fmeasure(precision, recall)

  return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _score_ngrams(target_ngrams, prediction_ngrams):
  """Compute n-gram based rouge scores.

  Args:
    target_ngrams: A Counter object mapping each ngram to number of
      occurrences for the target text.
    prediction_ngrams: A Counter object mapping each ngram to number of
      occurrences for the prediction text.
  Returns:
    A Score object containing computed scores.
  """

  intersection_ngrams_count = 0
  for ngram in six.iterkeys(target_ngrams):
    intersection_ngrams_count += min(target_ngrams[ngram],
                                     prediction_ngrams[ngram])
  target_ngrams_count = sum(target_ngrams.values())
  prediction_ngrams_count = sum(prediction_ngrams.values())

  precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
  recall = intersection_ngrams_count / max(target_ngrams_count, 1)
  fmeasure = scoring.fmeasure(precision, recall)

  return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)
