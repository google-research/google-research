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

# Lint as: python2, python3
"""Library for scoring and evaluation of text samples.

Aggregation functions use bootstrap resampling to compute confidence intervals
as per the original ROUGE perl implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import numpy as np
import six
from six.moves import range


class Score(
    collections.namedtuple("Score", ["precision", "recall", "fmeasure"])):
  """Tuple containing precision, recall, and f-measure values."""


class BaseScorer(object):
  """Base class for Scorer objects."""

  @abc.abstractmethod
  def score(self, target, prediction):
    """Calculates score between the target and prediction.

    Args:
      target: Text containing the target (ground truth) text.
      prediction: Text containing the predicted text.

    Returns:
      A dict mapping each score_type (string) to Score object.
    """


class AggregateScore(
    collections.namedtuple("AggregateScore", ["low", "mid", "high"])):
  """Tuple containing confidence intervals for scores."""


class BootstrapAggregator(object):
  """Aggregates scores to provide confidence intervals.

  Sample usage:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
    aggregator = Aggregator()
    aggregator.add_scores(scorer.score("one two three", "one two"))
    aggregator.add_scores(scorer.score("one two five six", "seven eight"))
    result = aggregator.aggregate()
    print result
    {'rougeL': AggregateScore(
         low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
         mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
         high=Score(precision=1.0, recall=0.66, fmeasure=0.80)),
     'rouge1': AggregateScore(
         low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
         mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
         high=Score(precision=1.0, recall=0.66, fmeasure=0.80))}
  """

  def __init__(self, confidence_interval=0.95, n_samples=1000):
    """Initializes a BootstrapAggregator object.

    Args:
      confidence_interval: Confidence interval to compute on the mean as a
        decimal.
      n_samples: Number of samples to use for bootstrap resampling.

    Raises:
      ValueError: If invalid argument is given.
    """

    if confidence_interval < 0 or confidence_interval > 1:
      raise ValueError("confidence_interval must be in range [0, 1]")
    if n_samples <= 0:
      raise ValueError("n_samples must be positive")

    self._n_samples = n_samples
    self._confidence_interval = confidence_interval
    self._scores = collections.defaultdict(list)

  def add_scores(self, scores):
    """Adds a sample for future aggregation.

    Args:
      scores: Dict mapping score_type strings to a namedtuple object/class
        representing a score.
    """

    for score_type, score in six.iteritems(scores):
      self._scores[score_type].append(score)

  def aggregate(self):
    """Aggregates scores previously added using add_scores.

    Returns:
      A dict mapping score_type to AggregateScore objects.
    """

    result = {}
    for score_type, scores in six.iteritems(self._scores):
      # Stack scores into a 2-d matrix of (sample, measure).
      score_matrix = np.vstack(tuple(scores))
      # Percentiles are returned as (interval, measure).
      percentiles = self._bootstrap_resample(score_matrix)
      # Extract the three intervals (low, mid, high).
      intervals = tuple(
          (scores[0].__class__(*percentiles[j, :]) for j in range(3)))
      result[score_type] = AggregateScore(
          low=intervals[0], mid=intervals[1], high=intervals[2])
    return result

  def _bootstrap_resample(self, matrix):
    """Performs bootstrap resampling on a matrix of scores.

    Args:
      matrix: A 2-d matrix of (sample, measure).

    Returns:
      A 2-d matrix of (bounds, measure). There are three bounds: low (row 0),
      mid (row 1) and high (row 2). Mid is always the mean, while low and high
      bounds are specified by self._confidence_interval (which defaults to 0.95
      meaning it will return the 2.5th and 97.5th percentiles for a 95%
      confidence interval on the mean).
    """

    # Matrix of (bootstrap sample, measure).
    sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
    for i in range(self._n_samples):
      sample_idx = np.random.choice(
          np.arange(matrix.shape[0]), size=matrix.shape[0])
      sample = matrix[sample_idx, :]
      sample_mean[i, :] = np.mean(sample, axis=0)

    # Take percentiles on the estimate of the mean using bootstrap samples.
    # Final result is a (bounds, measure) matrix.
    percentile_delta = (1 - self._confidence_interval) / 2
    q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
    return np.percentile(sample_mean, q, axis=0)


def fmeasure(precision, recall):
  """Computes f-measure given precision and recall values."""

  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0
