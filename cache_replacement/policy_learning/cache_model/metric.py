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

# Lint as: python3
"""Classes to track metrics about cache line evictions."""


import abc
import numpy as np
from scipy import stats
from cache_replacement.policy_learning.cache_model import utils


class CacheEvictionMetric(abc.ABC):
  """Tracks the value of a metric about predictions on cache lines."""

  @abc.abstractmethod
  def update(self, prediction_probs, eviction_mask, oracle_scores):
    """Updates the value of the metric based on a batch of data.

    Args:
      prediction_probs (torch.FloatTensor): batch of probability distributions
        over cache lines of shape (batch_size, num_cache_lines), each
        corresponding to a cache access. In each distribution, the lines are
        ordered from top-1 to top-num_cache_lines.
      eviction_mask (torch.ByteTensor): of shape (batch_size), where
        eviction_mask[i] = True if the i-th cache access results in an eviction.
        The metric value is tracked for two populations: (i) all cache accesses
        and (ii) for evictions.
      oracle_scores (list[list[float]]): the oracle scores of the cache lines,
        of shape (batch_size, num_cache_lines).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def write_to_tensorboard(self, tb_writer, tb_tag, step):
    """Writes the value of the tracked metric(s) to tensorboard.

    Args:
      tb_writer (tf.Writer): tensorboard writer to write to.
      tb_tag (str): metrics are written to tb_tag/metric_name(s).
      step (int): the step to use when writing to tensorboard.
    """
    raise NotImplementedError()


class SuccessRateMetric(CacheEvictionMetric):
  """Tracks the success rate of predicting the top-1 to top-k elements.

  Writes the following metrics to tensorboard:
    tb_tag/eviction_top_i for i = 1, ..., k
    tb_tag/total_top_i for i = 1, ..., k
  """

  def __init__(self, k):
    """Sets the value of k to track up to.

    Args:
      k (int): metric tracks top-1 to top-k.
    """
    self._k = k
    self._num_top_i_successes = {"total": [0] * k, "eviction": [0] * k}
    self._num_accesses = {"total": 0, "eviction": 0}

  def update(self, prediction_probs, eviction_mask, oracle_scores):
    del oracle_scores

    sorted_probs, _ = prediction_probs.sort(descending=True)
    num_elems = sorted_probs.shape[-1]
    for i in range(self._k):
      top_i_successes = (
          prediction_probs[:, 0] >= sorted_probs[:, min(i, num_elems - 1)])
      self._num_top_i_successes["total"][i] += top_i_successes.sum().item()
      self._num_top_i_successes["eviction"][i] += (
          eviction_mask * top_i_successes).sum().item()

    self._num_accesses["total"] += prediction_probs.shape[0]
    self._num_accesses["eviction"] += eviction_mask.sum().item()

  def write_to_tensorboard(self, tb_writer, tb_tag, step):
    for key in self._num_top_i_successes:
      for i in range(self._k):
        top_i_success_rate = (self._num_top_i_successes[key][i] /
                              (self._num_accesses[key] + 1e-8))
        utils.log_scalar(
            tb_writer, "{}/{}_top_{}".format(tb_tag, key, i + 1),
            top_i_success_rate, step)


class KendallWeightedTau(CacheEvictionMetric):
  """Tracks value of Kendall's weighted tau w.r.t. labeled ordering."""

  def __init__(self):
    self._weighted_taus = []
    self._masks = []

  def update(self, prediction_probs, eviction_mask, oracle_scores):
    del oracle_scores

    _, predicted_order = prediction_probs.sort(descending=True)
    for unbatched_order in predicted_order.cpu().data.numpy():
      # Need to negate arguments for rank: see weightedtau docs
      # NOTE: This is incorporating potentially masked & padded probs
      weighted_tau, _ = stats.weightedtau(
          -unbatched_order, -np.array(range(len(unbatched_order))), rank=False)
      self._weighted_taus.append(weighted_tau)
    self._masks.extend(eviction_mask.cpu().data.numpy())

  def write_to_tensorboard(self, tb_writer, tb_tag, step):
    weighted_taus = np.array(self._weighted_taus)
    eviction_masks = np.array(self._masks)

    eviction_mean_weighted_tau = np.sum(
        weighted_taus * eviction_masks) / (np.sum(eviction_masks) + 1e-8)
    utils.log_scalar(
        tb_writer, "{}/eviction_weighted_tau".format(tb_tag),
        eviction_mean_weighted_tau, step)

    utils.log_scalar(
        tb_writer, "{}/total_weighted_tau".format(tb_tag),
        np.mean(weighted_taus), step)


class OracleScoreGap(CacheEvictionMetric):
  """Tracks the gap between the oracle score of evicted vs. optimal line.

  Given lines l_1, ..., l_N with model probabilities p_1, ..., p_N and oracle
  scores s_1, ..., s_N, computes two gaps:
    - the optimal line is l_1 with score s_1
    - the evicted line is l_i, where i = argmax_j p_j
    - the quotient gap is computed as log(s_1 / s_i)
    - the difference gap is computed as log(s_i - s_1 + 1) [+1 to avoid log(0)]
    - scores are typically negative reuse distances.
  """

  def __init__(self):
    self._optimal_scores = []
    self._evicted_scores = []
    self._masks = []

  def update(self, prediction_probs, eviction_mask, oracle_scores):
    chosen_indices = prediction_probs.argmax(-1).cpu().data.numpy()

    # Default to optimal score = evicted score = 1, if no scores
    # Simpler than just excluding the scores, because of the masks.
    # Need to do explicit len check for numpy array
    # pylint: disable=g-explicit-length-test
    self._optimal_scores.append(
        [scores[0] if len(scores) > 0 else 1 for scores in oracle_scores])
    self._evicted_scores.append(
        [scores[index] if len(scores) > 0 else 1
         for (index, scores) in zip(chosen_indices, oracle_scores)])
    self._masks.append(eviction_mask.cpu().data.numpy())

  def write_to_tensorboard(self, tb_writer, tb_tag, step):
    eviction_masks = np.array(self._masks)
    difference_gap = np.log10(
        np.array(self._evicted_scores) - np.array(self._optimal_scores) + 1)
    quotient_gap = np.log10(
        np.array(self._optimal_scores) / np.array(self._evicted_scores))
    gaps = {
        "difference": difference_gap,
        "quotient": quotient_gap,
    }

    for gap_type, gap in gaps.items():
      eviction_mean_gap = np.sum(
          gap * eviction_masks) / (np.sum(eviction_masks) + 1e-8)
      utils.log_scalar(
          tb_writer, "{}/eviction_oracle_score_{}_gap".format(tb_tag, gap_type),
          eviction_mean_gap, step)
      utils.log_scalar(
          tb_writer, "{}/oracle_score_{}_gap".format(tb_tag, gap_type),
          np.mean(gap), step)
