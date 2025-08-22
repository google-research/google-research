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

"""Segmented ranking dataset with fast numpy support."""

from collections.abc import Mapping
import functools
import pathlib

import numpy as np


class SegmentedRankingDataset:
  """Ranking dataset with fast np support for query segmentation."""

  def __init__(
      self,
      scores,
      true_label_dist,
      pred_label_dist,
      query_segments,
      ranks = None,
      seed = 0,
      topn = 10,
  ):
    """Initializes the instance.

    Args:
      scores: An `[N]`-shaped array with the per-item scores to rank.
      true_label_dist: An `[N, L]`-shaped array with the per-item a one-hot
        distribution of the relevance labels.
      pred_label_dist: An `[N, L]`-shaped array with the per-item distribution
        of the predicted labels. This is usually the LLM's predicted probability
        of the suffix given the prompt, where the scored suffixes are the
        relevance labels ("0", "1", "2", ...).
      query_segments: An `[N]`-shaped array indicating the query segment of each
        item. Query segments should start at 0, be consecutive and be sorted in
        ascending order.
      ranks: An optional `[N]`-shaped array with the per-segment per-item
        1-based ranks. If not provided, the ranks will be computed from the
        scores.
      seed: The seed for the random number generator for random ops like
        tie-breaking ranks or creating bootstrap/batch samples.
      topn: The top-n to use for computing DCG metric.
    """
    self.scores = scores
    self.true_label_dist = true_label_dist
    self.pred_label_dist = pred_label_dist
    self.query_segments = query_segments
    self.num_queries = np.max(self.query_segments) + 1
    self.rng = np.random.default_rng(seed)
    self.topn = topn
    self.ranks = ranks
    if self.ranks is None:
      self.ranks = self.compute_ranks()
    self.bootstrap_seed = self.rng.integers(np.iinfo(np.int32).max)

  def compute_ranks(self):
    """Computes 1-based ranks for the scores per query."""
    # Do an argsort with random tiebreaking and grouping per query segment.
    ranks = np.lexsort((
        self.rng.uniform(size=self.scores.shape),
        -self.scores,
        self.query_segments,
    ))
    # Do a second argsort to get the per-item ranks with grouping per query
    # segment.
    ranks = np.lexsort((ranks, self.query_segments))
    # Get the per-segment minimum rank to subtract.
    _, counts = np.unique(self.query_segments, return_counts=True)
    min_ranks = (np.cumsum(counts) - counts)[self.query_segments]
    # Subtract the per-segment lowest rank so each segment starts its rank at 0.
    # Then add 1 to make sure the ranks are 1-based.
    return ranks - min_ranks + 1

  @functools.cached_property
  def _label_values(self):
    return np.arange(self.pred_label_dist.shape[-1])

  def _upper_bound_per_doc(self, degree):
    cum_pred = np.cumsum(self.pred_label_dist, axis=1)
    new_dist = self.pred_label_dist.copy()
    new_dist[:, 0] -= degree
    new_dist[:, 1:-1] -= np.maximum(0, degree - cum_pred[:, :-2])
    new_dist[:, :-1] = np.maximum(0, new_dist[:, :-1])
    new_dist[:, :] /= np.sum(new_dist, axis=1)[:, None]
    return np.sum(new_dist * self._label_values[None, :], axis=1)

  def _lower_bound_per_doc(self, degree):
    cum_pred = np.cumsum(self.pred_label_dist[:, ::-1], axis=1)[:, ::-1]
    new_dist = self.pred_label_dist.copy()
    new_dist[:, -1] -= degree
    new_dist[:, 1:-1] -= np.maximum(0, degree - cum_pred[:, 2:])
    new_dist[:, 1:] = np.maximum(0, new_dist[:, 1:])
    new_dist[:, :] /= np.sum(new_dist, axis=1)[:, None]
    return np.sum(new_dist * self._label_values[None, :], axis=1)

  def per_doc_pred_label(self, degree = 0.0):
    """Computes the predicted label per doc with degree-based bounding.

    Args:
      degree: The degree of the lower/upper bound in [-1, 1].

    Returns:
      An `[N]`-shaped array with the predicted labels.
    """
    if degree == 0.0:
      return np.sum(self.pred_label_dist * self._label_values[None, :], axis=-1)
    elif degree > 0:
      return self._upper_bound_per_doc(degree)
    else:
      return self._lower_bound_per_doc(-degree)

  def per_doc_true_label(self):
    """Computes the true label per doc."""
    return np.sum(self.true_label_dist * self._label_values[None, :], axis=-1)

  def per_doc_coverage(self, lower, upper):
    covered_lower = self.per_doc_pred_label(lower) <= self.per_doc_true_label()
    covered_upper = self.per_doc_pred_label(upper) >= self.per_doc_true_label()
    return float(np.mean(np.logical_and(covered_lower, covered_upper)))

  def _per_query_dcg(self, labels):
    label_fn = lambda x: (2.0**x) - 1.0
    weighted_ranks = 1.0 / np.log2(self.ranks + 1)
    weighted_ranks = np.where(self.ranks <= self.topn, weighted_ranks, 0.0)
    weighted_labels = label_fn(labels)
    per_query_result = np.zeros(self.num_queries)
    np.add.at(
        per_query_result, self.query_segments, weighted_ranks * weighted_labels
    )
    return per_query_result

  def per_query_pred_dcg(self, degree = 0.0):
    """Computes the predicted per-query performance."""
    return self._per_query_dcg(self.per_doc_pred_label(degree))

  def per_query_true_dcg(self):
    """Computes the true per-query performance."""
    return self._per_query_dcg(self.per_doc_true_label())

  def per_query_coverage(self, lower, upper):
    covered_lower = self.per_query_pred_dcg(lower) <= self.per_query_true_dcg()
    covered_upper = self.per_query_pred_dcg(upper) >= self.per_query_true_dcg()
    return float(np.mean(np.logical_and(covered_lower, covered_upper)))

  def per_query_bootstrap_true_dcg(self, n = 10_000):
    bootstrap_rng = np.random.RandomState(self.bootstrap_seed)
    idxs = bootstrap_rng.choice(
        np.arange(self.num_queries, dtype=np.int32), size=(n, self.num_queries)
    )
    return np.mean(self.per_query_true_dcg()[idxs], axis=-1)

  def per_query_bootstrap_pred_dcg(
      self,
      degree = 0.0,
      n = 10_000,
  ):
    bootstrap_rng = np.random.RandomState(self.bootstrap_seed)
    idxs = bootstrap_rng.choice(
        np.arange(self.num_queries, dtype=np.int32), size=(n, self.num_queries)
    )
    return np.mean(self.per_query_pred_dcg(degree)[idxs], axis=-1)

  def pred_dcg(self, degree = 0.0):
    """Computes the predicted performance on the entire dataset."""
    return float(np.mean(self.per_query_pred_dcg(degree)))

  def true_dcg(self):
    """Computes the true performance on the entire dataset."""
    return float(np.mean(self.per_query_true_dcg()))

  def sample(self, n):
    """Creates a sampled dataset consisting of `n` queries.

    Args:
      n: The number of queries to retain.

    Returns:
      A new `SegmentedRankingDataset` with a random subset of `n` queries.
    """
    query_indices = self.rng.choice(self.num_queries, n, replace=False)
    mask = np.in1d(self.query_segments, query_indices)
    _, new_query_segments = np.unique(
        self.query_segments[mask], return_inverse=True
    )
    return SegmentedRankingDataset(
        scores=self.scores[mask],
        true_label_dist=self.true_label_dist[mask, :],
        pred_label_dist=self.pred_label_dist[mask, :],
        query_segments=new_query_segments,
        seed=self.rng.integers(np.iinfo(np.int32).max),
        topn=self.topn,
        ranks=self.ranks[mask],
    )

  def split(
      self, n
  ):
    """Splits the dataset into two parts, for validation/testing.

    Args:
      n: The number of queries to retain in the first part.

    Returns:
      A tuple of two `SegmentedRankingDataset`s, the first with `n` queries
      randomly selected, and the second with the remaining queries.
    """
    query_indices_left = self.rng.choice(self.num_queries, n, replace=False)
    mask_left = np.in1d(self.query_segments, query_indices_left)
    mask_right = np.logical_not(mask_left)
    _, new_query_segments_left = np.unique(
        self.query_segments[mask_left], return_inverse=True
    )
    _, new_query_segments_right = np.unique(
        self.query_segments[mask_right], return_inverse=True
    )
    return (
        SegmentedRankingDataset(
            scores=self.scores[mask_left],
            true_label_dist=self.true_label_dist[mask_left, :],
            pred_label_dist=self.pred_label_dist[mask_left, :],
            query_segments=new_query_segments_left,
            seed=self.rng.integers(np.iinfo(np.int32).max),
            topn=self.topn,
            ranks=self.ranks[mask_left],
        ),
        SegmentedRankingDataset(
            scores=self.scores[mask_right],
            true_label_dist=self.true_label_dist[mask_right, :],
            pred_label_dist=self.pred_label_dist[mask_right, :],
            query_segments=new_query_segments_right,
            seed=self.rng.integers(np.iinfo(np.int32).max),
            topn=self.topn,
            ranks=self.ranks[mask_right],
        ),
    )

  def concatenate(
      self, other
  ):
    """Concatenates two datasets.

    Args:
      other: The other dataset to concatenate with.

    Returns:
      A new `SegmentedRankingDataset` that is the concatenation of the two.
    """
    return SegmentedRankingDataset(
        scores=np.concatenate((self.scores, other.scores), axis=0),
        true_label_dist=np.concatenate(
            (self.true_label_dist, other.true_label_dist), axis=0
        ),
        pred_label_dist=np.concatenate(
            (self.pred_label_dist, other.pred_label_dist), axis=0
        ),
        query_segments=np.concatenate(
            (self.query_segments, other.query_segments + self.num_queries),
            axis=0,
        ),
        seed=self.rng.integers(np.iinfo(np.int32).max),
        topn=self.topn,
        ranks=np.concatenate((self.ranks, other.ranks), axis=0),
    )

  def with_seed(self, seed):
    """Creates a copy of the dataset with a given RNG seed."""
    return SegmentedRankingDataset(
        scores=self.scores,
        true_label_dist=self.true_label_dist,
        pred_label_dist=self.pred_label_dist,
        query_segments=self.query_segments,
        ranks=self.ranks,
        seed=seed,
        topn=self.topn,
    )

  @classmethod
  def load(cls, file):
    """Loads the dataset from the given npz file."""
    return cls(**np.load(file))

  def adversarial_label_dist(self, ratio):
    new_dist = (
        ratio * (1.0 - self.pred_label_dist)
        + (1 - ratio) * self.pred_label_dist
    )
    new_dist /= np.sum(new_dist, axis=-1, keepdims=True)
    return SegmentedRankingDataset(
        scores=self.scores,
        true_label_dist=self.true_label_dist,
        pred_label_dist=new_dist,
        query_segments=self.query_segments,
        seed=self.rng.integers(np.iinfo(np.int32).max),
        topn=self.topn,
        ranks=self.ranks,
    )

  def oracle_label_dist(self, ratio):
    new_dist = (
        ratio * (self.true_label_dist) + (1 - ratio) * self.pred_label_dist
    )
    new_dist /= np.sum(new_dist, axis=-1, keepdims=True)
    return SegmentedRankingDataset(
        scores=self.scores,
        true_label_dist=self.true_label_dist,
        pred_label_dist=new_dist,
        query_segments=self.query_segments,
        seed=self.rng.integers(np.iinfo(np.int32).max),
        topn=self.topn,
        ranks=self.ranks,
    )

  def __len__(self):
    return self.num_queries


def read_datasets(
    data_dir,
):
  trecdl = SegmentedRankingDataset.load(data_dir / "trecdl.npz")
  robust04 = SegmentedRankingDataset.load(data_dir / "robust04.npz")
  # Split datasets into vali/test splits randomly.
  trecdl_vali, trecdl_test = trecdl.split(112)
  robust04_vali, robust04_test = robust04.split(125)
  return {
      "trecdl": {"vali": trecdl_vali, "test": trecdl_test},
      "robust04": {"vali": robust04_vali, "test": robust04_test},
  }
