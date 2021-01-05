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

"""Defines different eviction policies."""

import abc
import heapq
import numpy as np
import six


class EvictionPolicy(six.with_metaclass(abc.ABCMeta, object)):
  """Policy for determining what cache line to evict."""

  @abc.abstractmethod
  def __call__(self, cache_access, access_times):
    """Chooses a cache line to evict.

    Args:
      cache_access (CacheAccess): the cache access to make an eviction decision
        about.
      access_times (dict): maps int address to last time of access. Smaller
        times were accessed longer ago.

    Returns:
      line_to_evict (int | None): the cache line the policy has chosen to evict.
        None if there are no cache lines.
      scores (dict{int: int}): maps each cache line (int) to its score (int).
        Lower score means the policy prefers to evict the cache line more.
    """
    raise NotImplementedError()


class GreedyEvictionPolicy(EvictionPolicy):
  """Evicts the cache-line with the n-th lowest score."""

  def __init__(self, cache_line_scorer, n=0):
    """Constructs.

    Args:
      cache_line_scorer (CacheLineScorer): used to score the cache lines.
      n (int): policy selects the n-th smallest score according to the
        cache_line_scorer. If n is larger than the number of cache lines,
        returns the cache line with the highest score (least evictable).
    """
    super().__init__()

    self._cache_line_scorer = cache_line_scorer
    self._n = n

  def __call__(self, cache_access, access_times):
    scores = self._cache_line_scorer(cache_access, access_times)
    sorted_cache_lines = heapq.nsmallest(
        self._n + 1, scores.keys(), key=lambda line: scores[line])
    to_evict = sorted_cache_lines[-1] if sorted_cache_lines else None
    return to_evict, scores


class MixturePolicy(EvictionPolicy):
  """Follows different policies at each timestep.

  Given
    - N eviction policies: pi_1, ..., pi_N  and
    - N weights: p_1, ..., p_N with sum(p) = 1

  At each timestep:
    Acts according to policy pi_i with probability p_i.
  """

  def __init__(self, policies, weights=None, seed=0, scoring_policy_index=None):
    """Constructs.

    Args:
      policies (list[EvictionPolicy]): policies pi_1, ..., pi_N.
      weights (list[float] | None): weights p_1, ..., p_N. Defaults to uniform.
      seed (int): random seed for choosing between the policies.
      scoring_policy_index (int | None): policy always returns the scores
        according to policies[scoring_policy_index], even if that policy was not
        used to choose a cache line to evict. If None, uses the same policy for
        choosing a line to evict and scores.
    """
    super().__init__()

    if weights is None:
      weights = [1. / len(policies)] * len(policies)

    if len(policies) != len(weights):
      raise ValueError("Need the same number of weights ({}) as policies ({})"
                       .format(len(policies), len(weights)))

    if not np.isclose(sum(weights), 1.):
      raise ValueError("Weights must sum to 1 ({}).".format(sum(weights)))

    if scoring_policy_index is not None and not (
        0 <= scoring_policy_index < len(policies)):
      raise ValueError(
          "Invalid scoring policy index: {}".format(scoring_policy_index))

    self._policies = policies
    self._weights = weights
    self._random = np.random.RandomState(seed)
    self._scoring_policy_index = scoring_policy_index

  def __call__(self, cache_access, access_times):
    policy = self._random.choice(self._policies, p=self._weights)
    line_to_evict, scores = policy(cache_access, access_times)
    if self._scoring_policy_index is not None:
      scoring_policy = self._policies[self._scoring_policy_index]
      _, scores = scoring_policy(cache_access, access_times)
    return line_to_evict, scores


class RandomPolicy(EvictionPolicy):
  """Selects a cache-line uniformly at random."""

  def __init__(self, seed=0):
    """Constructs with a particular RNG seed.

    Args:
      seed (int): seed used to make random cache line choices.
    """
    super().__init__()

    self._random = np.random.RandomState(seed)

  def __call__(self, cache_access, access_times):
    del access_times

    # All scores the same.
    scores = {line: 0 for (line, _) in cache_access.cache_lines}
    if not scores:
      return None, scores

    selected = cache_access.cache_lines[self._random.randint(len(scores))][0]
    return selected, scores


class CacheLineScorer(six.with_metaclass(abc.ABCMeta, object)):
  """Scores cache lines based on how evictable each line is."""

  @abc.abstractmethod
  def __call__(self, cache_access, access_times):
    """Scores all the cache lines in the cache_access.

    Args:
      cache_access (CacheAccess): the cache access whose lines to score.
      access_times (dict): maps int address to last time of access. Smaller
        times were accessed longer ago.

    Returns:
      scores (dict{int: int}): maps each cache line (int) to its score (int).
        Lower score indicates more evictable.
    """
    raise NotImplementedError()


class LRUScorer(CacheLineScorer):
  """Returns scores equal to how recently each line was used.

  Specifically, the scores are just the access times.
  """

  def __call__(self, cache_access, access_times):
    scores = {line: access_times[line]
              for (line, _) in cache_access.cache_lines}
    return scores


class BeladyScorer(CacheLineScorer):
  """Returns scores equal to soon in the future each line is used.

  Specifically, the scores are the negative number of timesteps into the future
  each line is used, according to the provided memtrace.
  """

  def __init__(self, memtrace):
    """Constructs from the currently running memory trace.

    Args:
      memtrace (MemoryTrace): memory trace that cache is currently running on.
        The trace parameters affect how approximately the most future address is
        computed.
    """
    super(BeladyScorer, self).__init__()
    self._memtrace = memtrace

  def __call__(self, cache_access, access_times):
    del access_times

    scores = {line: -self._memtrace.next_access_time(line) for
              (line, _) in cache_access.cache_lines}
    return scores
