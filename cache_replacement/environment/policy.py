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
"""Simple cache replacement baselines."""
import abc
import collections


class ReplacementPolicy(abc.ABC):
  """A policy that takes cache states and produces eviction decisions."""

  @abc.abstractmethod
  def action(self, state):
    """Given a cache state, returns an eviction decision or no eviction.

    Args:
      state (State): returned by a CacheReplacementEnv.

    Returns:
      eviction_decision (int): index of the line to evict, or -1 for no
        eviction.
    """


class LRU(ReplacementPolicy):
  """Evicts the least-recently used line."""

  def __init__(self):
    self._time = 0
    self._access_times = collections.defaultdict()

  def action(self, state):
    address = state.access[0]
    self._access_times[address] = self._time
    self._time += 1
    if not state.evict:
      return -1

    return min((self._access_times[line], index)
               for index, line in enumerate(state.cache_lines))[1]


class RandomPolicy(ReplacementPolicy):
  """Evicts randomly when the cache is full (Belady_0)."""

  def __init__(self, random_state):
    """Constructs around a random state to query for decisions.

    Args:
      random_state (np.RandomState): used to generate random decisions.
    """
    self._random_state = random_state

  def action(self, state):
    if not state.evict:
      return -1

    return self._random_state.randint(len(state.cache_lines))
