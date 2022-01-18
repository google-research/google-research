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

# Lint as: python3
"""Replacement algorithms based on Belady's."""
import collections
import policy
import tqdm


class BeladyPolicy(policy.ReplacementPolicy):
  """Evicts the line with the greateste reuse distance."""

  def __init__(self, env):
    """Constructs around the environment to run on.

    Args:
      env (CacheReplacementEnv): the environment whose step method will receive
        actions from this policy.
    """
    self._env = env

  def action(self, state):
    if not state.evict:
      return -1

    reuse_distances = {
        line: self._env.next_access_time(line) for line in state.cache_lines}
    line_to_evict = max((reuse_distance, line)
                        for line, reuse_distance in reuse_distances.items())[1]
    return state.cache_lines.index(line_to_evict)


class BeladyNearestNeighborsPolicy(policy.ReplacementPolicy):
  """Policy that chooses actions based on the nearest-neighbor in training.

  Concretely, given a training trace, the policy follows this algorithm:
    1) The policy finds the longest matching suffix of (pc, address) accesses in
       the train trace.
    2) The policy computes the reuse distances (with respect to the train trace)
       of each of the cache lines in the train trace at the matching suffix.
    3) If the line with the maximal reuse distance is also in the current cache
       set, the policy evicts that line.
    4) Otherwise, the policy evicts LRU.
  """

  def __init__(self, train_env):
    """Constructs around the training trace.

    Args:
      train_env (CacheReplacementEnv): environment wrapping the training trace.
        The configs of the environment should be the same as those used at test
        time.
    """
    # set_id --> list of (state, ranked lines)
    self._train_accesses = collections.defaultdict(list)
    # Optimization: (set_id, address) --> list index in self._train_accesses
    # where self._train_accesses[set_id][index] = address
    self._address2index = collections.defaultdict(list)
    state = train_env.reset()
    with tqdm.tqdm(desc='"Training"') as pbar:
      while True:
        reuse_distances = {
            line: train_env.next_access_time(line)
            for line in state.cache_lines}

        # Ranked highest reuse distance to lowest reuse distance
        ranked_lines = sorted(
            state.cache_lines, key=lambda line: reuse_distances[line],
            reverse=True)
        self._address2index[(state.set_id, state.access)].append(
            len(self._train_accesses[state.set_id]))
        self._train_accesses[state.set_id].append((state, ranked_lines))
        line_to_evict = ranked_lines[0]
        action = state.cache_lines.index(line_to_evict) if state.evict else -1

        state, _, done, _ = train_env.step(action)
        pbar.update(1)
        if done:
          break

    # set_id --> history of State
    self._test_access_history = collections.defaultdict(list)
    # Dict for memoizing: (set_id, index, index) --> matching suffix length
    self._suffix_cache = {}
    # Fall-back policy
    self._lru = policy.LRU()

  def action(self, state):
    access_history = self._test_access_history[state.set_id]
    access_history.append(state)

    def suffix_len(train_index):
      # Returns the length of the longest matching suffix between:
      # - current access history (access_history at last index)
      # - training data (self._train_accesses at index train_index)
      # Dynamic programming relationship:
      # suffix length at (train_index, last_index) =
      #   1 + suffix length at (train_index - 1, last_index - 1)
      prev_key = (state.set_id, train_index - 1, len(access_history) - 1)
      length = self._suffix_cache.get(prev_key, 0) + 1

      key = (state.set_id, train_index, len(access_history))
      self._suffix_cache[key] = length
      return length

    # Find longest matching suffix
    # Choose arbitrary default, since we're going to back off to random anyway
    longest_suffix_index = max(
        self._address2index[(state.set_id, state.access)], key=suffix_len,
        default=0)

    # Need to call LRU policy whether or not using the LRU action for access
    # time computations.
    lru_action = self._lru.action(state)
    if not state.evict:
      return -1

    train_decisions = (
        self._train_accesses[state.set_id][longest_suffix_index][1])
    try:
      return state.cache_lines.index(train_decisions[0])
    except ValueError:
      # Back off and evict LRU
      return lru_action
