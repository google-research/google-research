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

"""Defines cache replacement OpenAI gym environment.

See policy.py for LRU and Belady's policies.

Example usage:

import config as cfg
env = CacheReplacementEnv(
    cfg.Config.from_files_and_bindings(
      ["example_cache_config.json"], []), "example_memtrace.csv")

state = env.reset()
while True:
  action = next(iter(state.action_set))  # random action
  state, reward, done, info = env.step(action)
  env.render()
  if done:
    break
"""

import collections
import sys

import cache as cache_mod
import gym
import memtrace as memtrace_mod


class State(collections.namedtuple(
    "State", ("access", "cache_lines", "access_history", "set_id", "evict"))):
  """Represents the state s_t.

    - access ((int, int)): representing currently accessed
        s^a_t = (address_t, pc_).
    - cache_lines (list[int]): cache lines in the currently accessed cache set
        s^c_t = [l_0, ..., l_{W - 1}].
    - access_history (list[(int, int)]): past H accesses to the cache set
        s^h_t = [(address_{t - H}, pc_{t - H}), ..., (address_t, pc_t)].
    - set_id (int): id of the set that was accessed.
    - evict (bool): True if an eviction decision must be made.
  """

  @property
  def action_set(self):
    """Returns the set of available actions at this state.

    Returns:
      action_set (list[int]): the available actions at the next timestep.
        When evict is True, this consists of all of {0, ..., W - 1}, where
        action a corresponds to evicting line l_a.  When evict is False, this
        also includes -1, the no-op action.
    """
    actions = set(range(len(self.cache_lines)))
    if not self.evict:
      actions.add(-1)
    return actions


class CacheReplacementEnv(gym.Env):
  """Environment for cache replacement. Returns State objects."""

  def __init__(self, cache_config, memtrace_filename, access_history_len=None):
    """Constructs.

    Args:
      cache_config (Config): specifies how the underlying cache should be
        configured. See Cache.from_config.
      memtrace_filename (str): filename of the memory trace to simulate on.
      access_history_len (int | None): specifies the length of history returned
        in states. None means that the history length is untruncated.
    """
    super().__init__()
    self._cache_config = cache_config
    self._memtrace_filename = memtrace_filename
    self._access_history_len = access_history_len
    self._memtrace = None

  def reset(self):
    if self._memtrace:
      self._memtrace.__exit__(*sys.exc_info())
    self._memtrace = memtrace_mod.MemoryTrace(
        self._memtrace_filename,
        cache_line_size=self._cache_config.get("cache_line_size")).__enter__()
    self._cache = cache_mod.Cache.from_config(self._cache_config)
    self._set_access_history = collections.defaultdict(
        lambda: collections.deque(maxlen=self._access_history_len))

    pc, address = self._memtrace.next()
    aligned_address, _, evicts, lines, set_ids = self._cache.access(address)
    self._prev_state = State(
        access=(aligned_address, pc), cache_lines=lines[0], access_history=[],
        set_id=set_ids[0], evict=evicts[0])
    self._set_access_history[set_ids[0]].append(aligned_address)
    return self._prev_state

  def step(self, action):
    if self._prev_state is not None:
      if action not in self._prev_state.action_set:
        raise ValueError(
            "Permitted actions: {}. Taken action: {}"
            .format(self._prev_state.action_set, action))

      if action != -1:
        replaced_line = self._prev_state.cache_lines[action]
        new_line = self._prev_state.access[0]
        self._cache.evict(replaced_line, self._prev_state.set_id)
        self._cache.cache(new_line, self._prev_state.set_id)

    pc, address = self._memtrace.next()
    aligned_address, hit, evicts, lines, set_ids = self._cache.access(address)

    # Copy, so doesn't get modified in place
    access_history = list(self._set_access_history[set_ids[0]])
    state = State(
        access=(aligned_address, pc), cache_lines=lines[0],
        access_history=access_history, set_id=set_ids[0], evict=evicts[0])
    reward = int(hit)
    done = self._memtrace.done()

    self._set_access_history[set_ids[0]].append(aligned_address)
    self._prev_state = state
    return state, reward, done, {}

  def render(self, mode="human"):
    """Prints a text representation of the cache."""
    print(self._cache)

  def next_access_time(self, address):
    return self._memtrace.next_access_time(address)
