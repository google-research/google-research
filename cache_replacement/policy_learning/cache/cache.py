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

"""Defines a simple cache."""

import collections
import numpy as np
from cache_replacement.policy_learning.cache import eviction_policy as eviction_policy_mod
from cache_replacement.policy_learning.cache_model import eviction_policy as model_eviction_policy_mod
from cache_replacement.policy_learning.common import config as cfg


class CacheSet(object):
  """A set of cache lines in cache. Evicts according to the eviction policy."""

  def __init__(self, set_id, num_cache_lines, eviction_policy,
               access_history_len=30):
    """Constructs.

    Args:
      set_id (int): ID of this set (value of set bits corresponding to this
        set).
      num_cache_lines (int): Number of cache lines in the set.
      eviction_policy (EvictionPolicy): determines what cache lines to evict
        during reads.
      access_history_len (int): returns an access history of this length in the
        CacheAccess for observers of read.
    """
    self._set_id = set_id
    self._cache_lines = {}  # maps address in cache to pc of that access
    self._access_times = {}  # maps address in cache --> time of last access
    self._num_cache_lines = num_cache_lines
    self._eviction_policy = eviction_policy
    self._access_history = collections.deque(maxlen=access_history_len)
    self._read_counter = 0  # Used to order access times

  def set_eviction_policy(self, eviction_policy):
    """Changes the eviction policy to be the passed one.

    Args:
      eviction_policy (EvictionPolicy): the new eviction policy to use.
    """
    self._eviction_policy = eviction_policy

  def read(self, pc, address, observers=None):
    """Adds data at the address in the set. Returns hit / miss.

    Args:
      pc (int): see documentation in Cache.read.
      address (int): cache line-aligned memory address to add.
      observers (list[Callable] | None): see documentation in Cache.read.

    Returns:
      hit (bool): True if data was already in working set.
    """
    # Pushed into an inner function to easily extract observer arguments.
    def update_cache_set(cache_access):
      """Performs all book-keeping for adding address to cache set.

      Args:
        cache_access (CacheAccess): the memory access to update on.

      Returns:
        hit (bool): True if address was in the set.
        eviction_decision (EvictionDecision): decision of the eviction policy.
      """
      self._access_history.append((cache_access.address, cache_access.pc))

      # Receive cache_line_scores for observers, even if cache line is evicted.
      # Needs to happen BEFORE the access_times is updated.
      line_to_evict, cache_line_scores = self._eviction_policy(
          cache_access, self._access_times)

      self._read_counter += 1
      self._access_times[address] = self._read_counter

      if address in self._cache_lines:
        # update the PC associated with the cache line
        self._cache_lines[address] = pc
        return True, EvictionDecision(False, cache_line_scores)

      evict = len(self._cache_lines) == self._num_cache_lines
      eviction_decision = EvictionDecision(evict, cache_line_scores)
      if evict:
        del self._cache_lines[line_to_evict]
      self._cache_lines[address] = pc
      return False, eviction_decision

    if observers is None:
      observers = []

    # Create CacheAccess with copies of cache_lines and access_history before
    # they get updated.
    cache_access = CacheAccess(
        pc, address, self._set_id, list(self._cache_lines.items()),
        list(self._access_history))
    hit, eviction_decision = update_cache_set(cache_access)

    for observer in observers:
      observer(cache_access, eviction_decision)
    return hit

  def __str__(self):
    cache_lines = [str((hex(line), hex(pc)))
                   for line, pc in self._cache_lines.items()]
    cache_lines += ["empty"] * (self._num_cache_lines - len(self._cache_lines))

    # Pad all elements to be same length for pretty printing
    pad_len = max(len(x) for x in cache_lines)
    cache_lines = [x.center(pad_len) for x in cache_lines]
    return " | ".join(cache_lines)


class Cache(object):
  """A hierarchical cache. Reads from child cache if data not present."""

  @classmethod
  def from_config(cls, config, eviction_policy=None, trace=None,
                  hit_rate_statistic=None):
    """Constructs Cache from config.

    Args:
      config (Config): how the Cache is to be configured.
      eviction_policy (EvictionPolicy | None): the eviction policy to use.
        Constructs an EvictionPolicy from the config if None.
      trace (MemoryTrace | None): the trace that the Cache is going to be
        simulated on. Only needs to be specified if the eviction_policy is None.
      hit_rate_statistic (BernoulliTrialStatistic | None): see constructor.

    Returns:
      Cache
    """
    def eviction_policy_from_config(config, trace):
      """Returns the EvictionPolicy specified by the config.

      Args:
        config (Config): config for the eviction policy.
        trace (MemoryTrace): memory trace to simulate on.

      Returns:
        EvictionPolicy
      """
      def scorer_from_config(config, trace):
        """Creates an eviction_policy.CacheLineScorer from the config.

        Args:
          config (Config): config for the cache line scorer.
          trace (MemoryTrace): see get_eviction_policy.

        Returns:
          CacheLineScorer
        """
        scorer_type = config.get("type")
        if scorer_type == "lru":
          return eviction_policy_mod.LRUScorer()
        elif scorer_type == "belady":
          return eviction_policy_mod.BeladyScorer(trace)
        elif scorer_type == "learned":
          with open(config.get("config_path"), "r") as model_config:
            return (model_eviction_policy_mod.LearnedScorer.
                    from_model_checkpoint(cfg.Config.from_file(model_config),
                                          config.get("checkpoint")))
        else:
          raise ValueError("Invalid scorer type: {}".format(scorer_type))

      policy_type = config.get("policy_type")
      if policy_type == "greedy":
        scorer = scorer_from_config(config.get("scorer"), trace)
        return eviction_policy_mod.GreedyEvictionPolicy(
            scorer, config.get("n", 0))
      elif policy_type == "random":
        return eviction_policy_mod.RandomPolicy()
      elif policy_type == "mixture":
        subpolicies = [eviction_policy_from_config(subconfig, trace) for
                       subconfig in config.get("subpolicies")]
        return eviction_policy_mod.MixturePolicy(
            subpolicies, config.get("weights"))
      else:
        raise ValueError("Invalid policy type: {}".format(policy_type))

    if eviction_policy is None:
      eviction_policy = eviction_policy_from_config(
          config.get("eviction_policy"), trace)
    return cls(config.get("capacity"), eviction_policy,
               config.get("associativity"), config.get("cache_line_size"),
               hit_rate_statistic=hit_rate_statistic)

  def __init__(self, cache_capacity, eviction_policy, associativity,
               cache_line_size=64, child_cache=None, hit_rate_statistic=None,
               access_history_len=30):
    """Constructs a hierarchical set-associative cache.

    Memory address is divided into:
      | ... | set_bits | cache_line_bits |

    Args:
      cache_capacity (int): number of bytes to store in cache.
      eviction_policy (EvictionPolicy): determines which cache lines to evict
        when necessary.
      associativity (int): number of cache lines per set.
      cache_line_size (int): number of bytes per cache line.
      child_cache (Cache | None): cache to access on reads, if data is not
        present in current cache.
      hit_rate_statistic (BernoulliTrialStatistic | None): logs cache hits /
        misses to this if provided. Defaults to vanilla
        BernoulliProcessStatistic if not provided.
      access_history_len (int): see CacheSet.
    """
    def is_pow_of_two(x):
      return (x & (x - 1)) == 0

    if not is_pow_of_two(cache_line_size):
      raise ValueError("Cache line size ({}) must be a power of two."
                       .format(cache_line_size))

    num_cache_lines = cache_capacity // cache_line_size
    num_sets = num_cache_lines // associativity

    if (cache_capacity % cache_line_size != 0 or
        num_cache_lines % associativity != 0):
      raise ValueError(
          ("Cache capacity ({}) must be an even multiple of "
           "cache_line_size ({}) and associativity ({})").format(
               cache_capacity, cache_line_size, associativity))

    if not is_pow_of_two(num_sets):
      raise ValueError("Number of cache sets ({}) must be a power of two."
                       .format(num_sets))

    if num_sets == 0:
      raise ValueError(
          ("Cache capacity ({}) is not great enough for {} cache lines per set "
           "and cache lines of size {}").format(cache_capacity, associativity,
                                                cache_line_size))

    set_bits = int(np.log2(num_sets))
    cache_line_bits = int(np.log2(cache_line_size))

    self._sets = [
        CacheSet(set_id, associativity, eviction_policy, access_history_len)
        for set_id in range(num_sets)
    ]
    self._set_bits = set_bits
    self._cache_line_bits = cache_line_bits
    self._child_cache = child_cache

    if hit_rate_statistic is None:
      hit_rate_statistic = BernoulliProcessStatistic()
    self._hit_rate_statistic = hit_rate_statistic

  def _align_address(self, address):
    """Returns the cache line aligned address and the corresponding set id.

    Args:
      address (int): a memory address.

    Returns:
      aligned_address (int): aligned with the size of the cache lines.
      set_id (int): the set this cache-line belongs to.
    """
    aligned_address = address >> self._cache_line_bits
    set_id = aligned_address & ((1 << self._set_bits) - 1)
    return aligned_address, set_id

  def read(self, pc, address, observers=None):
    """Adds data at address to cache. Logs hit / miss to hit_rate_statistic.

    Args:
      pc (int): program counter of the memory access.
      address (int): memory address to add to the cache.
      observers (list[Callable] | None): each observer is called with:
        - cache_access (CacheAccess): information about the current cache
            access.
        - eviction_decision (EvictionDecision): information about what cache
            line was evicted.
        observers are not called on reads in child caches.

    Returns:
      hit (bool): True if data was already in the cache.
    """
    aligned_address, set_id = self._align_address(address)
    hit = self._sets[set_id].read(pc, aligned_address, observers=observers)
    if not hit and self._child_cache is not None:
      self._child_cache.read(pc, address)
    self._hit_rate_statistic.trial(hit)
    return hit

  @property
  def hit_rate_statistic(self):
    """Returns the hit_rate_statistic provided to the constructor.

    Returns:
      BernoulliProcessStatistic
    """
    return self._hit_rate_statistic

  def set_eviction_policy(self, eviction_policy):
    """Changes the eviction policy to be the passed one.

    Args:
      eviction_policy (EvictionPolicy): the new eviction policy to use.
    """
    for cache_set in self._sets:
      cache_set.set_eviction_policy(eviction_policy)

  def __str__(self):
    s = []
    formatter = "{{:0{}b}}".format(self._set_bits)
    for set_id, cache_set in enumerate(self._sets):
      s.append(formatter.format(set_id))
      s.append(": {}\n".format(cache_set))
    return "".join(s)


class BernoulliProcessStatistic(object):
  """Tracks results of Bernoulli trials."""

  def __init__(self):
    self.reset()

  def trial(self, success):
    self._trials += 1
    if success:
      self._successes += 1

  @property
  def num_trials(self):
    return self._trials

  @property
  def num_successes(self):
    return self._successes

  def success_rate(self):
    if self.num_trials == 0:
      raise ValueError("Success rate is undefined when num_trials is 0.")
    return self.num_successes / self.num_trials

  def reset(self):
    self._successes = 0
    self._trials = 0


class CacheAccess(collections.namedtuple(
    "CacheAccess", ("pc", "address", "set_id", "cache_lines",
                    "access_history"))):
  """A single access to a cache set.

  Consists of:
    pc (int): the program counter of the memory access instruction.
    address (int): the cache-aligned memory address that was accessed.
    set_id (int): id of the cache set that was accessed.
    cache_lines (list[(int, int)]): list of (cache-aligned addresses, pc) in the
      cache set at the time of the access, where the pc is the program counter
      of the memory access of the address in the cache.
    access_history (list[(int, int)]): list of (cache-aligned address, pc) of
      past accesses to this set, ordered from most recent to oldest.
  """
  pass


class EvictionDecision(collections.namedtuple(
    "EvictionDecision", ("evict", "cache_line_scores"))):
  """Information about which cache line was evicted for a CacheAccess.

  Consists of:
    evict (bool): True if a cache line was evicted.
    cache_line_scores (dict): maps a cache line (int) to its score (int) as
      determined by an EvictionPolicy. Lower score --> more evictable.
  """

  def rank_cache_lines(self, cache_lines):
    """Returns the cache lines sorted by most evictable to least evictable.

    Args:
      cache_lines (list[tuple]): the cache lines (address, pc) [(int, int)] that
        this eviction decision was made about.

    Returns:
      ranked_cached_lines (list[int]): the cache lines sorted.
    """
    return sorted(cache_lines,
                  key=lambda cache_line: self.cache_line_scores[cache_line[0]])
