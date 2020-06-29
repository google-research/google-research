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
"""Defines a simple cache."""

import numpy as np


class CacheSet(object):
  """A set of cache lines in cache. Evicts according to the eviction policy."""

  def __init__(self, set_id, capacity):
    """Constructs.

    Args:
      set_id (int): ID of this set (value of set bits corresponding to this
        set).
      capacity (int): Maximum number of cache lines in the set.
    """
    self._set_id = set_id
    self._cache_lines = set()  # set of lines currently in the cache set.
    self._capacity = capacity

  def access(self, address):
    """Returns whether a line must be evicted from the cache set.

    Adds the line into the set upon a miss, if there is remaining capacity.

    Args:
      address (int): cache line-aligned memory address to access.

    Returns:
      hit (bool): True if the access is a cache hit
      evict (bool): True if a line must be evicted from the cache set.
      False on cache hits, and when the set is not yet at capacity.
    """
    hit = address in self._cache_lines
    if len(self._cache_lines) < self._capacity:
      self._cache_lines.add(address)
    return hit, address not in self._cache_lines

  def evict(self, line):
    """Removes a line from the cache set.

    Args:
      line (int): cache line-aligned memory address to remove.

    Raises:
      KeyError: If the line does not exist.
    """
    self._cache_lines.remove(line)

  def contents(self):
    """Returns the cache lines (set[int]) currently in the set."""
    return self._cache_lines

  def __str__(self):
    cache_lines = [str(hex(line)) for line in self._cache_lines]
    cache_lines += ["empty"] * (self._capacity - len(self._cache_lines))

    # Pad all elements to be same length for pretty printing
    pad_len = max(len(x) for x in cache_lines)
    cache_lines = [x.center(pad_len) for x in cache_lines]
    return " | ".join(cache_lines)


class Cache(object):
  """A hierarchical cache. Reads from child cache if data not present."""

  @classmethod
  def from_config(cls, config):
    """Constructs Cache from config.

    Args:
      config (Config): how the Cache is to be configured.

    Returns:
      Cache
    """
    return cls(config.get("capacity"), config.get("associativity"),
               config.get("cache_line_size"))

  def __init__(self, cache_capacity, associativity,
               cache_line_size=64, child_cache=None):
    """Constructs a hierarchical set-associative cache.

    Memory address is divided into:
      | ... | set_bits | cache_line_bits |

    Args:
      cache_capacity (int): number of bytes to store in cache.
      associativity (int): number of cache lines per set.
      cache_line_size (int): number of bytes per cache line.
      child_cache (Cache | None): cache to access on accesses, if data is not
        present in current cache.
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

    self._sets = [CacheSet(set_id, associativity) for set_id in range(num_sets)]
    self._set_bits = set_bits
    self._cache_line_bits = cache_line_bits
    self._child_cache = child_cache

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

  def access(self, address):
    """Attempts to access the address from the appropriate cache set.

    Returns whether a line must be evicted or not from this access, and the
    contents of the set.

    Args:
      address (int): memory address to access from the cache.

    Returns:
      aligned_address (int): cache-aligned version of the address.
      hit (bool): True if it is a cache hit at this level.
      evict (list[bool]): evict[0] corresponds to the current cache, and
        evict[1:] correspond to the child, grandchild, etc. caches.  An element
        is True if a line must be evicted from the cache. False on cache hits,
        and when the appropriate cache set is not yet at capacity.
      lines (list[list[int]]): lines currently present in the accessed cache
        sets.  The i-th element in the outer list corresponds to the i-th level
        grandchild.
      set_ids (list[int]): the set id that was accessed. The i-th element in the
        outer list corresponds to the i-th level grandchild.
    """
    aligned_address, set_id = self._align_address(address)
    hit, evict = self._sets[set_id].access(aligned_address)
    evicts = [evict]
    lines = [list(self._sets[set_id].contents())]
    set_ids = [set_id]
    if evict and self._child_cache is not None:
      _, _, children_evicts, children_lines, children_set_ids = (
          self._child_cache.access(address))
      evicts += children_evicts
      lines += children_lines
      set_ids += children_set_ids
    return aligned_address, hit, evicts, lines, set_ids

  def cache(self, line, set_id):
    """Adds the cache line to the given set_id if it's not already present.

    Args:
      line (int): cache-aligned memory address to add.
      set_id (int): identifies the cache set to add to.
    """
    _, evict = self._sets[set_id].access(line)
    assert not evict

  def evict(self, line, set_id):
    """Removes the cache line from the given set_id.

    Args:
      line (int): cache-aligned memory address to remove.
      set_id (int): identifies the cache set to remove from.

    Raises:
      KeyError: If the line is not present in the given set_id.
    """
    self._sets[set_id].evict(line)

  def __str__(self):
    s = []
    formatter = "{{:0{}b}}".format(self._set_bits)
    for set_id, cache_set in enumerate(self._sets):
      s.append(formatter.format(set_id))
      s.append(": {}\n".format(cache_set))
    return "".join(s)
