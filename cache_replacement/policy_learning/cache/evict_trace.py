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

"""Represents a trace of evictions from cache sets."""

import collections
import json
from cache_replacement.policy_learning.cache import cache


class EvictionEntry(collections.namedtuple(
    "EvictionEntry", ("cache_access", "eviction_decision"))):
  """A cache access combined with information about which line was evicted.

  Consists of:
    cache_access (cache.CacheAccess): an access to a cache. The cache lines
      (cache_access.cache_lines) are guaranteed to be ordered from most
      evictable to least evictable according to
      eviction_decision.cache_line_scores.
    eviction_decision (cache.EvictionDecision): information about which cache
      line (if any) was evicted during the cache access.
  """
  __slots__ = ()

  def __new__(cls, cache_access, eviction_decision):
    cache_access = cache_access._replace(
        cache_lines=eviction_decision.rank_cache_lines(
            cache_access.cache_lines))
    return super(EvictionEntry, cls).__new__(
        cls, cache_access, eviction_decision)


class EvictionTrace(object):
  """Ordered set of accesses with information about which lines were evicted.

  Serialization details:
    Files are written as JSON lines format, where each line is a separate JSON
    object with representing a single entry. Everything is written in hex except
    set_id is represented as a binary string for maximal human readability.
  """

  def __init__(self, filename, read_only=True):
    """Reads / writes cache access + eviction info from file.

    Args:
      filename (str): path to eviction trace file to read / write.
      read_only (bool): the trace is either read only or write only. If it is
        read only, it reads from the provided filename via the read method.
        Otherwise, it is write only and writes to the provided filename via the
        write method.
    """
    self._filename = filename
    self._read_only = read_only

  def read(self):
    """Returns the next memory access and eviction information.

    Raises StopIteration when file is exhausted. See class docstring for details
    on returned values.

    Returns:
      EvictionEntry
    """
    entry = json.loads(next(self._file))
    pc = int(entry["pc"], 16)
    address = int(entry["address"], 16)
    set_id = int(entry["set_id"], 2)
    cache_lines = [
        (int(line, 16), int(pc, 16)) for line, pc in entry["cache_lines"]]
    access_history = [(int(address, 16), int(pc, 16))
                      for address, pc in entry["access_history"]]
    cache_access = cache.CacheAccess(
        pc, address, set_id, cache_lines, access_history)

    evict = entry["evict"]
    cache_line_scores = dict(entry["cache_line_scores"])
    eviction_decision = cache.EvictionDecision(evict, cache_line_scores)
    return EvictionEntry(cache_access, eviction_decision)

  def write(self, entry):
    """Writes an eviction entry to file.

    Args:
      entry (EvictionEntry): entry generated from a memory access.
    """
    json.dump({
        "pc": hex(entry.cache_access.pc),
        "address": hex(entry.cache_access.address),
        "set_id": bin(entry.cache_access.set_id),
        "cache_lines": [(hex(line), hex(pc))
                        for line, pc in entry.cache_access.cache_lines],
        "access_history": [
            (hex(address), hex(pc))
            for address, pc in entry.cache_access.access_history],
        "evict": entry.eviction_decision.evict,
        # Serialize items instead of dict to prevent json from converting int
        # keys to string
        "cache_line_scores":
            list(entry.eviction_decision.cache_line_scores.items()),
    }, self._file)
    self._file.write("\n")

  def __enter__(self):
    if self._read_only:
      self._file = open(self._filename, "r")
    else:
      self._file = open(self._filename, "w")
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()
