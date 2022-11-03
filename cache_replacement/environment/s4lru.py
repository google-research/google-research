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

"""S4LRU cache replacement algorithm."""
import collections
import policy


class SegmentedQueue(object):
  """Queue divided into segments according to S4LRU algorithm."""

  def __init__(self, cache_size, num_queues=4):
    assert cache_size % num_queues == 0
    self._max_queue_size = cache_size // num_queues
    self._queues = [collections.deque(maxlen=self._max_queue_size)
                    for _ in range(num_queues)]

  def _insert(self, line, level):
    """Handles trickling down for adding the line to the given level.

    Args:
      line (int): address of line to add to level-th queue
      level (int): specifies the level to add to.

    Returns:
      evicted_line (int | None): the line that is evicted from the 0-th level
        cache, if it exists.
    """
    if level == -1:
      return line

    queue = self._queues[level]
    if len(queue) == self._max_queue_size:
      removed = queue.pop()
      queue.appendleft(line)
      return self._insert(removed, level - 1)
    else:
      queue.appendleft(line)
      return None

  def add(self, line):
    """Adds the line into the queue and returns the evicted line.

    Args:
      line (int): address of new line to place in queue.

    Returns:
      address (int | None): address of line to evict, if it exists.
    """
    for level, queue in enumerate(self._queues):
      try:
        queue.remove(line)
      except ValueError:
        pass
      else:
        insertion_level = min(len(self._queues) - 1, level + 1)
        return self._insert(line, insertion_level)
    else:
      return self._insert(line, 0)

  def __str__(self):
    return "\n".join(
        f"level {level}: {queue}" for level, queue in enumerate(self._queues))


class S4LRU(policy.ReplacementPolicy):
  """Implements S4LRU algorithm.

  See http://www.cs.cornell.edu/~qhuang/papers/sosp_fbanalysis.pdf
  """

  def __init__(self, cache_set_size):
    # set_id (int) -> SegmentedQueue
    self._queues = collections.defaultdict(
        lambda: SegmentedQueue(cache_set_size))

  def action(self, state):
    address = state.access[0]
    removed_line = self._queues[state.set_id].add(address)
    if removed_line is None:
      return -1
    return state.cache_lines.index(removed_line)
