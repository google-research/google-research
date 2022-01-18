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

import copy
import numpy as np
import heapq
from collections import namedtuple
from gtd.log import indent


class Entry(list):
  """Wrapper around priority entry with custom < operator for heappush."""

  def __lt__(self, other):
    if isinstance(other, Entry):
      return (self[0], id(self[1]), self[2]) < (other[0], id(
          other[1]), other[2])
    raise NotImplementedError


class PathPrioritizer(object):
  """Priority queue of paths."""

  def __init__(self, priority_fn):
    """
        Args:
            priority_fn (Callable): takes a path and returns a priority (float).
    """
    self._queue = []

    # Map from _path_id to (priority, path, in_queue) in queue
    # in_queue is bool: True if it is present in the queue
    self._enqueued_paths = {}
    self._priority_fn = priority_fn

  def __len__(self):
    # Only counts non-dead paths
    return sum(
        len(entry[1]) == 0 or not entry[1][-1].dead
        for entry in self._enqueued_paths.values())

  def next_path(self):
    """Returns the lowest priority path (does not remove it from queue).

        Returns:
            list[DirectedEdge]
        """
    while True:
      priority, path, in_queue = self._queue[0]

      if in_queue:
        return copy.copy(path)
      else:
        # Officially remove marked path
        heapq.heappop(self._queue)

  def pop(self):
    """Pops the lowest priority path off the queue and returns it.

        Returns:
            list[DirectedEdge]
        """
    # Remove any marked paths
    self.next_path()
    priority, path, in_queue = heapq.heappop(self._queue)
    assert in_queue
    del self._enqueued_paths[self._path_id(path)]
    # Does not need to be a copy because it doesn't live in queue anymore.
    return path

  def add_path(self, path):
    """Adds the path to the queue with the given priority.

    If the path is
        already in the queue, updates its priority.

        Args: path (list[DirectedEdge])
    """
    priority = self._priority_fn(path)
    path_id = self._path_id(path)
    if self._path_id(path) not in self._enqueued_paths:
      entry = Entry([priority, copy.copy(path), True])  # in queue
      self._enqueued_paths[path_id] = entry
      heapq.heappush(self._queue, entry)
    else:
      entry = self._enqueued_paths[path_id]
      if entry[0] != priority:
        entry[-1] = False  # Mark it as removed

        # Use the already copied path from the old entry
        new_entry = Entry([priority, entry[1], True])
        self._enqueued_paths[path_id] = new_entry
        heapq.heappush(self._queue, new_entry)

  def remove_path(self, path):
    path_id = self._path_id(path)
    entry = self._enqueued_paths[path_id]
    entry[-1] = False  # Mark it as removed
    del self._enqueued_paths[path_id]

  def _path_id(self, path):
    if len(path) == 0:
      return ()
    else:
      return path[-1]

  def __str__(self):
    MAX_LEN = int(1e3)

    s = "Priorities (lower is better): \n"
    dead_s = "Dead paths: \n"
    for i, (priority, path,
            in_queue) in enumerate(heapq.nsmallest(MAX_LEN, self._queue)):
      if in_queue:
        line = "{}) ".format(i)
        for edge in path:
          line += "{} --> {}, ".format(edge.start.uid, edge.end.uid)
        dead = len(path) > 0 and path[-1].dead
        line += "[{}] ({})\n".format(priority, "DEAD" if dead else "FINE")
        if dead:
          dead_s += line
        else:
          s += line
    return s + "\n" + dead_s


class MultiPriorityQueue(PathPrioritizer):
  """Maintains multiple priority queues and uniformly samples between them

    for the next path.
    """

  def __init__(self, priority_fns):
    """
        Args:
            priority_fns (list[Callable]): list of priority functions, each for
              a different queue
    """
    # Invariant: all queues always have exact same elements
    self._queues = [PathPrioritizer(fn) for fn in priority_fns]

  def __len__(self):
    return len(self._queues[0])

  def next_path(self):
    # Uniformly sample a queue and return its next_path
    queue_index = np.random.randint(len(self._queues))
    return self._queues[queue_index].next_path()

  def pop(self):
    # Can't know which queue to pop
    # Could keep a history if necessary
    raise ValueError("Popping a single queue violates equal queue invariant")

  def add_path(self, path):
    for queue in self._queues:
      queue.add_path(path)

  def remove_path(self, path):
    for queue in self._queues:
      queue.remove_path(path)

  def __str__(self):
    s = ""
    for i, queue in enumerate(self._queues):
      s += "Queue {} (size={})\n".format(i, len(queue))
      s += indent(str(queue))
      s += "\n"
    return s
