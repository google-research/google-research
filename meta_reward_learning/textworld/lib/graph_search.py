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

"""Graph search algorithms for exploration."""

import random
# pylint: disable=g-import-not-at-top
try:
  import queue
except ImportError:
  import six.moves.queue as queue
# pylint: enable=g-import-not-at-top


def check_valid(graph, pos):
  y, x = pos
  y_max, x_max = graph.shape
  if ((0 <= y and y < y_max) and (0 <= x and x < x_max)) and (graph[pos] >= 0):
    return True
  else:
    return False


def bfs_paths(graph, agent, goal, num_actions, maxlen):
  """Find paths from any start position to a goal position using BFS."""
  path_queue = queue.Queue()
  path_queue.put((agent.pos, []))
  while not path_queue.empty():
    curr_pos, path = path_queue.get()
    if len(path) >= maxlen:
      continue
    for action in range(num_actions):
      agent.reset(curr_pos)
      agent.act(action)
      if check_valid(graph, agent.pos):
        new_path = path + [action]
        if agent.pos == goal:
          yield new_path
        else:
          path_queue.put((agent.pos, new_path))


def dfs_paths(graph, agent, goal, num_actions, maxlen):
  """"Find paths from any start position to a goal position using DFS."""
  stack = [(agent.pos, [])]
  all_actions = list(range(num_actions))
  while stack:
    curr_pos, path = stack.pop()
    if len(path) >= maxlen:
      continue
    random.shuffle(all_actions)
    for action in all_actions:
      agent.reset(curr_pos)
      agent.act(action)
      if check_valid(graph, agent.pos):
        new_path = path + [action]
        if agent.pos == goal:
          yield new_path
        else:
          stack.append((agent.pos, new_path))
