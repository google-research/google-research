# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import numpy as np


class NodeBank(object):

  def __init__(
      self,
      src,
      dst,
  ):
    r"""maintains a dictionary of all nodes seen so far (specified by the input src and dst)

    Parameters:
        src: source node id of the edges
        dst: destination node id of the edges
        ts: timestamp of the edges
    """
    self.nodebank = {}
    self.update_memory(src, dst)

  def update_memory(
      self, update_src, update_dst
  ):
    r"""update self.memory with newly arrived src and dst

    Parameters:
        src: source node id of the edges
        dst: destination node id of the edges
    """
    for src, dst in zip(update_src, update_dst):
      if src not in self.nodebank:
        self.nodebank[src] = 1
      if dst not in self.nodebank:
        self.nodebank[dst] = 1

  def query_node(self, node):
    r"""query if node is in the memory

    Parameters:
        node: node id to query
    Returns:
        True if node is in the memory, False otherwise
    """
    return node in self.nodebank
