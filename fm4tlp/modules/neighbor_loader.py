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

"""Neighbor loader for storing graphs.
"""
from typing import Tuple

import torch


class LastNeighborLoader:

  def __init__(self, num_nodes, size, device=None):
    self.size = size

    self.neighbors = torch.empty(
        (num_nodes, size), dtype=torch.long, device=device
    )
    self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
    self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    self.reset_state()

  def __call__(
      self, n_id
  ):
    neighbors = self.neighbors[n_id]
    nodes = n_id.view(-1, 1).repeat(1, self.size)
    e_id = self.e_id[n_id]

    # Filter invalid neighbors (identified by `e_id < 0`).
    mask = e_id >= 0
    neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

    # Relabel node indices.
    n_id = torch.cat([n_id, neighbors]).unique()
    self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
    neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

    return n_id, torch.stack([neighbors, nodes]), e_id

  def insert(self, src, dst):
    # Inserts newly encountered interactions into an ever-growing
    # (undirected) temporal graph.

    # Collect central nodes, their neighbors and the current event ids.
    neighbors = torch.cat([src, dst], dim=0)
    nodes = torch.cat([dst, src], dim=0)
    e_id = torch.arange(
        self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
    ).repeat(2)
    self.cur_e_id += src.numel()

    # Convert newly encountered interaction ids so that they point to
    # locations of a "dense" format of shape [num_nodes, size].
    nodes, perm = nodes.sort()
    neighbors, e_id = neighbors[perm], e_id[perm]

    n_id = nodes.unique()
    self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

    dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
    dense_id += self._assoc[nodes].mul_(self.size)

    dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
    dense_e_id[dense_id] = e_id
    dense_e_id = dense_e_id.view(-1, self.size)

    dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
    dense_neighbors[dense_id] = neighbors
    dense_neighbors = dense_neighbors.view(-1, self.size)

    # Collect new and old interactions...
    e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
    neighbors = torch.cat(
        [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
    )

    # And sort them based on `e_id`.
    e_id, perm = e_id.topk(self.size, dim=-1)
    self.e_id[n_id] = e_id
    self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

  def reset_state(self):
    self.cur_e_id = 0
    self.e_id.fill_(-1)
