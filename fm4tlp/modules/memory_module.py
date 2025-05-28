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

"""Memory Module

Reference:
    -
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import copy
from typing import Dict, Tuple

import torch
from torch import nn
import torch_geometric

from fm4tlp.modules import message_agg
from fm4tlp.modules import message_func
from fm4tlp.modules import time_enc


TGNMessageStoreType = Dict[
    int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


_MESSAGE_MODULE_TYPE = message_agg.LastAggregator | message_agg.MeanAggregator


class TGNMemory(nn.Module):
  r"""The Temporal Graph Network (TGN) memory model.

  See `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
  <https://arxiv.org/abs/2006.10637>`_ paper.

  .. note:

      For an example of using TGN, see `examples/tgn.py
      <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
      tgn.py>`_.
  """

  def __init__(
      self,
      num_nodes,
      raw_msg_dim,
      memory_dim,
      time_dim,
      message_module,
      aggregator_module,
      memory_updater_cell = "gru",
  ):
    """Initializes the TGN memory module.

    Args:
      num_nodes: The number of nodes to save memories for.
      raw_msg_dim: The raw message dimensionality.
      memory_dim: The hidden memory dimensionality.
      time_dim: The time encoding dimensionality.
      message_module: The message function which combines source and desti-
        nation node memory embeddings, the raw message and the time encoding.
      aggregator_module: The message aggregator function which aggregates
        messages to the same destination into a single representation.
    """
    super().__init__()

    self.num_nodes = num_nodes
    self.raw_msg_dim = raw_msg_dim
    self.memory_dim = memory_dim
    self.time_dim = time_dim

    self.msg_s_module = message_module
    self.msg_d_module = copy.deepcopy(message_module)
    self.aggr_module = aggregator_module
    self.time_enc = time_enc.TimeEncoder(time_dim)
    # self.gru = GRUCell(message_module.out_channels, memory_dim)
    if memory_updater_cell == "gru":  # for TGN
      self.memory_updater = nn.GRUCell(message_module.out_channels, memory_dim)
    elif memory_updater_cell == "rnn":  # for JODIE & DyRep
      self.memory_updater = nn.RNNCell(message_module.out_channels, memory_dim)
    else:
      raise ValueError(
          "Undefined memory updater!!! Memory updater can be either 'gru' or"
          " 'rnn'."
      )

    self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
    last_update = torch.empty(self.num_nodes, dtype=torch.long)
    self.register_buffer("last_update", last_update)
    self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

    self.msg_s_store = {}
    self.msg_d_store = {}

    self.reset_parameters()

  @property
  def device(self):
    return self.time_enc.lin.weight.device

  def reset_parameters(self):
    r"""Resets all learnable parameters of the module."""
    if hasattr(self.msg_s_module, "reset_parameters"):
      self.msg_s_module.reset_parameters()
    if hasattr(self.msg_d_module, "reset_parameters"):
      self.msg_d_module.reset_parameters()
    if hasattr(self.aggr_module, "reset_parameters"):
      self.aggr_module.reset_parameters()
    self.time_enc.reset_parameters()
    self.memory_updater.reset_parameters()
    self.reset_state()

  def reset_state(self):
    """Resets the memory to its initial state."""
    torch_geometric.nn.inits.zeros(self.memory)
    torch_geometric.nn.inits.zeros(self.last_update)
    self._reset_message_store()

  def detach(self):
    """Detaches the memory from gradient computation."""
    self.memory.detach_()

  def forward(self, n_id):
    """Returns nodes' current memory and last updated timestamp.

    Args:
      n_id: a tensor of node indices.
    Returns:
      memory: a tensor of memories of nodes in `n_id`.
      last_update: a tensor of last updated timestamps of nodes in `n_id`.
    """
    if self.training:
      memory, last_update = self._get_updated_memory(n_id)
    else:
      memory, last_update = self.memory[n_id], self.last_update[n_id]

    return memory, last_update

  def update_state(
      self,
      *,
      src,
      dst,
      t,
      raw_msg,
  ):
    """Updates the memory with newly encountered interactions.

    Args:
      src: A tensor of source node indices.
      dst: A tensor of destination node indices.
      t: A tensor of timestamps.
      raw_msg: A tensor of raw messages.
    """
    n_id = torch.cat([src, dst]).unique()

    if self.training:
      self._update_memory(n_id)
      self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
      self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    else:
      self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
      self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
      self._update_memory(n_id)

  def _reset_message_store(self):
    i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
    msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
    # Message store format: (src, dst, t, msg)
    self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
    self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

  def _update_memory(self, n_id):
    memory, last_update = self._get_updated_memory(n_id)
    self.memory[n_id] = memory
    self.last_update[n_id] = last_update

  def _get_updated_memory(
      self, n_id
  ):
    self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

    # Compute messages (src -> dst).
    msg_s, t_s, src_s, unused_dst_s = self._compute_msg(
        n_id, self.msg_s_store, self.msg_s_module
    )

    # Compute messages (dst -> src).
    msg_d, t_d, src_d, unused_dst_d = self._compute_msg(
        n_id, self.msg_d_store, self.msg_d_module
    )

    # Aggregate messages.
    idx = torch.cat([src_s, src_d], dim=0)
    msg = torch.cat([msg_s, msg_d], dim=0)
    t = torch.cat([t_s, t_d], dim=0)
    aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

    # Get local copy of updated memory.
    memory = self.memory_updater(aggr, self.memory[n_id])

    # Get local copy of updated `last_update`.
    dim_size = self.last_update.size(0)
    last_update = torch_geometric.utils.scatter(
        t, idx, 0, dim_size, reduce="max"
    )[n_id]

    return memory, last_update

  def _update_msg_store(
      self,
      src,
      dst,
      t,
      raw_msg,
      msg_store,
  ):
    n_id, perm = src.sort()
    n_id, count = n_id.unique_consecutive(return_counts=True)
    for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
      msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

  def _compute_msg(
      self,
      n_id,
      msg_store,
      msg_module,
  ):
    data = [msg_store[i] for i in n_id.tolist()]
    src, dst, t, raw_msg = list(zip(*data))
    src = torch.cat(src, dim=0)
    dst = torch.cat(dst, dim=0)
    t = torch.cat(t, dim=0)
    raw_msg = torch.cat(raw_msg, dim=0)
    t_rel = t - self.last_update[src]
    t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

    msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

    return msg, t, src, dst

  def train(self, mode = True):
    """Sets the module in training mode."""
    if self.training and not mode:
      # Flush message store to memory in case we just entered eval mode.
      self._update_memory(
          torch.arange(self.num_nodes, device=self.memory.device)
      )
      self._reset_message_store()
    super().train(mode)


class DyRepMemory(nn.Module):
  r"""Based on intuitions from TGN Memory...

  Differences with the original TGN Memory:

      - can use source or destination embeddings in message generation
      - can use a RNN or GRU module as the memory updater
  """

  def __init__(
      self,
      num_nodes,
      raw_msg_dim,
      memory_dim,
      time_dim,
      message_module,
      aggregator_module,
      memory_updater_type,
      use_src_emb_in_msg = False,
      use_dst_emb_in_msg = False,
  ):
    """Initializes the DyRep memory module.

    Args:
      num_nodes: The number of nodes to save memories for.
      raw_msg_dim: The raw message dimensionality.
      memory_dim: The hidden memory dimensionality.
      time_dim: The time encoding dimensionality.
      message_module: The message function which combines source and destination
        node memory embeddings, the raw message and the time encoding.
      aggregator_module: The message aggregator function which aggregates
        messages to the same destination into a single representation.
      memory_updater_type: specifies whether the memory updater is GRU or RNN.
      use_src_emb_in_msg: whether to use the source embeddings in generation of
        messages.
      use_dst_emb_in_msg: whether to use the destination embeddings in
        generation of messages.
    """
    super().__init__()

    self.num_nodes = num_nodes
    self.raw_msg_dim = raw_msg_dim
    self.memory_dim = memory_dim
    self.time_dim = time_dim

    self.msg_s_module = message_module
    self.msg_d_module = copy.deepcopy(message_module)
    self.aggr_module = aggregator_module
    self.time_enc = time_enc.TimeEncoder(time_dim)

    assert memory_updater_type in [
        "gru",
        "rnn",
    ], "Memor updater can be either `rnn` or `gru`."
    if memory_updater_type == "gru":  # for TGN
      self.memory_updater = nn.GRUCell(message_module.out_channels, memory_dim)
    elif memory_updater_type == "rnn":  # for JODIE & DyRep
      self.memory_updater = nn.RNNCell(message_module.out_channels, memory_dim)
    else:
      raise ValueError(
          "Undefined memory updater!!! Memory updater can be either 'gru' or"
          " 'rnn'."
      )

    self.use_src_emb_in_msg = use_src_emb_in_msg
    self.use_dst_emb_in_msg = use_dst_emb_in_msg

    self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
    last_update = torch.empty(self.num_nodes, dtype=torch.long)
    self.register_buffer("last_update", last_update)
    self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

    self.msg_s_store = {}
    self.msg_d_store = {}

    self.reset_parameters()

  @property
  def device(self):
    return self.time_enc.lin.weight.device

  def reset_parameters(self):
    r"""Resets all learnable parameters of the module."""
    if hasattr(self.msg_s_module, "reset_parameters"):
      self.msg_s_module.reset_parameters()
    if hasattr(self.msg_d_module, "reset_parameters"):
      self.msg_d_module.reset_parameters()
    if hasattr(self.aggr_module, "reset_parameters"):
      self.aggr_module.reset_parameters()
    self.time_enc.reset_parameters()
    self.memory_updater.reset_parameters()
    self.reset_state()

  def reset_state(self):
    """Resets the memory to its initial state."""
    torch_geometric.nn.inits.zeros(self.memory)
    torch_geometric.nn.inits.zeros(self.last_update)
    self._reset_message_store()

  def detach(self):
    """Detaches the memory from gradient computation."""
    self.memory.detach_()

  def forward(self, n_id):
    """Returns nodes' current memory and last updated timestamp.

    Args:
      n_id: a tensor of node indices.
    Returns:
      memory: a tensor of memories of nodes in `n_id`.
      last_update: a tensor of last updated timestamps of nodes in `n_id`.
    """
    if self.training:
      memory, last_update = self._get_updated_memory(n_id)
    else:
      memory, last_update = self.memory[n_id], self.last_update[n_id]

    return memory, last_update

  def update_state(
      self,
      *,
      src,
      dst,
      t,
      raw_msg,
      embeddings = None,
      assoc = None,
  ):
    """Updates the memory with newly encountered interactions.

    Args:
      src: A tensor of source node indices.
      dst: A tensor of destination node indices.
      t: A tensor of timestamps.
      raw_msg: A tensor of raw messages.
      embeddings: A tensor of node embeddings.
      assoc: A transient tensor to store node indices for local operations.
    """
    n_id = torch.cat([src, dst]).unique()

    if self.training:
      self._update_memory(n_id, embeddings, assoc)
      self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
      self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    else:
      self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
      self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
      self._update_memory(n_id, embeddings, assoc)

  def _reset_message_store(self):
    i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
    msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
    # Message store format: (src, dst, t, msg)
    self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
    self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

  def _update_memory(
      self,
      n_id,
      embeddings = None,
      assoc = None,
  ):
    memory, last_update = self._get_updated_memory(n_id, embeddings, assoc)
    self.memory[n_id] = memory
    self.last_update[n_id] = last_update

  def _get_updated_memory(
      self,
      n_id,
      embeddings = None,
      assoc = None,
  ):
    self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

    # Compute messages (src -> dst).
    msg_s, t_s, src_s, unused_dst_s = self._compute_msg(
        n_id, self.msg_s_store, self.msg_s_module, embeddings, assoc
    )

    # Compute messages (dst -> src).
    msg_d, t_d, src_d, unused_dst_d = self._compute_msg(
        n_id, self.msg_d_store, self.msg_d_module, embeddings, assoc
    )

    # Aggregate messages.
    idx = torch.cat([src_s, src_d], dim=0)
    msg = torch.cat([msg_s, msg_d], dim=0)
    t = torch.cat([t_s, t_d], dim=0)
    aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

    # Get local copy of updated memory.
    memory = self.memory_updater(aggr, self.memory[n_id])

    # Get local copy of updated `last_update`.
    dim_size = self.last_update.size(0)
    last_update = torch_geometric.utils.scatter(
        t, idx, 0, dim_size, reduce="max"
    )[n_id]

    return memory, last_update

  def _update_msg_store(
      self,
      src,
      dst,
      t,
      raw_msg,
      msg_store,
  ):
    n_id, perm = src.sort()
    n_id, count = n_id.unique_consecutive(return_counts=True)
    for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
      msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

  def _compute_msg(
      self,
      n_id,
      msg_store,
      msg_module,
      embeddings = None,
      assoc = None,
  ):
    data = [msg_store[i] for i in n_id.tolist()]
    src, dst, t, raw_msg = list(zip(*data))
    src = torch.cat(src, dim=0)
    dst = torch.cat(dst, dim=0)
    t = torch.cat(t, dim=0)
    raw_msg = torch.cat(raw_msg, dim=0)
    t_rel = t - self.last_update[src]
    t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

    # source nodes: retrieve embeddings
    source_memory = self.memory[src]
    if self.use_src_emb_in_msg and embeddings != None:
      if src.size(0) > 0:
        curr_src, curr_src_idx = [], []
        for s_idx, s in enumerate(src):
          if s in n_id:
            curr_src.append(s.item())
            curr_src_idx.append(s_idx)

        source_memory[curr_src_idx] = embeddings[assoc[curr_src]]

    # destination nodes: retrieve embeddings
    destination_memory = self.memory[dst]
    if self.use_dst_emb_in_msg and embeddings != None:
      if dst.size(0) > 0:
        curr_dst, curr_dst_idx = [], []
        for d_idx, d in enumerate(dst):
          if d in n_id:
            curr_dst.append(d.item())
            curr_dst_idx.append(d_idx)
        destination_memory[curr_dst_idx] = embeddings[assoc[curr_dst]]

    msg = msg_module(source_memory, destination_memory, raw_msg, t_enc)

    return msg, t, src, dst

  def train(self, mode = True):
    """Sets the module in training mode."""
    if self.training and not mode:
      # Flush message store to memory in case we just entered eval mode.
      self._update_memory(
          torch.arange(self.num_nodes, device=self.memory.device)
      )
      self._reset_message_store()
    super().train(mode)
