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

"""Huggingface-like T5 Attention Modules with Flash Attention implemented.

Started with code from:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

Here we provide the Pytorch wrappers for the 4 styles of Flash Attention we
implement
alongside their huggingface-like implementations to be slotted into existing T5
models
from huggingface.
"""

import copy
import math
import os
import time

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from transformers.pytorch_utils import prune_linear_layer

from .flash_attn import attention_v1 as fused_att1
from .flash_attn import attention_v2 as fused_att2
from .flash_attn import attention_v6 as fused_att6
from .flash_attn import attention_v7 as fused_att7
from .huggingface_extension import T5ForInterpretableGeneration


class FlashT5AttentionV1(nn.Module):
  """T5 model with Flash Attention v1."""

  def __init__(
      self, config, oldlayer, is_causal=False, needs_decoder_positions=False
  ):
    super().__init__()
    self.is_causal = is_causal
    self.is_decoder = oldlayer.is_decoder
    self.has_relative_attention_bias = oldlayer.has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = (
        config.relative_attention_max_distance
    )
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = oldlayer.q
    self.k = oldlayer.k
    self.v = oldlayer.v
    self.o = oldlayer.o
    if self.verbose:
      print("q", self.q.weight.shape)
      print("k", self.k.weight.shape)
      print("v", self.v.weight.shape)
      print("o", self.o.weight.shape)

    if self.has_relative_attention_bias:
      self.relative_attention_bias = oldlayer.relative_attention_bias
    self.pruned_heads = set()
    self.gradient_checkpointing = False

    self.needs_speculative_decoder_position_biases = needs_decoder_positions
    self.decoder_position_bias_indices = None

  def prune_heads(self, heads):
    if self.verbose:
      print("pruning")

    if not heads:
      return
    heads, index = find_pruneable_heads_and_indices(
        heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    )
    # Prune linear layers
    self.q = prune_linear_layer(self.q, index)
    self.k = prune_linear_layer(self.k, index)
    self.v = prune_linear_layer(self.v, index)
    self.o = prune_linear_layer(self.o, index, dim=1)
    # Update hyper params
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.key_value_proj_dim * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  @staticmethod
  def _relative_position_bucket(
      relative_position, bidirectional=True, num_buckets=32, max_distance=128
  ):
    relative_buckets = 0
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
      relative_position = torch.abs(relative_position)
    else:
      relative_position = -torch.min(
          relative_position, torch.zeros_like(relative_position)
      )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins
    # in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias."""
    if self.verbose:
      print("compute_bias()")
    if device is None:
      device = self.relative_attention_bias.weight.device
    context_position = torch.arange(
        query_length, dtype=torch.long, device=device
    )[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
        None, :
    ]
    relative_position = (
        memory_position - context_position
    )  # shape (query_length, key_length)
    relative_position_bucket = self._relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(
        relative_position_bucket
    )  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(
        0
    )  # shape (1, num_heads, query_length, key_length)
    return values

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      position_bias=None,
      past_key_value=None,
      layer_head_mask=None,
      query_length=None,
      use_cache=False,
      output_attentions=False,
  ):
    self.st.record()
    if self.verbose:
      print("forward()")

    # Self-attention (if key_value_states is None) or
    # attention over source sentence (provided by key_value_states).
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal)
    # or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      if len(past_key_value) != 2:
        raise ValueError(
            "past_key_value should have 2 past states: keys and values. Got"
            f" { len(past_key_value)} past states"
        )
      real_seq_length += (
          past_key_value[0].shape[2] if query_length is None else query_length
      )

    key_length = (
        real_seq_length
        if key_value_states is None
        else key_value_states.shape[1]
    )

    def shape(states):
      """Projection."""
      return states.view(
          batch_size, -1, self.n_heads, self.key_value_proj_dim
      ).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      """Projects hidden states correctly to key/query states."""
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          # self-attn
          # (batch_size, n_heads, key_length, dim_per_head)
          hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        elif past_key_value.shape[2] != key_value_states.shape[1]:
          # checking that the `sequence_length` of the `past_key_value` is
          # the same as the provided `key_value_states`
          # to support prefix tuning cross-attn
          # (batch_size, n_heads, seq_length, dim_per_head)
          hidden_states = shape(proj_layer(key_value_states))
        else:
          # cross-attn
          hidden_states = past_key_value
      return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    if position_bias is None:
      if self.needs_decoder_positions:
        relative_position_bucket = self._relative_position_bucket(
            self.decoder_position_bias_indices,
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        decoder_position_bias = self.relative_attention_bias(
            relative_position_bucket
        )
        position_bias = decoder_position_bias.permute([2, 0, 1]).unsqueeze(0)
      else:
        if not self.has_relative_attention_bias:
          position_bias = torch.zeros(
              (1, self.n_heads, real_seq_length, key_length),
              device=query_states.device,
              dtype=query_states.dtype,
          )
          if self.gradient_checkpointing and self.training:
            position_bias.requires_grad = True
        else:
          position_bias = self.compute_bias(
              real_seq_length, key_length, device=query_states.device
          )

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value is not None:
        position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

      if mask is not None:
        position_bias = (
            position_bias + mask
        )  # (batch_size, n_heads, seq_length, key_length)
    else:
      pass
      if mask is not None:
        position_bias = position_bias + mask

    if self.pruned_heads:
      mask = torch.ones(position_bias.shape[1])
      mask[list(self.pruned_heads)] = 0
      position_bias_masked = position_bias[:, mask.bool()]
    else:
      position_bias_masked = position_bias

    layer_bias = position_bias_masked
    causal = self.is_causal
    sm_scale = 1.0

    attn_output2 = fused_att1(
        query_states, key_states, value_states, layer_bias, causal, sm_scale
    )
    attn_output = (
        attn_output2.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, self.inner_dim)
    )

    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
      raise NotImplementedError(
          "cannot save attention softmax in current Flash implmentation"
      )
    self.ed.record()
    return outputs


class FlashT5AttentionV2(nn.Module):
  """T5 model with Flash Attention v2."""

  def __init__(self, config, oldlayer, is_causal=False):
    super().__init__()
    self.is_causal = is_causal
    self.is_decoder = oldlayer.is_decoder
    self.has_relative_attention_bias = oldlayer.has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = (
        config.relative_attention_max_distance
    )
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = oldlayer.q
    self.k = oldlayer.k
    self.v = oldlayer.v
    self.o = oldlayer.o

    if self.has_relative_attention_bias:
      self.relative_attention_bias = oldlayer.relative_attention_bias
    self.pruned_heads = set()
    self.gradient_checkpointing = False

  def prune_heads(self, heads):
    if self.verbose:
      print("pruning")

    if not heads:
      return
    heads, index = find_pruneable_heads_and_indices(
        heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    )
    # Prune linear layers
    self.q = prune_linear_layer(self.q, index)
    self.k = prune_linear_layer(self.k, index)
    self.v = prune_linear_layer(self.v, index)
    self.o = prune_linear_layer(self.o, index, dim=1)
    # Update hyper params
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.key_value_proj_dim * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  @staticmethod
  def _relative_position_bucket(
      relative_position, bidirectional=True, num_buckets=32, max_distance=128
  ):
    relative_buckets = 0
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
      relative_position = torch.abs(relative_position)
    else:
      relative_position = -torch.min(
          relative_position, torch.zeros_like(relative_position)
      )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins
    # in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
    if device is None:
      device = self.relative_attention_bias.weight.device

    virtual_position = torch.arange(
        -query_length + 1, key_length, dtype=torch.long, device=device
    )
    virtual_position_bucket = self._relative_position_bucket(
        virtual_position,
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    virt_values = self.relative_attention_bias(virtual_position_bucket)
    virt_values = virt_values.permute([1, 0]).unsqueeze(0)
    # should be that only the first layer is computing the position
    # biases for all heads
    return virt_values

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      position_bias=None,
      past_key_value=None,
      layer_head_mask=None,
      query_length=None,
      use_cache=False,
      output_attentions=False,
  ):
    self.st.record()

    # Self-attention (if key_value_states is None)
    # or attention over source sentence (provided by key_value_states).
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal)
    # or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      if len(past_key_value) != 2:
        raise ValueError(
            "past_key_value should have 2 past states: keys and values. Got"
            f" { len(past_key_value)} past states"
        )
      real_seq_length += (
          past_key_value[0].shape[2] if query_length is None else query_length
      )

    key_length = (
        real_seq_length
        if key_value_states is None
        else key_value_states.shape[1]
    )

    def shape(states):
      """Projection."""
      return states.view(
          batch_size, -1, self.n_heads, self.key_value_proj_dim
      ).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      """Projects hidden states correctly to key/query states."""
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          # self-attn
          # (batch_size, n_heads, key_length, dim_per_head)
          hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        elif past_key_value.shape[2] != key_value_states.shape[1]:
          # checking that the `sequence_length` of the `past_key_value` is the
          # same as the provided `key_value_states` to support prefix tuning
          # cross-attn
          # (batch_size, n_heads, seq_length, dim_per_head)
          hidden_states = shape(proj_layer(key_value_states))
        else:
          # cross-attn
          hidden_states = past_key_value
      return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    if position_bias is None:
      if not self.has_relative_attention_bias:
        position_bias = torch.zeros(
            (1, self.n_heads, real_seq_length, key_length),
            device=query_states.device,
            dtype=query_states.dtype,
        )
        if self.gradient_checkpointing and self.training:
          position_bias.requires_grad = True
      else:
        position_bias = self.compute_bias(
            real_seq_length, key_length, device=query_states.device
        )

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value is not None:
        position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

    if self.pruned_heads:
      mask = torch.ones(position_bias.shape[1])
      mask[list(self.pruned_heads)] = 0
      position_bias_masked = position_bias[:, mask.bool()]
    else:
      position_bias_masked = position_bias

    layer_bias = position_bias_masked
    causal = self.is_causal
    sm_scale = 1.0

    attn_output2 = fused_att2(
        query_states, key_states, value_states, layer_bias, causal, sm_scale
    )
    attn_output = (
        attn_output2.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, self.inner_dim)
    )
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
      raise NotImplementedError(
          "cannot save attention softmax in current Flash implmentation"
      )
    self.ed.record()
    return outputs


class FlashT5AttentionV6(nn.Module):
  """T5 model with Flash Attention v6."""

  def __init__(self, config, oldlayer, is_causal=False):
    super().__init__()
    self.is_causal = is_causal
    self.is_decoder = oldlayer.is_decoder
    self.has_relative_attention_bias = oldlayer.has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = (
        config.relative_attention_max_distance
    )
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = oldlayer.q
    self.k = oldlayer.k
    self.v = oldlayer.v
    self.o = oldlayer.o

    if self.has_relative_attention_bias:
      self.relative_attention_bias = oldlayer.relative_attention_bias
    self.pruned_heads = set()
    self.gradient_checkpointing = False

  def prune_heads(self, heads):
    if not heads:
      return
    heads, index = find_pruneable_heads_and_indices(
        heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    )
    # Prune linear layers
    self.q = prune_linear_layer(self.q, index)
    self.k = prune_linear_layer(self.k, index)
    self.v = prune_linear_layer(self.v, index)
    self.o = prune_linear_layer(self.o, index, dim=1)
    # Update hyper params
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.key_value_proj_dim * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  @staticmethod
  def _relative_position_bucket(
      relative_position, bidirectional=True, num_buckets=32, max_distance=128
  ):
    relative_buckets = 0
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
      relative_position = torch.abs(relative_position)
    else:
      relative_position = -torch.min(
          relative_position, torch.zeros_like(relative_position)
      )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in
    # positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias."""
    if device is None:
      device = self.relative_attention_bias.weight.device

    virtual_position = torch.arange(
        -query_length + 1, key_length, dtype=torch.long, device=device
    )
    virtual_position_bucket = self._relative_position_bucket(
        virtual_position,
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    virt_values = self.relative_attention_bias(virtual_position_bucket)
    virt_values = virt_values.permute([1, 0]).unsqueeze(0)
    return virt_values

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      position_bias=None,
      past_key_value=None,
      layer_head_mask=None,
      query_length=None,
      use_cache=False,
      output_attentions=False,
  ):
    self.st.record()
    # Self-attention (if key_value_states is None)
    # or attention over source sentence (provided by key_value_states).
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal)
    # or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      if len(past_key_value) != 2:
        raise ValueError(
            "past_key_value should have 2 past states: keys and values. Got"
            f" { len(past_key_value)} past states"
        )
      real_seq_length += (
          past_key_value[0].shape[2] if query_length is None else query_length
      )

    key_length = (
        real_seq_length
        if key_value_states is None
        else key_value_states.shape[1]
    )

    def shape(states):
      """Projection."""
      return states.view(
          batch_size, -1, self.n_heads, self.key_value_proj_dim
      ).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      """Projects hidden states correctly to key/query states."""
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          # self-attn
          # (batch_size, n_heads, key_length, dim_per_head)
          hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        elif past_key_value.shape[2] != key_value_states.shape[1]:
          # checking that the `sequence_length` of the `past_key_value` is the
          # same as the provided `key_value_states` to support prefix tuning
          # cross-attn
          # (batch_size, n_heads, seq_length, dim_per_head)
          hidden_states = shape(proj_layer(key_value_states))
        else:
          # cross-attn
          hidden_states = past_key_value
      return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    if position_bias is None:
      if not self.has_relative_attention_bias:
        position_bias = torch.zeros(
            (1, self.n_heads, real_seq_length, key_length),
            device=query_states.device,
            dtype=query_states.dtype,
        )
        if self.gradient_checkpointing and self.training:
          position_bias.requires_grad = True
      else:
        position_bias = self.compute_bias(
            real_seq_length, key_length, device=query_states.device
        )

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value is not None:
        position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

    if self.pruned_heads:
      mask = torch.ones(position_bias.shape[1])
      mask[list(self.pruned_heads)] = 0
      position_bias_masked = position_bias[:, mask.bool()]
    else:
      position_bias_masked = position_bias

    layer_bias = position_bias_masked
    causal = self.is_causal
    layer_sparsity = self.sparsity_matrix
    layer_bias_shift = self.bias_shift_matrix
    layer_local_lens = self.local_lengths
    sm_scale = 1.0

    attn_output2 = fused_att6(
        query_states,
        key_states,
        value_states,
        layer_bias,
        causal,
        sm_scale,
        layer_sparsity,
        layer_bias_shift,
        layer_local_lens,
    )
    attn_output = (
        attn_output2.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, self.inner_dim)
    )
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
      raise NotImplementedError(
          "cannot save attention softmax in current Flash implmentation"
      )
    self.ed.record()
    return outputs


class FlashT5AttentionV7(nn.Module):
  """T5 model with Flash Attention v7."""

  def __init__(self, config, oldlayer, is_causal=False):
    super().__init__()
    self.is_causal = is_causal
    self.is_decoder = oldlayer.is_decoder
    self.has_relative_attention_bias = oldlayer.has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = (
        config.relative_attention_max_distance
    )
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = oldlayer.q
    self.k = oldlayer.k
    self.v = oldlayer.v
    self.o = oldlayer.o

    if self.has_relative_attention_bias:
      self.relative_attention_bias = oldlayer.relative_attention_bias
    self.pruned_heads = set()
    self.gradient_checkpointing = False

  def prune_heads(self, heads):
    if not heads:
      return
    heads, index = find_pruneable_heads_and_indices(
        heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    )
    # Prune linear layers
    self.q = prune_linear_layer(self.q, index)
    self.k = prune_linear_layer(self.k, index)
    self.v = prune_linear_layer(self.v, index)
    self.o = prune_linear_layer(self.o, index, dim=1)
    # Update hyper params
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.key_value_proj_dim * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  @staticmethod
  def _relative_position_bucket(
      relative_position, bidirectional=True, num_buckets=32, max_distance=128
  ):
    relative_buckets = 0
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
      relative_position = torch.abs(relative_position)
    else:
      relative_position = -torch.min(
          relative_position, torch.zeros_like(relative_position)
      )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in
    # positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets

  def compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias."""
    if device is None:
      device = self.relative_attention_bias.weight.device

    virtual_position = torch.arange(
        -query_length + 1, key_length, dtype=torch.long, device=device
    )
    virtual_position_bucket = self._relative_position_bucket(
        virtual_position,
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    virt_values = self.relative_attention_bias(virtual_position_bucket)
    virt_values = virt_values.permute([1, 0]).unsqueeze(0)
    return virt_values

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      position_bias=None,
      past_key_value=None,
      layer_head_mask=None,
      query_length=None,
      use_cache=False,
      output_attentions=False,
  ):
    self.st.record()
    # Self-attention (if key_value_states is None)
    # or attention over source sentence (provided by key_value_states).
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal)
    # or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      if len(past_key_value) != 2:
        raise ValueError(
            "past_key_value should have 2 past states: keys and values. Got"
            f" { len(past_key_value)} past states"
        )
      real_seq_length += (
          past_key_value[0].shape[2] if query_length is None else query_length
      )

    key_length = (
        real_seq_length
        if key_value_states is None
        else key_value_states.shape[1]
    )

    def shape(states):
      """Projection."""
      # altered to work for 'v7' with encoding once and decoding multiple
      # (better for memory)
      return states.view(
          -1, states.shape[1], self.n_heads, self.key_value_proj_dim
      ).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      """Projects hidden states correctly to key/query states."""
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          # self-attn
          # (batch_size, n_heads, key_length, dim_per_head)
          hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        elif past_key_value.shape[2] != key_value_states.shape[1]:
          # checking that the `sequence_length` of the `past_key_value` is the
          # same as the provided `key_value_states` to support prefix tuning
          # cross-attn
          # (batch_size, n_heads, seq_length, dim_per_head)
          hidden_states = shape(proj_layer(key_value_states))
        else:
          # cross-attn
          hidden_states = past_key_value
      return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    if position_bias is None:
      if not self.has_relative_attention_bias:
        position_bias = torch.zeros(
            (1, self.n_heads, real_seq_length, key_length),
            device=query_states.device,
            dtype=query_states.dtype,
        )
        if self.gradient_checkpointing and self.training:
          position_bias.requires_grad = True
      else:
        position_bias = self.compute_bias(
            real_seq_length, key_length, device=query_states.device
        )

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value is not None:
        position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

    if self.pruned_heads:
      mask = torch.ones(position_bias.shape[1])
      mask[list(self.pruned_heads)] = 0
      position_bias_masked = position_bias[:, mask.bool()]
    else:
      position_bias_masked = position_bias

    layer_bias = position_bias_masked
    causal = self.is_causal
    layer_sparsity = self.sparsity_matrix
    layer_bias_shift = self.bias_shift_matrix
    layer_local_lens = self.local_lengths
    sm_scale = 1.0

    attn_output2 = fused_att7(
        query_states,
        key_states,
        value_states,
        layer_bias,
        causal,
        sm_scale,
        layer_sparsity,
        layer_bias_shift,
        layer_local_lens,
    )
    attn_output = (
        attn_output2.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, self.inner_dim)
    )
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
      raise NotImplementedError(
          "cannot save attention softmax in current Flash implmentation"
      )
    self.ed.record()
    return outputs


# TODO(enouen): need to build dependency for 'jam_transformer'
#               this involves building the class extending GenerationMixin
def return_pretrained_model(model_name, verbose=True):
  """Returns pretrained model."""
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

  start_time = time.time()
  if model_name == "flan-t5-xxl":
    model = T5ForInterpretableGeneration.from_pretrained(
        "google/flan-t5-xxl", low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
  elif model_name == "flan-t5-large":
    model = T5ForInterpretableGeneration.from_pretrained(
        "google/flan-t5-large", torch_dtype=torch.float16
    )

  elif model_name == "FiD-t5-large":
    model_path2 = os.getcwd() + "/FiD/pretrained_models/nq_reader_large/"

    model = T5ForInterpretableGeneration.from_pretrained(
        model_path2, torch_dtype=torch.float16
    )
  else:
    raise NotImplementedError("model_name not properly defined", model_name)

  if verbose:
    print("model loaded in", time.time() - start_time, "seconds")
    print()

  model4 = copy.deepcopy(model)

  include_self = True
  include_cross = True
  include_causal = True

  # 'model_flash_attn_architecture_type' is the variable which decides which
  # of the flash attention implementations are required, the typical dense
  # version for typical T5 models or the block sparse version for FiD.
  if model_name == "FiD-t5-large":
    model_flash_attn_architecture_type = 7
  elif model_name in ["flan-t5-xxl", "flan-t5-large"]:
    model_flash_attn_architecture_type = 2
  else:
    raise NotImplementedError("model_name not properly defined", model_name)

  # TODO(enounen): feels like a slightly hacky way to extend these, would need
  # further refactoring alongside the 'FlashAttentionT5V_' classes
  if model_flash_attn_architecture_type == 2:
    for i in range(0, 24):
      if include_self:
        existing = model.encoder.block[i].layer[0].SelfAttention
        new = FlashT5AttentionV2(model.encoder.config, existing)
        model4.encoder.block[i].layer[0].SelfAttention = new
    for i in range(0, 24):
      if include_cross:
        # decoder cross attention
        existing = model.decoder.block[i].layer[1].EncDecAttention
        new = FlashT5AttentionV1(model.decoder.config, existing)
        model4.decoder.block[i].layer[1].EncDecAttention = new
      if include_causal:
        # THIS IS THE CAUSAL DECODER SELF ATTN
        existing = model.decoder.block[i].layer[0].SelfAttention
        new = FlashT5AttentionV1(
            model.decoder.config, existing, is_causal=False
        )
        model4.decoder.block[i].layer[0].SelfAttention = new

  elif model_flash_attn_architecture_type == 7:
    for i in range(0, 24):
      if include_self:
        existing = model.encoder.block[i].layer[0].SelfAttention
        new = FlashT5AttentionV6(model.encoder.config, existing)
        model4.encoder.block[i].layer[0].SelfAttention = new
    for i in range(0, 24):
      if include_cross:
        # decoder cross attention
        existing = model.decoder.block[i].layer[1].EncDecAttention
        new = FlashT5AttentionV7(model.decoder.config, existing)
        model4.decoder.block[i].layer[1].EncDecAttention = new
      if include_causal:
        # THIS IS THE CAUSAL DECODER SELF ATTN
        existing = model.decoder.block[i].layer[0].SelfAttention
        new = FlashT5AttentionV1(
            model.decoder.config, existing, is_causal=False
        )
        model4.decoder.block[i].layer[0].SelfAttention = new
  # del model # needed because of this refactoring?
  return model4, tokenizer


def fid_adjust_existing_model(
    model4,
    device,
    sparsity,
    bias_shift_np,
    local_lengths_np,
    nop,
):
  """Adjusts FiD model."""
  # built from 'return_shifts_from_prompt_list' method
  bias_shift = torch.LongTensor(bias_shift_np).to(device)
  local_lengths = torch.LongTensor(local_lengths_np).to(device)

  # gets size reinitialized inside
  sparsity2 = torch.ones((1, 1, 2 * nop))
  sparsity2 = sparsity2.to(device)
  bias_shift2 = torch.LongTensor(bias_shift_np[0][None]).to(device)

  # TODO(enounen): feels hacky, consider refactoring alongside T5 model
  for i in range(0, 24):
    model4.encoder.block[i].layer[0].SelfAttention.sparsity_matrix = sparsity
    model4.encoder.block[i].layer[
        0
    ].SelfAttention.bias_shift_matrix = bias_shift
    model4.encoder.block[i].layer[0].SelfAttention.local_lengths = local_lengths

    model4.decoder.block[i].layer[1].EncDecAttention.sparsity_matrix = sparsity2
    model4.decoder.block[i].layer[
        1
    ].EncDecAttention.bias_shift_matrix = bias_shift2
    model4.decoder.block[i].layer[
        1
    ].EncDecAttention.local_lengths = local_lengths
