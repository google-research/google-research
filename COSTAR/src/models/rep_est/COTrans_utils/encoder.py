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

"""COTrans encoder."""

import logging
import einops
from src.models.utils_transformer import get_fixed_sin_cos_encodings
from src.models.utils_transformer import MultiHeadedAttention
from src.models.utils_transformer import PositionwiseFeedForward
from src.models.utils_transformer import RelativePositionalEncoding as RelativeTemporalPositionalEncoding
import torch
from torch import nn

rearrange = einops.rearrange
repeat = einops.repeat
logger = logging.getLogger(__name__)


class AbsoluteTemporalPositionalEncoding(nn.Module):
  """Absolute temporal positional encoding."""

  def __init__(self, max_len, d_model, trainable=False):
    super().__init__()
    self.max_len = max_len
    self.trainable = trainable
    if trainable:
      self.pe = nn.Embedding(max_len, d_model)
    else:
      self.register_buffer('pe', get_fixed_sin_cos_encodings(d_model, max_len))

  def forward(self, x):
    # x: [B, T, F, D]
    batch_size = x.size(0)
    actual_len = x.shape[1]
    assert actual_len <= self.max_len

    pe = self.pe.weight if self.trainable else self.pe
    return (
        pe.unsqueeze(0).repeat(batch_size, 1, 1)[:, :actual_len, :].unsqueeze(2)
    )  # [B, T, 1, D]

  def get_pe(self, position):
    pe = self.pe.weight if self.trainable else self.pe
    return pe[position]


class AbsoluteFeaturePositionalEncoding(nn.Module):

  def __init__(self, feature_num, d_emb):
    super().__init__()
    self.feature_num = feature_num
    self.d_emb = d_emb
    self.emb = nn.Embedding(feature_num, d_emb)

  def forward(self, feature_num):
    # x: [B, T, F, D]
    assert feature_num <= self.feature_num
    return self.emb(
        torch.arange(feature_num).long().to(self.emb.weight.device)
    )  # [F, D]


class HierarchiFeaturePositionalEncoding(nn.Module):
  """Hierarchical feature positional encoding."""

  def __init__(self, max_branches, d_emb):
    super().__init__()
    self.max_branches = max_branches
    self.levels = len(max_branches)
    self.d_emb = d_emb
    embs = []
    for i in range(self.levels):
      embs.append(nn.Embedding(self.max_branches[i], d_emb))
    self.embs = nn.ModuleList(embs)

  def forward(self, coords):
    # coords: [F, L]
    assert coords.shape[1] <= self.levels
    embs = 0
    for i in range(coords.shape[1]):
      embs = embs + self.embs[i](coords[:, i])
    return embs


class BasicTransformerBlock(nn.Module):
  """Basic transformer block."""

  def __init__(
      self,
      hidden,
      attn_heads,
      head_size,
      feed_forward_hidden,
      dropout,
      attn_dropout,
      self_positional_encoding_k=None,
      self_positional_encoding_v=None,
      final_layer=False,
  ):
    super().__init__()

    attention_block_names = []
    self.self_attention = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=self_positional_encoding_k,
        positional_encoding_v=self_positional_encoding_v,
        final_layer=final_layer,
    )
    attention_block_names.append('self_attention')

    self.feed_forward = PositionwiseFeedForward(
        d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
    )

    self.attention_block_name2idx = {
        name: idx for idx, name in enumerate(attention_block_names)
    }

  def forward(self, x_vto, x_s, active_entries):
    # x_vto: [B, T, F, D]
    # x_s: [B, F, D]
    # active_entries: [B, T, 1]
    raise NotImplementedError()


class TemporalTranformerBlock(BasicTransformerBlock):
  """Temporal transformer block."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x_vto, x_s, active_entries):
    # causal attention along the temporal dimension for each F
    # skip x_s for self-attention
    _, timesteps, feature_num = (
        x_vto.shape[0],
        x_vto.shape[1],
        x_vto.shape[2],
    )
    x_vto = rearrange(x_vto, 'b t f d -> (b f) t d')
    active_entries_f = repeat(
        active_entries, 'b t d -> (b f) t d', f=feature_num
    )
    self_att_mask = active_entries_f.repeat(1, 1, timesteps).unsqueeze(1)
    x_vto = self.self_attention(x_vto, x_vto, x_vto, self_att_mask, True)
    x_vto_out = self.feed_forward(x_vto)

    s_feature_num = x_s.shape[1]
    x_s = rearrange(x_s, 'b f d -> (b f) 1 d')
    x_s_out = self.feed_forward(x_s)

    x_vto_out = rearrange(x_vto_out, '(b f) t d -> b t f d', f=feature_num)
    x_s_out = rearrange(x_s_out, '(b f) 1 d -> b f d', f=s_feature_num)
    return x_vto_out, x_s_out


class FeatureTransformerBlock(BasicTransformerBlock):
  """Feature transformer block."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x_vto, x_s, active_entries):
    # full attention along the feature dimension for each T
    # include x_s as well
    _, timesteps, vto_feature_num = (
        x_vto.shape[0],
        x_vto.shape[1],
        x_vto.shape[2],
    )
    x_all = torch.cat(
        [x_vto, repeat(x_s, 'b f d -> b t f d', t=timesteps)], dim=2
    )
    x_all = rearrange(x_all, 'b t f d -> (b t) f d')
    x_all = self.self_attention(x_all, x_all, x_all, None, False)
    x_vto = x_all[:, :vto_feature_num, :]
    x_vto_out = self.feed_forward(x_vto)

    x_s = self.self_attention(x_s, x_s, x_s, None, False)
    x_s_out = self.feed_forward(x_s)

    x_vto_out = rearrange(x_vto_out, '(b t) f d -> b t f d', t=timesteps)

    return x_vto_out, x_s_out


class TemporalFeatureTransformerBlock(nn.Module):
  """Temporal feature transformer block."""

  def __init__(
      self,
      hidden,
      attn_heads,
      head_size,
      feed_forward_hidden,
      dropout,
      attn_dropout,
      temporal_positional_encoding_k=None,
      temporal_positional_encoding_v=None,
      final_layer=False,
  ):
    super().__init__()

    self.temporal_block = TemporalTranformerBlock(
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=temporal_positional_encoding_k,
        self_positional_encoding_v=temporal_positional_encoding_v,
        final_layer=final_layer,
    )
    self.feature_block = FeatureTransformerBlock(
        hidden,
        attn_heads,
        head_size,
        feed_forward_hidden,
        dropout,
        attn_dropout,
        self_positional_encoding_k=None,
        self_positional_encoding_v=None,
        final_layer=final_layer,
    )

  def forward(self, x_vto, x_s, active_entries):
    x_vto, x_s = self.temporal_block(x_vto, x_s, active_entries)
    x_vto, x_s = self.feature_block(x_vto, x_s, active_entries)
    return x_vto, x_s


class COTransEncoder(nn.Module):
  """COTrans encoder."""

  def __init__(
      self,
      input_size,
      dim_treatments,
      dim_vitals,
      dim_outcome,
      dim_static_features,
      has_vitals,
      sub_args,
  ):
    super().__init__()
    self.input_size = input_size
    self.dim_treatments = dim_treatments
    self.dim_vitals = dim_vitals
    self.dim_outcome = dim_outcome
    self.dim_static_features = dim_static_features
    self.has_vitals = has_vitals

    self.max_seq_length = sub_args.max_seq_length
    self.seq_hidden_units = sub_args.seq_hidden_units
    self.dropout_rate = sub_args.dropout_rate
    self.num_layer = sub_args.num_layer
    self.num_heads = sub_args.num_heads
    self.attn_dropout = sub_args.attn_dropout

    self.temporal_positional_encoding_absolute = (
        sub_args.temporal_positional_encoding.absolute
    )
    self.temporal_positional_encoding_trainable = (
        sub_args.temporal_positional_encoding.trainable
    )
    self.temporal_positional_encoding_max_relative_position = (
        sub_args.temporal_positional_encoding.max_relative_position
    )

    self.feature_positional_encoding_absolute = (
        sub_args.feature_positional_encoding.absolute
    )

    self.head_size = self.seq_hidden_units // self.num_heads

    self.temporal_positional_encoding = (
        self.temporal_positional_encoding_k
    ) = self.temporal_positional_encoding_v = None
    if self.temporal_positional_encoding_absolute:
      self.temporal_positional_encoding = AbsoluteTemporalPositionalEncoding(
          self.max_seq_length,
          self.seq_hidden_units,
          self.temporal_positional_encoding_trainable,
      )
    else:
      # Relative positional encoding is shared across heads
      self.temporal_positional_encoding_k = RelativeTemporalPositionalEncoding(
          self.temporal_positional_encoding_max_relative_position,
          self.head_size,
          self.temporal_positional_encoding_trainable,
      )
      self.temporal_positional_encoding_v = RelativeTemporalPositionalEncoding(
          self.temporal_positional_encoding_max_relative_position,
          self.head_size,
          self.temporal_positional_encoding_trainable,
      )

    self.feature_positional_encoding = None
    if self.feature_positional_encoding_absolute:
      self.feature_positional_encoding = AbsoluteFeaturePositionalEncoding(
          self.dim_vitals
          + self.dim_treatments
          + self.dim_outcome
          + self.dim_static_features,
          self.seq_hidden_units,
      )
    else:
      self.feature_positional_encoding = HierarchiFeaturePositionalEncoding(
          max_branches=[
              4,
              max(
                  self.dim_vitals,
                  self.dim_treatments,
                  self.dim_outcome,
                  self.dim_static_features,
              ),
          ],
          d_emb=self.seq_hidden_units,
      )

    self.input_transformation = nn.Linear(1, self.seq_hidden_units)
    # feature order:
    # vitals (if exists), prev_treatments, prev_outcomes, static_features

    self.transformer_blocks = nn.ModuleList(
        [
            TemporalFeatureTransformerBlock(
                hidden=self.seq_hidden_units,
                attn_heads=self.num_heads,
                head_size=self.head_size,
                feed_forward_hidden=self.seq_hidden_units * 4,
                dropout=self.dropout_rate,
                attn_dropout=self.dropout_rate if self.attn_dropout else 0.0,
                temporal_positional_encoding_k=self.temporal_positional_encoding_k,
                temporal_positional_encoding_v=self.temporal_positional_encoding_v,
                final_layer=False,
            )
            for _ in range(self.num_layer)
        ]
    )

    self.output_dropout = nn.Dropout(self.dropout_rate)

  def forward(self, batch, return_comp_reps=False):
    prev_treatments = batch['prev_treatments']
    vitals = batch['vitals'] if self.has_vitals else None
    prev_outputs = batch['prev_outputs']
    static_features = batch['static_features']
    active_entries = batch['active_entries']

    to_concat = [vitals, prev_treatments, prev_outputs]
    x_vto = torch.cat(
        [x for x in to_concat if x is not None], dim=2
    )  # [B, T, F]
    x_vto = self.input_transformation(x_vto.unsqueeze(-1))  # [B, T, F, D]
    x_s = self.input_transformation(static_features.unsqueeze(-1))  # [B, F, D]

    # (if absolute) temporal positional encoding, only applies to x_vto
    if self.temporal_positional_encoding is not None:
      x_vto = x_vto + self.temporal_positional_encoding(x_vto)

    # feature positional encoding, applies to both x_vto and x_s
    if self.feature_positional_encoding_absolute:
      feature_positional_emb = self.feature_positional_encoding(
          x_vto.shape[2] + x_s.shape[1]
      )
    else:
      coords = []
      if self.has_vitals:
        coords.append(torch.tensor([[0, i] for i in range(self.dim_vitals)]))
      coords.append(torch.tensor([[1, i] for i in range(self.dim_treatments)]))
      coords.append(torch.tensor([[2, i] for i in range(self.dim_outcome)]))
      coords.append(
          torch.tensor([[3, i] for i in range(self.dim_static_features)])
      )
      coords = torch.cat(coords, dim=0).long().to(x_vto.device)
      feature_positional_emb = self.feature_positional_encoding(coords)
    vto_positional_emb, s_positional_emb = (
        feature_positional_emb[: -self.dim_static_features],
        feature_positional_emb[-self.dim_static_features :],
    )
    x_vto = x_vto + vto_positional_emb
    x_s = x_s + s_positional_emb

    # propagate through all layers
    for _, block in enumerate(self.transformer_blocks):
      x_vto, x_s = block(x_vto, x_s, active_entries)

    x_v = (
        x_vto[:, :, : self.dim_vitals].mean(dim=2) if self.has_vitals else None
    )
    x_t = x_vto[
        :, :, self.dim_vitals : self.dim_vitals + self.dim_treatments
    ].mean(dim=2)
    x_o = x_vto[:, :, self.dim_vitals + self.dim_treatments :].mean(dim=2)
    if self.has_vitals:
      comp_rep = [x_o, x_t, x_v]
      output = (x_o + x_t + x_v) / 3.0
    else:
      comp_rep = [x_o, x_t]
      output = (x_o + x_t) / 2.0

    output = self.output_dropout(output)

    if return_comp_reps:
      return output, comp_rep
    else:
      return output
