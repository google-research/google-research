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

"""Transformer utils."""

import math
import torch
from torch import nn
import torch.nn.functional as F


def get_fixed_sin_cos_encodings(d_model, max_len):
  """Sin-cos fixed positional encodddings.

  Args:
      d_model: hidden state dimensionality
      max_len: max sequence length

  Returns:
      PE
  """
  assert d_model % 2 == 0
  position = torch.arange(max_len).unsqueeze(1)
  div_term = torch.exp(
      torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
  )
  pe = torch.zeros(max_len, d_model)
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  return pe


class AbsolutePositionalEncoding(nn.Module):
  """Absolution positional encoding."""

  def __init__(self, max_len, d_model, trainable=False):
    super().__init__()
    self.max_len = max_len
    self.trainable = trainable
    if trainable:
      self.pe = nn.Embedding(max_len, d_model)
    else:
      self.register_buffer('pe', get_fixed_sin_cos_encodings(d_model, max_len))

  def forward(self, x):
    batch_size = x.size(0)
    actual_len = x.shape[1]
    assert actual_len <= self.max_len

    pe = self.pe.weight if self.trainable else self.pe
    return pe.unsqueeze(0).repeat(batch_size, 1, 1)[:, :actual_len, :]

  def get_pe(self, position):
    pe = self.pe.weight if self.trainable else self.pe
    return pe[position]


class RelativePositionalEncoding(nn.Module):
  """Relative positional encoding."""

  def __init__(
      self,
      max_relative_position,
      d_model,
      trainable=False,
      cross_attn=False,
  ):
    super().__init__()
    self.max_relative_position = max_relative_position
    self.trainable = trainable
    self.cross_attn = cross_attn
    self.num_embeddings = (
        (max_relative_position * 2 + 1)
        if not cross_attn
        else (max_relative_position + 1)
    )
    if trainable:
      self.embeddings_table = nn.Embedding(self.num_embeddings, d_model)
    else:
      self.register_buffer(
          'embeddings_table',
          get_fixed_sin_cos_encodings(d_model, max_relative_position * 2 + 1),
      )

  def forward(self, length_q, length_k):
    embeddings_table = (
        self.embeddings_table.weight
        if self.trainable
        else self.embeddings_table
    )

    if self.cross_attn:
      distance_mat = (
          torch.arange(length_k - 1, -1, -1)[None, :]
          + torch.arange(length_q)[:, None]
      )
    else:
      distance_mat = (
          torch.arange(length_k)[None, :] - torch.arange(length_q)[:, None]
      )
    distance_mat_clipped = torch.clamp(
        distance_mat, -self.max_relative_position, self.max_relative_position
    )
    if not self.cross_attn:
      distance_mat_clipped = distance_mat_clipped + self.max_relative_position
    final_mat = torch.LongTensor(distance_mat_clipped)
    embeddings = embeddings_table[final_mat]

    return embeddings


class UnalignedRelativePositionalEncoding(RelativePositionalEncoding):
  """Unaligned relative positional encoding."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, length_q, length_k):
    embeddings_table = (
        self.embeddings_table.weight
        if self.trainable
        else self.embeddings_table
    )

    if self.cross_attn:
      assert length_q == length_k
      distance_mat = (
          torch.arange(length_k - 1, -1, -1)[None, :]
          + torch.arange(length_q)[:, None]
          - (length_q - 1)
      )
      distance_mat = torch.clamp(distance_mat, 0)
    else:
      distance_mat = (
          torch.arange(length_k)[None, :] - torch.arange(length_q)[:, None]
      )
    distance_mat_clipped = torch.clamp(
        distance_mat, -self.max_relative_position, self.max_relative_position
    )
    if not self.cross_attn:
      distance_mat_clipped = distance_mat_clipped + self.max_relative_position
    final_mat = torch.LongTensor(distance_mat_clipped)
    embeddings = embeddings_table[final_mat]

    return embeddings


class LayerNorm(nn.Module):

  def __init__(self, features, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(features))
    self.bias = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.weight * (x - mean) / (std + self.eps) + self.bias


class Attention(nn.Module):
  """Multi head attention."""

  def __init__(
      self,
      positional_encoding_k = None,
      positional_encoding_v = None,
  ):
    super(Attention, self).__init__()
    self.positional_encoding_k = positional_encoding_k
    self.positional_encoding_v = positional_encoding_v

  def forward(
      self, query, key, value, mask=None, dropout=None, one_direction=False
  ):
    scores = torch.matmul(query, key.transpose(-2, -1))

    if self.positional_encoding_k is not None:
      bigr_k = self.positional_encoding_k(query.size(2), key.size(2))
      scores = scores + torch.einsum('b h q d, q k d -> b h q k', query, bigr_k)

    scores = scores / math.sqrt(query.size(-1))

    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)

    if (
        one_direction
    ):  # Required for self-attention, but not for cross-attention
      direction_mask = torch.ones_like(scores)
      direction_mask = torch.tril(direction_mask)
      scores = scores.masked_fill(direction_mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
      p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)

    if self.positional_encoding_v is not None:
      bigr_v = self.positional_encoding_v(query.size(2), value.size(2))
      output = output + torch.einsum(
          'b h q v, q v d -> b h q d', p_attn, bigr_v
      )

    return output, p_attn


class MultiHeadedAttention(nn.Module):
  """Multihead attention block."""

  def __init__(
      self,
      num_heads,
      d_model,
      head_size=None,
      dropout=0.0,
      positional_encoding_k=None,
      positional_encoding_v=None,
      final_layer=False,
  ):
    super().__init__()

    if d_model % num_heads != 0:
      raise AssertionError(f'd_model: {d_model} and num_heads: {num_heads}')

    self.num_heads = num_heads
    if head_size is not None:
      self.head_size = head_size
    else:
      self.head_size = d_model // num_heads

    self.linear_layers = nn.ModuleList(
        [nn.Linear(d_model, self.num_heads * self.head_size) for _ in range(3)]
    )
    self.attention = Attention(positional_encoding_k, positional_encoding_v)
    self.dropout = nn.Dropout(p=dropout)
    if final_layer:
      self.final_layer = nn.Linear(self.num_heads * self.head_size, d_model)
    self.layer_norm = LayerNorm(d_model)

  def forward(
      self, query, key, value, mask=None, one_direction=True, prefix=None
  ):
    batch_size = query.size(0)

    # 1) do all the linear projections in batch from d_model => num_heads x d_k
    query_, key_, value_ = [
        layer(x)
        .view(batch_size, -1, self.num_heads, self.head_size)
        .transpose(1, 2)
        for layer, x in zip(self.linear_layers, (query, key, value))
    ]
    if prefix is not None:
      prefix_key_, prefix_value_ = prefix[0].unsqueeze(0), prefix[1].unsqueeze(
          0
      )  # (1, T, num_heads, head_size)
      prefix_key_ = prefix_key_.repeat(batch_size, 1, 1, 1).transpose(
          1, 2
      )  # (B, num_heads, T, head_size)
      prefix_value_ = prefix_value_.repeat(batch_size, 1, 1, 1).transpose(1, 2)
      key_ = torch.cat([prefix_key_, key_], dim=2)
      value_ = torch.cat([prefix_value_, value_], dim=2)
      prefix_mask = torch.ones(
          (1, 1, 1, prefix_key_.shape[2]), dtype=mask.dtype
      ).to(mask.device)
      prefix_mask = prefix_mask.repeat(
          mask.shape[0], mask.shape[1], mask.shape[2], 1
      )
      mask = torch.cat([prefix_mask, mask], dim=3)
    # 2) apply self_attention on all the projected vectors in batch.
    x, _ = self.attention(
        query_,
        key_,
        value_,
        mask=mask,
        dropout=self.dropout,
        one_direction=one_direction,
    )

    # 3) "concat" using a view and apply a final linear.
    x = (
        x.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, self.num_heads * self.head_size)
    )
    if hasattr(self, 'final_layer'):
      x = self.final_layer(x)

    return self.layer_norm(x + query)


class PositionwiseFeedForward(nn.Module):
  """Position-wise feed forward module."""

  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.conv1 = nn.Conv1d(
        in_channels=d_model, out_channels=d_ff, kernel_size=1
    )
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv1d(
        in_channels=d_ff, out_channels=d_model, kernel_size=1
    )
    self.layer_norm = LayerNorm(d_model)

  def forward(self, x):
    x_ = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
    return self.layer_norm(self.dropout(self.conv2(x_)).permute(0, 2, 1) + x)


class TransformerEncoderBlock(nn.Module):
  """Transformer encoder block."""

  def __init__(
      self,
      hidden,
      attn_heads,
      head_size,
      feed_forward_hidden,
      dropout,
      attn_dropout=0.1,
      self_positional_encoding_k=None,
      self_positional_encoding_v=None,
      final_layer=True,
      **kwargs,
  ):
    super().__init__()
    # self.layer_norm = LayerNorm(hidden)
    # already used in MultiHeadedAttention and PositionwiseFeedForward
    self.self_attention = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=self_positional_encoding_k,
        positional_encoding_v=self_positional_encoding_v,
        final_layer=final_layer,
    )
    self.feed_forward = PositionwiseFeedForward(
        d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
    )

  def forward(self, x, active_entries):
    self_att_mask = (
        (active_entries.unsqueeze(1) * active_entries.unsqueeze(2))
        .squeeze(-1)
        .unsqueeze(1)
    )
    x = self.self_attention(x, x, x, self_att_mask, True)
    x = self.feed_forward(x)
    return x


class TransformerDecoderBlock(nn.Module):
  """Transformer decoder block."""

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
      cross_positional_encoding_k=None,
      cross_positional_encoding_v=None,
      final_layer=False,
      **kwargs,
  ):
    super().__init__()
    self.layer_norm = LayerNorm(hidden)
    self.self_attention = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=self_positional_encoding_k,
        positional_encoding_v=self_positional_encoding_v,
        final_layer=final_layer,
    )
    self.cross_attention = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=cross_positional_encoding_k,
        positional_encoding_v=cross_positional_encoding_v,
        final_layer=final_layer,
    )
    self.feed_forward = PositionwiseFeedForward(
        d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
    )

  def forward(self, x, encoder_x, active_entries, active_encoder_br):
    self_att_mask = (
        (active_entries.unsqueeze(1) * active_entries.unsqueeze(2))
        .squeeze(-1)
        .unsqueeze(1)
    )
    cross_att_mask = (
        active_encoder_br.unsqueeze(1) * active_entries
    ).unsqueeze(1)

    x = self.self_attention(x, x, x, self_att_mask, True)
    x = self.cross_attention(x, encoder_x, encoder_x, cross_att_mask, False)
    x = self.feed_forward(x)
    return x


class TransformerMultiInputBlock(nn.Module):
  """Transformer multiple input block."""

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
      n_inputs=2,
      final_layer=False,
      disable_cross_attention=False,
      isolate_subnetwork='',
      #  prefix_tuning=False, prefix_length=5, prefix_dim=64, prefix_mid_dim=64,
      **kwargs,
  ):
    super().__init__()
    self.n_inputs = n_inputs
    self.disable_cross_attention = disable_cross_attention
    self.isolate_subnetwork = isolate_subnetwork

    attention_block_names = []

    self.self_attention_o = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=self_positional_encoding_k,
        positional_encoding_v=self_positional_encoding_v,
        final_layer=final_layer,
    )
    attention_block_names.append('self_attention_o')
    self.self_attention_t = MultiHeadedAttention(
        num_heads=attn_heads,
        d_model=hidden,
        head_size=head_size,
        dropout=attn_dropout,
        positional_encoding_k=self_positional_encoding_k,
        positional_encoding_v=self_positional_encoding_v,
        final_layer=final_layer,
    )
    attention_block_names.append('self_attention_t')
    if not disable_cross_attention:
      self.cross_attention_ot = MultiHeadedAttention(
          num_heads=attn_heads,
          d_model=hidden,
          head_size=head_size,
          dropout=attn_dropout,
          positional_encoding_k=self_positional_encoding_k,
          positional_encoding_v=self_positional_encoding_v,
          final_layer=final_layer,
      )
      attention_block_names.append('cross_attention_ot')
      self.cross_attention_to = MultiHeadedAttention(
          num_heads=attn_heads,
          d_model=hidden,
          head_size=head_size,
          dropout=attn_dropout,
          positional_encoding_k=self_positional_encoding_k,
          positional_encoding_v=self_positional_encoding_v,
          final_layer=final_layer,
      )
      attention_block_names.append('cross_attention_to')

    if n_inputs == 3:
      self.self_attention_v = MultiHeadedAttention(
          num_heads=attn_heads,
          d_model=hidden,
          head_size=head_size,
          dropout=attn_dropout,
          positional_encoding_k=self_positional_encoding_k,
          positional_encoding_v=self_positional_encoding_v,
          final_layer=final_layer,
      )
      attention_block_names.append('self_attention_v')
      if not disable_cross_attention:
        self.cross_attention_tv = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('cross_attention_tv')
        self.cross_attention_vt = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('cross_attention_vt')
        self.cross_attention_ov = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('cross_attention_ov')
        self.cross_attention_vo = MultiHeadedAttention(
            num_heads=attn_heads,
            d_model=hidden,
            head_size=head_size,
            dropout=attn_dropout,
            positional_encoding_k=self_positional_encoding_k,
            positional_encoding_v=self_positional_encoding_v,
            final_layer=final_layer,
        )
        attention_block_names.append('cross_attention_vo')

    self.feed_forwards = [
        PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        for _ in range(n_inputs)
    ]
    self.feed_forwards = nn.ModuleList(self.feed_forwards)

    self.n_inputs = n_inputs

    self.attention_block_name2idx = {
        name: idx for idx, name in enumerate(attention_block_names)
    }

  def _fetch_prefix(self, prefix_list, attn_name):
    if prefix_list is None:
      return None
    else:
      return prefix_list[self.attention_block_name2idx[attn_name]]

  def forward(
      self,
      x_tov,
      x_s,
      active_entries_treat_outcomes,
      active_entries_vitals=None,
      prefix_list=None,
  ):
    assert len(x_tov) == self.n_inputs
    if self.n_inputs == 2:
      x_t, x_o = x_tov
      x_v = None
    else:
      x_t, x_o, x_v = x_tov

    self_att_mask_ot = active_entries_treat_outcomes.repeat(
        1, 1, x_t.size(1)
    ).unsqueeze(1)
    cross_att_mask_ot = cross_att_mask_to = self_att_mask_ot

    x_t_ = self.self_attention_t(
        x_t,
        x_t,
        x_t,
        self_att_mask_ot,
        True,
        prefix=self._fetch_prefix(prefix_list, 'self_attention_t'),
    )

    if (
        not self.disable_cross_attention
        and self.isolate_subnetwork != 't'
        and self.isolate_subnetwork != 'o'
    ):
      x_to_ = self.cross_attention_to(
          x_t_,
          x_o,
          x_o,
          cross_att_mask_ot,
          True,
          prefix=self._fetch_prefix(prefix_list, 'cross_attention_to'),
      )
    else:
      x_to_ = x_t_

    x_o_ = self.self_attention_o(
        x_o,
        x_o,
        x_o,
        self_att_mask_ot,
        True,
        prefix=self._fetch_prefix(prefix_list, 'self_attention_o'),
    )
    if (
        not self.disable_cross_attention
        and self.isolate_subnetwork != 'o'
        and self.isolate_subnetwork != 't'
    ):
      x_ot_ = self.cross_attention_ot(
          x_o_,
          x_t,
          x_t,
          cross_att_mask_to,
          True,
          prefix=self._fetch_prefix(prefix_list, 'cross_attention_ot'),
      )
    else:
      x_ot_ = x_o_

    if self.n_inputs == 2:
      out_t = self.feed_forwards[0](x_to_ + x_s)
      out_o = self.feed_forwards[1](x_ot_ + x_s)

      return out_t, out_o

    else:
      self_att_mask_v = active_entries_vitals.repeat(
          1, 1, x_v.size(1)
      ).unsqueeze(1)
      cross_att_mask_ot_v = (
          active_entries_vitals.squeeze(-1).unsqueeze(1)
          * active_entries_treat_outcomes
      ).unsqueeze(
          1
      )  # (B, 1, Tq, Tk)
      cross_att_mask_v_ot = (
          active_entries_treat_outcomes.squeeze(-1).unsqueeze(1)
          * active_entries_vitals
      ).unsqueeze(1)

      if (
          not self.disable_cross_attention
          and self.isolate_subnetwork != 't'
          and self.isolate_subnetwork != 'v'
      ):
        x_tv_ = self.cross_attention_to(
            x_t_,
            x_v,
            x_v,
            cross_att_mask_ot_v,
            True,
            prefix=self._fetch_prefix(prefix_list, 'cross_attention_to'),
        )
      else:
        x_tv_ = 0.0

      if (
          not self.disable_cross_attention
          and self.isolate_subnetwork != 'o'
          and self.isolate_subnetwork != 'v'
      ):
        x_ov_ = self.cross_attention_to(
            x_o_,
            x_v,
            x_v,
            cross_att_mask_ot_v,
            True,
            prefix=self._fetch_prefix(prefix_list, 'cross_attention_to'),
        )
      else:
        x_ov_ = 0.0

      x_v_ = self.self_attention_o(
          x_v,
          x_v,
          x_v,
          self_att_mask_v,
          True,
          prefix=self._fetch_prefix(prefix_list, 'self_attention_o'),
      )

      if (
          not self.disable_cross_attention
          and self.isolate_subnetwork != 'v'
          and self.isolate_subnetwork != 't'
      ):
        x_vt_ = self.cross_attention_ot(
            x_v_,
            x_t,
            x_t,
            cross_att_mask_v_ot,
            True,
            prefix=self._fetch_prefix(prefix_list, 'cross_attention_ot'),
        )
      else:
        x_vt_ = x_v_

      if (
          not self.disable_cross_attention
          and self.isolate_subnetwork != 'v'
          and self.isolate_subnetwork != 'o'
      ):
        x_vo_ = self.cross_attention_ot(
            x_v_,
            x_o,
            x_o,
            cross_att_mask_v_ot,
            True,
            prefix=self._fetch_prefix(prefix_list, 'cross_attention_ot'),
        )
      else:
        x_vo_ = 0.0

      out_t = self.feed_forwards[0](x_to_ + x_tv_ + x_s)
      out_o = self.feed_forwards[1](x_ot_ + x_ov_ + x_s)
      out_v = self.feed_forwards[2](x_vt_ + x_vo_ + x_s)

      return out_t, out_o, out_v


class TransformerSingleInputBlock(nn.Module):
  """Transformer block with single input."""

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
      n_inputs=1,
      final_layer=False,
      disable_cross_attention=False,
      isolate_subnetwork='',
      **kwargs,
  ):
    super().__init__()
    self.n_inputs = n_inputs
    self.disable_cross_attention = disable_cross_attention
    self.isolate_subnetwork = isolate_subnetwork

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

    self.feed_forwards = [
        PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        for _ in range(n_inputs)
    ]
    self.feed_forwards = nn.ModuleList(self.feed_forwards)

    self.n_inputs = n_inputs

    self.attention_block_name2idx = {
        name: idx for idx, name in enumerate(attention_block_names)
    }

  def _fetch_prefix(self, prefix_list, attn_name):
    if prefix_list is None:
      return None
    else:
      return prefix_list[self.attention_block_name2idx[attn_name]]

  def forward(
      self,
      x_tov,
      x_s,
      active_entries_treat_outcomes,
      active_entries_vitals=None,
      prefix_list=None,
  ):
    x_t = x_tov  # already concatenated
    self_att_mask_ot = active_entries_treat_outcomes.repeat(
        1, 1, x_t.size(1)
    ).unsqueeze(1)
    x_t_ = self.self_attention(
        x_t,
        x_t,
        x_t,
        self_att_mask_ot,
        True,
        prefix=self._fetch_prefix(prefix_list, 'self_attention_t'),
    )
    out_t = self.feed_forwards[0](x_t_ + x_s)
    return out_t
