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

"""Causal transformer encoder."""

import logging

from src.models.utils_transformer import AbsolutePositionalEncoding
from src.models.utils_transformer import RelativePositionalEncoding
from src.models.utils_transformer import TransformerMultiInputBlock
import torch
from torch import nn

logger = logging.getLogger(__name__)


class CTEncoder(nn.Module):
  """Causal transformer encoder."""

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
    self.basic_block_cls = TransformerMultiInputBlock
    self.max_seq_length = sub_args.max_seq_length
    self.seq_hidden_units = sub_args.seq_hidden_units
    self.fc_hidden_units = sub_args.fc_hidden_units
    self.dropout_rate = sub_args.dropout_rate
    self.num_layer = sub_args.num_layer
    self.num_heads = sub_args.num_heads
    self.attn_dropout = sub_args.attn_dropout
    self.disable_cross_attention = sub_args.disable_cross_attention
    self.isolate_subnetwork = sub_args.isolate_subnetwork
    self.self_positional_encoding_absolute = (
        sub_args.self_positional_encoding.absolute
    )
    self.self_positional_encoding_trainable = (
        sub_args.self_positional_encoding.trainable
    )
    self.self_positional_encoding_max_relative_position = (
        sub_args.self_positional_encoding.max_relative_position
    )
    if 'cross_positional_encoding' in sub_args:
      self.cross_positional_encoding_absolute = (
          sub_args.cross_positional_encoding.absolute
      )
      self.cross_positional_encoding_trainable = (
          sub_args.cross_positional_encoding.trainable
      )
      self.cross_positional_encoding_max_relative_position = (
          sub_args.cross_positional_encoding.max_relative_position
      )
    else:
      self.cross_positional_encoding_absolute = None
      self.cross_positional_encoding_trainable = None
      self.cross_positional_encoding_max_relative_position = None

    self.head_size = self.seq_hidden_units // self.num_heads

    # Init of positional encodings
    self.self_positional_encoding = (
        self.self_positional_encoding_k
    ) = self.self_positional_encoding_v = None
    if self.self_positional_encoding_absolute:
      self.self_positional_encoding = AbsolutePositionalEncoding(
          self.max_seq_length,
          self.seq_hidden_units,
          self.self_positional_encoding_trainable,
      )
    else:
      # Relative positional encoding is shared across heads
      self.self_positional_encoding_k = RelativePositionalEncoding(
          self.self_positional_encoding_max_relative_position,
          self.head_size,
          self.self_positional_encoding_trainable,
      )
      self.self_positional_encoding_v = RelativePositionalEncoding(
          self.self_positional_encoding_max_relative_position,
          self.head_size,
          self.self_positional_encoding_trainable,
      )

    self.cross_positional_encoding = (
        self.cross_positional_encoding_k
    ) = self.cross_positional_encoding_v = None
    if self.cross_positional_encoding_absolute:
      self.cross_positional_encoding = AbsolutePositionalEncoding(
          self.max_seq_length,
          self.seq_hidden_units,
          self.cross_positional_encoding_trainable,
      )
    elif self.cross_positional_encoding_max_relative_position is not None:
      # Relative positional encoding is shared across heads
      self.cross_positional_encoding_k = RelativePositionalEncoding(
          self.cross_positional_encoding_max_relative_position,
          self.head_size,
          self.cross_positional_encoding_trainable,
          cross_attn=True,
      )
      self.cross_positional_encoding_v = RelativePositionalEncoding(
          self.cross_positional_encoding_max_relative_position,
          self.head_size,
          self.cross_positional_encoding_trainable,
          cross_attn=True,
      )

    self.treatments_input_transformation = nn.Linear(
        self.dim_treatments, self.seq_hidden_units
    )
    self.vitals_input_transformation = (
        nn.Linear(self.dim_vitals, self.seq_hidden_units)
        if self.has_vitals
        else None
    )
    self.vitals_input_transformation = (
        nn.Linear(self.dim_vitals, self.seq_hidden_units)
        if self.has_vitals
        else None
    )
    self.outputs_input_transformation = nn.Linear(
        self.dim_outcome, self.seq_hidden_units
    )
    self.static_input_transformation = nn.Linear(
        self.dim_static_features, self.seq_hidden_units
    )

    self.n_inputs = (
        3 if self.has_vitals else 2
    )  # prev_outcomes and prev_treatments

    self.transformer_blocks = nn.ModuleList(
        [
            self.basic_block_cls(
                self.seq_hidden_units,
                self.num_heads,
                self.head_size,
                self.seq_hidden_units * 4,
                self.dropout_rate,
                self.dropout_rate if self.attn_dropout else 0.0,
                self_positional_encoding_k=self.self_positional_encoding_k,
                self_positional_encoding_v=self.self_positional_encoding_v,
                n_inputs=self.n_inputs,
                disable_cross_attention=self.disable_cross_attention,
                isolate_subnetwork=self.isolate_subnetwork,
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
    return self.build_br(
        prev_treatments,
        vitals,
        prev_outputs,
        static_features,
        active_entries,
        fixed_split=None,
        return_comp_reps=return_comp_reps,
    )

  def build_br(
      self,
      prev_treatments,
      vitals,
      prev_outputs,
      static_features,
      active_entries,
      fixed_split,
      return_comp_reps=False,
  ):
    active_entries_treat_outcomes = torch.clone(active_entries)
    active_entries_vitals = torch.clone(active_entries)

    if (
        fixed_split is not None and self.has_vitals
    ):  # Test sequence data / Train augmented data
      for i in range(len(active_entries)):
        # Masking vitals in range [fixed_split: ]
        active_entries_vitals[i, int(fixed_split[i]) :, :] = 0.0
        vitals[i, int(fixed_split[i]) :] = 0.0

    x_t = self.treatments_input_transformation(prev_treatments)
    x_o = self.outputs_input_transformation(prev_outputs)
    x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
    x_s = self.static_input_transformation(
        static_features.unsqueeze(1)
    )  # .expand(-1, x_t.size(1), -1)

    # if active_encoder_br is None and encoder_r is None:  # Only self-attention
    for _, block in enumerate(self.transformer_blocks):
      if self.self_positional_encoding is not None:
        x_t = x_t + self.self_positional_encoding(x_t)
        x_o = x_o + self.self_positional_encoding(x_o)
        x_v = (
            x_v + self.self_positional_encoding(x_v)
            if self.has_vitals
            else None
        )

      prefix_list = None

      if self.has_vitals:
        x_t, x_o, x_v = block(
            (x_t, x_o, x_v),
            x_s,
            active_entries_treat_outcomes,
            active_entries_vitals,
            prefix_list=prefix_list,
        )
      else:
        x_t, x_o = block(
            (x_t, x_o),
            x_s,
            active_entries_treat_outcomes,
            prefix_list=prefix_list,
        )

    if not self.has_vitals:
      x = (x_o + x_t) / 2
    else:
      if fixed_split is not None:  # Test seq data
        x = torch.empty_like(x_o)
        for i in range(len(active_entries)):
          # Masking vitals in range [fixed_split: ]
          x[i, : int(fixed_split[i])] = (
              x_o[i, : int(fixed_split[i])]
              + x_t[i, : int(fixed_split[i])]
              + x_v[i, : int(fixed_split[i])]
          ) / 3
          x[i, int(fixed_split[i]) :] = (
              x_o[i, int(fixed_split[i]) :] + x_t[i, int(fixed_split[i]) :]
          ) / 2
      else:  # Train data always has vitals
        x = (x_o + x_t + x_v) / 3

    if not self.has_vitals:
      comp_rep = [x_o, x_t]
    else:
      comp_rep = [x_o, x_t, x_v]

    output = self.output_dropout(x)

    if return_comp_reps:
      return output, comp_rep
    else:
      return output
