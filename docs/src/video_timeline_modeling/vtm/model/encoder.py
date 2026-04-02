# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""The Transformer based encoder, proposed in the following paper.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan
N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
"Attention Is All You Need."
Advances in neural information processing systems 31 (2017).
"""

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
  """Positional Encoding module."""

  def __init__(self, d_model, max_len=24):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """Forward pass.

    Args:
      x: input embeddings.
    Shape:
      x: (B, N, num_input_hidden)

    Returns:
      The embeddings with added positional encodings with shape (B, N,
      num_input_hidden)
    """
    # Do not use += since the right side is a Variable with require_grad args
    x = x + Variable(
        self.pe.expand(x.shape[0], -1, -1)[:, :x.shape[1], :],
        requires_grad=False)
    return x


class Encoder(nn.Module):
  """Transformer based encoder."""

  def __init__(self, num_input_hidden, num_hidden, num_head, num_layers,
               dropout):
    super().__init__()
    self.num_input_hidden = num_input_hidden
    self.num_hidden = num_hidden
    self.num_head = num_head
    self.num_layers = num_layers
    encoder_layers = nn.TransformerEncoderLayer(num_input_hidden, num_head,
                                                num_hidden, dropout)
    self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

  def forward(self, batch_x, src_key_padding_mask=None):
    """Forward pass.

    Args:
      batch_x: input embeddings.
      src_key_padding_mask: mask for padding tokens.
    Shape:
      batch_x: (B, N, num_input_hidden)
      src_key_padding_mask: (B, N)

    Returns:
      The encoded embeddings with shape (B, N, num_input_hidden)
    """
    x_encoder = self.encoder(
        batch_x.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
    return x_encoder.permute(1, 0, 2)
