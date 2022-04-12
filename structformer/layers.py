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

"""Convolution and transformer layers used by StructFormer."""

import torch
from torch import nn
from torch.nn import init


def _get_activation_fn(activation):
  """Get specified activation function."""
  if activation == "relu":
    return nn.ReLU()
  elif activation == "gelu":
    return nn.GELU()
  elif activation == "leakyrelu":
    return nn.LeakyReLU()

  raise RuntimeError(
      "activation should be relu/gelu, not {}".format(activation))


class Conv1d(nn.Module):
  """1D convolution layer."""

  def __init__(self, hidden_size, kernel_size, dilation=1):
    """Initialization.

    Args:
      hidden_size: dimension of input embeddings
      kernel_size: convolution kernel size
      dilation: the spacing between the kernel points
    """
    super(Conv1d, self).__init__()

    if kernel_size % 2 == 0:
      padding = (kernel_size // 2) * dilation
      self.shift = True
    else:
      padding = ((kernel_size - 1) // 2) * dilation
      self.shift = False
    self.conv = nn.Conv1d(
        hidden_size,
        hidden_size,
        kernel_size,
        padding=padding,
        dilation=dilation)

  def forward(self, x):
    """Compute convolution.

    Args:
      x: input embeddings
    Returns:
      conv_output: convolution results
    """

    if self.shift:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
    else:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiheadAttention(nn.Module):
  """Multi-head self-attention layer."""

  def __init__(self,
               embed_dim,
               num_heads,
               dropout=0.,
               bias=True,
               v_proj=True,
               out_proj=True,
               relative_bias=True):
    """Initialization.

    Args:
      embed_dim: dimension of input embeddings
      num_heads: number of self-attention heads
      dropout: dropout rate
      bias: bool, indicate whether include bias for linear transformations
      v_proj: bool, indicate whether project inputs to new values
      out_proj: bool, indicate whether project outputs to new values
      relative_bias: bool, indicate whether use a relative position based
        attention bias
    """

    super(MultiheadAttention, self).__init__()
    self.embed_dim = embed_dim

    self.num_heads = num_heads
    self.drop = nn.Dropout(dropout)
    self.head_dim = embed_dim // num_heads
    assert self.head_dim * num_heads == self.embed_dim, ("embed_dim must be "
                                                         "divisible by "
                                                         "num_heads")

    self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    if v_proj:
      self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    else:
      self.v_proj = nn.Identity()

    if out_proj:
      self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    else:
      self.out_proj = nn.Identity()

    if relative_bias:
      self.relative_bias = nn.Parameter(torch.zeros((self.num_heads, 512)))
    else:
      self.relative_bias = None

    self._reset_parameters()

  def _reset_parameters(self):
    """Initialize attention parameters."""

    init.xavier_uniform_(self.q_proj.weight)
    init.constant_(self.q_proj.bias, 0.)

    init.xavier_uniform_(self.k_proj.weight)
    init.constant_(self.k_proj.bias, 0.)

    if isinstance(self.v_proj, nn.Linear):
      init.xavier_uniform_(self.v_proj.weight)
      init.constant_(self.v_proj.bias, 0.)

    if isinstance(self.out_proj, nn.Linear):
      init.xavier_uniform_(self.out_proj.weight)
      init.constant_(self.out_proj.bias, 0.)

  def forward(self, query, key_padding_mask=None, attn_mask=None):
    """Compute multi-head self-attention.

    Args:
      query: input embeddings
      key_padding_mask: 3D mask that prevents attention to certain positions
      attn_mask: 3D mask that rescale the attention weight at each position
    Returns:
      attn_output: self-attention output
    """

    length, bsz, embed_dim = query.size()
    assert embed_dim == self.embed_dim

    head_dim = embed_dim // self.num_heads
    assert head_dim * self.num_heads == embed_dim, ("embed_dim must be "
                                                    "divisible by num_heads")
    scaling = float(head_dim)**-0.5

    q = self.q_proj(query)
    k = self.k_proj(query)
    v = self.v_proj(query)

    q = q * scaling

    if attn_mask is not None:
      assert list(attn_mask.size()) == [bsz * self.num_heads,
                                        query.size(0), query.size(0)]

    q = q.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)
    k = k.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)
    v = v.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(
        attn_output_weights.size()) == [bsz * self.num_heads, length, length]

    if self.relative_bias is not None:
      pos = torch.arange(length, device=query.device)
      relative_pos = torch.abs(pos[:, None] - pos[None, :]) + 256
      relative_pos = relative_pos[None, :, :].expand(bsz * self.num_heads, -1,
                                                     -1)

      relative_bias = self.relative_bias.repeat_interleave(bsz, dim=0)
      relative_bias = relative_bias[:, None, :].expand(-1, length, -1)
      relative_bias = torch.gather(relative_bias, 2, relative_pos)
      attn_output_weights = attn_output_weights + relative_bias

    if key_padding_mask is not None:
      attn_output_weights = attn_output_weights + key_padding_mask

    if attn_mask is None:
      attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
    else:
      attn_output_weights = torch.sigmoid(attn_output_weights) * attn_mask

    attn_output_weights = self.drop(attn_output_weights)

    attn_output = torch.bmm(attn_output_weights, v)

    assert list(attn_output.size()) == [bsz * self.num_heads, length, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        length, bsz, embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output


class TransformerLayer(nn.Module):
  """TransformerEncoderLayer is made up of self-attn and feedforward network."""

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               dropatt=0.1,
               activation="leakyrelu",
               relative_bias=True):
    """Initialization.

    Args:
      d_model: dimension of inputs
      nhead: number of self-attention heads
      dim_feedforward: dimension of hidden layer in feedforward layer
      dropout: dropout rate
      dropatt: drop attention rate
      activation: activation function
      relative_bias: bool, indicate whether use a relative position based
        attention bias
    """

    super(TransformerLayer, self).__init__()
    self.self_attn = MultiheadAttention(
        d_model, nhead, dropout=dropatt, relative_bias=relative_bias)
    # Implementation of Feedforward model
    self.feedforward = nn.Sequential(
        nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward),
        _get_activation_fn(activation), nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model))

    self.norm = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.nhead = nhead

  def forward(self, src, attn_mask=None, key_padding_mask=None):
    """Pass the input through the encoder layer.

    Args:
      src: the sequence to the encoder layer (required).
      attn_mask: the mask for the src sequence (optional).
      key_padding_mask: the mask for the src keys per batch (optional).
    Returns:
      src3: the output of transformer layer, share the same shape as src.
    """
    src2 = self.self_attn(
        self.norm(src), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    src2 = src + self.dropout1(src2)
    src3 = self.feedforward(src2)
    src3 = src2 + self.dropout2(src3)

    return src3
