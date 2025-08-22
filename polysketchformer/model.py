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

"""Defines a decoder-only transformer model."""

import functools
from typing import Optional, Tuple

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from polysketchformer import linear_squared_attention
from polysketchformer import polynomial_attention as pa
from polysketchformer import polysketch_attention as psa
from polysketchformer import sketch_layer as skl
from polysketchformer import utils


@struct.dataclass
class TransformerConfig:
  """Model parameters."""

  vocab_size: int
  context_length: int
  emb_dim: int
  num_heads: int
  num_layers: int
  dropout_rate: float
  attention: str = 'softmax'
  power: Optional[float] = None
  sketch_size: Optional[int] = None
  grain_size: Optional[int] = None
  sketch_key: Optional[jax.Array] = None  # For initializing random sketches
  checkpoint_attention: bool = True


class GLU(nn.Module):
  """Gated Linear Unit.

  See https://arxiv.org/pdf/2002.05202.pdf and references therein.

  Attributes:
    emb_dim: Output dimension of the gated linear unit
    dropout_rate: Dropout rate
  """

  emb_dim: int
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, training):
    hidden_dim = (8 * self.emb_dim // 3) // 8 * 8  # Ensures hidden_dim % 8 == 0
    # hidden_dim is chosen to make the number of parameters close to that of
    # the a simple FeedForward layer with an expansion factor 4.
    x = nn.Dense(2 * hidden_dim, use_bias=False)(x)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x2 = nn.gelu(x2)
    x = x1 * x2
    x = nn.Dense(self.emb_dim, use_bias=False)(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
    return x


class MultiHeadAttention(nn.Module):
  """Projection + self attention layer.

  Implementation of the multi-head attention. Projects the input vectors to
  obtain Q, K, V matrices and then applies the self attention mechanism to
  obtain outputs. Finally, applies a projection on the output of the self
  attention mechanism.

  Attributes:
    num_heads: Number of attention heads
    emb_dim: Dimension of the input vectors
    causal: True/False denoting if causal masking is to be applied
    attention: Attention mechanism to be used. Accepted inputs: 'softmax',
      'polynomial', 'random_sketch', 'learned_sketch'
    power: Degree to be used for 'polynomial' attention
    random_key: Key to be used to generate sketches if attention ==
      'random_sketch'
    grain_size: block size to be used in lower triangular algorithm
    dropout_rate:  Dropout rate
    checkpoint_attention: Whether attention is to be recomputed during backward
      pass to decrease the memory required.
  """

  num_heads: int
  emb_dim: int
  context_length: int
  causal: bool = False
  attention: str = 'softmax'
  power: Optional[float] = None
  random_key: Optional[jax.Array] = None
  sketch_size: Optional[int] = None
  grain_size: Optional[int] = None
  dropout_rate: Optional[float] = None
  checkpoint_attention: bool = False

  def setup(self):
    self.head_size = self.emb_dim // self.num_heads
    self.rope = utils.RoPE(self.context_length, self.head_size)
    self.attn = lambda q, k, v: v  # Passthrough the value
    self.transpose = True  # q, k, v of shape [batch, heads, length, head_sz]
    self.pre_attn_layer = None
    self.mixed = False

    if self.attention == 'polynomial':
      self.pre_attn_layer = nn.LayerNorm(epsilon=1e-5)
      self.attn = functools.partial(
          pa.polynomial_attention,
          degree=self.power,
          is_causal=self.causal,
      )
    elif self.attention == 'softmax':
      self.attn = functools.partial(
          nn.dot_product_attention,
          deterministic=True,
          mask=nn.make_causal_mask(
              jnp.ones(self.context_length) if self.causal else None
          ),
      )
      self.transpose = False  # q, k, v in shape [batch, length, heads, head_sz]

    elif self.attention == 'random_sketch':
      self.pre_attn_layer = nn.LayerNorm(epsilon=1e-5)
      self.attn = psa.make_polysketch_attn_fn(
          feature_dimension=self.head_size,
          sketch_key=self.random_key,
          sketch_size=self.sketch_size,
          is_causal=self.causal,
          grain_size=self.grain_size,
      )

    elif self.attention == 'learned_sketch':
      self.pre_attn_layer = skl.SketchLayer(
          sketch_dim=self.sketch_size, sketch_levels=2, expansion_factor=8
      )
      self.attn = linear_squared_attention.make_squared_attn_fn(
          is_causal=self.causal, grain_size=self.grain_size
      )
    elif self.attention == 'learned_sketch_mixed':
      # Note self.attn handles both sketching and mixing exact attention
      self.mixed = True
      self.pre_attn_layer = skl.SketchLayer(
          sketch_dim=self.sketch_size, sketch_levels=2, expansion_factor=8
      )
      self.attn = linear_squared_attention.make_mixed_squared_attn_fn(
          is_causal=self.causal, grain_size=self.grain_size
      )
    elif self.attention == 'random_sketch_mixed':
      # Note self.attn handles both sketching and mixing exact attention
      self.mixed = True
      self.pre_attn_layer = nn.LayerNorm(epsilon=1e-5)
      self.attn = psa.make_mixed_polysketch_attn_fn(
          feature_dimension=self.head_size,
          sketch_key=self.random_key,
          sketch_size=self.sketch_size,
          is_causal=self.causal,
          grain_size=self.grain_size,
      )
    else:
      raise ValueError(f'Unknown attention: {self.attention}')

    if self.checkpoint_attention:
      self.attn = jax.checkpoint(self.attn)

  @nn.compact
  def __call__(self, x, training):
    """Applies the multi-head attention.

    Args:
      x: An array of shape [batch, context_length, emb_dim]
      training: True/False denoting training/inference

    Returns:
      An array of shape same as x.
    """
    batch_size, context_length, _ = x.shape

    projection = nn.Dense(3 * self.emb_dim, use_bias=False)(x)
    q, k, v = jnp.array_split(
        projection, 3, axis=-1
    )  # Each has shape [..., n, h * d]

    q = jnp.reshape(q, (batch_size, context_length, self.num_heads, -1))
    k = jnp.reshape(k, (batch_size, context_length, self.num_heads, -1))
    v = jnp.reshape(v, (batch_size, context_length, self.num_heads, -1))

    # Apply RoPE
    q = self.rope.apply(q)
    k = self.rope.apply(k)

    if self.transpose:
      # Switch head and context_length dimensions
      q = q.transpose((0, 2, 1, 3))
      k = k.transpose((0, 2, 1, 3))
      v = v.transpose((0, 2, 1, 3))

    q_pre_sketch, k_pre_sketch = None, None

    if self.mixed:
      # Apply layer norm to original query and key vectors and store them.
      q_pre_sketch = nn.LayerNorm(epsilon=1e-5)(q)
      k_pre_sketch = nn.LayerNorm(epsilon=1e-5)(k)

    if self.pre_attn_layer is not None:
      # Compute the transformed vectors
      q = self.pre_attn_layer(q)
      k = self.pre_attn_layer(k)

    if not self.mixed:
      output = self.attn(q, k, v)
    else:
      output = self.attn(q, k, v, q_pre_sketch, k_pre_sketch)

    if self.transpose:
      # Switch head and context_length dimensions back
      output = output.transpose((0, 2, 1, 3))

    output = output.reshape((batch_size, context_length, -1))
    output = nn.Dense(self.emb_dim, use_bias=False)(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=not training
    )
    return output


class Block(nn.Module):
  """Transformer block.

  Applies Self-Attention and a Gated Linear Unit using skip connections.

  Attributes:
    config: Of type TransformerConfig which defines the model structure
    layer_num: Index of the layer used to derive sketch_key for this layer
  """

  config: TransformerConfig
  layer_num: int

  def setup(self):
    random_key = (
        jax.random.fold_in(self.config.sketch_key, self.layer_num)
        if self.config.sketch_key is not None
        else None
    )
    self.attn = MultiHeadAttention(
        num_heads=self.config.num_heads,
        emb_dim=self.config.emb_dim,
        context_length=self.config.context_length,
        causal=True,
        attention=self.config.attention,
        power=self.config.power,
        random_key=random_key,
        sketch_size=self.config.sketch_size,
        grain_size=self.config.grain_size,
        dropout_rate=self.config.dropout_rate,
        checkpoint_attention=self.config.checkpoint_attention,
    )
    self.ln_1 = nn.LayerNorm(epsilon=1e-5)
    self.ln_2 = nn.LayerNorm(epsilon=1e-5)
    self.ff = GLU(self.config.emb_dim, self.config.dropout_rate)

  def __call__(self, x, training):
    """Applies Self-attention followed by a feedforward layer to the inputs.

    Args:
      x: Input array of shape [batch,..., context_length, emb_dim]
      training: True/False denoting training/inference

    Returns:
      An array of the same shape as the input x
    """
    x = x + self.attn(self.ln_1(x), training=training)
    x = x + self.ff(self.ln_2(x), training=training)
    return x


class Transformer(nn.Module):
  """Decoder-only transformer model.

  Attributes:
    config: Of type TransformerConfig which defines the structure of the model.
  """

  config: TransformerConfig

  def setup(self):
    self.wte = nn.Embed(self.config.vocab_size, self.config.emb_dim)
    self.pe = utils.sinusoidal_position_embedding(
        self.config.context_length, self.config.emb_dim
    )
    self.ln = nn.LayerNorm(epsilon=1e-5)
    self.softmax_bias = self.param(
        'softmax_bias', nn.initializers.zeros_init(), (self.config.vocab_size,)
    )  # Learnable bias to add to logits

  @nn.compact
  def __call__(
      self, idx, training
  ):
    """Applies the transformer model to the input tokens.

    Args:
      idx: Input of shape [batch,...,context_length] of dtype jnp.int32. Each
           entry of idx must be in the interval [0, config.context_length]
      training: True/False denoting training/inference

    Returns:
      logits: An array of shape idx.shape + (config.context_length, ) giving
              unnormalized logits
      cache: A cache that can be used to speedup inference. *Unimplemented*
    """
    token_embd = self.wte(idx)
    x = token_embd + self.pe

    for i in range(self.config.num_layers):
      x = Block(self.config, i)(x, training)

    x = self.ln(x)
    logits = self.wte.attend(x) + self.softmax_bias
    return logits, None
