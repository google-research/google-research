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

"""Model Architecture."""

import functools
from typing import Any, Callable

from flax import linen as nn
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  dataset_fn: str = "sudoku"  # Choices=['sudoku', 'othello']
  vocab_size: int = 1
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  seq_len: int = 2048  # Maximum sequence length
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  kernel_init: Callable[Ellipsis, Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[Ellipsis, Any] = nn.initializers.normal(stddev=1e-6)


class TransformerBlock(nn.Module):
  """A Transformer block which includes self attention and MLP layers.

  Attributes:
    vocab_size: size of the token vocabulary
    emb_dim: embedding dimension
    num_layers: number of layers
    config: model config as a ConfigDict
  """
  config: Any = None

  def setup(self):
    self.vocab_size = self.config.vocab_size
    self.emb_dim = self.config.emb_dim
    self.num_layers = self.config.num_layers

  @nn.compact
  def __call__(self, inputs, causal_mask_inputs, use_cache=True, training=True):

    print(inputs.shape, causal_mask_inputs.shape)
    x = inputs + nn.SelfAttention(
        num_heads=self.config.num_heads, dtype=self.config.dtype,
        qkv_features=self.config.qkv_dim,
        kernel_init=self.config.kernel_init,
        bias_init=self.config.bias_init, use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.config.attention_dropout_rate, normalize_qk=True,
        deterministic=self.config.deterministic)(inputs, causal_mask_inputs)

    def mlp(x):
      dense_with_init = functools.partial(
          nn.Dense,
          kernel_init=self.config.kernel_init,
          bias_init=self.config.bias_init
          )
      x = dense_with_init(features=self.config.mlp_dim)(x)
      x = nn.gelu(x)
      x = dense_with_init(features=self.config.emb_dim)(x)
      x = nn.Dropout(rate=self.config.dropout_rate,
                     deterministic=self.config.deterministic)(x)
      return x

    x = x + mlp(x)
    return x


class TransformerLMHeadModel(nn.Module):
  """A Transformer based Language Model.

  Attributes:
    vocab_size: size of the token vocabulary
    emb_dim: embedding dimension
    num_layers: number of layers
    config: model config as a ConfigDict
  """
  config: Any = None

  def setup(self):
    self.vocab_size = self.config.vocab_size
    self.emb_dim = self.config.emb_dim
    self.num_layers = self.config.num_layers

  @nn.compact
  def __call__(self, inputs, use_cache=True, training=True):
    batch_size, seq_size = inputs.shape

    causal_mask_x = nn.make_causal_mask(inputs, dtype=self.config.dtype)

    embed_with_init = functools.partial(
        nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))
    token_embeddings = embed_with_init(
        num_embeddings=self.config.vocab_size,
        features=self.config.emb_dim,
    )(inputs)

    assert token_embeddings.shape == (batch_size, seq_size,
                                      self.config.emb_dim)

    pos_embedding_variable = self.variable(
        "params",
        "position_embeddings",
        jnp.zeros,
        (self.config.seq_len, self.config.emb_dim),
    )
    pos_embeddings = pos_embedding_variable.value[:seq_size, :]
    output_tuple = (pos_embeddings.shape, token_embeddings.shape[1:])
    assert pos_embeddings.shape == token_embeddings.shape[1:], output_tuple

    x = token_embeddings + pos_embeddings[None, :, :]
    x = nn.Dropout(rate=self.config.dropout_rate,
                   deterministic=self.config.deterministic)(x)

    for _ in range(self.num_layers):
      x = TransformerBlock(config=self.config)(
          x, causal_mask_x, use_cache=True, training=training)

    x = nn.LayerNorm()(x)
    logits = nn.Dense(features=self.config.vocab_size,
                      kernel_init=self.config.kernel_init,
                      bias_init=self.config.bias_init,
                      use_bias=False)(x)

    assert logits.shape == (batch_size, seq_size, self.config.vocab_size)
    return logits
