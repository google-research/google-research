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

"""BERT-style layout generation network."""

from typing import Callable, Dict, Iterable, Optional, Text, Tuple

from . import simplified_bert
from flax import linen as nn
import jax.numpy as jnp

# BERT layer norm
TF_LAYERNORM_EPSILON = 1e-12

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


class LayoutEmbed(nn.Module):
  """Layout embeddings.

  There are four types of embeddings for layout generation.
  1. word_embedder: embeddings for layout sequence tokens.
  2. position_embeder: embeddings for differnt positions in the layout sequence.
  3. asset_num_embedder: embeddings of how many assets in the layout sequences.
    Different asset numbers denote differnt sizes and positions of these assets.
  4. asset_embedder: tokens belonging to the same asset will share the same
    embedding, similar to the segment embedding in the BERT.

  Attributes:
    use_vertical: whether add vertical info nor not.
    embedding_size: embedding dimension.
    hidden_dropout_prob: dropout rate.
    vocab_size: vocabulary size.
    max_position_embeddings: maximum positions in layout sequences.
    initializer_fn: embedding initialization method.
    layout_dim: dimension of layout.
    hidden_size: embedding dimension projected length.
  """
  use_vertical: bool
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  layout_dim: int
  hidden_size: Optional[int] = None

  def setup(self):
    # Token embeddings.
    self.word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    # Position embeddings.
    self.position_embedder = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')
    # How many assets in the layout sample.
    self.asset_num_embdder = nn.Embed(
        num_embeddings=50,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='asset_num_embeddings')
    # Asset segment embeddings.
    self.asset_embedder = nn.Embed(
        num_embeddings=50,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='asset_embeddings')
    if self.use_vertical:
      # Vertical info embeddings.
      self.label_embedder = nn.Embed(
          num_embeddings=32,
          features=self.embedding_size,
          embedding_init=self.initializer_fn,
          name='label_embedding')

  @nn.compact
  def __call__(self,
               input_ids,
               labels,
               deterministic):
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]
    asset_ids = position_ids // (self.layout_dim * 2 + 1)
    asset_num = jnp.expand_dims(
        jnp.sum(input_ids != 0, axis=1) // (self.layout_dim * 2 + 1), 1)

    word_embeddings = self.word_embedder(input_ids)
    # position_embeddings = self.position_embedder(position_ids)
    asset_embeddings = self.asset_embedder(asset_ids)
    asset_num_embeddings = self.asset_num_embdder(asset_num)
    input_embeddings = word_embeddings + asset_embeddings + asset_num_embeddings
    if labels is not None and self.use_vertical:
      labels = labels.astype('int32')
      label_emb = self.label_embedder(labels)
      input_embeddings += label_emb

    input_embeddings = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON,
        name='embeddings_ln')(input_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class NALayoutNet(nn.Module):
  """BERT as a Flax module."""
  use_vertical: bool
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  initializer_range: float = 0.02
  pad_token_id: int = -1
  layout_dim: int = 2

  def setup(self):
    self.embedder = LayoutEmbed(
        use_vertical=self.use_vertical,
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=simplified_bert.truncated_normal(self.initializer_range),
        layout_dim=self.layout_dim)

  @nn.compact
  def __call__(self,
               input_ids,
               labels,
               deterministic = True):
    # We assume that all pad tokens should be masked out.
    input_ids = input_ids.astype('int32')
    input_mask = jnp.asarray(input_ids != 0, dtype=jnp.int32)

    input_embeddings = self.embedder(
        input_ids=input_ids, labels=labels, deterministic=deterministic)

    # Stack BERT layers.
    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = simplified_bert.BertLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=simplified_bert.truncated_normal(
              self.initializer_range))(
                  layer_input=layer_input,
                  input_mask=input_mask,
                  deterministic=deterministic)
      layer_input = layer_output

    # Word embedding weights.
    word_embedding_matrix = self.variables['params']['embedder'][
        'word_embeddings']['embedding']
    logits = simplified_bert.BertMlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=simplified_bert.truncated_normal(
            self.initializer_range))(
                last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits
