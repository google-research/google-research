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

"""Heterogeneous GNN library using Sparse Deferred."""

from typing import Any, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import sparse_deferred as sd


class TransformerFeedForward(nn.Module):
  """Two-layer feed-forward layer for use in a transformer.

  Attributes:
    dim: The embedding dimension of the transformer (input and output dims).
    dropout_rate: The frequency of dropout.
  """

  dim: int
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x, training):
    x = nn.Dense(4 * self.dim, use_bias=False)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.dim, use_bias=False)(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
    return x


class HeteroGraphTransformerBlock(nn.Module):
  """Transformer block corresponding to a single layer of a transformer.

  [Graph Transformer](https://arxiv.org/pdf/2012.09699) with
  [pre layer normalization](https://arxiv.org/pdf/2002.04745).

  Attributes:
    hidden_dim: The embedding dimension of the transformer (input and output
      dims).
    num_heads: The number of attention heads.
    dropout_rate: The frequency of dropout.
    precision: The precision of the dot product attention.
  """

  hidden_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  def setup(self):
    self.ln_1 = nn.LayerNorm(epsilon=1e-5)
    self.attn = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.hidden_dim,
        precision=self.precision,
        dropout_rate=self.dropout_rate,
    )
    self.ln_2 = nn.LayerNorm(epsilon=1e-5)
    self.ff = TransformerFeedForward(self.hidden_dim, self.dropout_rate)

  @classmethod
  def create_dense_mask(cls, adj):
    """Sparse matrix representing edges from source to target.

    Args:
      adj: Adjacency represented as a sd.SparseMatrix. The expected shape is [T,
        S] as sparse deferred assumes right multiplication with features (A @
        X).

    Returns:
      Dense mask with shape [S, T] (transpose of adj) to fit expected
        attention shape.
    """
    return HeteroGraphTransformerBlock.create_mask_from_edges(
        adj.shape, adj.row_ids, adj.col_ids
    )

  @classmethod
  def create_mask_from_edges(
      cls,
      shape,
      source_ids,
      target_ids,
      dtype=jnp.int32,
  ):
    mask = jnp.zeros(shape).astype(dtype)
    mask = mask.at[source_ids, target_ids].set(1)
    # Take the transpose b/c the adjacency matrix is formulated for right
    # multiplication with features, A @ X
    return mask.T

  def __call__(
      self,
      h_source,
      h_target,
      source_mask = None,
      target_mask = None,
      edge_mask = None,
      training = True,
  ):
    """Apply the transformer block to h_target.

    Returns h_target^{l+1}, the next level representation of h_target. The
    `h_source`, `h_target` argument order is just to express the familiar
    source->target relationship in the function signature. However, the output
    of this function will be the next level representations of the h_target.

    The mask controls the attention propagation over the graph structure, which
    can be built from a sd.SparseMatrix edge set.

    If h_source is [S, D] and h_target is [T, D] then the mask must have size
    [T, S].

    Args:
      h_source: Hidden representation of the source node set (passed as KV to
        attn).
      h_target: Hidden representation of the target node set (passed as Q to
        attn).
      source_mask: Optional mask with shape (S,) to account for padding on
        h_source. This is necessary since a row of all zeros on `edge_mask` will
        result in uniform attention due to softmax normalization.
      target_mask: Optional mask with shape (T,) to account for padding on
        h_target. This is necessary since a row of all zeros on `edge_mask` will
        result in uniform attention due to softmax normalization.
      edge_mask: Optional mask with shape (T, S) to control attention. Controls
        what targets can attend to which sources based on adjacency.
      training: Whether the model is in training mode (for dropout).

    Returns:
      h^{l+1} representation of h_target.
    """

    # Prenorm to bring h_source and h_target to similar magnitudes
    # It may be slightly better to have different layer norms for each node set
    # to account for different distributions.
    h_source = self.ln_1(h_source)
    h_target = self.ln_1(h_target)

    if source_mask is not None:
      if source_mask.shape[-1] != 1:
        # Add one to last dim to account for broadcasting.
        source_mask = jnp.expand_dims(source_mask, axis=-1)

    if target_mask is not None:
      if target_mask.shape[-1] != 1:
        # Add one to last dim to account for broadcasting.
        target_mask = jnp.expand_dims(target_mask, axis=-1)

    # Bias from LN implementation causes non-zero values for padding.
    # Remove them.
    if source_mask is not None:
      h_source = jnp.where(source_mask == 1, h_source, 0)

    if target_mask is not None:
      h_target = jnp.where(target_mask == 1, h_target, 0)

    attention = self.attn(
        inputs_q=h_target,
        inputs_kv=h_source,
        mask=edge_mask,
        deterministic=not training,
    )

    if target_mask is not None:
      attention = jnp.where(target_mask == 1, attention, 0)

    h_target = h_target + attention

    # It looks like the MHDPA impl has the final projection
    # (O matrix in the paper) implemented internally.
    h_target = self.ln_2(h_target)

    # Again get rid of ln bias in padded nodes.
    if target_mask is not None:
      h_target = jnp.where(target_mask == 1, h_target, 0)

    h_target = h_target + self.ff(h_target, training=training)

    return h_target
