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

"""Attention module library."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
from invariant_slot_attention.modules import misc

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class SlotAttention(nn.Module):
  """Slot Attention module.

  Note: This module uses pre-normalization by default.
  """

  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  num_heads: int = 1

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention module forward pass."""
    del padding_mask, train  # Unused.

    qkv_size = self.qkv_size or slots.shape[-1]
    head_dim = qkv_size // self.num_heads
    dense = functools.partial(nn.DenseGeneral,
                              axis=-1, features=(self.num_heads, head_dim),
                              use_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = nn.LayerNorm()
    inverted_attention = InvertedDotProductAttention(
        norm_type="mean", multi_head=self.num_heads > 1)
    gru = misc.GRU()

    if self.mlp_size is not None:
      mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    # inputs.shape = (..., n_inputs, inputs_size).
    inputs = nn.LayerNorm()(inputs)
    # k.shape = (..., n_inputs, slot_size).
    k = dense(name="general_dense_k_0")(inputs)
    # v.shape = (..., n_inputs, slot_size).
    v = dense(name="general_dense_v_0")(inputs)

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).
      updates = inverted_attention(query=q, key=k, value=v)

      # Recurrent update.
      slots = gru(slots, updates)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None:
        slots = mlp(slots)

    return slots


class InvertedDotProductAttention(nn.Module):
  """Inverted version of dot-product attention (softmax over query axis)."""

  norm_type: Optional[str] = "mean"  # mean, layernorm, or None
  multi_head: bool = False
  epsilon: float = 1e-8
  dtype: DType = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  return_attn_weights: bool = False

  @nn.compact
  def __call__(self, query, key, value,
               train = False):
    """Computes inverted dot-product attention.

    Args:
      query: Queries with shape of `[batch..., q_num, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, qk_features]`.
      value: Values with shape of `[batch..., kv_num, v_features]`.
      train: Indicating whether we're training or evaluating.

    Returns:
      Output of shape `[batch_size..., n_queries, v_features]`
    """
    del train  # Unused.

    attn = GeneralizedDotProductAttention(
        inverted_attn=True,
        renormalize_keys=True if self.norm_type == "mean" else False,
        epsilon=self.epsilon,
        dtype=self.dtype,
        precision=self.precision,
        return_attn_weights=True)

    # Apply attention mechanism.
    output, attn = attn(query=query, key=key, value=value)

    if self.multi_head:
      # Multi-head aggregation. Equivalent to concat + dense layer.
      output = nn.DenseGeneral(features=output.shape[-1], axis=(-2, -1))(output)
    else:
      # Remove head dimension.
      output = jnp.squeeze(output, axis=-2)
      attn = jnp.squeeze(attn, axis=-3)

    if self.norm_type == "layernorm":
      output = nn.LayerNorm()(output)

    if self.return_attn_weights:
      return output, attn

    return output


class GeneralizedDotProductAttention(nn.Module):
  """Multi-head dot-product attention with customizable normalization axis.

  This module supports logging of attention weights in a variable collection.
  """

  dtype: DType = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  epsilon: float = 1e-8
  inverted_attn: bool = False
  renormalize_keys: bool = False
  attn_weights_only: bool = False
  return_attn_weights: bool = False

  @nn.compact
  def __call__(self, query, key, value,
               train = False, **kwargs
               ):
    """Computes multi-head dot-product attention given query, key, and value.

    Args:
      query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
      key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
      value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
      train: Indicating whether we're training or evaluating.
      **kwargs: Additional keyword arguments are required when used as attention
        function in nn.MultiHeadDotProductAttention, but they will be ignored
        here.

    Returns:
      Output of shape `[batch..., q_num, num_heads, v_features]`.
    """

    assert query.ndim == key.ndim == value.ndim, (
        "Queries, keys, and values must have the same rank.")
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        "Query, key, and value batch dimensions must match.")
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        "Query, key, and value num_heads dimensions must match.")
    assert key.shape[-3] == value.shape[-3], (
        "Key and value cardinality dimensions must match.")
    assert query.shape[-1] == key.shape[-1], (
        "Query and key feature dimensions must match.")

    if kwargs.get("bias") is not None:
      raise NotImplementedError(
          "Support for masked attention is not yet implemented.")

    if "dropout_rate" in kwargs:
      if kwargs["dropout_rate"] > 0.:
        raise NotImplementedError("Support for dropout is not yet implemented.")

    # Temperature normalization.
    qk_features = query.shape[-1]
    query = query / jnp.sqrt(qk_features).astype(self.dtype)

    # attn.shape = (batch..., num_heads, q_num, kv_num)
    attn = jnp.einsum("...qhd,...khd->...hqk", query, key,
                      precision=self.precision)

    if self.inverted_attn:
      attention_axis = -2  # Query axis.
    else:
      attention_axis = -1  # Key axis.

    # Softmax normalization (by default over key axis).
    attn = jax.nn.softmax(attn, axis=attention_axis).astype(self.dtype)

    # Defines intermediate for logging.
    if not train:
      self.sow("intermediates", "attn", attn)

    if self.renormalize_keys:
      # Corresponds to value aggregation via weighted mean (as opposed to sum).
      normalizer = jnp.sum(attn, axis=-1, keepdims=True) + self.epsilon
      attn = attn / normalizer

    if self.attn_weights_only:
      return attn

    # Aggregate values using a weighted sum with weights provided by `attn`.
    output = jnp.einsum(
        "...hqk,...khd->...qhd", attn, value, precision=self.precision)

    if self.return_attn_weights:
      return output, attn

    return output


class Transformer(nn.Module):
  """Transformer with multiple blocks."""

  num_heads: int
  qkv_size: int
  mlp_size: int
  num_layers: int
  pre_norm: bool = False

  @nn.compact
  def __call__(self, queries, inputs = None,
               padding_mask = None,
               train = False):
    x = queries
    for lyr in range(self.num_layers):
      x = TransformerBlock(
          num_heads=self.num_heads, qkv_size=self.qkv_size,
          mlp_size=self.mlp_size, pre_norm=self.pre_norm,
          name=f"TransformerBlock{lyr}")(  # pytype: disable=wrong-arg-types
              x, inputs, padding_mask, train)
    return x


class TransformerBlock(nn.Module):
  """Transformer decoder block."""

  num_heads: int
  qkv_size: int
  mlp_size: int
  pre_norm: bool = False

  @nn.compact
  def __call__(self, queries, inputs = None,
               padding_mask = None,
               train = False):
    del padding_mask  # Unused.
    assert queries.ndim == 3

    attention_fn = GeneralizedDotProductAttention()

    attn = functools.partial(
        nn.MultiHeadDotProductAttention,
        num_heads=self.num_heads,
        qkv_features=self.qkv_size,
        attention_fn=attention_fn)

    mlp = misc.MLP(hidden_size=self.mlp_size)  # type: ignore

    if self.pre_norm:
      # Self-attention on queries.
      x = nn.LayerNorm()(queries)
      x = attn()(inputs_q=x, inputs_kv=x, deterministic=not train)
      x = x + queries

      # Cross-attention on inputs.
      if inputs is not None:
        assert inputs.ndim == 3
        y = nn.LayerNorm()(x)
        y = attn()(inputs_q=y, inputs_kv=inputs, deterministic=not train)
        y = y + x
      else:
        y = x

      # MLP
      z = nn.LayerNorm()(y)
      z = mlp(z, train)
      z = z + y
    else:
      # Self-attention on queries.
      x = queries
      x = attn()(inputs_q=x, inputs_kv=x, deterministic=not train)
      x = x + queries
      x = nn.LayerNorm()(x)

      # Cross-attention on inputs.
      if inputs is not None:
        assert inputs.ndim == 3
        y = attn()(inputs_q=x, inputs_kv=inputs, deterministic=not train)
        y = y + x
        y = nn.LayerNorm()(y)
      else:
        y = x

      # MLP.
      z = mlp(y, train)
      z = z + y
      z = nn.LayerNorm()(z)
    return z
