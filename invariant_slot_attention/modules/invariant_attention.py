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

"""Equivariant attention module library."""
import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
from invariant_slot_attention.modules import attention
from invariant_slot_attention.modules import misc

Shape = Tuple[int]

DType = Any
Array = Any  # jnp.ndarray
PRNGKey = Array


class InvertedDotProductAttentionKeyPerQuery(nn.Module):
  """Inverted dot-product attention with a different set of keys per query.

  Used in SlotAttentionTranslEquiv, where each slot has a position.
  The positions are used to create relative coordinate grids,
  which result in a different set of inputs (keys) for each slot.
  """

  dtype: DType = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  epsilon: float = 1e-8
  renormalize_keys: bool = False
  attn_weights_only: bool = False
  softmax_temperature: float = 1.0
  value_per_query: bool = False

  @nn.compact
  def __call__(self, query, key, value, train):
    """Computes inverted dot-product attention with key per query.

    Args:
      query: Queries with shape of `[batch..., q_num, qk_features]`.
      key: Keys with shape of `[batch..., q_num, kv_num, qk_features]`.
      value: Values with shape of `[batch..., kv_num, v_features]`.
      train: Indicating whether we're training or evaluating.

    Returns:
      Tuple of two elements: (1) output of shape
      `[batch_size..., q_num, v_features]` and (2) attention mask of shape
      `[batch_size..., q_num, kv_num]`.
    """
    qk_features = query.shape[-1]
    query = query / jnp.sqrt(qk_features).astype(self.dtype)

    # Each query is multiplied with its own set of keys.
    attn = jnp.einsum(
        "...qd,...qkd->...qk", query, key, precision=self.precision
    )

    # axis=-2 for a softmax over query axis (inverted attention).
    attn = jax.nn.softmax(
        attn / self.softmax_temperature, axis=-2
    ).astype(self.dtype)

    # We expand dims because the logger expect a #heads dimension.
    self.sow("intermediates", "attn", jnp.expand_dims(attn, -3))

    if self.renormalize_keys:
      normalizer = jnp.sum(attn, axis=-1, keepdims=True) + self.epsilon
      attn = attn / normalizer

    if self.attn_weights_only:
      return attn

    output = jnp.einsum(
        "...qk,...qkd->...qd" if self.value_per_query else "...qk,...kd->...qd",
        attn,
        value,
        precision=self.precision
    )

    return output, attn


class SlotAttentionExplicitStats(nn.Module):
  """Slot Attention module with explicit slot statistics.

  Slot statistics, such as position and scale, are appended to the
  output slot representations.

  Note: This module expects a 2D coordinate grid to be appended
  at the end of inputs.

  Note: This module uses pre-normalization by default.
  """
  grid_encoder: Callable[[], nn.Module]
  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  softmax_temperature: float = 1.0
  gumbel_softmax: bool = False
  gumbel_softmax_straight_through: bool = False
  num_heads: int = 1
  min_scale: float = 0.01
  max_scale: float = 5.
  return_slot_positions: bool = True
  return_slot_scales: bool = True

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention with explicit slot statistics module forward pass."""
    del padding_mask  # Unused.
    # Slot scales require slot positions.
    assert self.return_slot_positions or not self.return_slot_scales

    # Separate a concatenated linear coordinate grid from the inputs.
    inputs, grid = inputs[Ellipsis, :-2], inputs[Ellipsis, -2:]

    # Hack so that the input and output slot dimensions are the same.
    to_remove = 0
    if self.return_slot_positions:
      to_remove += 2
    if self.return_slot_scales:
      to_remove += 2
    if to_remove > 0:
      slots = slots[Ellipsis, :-to_remove]

    # Add position encodings to inputs
    n_features = inputs.shape[-1]
    grid_projector = nn.Dense(n_features, name="dense_pe_0")
    inputs = self.grid_encoder()(inputs + grid_projector(grid))

    qkv_size = self.qkv_size or slots.shape[-1]
    head_dim = qkv_size // self.num_heads
    dense = functools.partial(nn.DenseGeneral,
                              axis=-1, features=(self.num_heads, head_dim),
                              use_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = nn.LayerNorm()
    inverted_attention = attention.InvertedDotProductAttention(
        norm_type="mean",
        multi_head=self.num_heads > 1,
        return_attn_weights=True)
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
      updates, attn = inverted_attention(query=q, key=k, value=v, train=train)

      # Recurrent update.
      slots = gru(slots, updates)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None:
        slots = mlp(slots)

    if self.return_slot_positions:
      # Compute the center of mass of each slot attention mask.
      positions = jnp.einsum("...qk,...kd->...qd", attn, grid)
      slots = jnp.concatenate([slots, positions], axis=-1)

    if self.return_slot_scales:
      # Compute slot scales. Take the square root to make the operation
      # analogous to normalizing data drawn from a Gaussian.
      spread = jnp.square(
          jnp.expand_dims(grid, axis=-3) - jnp.expand_dims(positions, axis=-2))
      scales = jnp.sqrt(
          jnp.einsum("...qk,...qkd->...qd", attn + self.epsilon, spread))
      scales = jnp.clip(scales, self.min_scale, self.max_scale)
      slots = jnp.concatenate([slots, scales], axis=-1)

    return slots


class SlotAttentionPosKeysValues(nn.Module):
  """Slot Attention module with positional encodings in keys and values.

  Feature position encodings are added to keys and values instead
  of the inputs.

  Note: This module expects a 2D coordinate grid to be appended
  at the end of inputs.

  Note: This module uses pre-normalization by default.
  """
  grid_encoder: Callable[[], nn.Module]
  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  softmax_temperature: float = 1.0
  gumbel_softmax: bool = False
  gumbel_softmax_straight_through: bool = False
  num_heads: int = 1

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention with explicit slot statistics module forward pass."""
    del padding_mask  # Unused.

    # Separate a concatenated linear coordinate grid from the inputs.
    inputs, grid = inputs[Ellipsis, :-2], inputs[Ellipsis, -2:]

    qkv_size = self.qkv_size or slots.shape[-1]
    head_dim = qkv_size // self.num_heads
    dense = functools.partial(nn.DenseGeneral,
                              axis=-1, features=(self.num_heads, head_dim),
                              use_bias=False)

    # Shared modules.
    dense_q = dense(name="general_dense_q_0")
    layernorm_q = nn.LayerNorm()
    inverted_attention = attention.InvertedDotProductAttention(
        norm_type="mean",
        multi_head=self.num_heads > 1)
    gru = misc.GRU()

    if self.mlp_size is not None:
      mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    # inputs.shape = (..., n_inputs, inputs_size).
    inputs = nn.LayerNorm()(inputs)
    # k.shape = (..., n_inputs, slot_size).
    k = dense(name="general_dense_k_0")(inputs)
    # v.shape = (..., n_inputs, slot_size).
    v = dense(name="general_dense_v_0")(inputs)

    # Add position encodings to keys and values.
    grid_projector = dense(name="general_dense_p_0")
    grid_encoder = self.grid_encoder()
    k = grid_encoder(k + grid_projector(grid))
    v = grid_encoder(v + grid_projector(grid))

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_inputs, slot_size).
      updates = inverted_attention(query=q, key=k, value=v, train=train)

      # Recurrent update.
      slots = gru(slots, updates)

      # Feedforward block with pre-normalization.
      if self.mlp_size is not None:
        slots = mlp(slots)

    return slots


class SlotAttentionTranslEquiv(nn.Module):
  """Slot Attention module with slot positions.

  A position is computed for each slot. Slot positions are used to create
  relative coordinate grids, which are used as position embeddings reapplied
  in each iteration of slot attention. The last two channels in inputs
  must contain the flattened position grid.

  Note: This module uses pre-normalization by default.
  """

  grid_encoder: Callable[[], nn.Module]
  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  softmax_temperature: float = 1.0
  gumbel_softmax: bool = False
  gumbel_softmax_straight_through: bool = False
  num_heads: int = 1
  zero_position_init: bool = True
  ablate_non_equivariant: bool = False
  stop_grad_positions: bool = False
  mix_slots: bool = False
  add_rel_pos_to_values: bool = False
  append_statistics: bool = False

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention translation equiv. module forward pass."""
    del padding_mask  # Unused.

    if self.num_heads > 1:
      raise NotImplementedError("This prototype only uses one attn. head.")

    # Separate a concatenated linear coordinate grid from the inputs.
    inputs, grid = inputs[Ellipsis, :-2], inputs[Ellipsis, -2:]

    # Separate position (x,y) from slot embeddings.
    slots, positions = slots[Ellipsis, :-2], slots[Ellipsis, -2:]
    qkv_size = self.qkv_size or slots.shape[-1]
    num_slots = slots.shape[-2]

    # Prepare initial slot positions.
    if self.zero_position_init:
      # All slots start in the middle of the image.
      positions *= 0.

    # Learnable initial positions might deviate from the allowed range.
    positions = jnp.clip(positions, -1., 1.)

    # Pre-normalization.
    inputs = nn.LayerNorm()(inputs)

    grid_per_slot = jnp.repeat(
        jnp.expand_dims(grid, axis=-3), num_slots, axis=-3)

    # Shared modules.
    dense_q = nn.Dense(qkv_size, use_bias=False, name="general_dense_q_0")
    dense_k = nn.Dense(qkv_size, use_bias=False, name="general_dense_k_0")
    dense_v = nn.Dense(qkv_size, use_bias=False, name="general_dense_v_0")
    grid_proj = nn.Dense(qkv_size, name="dense_gp_0")
    grid_enc = self.grid_encoder()
    layernorm_q = nn.LayerNorm()
    inverted_attention = InvertedDotProductAttentionKeyPerQuery(
        epsilon=self.epsilon,
        renormalize_keys=True,
        softmax_temperature=self.softmax_temperature,
        value_per_query=self.add_rel_pos_to_values
    )
    gru = misc.GRU()

    if self.mlp_size is not None:
      mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    if self.append_statistics:
      embed_statistics = nn.Dense(slots.shape[-1], name="dense_embed_0")

    # k.shape and v.shape = (..., n_inputs, slot_size).
    v = dense_v(inputs)
    k = dense_k(inputs)
    k_expand = jnp.expand_dims(k, axis=-3)
    v_expand = jnp.expand_dims(v, axis=-3)

    # Multiple rounds of attention. Last iteration updates positions only.
    for attn_round in range(self.num_iterations + 1):

      if self.ablate_non_equivariant:
        # Add an encoded coordinate grid with absolute positions.
        grid_emb_per_slot = grid_proj(grid_per_slot)
        k_rel_pos = grid_enc(k_expand + grid_emb_per_slot)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + grid_emb_per_slot)
      else:
        # Relativize positions, encode them and add them to the keys
        # and optionally to values.
        relative_grid = grid_per_slot - jnp.expand_dims(positions, axis=-2)
        grid_emb_per_slot = grid_proj(relative_grid)
        k_rel_pos = grid_enc(k_expand + grid_emb_per_slot)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + grid_emb_per_slot)

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_slots, slot_size).
      updates, attn = inverted_attention(
          query=q,
          key=k_rel_pos,
          value=v_rel_pos if self.add_rel_pos_to_values else v,
          train=train)

      # Compute the center of mass of each slot attention mask.
      # Guaranteed to be in [-1, 1].
      positions = jnp.einsum("...qk,...kd->...qd", attn, grid)

      if self.stop_grad_positions:
        # Do not backprop through positions and scales.
        positions = jax.lax.stop_gradient(positions)

      if attn_round < self.num_iterations:
        if self.append_statistics:
          # Projects and add 2D slot positions into slot latents.
          tmp = jnp.concatenate([slots, positions], axis=-1)
          slots = embed_statistics(tmp)

        # Recurrent update.
        slots = gru(slots, updates)

        # Feedforward block with pre-normalization.
        if self.mlp_size is not None:
          slots = mlp(slots)

    # Concatenate position information to slots.
    output = jnp.concatenate([slots, positions], axis=-1)

    if self.mix_slots:
      output = misc.MLP(hidden_size=128, layernorm="pre")(output)

    return output


class SlotAttentionTranslScaleEquiv(nn.Module):
  """Slot Attention module with slot positions and scales.

  A position and scale is computed for each slot. Slot positions and scales
  are used to create relative coordinate grids, which are used as position
  embeddings reapplied in each iteration of slot attention. The last two
  channels in input must contain the flattened position grid.

  Note: This module uses pre-normalization by default.
  """

  grid_encoder: Callable[[], nn.Module]
  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  softmax_temperature: float = 1.0
  gumbel_softmax: bool = False
  gumbel_softmax_straight_through: bool = False
  num_heads: int = 1
  zero_position_init: bool = True
  # Scale of 0.1 corresponds to fairly small objects.
  init_with_fixed_scale: Optional[float] = 0.1
  ablate_non_equivariant: bool = False
  stop_grad_positions_and_scales: bool = False
  mix_slots: bool = False
  add_rel_pos_to_values: bool = False
  scales_factor: float = 1.
  # Slot scales cannot be negative and should not be too close to zero
  # or too large.
  min_scale: float = 0.001
  max_scale: float = 2.
  append_statistics: bool = False

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention translation and scale equiv. module forward pass."""
    del padding_mask  # Unused.

    if self.num_heads > 1:
      raise NotImplementedError("This prototype only uses one attn. head.")

    # Separate a concatenated linear coordinate grid from the inputs.
    inputs, grid = inputs[Ellipsis, :-2], inputs[Ellipsis, -2:]

    # Separate position (x,y) and scale from slot embeddings.
    slots, positions, scales = (slots[Ellipsis, :-4],
                                slots[Ellipsis, -4: -2],
                                slots[Ellipsis, -2:])
    qkv_size = self.qkv_size or slots.shape[-1]
    num_slots = slots.shape[-2]

    # Prepare initial slot positions.
    if self.zero_position_init:
      # All slots start in the middle of the image.
      positions *= 0.

    if self.init_with_fixed_scale is not None:
      scales = scales * 0. + self.init_with_fixed_scale

    # Learnable initial positions and scales could have arbitrary values.
    positions = jnp.clip(positions, -1., 1.)
    scales = jnp.clip(scales, self.min_scale, self.max_scale)

    # Pre-normalization.
    inputs = nn.LayerNorm()(inputs)

    grid_per_slot = jnp.repeat(
        jnp.expand_dims(grid, axis=-3), num_slots, axis=-3)

    # Shared modules.
    dense_q = nn.Dense(qkv_size, use_bias=False, name="general_dense_q_0")
    dense_k = nn.Dense(qkv_size, use_bias=False, name="general_dense_k_0")
    dense_v = nn.Dense(qkv_size, use_bias=False, name="general_dense_v_0")
    grid_proj = nn.Dense(qkv_size, name="dense_gp_0")
    grid_enc = self.grid_encoder()
    layernorm_q = nn.LayerNorm()
    inverted_attention = InvertedDotProductAttentionKeyPerQuery(
        epsilon=self.epsilon,
        renormalize_keys=True,
        softmax_temperature=self.softmax_temperature,
        value_per_query=self.add_rel_pos_to_values
    )
    gru = misc.GRU()

    if self.mlp_size is not None:
      mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    if self.append_statistics:
      embed_statistics = nn.Dense(slots.shape[-1], name="dense_embed_0")

    # k.shape and v.shape = (..., n_inputs, slot_size).
    v = dense_v(inputs)
    k = dense_k(inputs)
    k_expand = jnp.expand_dims(k, axis=-3)
    v_expand = jnp.expand_dims(v, axis=-3)

    # Multiple rounds of attention.
    # Last iteration updates positions and scales only.
    for attn_round in range(self.num_iterations + 1):

      if self.ablate_non_equivariant:
        # Add an encoded coordinate grid with absolute positions.
        tmp_grid = grid_proj(grid_per_slot)
        k_rel_pos = grid_enc(k_expand + tmp_grid)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + tmp_grid)
      else:
        # Relativize and scale positions, encode them and add them to inputs.
        relative_grid = grid_per_slot - jnp.expand_dims(positions, axis=-2)
        # Scales are usually small so the grid might get too large.
        relative_grid = relative_grid / self.scales_factor
        relative_grid = relative_grid / jnp.expand_dims(scales, axis=-2)
        tmp_grid = grid_proj(relative_grid)
        k_rel_pos = grid_enc(k_expand + tmp_grid)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + tmp_grid)

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_slots, slot_size).
      updates, attn = inverted_attention(
          query=q,
          key=k_rel_pos,
          value=v_rel_pos if self.add_rel_pos_to_values else v,
          train=train)

      # Compute the center of mass of each slot attention mask.
      positions = jnp.einsum("...qk,...kd->...qd", attn, grid)

      # Compute slot scales. Take the square root to make the operation
      # analogous to normalizing data drawn from a Gaussian.
      spread = jnp.square(grid_per_slot - jnp.expand_dims(positions, axis=-2))
      scales = jnp.sqrt(
          jnp.einsum("...qk,...qkd->...qd", attn + self.epsilon, spread))

      # Computed positions are guaranteed to be in [-1, 1].
      # Scales are unbounded.
      scales = jnp.clip(scales, self.min_scale, self.max_scale)

      if self.stop_grad_positions_and_scales:
        # Do not backprop through positions and scales.
        positions = jax.lax.stop_gradient(positions)
        scales = jax.lax.stop_gradient(scales)

      if attn_round < self.num_iterations:
        if self.append_statistics:
          # Project and add 2D slot positions and scales into slot latents.
          tmp = jnp.concatenate([slots, positions, scales], axis=-1)
          slots = embed_statistics(tmp)

        # Recurrent update.
        slots = gru(slots, updates)

        # Feedforward block with pre-normalization.
        if self.mlp_size is not None:
          slots = mlp(slots)

    # Concatenate position and scale information to slots.
    output = jnp.concatenate([slots, positions, scales], axis=-1)

    if self.mix_slots:
      output = misc.MLP(hidden_size=128, layernorm="pre")(output)

    return output


class SlotAttentionTranslRotScaleEquiv(nn.Module):
  """Slot Attention module with slot positions, rotations and scales.

  A position, rotation and scale is computed for each slot.
  Slot positions, rotations and scales are used to create relative
  coordinate grids, which are used as position embeddings reapplied in each
  iteration of slot attention. The last two channels in input must contain
  the flattened position grid.

  Note: This module uses pre-normalization by default.
  """

  grid_encoder: Callable[[], nn.Module]
  num_iterations: int = 1
  qkv_size: Optional[int] = None
  mlp_size: Optional[int] = None
  epsilon: float = 1e-8
  softmax_temperature: float = 1.0
  gumbel_softmax: bool = False
  gumbel_softmax_straight_through: bool = False
  num_heads: int = 1
  zero_position_init: bool = True
  # Scale of 0.1 corresponds to fairly small objects.
  init_with_fixed_scale: Optional[float] = 0.1
  ablate_non_equivariant: bool = False
  stop_grad_positions: bool = False
  stop_grad_scales: bool = False
  stop_grad_rotations: bool = False
  mix_slots: bool = False
  add_rel_pos_to_values: bool = False
  scales_factor: float = 1.
  # Slot scales cannot be negative and should not be too close to zero
  # or too large.
  min_scale: float = 0.001
  max_scale: float = 2.
  limit_rot_to_45_deg: bool = True
  append_statistics: bool = False

  @nn.compact
  def __call__(self, slots, inputs,
               padding_mask = None,
               train = False):
    """Slot Attention translation and scale equiv. module forward pass."""
    del padding_mask  # Unused.

    if self.num_heads > 1:
      raise NotImplementedError("This prototype only uses one attn. head.")

    # Separate a concatenated linear coordinate grid from the inputs.
    inputs, grid = inputs[Ellipsis, :-2], inputs[Ellipsis, -2:]

    # Separate position (x,y) and scale from slot embeddings.
    slots, positions, scales, rotm = (slots[Ellipsis, :-8],
                                      slots[Ellipsis, -8: -6],
                                      slots[Ellipsis, -6: -4],
                                      slots[Ellipsis, -4:])
    rotm = jnp.reshape(rotm, (*rotm.shape[:-1], 2, 2))
    qkv_size = self.qkv_size or slots.shape[-1]
    num_slots = slots.shape[-2]

    # Prepare initial slot positions.
    if self.zero_position_init:
      # All slots start in the middle of the image.
      positions *= 0.

    if self.init_with_fixed_scale is not None:
      scales = scales * 0. + self.init_with_fixed_scale

    # Learnable initial positions and scales could have arbitrary values.
    positions = jnp.clip(positions, -1., 1.)
    scales = jnp.clip(scales, self.min_scale, self.max_scale)

    # Pre-normalization.
    inputs = nn.LayerNorm()(inputs)

    grid_per_slot = jnp.repeat(
        jnp.expand_dims(grid, axis=-3), num_slots, axis=-3)

    # Shared modules.
    dense_q = nn.Dense(qkv_size, use_bias=False, name="general_dense_q_0")
    dense_k = nn.Dense(qkv_size, use_bias=False, name="general_dense_k_0")
    dense_v = nn.Dense(qkv_size, use_bias=False, name="general_dense_v_0")
    grid_proj = nn.Dense(qkv_size, name="dense_gp_0")
    grid_enc = self.grid_encoder()
    layernorm_q = nn.LayerNorm()
    inverted_attention = InvertedDotProductAttentionKeyPerQuery(
        epsilon=self.epsilon,
        renormalize_keys=True,
        softmax_temperature=self.softmax_temperature,
        value_per_query=self.add_rel_pos_to_values
    )
    gru = misc.GRU()

    if self.mlp_size is not None:
      mlp = misc.MLP(hidden_size=self.mlp_size, layernorm="pre", residual=True)  # type: ignore

    if self.append_statistics:
      embed_statistics = nn.Dense(slots.shape[-1], name="dense_embed_0")

    # k.shape and v.shape = (..., n_inputs, slot_size).
    v = dense_v(inputs)
    k = dense_k(inputs)
    k_expand = jnp.expand_dims(k, axis=-3)
    v_expand = jnp.expand_dims(v, axis=-3)

    # Multiple rounds of attention.
    # Last iteration updates positions and scales only.
    for attn_round in range(self.num_iterations + 1):

      if self.ablate_non_equivariant:
        # Add an encoded coordinate grid with absolute positions.
        tmp_grid = grid_proj(grid_per_slot)
        k_rel_pos = grid_enc(k_expand + tmp_grid)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + tmp_grid)
      else:
        # Relativize and scale positions, encode them and add them to inputs.
        relative_grid = grid_per_slot - jnp.expand_dims(positions, axis=-2)

        # Rotation.
        relative_grid = self.transform(rotm, relative_grid)

        # Scales are usually small so the grid might get too large.
        relative_grid = relative_grid / self.scales_factor
        relative_grid = relative_grid / jnp.expand_dims(scales, axis=-2)
        tmp_grid = grid_proj(relative_grid)
        k_rel_pos = grid_enc(k_expand + tmp_grid)
        if self.add_rel_pos_to_values:
          v_rel_pos = grid_enc(v_expand + tmp_grid)

      # Inverted dot-product attention.
      slots_n = layernorm_q(slots)
      q = dense_q(slots_n)  # q.shape = (..., n_slots, slot_size).
      updates, attn = inverted_attention(
          query=q,
          key=k_rel_pos,
          value=v_rel_pos if self.add_rel_pos_to_values else v,
          train=train)

      # Compute the center of mass of each slot attention mask.
      positions = jnp.einsum("...qk,...kd->...qd", attn, grid)

      # Find the axis with the highest spread.
      relp = grid_per_slot - jnp.expand_dims(positions, axis=-2)
      if self.limit_rot_to_45_deg:
        rotm = self.compute_rotation_matrix_45_deg(relp, attn)
      else:
        rotm = self.compute_rotation_matrix_90_deg(relp, attn)

      # Compute slot scales. Take the square root to make the operation
      # analogous to normalizing data drawn from a Gaussian.
      relp = self.transform(rotm, relp)

      spread = jnp.square(relp)
      scales = jnp.sqrt(
          jnp.einsum("...qk,...qkd->...qd", attn + self.epsilon, spread))

      # Computed positions are guaranteed to be in [-1, 1].
      # Scales are unbounded.
      scales = jnp.clip(scales, self.min_scale, self.max_scale)

      if self.stop_grad_positions:
        positions = jax.lax.stop_gradient(positions)
      if self.stop_grad_scales:
        scales = jax.lax.stop_gradient(scales)
      if self.stop_grad_rotations:
        rotm = jax.lax.stop_gradient(rotm)

      if attn_round < self.num_iterations:
        if self.append_statistics:
          # For the slot rotations, we append both the 2D rotation matrix
          # and the angle by which we rotate.
          # We can compute the angle using atan2(R[0, 0], R[1, 0]).
          tmp = jnp.concatenate(
              [slots, positions, scales,
               rotm.reshape(*rotm.shape[:-2], 4),
               jnp.arctan2(rotm[Ellipsis, 0, 0], rotm[Ellipsis, 1, 0])[Ellipsis, None]],
              axis=-1)
          slots = embed_statistics(tmp)

        # Recurrent update.
        slots = gru(slots, updates)

        # Feedforward block with pre-normalization.
        if self.mlp_size is not None:
          slots = mlp(slots)

    # Concatenate position and scale information to slots.
    output = jnp.concatenate(
        [slots, positions, scales, rotm.reshape(*rotm.shape[:-2], 4)], axis=-1)

    if self.mix_slots:
      output = misc.MLP(hidden_size=128, layernorm="pre")(output)

    return output

  @classmethod
  def compute_weighted_covariance(cls, x, w):
    # The coordinate grid is (y, x), we want (x, y).
    x = jnp.stack([x[Ellipsis, 1], x[Ellipsis, 0]], axis=-1)

    # Pixel coordinates weighted by attention mask.
    cov = x * w[Ellipsis, None]
    cov = jnp.einsum(
        "...ji,...jk->...ik", cov, x, precision=jax.lax.Precision.HIGHEST)

    return cov

  @classmethod
  def compute_reference_frame_45_deg(cls, x, w):
    cov = cls.compute_weighted_covariance(x, w)

    # Compute eigenvalues.
    pm = jnp.sqrt(4. * jnp.square(cov[Ellipsis, 0, 1]) +
                  jnp.square(cov[Ellipsis, 0, 0] - cov[Ellipsis, 1, 1]) + 1e-16)

    eig1 = (cov[Ellipsis, 0, 0] + cov[Ellipsis, 1, 1] + pm) / 2.
    eig2 = (cov[Ellipsis, 0, 0] + cov[Ellipsis, 1, 1] - pm) / 2.

    # Compute eigenvectors, note that both have a positive y-axis.
    # This means we have eliminated half of the possible rotations.
    div = cov[Ellipsis, 0, 1] + 1e-16

    v1 = (eig1 - cov[Ellipsis, 1, 1]) / div
    v2 = (eig2 - cov[Ellipsis, 1, 1]) / div

    v1 = jnp.stack([v1, jnp.ones_like(v1)], axis=-1)
    v2 = jnp.stack([v2, jnp.ones_like(v2)], axis=-1)

    # RULE 1:
    # We catch two failure modes here.
    # 1. If all attention weights are zero the covariance is also zero.
    # Then the above computation is meaningless.
    # 2. If the attention pattern is exactly aligned with the axes
    # (e.g. a horizontal/vertical bar), the off-diagonal covariance
    # values are going to be very low. If we use float32, we get
    # basis vectors that are not orthogonal.
    # Solution: use the default reference frame if the off-diagonal
    # covariance value is too low.
    default_1 = jnp.stack([jnp.ones_like(div), jnp.zeros_like(div)], axis=-1)
    default_2 = jnp.stack([jnp.zeros_like(div), jnp.ones_like(div)], axis=-1)

    mask = (jnp.abs(div) < 1e-6).astype(jnp.float32)[Ellipsis, None]
    v1 = (1. - mask) * v1 + mask * default_1
    v2 = (1. - mask) * v2 + mask * default_2

    # Turn eigenvectors into unit vectors, so that we can construct
    # a basis of a new reference frame.
    norm1 = jnp.sqrt(jnp.sum(jnp.square(v1), axis=-1, keepdims=True))
    norm2 = jnp.sqrt(jnp.sum(jnp.square(v2), axis=-1, keepdims=True))

    v1 = v1 / norm1
    v2 = v2 / norm2

    # RULE 2:
    # If the first basis vector is "pointing up" we assume the object
    # is vertical (e.g. we say a door is vertical, whereas a car is horizontal).
    # In the case of vertical objects, we swap the two basis vectors.
    # This limits the possible rotations to +- 45deg instead of +- 90deg.
    # We define "pointing up" as the first coordinate of the first basis vector
    # being between +- sin(pi/4). The second coordinate is always positive.
    mask = (jnp.logical_and(v1[Ellipsis, 0] < 0.707, v1[Ellipsis, 0] > -0.707)
            ).astype(jnp.float32)[Ellipsis, None]
    v1_ = (1. - mask) * v1 + mask * v2
    v2_ = (1. - mask) * v2 + mask * v1
    v1 = v1_
    v2 = v2_

    # RULE 3:
    # Mirror the first basis vector if the first coordinate is negative.
    # Here, we ensure that our coordinate system is always left-handed.
    # Otherwise, we would sometimes unintentionally mirror the grid.
    mask = (v1[Ellipsis, 0] < 0).astype(jnp.float32)[Ellipsis, None]
    v1 = (1. - mask) * v1 - mask * v1

    return v1, v2

  @classmethod
  def compute_reference_frame_90_deg(cls, x, w):
    cov = cls.compute_weighted_covariance(x, w)

    # Compute eigenvalues.
    pm = jnp.sqrt(4. * jnp.square(cov[Ellipsis, 0, 1]) +
                  jnp.square(cov[Ellipsis, 0, 0] - cov[Ellipsis, 1, 1]) + 1e-16)

    eig1 = (cov[Ellipsis, 0, 0] + cov[Ellipsis, 1, 1] + pm) / 2.
    eig2 = (cov[Ellipsis, 0, 0] + cov[Ellipsis, 1, 1] - pm) / 2.

    # Compute eigenvectors, note that both have a positive y-axis.
    # This means we have eliminated half of the possible rotations.
    div = cov[Ellipsis, 0, 1] + 1e-16

    v1 = (eig1 - cov[Ellipsis, 1, 1]) / div
    v2 = (eig2 - cov[Ellipsis, 1, 1]) / div

    v1 = jnp.stack([v1, jnp.ones_like(v1)], axis=-1)
    v2 = jnp.stack([v2, jnp.ones_like(v2)], axis=-1)

    # RULE 1:
    # We catch two failure modes here.
    # 1. If all attention weights are zero the covariance is also zero.
    # Then the above computation is meaningless.
    # 2. If the attention pattern is exactly aligned with the axes
    # (e.g. a horizontal/vertical bar), the off-diagonal covariance
    # values are going to be very low. If we use float32, we get
    # basis vectors that are not orthogonal.
    # Solution: use the default reference frame if the off-diagonal
    # covariance value is too low.
    default_1 = jnp.stack([jnp.ones_like(div), jnp.zeros_like(div)], axis=-1)
    default_2 = jnp.stack([jnp.zeros_like(div), jnp.ones_like(div)], axis=-1)

    # RULE 1.5:
    # RULE 1 is activated if we see a vertical or a horizontal bar.
    # We make sure that the coordinate grid for a horizontal bar is not rotated,
    # whereas the coordinate grid for a vertical bar is rotated by 90deg.
    # If cov[0, 0] > cov[1, 1], the bar is vertical.
    mask = (cov[Ellipsis, 0, 0] <= cov[Ellipsis, 1, 1]).astype(jnp.float32)[Ellipsis, None]
    # Furthermore, we have to mirror one of the basis vectors (if mask==1)
    # so that we always have a left-handed coordinate grid.
    default_v1 = (1. - mask) * default_1 - mask * default_2
    default_v2 = (1. - mask) * default_2 + mask * default_1

    # Continuation of RULE 1.
    mask = (jnp.abs(div) < 1e-6).astype(jnp.float32)[Ellipsis, None]
    v1 = mask * default_v1 + (1. - mask) * v1
    v2 = mask * default_v2 + (1. - mask) * v2

    # Turn eigenvectors into unit vectors, so that we can construct
    # a basis of a new reference frame.
    norm1 = jnp.sqrt(jnp.sum(jnp.square(v1), axis=-1, keepdims=True))
    norm2 = jnp.sqrt(jnp.sum(jnp.square(v2), axis=-1, keepdims=True))

    v1 = v1 / norm1
    v2 = v2 / norm2

    # RULE 2:
    # Mirror the first basis vector if the first coordinate is negative.
    # Here, we ensure that the our coordinate system is always left-handed.
    # Otherwise, we would sometimes unintentionally mirror the grid.
    mask = (v1[Ellipsis, 0] < 0).astype(jnp.float32)[Ellipsis, None]
    v1 = (1. - mask) * v1 - mask * v1

    return v1, v2

  @classmethod
  def compute_rotation_matrix_45_deg(cls, x, w):
    v1, v2 = cls.compute_reference_frame_45_deg(x, w)
    return jnp.stack([v1, v2], axis=-1)

  @classmethod
  def compute_rotation_matrix_90_deg(cls, x, w):
    v1, v2 = cls.compute_reference_frame_90_deg(x, w)
    return jnp.stack([v1, v2], axis=-1)

  @classmethod
  def transform(cls, rotm, x):
    # The coordinate grid x is in the (y, x) format, so we need to swap
    # the coordinates on the input and output.
    x = jnp.stack([x[Ellipsis, 1], x[Ellipsis, 0]], axis=-1)
    # Equivalent to inv(R) * x^T = R^T * x^T = (x * R)^T.
    # We are multiplying by the inverse of the rotation matrix because
    # we are rotating the coordinate grid *against* the rotation of the object.
    # y = jnp.matmul(x, R)
    y = jnp.einsum("...ij,...jk->...ik", x, rotm)
    # Swap coordinates again.
    y = jnp.stack([y[Ellipsis, 1], y[Ellipsis, 0]], axis=-1)
    return y
