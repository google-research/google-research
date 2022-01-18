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

"""Utility functions for Multiscale Image Quality Transformer."""
from flax import nn
import jax
import jax.numpy as jnp
import numpy as np

# Maximum frequency-scale in sine grating.
SINE_MAX_SCALE = 10000


def get_sinusoid_encoding(n_position, hidden_size):
  """Sinusoid position encoding table.

  Args:
    n_position: the number of total positions.
    hidden_size: the hidden dimension for the encoding table.

  Returns:
    The sinusoid_table
  """

  def get_position_angle_vec(position):
    return [
        position / np.power(SINE_MAX_SCALE, 2 * (hid_j // 2) / hidden_size)
        for hid_j in range(hidden_size)
    ]

  sinusoid_table = np.array(
      [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

  return sinusoid_table


class AddHashSpatialPositionEmbs(nn.Module):
  """Adds learnable hash-based spatial embeddings to the inputs."""

  def apply(self,
            inputs,
            spatial_pos_grid_size,
            inputs_positions,
            posemb_init=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: input data.
      spatial_pos_grid_size: spatial positional encoding hash grid size.
      inputs_positions: input position indices for packed sequences.
      posemb_init: positional embedding initializer.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    pos_emb_shape = (1, spatial_pos_grid_size * spatial_pos_grid_size,
                     inputs.shape[2])
    pe = self.param("pos_embedding", pos_emb_shape, posemb_init)

    # Fetches embedding according to hashed position indices:
    return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class AddScaleEmbs(nn.Module):
  """Adds learnable scale embeddings to the inputs."""

  def apply(self,
            inputs,
            num_scales,
            inputs_positions=None,
            scale_emb_init=None):
    """Applies AddScaleEmbs module.

    Args:
      inputs: input data.
      num_scales: number of scales input.
      inputs_positions: input position indices for packed sequences.
      scale_emb_init: scale embedding initializer.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    scale_emb_shape = (1, num_scales, inputs.shape[2])
    scale_emb = self.param("scale_embedding", scale_emb_shape, scale_emb_init)

    # Fetches embedding according to hashed position indices:
    return inputs + jnp.take(scale_emb[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=True,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(
        inputs,
        mlp_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x,
        actual_out_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer."""

  def get_drop_pattern(self, x, layer_drop_p):
    if nn.is_stochastic() and layer_drop_p:
      rng = nn.make_rng()
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(rng, layer_drop_p, shape).astype("float32")
    else:
      return 0.0

  def apply(self,
            inputs,
            mlp_dim,
            inputs_masks=None,
            dtype=jnp.float32,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=True,
            layer_drop_p=None,
            **attention_kwargs):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      inputs_masks: bool, input mask.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      layer_drop_p: probability of dropping a layer.
      **attention_kwargs: kwargs passed to nn.SelfAttention

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(inputs, dtype=dtype)
    x = nn.SelfAttention(
        x,
        dtype=dtype,
        inputs_kv=x,
        attention_axis=(1,),
        causal_mask=False,
        padding_mask=inputs_masks,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=attention_dropout_rate,
        **attention_kwargs)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, layer_drop_p)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(x, dtype=dtype)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, layer_drop_p)
    return y * (1.0 - drop_pattern) + x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def apply(self,
            inputs,
            inputs_spatial_positions,
            inputs_scale_positions,
            inputs_masks,
            spatial_pos_grid_size,
            num_scales,
            num_layers,
            mlp_dim,
            use_sinusoid_pos_emb=False,
            use_scale_emb=True,
            dropout_rate=0.1,
            train=False,
            dtype=jnp.float32,
            stochastic_layer_drop_rate=0.0,
            **attention_kwargs):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_spatial_positions: input spatial positions for each embedding.
      inputs_scale_positions: input scale positions for each embedding.
      inputs_masks: bool, input mask.
      spatial_pos_grid_size: spatial positional encoding hash grid size.
      num_scales: number of scales input.
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block.
      use_sinusoid_pos_emb: whether to use Sinusoidal Positional Embedding.
      use_scale_emb: use scale embedding.
      dropout_rate: dropout rate
      train: if it is training,
      dtype: dtype of activations.
      stochastic_layer_drop_rate: probability of dropping a layer linearly grows
        from 0 to the provided value. Our implementation of stochastic depth
        follows timm library, which does per-example layer dropping and uses
        independent dropping patterns for each skip-connection.
      **attention_kwargs: kwargs passed to nn.SelfAttention

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)
    dtype = jax.dtypes.canonicalize_dtype(dtype)

    if not use_sinusoid_pos_emb:
      x = AddHashSpatialPositionEmbs(
          inputs,
          spatial_pos_grid_size,
          inputs_positions=inputs_spatial_positions,
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name="posembed_input")
    else:
      pos_emb_shape = (1, spatial_pos_grid_size * spatial_pos_grid_size,
                       inputs.shape[2])
      pe = get_sinusoid_encoding(pos_emb_shape[1], pos_emb_shape[2])
      pe = jnp.expand_dims(pe, axis=0)
      x = inputs + jnp.take(pe[0], inputs_spatial_positions, axis=0)

    if use_scale_emb:
      x = AddScaleEmbs(
          x,
          num_scales=num_scales,
          inputs_positions=inputs_scale_positions,
          scale_emb_init=nn.initializers.normal(stddev=0.02),
          name="scaleembed_input")

    n, _, c = x.shape
    cls = self.param("cls", (1, 1, c), nn.initializers.zeros)
    cls = jnp.tile(cls, [n, 1, 1])
    x = jnp.concatenate([cls, x], axis=1)

    cls_mask = jnp.ones((n, 1), dtype=inputs_masks.dtype)
    inputs_masks = jnp.concatenate([cls_mask, inputs_masks], axis=1)

    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    # Input Encoder
    for lyr in range(num_layers):
      layer_drop_p = (lyr / max(num_layers - 1, 1)) * stochastic_layer_drop_rate
      x = Encoder1DBlock(
          x,
          mlp_dim=mlp_dim,
          inputs_masks=inputs_masks,
          dropout_rate=dropout_rate,
          deterministic=not train,
          name=f"encoderblock_{lyr}",
          dtype=dtype,
          layer_drop_p=layer_drop_p,
          **attention_kwargs)
    encoded = nn.LayerNorm(x, name="encoder_norm")

    return encoded
