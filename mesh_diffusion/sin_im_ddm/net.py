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

"""FC based architectures."""
from typing import Tuple

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp

from mesh_diffusion.latents.layers import downsample_tangent
from mesh_diffusion.latents.layers import FCENextBlock
from mesh_diffusion.latents.layers import FCEResBlock
from mesh_diffusion.latents.layers import FCEResBlock2
from mesh_diffusion.latents.layers import quantized_embedding
from mesh_diffusion.latents.layers import upsample_tangent
from mesh_diffusion.sin_im_ddm.diffusion import batched_timestep_embedding
from mesh_diffusion.sin_im_ddm.diffusion import timestep_embedding


class fc_unet_3(nn.module.Module):  # pylint: disable=invalid-name
  """Three layer FC UNet w/timestep embeddings. Input/output dims are equal."""

  features: int = 64  # Number of features at finest resolution.

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
  ):
    # Assummes x is complex (num_verts x latent_dim) tensor.
    in_features = x.shape[1]
    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )
    geom_2 = (
        mass[2],
        neigh[2],
        logs[2],
        mass[2][neigh[2]]
        * jax.lax.complex(jnp.cos(xport[2]), jnp.sin(xport[2])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])
    down_2_1 = (sample_neigh[1], sample_vals[1], mass[1], mass[2].shape[0])

    # Upsampling data
    up_1_2 = (sample_neigh[1], sample_vals[1])
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 4 * self.features
    emb = timestep_embedding(t, emb_features)
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0])(x, emb, *geom_0)
    x0 = FCEResBlock(features=f[0])(x, emb, *geom_0)

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1])(x, emb, *geom_1)
    x1 = FCEResBlock(features=f[1])(x, emb, *geom_1)

    x = downsample_tangent(x1, *down_2_1)

    # Layer 2 (Bottom)
    x = FCEResBlock(features=f[2])(x, emb, *geom_2)
    x = FCEResBlock(features=f[2])(x, emb, *geom_2)

    # Layer 1 (Up)
    x = upsample_tangent(x, *up_1_2)
    x = nn.Dense(features=f[1], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x1), axis=-1)
    )

    x = FCEResBlock(features=f[1])(x, emb, *geom_1)
    x = FCEResBlock(features=f[1])(x, emb, *geom_1)

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0])(x, emb, *geom_0)
    x = FCEResBlock(features=f[0], last=True)(x, emb, *geom_0)

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_unet_2(nn.module.Module):  # pylint: disable=invalid-name
  """Same as fc_unet_3 but two layers instead of three."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1
  hdim: int = 8

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 4 * self.features
    emb = timestep_embedding(t, emb_features)
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )
    x0 = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_1
    )

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_1
    )

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb, *geom_0
    )

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_unet_2_ext(nn.module.Module):  # pylint: disable=invalid-name
  """fc_unet_2_ext module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 4 * self.features
    emb = timestep_embedding(t, emb_features)
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x0 = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_unet_2_lab(nn.module.Module):  # pylint: disable=invalid-name
  """fc_unet_2_lab module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1
  hdim: int = 8

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
      labels,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 2 * self.features
    emb = timestep_embedding(t, emb_features)

    lab_0 = batched_timestep_embedding(labels[0] + 1, emb_features)
    lab_1 = batched_timestep_embedding(labels[1] + 1, emb_features)

    emb_0 = jnp.concatenate(
        (jnp.tile(emb[None, :], (labels[0].shape[0], 1)), lab_0), axis=-1
    )
    emb_1 = jnp.concatenate(
        (jnp.tile(emb[None, :], (labels[1].shape[0], 1)), lab_1), axis=-1
    )

    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))
    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))

    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))
    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )
    x0 = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_1, *geom_1
    )

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_1, *geom_1
    )

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers, hdim=self.hdim)(
        x, emb_0, *geom_0
    )

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_unet_2_signal(nn.module.Module):  # pylint: disable=invalid-name
  """fc_unet_2_signal module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
      signal,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 2 * self.features
    emb = timestep_embedding(t, emb_features)

    signal_0 = jax.nn.silu(nn.Dense(features=emb_features)(signal[0]))
    signal_1 = jax.nn.silu(nn.Dense(features=emb_features)(signal[1]))

    emb_0 = jnp.concatenate(
        (jnp.tile(emb[None, :], (signal[0].shape[0], 1)), signal_0), axis=-1
    )
    emb_1 = jnp.concatenate(
        (jnp.tile(emb[None, :], (signal[1].shape[0], 1)), signal_1), axis=-1
    )

    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))
    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))

    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))
    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x0 = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_unet_2_quantize(nn.module.Module):  # pylint: disable=invalid-name
  """fc_unet_2_quantize module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
      signal,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 2 * self.features
    emb = timestep_embedding(t, emb_features)

    signal_stack = jnp.concatenate((signal[0], signal[1]), axis=0)
    quant_stack, loss = quantized_embedding(features=emb_features)(signal_stack)

    quant_0 = quant_stack[: signal[0].shape[0], Ellipsis]
    quant_1 = quant_stack[signal[0].shape[0] :, Ellipsis]

    emb_0 = jnp.concatenate(
        (jnp.tile(emb[None, :], (signal[0].shape[0], 1)), quant_0), axis=-1
    )
    emb_1 = jnp.concatenate(
        (jnp.tile(emb[None, :], (signal[1].shape[0], 1)), quant_1), axis=-1
    )

    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))
    emb_0 = jax.nn.silu(nn.Dense(features=emb_features)(emb_0))

    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))
    emb_1 = jax.nn.silu(nn.Dense(features=emb_features)(emb_1))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x0 = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )

    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )
    x = FCEResBlock(features=f[1], num_layers=self.mlp_layers)(
        x, emb_1, *geom_1
    )

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )
    x = FCEResBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb_0, *geom_0
    )

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x, loss


class fc_unet_2_proj(nn.module.Module):  # pylint: disable=invalid-name
  """fc_unet_2_proj module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      bases,
      sample_neigh,
      sample_vals,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
        bases[0],
    )
    geom_1 = (
        mass[1],
        neigh[1],
        logs[1],
        mass[1][neigh[1]]
        * jax.lax.complex(jnp.cos(xport[1]), jnp.sin(xport[1])),
        bases[1],
    )

    # Downsampling data
    down_1_0 = (sample_neigh[0], sample_vals[0], mass[0], mass[1].shape[0])

    # Upsampling data
    up_0_1 = (sample_neigh[0], sample_vals[0])

    # Initialize embedding
    emb_features = 4 * self.features
    emb = timestep_embedding(t, emb_features)
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x0 = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(
        x, emb, *geom_0
    )

    x = downsample_tangent(x0, *down_1_0)

    # Layer 1 (Down)

    x = FCEResBlock2(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)
    x = FCEResBlock2(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)

    x = FCEResBlock2(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)
    x = FCEResBlock2(features=f[1], num_layers=self.mlp_layers)(x, emb, *geom_1)

    # Layer 0 (Up)
    x = upsample_tangent(x, *up_0_1)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(
        jnp.concatenate((x, x0), axis=-1)
    )

    x = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)
    x = FCEResBlock2(features=f[0], num_layers=self.mlp_layers)(x, emb, *geom_0)

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x


class fc_next(nn.module.Module):  # pylint: disable=invalid-name
  """fc_next module."""

  features: int = 64  # Number of features at finest resolution
  mlp_layers: int = 1

  @nn.module.compact
  def __call__(
      self,
      x,
      t,
      mass,
      neigh,
      logs,
      xport,
      sample_neigh,
      sample_vals,
  ):

    # Assummes x is complex (num_verts x latent_dim) tensor.

    in_features = x.shape[1]

    f = [self.features, 2 * self.features, 4 * self.features]

    # Organize geometry data for each layer
    geom_0 = (
        mass[0],
        neigh[0],
        logs[0],
        mass[0][neigh[0]]
        * jax.lax.complex(jnp.cos(xport[0]), jnp.sin(xport[0])),
    )

    # Initialize embedding
    emb_features = 4 * self.features
    emb = timestep_embedding(t, emb_features)
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))
    emb = jax.nn.silu(nn.Dense(features=emb_features)(emb))

    # Going down

    # Layer 0 (Down)
    x = nn.Dense(features=f[0], use_bias=False, param_dtype=jnp.complex64)(x)

    x1 = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        x, emb, *geom_0
    )
    x2 = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        x1, emb, *geom_0
    )
    x3 = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        x2, emb, *geom_0
    )
    x4 = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        x3, emb, *geom_0
    )
    x = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        x4, emb, *geom_0
    )
    x = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        jnp.concatenate((x, x3), axis=-1), emb, *geom_0
    )
    x = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        jnp.concatenate((x, x2), axis=-1), emb, *geom_0
    )
    x = FCENextBlock(features=f[0], num_layers=self.mlp_layers)(
        jnp.concatenate((x, x1), axis=-1), emb, *geom_0
    )

    x = nn.Dense(
        features=in_features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    return x
