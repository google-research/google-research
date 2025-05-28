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

"""One ring encoder + decoder for latent tangent features."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from mesh_diffusion.latents.complex_attn import ComplexMultiHeadDotProductAttention
from mesh_diffusion.latents.layers import dense_neuron
from mesh_diffusion.latents.layers import mlp5


EPS = 1.0e-6

################################################################################
# One-ring encoder.
################################################################################


class square_mag_norm_encoder(nn.module.Module):  # pylint: disable=invalid-name
  """Square magnitude normalization for the one-ring encoder."""

  @nn.module.compact
  def __call__(self, x):
    features = x.shape[2]
    eps = self.param(
        'eps', nn.initializers.constant(1.0e-3), (features,), jnp.float32
    )
    scale = self.param('scale', nn.initializers.ones, (features,), jnp.float32)
    square_mean = jnp.mean(
        x.imag * x.imag + x.real * x.real, axis=(0, 1), keepdims=True
    )
    factor = scale[None, None, :] * jax.lax.rsqrt(
        square_mean + EPS + jnp.abs(eps[None, None, :])
    )
    return x * factor


class init_conv(nn.module.Module):  # pylint: disable=invalid-name
  """Gathering operation that initalizes token feature for one-ring encoder."""

  features: int

  @nn.module.compact
  def __call__(self, ring_logs, ring_vals):
    ring_logs = jax.lax.complex(ring_logs[Ellipsis, 0], ring_logs[Ellipsis, 1])
    mags = jnp.abs(ring_logs)
    feat = jnp.concatenate(
        (
            jnp.tile(ring_vals[:, 0, None, :], (1, ring_vals.shape[1] - 1, 1)),
            ring_vals[:, 1:, :],
            mags[:, 1:, None],
        ),
        axis=-1,
    )
    feat = jax.nn.silu(nn.Dense(features=self.features)(feat))
    feat = jax.nn.silu(nn.Dense(features=self.features)(feat))
    return jnp.mean(feat * ring_logs[:, 1:, None], axis=1)


class complex_mhattn(nn.module.Module):  # pylint: disable=invalid-name
  """Multi-headed complex attention block (equivariant) w/ res. conn."""

  features: int
  num_heads: int = 8

  @nn.module.compact
  def __call__(self, x):
    # head_feat = self.features // self.num_heads
    x = ComplexMultiHeadDotProductAttention(
        num_heads=self.num_heads, out_features=self.features, use_bias=False
    )(x, x) + nn.Dense(features=self.features, use_bias=False)(x)
    return x


class quad_mhattn(nn.module.Module):  # pylint: disable=invalid-name
  """Four multi-headed attention blocks."""

  features: int
  num_heads: int = 8

  @nn.module.compact
  def __call__(self, x):
    x0 = square_mag_norm_encoder()(x)
    x0 = complex_mhattn(features=self.features, num_heads=self.num_heads)(x0)
    x = x + x0
    x0 = square_mag_norm_encoder()(x)
    x0 = dense_neuron(features=self.features)(x0)
    x0 = dense_neuron(features=self.features)(x0)
    x = x + x0

    x0 = square_mag_norm_encoder()(x)
    x0 = complex_mhattn(features=self.features, num_heads=self.num_heads)(x0)
    x = x + x0
    x0 = square_mag_norm_encoder()(x)
    x0 = dense_neuron(features=self.features)(x0)
    x0 = dense_neuron(features=self.features)(x0)
    x = x + x0

    x0 = square_mag_norm_encoder()(x)
    x0 = complex_mhattn(features=self.features, num_heads=self.num_heads)(x0)
    x = x + x0
    x0 = square_mag_norm_encoder()(x)
    x0 = dense_neuron(features=self.features)(x0)
    x0 = dense_neuron(features=self.features)(x0)
    x = x + x0

    x0 = square_mag_norm_encoder()(x)
    x0 = complex_mhattn(features=self.features, num_heads=self.num_heads)(x0)
    x = x + x0
    x0 = square_mag_norm_encoder()(x)
    x0 = dense_neuron(features=self.features)(x0)
    x0 = dense_neuron(features=self.features)(x0)
    x = x + x0

    return x


class directional_features(nn.module.Module):  # pylint: disable=invalid-name
  """Scalar directonal features from tangent vector features."""

  @nn.module.compact
  def __call__(self, x):
    q = nn.Dense(features=x.shape[-1], use_bias=False)(x)
    return x.real * q.real + x.imag * q.imag


class ring_encoder(nn.module.Module):  # pylint: disable=invalid-name
  """One-ring encoder for tangent vector latents."""

  features: int
  latent_dim: int
  num_heads: int = 8

  @nn.module.compact
  def __call__(self, ring_logs, ring_pix):
    x0 = init_conv(features=self.features)(ring_logs, ring_pix)
    pix_emb = jax.nn.silu(nn.Dense(features=self.features)(ring_pix))
    pix_emb = jax.nn.silu(nn.Dense(features=self.features)(pix_emb))
    pix_emb = nn.Dense(features=self.features)(pix_emb)
    x = (
        pix_emb
        * jax.lax.complex(ring_logs[Ellipsis, 0], ring_logs[Ellipsis, 1])[Ellipsis, None]
    )
    # x0 are token features
    x = jnp.concatenate((x[:, 1:, :], x0[:, None, :]), axis=1)

    x = quad_mhattn(features=self.features, num_heads=self.num_heads)(x)
    x = quad_mhattn(features=self.features, num_heads=self.num_heads)(x)

    # Extract token
    x = x[:, -1, Ellipsis]

    # VAE Mean
    mean = nn.Dense(
        features=self.latent_dim, use_bias=False, param_dtype=jnp.complex64
    )(x)
    # VAE log variance
    ln_var = jnp.tanh(directional_features()(x))
    ln_var = nn.Dense(features=2 * self.latent_dim)(ln_var)
    ln_var = jnp.clip(jnp.reshape(ln_var, (ln_var.shape[0], -1, 2)), max=3.0)
    return mean, jax.lax.complex(ln_var[Ellipsis, 0], ln_var[Ellipsis, 1])


################################################################################
# Sampler.
################################################################################


class Sampling(nn.module.Module):
  """VAE sampling bottleneck."""

  @nn.module.compact
  def __call__(self, z_mean, z_log_var, key):
    z_mean = jnp.concatenate(
        (jnp.real(z_mean)[Ellipsis, None], jnp.imag(z_mean)[Ellipsis, None]), axis=-1
    )
    z_log_var = jnp.concatenate(
        (jnp.real(z_log_var[Ellipsis, None]), jnp.imag(z_log_var)[Ellipsis, None]),
        axis=-1,
    )

    key, sample_key = jax.random.split(key)
    eps = jax.random.normal(sample_key, shape=z_mean.shape, dtype=z_mean.dtype)

    sample = z_mean + jnp.exp(0.5 * z_log_var) * eps

    return jax.lax.complex(sample[Ellipsis, 0], sample[Ellipsis, 1]), key


################################################################################
# One-ring decoder.
################################################################################


class ring_decoder(nn.module.Module):  # pylint: disable=invalid-name
  """One-ring decoder."""

  features: int
  out_features: int = 3

  @nn.module.compact
  def __call__(self, vlatents, pix_tri, pix_logs):
    def _invar_outer(x, rows, cols):
      outer = (x[Ellipsis, None] * jnp.conjugate(x)[:, None, :])[:, rows, cols]
      return jnp.concatenate(
          (jnp.real(x * jnp.conjugate(x)), jnp.real(outer), jnp.imag(outer)),
          axis=-1,
      )

    latent_dim = vlatents.shape[1]
    # num_tris = pix_tri.shape[0]
    (irows, icols) = jnp.triu_indices(latent_dim, k=1)

    # Compute invariant features per-vertex
    latents = _invar_outer(vlatents[pix_tri, Ellipsis], irows, icols)

    # Directional features
    pix_logs = jax.lax.complex(pix_logs[Ellipsis, 0], pix_logs[Ellipsis, 1])
    dir_latents = pix_logs[Ellipsis, None] * jnp.conjugate(vlatents)[pix_tri, Ellipsis]

    latents = jnp.concatenate(
        (latents, jnp.real(dir_latents), jnp.imag(dir_latents)), axis=-1
    )
    pred = mlp5(features=self.features, out_features=self.out_features)(latents)

    return pred


class scalar_decoder(nn.module.Module):  # pylint: disable=invalid-name
  """One-ring decoder."""

  features: int
  out_features: int = 3

  @nn.module.compact
  def __call__(self, vlatents, pix_bary):
    def _invar_outer(x, rows, cols):
      outer = (x[Ellipsis, None] * jnp.conjugate(x)[Ellipsis, None, :])[Ellipsis, rows, cols]
      return jnp.concatenate(
          (jnp.real(x * jnp.conjugate(x)), jnp.real(outer), jnp.imag(outer)),
          axis=-1,
      )

    latent_dim = vlatents.shape[-1]
    (irows, icols) = jnp.triu_indices(latent_dim, k=1)

    # Compute invariant features per-vertex
    latents = _invar_outer(vlatents, irows, icols)
    latents = jnp.sum(latents * pix_bary[Ellipsis, None], axis=1)
    latents = jnp.concatenate((jnp.real(latents), jnp.imag(latents)), axis=-1)
    pred = mlp5(features=self.features, out_features=self.out_features)(latents)

    return pred
