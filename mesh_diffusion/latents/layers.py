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

"""Layers for processing tangent vector features on meshes."""

import functools
import math
from typing import Any
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp

EPS = 1.0e-6


class square_mag_norm(nn.module.Module):  # pylint: disable=invalid-name
  """Square magnitude normalization."""

  @nn.module.compact
  def __call__(self, x, mass):

    features = x.shape[1]

    eps = self.param(
        'eps', nn.initializers.constant(1.0e-3), (features,), jnp.float32
    )

    scale = self.param(
        'scale', nn.initializers.ones, (features,), jnp.complex64
    )

    mass = mass / jnp.sum(mass)

    square_mean = jnp.sum(
        (x.imag * x.imag + x.real * x.real) * mass[Ellipsis, None],
        axis=0,
        keepdims=True,
    )

    factor = scale[None, :] * jax.lax.rsqrt(
        square_mean + EPS + jnp.abs(eps[None, :])
    )

    return x * factor


class dense_neuron(nn.module.Module):  # pylint: disable=invalid-name
  """Linear layer + SO(3) neuron modified for tangent vector features + silu."""

  features: int
  zero: bool = False

  @nn.module.compact
  def __call__(self, x):

    q = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)
    k = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    dot = q.real * k.real + q.imag * k.imag

    k2 = k.real * k.real + k.imag * k.imag

    k = (jax.nn.silu(-1.0 * dot) / (k2 + EPS)) * k

    return q + k


class act_neuron(nn.module.Module):  # pylint: disable=invalid-name
  """SO(3) neuron modified for tangent vector features + silu."""

  @nn.module.compact
  def __call__(self, x):

    k = nn.Dense(
        features=x.shape[-1], use_bias=False, param_dtype=jnp.complex64
    )(x)

    dot = x.real * k.real + x.imag * k.imag

    k2 = k.real * k.real + k.imag * k.imag

    k = (jax.nn.silu(-1.0 * dot) / (k2 + EPS)) * k

    return x + k


class neuron_mlp(nn.module.Module):  # pylint: disable=invalid-name
  """Vector (Tangent) neuron MLP with normalization."""

  features: int
  num_layers: int = 1
  factor: int = 1

  def setup(self):

    dense = []
    norm = []

    for l in range(self.num_layers):

      if l == self.num_layers - 1:
        f = self.features
      else:
        f = self.factor * self.features

      norm.append(square_mag_norm())
      dense.append(dense_neuron(features=f))

    self.dense = dense
    self.norm = norm

  def __call__(self, x, mass):

    for l in range(self.num_layers):
      x = self.norm[l](x, mass)
      x = self.dense[l](x)

    return x


class mlp5(nn.module.Module):  # pylint: disable=invalid-name
  """5-layer MLP."""

  features: int
  out_features: int = 3

  @nn.module.compact
  def __call__(self, x):

    embed = jax.nn.silu(nn.Dense(features=self.features)(x))
    embed = jax.nn.silu(nn.Dense(features=self.features)(embed))
    embed = jax.nn.silu(nn.Dense(features=self.features)(embed))
    embed = jax.nn.silu(nn.Dense(features=self.features)(embed))
    embed = nn.Dense(features=self.out_features)(embed)

    return embed


class INF(nn.module.Module):  # pylint: disable=invalid-name
  """Inf."""
  features: int
  out_features: int = 3

  @nn.module.compact
  def __call__(self, x):

    embed = jax.nn.relu(nn.Dense(features=self.features)(x))
    embed = jax.nn.relu(nn.Dense(features=self.features)(embed))
    embed = jax.nn.relu(nn.Dense(features=self.features)(embed))
    embed = jnp.concatenate((embed, x), axis=-1)
    embed = jax.nn.relu(nn.Dense(features=self.features)(embed))
    embed = jax.nn.relu(nn.Dense(features=self.features)(embed))
    embed = jax.nn.sigmoid(nn.Dense(features=self.out_features)(embed))

    return embed


################################################################################
# Field Convolutions.
################################################################################


def get_rel_pos(x, neigh, logs):
  """Get relative position of logs in frame of features."""

  mask = 1.0 * jnp.greater(jnp.abs(x), 1.0e-4)

  angle = jnp.angle(mask * x + (1.0 - mask))

  frame = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))

  return (
      jax.lax.complex(logs[Ellipsis, 0], logs[Ellipsis, 1])[Ellipsis, None]
      * jnp.conjugate(frame)[neigh, :]
  )


class FieldConv(nn.module.Module):
  """Field convolution."""

  features: int  # Number of output channels
  hdim: int = 8  # Number of intermediate channels for MLP-parameterized kernel
  grouped: bool = False

  @nn.module.compact
  def __call__(self, x, neigh, rel, weights):
    """Field convolution."""

    # Normalize weights
    weights = weights / jnp.sum(jnp.abs(weights), axis=1, keepdims=True)

    # Evaluate kernels at relative positions
    rel = nn.Dense(features=self.hdim)(
        jnp.concatenate(
            (jnp.real(rel)[Ellipsis, None], jnp.imag(rel)[Ellipsis, None]), axis=-1
        )
    )
    rel = jax.nn.silu(rel)
    rel = nn.Dense(features=self.hdim)(rel)
    rel = jax.nn.silu(rel)

    # Aggregate
    feat = jnp.sum(x[neigh, :, None] * weights[Ellipsis, None, None] * rel, axis=1)

    # Final linear layer without bias to remain equivariant
    if not self.grouped:
      feat = nn.Dense(
          features=self.features, use_bias=False, param_dtype=jnp.complex64
      )(jnp.reshape(feat, (feat.shape[0], -1)))
    else:
      group = self.param(
          'group',
          nn.initializers.lecun_normal(),
          (feat.shape[-2], feat.shape[-1], 2),
          jnp.float32,
      )
      group = jax.lax.complex(group[Ellipsis, 0], group[Ellipsis, 1])
      feat = jnp.sum(feat * group[None, Ellipsis], axis=-1)

    return feat


class FieldConvEmb(nn.module.Module):
  """Field convolution with timestep embedding."""

  features: int  # Number of output channels
  hdim: int = 8  # Number of intermediate channels for MLP-parameterized kernel
  grouped: bool = False

  @nn.module.compact
  def __call__(
      self, x, neigh, rel, weights, emb
  ):
    """Field convolution."""
    weights = weights / jnp.sum(jnp.abs(weights), axis=1, keepdims=True)

    # Evaluate kernels at relative positions concatenated with timestep
    # embedding.
    if jnp.ndim(emb) == 1:
      rel = nn.Dense(features=self.hdim)(
          jnp.concatenate(
              (
                  jnp.real(rel)[Ellipsis, None],
                  jnp.imag(rel)[Ellipsis, None],
                  jnp.tile(
                      emb[None, None, :, None],
                      (rel.shape[0], rel.shape[1], 1, 1),
                  ),
              ),
              axis=-1,
          )
      )
    else:
      rel = nn.Dense(features=self.hdim)(
          jnp.concatenate(
              (
                  jnp.real(rel)[Ellipsis, None],
                  jnp.imag(rel)[Ellipsis, None],
                  jnp.tile(emb[:, None, :, None], (1, rel.shape[1], 1, 1)),
              ),
              axis=-1,
          )
      )

    rel = jax.nn.silu(rel)
    rel = nn.Dense(features=self.hdim)(rel)
    rel = jax.nn.silu(rel)

    # Aggregate
    feat = jnp.sum(x[neigh, :, None] * weights[Ellipsis, None, None] * rel, axis=1)

    # Final linear layer without bias to remain equivariant
    if not self.grouped:
      feat = nn.Dense(
          features=self.features, use_bias=False, dtype=jnp.complex64
      )(jnp.reshape(feat, (feat.shape[0], -1)))
    else:
      group = self.param(
          'group',
          nn.initializers.lecun_normal(),
          (feat.shape[-2], feat.shape[-1], 2),
          jnp.float32,
      )
      group = jax.lax.complex(group[Ellipsis, 0], group[Ellipsis, 1])
      feat = jnp.sum(feat * group[None, Ellipsis], axis=-1)
    return feat


class fconv_layer(nn.module.Module):  # pylint: disable=invalid-name
  """Field conv layer. Normalization -> Activation -> Field Conv."""

  features: int
  num_layers: int = 1
  hdim: int = 8
  grouped: bool = False

  def setup(self):

    self.mlp = neuron_mlp(features=self.features, num_layers=self.num_layers)

    self.conv = nn.checkpoint(FieldConv)(
        features=self.features, grouped=self.grouped, hdim=self.hdim
    )

  def __call__(
      self, x, mass, neigh, logs, weights
  ):

    # x = self.act(self.norm(x, mass))

    x = self.mlp(x, mass)

    rel = get_rel_pos(x, neigh, logs)

    x = self.conv(x, neigh, rel, weights)

    return x


class fconv_emb_layer(nn.module.Module):  # pylint: disable=invalid-name
  """Field conv layer w/ timestep embedding."""

  # Normalization -> Activation -> Field Conv w/ embedding.
  features: int
  num_layers: int = 1
  hdim: int = 8
  grouped: bool = False

  def setup(self):
    self.mlp = neuron_mlp(features=self.features, num_layers=self.num_layers)
    self.conv = nn.checkpoint(FieldConvEmb)(
        features=self.features, grouped=self.grouped, hdim=self.hdim
    )

  def __call__(
      self,
      x,
      mass,
      neigh,
      logs,
      weights,
      emb):
    x = self.mlp(x, mass)
    rel = get_rel_pos(x, neigh, logs)
    x = self.conv(x, neigh, rel, weights, emb)
    return x


class FCEResBlock(nn.module.Module):
  """Field conv. ResNet block with temporal embedding."""

  features: int
  hdim: int = 8
  last: bool = False
  num_layers: int = 1

  def setup(self):
    """Setup."""
    if self.last:
      self.norm = square_mag_norm()
      self.act = act_neuron()

  @nn.module.compact
  def __call__(
      self,
      x,
      emb,
      mass,
      neigh,
      logs,
      weights,
  ):

    # Residual connection
    res = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    x = fconv_layer(
        features=self.features, num_layers=self.num_layers, hdim=self.hdim
    )(x, mass, neigh, logs, weights)

    # Embedding
    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))
    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))

    x = fconv_emb_layer(
        features=self.features, num_layers=self.num_layers, hdim=self.hdim
    )(x, mass, neigh, logs, weights, emb)

    if not self.last:
      return x + res
    else:
      return self.act(self.norm(x + res, mass))


class FCEResBlock2(nn.module.Module):
  """Field conv. ResNet block with temporal embedding."""

  features: int
  last: bool = False
  num_layers: int = 1

  def setup(self):
    """Setup."""
    if self.last:
      self.norm = square_mag_norm()
      self.act = act_neuron()

  @nn.module.compact
  def __call__(
      self,
      x,
      emb,
      mass,
      neigh,
      logs,
      weights,
      bases,
  ):

    # Residual connection
    res = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    x = fconv_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh, logs, weights
    )

    # Embedding
    emb = jax.nn.silu(nn.Dense(features=3 * self.features)(emb))
    emb = jax.nn.silu(nn.Dense(features=3 * self.features)(emb))

    emb = jnp.reshape(emb, (self.features, 3))

    emb = jnp.matmul(bases[:, None, Ellipsis], emb[None, Ellipsis, None])[Ellipsis, 0]
    emb = jax.lax.complex(emb[Ellipsis, 0], emb[Ellipsis, 1])

    x = x + emb

    x = fconv_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh, logs, weights
    )

    if not self.last:
      return x + res
    else:
      return self.act(self.norm(x + res, mass))


class FCEResBlock_mult(nn.module.Module):  # pylint: disable=invalid-name
  """Field conv. ResNet block with temporal embedding."""

  features: int
  last: bool = False
  num_layers: int = 1

  def setup(self):
    """Setup."""
    if self.last:
      self.norm = square_mag_norm()
      self.act = act_neuron()

  @nn.module.compact
  def __call__(
      self,
      x,
      emb,
      mass,
      neigh,
      logs,
      weights,
  ):

    # Residual connection
    res = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    x = fconv_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh, logs, weights
    )

    # Embedding
    emb = jax.nn.silu(nn.Dense(features=2 * self.features)(emb))
    emb = nn.Dense(features=2 * self.features)(emb)

    if jnp.ndim(emb) == 1:
      emb = jax.lax.complex(emb[: self.features], emb[self.features :])
      x = x * emb[None, :]
    else:
      emb = jax.lax.complex(emb[:, : self.features], emb[:, self.features :])
      x = x * emb

    x = fconv_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh, logs, weights
    )

    if not self.last:
      return x + res
    else:
      return self.act(self.norm(x + res, mass))


class FCENextBlock(nn.module.Module):
  """Field conv. ResNet block with temporal embedding."""

  features: int
  last: bool = False
  num_layers: int = 1

  def setup(self):
    """Setup."""
    if self.last:
      self.norm = square_mag_norm()
      self.act = act_neuron()

  @nn.module.compact
  def __call__(
      self,
      x,
      emb,
      mass,
      neigh,
      logs,
      weights,
  ):

    # Residual connection
    res = nn.Dense(
        features=self.features, use_bias=False, param_dtype=jnp.complex64
    )(x)

    x = fconv_layer(
        features=self.features, num_layers=self.num_layers, grouped=True
    )(x, mass, neigh, logs, weights)

    # Embedding
    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))
    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))

    x = fconv_emb_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh[Ellipsis, :8], logs[Ellipsis, :8, :], weights[Ellipsis, :8], emb
    )
    x = fconv_layer(features=self.features, num_layers=self.num_layers)(
        x, mass, neigh[Ellipsis, :8], logs[Ellipsis, :8, :], weights[Ellipsis, :8]
    )

    if not self.last:
      return x + res
    else:
      return self.act(self.norm(x + res, mass))


################################################################################
# Pooling / Unpooling.
#################################################################################


def scatter_stencil(
    signal, target, cols, weights
):
  """Scattering helper function."""
  rows = jnp.reshape(
      jnp.tile(jnp.arange(cols.shape[0])[:, None], (1, cols.shape[1])), (-1,)
  )
  cols = jnp.reshape(cols, (-1,))
  weights = jnp.reshape(weights, (-1,))
  return target.at[cols].add(weights * signal[rows])


def upsample_tangent(x, cols, xport):
  """NN upsampling of tangent vector features w/ transport."""
  return (
      x[cols[:, 0], Ellipsis]
      * jax.lax.complex(jnp.cos(xport[:, 0, 1]), jnp.sin(xport[:, 0, 1]))[
          :, None
      ]
  )


@functools.partial(jax.jit, static_argnames=['out_dim'])
def downsample_tangent(
    x, cols, xport, mass, out_dim
):
  """Avg. pool of tangent vector features w/ paralell transport."""
  num_low_res = out_dim
  weights = jnp.tile(mass[:, None], (1, cols.shape[1]))
  target = jnp.zeros((num_low_res, x.shape[1]), dtype=x.dtype)

  weight_sum = jnp.zeros((num_low_res,), dtype=x.dtype)
  weight_sum = (
      scatter_stencil(jnp.ones_like(x[:, 0]), weight_sum, cols, weights)
      + 1.0e-6
  )
  weights = weights * jax.lax.complex(
      jnp.cos(xport[Ellipsis, 0]), jnp.sin(xport[Ellipsis, 0])
  )
  target = jax.vmap(scatter_stencil, (1, 1, None, None), 1)(
      x, target, cols, weights
  )
  return target / weight_sum[:, None]


################################################################################
# Quantized projection.
#################################################################################


Initializer = nn.initializers.Initializer
DTypeLikeInexact = Any


def embed_initializer(
    dtype = jnp.float_,
    scale = 1.0):
  """Embed initializer."""
  def init(
      key, shape,
      dtype = dtype):
    """Initializer."""
    return jax.random.uniform(
        key, shape, dtype=dtype, minval=-1.0 * scale, maxval=1.0 * scale)
  return init


class quantized_projection(nn.module.Module):  # pylint: disable=invalid-name
  """Quantized projection."""
  features: int
  code_dim: int = 16
  beta: float = 1.0

  @nn.module.compact
  def __call__(self, emb, bases):
    codes = self.param(
        'codes',
        embed_initializer(),
        (self.code_dim, self.features, 2),
        jnp.float32,
    )
    codes = jax.lax.complex(codes[Ellipsis, 0], codes[Ellipsis, 1])

    def _quantize(z, codes):
      """Quantize."""
      inner = jnp.sum(jnp.conjugate(codes)[None, Ellipsis] * z[:, None, :], axis=-1)
      alpha = jnp.real(inner)
      beta = jnp.imag(inner)
      pi_mask = 1.0 * jnp.greater(0.0, alpha)

      alpha_mask = jnp.greater(jnp.abs(alpha), 1.0e-4)

      alpha_masked = alpha * alpha_mask + 1.0e-4 * (1.0 - alpha_mask)

      theta = jnp.arctan2(beta, alpha_masked) + math.pi * pi_mask

      diff = (
          jnp.sum(jnp.real(jnp.conjugate(z)) * z, axis=-1)[:, None]
          + jnp.sum(jnp.real(jnp.conjugate(codes) * codes), axis=-1)[None, :]
          - 2.0 * (alpha * jnp.cos(theta) + beta * jnp.sin(theta))
      )

      ind = jnp.argmin(diff, axis=1)

      theta_q = theta[jnp.arange(ind.shape[0]), ind]
      rot = jax.lax.complex(jnp.cos(theta_q), jnp.sin(theta_q))

      z_q = codes[ind, :]

      loss_comm = jnp.mean(
          jnp.abs(jax.lax.stop_gradient(rot[Ellipsis, None] * z_q) - z)
      )
      loss_code = jnp.mean(
          jnp.abs(
              z_q - jax.lax.stop_gradient(jnp.conjugate(rot)[Ellipsis, None] * z)
          )
      )

      z_q = z + jax.lax.stop_gradient(rot[:, None] * z_q - z)

      loss = 0.5 * (loss_comm - loss_code)

      return z_q, loss

    emb = jax.nn.silu(nn.Dense(features=3 * self.features)(emb))
    emb = jax.nn.silu(nn.Dense(features=3 * self.features)(emb))

    emb = jnp.reshape(emb, (self.features, 3))

    emb = jnp.matmul(bases[:, None, Ellipsis], emb[None, Ellipsis, None])[Ellipsis, 0]
    emb = jax.lax.complex(emb[Ellipsis, 0], emb[Ellipsis, 1])

    return _quantize(emb, codes)


class quantized_embedding(nn.module.Module):  # pylint: disable=invalid-name
  """Quantized embedding."""
  features: int
  code_dim: int = 8
  beta: float = 1.0

  @nn.module.compact
  def __call__(self, emb):

    codes = self.param(
        'codes',
        embed_initializer(),
        (self.code_dim, self.features),
        jnp.float32,
    )

    def _quantize(z, codes):
      """Quantize."""
      q_ind = jnp.argmin(
          jnp.sum(jnp.abs(z[:, None, :] - codes[None, Ellipsis]), axis=-1), axis=-1
      )

      z_q = codes[q_ind, :]

      loss_comm = jnp.mean(
          jnp.sum(jnp.abs(jax.lax.stop_gradient(z_q) - z), axis=-1)
      )
      loss_code = jnp.mean(
          jnp.sum(jnp.abs(z_q - jax.lax.stop_gradient(z)), axis=-1)
      )

      z_q = z + jax.lax.stop_gradient(z_q - z)

      return z_q, loss_comm, loss_code

    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))
    emb = jax.nn.silu(nn.Dense(features=self.features)(emb))
    emb = nn.Dense(features=self.features)(emb)

    z_q, loss_comm, loss_code = _quantize(emb, codes)

    loss = 0.5 * (loss_comm + loss_code) / self.features

    return z_q, loss
