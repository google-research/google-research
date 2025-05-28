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

"""Latent Diffusion Model utilities."""
from dataclasses import field
from typing import Sequence, Optional, Iterable
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def constant_init(value, dtype='float32'):

  def _init(key, shape, dtype=dtype):
    return value * jnp.ones(shape, dtype)
  return _init


class ResNet1D(nn.Module):
  """ResNet backbone (1D)."""
  hidden_size: int = 512
  n_layers: int = 1
  middle_size: int = 1024

  @nn.compact
  def __call__(self, x, cond=None):
    assert x.shape[-1] == self.hidden_size, 'Input must be hidden size.'
    z = x
    for _ in range(self.n_layers):
      h = nn.gelu(nn.LayerNorm()(z))
      h = nn.Dense(self.middle_size)(h)
      if cond is not None:
        h += nn.Dense(self.middle_size, use_bias=False)(cond)
      h = nn.gelu(nn.LayerNorm()(h))
      h = nn.Dense(self.hidden_size, kernel_init=jax.nn.initializers.zeros)(h)
      z = z + h
    return z


class ResNet2D(nn.Module):
  """ResNet backbone (with conv)."""
  hidden_size: int = 256
  n_layers: int = 1
  middle_size: int = 512
  down_sample: Sequence[int] = field(default_factory=lambda: [])
  up_sample: Sequence[int] = field(default_factory=lambda: [])

  @nn.compact
  def __call__(self, x, cond=None):
    assert x.shape[-1] == self.hidden_size, "Input must be hidden size."
    z = x
    for i in range(self.n_layers):
      h = nn.gelu(nn.LayerNorm()(z))
      h = nn.Conv(self.middle_size, kernel_size=(3, 3))(h)
      if cond is not None:
        hc = nn.Dense(self.middle_size, use_bias=False)(cond)
        hc_2d = hc.reshape(*x.shape[:-3], 1, 1, -1)
        h += hc_2d
      h = nn.gelu(nn.LayerNorm()(h))
      h = nn.Conv(
          self.hidden_size,
          kernel_size=(3, 3),
          kernel_init=jax.nn.initializers.zeros)(
              h)
      z = z + h

      if i in self.down_sample:
        B, H, W, C = z.shape
        z = jax.image.resize(
            z, shape=(B, H // 2, W // 2, C),
            method=jax.image.ResizeMethod.LINEAR)
      if i in self.up_sample:
        B, H, W, C = z.shape
        z = jax.image.resize(
            z, shape=(B, H * 2, W * 2, C),
            method=jax.image.ResizeMethod.LINEAR)
    return z


# Noise scheduling!


class NoiseSchedule_Scalar(nn.Module):
  """Noise scheduler Scalar."""

  gamma_min: float = -13.3
  gamma_max: float = 5.0

  def setup(self):
    init_bias = self.gamma_min
    init_scale = jnp.log(self.gamma_max - init_bias)
    self.w = self.param('w', constant_init(init_scale), (1,))
    self.b = self.param('b', constant_init(init_bias), (1,))

  @nn.compact
  def __call__(self, t):
    return self.b + jnp.exp(self.w) * t


class NoiseSchedule_FixedLinear(nn.Module):
  """Noise scheduler FIxed Linear."""

  gamma_min: float = -13.3
  gamma_max: float = 5.0

  @nn.compact
  def __call__(self, t):
    out = self.gamma_min + (self.gamma_max - self.gamma_min) * t
    return out


######### Score model #########


class ScoreUNet(nn.Module):
  """ScoreUNet adapted from the project."""

  n_embd: int
  n_layers: int
  gamma_min: float
  gamma_max: float
  with_fourier_features: bool
  with_attention: bool
  p_drop: float

  @nn.compact
  def __call__(self, z, g_t, conditioning, deterministic=True):
    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.n_embd

    lb = self.gamma_min
    ub = self.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
      t = jnp.ones((z.shape[0],), z.dtype) * t
    elif len(t.shape) == 0:
      t = jnp.tile(t[None], z.shape[0])

    temb = get_timestep_embedding(t, n_embd)
    # print(temb.shape, conditioning.shape, conditioning[:, None].shape)
    # cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = jnp.concatenate([temb, conditioning], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if self.with_fourier_features:
      z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h]

    # TODO(guandao) but not downsampling it?
    # Downsampling
    for i_block in range(self.n_layers):
      block = ResnetBlock(out_ch=n_embd, p_drop=self.p_drop,
                          name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if self.with_attention:
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(
        p_drop=self.p_drop, name='mid.block_1')(h, cond, deterministic)[0]
    h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(
        p_drop=self.p_drop, name='mid.block_2')(h, cond, deterministic)[0]

    # TODO(guandao) but not downsampling it?
    # Upsampling
    for i_block in range(self.n_layers + 1):
      b = ResnetBlock(
          out_ch=n_embd, p_drop=self.p_drop, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if self.with_attention:
        h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    eps_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)

    # Base measure
    eps_pred += z

    return eps_pred


def get_timestep_embedding(timesteps, embedding_dim, dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  assert len(timesteps.shape) == 1
  timesteps *= 1000.

  half_dim = embedding_dim // 2
  emb = np.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


######### ResNet block #########

class ResnetBlock(nn.Module):
  """Convolutional residual block with two convs."""
  # config: VDMConfig
  p_drop: Optional[float] = None
  out_ch: Optional[int] = None

  @nn.compact
  def __call__(self, x, cond, deterministic, enc=None):
    nonlinearity = nn.swish
    normalize1 = nn.normalization.GroupNorm()
    normalize2 = nn.normalization.GroupNorm()

    if enc is not None:
      x = jnp.concatenate([x, enc], axis=-1)

    B, _, _, C = x.shape  # pylint: disable=invalid-name
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in conditioning
    if cond is not None:
      assert cond.shape[0] == B and len(cond.shape) == 2
      h += nn.Dense(
          features=out_ch, use_bias=False, kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)[:, None, None, :]

    h = nonlinearity(normalize2(h))
    if self.p_drop is not None:
      h = nn.Dropout(rate=self.p_drop)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    x = x + h
    return x, x


class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: int

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0

    normalize = nn.normalization.GroupNorm()

    h = normalize(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h)
      k = nn.Dense(features=C, name='k')(h)
      v = nn.Dense(features=C, name='v')(h)
      h = dot_product_attention(
          q[:, :, :, None, :],
          k[:, :, :, None, :],
          v[:, :, :, None, :],
          axis=(1, 2))[:, :, :, 0, :]
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h)
    else:
      head_dim = C // self.num_heads
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (
          B, H, W, self.num_heads, head_dim)
      h = dot_product_attention(q, k, v, axis=(1, 2))
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(h)

    assert h.shape == x.shape
    return x + h


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          # broadcast_dropout=True,
                          # dropout_rng=None,
                          # dropout_rate=0.,
                          # deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.

  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])
  assert query.dtype == key.dtype == value.dtype
  input_dtype = query.dtype

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  key = key.astype(dtype)
  query = query.astype(dtype) / np.sqrt(depth)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  assert attn_weights.dtype == dtype
  attn_weights = attn_weights.astype(input_dtype)

  # compute the new values given the attention weights
  assert attn_weights.dtype == value.dtype
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  assert y.dtype == input_dtype
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class Base2FourierFeatures(nn.Module):
  start: int = 0
  stop: int = 8
  step: int = 1

  @nn.compact
  def __call__(self, inputs):
    freqs = range(self.start, self.stop, self.step)

    # Create Base 2 Fourier features
    w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
    w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

    # Compute features
    h = jnp.repeat(inputs, len(freqs), axis=-1)
    h = w * h
    h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
    return h


@flax.struct.dataclass
class VDMOutput:
  loss_recon: chex.Array  # [B]
  loss_klz: chex.Array  # [B]
  loss_diff: chex.Array  # [B]
  var_0: float
  var_1: float


######### Likelihood Functions #########


def rgb_categorical_likelihood(rgb_img, noise_level, vocab_size=256):
  """Create categorical likelihood for RGB images.

  Args:
    [rgb_img] shape=(bs, h, w, 3), range should be [-1, 1]
    [noise_level] float
    [vocab_size]

  Returns:
    tensorflow distribution
  """
  assert rgb_img.shape[-1] == 3 and len(rgb_img.shape) == 4
  # Logits are exact if there are no dependencies between dimensions of x
  x_vals = jnp.arange(0, vocab_size)[None, :] # (1, vocab_size)
  x_vals = jnp.repeat(x_vals, 3, 0) # (3, vocab_size)
  # x_vals = x_vals.round()
  # shape=(vocab_size, 3) range=[-1, 1])
  x_vals = 2 * ((x_vals +.5) / float(vocab_size)) - 1
  # shape=(1, 1,  3, vocab_size)
  x_vals = x_vals[None, None, :, :]

  inv_stdev = jnp.exp(-0.5 * noise_level)
  logits = -0.5 * jnp.square(
      (rgb_img[Ellipsis, None] - x_vals) * inv_stdev[Ellipsis, None])
  dist = tfd.Categorical(logits=logits)
  return dist
