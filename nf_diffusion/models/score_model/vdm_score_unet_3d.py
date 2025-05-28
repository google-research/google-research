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

"""ScoreUNet adapted from VDM public cobase."""
from typing import Optional
import flax.linen as nn
import jax.numpy as jnp
from nf_diffusion.models.layers import ldm_utils


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

    B, _, _, _, C = x.shape  # pylint: disable=invalid-name
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3, 3), strides=(1, 1, 1),
        name='conv1')(
            h)

    # add in conditioning
    if cond is not None:
      assert cond.shape[0] == B and len(cond.shape) == 2
      h += nn.Dense(
          features=out_ch,
          use_bias=False,
          kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)[:, None, None, None, :]

    h = nonlinearity(normalize2(h))
    if self.p_drop is not None:
      h = nn.Dropout(rate=self.p_drop)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(
            h)

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
    # B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    B, H, W, D, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0

    normalize = nn.normalization.GroupNorm()

    h = normalize(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h)
      k = nn.Dense(features=C, name='k')(h)
      v = nn.Dense(features=C, name='v')(h)
      h = ldm_utils.dot_product_attention(
          q[:, :, :, :, None, :],
          k[:, :, :, :, None, :],
          v[:, :, :, :, None, :],
          axis=(1, 2, 3))[:, :, :, :, 0, :]
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(
              h)
    else:
      head_dim = C // self.num_heads
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (B, H, W, self.num_heads,
                                               head_dim)
      h = ldm_utils.dot_product_attention(q, k, v, axis=(1, 2))
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(
              h)

    assert h.shape == x.shape
    return x + h


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
  def __call__(self, z, t=None, conditioning=None, train=True, **kwargs):
    g_t = t
    deterministic = not train
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

    temb = ldm_utils.get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, conditioning], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if self.with_fourier_features:
      z_f = ldm_utils.Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(
        features=n_embd,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        name='conv_in')(
            h)
    hs = [h]

    # TODO(guandao) but not downsampling it?
    # Downsampling
    for i_block in range(self.n_layers):
      block = ResnetBlock(
          out_ch=n_embd, p_drop=self.p_drop, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if self.with_attention:
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(
        p_drop=self.p_drop, name='mid.block_1')(h, cond, deterministic)[0]
    if self.with_attention:
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
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(
            h)

    # Base measure
    eps_pred += z

    return eps_pred
