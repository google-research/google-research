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

"""Latent Diffusion Model."""

import dataclasses
from typing import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from nf_diffusion.models.layers import ldm_utils
from nf_diffusion.models.ldm import get_timestep_embedding
from nf_diffusion.models.ldm import VDMOutput


tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(nn.Module):
  """Encoder."""

  hidden_size: int = 128
  n_layers: int = 3
  z_dim: int = 64
  down_sample: Sequence[int] = dataclasses.field(default_factory=lambda: [])

  @nn.compact
  def __call__(self, ims, cond=None, mask=None):
    x = 2 * ims.astype("float32") - 1.0
    if mask is not None:
      x = jnp.concatenate([x, mask], axis=-1)
    else:
      x = jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)

    # This is fully convolutional
    x = nn.Dense(self.hidden_size)(x)
    x = ldm_utils.ResNet2D(
        self.hidden_size, self.n_layers, down_sample=self.down_sample
    )(x, cond=cond)
    x = einops.rearrange(x, "... x y d -> ... (x y d)")
    x = nn.Dense(self.z_dim)(x)
    return x


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[Ellipsis, None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1],
    )
  else:
    xb = jnp.reshape(
        (x[Ellipsis, None, :] * scales[:, None]), list(x.shape[:-1]) + [-1]
    )
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)


class Decoder(nn.Module):
  """CodeNF decoder."""

  hidden_size: int = 512
  n_layers: int = 3
  H: int = 32
  W: int = 32
  min_deg: int = 0
  max_deg: int = 10

  def make_2d_grid(self):
    xs, ys = jnp.meshgrid(jnp.arange(self.H), jnp.arange(self.W))
    xs, ys = xs.astype(jnp.float32), ys.astype(jnp.float32)
    xs = xs / float(self.H - 1)
    ys = ys / float(self.W - 1)
    ttl = self.H * self.W
    coord = jnp.concatenate([xs.reshape(ttl, 1), ys.reshape(ttl, 1)], axis=1)
    coord -= 0.5
    return coord

  def setup(self):
    self.grid = jnp.array(self.make_2d_grid()).reshape(1, -1, 2)
    self.grid_posenc = posenc(
        self.grid, min_deg=self.min_deg, max_deg=self.max_deg
    )

  @nn.compact
  def __call__(self, z, cond=None, mask=None):
    npts = self.grid.shape[1]
    z = jnp.repeat(jnp.expand_dims(z, axis=1), npts, axis=1)
    grid_posenc = jnp.repeat(self.grid_posenc, z.shape[0], axis=0)
    z = jnp.concatenate([z, grid_posenc], axis=-1)
    if cond is not None:
      cond = jnp.expand_dims(cond, axis=1)
    z = nn.Dense(self.hidden_size)(z)
    z = ldm_utils.ResNet1D(self.hidden_size, self.n_layers)(z, cond=cond)

    logits = nn.Dense(1)(z)
    logits = einops.rearrange(
        logits, "... (x y) d -> ... x y d", x=self.H, y=self.W, d=1
    )
    dist = tfd.Independent(tfd.Bernoulli(logits=logits), 3)  # (... 28 28 1)
    if mask is not None:
      dist = tfd.Masked(dist, mask)
    return dist


class ScoreNet(nn.Module):
  """ScoreNet."""

  embedding_dim: int = 64
  n_layers: int = 10

  @nn.compact
  def __call__(self, z, g_t, cond, mask=None, deterministic=True):
    n_embd = self.embedding_dim

    t = g_t
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    t = t * jnp.ones(z.shape[0])  # ensure t is a vector

    temb = get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, cond], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense0")(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense1")(cond))
    cond = nn.Dense(n_embd)(cond)

    h = nn.Dense(n_embd)(z)
    h = ldm_utils.ResNet1D(n_embd, self.n_layers)(h, cond)
    return z + h


class LDM(nn.Module):
  """LDM Model."""

  timesteps: int = 1000
  gamma_min: float = -3.0  # -13.3
  gamma_max: float = 3.0  # 5.0
  embedding_dim: int = 128
  antithetic_time_sampling: bool = True
  sm_layers: int = 32
  classes: int = 10 + 26 + 26 + 1
  enc_layers: int = 4
  dec_layers: int = 4
  downsample_layers: Sequence[int] = dataclasses.field(
      default_factory=lambda: []
  )
  latent_shape: Sequence[int] = dataclasses.field(default_factory=lambda: [])
  dec_type: str = "default"
  res: int = 32
  pe_min_deg: int = 0
  pe_max_deg: int = 10

  def setup(self):
    # TODO(guandao) choose noise schedule
    self.gamma = ldm_utils.NoiseSchedule_FixedLinear(
        gamma_min=self.gamma_min, gamma_max=self.gamma_max
    )
    self.score_model = ScoreNet(
        n_layers=self.sm_layers, embedding_dim=self.embedding_dim
    )
    self.encoder = Encoder(
        z_dim=self.embedding_dim,
        n_layers=self.enc_layers,
        down_sample=self.downsample_layers,
    )
    self.decoder = Decoder(
        n_layers=self.dec_layers,
        H=self.res,
        W=self.res,
        min_deg=self.pe_min_deg,
        max_deg=self.pe_max_deg,
    )
    self.embedding_vectors = nn.Embed(self.classes, self.embedding_dim)

  def gammat(self, t):
    return self.gamma(t)

  def __call__(self, images, cond, mask=None, deterministic = True):
    g_0, g_1 = self.gamma(0.0), self.gamma(1.0)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    x = images
    n_batch = images.shape[0]

    cond_v = self.embedding_vectors(cond)

    # 1. RECONSTRUCTION LOSS
    f = self.encoder(x, cond_v, mask=mask)

    eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0  # = z_0 / sqrt(1-var)
    likelihood = self.decoder(z_0_rescaled, cond_v, mask=mask)
    loss_recon = -likelihood.log_prob(x.astype("int"))

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = (1.0 - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1.0, axis=-1)

    # 3. DIFFUSION LOSS
    # sample time steps
    rng1 = self.make_rng("sample")
    if self.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / n_batch), 1.0)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    ttl_t = self.timesteps
    t = jnp.ceil(t * ttl_t) / ttl_t

    # sample z_t
    g_t = self.gamma(t)
    var_t = nn.sigmoid(g_t)[:, None, None, None]
    eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_t = jnp.sqrt(1.0 - var_t) * f + jnp.sqrt(var_t) * eps
    # compute predicted noise
    eps_hat = self.score_model(z_t, g_t, cond_v, deterministic)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=(-3, -2, -1))

    # loss for finite depth T, i.e. discrete time
    s = t - (1.0 / ttl_t)
    g_s = self.gamma(s)
    loss_diff = 0.5 * ttl_t * jnp.expm1(g_t - g_s) * loss_diff_mse

    # End of diffusion loss computation
    img = likelihood.mean()
    return (
        VDMOutput(
            loss_recon=loss_recon,
            loss_klz=loss_klz,
            loss_diff=loss_diff,
            var_0=var_0,
            var_1=var_1,
        ),
        img,
    )

  def sample_step(self, i, T, z_t, cond, rng):
    """Sample a step [i] out total [ttl_t]."""
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)
    t = (T - i) / T
    s = (T - i - 1) / T

    g_s = self.gamma(s)
    g_t = self.gamma(t)

    cond_v = self.embedding_vectors(cond)

    eps_hat = self.score_model(
        z_t,
        g_t * jnp.ones((z_t.shape[0]), z_t.dtype),
        cond_v,
        deterministic=True,
    )

    a = nn.sigmoid(-g_s)
    c = -jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s = (
        jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t))
        * (z_t - sigma_t * c * eps_hat)
        + jnp.sqrt((1.0 - a) * c) * eps
    )
    return z_s

  # TODO(guandao) colab has [rescale] == True
  def decode(
      self, z_0, cond=None, mask=None, rescale=False, rng=None, sample=False
  ):
    """Take a sampled z_0 and decode image out."""
    if rescale:
      g_0 = self.gamma(0.0)
      var_0 = nn.sigmoid(g_0)
      z_0_rescaled = z_0 / jnp.sqrt(1.0 - var_0)
    else:
      z_0_rescaled = z_0
    cond_v = self.embedding_vectors(cond)
    out_tfp = self.decoder(z_0_rescaled, cond=cond_v, mask=mask)
    if sample:
      if rng is None:
        rng = self.make_rng("sample")
      # TODO(guandao) this will give "int", if there is only two class, then it
      #   will create "int" with [0, 1], which won't write into images since the
      #   tensorbaord writer expect "int" to range from [0, 255]
      return out_tfp.sample(seed=rng).astype("float")
    else:
      return out_tfp.mean()

  def encode(self, ims, cond=None, mask=None):
    """Encode an image into z_0."""
    cond_v = self.embedding_vectors(cond)
    return self.encoder(ims, cond_v, mask=mask)
