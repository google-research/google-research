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

from dataclasses import field
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from nf_diffusion.models.layers import ldm_utils
from nf_diffusion.models.utils import resample
from nf_diffusion.trainers.metrics.image import compute_psnr
from nf_diffusion.trainers.metrics.image import compute_ssim


tfd = tfp.distributions
tfb = tfp.bijectors
ResNet = ldm_utils.ResNet2D
VDMOutput = ldm_utils.VDMOutput
get_timestep_embedding = ldm_utils.get_timestep_embedding
ScoreUNet = ldm_utils.ScoreUNet


class Encoder(nn.Module):
  """Encoder."""

  hidden_size: int = 128
  n_layers: int = 3
  z_dim: int = 64
  down_sample: Sequence[int] = field(default_factory=lambda: [])
  num_colors: int = 256

  @nn.compact
  def __call__(self, ims, cond=None, mask=None):
    if ims.dtype in [jnp.uint8, jnp.int32]:
      ims = ims.astype("float32") / float(self.num_colors - 1)
    assert ims.dtype == jnp.float32
    x = 2 * ims - 1.0
    if mask is not None:
      x = jnp.concatenate([x, mask], axis=-1)
    else:
      x = jnp.concatenate([x, jnp.ones((*x.shape[:-1], 1))], axis=-1)

    # This is fully convolutional
    x = nn.Dense(self.hidden_size)(x)
    x = ResNet(
        self.hidden_size, self.n_layers, down_sample=self.down_sample)(
            x, cond=cond)

    x = nn.Dense(self.z_dim)(x)
    return x


class DecoderNF(nn.Module):
  """Decoder with Neural Field."""

  h: int = 32
  w: int = 32
  hidden_size: int = 256
  n_layers: int = 3
  is_rgb: bool = False
  likelihood_scale: float = 1.
  likelihood_type: str = "bernoulli"
  num_colors: int = 256
  nf_type: str = "res"

  def make_2d_grid(self):
    xs, ys = jnp.meshgrid(jnp.arange(self.w), jnp.arange(self.h))
    xs, ys = xs.astype(jnp.float32), ys.astype(jnp.float32)
    xs = xs / float(self.w - 1)
    ys = ys / float(self.h - 1)
    ttl = self.h * self.w
    coord = jnp.concatenate([
        xs.reshape(ttl, 1), ys.reshape(ttl, 1)], axis=1)
    coord -= 0.5
    return coord

  def setup(self):
    self.grid = self.make_2d_grid()
    self.grid = self.grid.reshape(1, self.h * self.w, 2)

  @nn.compact
  def __call__(self, z, cond=None, mask=None, inp_grid=None, scale=None):
    assert len(z.shape) == 4
    bs = z.shape[0]
    z = nn.Dense(self.hidden_size)(z)  # (..., h, w, hid_size)
    if inp_grid is None:
      grid = self.grid
    else:
      grid = inp_grid
    assert grid.shape[0] == 1 and len(grid.shape) == 3
    npts = grid.shape[1]

    # (..., npts/h*w, hid_size)
    grid = (grid + 0.5) * (
        jnp.array(z.shape[1:3]).astype(jnp.float32).reshape(1, 1, 2) - 1)
    z = resample.resample_2d(z, grid).reshape(bs, npts, -1)
    cond = jnp.expand_dims(cond, 1)
    z = ldm_utils.ResNet1D(self.hidden_size, self.n_layers)(z, cond=cond)

    if inp_grid is None:
      if self.is_rgb:
        loc = nn.Dense(3)(z)
        # (bs, H, W, 3, 256) -> this is because [z] is guaranteed to be 4D input
        loc = loc.reshape(-1, self.h, self.w, 3)
        # TODO(guandao) - ELBO needs to predict the scale
        if scale is None:
          scale = jnp.ones_like(loc) * self.likelihood_scale
        if self.likelihood_type == "categorical":
          dist = ldm_utils.rgb_categorical_likelihood(
              loc, scale, vocab_size=self.num_colors)
        elif self.likelihood_type == "normal":
          dist = tfd.Normal(loc=loc, scale=scale)  #(... H W 3)
        else:
          raise NotImplementedError
      else:
        assert self.likelihood_type == "bernoulli"
        logits = nn.Dense(1)(z)
        logits = logits.reshape(-1, self.h, self.w, 1)  # (bs, H, W, 1)
        dist = tfd.Bernoulli(logits=logits)  #(... H W 1)

      if mask is not None:
        dist = tfd.Masked(dist, mask)
      dist = tfd.Independent(dist, 3)  #(... H W 1)
      return dist
    else:
      raise NotImplementedError


class Decoder(nn.Module):
  """Decoder."""

  hidden_size: int = 64
  n_layers: int = 3
  up_sample: Sequence[int] = field(default_factory=lambda: [])
  is_rgb: bool = False
  likelihood_scale: float = 1.
  likelihood_type: str = "categorical"
  num_colors: int = 256
  h: int = 32
  w: int = 32

  @nn.compact
  def __call__(self, z, cond=None, mask=None, scale=None):
    z = nn.Dense(self.hidden_size)(z)
    z = ResNet(
        self.hidden_size, self.n_layers, up_sample=self.up_sample)(
            z, cond=cond)

    if self.is_rgb:
      loc = nn.Dense(3)(z)
      # (bs, H, W, 3, 256) -> this is because [z] is guaranteed to be 4D input
      loc = loc.reshape(-1, self.h, self.w, 3)
      # TODO(guandao) - ELBO needs to predict the scale
      if scale is None:
        scale = jnp.ones_like(loc) * self.likelihood_scale
      if self.likelihood_type == "categorical":
        dist = ldm_utils.rgb_categorical_likelihood(
            loc, scale, vocab_size=self.num_colors)
      elif self.likelihood_type == "normal":
        dist = tfd.Normal(loc=loc, scale=scale)  #(... H W 3)
      else:
        raise NotImplementedError
    else:
      assert self.likelihood_type == "bernoulli"
      logits = nn.Dense(1)(z)
      logits = logits.reshape(-1, self.h, self.w, 1)  # (bs, H, W, 1)
      dist = tfd.Bernoulli(logits=logits)  #(... H W 1/3)

    if mask is not None:
      dist = tfd.Masked(dist, mask)
    dist = tfd.Independent(dist, 3)  # (... H W 1)
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
    h = ResNet(n_embd, self.n_layers)(h, cond)
    return z + h


class LDM(nn.Module):
  """LDM Model."""

  timesteps: int = 1000
  gamma_min: float = -3.0  # -13.3
  gamma_max: float = 3.0   # 5.0
  embedding_dim: int = 128
  antithetic_time_sampling: bool = True
  sm_layers: int = 32
  classes: int = 10 + 26 + 26 + 1
  enc_layers: int = 4
  enc_hid_size: int = 128
  dec_layers: int = 4
  dec_hid_size: int = 128
  change_res_layers: Sequence[int] = field(default_factory=lambda: [])
  latent_shape: Sequence[int] = field(default_factory=lambda: [])
  dec_type: str = "default"  # default/nfdec
  sm_type: str = "default"  # default/unet
  res: int = 32
  is_rgb: bool = False
  likelihood_scale: float = 1.
  likelihood_type: str = "bernoulli"

  with_fourier_features: bool = True
  with_attention: bool = False
  p_drop: float = 0.1
  num_colors: int = 256
  sm_n_embd: int = 256

  def setup(self):
    # TODO(guandao) choose noise schedule
    self.gamma = ldm_utils.NoiseSchedule_FixedLinear(
        gamma_min=self.gamma_min, gamma_max=self.gamma_max)

    if self.sm_type == "default":
      self.score_model = ScoreNet(n_layers=self.sm_layers,
                                  embedding_dim=self.embedding_dim)
    elif self.sm_type == "unet":
      self.score_model = ScoreUNet(
          n_layers=self.sm_layers,
          n_embd=self.sm_n_embd,
          gamma_min=self.gamma_min,
          gamma_max=self.gamma_max,
          with_fourier_features=self.with_fourier_features,
          with_attention=self.with_attention,
          p_drop=self.p_drop)
    else:
      raise ValueError

    self.encoder = Encoder(
        z_dim=self.embedding_dim,
        n_layers=self.enc_layers,
        hidden_size=self.enc_hid_size,
        down_sample=self.change_res_layers)

    if self.dec_type == "default":
      self.decoder = Decoder(
          up_sample=self.change_res_layers,
          n_layers=self.dec_layers,
          h=self.res,
          w=self.res,
          is_rgb=self.is_rgb,
          likelihood_scale=self.likelihood_scale,
          likelihood_type=self.likelihood_type,
          hidden_size=self.dec_hid_size,
          num_colors=self.num_colors)
    elif self.dec_type == "nfdec":
      self.decoder = DecoderNF(
          n_layers=self.dec_layers,
          h=self.res,
          w=self.res,
          is_rgb=self.is_rgb,
          likelihood_scale=self.likelihood_scale,
          likelihood_type=self.likelihood_type,
          num_colors=self.num_colors)
    else:
      raise ValueError

    self.embedding_vectors = nn.Embed(self.classes, self.embedding_dim)

  def get_embedding_vectors(self, cond=None):
    if cond is not None:
      return self.embedding_vectors(cond)
    else:
      return None

  def get_score(self, z, t, g_t, cond, cond_v, train, g_t_conditioned=True):
    if g_t_conditioned:
      eps_hat = self.score_model(z, g_t, cond_v, not train)
    else:
      eps_hat = self.score_model(z, t, cond_v, not train)
    return eps_hat

  def gammat(self, t):
    return self.gamma(t)

  def __call__(self, images, cond, mask=None,
               train = False,
               deterministic = True):
    assert (not train) == deterministic
    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    x = images
    n_batch = images.shape[0]

    cond_v = self.embedding_vectors(cond)

    # 1. RECONSTRUCTION LOSS
    f = self.encoder(x, cond_v, mask=mask)

    eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0  # = z_0 / sqrt(1-var)

    likelihood = self.decoder(z_0_rescaled, cond_v, mask=mask)
    if self.likelihood_type == "gaussian":
      loss_recon = -likelihood.log_prob(x.astype("float"))
    else:
      loss_recon = -likelihood.log_prob(x.astype("int"))

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = (1. - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=-1)

    # 3. DIFFUSION LOSS
    # sample time steps
    rng1 = self.make_rng("sample")
    if self.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.0)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    ttl_t = self.timesteps
    t = jnp.ceil(t * ttl_t) / ttl_t

    # sample z_t
    g_t = self.gamma(t)
    var_t = nn.sigmoid(g_t)[:, None, None, None]
    eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    # compute predicted noise
    eps_hat = self.score_model(z_t, g_t, cond_v, deterministic)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=(-3, -2, -1))

    # loss for finite depth T, i.e. discrete time
    s = t - (1./ ttl_t)
    g_s = self.gamma(s)
    loss_diff = .5 * ttl_t * jnp.expm1(g_t - g_s) * loss_diff_mse

    # End of diffusion loss computation
    out = VDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=var_0,
        var_1=var_1,
    )
    if self.likelihood_type == "categorical":
      # TODO(guandao) make 255 a tunable nummber
      img = likelihood.mode() / float(self.num_colors - 1)
    else:
      img = likelihood.mean()
    return out, img

  def sample_step(self, i, T, z_t, cond, rng):
    """Sample a step [i] out total [ttl_t]."""
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)
    t = (T - i)/T
    s = (T - i - 1) / T

    g_s = self.gamma(s)
    g_t = self.gamma(t)

    cond_v = None
    if cond is not None:
      cond_v = self.embedding_vectors(cond)
    eps_hat = self.score_model(
        z_t,
        g_t * jnp.ones((z_t.shape[0]), z_t.dtype),
        cond_v,
        deterministic=True)

    a = nn.sigmoid(-g_s)
    c = -jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s = jnp.sqrt(nn.sigmoid(-g_s) /
                   nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) + jnp.sqrt(
                       (1. - a) * c) * eps
    return z_s

  # TODO(guandao) colab has [rescale] == True
  def decode(self, z_0, cond=None, mask=None, rescale=True, rng=None,
             sample=False, return_likelihood=False):
    """Take a sampled z_0 and decode image out."""
    if rescale:
      g_0 = self.gamma(0.)
      var_0 = nn.sigmoid(g_0)
      z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
    else:
      z_0_rescaled = z_0
    cond_v = self.embedding_vectors(cond)
    out_tfp = self.decoder(z_0_rescaled, cond=cond_v, mask=mask)
    if return_likelihood:
      return out_tfp
    if sample:
      if rng is None:
        rng = self.make_rng("sample")
      # TODO(guandao) this will give "int", if there is only two class, then it
      #   will create "int" with [0, 1], which won't write into images since the
      #   tensorbaord writer expect "int" to range from [0, 255]
      return out_tfp.sample(seed=rng).astype("float")
    else:
      if self.likelihood_type == "categorical":
        return out_tfp.mode() / float(self.num_colors - 1)
      else:
        return out_tfp.mean().astype("float")

  def encode(self, ims, cond=None, mask=None):
    """Encode an image into z_0."""
    cond_v = self.embedding_vectors(cond)
    return self.encoder(ims, cond_v, mask=mask)


def make_sample_fn(ldm):
  """Return a function that generate an image."""

  def sample_fn(params, rng, cond):
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    variables = {"params": params}
    # first generate latent
    rng, spl, spl2 = jax.random.split(rng, num=3)
    zt = jax.random.normal(spl,
                           (*cond.shape, *ldm.latent_shape, ldm.embedding_dim))
    print(params.keys())

    def body_fn(i, z_t):
      # sample_step(self, i, T, z_t, cond, rng):
      curr_rng = jax.random.fold_in(rng, i)
      return ldm.apply(
          variables,
          i=i,
          T=ldm.timesteps,
          z_t=z_t,
          cond=cond,
          rng=curr_rng,
          method=ldm.sample_step)

    z0 = jax.lax.fori_loop(
        lower=0, upper=ldm.timesteps, body_fun=body_fn, init_val=zt)
    return ldm.apply(variables, z0, cond, rng=spl2, method=ldm.decode)

  return sample_fn


def make_recon_fn(ldm, short_cut=False, return_zs=False):
  """Return a reconstruct the signal [ims]."""

  def recon(params, rng, t, ims, cond, mask=None):
    rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    rng, rng_dec = jax.random.split(rng)
    variables = {"params": params}
    # NOTE: this assumes [t] is from <=1 and >= 0
    T = ldm.timesteps
    tn = jnp.ceil(t * T)
    t = tn / T

    # NOTE: encode(self, ims, cond=None, mask=None):
    z_t = ldm.apply(variables, ims, cond=cond,
                    mask=mask, method=ldm.encode)

    if short_cut:
      # TODO(guandao) this assume t = 0, should we use one step to recover it?
      z0 = z_t
    else:
      # Reparameterization (i.e. sample z_t from the distribution predicted)
      g_t = ldm.apply(variables, t, method=ldm.gammat)
      var_t = nn.sigmoid(g_t)
      rng, spl1 = jax.random.split(rng)
      eps = jax.random.normal(spl1, shape=z_t.shape)
      zt = jnp.sqrt(1. - var_t) * z_t + jnp.sqrt(var_t) * eps

      # Run reverse procedure to get l_0
      def body_fn(i, z_t):
        # def sample_step(self, i, T, z_t, cond, rng):
        curr_rng = jax.random.fold_in(rng, i)
        return ldm.apply(
            variables,
            i=i,
            T=ldm.timesteps,
            z_t=z_t,
            cond=cond,
            rng=curr_rng,
            method=ldm.sample_step)

      z0 = jax.lax.fori_loop(
          lower=(T - tn).astype(jnp.int32),
          upper=ldm.timesteps, body_fun=body_fn, init_val=zt)
    out = ldm.apply(variables, z0, cond, rng=rng_dec, method=ldm.decode)

    metric_dict = {
        "psnr": compute_psnr(img0=ims, img1=out),
        "ssim": compute_ssim(img0=ims, img1=out, max_val=1.)
    }
    if return_zs:
      out = (out, z_t, z0)
    return out, metric_dict

  return recon
