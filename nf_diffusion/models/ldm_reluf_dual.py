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

"""Latent Diffusion Model trained with ReLU fields."""
from typing import Optional, Sequence

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from nf_diffusion.models.layers import ldm_utils
from nf_diffusion.models.score_model import ddpm_univdiff_3dreluf


def get_scheduler(name, config):
  """Helper function to get scheduler model."""
  if name == "fixed_linear":
    gamma = ldm_utils.NoiseSchedule_FixedLinear(**config)
  else:
    raise ValueError
  return gamma


def get_score_model(name, config):
  """Helper function to get score model."""
  if name == "ddpm_univdiff_reluf":
    score_model = ddpm_univdiff_3dreluf.UNet3D(
        density_config=config.density_config,
        color_config=config.color_config)
  else:
    raise ValueError
  return score_model


class LDM(nn.Module):
  """LDM Model."""

  # Score-based model configurations
  score_name: str
  score_config: ml_collections.ConfigDict
  gamma_name_den: str
  gamma_args_den: ml_collections.ConfigDict
  gamma_name_rgb: str
  gamma_args_rgb: ml_collections.ConfigDict
  g_t_conditioned: bool = True
  g_t_prenorm: bool = False
  timesteps: int = 1000
  cond_embedding_dim: int = 128
  antithetic_time_sampling: bool = True
  classes: int = 1
  latent_shape: Optional[Sequence[int]] = None
  embedding_dim: Optional[int] = None

  scale_factor: Optional[Sequence[float]] = None
  shift_factor: Optional[Sequence[float]] = None

  def setup(self):
    self.gamma_den = get_scheduler(self.gamma_name_den, self.gamma_args_den)
    self.gamma_rgb = get_scheduler(self.gamma_name_rgb, self.gamma_args_rgb)
    self.embedding_vectors = nn.Embed(self.classes, self.cond_embedding_dim)
    self.score_model = get_score_model(self.score_name, self.score_config)

  def gammat(self, t):
    return (self.gamma_den(t), self.gamma_rgb(t))

  def normalize_gt(self, g_t):
    g_t_den, g_t_rgb = g_t
    gmin_den = self.gamma_args_den.gamma_min
    gmax_den = self.gamma_args_den.gamma_max
    g_t_den = (g_t_den - gmin_den) / (gmax_den - gmin_den)

    gmin_rgb = self.gamma_args_rgb.gamma_min
    gmax_rgb = self.gamma_args_rgb.gamma_max
    g_t_rgb = (g_t_rgb - gmin_rgb) / (gmax_rgb - gmin_rgb)
    g_t = (g_t_den, g_t_rgb)
    return g_t

  def get_embedding_vectors(self, cond=None):
    if cond is not None:
      return self.embedding_vectors(cond)
    else:
      return None

  def get_score(self, z, t, g_t, cond, cond_v, train):
    # Some model expect condition input to be in a dictionary
    if self.score_name in ["ddpm_univdiff_reluf"]:
      if cond is not None:
        cond_v = {"class_id": cond}
      else:
        cond_v = {}

    if self.g_t_conditioned:
      if self.g_t_prenorm:
        g_t = self.normalize_gt(g_t)
      eps_hat = self.score_model(z, g_t, cond_v, train)
    else:
      eps_hat = self.score_model(z, t, cond_v, train)
    return eps_hat

  def normalize_vox(self, f):
    if self.scale_factor is not None and self.shift_factor is not None:
      nb = len(f.shape[:-1])
      scale_factor = jnp.array(self.scale_factor).reshape(*([1] * nb), -1)
      shift_factor = jnp.array(self.shift_factor).reshape(*([1] * nb), -1)
      f = (f - shift_factor) / scale_factor
    return f

  def denormalize_vox(self, f):
    if self.scale_factor is not None and self.shift_factor is not None:
      nb = len(f.shape[:-1])
      scale_factor = jnp.array(self.scale_factor).reshape(*([1] * nb), -1)
      shift_factor = jnp.array(self.shift_factor).reshape(*([1] * nb), -1)
      f = f * scale_factor + shift_factor
    return f

  def diffuse(self, f, t, return_eps=False):
    f_den, f_rgb = f[Ellipsis, :1], f[Ellipsis, 1:]
    g_t = self.gammat(t)
    var_t_den, var_t_rgb = jax.tree_util.tree_map(
        lambda x: nn.sigmoid(x)[:, None, None, None, None],
        g_t)
    eps_den = jax.random.normal(self.make_rng("sample"), shape=f_den.shape)
    z_t_den = jnp.sqrt(1. - var_t_den) * f_den + jnp.sqrt(var_t_den) * eps_den

    eps_rgb = jax.random.normal(self.make_rng("sample"), shape=f_rgb.shape)
    z_t_rgb = jnp.sqrt(1. - var_t_rgb) * f_rgb + jnp.sqrt(var_t_rgb) * eps_rgb

    z_t = jnp.concatenate([z_t_den, z_t_rgb], axis=-1)
    if return_eps:
      return z_t, (eps_den, eps_rgb)
    else:
      return z_t

  def compute_latent_loss(self, f, var_1):
    mean1_sqr = (1. - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1.,
                             axis=(-4, -3, -2, -1))
    return loss_klz

  def __call__(
      self,
      vox,
      cond=None,
      train = False,
  ):
    g_0, g_1 = self.gammat(0.), self.gammat(1.)
    var_0_den, var_0_rgb = jax.tree_util.tree_map(nn.sigmoid, g_0)
    var_1_den, var_1_rgb = jax.tree_util.tree_map(nn.sigmoid, g_1)
    n_batch = vox.shape[0]
    f = self.normalize_vox(vox)
    f_den, f_rgb = f[Ellipsis, :1], f[Ellipsis, 1:]
    cond_v = self.get_embedding_vectors(cond)

    # 2. LATENT LOSS.
    # KL z1 with N(0,1) prior
    # Normally this is not needed as with enough T this should be 0
    # But we will learn 1) encoder and 2) noise scheduler
    # So the latent loss makes it important to regularize the latent space.
    loss_klz_den = self.compute_latent_loss(f_den, var_1_den)
    loss_klz_rgb = self.compute_latent_loss(f_rgb, var_1_rgb)
    loss_klz = loss_klz_den + loss_klz_rgb

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

    # # sample z_t
    # g_t_den, g_t_rgb = self.gammat(t)
    # var_t_den = nn.sigmoid(g_t_den)[:, None, None, None, None]
    # eps_den = jax.random.normal(self.make_rng("sample"), shape=f_den.shape)
    # z_t_den = jnp.sqrt(1. - var_t_den) * f_den + jnp.sqrt(var_t_den) * eps_den
    # var_t_rgb = nn.sigmoid(g_t_rgb)[:, None, None, None, None]
    # eps_rgb = jax.random.normal(self.make_rng("sample"), shape=f_rgb.shape)
    # z_t_rgb = jnp.sqrt(1. - var_t_rgb) * f_rgb + jnp.sqrt(var_t_rgb) * eps_rgb
    # z_t = jnp.concatenate([z_t_den, z_t_rgb], axis=-1)

    # compute predicted noise
    g_t_den, g_t_rgb = self.gammat(t)
    z_t, (eps_den, eps_rgb) = self.diffuse(f, t, return_eps=True)
    eps_hat = self.get_score(z_t, t, (g_t_den, g_t_rgb), cond, cond_v, train)
    eps_hat_den = eps_hat[Ellipsis, :1]
    eps_hat_rgb = eps_hat[Ellipsis, 1:]
    # compute MSE of predicted noise
    loss_diff_mse_den = jnp.sum(
        jnp.square(eps_den - eps_hat_den), axis=(-4, -3, -2, -1))
    loss_diff_mse_rgb = jnp.sum(
        jnp.square(eps_rgb - eps_hat_rgb), axis=(-4, -3, -2, -1))

    # loss for finite depth T, i.e. discrete time
    s = t - (1. / ttl_t)
    g_s_den, g_s_rgb = self.gammat(s)
    # TODO(guandao) continuous version of the objective
    loss_diff_den = .5 * ttl_t * jnp.expm1(g_t_den -
                                           g_s_den) * loss_diff_mse_den
    loss_diff_rgb = .5 * ttl_t * jnp.expm1(g_t_rgb -
                                           g_s_rgb) * loss_diff_mse_rgb
    loss_diff = loss_diff_den + loss_diff_rgb

    # End of diffusion loss computation
    return {
        "loss_klz_den": loss_klz_den,
        "loss_klz_rgb": loss_klz_rgb,
        "loss_klz": loss_klz,
        "loss_diff": loss_diff,
        "loss_diff_den": loss_diff_den,
        "loss_diff_rgb": loss_diff_rgb,
        "var_0_den": var_0_den,
        "var_0_rgb": var_0_rgb,
        "var_1_den": var_1_den,
        "var_1_rgb": var_1_rgb,
    }

  def sample_step_i_d(self, i, d, T, z_t, cond, rng):
    """Sample a step [i] out total [ttl_t].

      NOTE: i, d are in the range of [0, max_steps].
      NOTE: i=0 means go from T_max to T_max - d
      NOTE: here, small i means small amount of denoising steps.
    """
    z_t_den, z_t_rgb = z_t[Ellipsis, :1], z_t[Ellipsis, 1:]
    rng_body = jax.random.fold_in(rng, i)
    eps_den = jax.random.normal(rng_body, z_t_den.shape)
    eps_rgb = jax.random.normal(rng_body, z_t_rgb.shape)
    t = (T - i) / T
    s = (T - i - d) / T

    g_s_den, g_s_rgb = self.gammat(s)
    g_t_den, g_t_rgb = self.gammat(t)

    cond_v = self.embedding_vectors(cond)
    t_inp = t * jnp.ones((z_t.shape[0],), z_t.dtype)
    g_t_den_inp = g_t_den * jnp.ones((z_t.shape[0],), z_t.dtype)
    g_t_rgb_inp = g_t_rgb * jnp.ones((z_t.shape[0],), z_t.dtype)
    eps_hat = self.get_score(z_t, t_inp, (g_t_den_inp, g_t_rgb_inp), cond,
                             cond_v, False)

    def _step_(z_t, eps_hat, eps, g_t, g_s):
      a = nn.sigmoid(-g_s)
      c = -jnp.expm1(g_s - g_t)
      sigma_t = jnp.sqrt(nn.sigmoid(g_t))
      z_s = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * (
          z_t - sigma_t * c * eps_hat) + jnp.sqrt((1. - a) * c) * eps
      return z_s

    eps_hat_den, eps_hat_rgb = eps_hat[Ellipsis, :1], eps_hat[Ellipsis, 1:]
    z_t_den, z_t_rgb = z_t[Ellipsis, :1], z_t[Ellipsis, 1:]
    z_s_den = _step_(z_t_den, eps_hat_den, eps_den, g_t_den, g_s_den)
    z_s_rgb = _step_(z_t_rgb, eps_hat_rgb, eps_rgb, g_t_rgb, g_s_rgb)
    z_s = jnp.concatenate((z_s_den, z_s_rgb), axis=-1)
    return z_s, T - i - d

  def sample_step(self, i, T, z_t, cond, rng):
    return self.sample_step_i_d(i, 1, T, z_t, cond, rng)[0]

  def rescale_z0(self, z0):
    g_0_den, g_0_rgb = self.gammat(0)
    var_0_den = nn.sigmoid(g_0_den)
    var_0_rgb = nn.sigmoid(g_0_rgb)
    z0_den, z0_rgb = z0[Ellipsis, :1], z0[Ellipsis, 1:]
    z0_den = z0_den / jnp.sqrt(1. - var_0_den)
    z0_rgb = z0_rgb / jnp.sqrt(1. - var_0_rgb)
    z0 = jnp.concatenate((z0_den, z0_rgb), axis=-1)
    logging.info("Rescaling factor den=%s rgb=%s", jnp.sqrt(1. - var_0_den),
                 jnp.sqrt(1. - var_0_rgb))
    return z0

def make_sample_fn(ldm, config, multi=True, rescale=True):
  """Return a function that generate an image."""

  # TODO(guandao) : need to do a single/multi GPU version of this
  def sample_fn(variables, rng, cond):
    # first generate latent
    logging.info("Initialize.")
    if multi:
      rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    rng, spl = jax.random.split(rng)
    zt = jax.random.normal(spl,
                           (*cond.shape, *ldm.latent_shape, ldm.embedding_dim))

    def body_fn(i, z_t):
      curr_rng = jax.random.fold_in(rng, i)
      return ldm.apply(
          variables,
          i=i,
          d=1,
          T=ldm.timesteps,
          z_t=z_t,
          cond=cond,
          rng=curr_rng,
          mutable=False,
          method=ldm.sample_step_i_d)[0]

    logging.info("Sampling.")
    z0 = jax.lax.fori_loop(
        lower=0, upper=ldm.timesteps, body_fun=body_fn, init_val=zt)

    logging.info("Rescaling final z0.")
    if rescale:
      z0 = ldm.apply(variables, z0, method=ldm.rescale_z0)
    z0 = ldm.denormalize_vox(z0)
    return z0

  return sample_fn
