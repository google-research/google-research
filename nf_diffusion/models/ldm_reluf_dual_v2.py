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
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from nf_diffusion.models.layers import ldm_utils
from nf_diffusion.models.score_model import ddpm_univdiff_3d
from nf_diffusion.models.score_model import vdm_score_unet_3d


def get_scheduler(name, config):
  """Helper function to get scheduler model."""
  if name == "fixed_linear":
    gamma = ldm_utils.NoiseSchedule_FixedLinear(**config)
  else:
    raise ValueError
  return gamma


def get_score_model(name, config):
  """Helper function to get score model."""
  if name == "vdm_unet":
    score_model = vdm_score_unet_3d.ScoreUNet(**config)
  elif name == "ddpm_univdiff":
    score_model = ddpm_univdiff_3d.UNet3D(config=config)
  else:
    raise ValueError
  return score_model


class LDM(nn.Module):
  """LDM Model."""

  # Schedule
  gamma_args_den: ml_collections.ConfigDict
  gamma_name_rgb: str
  gamma_args_rgb: ml_collections.ConfigDict

  # Score-based model configurations
  score_name_den: str
  score_config_den: ml_collections.ConfigDict
  score_name_rgb: str
  score_config_rgb: ml_collections.ConfigDict
  gamma_name_den: str

  # Conditioning for RGB Unet
  stop_den_grad: bool = False
  color_conditioning: Sequence[str] = ("output",)
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

  # Now, here are some options to evaluate only on some filtered part of RGB
  visibility_type: Optional[str] = None
  visibility_thr: Optional[float] = None
  visibility_nograd: bool = True

  # Only sampling a small amount of time steps
  # This assume time scale of [0, 1]
  sample_t_min: float = 0.0
  sample_t_max: float = 1.0

  def setup(self):
    self.gamma_den = get_scheduler(self.gamma_name_den, self.gamma_args_den)
    self.gamma_rgb = get_scheduler(self.gamma_name_rgb, self.gamma_args_rgb)
    self.embedding_vectors = nn.Embed(self.classes, self.cond_embedding_dim)
    self.score_model_den = get_score_model(
        self.score_name_den, self.score_config_den
    )
    self.score_model_rgb = get_score_model(
        self.score_name_rgb, self.score_config_rgb
    )

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

  def compute_visibility_binary(self, density_gtr):
    visibility = jnp.ones(density_gtr.shape)
    if self.visibility_type is not None:
      if self.visibility_type == "thr_gtr":
        assert self.visibility_thr is not None
        visibility = (density_gtr >= self.visibility_thr).astype(jnp.float32)
      else:
        print("Visibility type %s not implemented" % self.visibility_type)
        raise NotImplementedError
    if self.visibility_nograd:
      visibility = jax.lax.stop_gradient(visibility)
    return visibility

  def get_embedding_vectors(self, cond=None):
    if cond is not None:
      return self.embedding_vectors(cond)
    else:
      return None

  def get_clean_density(self, z_den, t, eps_hat_den):
    g_t_den = self.gammat(t)[0]
    var_t_den = jax.nn.sigmoid(g_t_den)[:, None, None, None, None]
    clean_den = (z_den - jnp.sqrt(var_t_den) * eps_hat_den) / jnp.sqrt(
        1.0 - var_t_den
    )
    return clean_den

  def get_clean_rgb(self, z_rgb, t, eps_hat_rgb):
    g_t_rgb = self.gammat(t)[1]
    var_t_rgb = jax.nn.sigmoid(g_t_rgb)[:, None, None, None, None]
    clean_rgb = (z_rgb - jnp.sqrt(var_t_rgb) * eps_hat_rgb) / jnp.sqrt(
        1.0 - var_t_rgb
    )
    return clean_rgb

  def get_clean_vox(self, vox, t, eps_hat):
    return jnp.concatenate(
        [
            self.get_clean_density(vox[Ellipsis, :1], t, eps_hat[Ellipsis, :1]),
            self.get_clean_rgb(vox[Ellipsis, 1:], t, eps_hat[Ellipsis, 1:]),
        ],
        axis=-1,
    )

  def denoise(self, z_t, t, cond):
    cond_v = self.embedding_vectors(cond)
    g_t_den, g_t_rgb = self.gammat(t)
    eps_hat = self.get_score(z_t, t, (g_t_den, g_t_rgb), cond, cond_v, False)
    return self.get_clean_vox(z_t, t, eps_hat)

  def get_score(self, z, t, g_t, cond, cond_v, train):
    # Some model expect condition input to be in a dictionary
    # NOTE: only work with DDPM UNet from now on.
    # NOTE: DDPM model assume cond to be integer, so we cannot use condv
    if cond is not None:
      cond_v = {"class_id": cond}
    else:
      cond_v = {}

    if self.g_t_conditioned:
      if self.g_t_prenorm:
        g_t = self.normalize_gt(g_t)
      t_inp_den, t_inp_rgb = g_t
    else:
      t_inp_den, t_inp_rgb = t

    # Get density
    z_den = z[Ellipsis, :1]
    eps_hat_den = self.score_model_den(z_den, t_inp_den, cond_v, train)

    # NOTE(guandao) : DO NOT CHANGE THE ORDER!
    all_den_cond = []
    clean_den = None
    for cond_type in self.color_conditioning:
      if "denoised" == cond_type:
        # logging.info("Add denoised density to conditioning time")
        # print("Add denoised density to conditioning time")
        # g_t_den = self.gammat(t)[0]
        # var_t_den = jax.nn.sigmoid(g_t_den)[:, None, None, None, None]
        # clean_den = (z_den - jnp.sqrt(var_t_den) *
        # eps_hat_den) / jnp.sqrt(1. - var_t_den)
        if clean_den is None:
          clean_den = self.get_clean_density(z_den, t, eps_hat_den)
        all_den_cond.append(clean_den)
        continue
      if "visibility" == cond_type:
        if clean_den is None:
          clean_den = self.get_clean_density(z_den, t, eps_hat_den)
        vis = self.compute_visibility_binary(clean_den).astype(jnp.float32)
        all_den_cond.append(vis)
        continue
      if "input" == cond_type:
        # logging.info("Add density input to conditioning time")
        # print("Add density input to conditioning time")
        all_den_cond.append(z_den)
        continue
      if "output" == cond_type:
        # logging.info("Add density output to conditioning time")
        # print("Add density output to conditioning time")
        all_den_cond.append(eps_hat_den)
        continue
      raise ValueError

    z_rgb = z[Ellipsis, 1:]
    if all_den_cond:
      density_out_cond = jnp.concatenate(all_den_cond, axis=-1)
      if self.stop_den_grad:
        density_out_cond = jax.lax.stop_gradient(density_out_cond)
      colors_input = jnp.concatenate([density_out_cond, z_rgb], axis=-1)
    else:
      colors_input = z_rgb
    eps_hat_rgb = self.score_model_rgb(colors_input, t_inp_rgb, cond_v, train)

    eps_hat = jnp.concatenate([eps_hat_den, eps_hat_rgb], axis=-1)
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
        lambda x: nn.sigmoid(x)[:, None, None, None, None], g_t
    )
    eps_den = jax.random.normal(self.make_rng("sample"), shape=f_den.shape)
    z_t_den = jnp.sqrt(1.0 - var_t_den) * f_den + jnp.sqrt(var_t_den) * eps_den

    eps_rgb = jax.random.normal(self.make_rng("sample"), shape=f_rgb.shape)
    z_t_rgb = jnp.sqrt(1.0 - var_t_rgb) * f_rgb + jnp.sqrt(var_t_rgb) * eps_rgb

    z_t = jnp.concatenate([z_t_den, z_t_rgb], axis=-1)
    if return_eps:
      return z_t, (eps_den, eps_rgb)
    else:
      return z_t

  def compute_latent_loss(self, f, var_1):
    mean1_sqr = (1.0 - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(
        mean1_sqr + var_1 - jnp.log(var_1) - 1.0, axis=(-4, -3, -2, -1)
    )
    return loss_klz

  def __call__(
      self,
      vox,
      cond=None,
      train = False,
  ):
    g_0, g_1 = self.gammat(0.0), self.gammat(1.0)
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
      t0 = t0 * (self.sample_t_max - self.sample_t_min) + self.sample_t_min
      t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / n_batch), 1.0)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))
      t = t * (self.sample_t_max - self.sample_t_min) + self.sample_t_min

    # discretize time steps if we're working with discrete time
    ttl_t = self.timesteps
    t = jnp.ceil(t * ttl_t) / ttl_t

    # compute predicted noise
    g_t_den, g_t_rgb = self.gammat(t)
    z_t, (eps_den, eps_rgb) = self.diffuse(f, t, return_eps=True)
    eps_hat = self.get_score(z_t, t, (g_t_den, g_t_rgb), cond, cond_v, train)
    eps_hat_den = eps_hat[Ellipsis, :1]
    eps_hat_rgb = eps_hat[Ellipsis, 1:]
    # compute MSE of predicted noise
    loss_diff_mse_den = jnp.sum(
        jnp.square(eps_den - eps_hat_den), axis=(-4, -3, -2, -1)
    )
    loss_diff_mse_rgb = jnp.sum(
        jnp.square(eps_rgb - eps_hat_rgb), axis=(-4, -3, -2, -1)
    )

    # loss for finite depth T, i.e. discrete time
    s = t - (1.0 / ttl_t)
    g_s_den, g_s_rgb = self.gammat(s)
    # TODO(guandao) continuous version of the objective
    loss_diff_den = (
        0.5 * ttl_t * jnp.expm1(g_t_den - g_s_den) * loss_diff_mse_den
    )
    loss_diff_rgb = (
        0.5 * ttl_t * jnp.expm1(g_t_rgb - g_s_rgb) * loss_diff_mse_rgb
    )
    loss_diff = loss_diff_den + loss_diff_rgb

    # Compute loss with visibility
    visibility_mask = self.compute_visibility_binary(f_den)
    loss_diff_mse_den_vis = jnp.sum(
        jnp.square(eps_den - eps_hat_den) * visibility_mask,
        axis=(-4, -3, -2, -1),
    )
    loss_diff_den_vis = (
        0.5 * ttl_t * jnp.expm1(g_t_den - g_s_den) * loss_diff_mse_den_vis
    )
    loss_diff_mse_rgb_vis = jnp.sum(
        jnp.square(eps_rgb - eps_hat_rgb) * visibility_mask,
        axis=(-4, -3, -2, -1),
    )
    loss_diff_rgb_vis = (
        0.5 * ttl_t * jnp.expm1(g_t_rgb - g_s_rgb) * loss_diff_mse_rgb_vis
    )
    loss_diff_vis = loss_diff_den_vis + loss_diff_rgb_vis

    # TODO(guandao) group losses accodring to time

    # End of diffusion loss computation
    return {
        "loss_klz_den": loss_klz_den,
        "loss_klz_rgb": loss_klz_rgb,
        "loss_klz": loss_klz,
        "loss_diff": loss_diff,
        "loss_diff_den": loss_diff_den,
        "loss_diff_rgb": loss_diff_rgb,
        "loss_diff_vis": loss_diff_vis,
        "loss_diff_den_vis": loss_diff_den_vis,
        "loss_diff_rgb_vis": loss_diff_rgb_vis,
        "var_0_den": var_0_den,
        "var_0_rgb": var_0_rgb,
        "var_1_den": var_1_den,
        "var_1_rgb": var_1_rgb,
    }

  def sample_step_i_d(self, i, d, T, z_t, cond, rng):  # pylint: disable=invalid-name
    """Sample a step [i] out total [ttl_t]."""

    # NOTE: i, d are in the range of [0, max_steps].
    # NOTE: i=0 means go from T_max to T_max - d
    # NOTE: here, small i means small amount of denoising steps.

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
    eps_hat = self.get_score(
        z_t, t_inp, (g_t_den_inp, g_t_rgb_inp), cond, cond_v, False
    )

    def _step_(z_t, eps_hat, eps, g_t, g_s):
      a = nn.sigmoid(-g_s)
      c = -jnp.expm1(g_s - g_t)
      sigma_t = jnp.sqrt(nn.sigmoid(g_t))
      z_s = (
          jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t))
          * (z_t - sigma_t * c * eps_hat)
          + jnp.sqrt((1.0 - a) * c) * eps
      )
      return z_s

    eps_hat_den, eps_hat_rgb = eps_hat[Ellipsis, :1], eps_hat[Ellipsis, 1:]
    z_t_den, z_t_rgb = z_t[Ellipsis, :1], z_t[Ellipsis, 1:]
    z_s_den = _step_(z_t_den, eps_hat_den, eps_den, g_t_den, g_s_den)
    z_s_rgb = _step_(z_t_rgb, eps_hat_rgb, eps_rgb, g_t_rgb, g_s_rgb)
    z_s = jnp.concatenate((z_s_den, z_s_rgb), axis=-1)
    return z_s, T - i - d

  def rescale_z0(self, z0):
    logging.info("Does it get here at all?")
    g_0_den, g_0_rgb = self.gammat(jnp.zeros((1,), jnp.float32))
    logging.info("Where does it get stucked??")
    var_0_den = nn.sigmoid(g_0_den)
    var_0_rgb = nn.sigmoid(g_0_rgb)
    z0_den, z0_rgb = z0[Ellipsis, :1], z0[Ellipsis, 1:]
    z0_den = z0_den / jnp.sqrt(1.0 - var_0_den)
    z0_rgb = z0_rgb / jnp.sqrt(1.0 - var_0_rgb)
    z0 = jnp.concatenate((z0_den, z0_rgb), axis=-1)
    logging.info(
        "Rescaling factor den=%s rgb=%s",
        jnp.sqrt(1.0 - var_0_den),
        jnp.sqrt(1.0 - var_0_rgb),
    )
    return z0

  def get_mu_s_t(self, z_t, xhat, s, t):
    """Compute mu_{s|t} from z_t and xhat.

    Args:
      z_t: (..., r, r, r, d)
      xhat: (..., r, r, r, d)
      s: should be [0, t), 0=clean, 1=noisy, shape (...,)
      t: should be [0, 1], 0=clean, 1=noisy, shape (...,)

    Returns:
      mu_{s|t} (1, r, r, r, d)
    """
    gamma_t = self.gammat(t)
    gamma_s = self.gammat(s)

    def get_mu(zt_this, x_this, g_s, g_t):
      c = jnp.exp(g_s - g_t)
      alpha_s = jnp.sqrt(jax.nn.sigmoid(-g_s))
      alpha_t = jnp.sqrt(jax.nn.sigmoid(-g_t))
      return c * alpha_s / alpha_t * zt_this + (1 - c) * alpha_s * x_this

    return jnp.concatenate(
        [
            get_mu(z_t[Ellipsis, :1], xhat[Ellipsis, :1], gamma_s[0], gamma_t[0]),
            get_mu(z_t[Ellipsis, 1:], xhat[Ellipsis, 1:], gamma_s[1], gamma_t[1]),
        ],
        axis=-1,
    )

  def get_zs_from_mu_st(self, mu, s, t, rng, gamma=0.0):
    """Sample z_s given mu_{s|t}."""
    gamma_t = self.gammat(t)
    gamma_s = self.gammat(s)

    def get_combine_sigma_t_s(gamma_t, gamma_s):
      c = jnp.exp(gamma_s - gamma_t)
      sigma_t_2 = jax.nn.sigmoid(gamma_t)
      sigma_s_2 = jax.nn.sigmoid(gamma_s)
      sigma_t_s_2 = (1 - c) * sigma_t_2
      tl_sigma_t_s_2 = (1 - c) * sigma_s_2
      return jnp.sqrt(sigma_t_s_2**gamma * tl_sigma_t_s_2 ** (1 - gamma))

    combine_sigma_t_s = jax.tree_util.tree_map(
        get_combine_sigma_t_s, gamma_t, gamma_s
    )
    eps = jax.random.normal(rng, mu.shape)
    return jnp.concatenate(
        [
            mu[Ellipsis, :1] + combine_sigma_t_s[0] * eps[Ellipsis, :1],
            mu[Ellipsis, 1:] + combine_sigma_t_s[1] * eps[Ellipsis, 1:],
        ],
        axis=-1,
    )

  def get_zs_from_xhat(self, z_t, xhat, s, t, rng, gamma=0.0):
    """Sample z_s given xhat."""
    mu_st = self.get_mu_s_t(z_t, xhat, s, t)
    return self.get_zs_from_mu_st(mu_st, s, t, rng, gamma=gamma)

  def get_eps_from_zt_xt(self, z_t, x_t, t):
    gamma_t = self.gammat(t)

    def helper(zt, xt, gt):
      sigma_t = jnp.sqrt(jax.nn.sigmoid(gt))
      alpha_t = jnp.sqrt(jax.nn.sigmoid(-gt))
      return (zt - xt * alpha_t) / sigma_t

    return jnp.concatenate(
        [
            helper(z_t[Ellipsis, :1], x_t[Ellipsis, :1], gamma_t[0]),
            helper(z_t[Ellipsis, 1:], x_t[Ellipsis, 1:], gamma_t[1]),
        ],
        axis=-1,
    )

  def get_zs_langevin_correction(self, z_s, s, x_hat, rng, delta=0.1):
    g_s = self.gammat(s)
    eps_hat = self.get_eps_from_zt_xt(z_s, x_hat, s)

    def f(spl, zs, epsh, gs):
      sigma_s = jnp.sqrt(jax.nn.sigmoid(gs))
      eps = jax.random.normal(spl, epsh.shape)
      return zs - 0.5 * delta * sigma_s * epsh + jnp.sqrt(delta) * sigma_s * eps

    spl1, spl2 = jax.random.split(rng)
    return jnp.concatenate(
        [
            f(spl1, z_s[Ellipsis, :1], eps_hat[Ellipsis, :1], g_s[0]),
            f(spl2, z_s[Ellipsis, 1:], eps_hat[Ellipsis, 1:], g_s[1]),
        ],
        axis=-1,
    )


def make_sample_fn(ldm, unused_config, multi=True, rescale=True, verbose=False):
  """Return a function that generate an image."""

  # TODO(guandao) : need to do a single/multi GPU version of this
  def sample_fn(variables, rng, cond):
    # first generate latent
    logging.info("Initialize.")
    if multi:
      rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
    rng, spl = jax.random.split(rng)
    zt = jax.random.normal(
        spl, (*cond.shape, *ldm.latent_shape, ldm.embedding_dim)
    )

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
          method=ldm.sample_step_i_d,
      )[0]

    logging.info("Sampling.")
    if verbose:
      body_fn_jit = jax.jit(body_fn)
      for i in range(ldm.timesteps):
        logging.info("z0 iters %d %d", int(i), int(ldm.timesteps))
        zt = body_fn_jit(i, zt)
      z0 = zt
    else:
      z0 = jax.lax.fori_loop(
          lower=0, upper=ldm.timesteps, body_fun=body_fn, init_val=zt
      )
    logging.info("Rescaling final z0 %s", z0.shape)
    if rescale:
      z0 = ldm.apply(variables, z0, method=ldm.rescale_z0)
    logging.info("Denoramlize final z0.")
    z0 = ldm.denormalize_vox(z0)
    logging.info("Fainl s0 done: %s", z0.shape)
    return z0

  return sample_fn


def make_sample_fn_even_d(
    ldm,
    unused_config,
    d = 5,
    unused_multi=True,
    rescale=True,
    unused_verbose=False,
):
  """Return a function that generate an image."""

  def sample_fn(variables, rng, cond):
    """Sample function will loop through the body func with pmap."""
    # This function will generate #devices number of voxels.
    logging.info("Initialize.")
    rng, spl = jax.random.split(rng)
    zt = jax.random.normal(
        spl,
        (
            jax.local_device_count(),
            *cond.shape,
            *ldm.latent_shape,
            ldm.embedding_dim,
        ),
    )

    def body_fn(i, z_t):
      curr_rng = jax.random.fold_in(rng, i)
      curr_rng = jax.random.fold_in(curr_rng, jax.lax.axis_index("batch"))
      return ldm.apply(
          variables,
          i=i,
          d=d,
          T=ldm.timesteps,
          z_t=z_t,
          cond=cond,
          rng=curr_rng,
          mutable=False,
          method=ldm.sample_step_i_d,
      )[0]

    p_body_fn = jax.pmap(body_fn, axis_name="batch")

    logging.info("Sampling.")
    for i in range(0, ldm.timesteps, d):
      logging.info("z0 iters %d %d", int(i), int(ldm.timesteps))
      zt = p_body_fn(flax_utils.replicate(i), zt)
    z0 = zt
    logging.info("Rescaling final z0 %s", z0.shape)
    if rescale:
      z0 = ldm.apply(variables, z0, method=ldm.rescale_z0)

    logging.info("Denoramlize final z0.")
    z0 = ldm.denormalize_vox(z0)

    logging.info("Fainl s0 done: %s", z0.shape)
    return z0

  return sample_fn
