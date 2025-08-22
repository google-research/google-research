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
from nf_diffusion.models.score_model import ddpm_univdiff_3d
from nf_diffusion.models.score_model import vdm_score_unet_3d

VDMOutput = ldm_utils.VDMOutput


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

  # Score-based model configurations
  score_name: str
  score_config: ml_collections.ConfigDict
  gamma_name: str
  gamma_args: ml_collections.ConfigDict
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
    self.gamma = get_scheduler(self.gamma_name, self.gamma_args)
    self.embedding_vectors = nn.Embed(self.classes, self.cond_embedding_dim)
    self.score_model = get_score_model(self.score_name, self.score_config)

  def gammat(self, t):
    return self.gamma(t)

  def get_embedding_vectors(self, cond=None):
    if cond is not None:
      return self.embedding_vectors(cond)
    else:
      return None

  def get_score(self, z, t, g_t, cond, cond_v, train):
    # Some model expect condition input to be in a dictionary
    if self.score_name in ["ddpm_univdiff"]:
      if cond is not None:
        cond_v = {"class_id": cond}
      else:
        cond_v = {}

    if self.g_t_conditioned:
      if self.g_t_prenorm:
        gmin, gmax = self.gamma_args.gamma_min, self.gamma_args.gamma_max
        g_t = (g_t - gmin) / (gmax - gmin)
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

  def __call__(
      self,
      vox,
      cond=None,
      train = False,
  ):
    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    n_batch = vox.shape[0]
    f = vox
    f = self.normalize_vox(f)

    # if self.scale_factor is not None and self.shift_factor is not None:
    # nb = len(f.shape[:-1])
    # scale_factor = jnp.array(self.scale_factor).reshape(*([1] * nb), -1)
    # shift_factor = jnp.array(self.shift_factor).reshape(*([1] * nb), -1)
    # f = (f - shift_factor) / scale_factor
    # if cond is not None:
    # cond_v = self.embedding_vectors(cond)
    # else:
    # cond_v = None
    cond_v = self.get_embedding_vectors(cond)

    # 2. LATENT LOSS.
    # KL z1 with N(0,1) prior
    # Normally this is not needed as with enough T this should be 0
    # But we will learn 1) encoder and 2) noise scheduler
    # So the latent loss makes it important to regularize the latent space.
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
    var_t = nn.sigmoid(g_t)[:, None, None, None, None]
    eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    # compute predicted noise
    eps_hat = self.get_score(z_t, t, g_t, cond, cond_v, train)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=(-4, -3, -2, -1))

    # loss for finite depth T, i.e. discrete time
    s = t - (1. / ttl_t)
    g_s = self.gamma(s)
    loss_diff = .5 * ttl_t * jnp.expm1(g_t - g_s) * loss_diff_mse

    # End of diffusion loss computation
    out = VDMOutput(
        loss_recon=0,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=var_0,
        var_1=var_1,
    )
    return out

  def sample_step_i_d(self, i, d, T, z_t, cond, rng):
    """Sample a step [i] out total [ttl_t].

      NOTE: i, d are in the range of [0, max_steps].
      NOTE: i=0 means go from T_max to T_max - d
      NOTE: here, small i means small amount of denoising steps.
    """
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)
    t = (T - i) / T
    s = (T - i - d) / T

    g_s = self.gamma(s)
    g_t = self.gamma(t)

    cond_v = self.embedding_vectors(cond)
    t_inp = t * jnp.ones((z_t.shape[0],), z_t.dtype)
    g_t_inp = g_t * jnp.ones((z_t.shape[0],), z_t.dtype)
    eps_hat = self.get_score(z_t, t_inp, g_t_inp, cond, cond_v, False)

    a = nn.sigmoid(-g_s)
    c = -jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s = jnp.sqrt(nn.sigmoid(-g_s) /
                   nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) + jnp.sqrt(
                       (1. - a) * c) * eps
    return z_s, s

  def sample_step(self, i, T, z_t, cond, rng):
    """Sample a step [i] out total [ttl_t].

      NOTE: i=0 means go from T_max to T_max - d
      NOTE: here, small i means small amount of denoising steps.
    """
    return self.sample_step_i_d(i, 1, T, z_t, cond, rng)[0]

  def decode(self,
             z_0,
             rays,
             cond=None,
             rescale=True,
             rng=None,
             train=False,
             **kwargs):
    """Take a sampled z_0 and decode image out."""
    # TODO(guandao) colab has [rescale] == True
    if rescale:
      g_0 = self.gamma(0.)
      var_0 = nn.sigmoid(g_0)
      z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
    else:
      z_0_rescaled = z_0

    # Which images to decode needs camera poses
    # TODO(guandao) put the reference test-image pipeline here!
    raise NotImplementedError


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
          T=ldm.timesteps,
          z_t=z_t,
          cond=cond,
          rng=curr_rng,
          mutable=False,
          method=ldm.sample_step)

    logging.info("Sampling.")
    z0 = jax.lax.fori_loop(
        lower=0, upper=ldm.timesteps, body_fun=body_fn, init_val=zt)

    logging.info("Rescaling final z0.")
    if rescale:
      g_0 = ldm.apply(variables, 0., method=ldm.gammat)
      var_0 = nn.sigmoid(g_0)
      z0 = z0 / jnp.sqrt(1. - var_0)
    z0 = ldm.denormalize_vox(z0)
    # if ldm.scale_factor is not None and ldm.shift_factor is not None:
    # nb = len(z0.shape[:-1])
    # scale_factor = jnp.array(ldm.scale_factor).reshape(*([1] * nb), -1)
    # shift_factor = jnp.array(ldm.shift_factor).reshape(*([1] * nb), -1)
    # z0 = z0 / scale_factor + shift_factor

    # NOTE: the following code is TOO SLOW - I will change to return the voxel
    #       only and then do the rendering outside.
    # # Test-reference image pipeline here, will use multi GPU to do inference.
    # logging.info("Rendering z0")
    # all_frames = []
    # for i in range(z0.shape[0]):
    # logging.info("Render scene [%02d/%02d]", i, z0.shape[0])
    # vox = z0[i]
    # render_loop = instant_ngp_utils.make_render_loop(
    # vox, render_config, multi=True)
    # frames = []
    # for pi, pose in enumerate(poses):
    # logging.info("scene [%02d/%02d] pose [%03d/%03d]", i, z0.shape[0], pi,
    # len(poses))
    # spl = jax.random.fold_in(spl, pi)
    # frames.append(
    # render_loop(instant_ngp_utils.camera_ray_batch(pose, hwf), spl)[0])
    # frames = jnp.concatenate([f[None, ...] for f in frames], axis=0)
    # all_frames.append(frames[None, ...])
    # # Return: (#batch, #imgs, #res_img, #res_img, #c_img)
    # frames = jnp.concatenate(all_frames, axis=0)
    # return frames
    return z0

  return sample_fn


# TODO(guandao): still need to finalize the few-view recon pipeline.
def make_recon_fn(ldm):
  raise NotImplementedError
