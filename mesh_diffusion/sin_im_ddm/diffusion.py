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

"""Denoising diffusion timescale and sampling helper functions."""

import functools
import math
from typing import Any, Tuple
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnames=['dim'])
def timestep_embedding(t, dim, max_period = 10000):
  """Timestep embedding."""
  half = dim // 2
  freqs = jnp.exp(
      -1.0
      * jnp.log(max_period)
      * jnp.arange(half, dtype=jnp.float32)
      / (half - 1)
  )
  args = t * freqs
  embedding = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=0)
  return embedding


@functools.partial(jax.jit, static_argnames=['dim'])
def batched_timestep_embedding(t, dim, max_period = 10000):
  """Batched timestep embedding."""
  half = dim // 2
  freqs = jnp.exp(
      -1.0
      * jnp.log(max_period)
      * jnp.arange(half, dtype=jnp.float32)
      / (half - 1)
  )
  args = t[:, None] * freqs[None, :]
  embedding = jnp.concatenate((jnp.cos(args), jnp.sin(args)), axis=-1)
  return embedding


def linear_beta(t, beta_start = 1.0e-4, beta_end = 2.0e-2):
  """Linear beta."""
  beta = np.linspace(beta_start**0.5, beta_end**0.5, t)
  return beta * beta


def cosine_beta(t, s = 0.008):
  """Cosine beta."""
  timesteps = (np.arange(t + 1) / t) + s
  alpha = (timesteps / (1 + s)) * math.pi / 2.0
  alpha = jnp.cos(alpha)
  alpha = alpha * alpha
  alpha = alpha / alpha[0]
  beta = 1.0 - (alpha[1:] / alpha[:-1])
  beta = np.clip(beta, a_min=0, a_max=0.999)
  return beta


def exp_beta(t):
  """Exp beta."""
  timesteps = np.arange(t + 1).astype(np.float32) / t
  alpha_bar = np.exp(-10.0 * timesteps * timesteps - 1.0e-4)
  alpha = np.concatenate(
      (alpha_bar[0, None], alpha_bar[1:] / alpha_bar[:-1]), axis=0)
  beta = 1.0 - alpha
  return beta


def get_diffusion_parameters(t, schedule = 'linear'):
  """Get standard diffusion parameters with linear beta timescale."""
  if schedule == 'cos':
    beta = cosine_beta(t)
  elif schedule == 'exp':
    beta = exp_beta(t)
  else:
    beta = linear_beta(t)

  alpha = np.ones_like(beta) - beta
  alpha_bar = np.cumprod(alpha)
  alpha_bar_prev = np.concatenate((np.asarray([1.0]), alpha_bar[:-1]), axis=0)
  post_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
  ln_var = np.log(np.maximum(post_var, 1.0e-20))
  return (
      jnp.asarray(beta),
      jnp.asarray(alpha),
      jnp.asarray(alpha_bar),
      jnp.asarray(alpha_bar_prev),
      jnp.asarray(ln_var),
  )


def get_alphas(t, tau = 1, schedule = 'linear'):
  """Get only alpha_bar, and alpha_bar_prev from diffusion timescale."""
  if schedule == 'cos':
    beta = cosine_beta(t)
  elif schedule == 'exp':
    beta = exp_beta(t)
  else:
    beta = linear_beta(t)

  tau_step = t // tau
  tlist = np.arange(1, t + 1, tau_step)
  alpha = np.ones_like(beta) - beta
  alpha_bar = np.cumprod(alpha)
  alpha_bar = alpha_bar[tlist - 1]
  alpha_bar_prev = np.concatenate((np.asarray([1.0]), alpha_bar[:-1]), axis=0)
  return alpha_bar, alpha_bar_prev, tlist


def predict_z0(alpha_bar_t, zt, epst):
  """Predict z0."""
  alpha_bar_recip = 1.0 / (alpha_bar_t)
  z0 = jnp.sqrt(alpha_bar_recip) * zt - jnp.sqrt(alpha_bar_recip - 1.0) * epst
  return z0


def predict_mean(
    beta_t,
    alpha_t,
    alpha_bar_t,
    alpha_bar_prev_t,
    zt,
    epst):
  """Predict mean."""
  # Potentially clip this
  z0 = predict_z0(alpha_bar_t, zt, epst)

  mean_coeff_1 = (beta_t * jnp.sqrt(alpha_bar_prev_t)) / (1.0 - alpha_bar_t)
  mean_coeff_2 = (
      (1.0 - alpha_bar_prev_t) * jnp.sqrt(alpha_t) / (1.0 - alpha_bar_t)
  )

  post_mean = mean_coeff_1 * z0 + mean_coeff_2 * zt
  return post_mean


def predict_mean_from_z0(
    beta_t,
    alpha_t,
    alpha_bar_t,
    alpha_bar_prev_t,
    zt,
    z0):
  """Predict mean from z0."""
  mean_coeff_1 = (beta_t * jnp.sqrt(alpha_bar_prev_t)) / (1.0 - alpha_bar_t)
  mean_coeff_2 = (
      (1.0 - alpha_bar_prev_t) * jnp.sqrt(alpha_t) / (1.0 - alpha_bar_t))
  post_mean = mean_coeff_1 * z0 + mean_coeff_2 * zt
  return post_mean


@jax.jit
def predict_prev(
    beta_t,
    alpha_t,
    alpha_bar_t,
    alpha_bar_prev_t,
    ln_var_t,
    zt,
    epst,
    noise):
  """Predict previous."""
  post_mean = predict_mean(
      beta_t, alpha_t, alpha_bar_t, alpha_bar_prev_t, zt, epst)
  ztm1 = post_mean + jnp.exp(0.5 * ln_var_t) * noise
  return ztm1


@jax.jit
def predict_prev_from_z0(
    beta_t,
    alpha_t,
    alpha_bar_t,
    alpha_bar_prev_t,
    ln_var_t,
    zt,
    z0,
    noise):
  """Predict previous from z0."""
  post_mean = predict_mean_from_z0(
      beta_t, alpha_t, alpha_bar_t, alpha_bar_prev_t, zt, z0)
  ztm1 = post_mean + jnp.exp(0.5 * ln_var_t) * noise
  return ztm1


@jax.jit
def alpha_implicit_predict_prev_from_z0(
    alpha_t, alpha_tm1, zt, z0):
  """Alpha implicit prev prediction from z0."""
  epst = (zt - jnp.sqrt(alpha_t) * z0) / jnp.sqrt(1.0 - alpha_t)
  ztm1 = jnp.sqrt(alpha_tm1) * z0 + jnp.sqrt(1.0 - alpha_tm1) * epst
  return ztm1


@jax.jit
def alpha_predict_prev_from_z0(
    alpha_t,
    alpha_tm1,
    zt,
    z0,
    noise,
    eta = 1.0):
  """Alpha prev prediction from z0."""
  epst = (zt - jnp.sqrt(alpha_t) * z0) / jnp.sqrt(1.0 - alpha_t)
  sigma_t = (
      eta
      * jnp.sqrt((1.0 - alpha_tm1) / (1.0 - alpha_t))
      * jnp.sqrt(1.0 - (alpha_t / alpha_tm1))
  )

  ztm1 = (
      jnp.sqrt(alpha_tm1) * z0
      + jnp.sqrt(1.0 - alpha_tm1 - sigma_t * sigma_t) * epst
      + sigma_t * noise
  )

  return ztm1


def alpha_predict_mean_from_z0(
    alpha_t, alpha_tm1, zt, z0, eta = 1.0):
  """Alpha predict mean from z0."""
  epst = (zt - jnp.sqrt(alpha_t) * z0) / jnp.sqrt(1.0 - alpha_t)
  sigma_t = (
      eta
      * jnp.sqrt((1.0 - alpha_tm1) / (1.0 - alpha_t))
      * jnp.sqrt(1.0 - (alpha_t / alpha_tm1))
  )
  return (
      jnp.sqrt(alpha_tm1) * z0
      + jnp.sqrt(1.0 - alpha_tm1 - sigma_t * sigma_t) * epst,
      sigma_t * sigma_t,
  )


@functools.partial(jax.jit, static_argnames=['batch_size'])
def encode_batched_image_samples(
    z, alpha_bar, key, batch_size):
  """Encode batched image samples."""
  key, t_key, eps_key = jax.random.split(key, num=3)
  t = jax.random.randint(
      t_key, (batch_size,), minval=1, maxval=alpha_bar.shape[0] + 1
  )

  epst = jax.random.normal(eps_key, shape=(batch_size,) + z.shape)
  alpha = alpha_bar[t - 1]
  zt = (
      jnp.sqrt(alpha)[:, None, None, None] * z[None, Ellipsis]
      + jnp.sqrt(1.0 - alpha)[:, None, None, None] * epst
  )

  return zt, epst, t, key


def encode_batched_tangent_latent_samples(
    z, alpha_bar, key,
    sigma = 1.0):
  """Noised latent codes."""
  # Blends latent codes with noise based on diffusion timescale to create
  # training samples.
  z = jnp.concatenate((jnp.real(z)[Ellipsis, None], jnp.imag(z)[Ellipsis, None]), axis=-1)
  batch_size = z.shape[0]
  key, t_key, eps_key = jax.random.split(key, num=3)
  t = jax.random.randint(
      t_key, (batch_size,), minval=1, maxval=alpha_bar.shape[0] + 1
  )

  epst = sigma * jax.random.normal(eps_key, shape=z.shape)
  alpha = alpha_bar[t - 1]
  zt = (
      jnp.sqrt(alpha)[:, None, None, None] * z
      + jnp.sqrt(1.0 - alpha)[:, None, None, None] * epst
  )

  zt = jax.lax.complex(zt[Ellipsis, 0], zt[Ellipsis, 1])
  epst = jax.lax.complex(epst[Ellipsis, 0], epst[Ellipsis, 1])

  return zt, epst, t, key


def encode_inpaint_mask(
    z,
    z0,
    mask,
    alpha_bar,
    key,
    sigma = 1.0,
):
  """Noised latent codes."""
  # Blends latent codes with noise based on diffusion timescale to create
  # training samples.
  z0 = jnp.concatenate(
      (jnp.real(z0)[Ellipsis, None], jnp.imag(z0)[Ellipsis, None]), axis=-1
  )

  key, eps_key = jax.random.split(key, num=2)
  epst = sigma * jax.random.normal(eps_key, shape=z0.shape)
  zt = jnp.sqrt(alpha_bar) * z0 + jnp.sqrt(1.0 - alpha_bar) * epst
  z = (1.0 - mask)[Ellipsis, None, None] * z + mask[Ellipsis, None, None] * zt
  return z, key
