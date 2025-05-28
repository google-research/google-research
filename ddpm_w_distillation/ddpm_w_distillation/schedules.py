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

"""LogSNR schedules (t==0 => logsnr_max, t==1 => logsnr_min)."""

import functools
import jax.numpy as jnp
import numpy as onp


def _logsnr_schedule_uniform(t, *, logsnr_min, logsnr_max):
  return logsnr_min * t + logsnr_max * (1. - t)


def _onp_softplus(x):
  return onp.logaddexp(x, 0)


def _logsnr_schedule_beta_const(t, *, logsnr_min, logsnr_max):
  b = _onp_softplus(-logsnr_max)
  a = _onp_softplus(-logsnr_min) - b
  return -jnp.log(jnp.expm1(a * t + b))


def _logsnr_schedule_beta_linear(t, *, logsnr_min, logsnr_max):
  b = _onp_softplus(-logsnr_max)
  a = _onp_softplus(-logsnr_min) - b
  return -jnp.log(jnp.expm1(a * t**2 + b))


def _logsnr_schedule_beta_interpolated(t, *, betas):
  betas = onp.asarray(betas, dtype=onp.float64)
  assert betas.ndim == 1
  alphas = 1. - betas
  alphas_cumprod = onp.cumprod(alphas, axis=0)
  logsnr = onp.log(alphas_cumprod) - onp.log1p(-alphas_cumprod)
  return jnp.interp(t, onp.linspace(0, 1, len(betas)), logsnr)


def _logsnr_schedule_cosine(t, *, logsnr_min, logsnr_max):
  b = onp.arctan(onp.exp(-0.5 * logsnr_max))
  a = onp.arctan(onp.exp(-0.5 * logsnr_min)) - b
  return -2. * jnp.log(jnp.tan(a * t + b))


def _logsnr_schedule_iddpm_cosine_interpolated(t, *, num_timesteps):
  steps = onp.arange(num_timesteps + 1, dtype=onp.float64) / num_timesteps
  alpha_bar = onp.cos((steps + 0.008) / 1.008 * onp.pi / 2) ** 2
  betas = onp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
  return _logsnr_schedule_beta_interpolated(t, betas=betas)


def _logsnr_schedule_iddpm_cosine_respaced(t, *, num_timesteps,
                                           num_respaced_timesteps):
  """Improved DDPM respaced discrete time cosine schedule."""
  # original schedule
  steps = onp.arange(num_timesteps + 1, dtype=onp.float64) / num_timesteps
  alpha_bar = onp.cos((steps + 0.008) / 1.008 * onp.pi / 2)**2
  betas = onp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)

  # respace the schedule
  respaced_inds = onp.round(
      onp.linspace(0, 1, num_respaced_timesteps) *
      (num_timesteps - 1)).astype(int)
  alpha_bar = onp.cumprod(1. - betas)[respaced_inds]
  assert alpha_bar.shape == (num_respaced_timesteps,)
  logsnr = onp.log(alpha_bar) - onp.log1p(-alpha_bar)
  return jnp.interp(t, onp.linspace(0, 1, len(logsnr)), logsnr)


def get_logsnr_schedule(name, **kwargs):
  """Get log SNR schedule (t==0 => logsnr_max, t==1 => logsnr_min)."""
  schedules = {
      'uniform': _logsnr_schedule_uniform,
      'beta_const': _logsnr_schedule_beta_const,
      'beta_linear': _logsnr_schedule_beta_linear,
      'beta_interp': _logsnr_schedule_beta_interpolated,
      'cosine': _logsnr_schedule_cosine,
      'iddpm_cosine_interp': _logsnr_schedule_iddpm_cosine_interpolated,
      'iddpm_cosine_respaced': _logsnr_schedule_iddpm_cosine_respaced,
  }
  return functools.partial(schedules[name], **kwargs)
