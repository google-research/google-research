# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""All functions and modules related to model definition."""
from typing import Any

import flax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class State:
  step: int
  optimizer: flax.optim.Optimizer
  lr: float
  model_state: flax.deprecated.nn.Collection
  ema_rate: float
  params_ema: Any
  rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""
  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls
  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels from config files.

  Args:
    config: A ConfigDict object parsed from the config file

  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  if config.model.sigma_dist == 'geometric':
    sigmas = jnp.exp(
        jnp.linspace(
            jnp.log(config.model.sigma_begin), jnp.log(config.model.sigma_end),
            config.model.num_classes))
  elif config.model.sigma_dist == 'uniform':
    sigmas = jnp.linspace(config.model.sigma_begin, config.model.sigma_end,
                          config.model.num_classes)

  return sigmas


def get_ddpm_params():
  """Get betas and alphas --- parameters used in the original DDPM paper."""

  beta_schedule = 'linear'
  beta_start = 0.0001
  beta_end = 0.02
  num_diffusion_timesteps = 1000

  def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

  if beta_schedule == 'quad':
    betas = np.linspace(
        beta_start**0.5,
        beta_end**0.5,
        num_diffusion_timesteps,
        dtype=np.float64)**2
  elif beta_schedule == 'linear':
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(
        num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
      'betas': betas,
      'alphas': alphas,
      'alphas_cumprod': alphas_cumprod,
      'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
      'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod
  }
