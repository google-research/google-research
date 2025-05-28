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

"""Utils for score_prior."""
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree  # pylint:disable=g-multiple-import
import ml_collections
import numpy as np
from PIL import Image

from score_sde import sde_lib
# Keep the import below for registering models for `initialize_training_state`.
from score_sde.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import
from score_sde.models import utils as mutils

from score_prior.score_sde import losses


def get_sde(config
            ):
  """Return the SDE and time-0 epsilon based on the given config."""
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(
        beta_min=config.model.beta_min, beta_max=config.model.beta_max,
        N=config.model.num_scales)
    t0_eps = 1e-3  # epsilon for stability near time 0
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(
        beta_min=config.model.beta_min, beta_max=config.model.beta_max,
        N=config.model.num_scales)
    t0_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(
        sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
        N=config.model.num_scales)
    t0_eps = 1e-5
  else:
    raise NotImplementedError(f'SDE {config.training.sde} unknown.')
  return sde, t0_eps


def get_marginal_dist_fn(config
                         ):
  """Return a function that gives the scale and std. dev. of $p_0t$.

  See https://github.com/yang-song/score_sde/blob/main/sde_lib.py.
  `alpha_t` and `beta_t` are determined by the method
  `score_sde.sde_lib.SDE.marginal_prob`, where `alpha_t` is the coefficient of
  the mean, and `beta_t` is the std. dev.

  Args:
    config: An ml_collections.ConfigDict with the SDE configuration.

  Returns:
    _marginal_dist_fn: A callable that returns the mean coefficient `alpha_t`
      and std. dev. `beta_t` for a given diffusion time `t`.
  """
  if config.training.sde.lower() == 'vpsde':
    beta_0, beta_1 = config.model.beta_min, config.model.beta_max
    def _marginal_dist_fn(t):
      log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
      alpha_t = jnp.exp(log_mean_coeff)
      beta_t = jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff))
      return alpha_t, beta_t

  elif config.training.sde.lower() == 'vesde':
    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
    def _marginal_dist_fn(t):
      alpha_t = jnp.ones_like(t)
      beta_t = sigma_min * (sigma_max / sigma_min) ** t
      return alpha_t, beta_t

  elif config.training.sde.lower() == 'subvpsde':
    beta_0, beta_1 = config.model.beta_min, config.model.beta_max
    def _marginal_dist_fn(t):
      log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
      alpha_t = jnp.exp(log_mean_coeff)
      beta_t = 1 - jnp.exp(2. * log_mean_coeff)
      return alpha_t, beta_t
  else:
    raise NotImplementedError(f'Unsupported SDE: {config.training.sde}')

  return _marginal_dist_fn


def initialize_training_state(config
                              ):
  """Initialize training state, score model, and optimizer.

  Args:
    config: An ml_collections ConfigDict specifying the model and training
      parameters.

  Returns:
    state: The initial training state, a `PyTree`.
    score_model: The score model, a `flax.linen.Module`.
    tx: The optimizer, an `optax.GradientTransformation`.
  """
  rng = jax.random.PRNGKey(config.seed)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, init_params = mutils.init_model(
      step_rng, config)

  # Initialize optimizer.
  tx = losses.get_optimizer(config)
  opt_state = tx.init(init_params)

  # Construct initial state.
  state = mutils.State(
      step=0,
      model_state=init_model_state,
      opt_state=opt_state,
      ema_rate=config.model.ema_rate,
      params=init_params,
      params_ema=init_params,
      rng=rng)

  return state, score_model, tx


def psplit(
    rng
):
  """Split a JAX RNG into pmapped RNGs."""
  rng, *step_rngs = jax.random.split(rng, jax.local_device_count() + 1)
  step_rngs = jnp.asarray(step_rngs)
  return rng, step_rngs


def gaussian_logp(x, mu, sigma):
  """Evaluates the log-probability of x under N(mu, sigma**2)."""
  dim = x.size
  return (-dim / 2. * jnp.log(2 * jnp.pi * sigma**2) - jnp.sum((x - mu)**2) /
          (2 * sigma**2))


def save_image_grid(ndarray,
                    fp,
                    nrow = 8,
                    padding = 2,
                    image_format = None):
  """Make a grid of images and save it into an image file.

  This implementation is modified from the one in
  https://github.com/yang-song/score_sde/blob/main/utils.py.

  Pixel values are assumed to be within [0, 1].

  Args:
    ndarray: 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow: Number of images displayed in each row of the grid.
      The final grid size is ``(nrow, B // nrow)``.
    padding: Amount of zero-padding on each image.
    image_format:  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, (jnp.ndarray, np.ndarray)) or
          (isinstance(ndarray, list) and
           all(isinstance(t, (jnp.ndarray, np.ndarray)) for t in ndarray))):
    raise TypeError('array_like of tensors expected, got {}'.format(
        type(ndarray)))
  ndarray = np.asarray(ndarray)

  # Keep largest-possible number of images for given `nrow`.
  ncol = len(ndarray) // nrow
  ndarray = ndarray[:nrow * ncol]

  def _pad(image):
    # Pads a 3D array in the height and width dimensions.
    return np.pad(image, ((padding, padding), (padding, padding), (0, 0)))

  grid = np.concatenate([
      np.concatenate([
          _pad(im) for im in ndarray[row * ncol:(row + 1) * ncol]], axis=1)
      for row in range(nrow)], axis=0)

  # For grayscale images, need to remove the third axis.
  grid = np.squeeze(grid)

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer.
  ndarr = np.clip(grid * 255. + 0.5, 0, 255).astype(np.uint8)

  im = Image.fromarray(ndarr)
  im.save(fp, format=image_format)


def is_coordinator():
  return jax.process_index() == 0
