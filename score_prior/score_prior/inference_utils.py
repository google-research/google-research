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

"""Utility functions for posterior inference with score-based prior."""
from typing import Callable, Tuple

import diffrax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint:disable=g-multiple-import
import ml_collections
import numpy as np
from score_sde.models import utils as mutils
from tensorflow.io import gfile
import tensorflow_probability.substrates.jax as tfp

from score_prior import datasets
from score_prior import forward_models
from score_prior import probability_flow
from score_prior import utils

tfd = tfp.distributions


def get_score_fn(config,
                 score_model_config
                 ):
  """Return score function for a given model checkpoint."""
  score_model_config.data.num_channels = config.data.num_channels
  # Initialize score model.
  state, score_model, _ = utils.initialize_training_state(
      score_model_config)

  if gfile.isdir(config.prob_flow.score_model_dir):
    # Try to find the latest checkpoint.
    ckpt_path = checkpoints.latest_checkpoint(config.prob_flow.score_model_dir)
  else:
    ckpt_path = config.prob_flow.score_model_dir

  if ckpt_path is None or not gfile.exists(ckpt_path):
    raise FileNotFoundError(
        'No pretrained model found in %s' % config.prob_flow.score_model_dir)

  # Load checkpoint.
  state = checkpoints.restore_checkpoint(ckpt_path, state)

  # Get SDE.
  sde, _ = utils.get_sde(score_model_config)
  # Get score function.
  score_fn = mutils.get_score_fn(
      sde,
      score_model,
      state.params_ema,
      state.model_state,
      train=False,
      continuous=True)
  return score_fn


def _get_solver(solver):
  """Return `diffrax.AbstractSolver` instance."""
  return getattr(diffrax, solver)()


def _get_stepsize_controller(stepsize_controller,
                             rtol,
                             atol,
                             adjoint_rms_seminorm = False
                             ):
  """Return `diffrax.AbstractStepSizeController` instance."""
  if stepsize_controller == 'ConstantStepSize':
    return diffrax.ConstantStepSize(compile_steps=True)
  elif stepsize_controller == 'PIDController':
    if adjoint_rms_seminorm:
      return diffrax.PIDController(
          norm=diffrax.adjoint_rms_seminorm, rtol=rtol, atol=atol)
    else:
      return diffrax.PIDController(rtol=rtol, atol=atol)
  else:
    raise ValueError(f'Unsupported stepsize controller: {stepsize_controller}')


def get_prob_flow(config,
                  score_model_config
                  ):
  """Return `ProbabilityFlow` module.

  Args:
    config: Config for inference setup (e.g., DPI, grad ascent). Includes
      parameters for score-model checkpoint, dataset, etc.
    score_model_config: Config for score model. Includes parameters for
      score-model architecture, SDE, etc.

  Returns:
    A `probability_flow.ProbabilityFlow` instance.
  """
  # Get SDE.
  sde, _ = utils.get_sde(score_model_config)
  # Get score function.
  score_fn = get_score_fn(config, score_model_config)

  # ODE solver and step-size controller.
  solver = _get_solver(config.prob_flow.solver)
  stepsize_controller = _get_stepsize_controller(
      config.prob_flow.stepsize_controller,
      config.prob_flow.rtol,
      config.prob_flow.atol)

  # Adjoint solver and step-size controller.
  if config.prob_flow.adjoint_method == 'RecursiveCheckpointAdjoint':
    adjoint = diffrax.RecursiveCheckpointAdjoint()
  elif config.prob_flow.adjoint_method == 'BacksolveAdjoint':
    adjoint_solver = _get_solver(config.prob_flow.adjoint_solver)
    adjoint_stepsize_controller = _get_stepsize_controller(
        config.prob_flow.adjoint_stepsize_controller,
        config.prob_flow.adjoint_rtol,
        config.prob_flow.adjoint_atol,
        config.prob_flow.adjoint_rms_seminorm)
    adjoint = diffrax.BacksolveAdjoint(
        solver=adjoint_solver,
        stepsize_controller=adjoint_stepsize_controller)
  else:
    raise ValueError(
        f'Unsupported adjoint method: {config.prob_flow.adjoint_method}')

  prob_flow = probability_flow.ProbabilityFlow(
      sde=sde,
      score_fn=score_fn,
      solver=solver,
      stepsize_controller=stepsize_controller,
      adjoint=adjoint,
      n_trace_estimates=config.prob_flow.n_trace_estimates)

  return prob_flow


def _get_eht_image(config):
  with gfile.GFile(config.likelihood.eht_image_path, 'r') as f:
    image = np.load(f)
  # Rescale to [0, 1].
  return image / image.max()


def get_likelihood(config
                   ):
  """Return the likelihood module matching the config."""
  image_size = config.data.image_size
  image_shape = (
      config.data.image_size, config.data.image_size, config.data.num_channels)
  noise_scale = config.likelihood.noise_scale

  if config.likelihood.likelihood == 'Denoising':
    likelihood = forward_models.Denoising(
        scale=noise_scale,
        image_shape=image_shape)
  elif config.likelihood.likelihood == 'Deblurring':
    sigmas = jnp.ones(config.likelihood.n_dft**2) * noise_scale
    likelihood = forward_models.Deblurring(
        config.likelihood.n_dft,
        sigmas=sigmas,
        image_shape=image_shape)
  elif config.likelihood.likelihood == 'EHT':
    assert config.data.num_channels == 1
    # EHT forward model matrix and noise sigmas.
    with gfile.GFile(config.likelihood.eht_matrix_path, 'r') as f:
      eht_matrix = np.load(f)
    with gfile.GFile(config.likelihood.eht_sigmas_path, 'r') as f:
      eht_sigmas = np.load(f)

    # EHT target image.
    source_image = _get_eht_image(config)

    # Multiply noise scale by flux of image.
    eht_sigmas = eht_sigmas * np.sum(source_image)
    likelihood = forward_models.EHT(eht_matrix, eht_sigmas, image_size)
  return likelihood


def get_measurement(config,
                    likelihood,
                    single_image = True
                    ):
  """Return true image and measurement.

  Args:
    config: Config for the inference module (e.g., DPI, GradientAscent).
    likelihood: Likelihood module.
    single_image: If `True`, get one image and measurement.
      If `False`, use a batch of images.

  Returns:
    image: The true image, of shape (h, w, c) if `single_image` is True,
      else (b, h, w, c).
    y: Noisy measurement, of shape (1, m) if `single_image` is True,
      else (b, m).
  """
  if config.likelihood.likelihood == 'EHT' and config.data.centered:
    raise ValueError('Do not center data for EHT likelihood.')
  if config.data.dataset == 'EHT':
    # 'EHT' dataset refers to one Sgr A* image taken over the course of a night.
    image = _get_eht_image(config)  # shape: (image_size, image_size)
    image = np.expand_dims(image, axis=-1)
    # Get measurement.
    x = np.expand_dims(image, axis=0)
    y = likelihood.get_measurement(jax.random.PRNGKey(0), x)
    return image, y

  data_config = ml_collections.ConfigDict()
  data_config.data = ml_collections.ConfigDict()
  data_config.data.image_size = config.data.image_size
  data_config.data.num_channels = config.data.num_channels
  data_config.data.dataset = config.data.dataset
  data_config.data.centered = config.data.centered
  data_config.data.random_flip = False
  data_config.eval = ml_collections.ConfigDict()
  if single_image:
    data_config.eval.batch_size = jax.device_count()
  else:
    data_config.eval.batch_size = config.eval.batch_size

  # Get true image.
  _, _, test_ds = datasets.get_dataset(
      data_config, evaluation=True, shuffle_seed=config.data.shuffle_seed,
      device_batch=False)

  if single_image:
    image = next(iter(test_ds))['image'][0].numpy()
  else:
    image = next(iter(test_ds))['image'].numpy()

  scaler = datasets.get_data_scaler(data_config)
  image = scaler(image)

  # Get measurement.
  x = np.expand_dims(image, axis=0) if single_image else image
  y = likelihood.get_measurement(jax.random.PRNGKey(0), x)

  return image, y
