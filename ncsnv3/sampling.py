# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

from absl import logging
import flax
import flax.jax_utils as flax_utils
import flax.nn as nn
import jax
import jax.numpy as jnp
import numpy as np

from .models import utils as mutils


_SAMPLERS = {}


def register_sampler(func=None, *, name=None):
  """Decorator for registering sampler functions."""
  def wrapper(func):
    if name is None:
      local_name = func.__name__
    else:
      local_name = name
    if local_name in _SAMPLERS:
      raise ValueError(f'Sampler {local_name} has already been registered!')
    _SAMPLERS[local_name] = func
    return func
  if func is None:
    return wrapper
  else:
    return wrapper(func)


def get_samples(rng, config, state, shape, scaler, inverse_scaler,
                class_conditional=False, colab=False):
  """Generate samples. Assume state is unreplicated."""

  continuous_sigmas = 'continuous' in config.training.loss
  rng1, rng2 = jax.random.split(rng)
  sampler_name = config.sampling.method
  if sampler_name not in _SAMPLERS:
    raise ValueError(
        f'Sampler {sampler_name} not found. Supported samplers are {list(_SAMPLERS.keys())}'
    )
  else:
    sampler = _SAMPLERS[sampler_name]

  if class_conditional:
    class_labels = jax.random.choice(rng1, config.data.num_classes,
                                     shape=(shape[0], shape[1]))
    rng1, _ = jax.random.split(rng1)
  else:
    class_labels = None

  if sampler_name in ['diffusion_sampling', 'reverse_diffusion', 'gradient_flow']:
    sigmas = mutils.get_sigmas(config)
    init_sample = jax.random.normal(rng1, shape) * sigmas[0]
    samples = sampler(
        rng2,
        init_sample,
        state,
        sigmas,
        inverse_scaler,
        class_labels=class_labels,
        continuous_sigmas=continuous_sigmas,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab)

  elif sampler_name in ['ddpm', 'ddpm_reproduce',
                        'ddpm_reverse_diffusion', 'ddpm_gradient_flow']:
    ddpm_params = mutils.get_ddpm_params()
    init_sample = jax.random.normal(rng1, shape)
    samples = sampler(
        rng2,
        init_sample,
        state,
        ddpm_params,
        inverse_scaler,
        noise_removal=config.sampling.noise_removal,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab)

  elif sampler_name in ['ddpm_ald_fix_snr_diffusion_sampling',
                        'ddpm_ald_fix_snr',
                        'ddpm_ald_fix_snr_reverse_diffusion',
                        'ddpm_ald_fix_snr_gradient_flow']:
    ddpm_params = mutils.get_ddpm_params()
    init_sample = jax.random.normal(rng1, shape)
    samples = sampler(
        rng2,
        init_sample,
        state,
        ddpm_params,
        inverse_scaler,
        n_steps_each=config.sampling.n_steps_each,
        target_snr=config.sampling.target_snr,
        noise_removal=config.sampling.noise_removal,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab)

  elif sampler_name == 'ald':
    sigmas = mutils.get_sigmas(config)
    init_sample = jax.random.uniform(rng1, shape)
    init_sample = scaler(init_sample)
    samples = sampler(
        rng2,
        init_sample,
        state,
        sigmas,
        inverse_scaler,
        n_steps_each=config.sampling.n_steps_each,
        step_size=config.sampling.step_size,
        class_labels=class_labels,
        continuous_sigmas=continuous_sigmas,
        noise_removal=config.sampling.noise_removal,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab)

  elif sampler_name.startswith('ald_fix_snr'):
    sigmas = mutils.get_sigmas(config)
    init_sample = jax.random.uniform(rng1, shape)
    init_sample = scaler(init_sample)
    rng3, _ = jax.random.split(rng1)
    init_sample = init_sample + jax.random.normal(rng3, shape) * sigmas[0]
    samples = sampler(
        rng2,
        init_sample,
        state,
        sigmas,
        inverse_scaler,
        class_labels=class_labels,
        continuous_sigmas=continuous_sigmas,
        n_steps_each=config.sampling.n_steps_each,
        target_snr=config.sampling.target_snr,
        noise_removal=config.sampling.noise_removal,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab)

  elif sampler_name == 'ddpm_ald_fix_snr':
    init_sample = jax.random.normal(rng1, shape)
    ddpm_params = mutils.get_ddpm_params()
    samples = sampler(
        rng2,
        init_sample,
        state,
        ddpm_params,
        inverse_scaler,
        n_steps_each=config.sampling.n_steps_each,
        target_snr=config.sampling.target_snr,
        final_only=config.sampling.final_only,
        verbose=True,
        colab=colab
    )

  return samples


@register_sampler(name='ald')
def anneal_langevin_dynamics(rng,
                             init,
                             state,
                             sigmas,
                             inverse_scaler,
                             class_labels=None,
                             continuous_sigmas=False,
                             n_steps_each=200,
                             step_size=0.000008,
                             noise_removal=True,
                             final_only=False,
                             verbose=False,
                             colab=False):
  """The original annealed Langevin dynamics sampling used in NCSNv1/v2.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class. should be replicated.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    n_steps_each: the number of Langevin steps for each noise level.
    step_size: the step size for running Langevin dynamics
    noise_removal: FID will increase by a large amount if set to False.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for c, sigma in enumerate(sigmas):
    step = step_size * (sigma / sigmas[-1])**2
    if continuous_sigmas:
      noise_level = sigma
    else:
      noise_level = c
    noise_level = flax.jax_utils.replicate(c)
    for _ in range(n_steps_each):
      grad = score_eval(x, noise_level)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      x = x + step * grad + noise * jnp.sqrt(step * 2)
      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0] * noise.shape[1], -1)),
            axis=-1).mean()
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        snr = jnp.sqrt(step / 2.) * grad_norm / noise_norm
        grad_mean_norm = jnp.linalg.norm(grad.mean(axis=(0, 1)).reshape(
            (-1,)))**2 * sigma**2
        if colab:
          print(
              'level: %d, step_size: %.5e, grad_norm: %.5e, image_norm: %.5e, snr: %.5e, grad_mean_norm: %.5e'
              % (c, step, grad_norm, image_norm, snr, grad_mean_norm))
        else:
          logging.info(
              'level: %d, step_size: %.5e, grad_norm: %.5e, image_norm: %.5e, snr: %.5e, grad_mean_norm: %.5e',
              c, step, grad_norm, image_norm, snr, grad_mean_norm)

  if noise_removal:
    if continuous_sigmas:
      last_noise = flax.jax_utils.replicate(sigmas[-1])
    else:
      last_noise = flax.jax_utils.replicate(len(sigmas) - 1)
    x = x + sigmas[-1]**2 * score_eval(x, last_noise)
    logging.info('Finished noise removal!')
  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def diffusion_sampling(rng,
                 init,
                 state,
                 sigmas,
                 inverse_scaler,
                 class_labels=None,
                 continuous_sigmas=False,
                 final_only=False,
                 verbose=False,
                 colab=False):
  """Discrete diffusion sampling (the method in DDPM) applied to NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)


  for T in range(len(sigmas)):  # pylint: disable=invalid-name
    if continuous_sigmas:
      replicated_T = flax.jax_utils.replicate(sigmas[T])
    else:
      replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    grad = score_eval(x, replicated_T)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)

    x0 = x + sigmas[T]**2 * grad
    if T == len(sigmas) - 1:
      x = x0
    else:
      std = sigmas[T + 1] * jnp.sqrt(sigmas[T]**2 -
                                     sigmas[T + 1]**2) / sigmas[T]
      coeff = sigmas[T + 1]**2 / sigmas[T]**2
      x = coeff * x + (1. - coeff) * x0 + std * noise

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e'
              % (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def reverse_diffusion(rng,
                      init,
                      state,
                      sigmas,
                      inverse_scaler,
                      class_labels=None,
                      continuous_sigmas=False,
                      final_only=False,
                      verbose=False,
                      colab=False):
  """Reverse diffusion sampling for NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for T in range(len(sigmas)):  # pylint: disable=invalid-name
    if continuous_sigmas:
      replicated_T = flax.jax_utils.replicate(sigmas[T])
    else:
      replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name

    grad = score_eval(x, replicated_T)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)

    variance = sigmas[T]**2 - sigmas[T+1]**2 if T < len(sigmas) - 1 else sigmas[T]**2
    x0 = x + variance * grad
    if T == len(sigmas) - 1:
      x = x0
    else:
      std = jnp.sqrt(variance)
      x = x0 + std * noise

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e'
              % (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def gradient_flow(rng,
                  init,
                  state,
                  sigmas,
                  inverse_scaler,
                  class_labels=None,
                  continuous_sigmas=False,
                  final_only=False,
                  verbose=False,
                  colab=False):
  """Gradient flow sampling for NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for T in range(len(sigmas)):  # pylint: disable=invalid-name
    if continuous_sigmas:
      replicated_T = flax.jax_utils.replicate(sigmas[T])
    else:
      replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name

    grad = score_eval(x, replicated_T)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)

    variance = sigmas[T]**2 - sigmas[T+1]**2 if T < len(sigmas) - 1 else sigmas[T]**2
    x = x + variance * grad / 2.

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e'
              % (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler(name='ald_fix_snr')
def anneal_langevin_dynamics_fix_snr(rng,
                                     init,
                                     state,
                                     sigmas,
                                     inverse_scaler,
                                     class_labels=None,
                                     continuous_sigmas=False,
                                     n_steps_each=200,
                                     target_snr=0.2,
                                     noise_removal=True,
                                     final_only=False,
                                     verbose=False,
                                     colab=False):
  """Annealed Langevin dynamics sampling for NCSNs in the form of fixed SNR.

  Scales are estimated by the norm of score functions.
  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    n_steps_each: the number of Langevin steps for each noise level.
    target_snr: the target signal to noise ratio.
    noise_removal: FID will increase by a lot if set to False.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for c, sigma in enumerate(sigmas):
    if continuous_sigmas:
      noise_level = flax.jax_utils.replicate(sigma)
    else:
      noise_level = flax.jax_utils.replicate(c)
    for _ in range(n_steps_each):
      grad = score_eval(x, noise_level)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.
      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)
      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e' %
                (c, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', c,
                       step_size, image_norm)
  if noise_removal:
    if continuous_sigmas:
      last_noise = sigmas[-1]
    else:
      last_noise = len(sigmas) - 1
    x = x + sigmas[-1]**2 * score_eval(x, flax.jax_utils.replicate(last_noise))
    logging.info('Finished noise removal!')

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler(name='ddpm')
def ddpm_sampling(rng,
                  init,
                  state,
                  ddpm_params,
                  inverse_scaler,
                  noise_removal=False,
                  final_only=False,
                  verbose=False,
                  colab=False):
  """The sampling procedure of DDPMs as described in the DDPM paper.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    grad = score_eval(x, replicated_T)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)
    x = 1. / np.sqrt(alphas[T, None, None, None, None]) * (
        x -
        (betas[T, None, None, None, None] /
         sqrt_1m_alphas_cumprod[T, None, None, None, None]) * grad)

    if not noise_removal or T > 0:
      x = x + np.sqrt(betas[T, None, None, None, None]) * noise

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e' %
              (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler(name='ddpm_reverse_diffusion')
def ddpm_reverse_diffusion(rng,
                           init,
                           state,
                           ddpm_params,
                           inverse_scaler,
                           noise_removal=False,
                           final_only=False,
                           verbose=False,
                           colab=False):
  """Reverse diffusion sampling for DDPMs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    grad = - score_eval(x, replicated_T) / sqrt_1m_alphas_cumprod[T]
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)

    x = (2. - np.sqrt(1 - betas[T])) * x + grad * betas[T]

    if not noise_removal or T > 0:
      x = x + np.sqrt(betas[T, None, None, None, None]) * noise

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e' %
              (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ddpm_gradient_flow(rng,
                       init,
                       state,
                       ddpm_params,
                       inverse_scaler,
                       noise_removal=False,
                       final_only=False,
                       verbose=False,
                       colab=False):
  """Gradient flow sampling for DDPMs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    grad = - score_eval(x, replicated_T) / sqrt_1m_alphas_cumprod[T]
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)

    x = (2. - np.sqrt(1 - betas[T])) * x + grad * betas[T] / 2.

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e' %
              (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ddpm_ald_fix_snr(rng,
                     init,
                     state,
                     ddpm_params,
                     inverse_scaler,
                     n_steps_each=2,
                     target_snr=0.15,
                     noise_removal=False,
                     final_only=False,
                     verbose=False,
                     colab=False):
  """Annealed Langevin dynamics sampling for DDPMs.

  In the form of fixed SNR and estimated scales with norm of score functions.
  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.

  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  alphas_cumprod = ddpm_params['alphas_cumprod']
  sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']
  x = x / sqrt_alphas_cumprod[-1]

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    for _ in range(n_steps_each):
      grad = -score_eval(
          sqrt_alphas_cumprod[T] * x,
          replicated_T) * sqrt_alphas_cumprod[T] / sqrt_1m_alphas_cumprod[T]

      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)

      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.
      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e'
                % (T, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', T,
                       step_size, image_norm)

  if noise_removal:
    x = x - sqrt_1m_alphas_cumprod[0] / sqrt_alphas_cumprod[0] * score_eval(
        x * sqrt_alphas_cumprod[0], flax.jax_utils.replicate(0))

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler(name='ddpm_reproduce')
def ddpm_sampling_reproduce(rng,
                            init,
                            state,
                            ddpm_params,
                            inverse_scaler,
                            noise_removal=False,
                            final_only=False,
                            verbose=False,
                            colab=False):
  """The sampling procedure of DDPMs. Same implementation as the DDPM codebase.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)
  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  alphas_cumprod = ddpm_params['alphas_cumprod']
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']
  sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
  sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

  alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
  posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. -
                                                                 alphas_cumprod)
  posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (
      1. - alphas_cumprod)

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    grad = score_eval(x, replicated_T)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, x.shape)
    x0 = sqrt_recip_alphas_cumprod[T, None, None, None,
                                   None] * x - sqrt_recipm1_alphas_cumprod[
                                       T, None, None, None, None] * grad
    x0 = jnp.clip(x0, -1., 1.)
    x = posterior_mean_coef1[T, None, None, None,
                             None] * x0 + posterior_mean_coef2[T, None, None,
                                                               None, None] * x

    if not noise_removal or T > 0:
      x = x + np.sqrt(betas[T, None, None, None, None]) * noise

    if not final_only:
      images.append(inverse_scaler(np.asarray(x)))
    if verbose and jax.host_id() == 0:
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      image_norm = jnp.linalg.norm(
          x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
      if colab:
        print('level: %d, grad_norm: %.5e, image_norm: %.5e'
              % (T, grad_norm, image_norm))
      else:
        logging.info('level: %d, grad_norm: %.5e, image_norm: %.5e', T,
                     grad_norm, image_norm)

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images



@register_sampler(name='ddpm_ald_fix_snr_diffusion_sampling')
def ddpm_ald_fix_snr_diffusion_sampling(rng,
                           init,
                           state,
                           ddpm_params,
                           inverse_scaler,
                           n_steps_each=2,
                           target_snr=0.15,
                           noise_removal=False,
                           final_only=False,
                           verbose=False,
                           colab=False):
  """Discrete diffusion + annealed Langevin dynamics for sampling from DDPMs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.

  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  alphas_cumprod = ddpm_params['alphas_cumprod']
  sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']
  sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
  sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

  alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
  posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. -
                                                                 alphas_cumprod)
  posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (
      1. - alphas_cumprod)

  x = x / sqrt_alphas_cumprod[-1]

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    if T < len(betas) - 1:
      y = sqrt_alphas_cumprod[T+1] * x
      grad = score_eval(y, flax.jax_utils.replicate(T+1))
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      y0 = sqrt_recip_alphas_cumprod[T+1, None, None, None,
                                     None] * y - sqrt_recipm1_alphas_cumprod[
                                         T+1, None, None, None, None] * grad
      y0 = jnp.clip(y0, -1., 1.)
      y = posterior_mean_coef1[T + 1, None, None, None,
                               None] * y0 + posterior_mean_coef2[
                                   T + 1, None, None, None, None] * y

      y = y + np.sqrt(betas[T + 1, None, None, None, None]) * noise
      x = y / sqrt_alphas_cumprod[T]

    for _ in range(n_steps_each):
      grad = -score_eval(
          sqrt_alphas_cumprod[T] * x,
          replicated_T) * sqrt_alphas_cumprod[T] / sqrt_1m_alphas_cumprod[T]

      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)

      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.
      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e'
                % (T, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', T,
                       step_size, image_norm)

  if noise_removal:
    x = x - sqrt_1m_alphas_cumprod[0] / sqrt_alphas_cumprod[0] * score_eval(
        x * sqrt_alphas_cumprod[0], flax.jax_utils.replicate(0))

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ddpm_ald_fix_snr_reverse_diffusion(rng,
                                       init,
                                       state,
                                       ddpm_params,
                                       inverse_scaler,
                                       n_steps_each=2,
                                       target_snr=0.15,
                                       noise_removal=False,
                                       final_only=False,
                                       verbose=False,
                                       colab=False):
  """Reverse diffusion + annealed Langevin dynamics for sampling from DDPMs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.

  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  alphas_cumprod = ddpm_params['alphas_cumprod']
  sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']
  sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
  sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

  alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
  posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. -
                                                                 alphas_cumprod)
  posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (
      1. - alphas_cumprod)

  x = x / sqrt_alphas_cumprod[-1]

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    if T < len(betas) - 1:
      y = sqrt_alphas_cumprod[T+1] * x
      grad = -score_eval(y, flax.jax_utils.replicate(T+1)) / sqrt_1m_alphas_cumprod[T+1]
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      y = (2. - np.sqrt(1 - betas[T+1])) * y + grad * betas[T+1]
      y = y + np.sqrt(betas[T+1]) * noise
      x = y / sqrt_alphas_cumprod[T]

    for _ in range(n_steps_each):
      grad = -score_eval(
          sqrt_alphas_cumprod[T] * x,
          replicated_T) * sqrt_alphas_cumprod[T] / sqrt_1m_alphas_cumprod[T]

      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)

      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.
      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e'
                % (T, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', T,
                       step_size, image_norm)

  if noise_removal:
    x = x - sqrt_1m_alphas_cumprod[0] / sqrt_alphas_cumprod[0] * score_eval(
        x * sqrt_alphas_cumprod[0], flax.jax_utils.replicate(0))

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ddpm_ald_fix_snr_gradient_flow(rng,
                                   init,
                                   state,
                                   ddpm_params,
                                   inverse_scaler,
                                   n_steps_each=2,
                                   target_snr=0.15,
                                   noise_removal=False,
                                   final_only=False,
                                   verbose=False,
                                   colab=False):
  """Gradient flow + annealed Langevin dynamics for sampling from DDPMs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    ddpm_params: a dictionary containing hyperparameters of the DDPM model.
    noise_removal: whether to remove the noise at the final sampling step.
    inverse_scaler: a function to scale generated inputs back to valid images.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.

  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  @functools.partial(jax.pmap, axis_name='batch')
  def score_eval(sample, noise_level):
    labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
    with nn.stateful(state.model_state, mutable=False):
      score = model_ema(sample, labels, train=False)
    return score

  betas = ddpm_params['betas']
  alphas = ddpm_params['alphas']
  alphas_cumprod = ddpm_params['alphas_cumprod']
  sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = ddpm_params['sqrt_1m_alphas_cumprod']
  sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
  sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

  alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
  posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. -
                                                                 alphas_cumprod)
  posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (
      1. - alphas_cumprod)

  x = x / sqrt_alphas_cumprod[-1]

  for T in reversed(range(len(betas))):  # pylint: disable=invalid-name
    replicated_T = flax.jax_utils.replicate(T)  # pylint: disable=invalid-name
    if T < len(betas) - 1:
      y = sqrt_alphas_cumprod[T+1] * x
      grad = -score_eval(y, flax.jax_utils.replicate(T+1)) / sqrt_1m_alphas_cumprod[T+1]
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      y = (2. - np.sqrt(1 - betas[T+1])) * y + grad * betas[T+1] / 2.
      x = y / sqrt_alphas_cumprod[T]

    for _ in range(n_steps_each):
      grad = -score_eval(
          sqrt_alphas_cumprod[T] * x,
          replicated_T) * sqrt_alphas_cumprod[T] / sqrt_1m_alphas_cumprod[T]

      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)

      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.
      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e'
                % (T, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', T,
                       step_size, image_norm)

  if noise_removal:
    x = x - sqrt_1m_alphas_cumprod[0] / sqrt_alphas_cumprod[0] * score_eval(
        x * sqrt_alphas_cumprod[0], flax.jax_utils.replicate(0))

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ald_fix_snr_diffusion_sampling(rng,
                      init,
                      state,
                      sigmas,
                      inverse_scaler,
                      class_labels=None,
                      continuous_sigmas=False,
                      n_steps_each=200,
                      target_snr=0.2,
                      noise_removal=True,
                      final_only=False,
                      verbose=False,
                      colab=False):
  """Discrete diffusion sampling + annealed Langevin dynamics for NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    n_steps_each: the number of Langevin steps for each noise level.
    target_snr: the target signal to noise ratio.
    noise_removal: FID will increase by a lot if set to False.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for c, sigma in enumerate(sigmas):
    if continuous_sigmas:
      noise_level = flax.jax_utils.replicate(sigma)
    else:
      noise_level = flax.jax_utils.replicate(c)
    if c > 0:
      prev_noise = sigmas[c - 1] if continuous_sigmas else c - 1
      grad = score_eval(x, flax.jax_utils.replicate(prev_noise))
      x0 = x + sigmas[c-1]**2 * grad
      coeff = sigmas[c]**2 / sigmas[c - 1]**2
      std = sigmas[c] / sigmas[c - 1] * jnp.sqrt(sigmas[c-1]**2 - sigmas[c]**2)
      x = coeff * x + (1. - coeff) * x0
      rng, noise_rng = jax.random.split(rng)
      noise = jax.random.normal(noise_rng, x.shape)
      x = x + std * noise

    for step in range(n_steps_each):
      grad = score_eval(x, noise_level)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.

      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))
      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e' %
                (c, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', c,
                       step_size, image_norm)
  if noise_removal:
    last_noise = sigmas[-1] if continuous_sigmas else len(sigmas) - 1
    x = x + sigmas[-1]**2 * score_eval(x, flax.jax_utils.replicate(last_noise))
    logging.info('Finished noise removal!')

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ald_fix_snr_reverse_diffusion(rng,
                                  init,
                                  state,
                                  sigmas,
                                  inverse_scaler,
                                  class_labels=None,
                                  continuous_sigmas=False,
                                  n_steps_each=200,
                                  target_snr=0.2,
                                  noise_removal=True,
                                  final_only=False,
                                  verbose=False,
                                  colab=False,
                                  temperature=1.):
  """Reverse diffusion + annealed Langevin dynamics for NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    n_steps_each: the number of Langevin steps for each noise level.
    target_snr: the target signal to noise ratio.
    noise_removal: FID will increase by a lot if set to False.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score * 1. / temperature
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score * 1. / temperature
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for c, sigma in enumerate(sigmas):
    if c > 0:
      prev_noise = sigmas[c - 1] if continuous_sigmas else c - 1
      grad = score_eval(x, flax.jax_utils.replicate(prev_noise))
      variance = (sigmas[c-1]**2 - sigma**2)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      x = x + variance * grad + jnp.sqrt(variance) * noise

    noise_level = sigma if continuous_sigmas else c
    noise_level = flax.jax_utils.replicate(noise_level)
    for step in range(n_steps_each):
      grad = score_eval(x, noise_level)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.

      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))

      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e' %
                (c, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', c,
                       step_size, image_norm)
  if noise_removal:
    last_noise = sigmas[-1] if continuous_sigmas else len(sigmas) - 1
    x = x + sigmas[-1]**2 * score_eval(x, flax.jax_utils.replicate(last_noise))
    logging.info('Finished noise removal!')

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images


@register_sampler
def ald_fix_snr_gradient_flow(rng,
                              init,
                              state,
                              sigmas,
                              inverse_scaler,
                              class_labels=None,
                              continuous_sigmas=False,
                              n_steps_each=200,
                              target_snr=0.2,
                              noise_removal=True,
                              final_only=False,
                              verbose=False,
                              colab=False,
                              temperature=1.):
  """Gradient flow + annealed Langevin dynamics for NCSNs.

  This function leverages `pmap` internally and shouldn't be pmapped itself.
  sample with EMA.

  Args:
    rng: jax random state for Langevin dynamics sample generation.
    init: the randomly initialized starting point for sampling.
    state: the full state class.
    sigmas: noise levels.
    inverse_scaler: scale generated samples back to valid images.
    class_labels: the target class labels for class-conditional generation.
    continuous_sigmas: use a continuous distribution of sigmas.
    n_steps_each: the number of Langevin steps for each noise level.
    target_snr: the target signal to noise ratio.
    noise_removal: FID will increase by a lot if set to False.
    final_only: if True store only the last sample. Otherwise store the whole
      sample history.
    verbose: if True log running information.
    colab: if True log to stdout.
  Returns:
    samples: list of image samples.
  """
  images = []
  x = init.copy()

  model_ema = state.optimizer.target.replace(params=state.params_ema)

  if class_labels is None:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, train=False)
      return score * 1. / temperature
  else:
    @functools.partial(jax.pmap, axis_name='batch')
    def score_eval(sample, noise_level, class_label):
      labels = jnp.ones((sample.shape[0],), dtype=jnp.int32) * noise_level
      with nn.stateful(state.model_state, mutable=False):
        score = model_ema(sample, labels, y=class_label, train=False)
      return score * 1. / temperature
    score_eval = functools.partial(score_eval, class_label=class_labels)

  for c, sigma in enumerate(sigmas):
    if c > 0:
      prev_noise = sigmas[c - 1] if continuous_sigmas else c - 1
      grad = score_eval(x, flax.jax_utils.replicate(prev_noise))
      variance = (sigmas[c-1]**2 - sigma**2)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      x = x + variance * grad / 2.

    noise_level = sigma if continuous_sigmas else c
    noise_level = flax.jax_utils.replicate(noise_level)
    for step in range(n_steps_each):
      grad = score_eval(x, noise_level)
      rng, sample_rng = jax.random.split(rng)
      noise = jax.random.normal(sample_rng, x.shape)
      grad_norm = jnp.linalg.norm(
          grad.reshape((grad.shape[1] * grad.shape[0], -1)), axis=-1).mean()
      noise_norm = jnp.linalg.norm(
          noise.reshape((noise.shape[0] * noise.shape[1], -1)), axis=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm)**2 * 2.

      x = x + step_size * grad + noise * jnp.sqrt(step_size * 2)

      if not final_only:
        images.append(inverse_scaler(np.asarray(x)))

      if verbose and jax.host_id() == 0:
        image_norm = jnp.linalg.norm(
            x.reshape((x.shape[0] * x.shape[1], -1)), axis=-1).mean()
        if colab:
          print('level: %d, step_size: %.5e, image_norm: %.5e' %
                (c, step_size, image_norm))
        else:
          logging.info('level: %d, step_size: %.5e, image_norm: %.5e', c,
                       step_size, image_norm)
  if noise_removal:
    last_noise = sigmas[-1] if continuous_sigmas else len(sigmas) - 1
    x = x + sigmas[-1]**2 * score_eval(x, flax.jax_utils.replicate(last_noise))
    logging.info('Finished noise removal!')

  if final_only:
    images.append(inverse_scaler(np.asarray(x)))
  return images
