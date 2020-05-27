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

# Lint as: python3
"""Sequential Monte-Carlo Sampler in jax."""

from typing import Dict

import gin
import jax
import jax.numpy as np
import numpy as onp

from grouptesting import bayes
from grouptesting import utils
from grouptesting.samplers import kernels
from grouptesting.samplers import sampler
from grouptesting.samplers import temperature


@gin.configurable
class SmcSampler(sampler.Sampler):
  """Sequential monte carlo sampler."""

  NAME = 'smc'

  def __init__(self,
               kernel=kernels.Gibbs(),
               resample_at_each_iteration: bool = True,
               num_particles: int = 10000,
               min_kernel_iterations: int = 2,
               max_kernel_iterations: int = 20,
               min_ratio_delta: float = 0.02,
               target_unique_ratio: float = 0.95):
    super().__init__()
    self._kernel = kernel
    self._resample_at_each_iteration = resample_at_each_iteration
    self._num_particles = num_particles
    self._min_kernel_iterations = min_kernel_iterations
    self._max_kernel_iterations = max_kernel_iterations
    self._min_ratio_delta = min_ratio_delta
    self._target_unique_ratio = target_unique_ratio

  @property
  def is_cheap(self):
    return False

  def produce_sample(self, rng, state):
    """Produces a particle approx to posterior distribution given tests.

    If no tests have been carried out so far, naively sample from
    prior distribution.

    Otherwise take into account previous tests to form posterior
    and sample from it using a SMC sampler.

    Args:
     rng: a random key
     state: the current state of what has been tested, etc.

    Returns:
     Nothing but sets the particle_weights and particles members.
    """
    shape = (self._num_particles, state.num_patients)
    if np.size(state.past_test_results) == 0:
      self.particles = (
          jax.random.uniform(rng, shape=shape) < state.prior_infection_rate)
      self.particle_weights = np.ones(
          (self._num_particles,)) / self._num_particles
    else:
      rngs = jax.random.split(rng, 2)
      if self._resample_at_each_iteration or state.particles is None:
        particles = jax.random.uniform(rngs[0], shape=shape) < 0.5
      else:
        particles = state.particles
      # sample from posterior
      self.particle_weights, self.particles = self.resample_move(
          rngs[1], particles, state)

  def resample_move(self, rng, particles, state):
    """Resample / Move sequence."""
    log_posterior_params = state.log_posterior_params
    log_posterior = bayes.log_posterior(particles, **log_posterior_params)
    alpha, logpialphaparticles = temperature.find_step_length(0, log_posterior)
    rho = alpha
    particle_weights = temperature.importance_weights(logpialphaparticles)
    while rho < 1:
      print(f'Sampling {rho:.0%}', end='\r')
      rng, *rngs = jax.random.split(rng, 3)
      self._kernel.fit_model(particle_weights, particles)
      particles = particles[self.resample(rngs[0], particle_weights), :]
      particles = self.move(rngs[1], particles, rho, log_posterior_params)
      alpha, logpialphaparticles = temperature.find_step_length(
          rho, bayes.log_posterior(particles, **log_posterior_params))
      particle_weights = temperature.importance_weights(logpialphaparticles)
      rho = rho + alpha
    return particle_weights, particles

  def resample(self,
               rng: int,
               particle_weights: np.ndarray) -> np.ndarray:
    """Systematic resampling given weights.

    This is coded in onp because it seems to be faster than lax here.

    Args:
     rng: the random number generator, like elsewhere.
     particle_weights: the weights of the particles.

    Returns:
     Return a list of indices.
    """
    num_particles = particle_weights.shape[0]
    noise = onp.array(jax.random.uniform(rng, shape=((1,))))[0]
    positions = (noise + onp.arange(num_particles)) / num_particles
    cum_sum = onp.cumsum(onp.array(particle_weights))
    cum_sum[-1] = 1  # fix to handle numerical issues where sum isn't exactly 1
    i, j = 0, 0
    indices = list()
    while i < num_particles:
      if positions[i] < cum_sum[j]:
        indices.append(j)
        i += 1
      else:
        j += 1
    return indices

  def move(self,
           rng: int,
           particles: np.ndarray,
           rho: float,
           log_posterior_params: Dict[str, np.ndarray]):
    """Applies the kernel until particles are sufficiently refreshed.

    Args:
      rng : random seed
      particles : current particles
      rho : temperature, between 0 and 1
      log_posterior_params : dict of useful quantities to evaluate posterior

    Returns:
      a np.ndarray of particles of the same size as the input.
    """
    num_particles = particles.shape[0]
    rng, rng_uniques = jax.random.split(rng, 2)
    unique_ratio = utils.unique(rng_uniques, particles) / num_particles

    for it in range(self._max_kernel_iterations):
      rng, *rngs = jax.random.split(rng, 3)
      particles = self._kernel(rngs[0], particles, rho, log_posterior_params)

      old_unique_ratio = unique_ratio
      unique_ratio = utils.unique(rngs[1], particles) / num_particles
      if it < self._min_kernel_iterations - 1:
        continue
      ratio_delta = np.abs(old_unique_ratio - unique_ratio)
      if (ratio_delta <= self._min_ratio_delta or
          unique_ratio > self._target_unique_ratio):
        break

    return particles
