# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Pairs of distributions to be compared through two-sample tests."""

from collections.abc import Callable
import enum
import itertools

from absl import logging
import attrs
import jax
import jax.numpy as jnp
from scipy import stats

from f_divergence_tests import testing_typing


class ParametricDistribution(enum.Enum):
  GAUSSIAN = 'gaussian'
  LAPLACE = 'laplace'


def _get_delta_gaussian_std1(
    epsilon, sensitivity
):
  delta = stats.norm.cdf((sensitivity / 2) - epsilon / sensitivity) - jnp.exp(
      epsilon
  ) * stats.norm.cdf((-sensitivity / 2) - epsilon / sensitivity)
  return delta


@attrs.define
class ParametricDistributionSamples:
  """Paired samples from a parametric family of distributions.

  Generates two arrays with `n_samples` samples from two distribution of the
  same parametric family. Possible values for the parametric family are
  "gaussian" and "laplace", and the distributions have same variance but
  different means.

  Attributes:
    name: Name of the distribution, "gaussian" or "laplace".
    epsilon: logarithm of the order of the HSD between two distributions.
    n_samples: Number of samples to generate from each distribution.
    key_seed: Seed for the random number generator.
    mu_x: mean of the first distribution.
    mu_y: mean of the second distribution.
    delta: HSD between the two distributions.
    samples_x: Samples for the first distribution.
    samples_y: Samples for the second distribution.
  """

  name: str = attrs.field()
  epsilon: float = attrs.field()
  n_samples: int = attrs.field()

  key_seed: int = attrs.field()
  mu_x: float = attrs.field(default=0.0)
  mu_y: float = attrs.field(default=1.0)

  samples_x: jnp.ndarray = attrs.field(init=False)
  samples_y: jnp.ndarray = attrs.field(init=False)
  delta: float = attrs.field(init=False)

  def __attrs_post_init__(self):
    key = jax.random.PRNGKey(self.key_seed)
    _, subkey_x, subkey_y = jax.random.split(key, 3)
    sensitivity = jnp.abs(self.mu_x - self.mu_y)

    if self.name == 'gaussian':
      self.samples_x = (
          jax.random.normal(key=subkey_x, shape=(self.n_samples, 1)) + self.mu_x
      )
      self.samples_y = (
          jax.random.normal(key=subkey_y, shape=(self.n_samples, 1)) + self.mu_y
      )

      self.delta = _get_delta_gaussian_std1(
          epsilon=self.epsilon, sensitivity=sensitivity
      )
    elif self.name == 'laplace':
      self.samples_x = (
          sensitivity
          / self.epsilon
          * jax.random.laplace(key=subkey_x, shape=(self.n_samples, 1))
          + self.mu_x
      )
      self.samples_y = (
          sensitivity
          / self.epsilon
          * jax.random.laplace(key=subkey_y, shape=(self.n_samples, 1))
          + self.mu_y
      )
      self.delta = 0.0
    else:
      raise NotImplementedError(f'Unknown parametric distribution: {self.name}')


def rejection_sampling(
    seed,
    density_fn,
    dimension,
    density_max,
    minval,
    maxval,
    n_samples,
):
  """Returns samples from a density function using rejection sampling.

  This function is based on code from:
  Repository:
  https://github.com/antoninschrab/mmdagg-paper/blob/master/sampling.py
  Author: Antonin Schrab
  License: MIT License

  Samples from a bounded density function on a compact domain. The density
  satisfies density_fn(x)=0 if x_min > x_i, or x_i > x_max for some i in
  1, ..., d.

  Args:
      seed: Seed for JAX's random number generator.
      density_fn: Probability density function for a random variable.
      dimension: Dimension of the random variable.
      density_max: Maximum value of the density function.
      minval: Lower bound for all dimension. Density is zero outside this bound.
      maxval: Upper bound for all dimension. Density is zero outside this bound.
      n_samples: Number of samples to generate.
  """
  n_generated_samples = 0
  samples = jnp.zeros((n_samples, dimension))
  key = jax.random.PRNGKey(seed)

  while n_generated_samples < n_samples:
    key, k1, k2 = jax.random.split(key, 3)
    new_sample = jax.random.uniform(
        key=k1, shape=(dimension,), minval=minval, maxval=maxval
    )
    density = jax.random.uniform(key=k2, minval=0, maxval=density_max)

    if density <= density_fn(new_sample):
      samples = samples.at[n_generated_samples].set(new_sample)
      n_generated_samples += 1

  return samples


def g_fn(x):
  """Function G used in equation 17 in https://arxiv.org/pdf/2110.15073.pdf.

  This function describes intensity of the perturbations for a perturbed uniform
  distribution. See Figure 2 in the paper for an illustration.

  Args:
    x: array in R^d.

  Returns:
    float that corresponds to the G function evaluated at x.
  """
  if -1 < x and x < -0.5:
    return jnp.exp(-1 / (1 - (4 * x + 3) ** 2))
  if -0.5 < x and x < 0:
    return -jnp.exp(-1 / (1 - (4 * x + 1) ** 2))
  return 0


def f_theta_density_fn(
    x, num_perturbations, smoothness, perturbation_multiplier, key
):
  """Return a perturbed uniform density function value at x.

  This function implements the densitity function of a perturbed uniform random
  variable in R^d. The density is defined in equation 17 in
  https://arxiv.org/pdf/2110.15073. A two-sample test between samples from this
  perturbed uniform and the standard uniform distribution provides a lower bound
  on the minimax rate of testing over the Sobolev ball of dimension d and
  smoothness parameter s.

  Args:
      x: Point in R^d.
      num_perturbations: Number of perturbations for the uniform distribution.
      smoothness: Smoothness parameter of Sobolev ball.
      perturbation_multiplier: Scaling factor emphazising the effect of the
        pertubation (c_d in Eq. (17)).
      key: JAX random number generator key.
  """
  x = jnp.atleast_1d(x)
  d = x.shape[0]
  if (
      perturbation_multiplier * num_perturbations ** (-smoothness) * jnp.exp(-d)
      > 1 + 1e-6
  ):
    raise ValueError('The density is negative.')

  _, subkey = jax.random.split(key)

  theta = jax.random.choice(
      key=subkey, a=jnp.array([-1, 1]), shape=(num_perturbations**d,)
  )
  output = 0
  indices = itertools.product(
      [i + 1 for i in range(num_perturbations)], repeat=d
  )

  for nu, permutation_nu in enumerate(indices):
    output += theta[nu] * jnp.prod(
        jnp.array([
            g_fn(num_perturbations * x[i] - permutation_nu[i]) for i in range(d)
        ])
    )

  output *= perturbation_multiplier * num_perturbations ** (-smoothness)
  # Uniform distribution densiti is 1 in [0, 1] and 0 otherwise.
  if jnp.min(x) >= 0 and jnp.max(x) <= 1:
    output += 1
  return output


@attrs.define
class PerturbedUniformSamples:
  """Samples from perturbed and standard uniform distributions.

  Generates two arrays with `n_samples` samples from  uniform and a perturbed
  uniform distribution defined in equation 17 in
  https://arxiv.org/pdf/2110.15073.pdf.

  Attributes:
    n_samples: Number of samples to generate from each distribution.
    key_seed: Seed for the random number generator.
    num_perturbations: Number of perturbations for the perturbed uniform
      distribution.
    dimension: Dimension of the random variables.
    smoothness: Smoothness parameter for the perturbation.
    scale: Scaling factor for the perturbation.
    perturbation_multiplier: Scaling factor emphazising the effect of the
      pertubation (c_d in Eq. (17)).
    samples_x: Samples for the first distribution.
    samples_y: Samples for the second distribution.
  """

  n_samples: int
  key_seed: int
  num_perturbations: int
  dimension: int
  smoothness: float
  scale: float
  perturbation_multiplier: testing_typing.SCALAR = attrs.field(init=False)
  samples_x: jnp.ndarray = attrs.field(init=False)
  samples_y: jnp.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    self.perturbation_multiplier = (
        jnp.exp(self.dimension)
        * self.num_perturbations**self.smoothness
        * self.scale
    )
    key = jax.random.PRNGKey(self.key_seed)
    _, subkey_x, subkey_y = jax.random.split(key, 3)

    density_max = 1 + self.perturbation_multiplier * self.num_perturbations ** (
        -self.smoothness
    ) * jnp.exp(-self.dimension)

    self.samples_x = rejection_sampling(
        seed=self.key_seed + 1,
        density_fn=lambda x: f_theta_density_fn(
            x,
            num_perturbations=self.num_perturbations,
            smoothness=self.smoothness,
            perturbation_multiplier=self.perturbation_multiplier,
            key=subkey_x,
        ),
        dimension=self.dimension,
        density_max=density_max,
        minval=0.0,
        maxval=1.0,
        n_samples=self.n_samples,
    )
    self.samples_y = jax.random.uniform(
        key=subkey_y, shape=(self.n_samples, self.dimension)
    )


def null_expo1d_density_fn(x):
  return 2000 * stats.expon.pdf(x)


def alternative_0_expo1d_density_fn(x, location, scale, multiplier):
  return null_expo1d_density_fn(x) + multiplier * stats.norm.pdf(
      x, loc=location, scale=scale
  )


@attrs.define
class Expo1dSamples:
  """Samples from the physics Expo-1D distribution."""

  location: float
  scale: float
  multiplier: float
  n_samples: int
  key_seed: int
  samples_x: jnp.ndarray = attrs.field(init=False)
  samples_y: jnp.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):

    self.samples_x = rejection_sampling(
        seed=self.key_seed,
        density_fn=null_expo1d_density_fn,
        dimension=1,
        density_max=2000,
        n_samples=self.n_samples,
        minval=0,
        maxval=10,
    )
    if self.multiplier <= 0.001:
      logging.info('Multiplier is too small, using null density.')
      alternative_density = null_expo1d_density_fn
    else:
      alternative_density = lambda x: alternative_0_expo1d_density_fn(
          x,
          location=self.location,
          scale=self.scale,
          multiplier=self.multiplier,
      )
    self.samples_y = rejection_sampling(
        seed=self.key_seed + 49,
        density_fn=alternative_density,
        dimension=1,
        density_max=2000,
        n_samples=self.n_samples,
        minval=0,
        maxval=10,
    )
