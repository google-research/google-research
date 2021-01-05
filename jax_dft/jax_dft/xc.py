# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Library of exchange and correlation functionals."""

from jax import tree_util
import jax.numpy as jnp

from jax_dft import constants

# NOTE(leeley): Use tree_util.Partial so the physics xc_energy_density_fn can
# be jitted in the same way as the neural xc_energy_density_fn.


@tree_util.Partial
def exponential_coulomb_uniform_exchange_density(
    density,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA,
    epsilon=1e-15):
  """Exchange energy density for uniform gas with exponential coulomb.

  Equation 17 in the following paper provides the exchange energy per length
  for 1d uniform gas with exponential coulomb interaction.

  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  y = pi * density / kappa
  exchange energy per length
      = amplitude * kappa * (ln(1 + y ** 2) - 2 * y * arctan(y)) / (2 * pi ** 2)

  exchange energy density
      = exchange energy per length * pi / (kappa * y)
      = amplitude / (2 * pi) * (ln(1 + y ** 2) / y - 2 * arctan(y))

  Dividing by y may cause numerical instability when y is close to zero. Small
  value epsilon is introduced to prevent it.

  When density is smaller than epsilon, the exchange energy density is computed
  by its series expansion at y=0:

  exchange energy density = amplitude / (2 * pi) * (-y + y ** 3 / 6)

  Note the exchange energy density converge to constant -amplitude / 2 at high
  density limit.

  Args:
    density: Float numpy array with shape (num_grids,).
    amplitude: Float, parameter of exponential Coulomb interaction.
    kappa: Float, parameter of exponential Coulomb interaction.
    epsilon: Float, a constant for numerical stability.

  Returns:
    Float numpy array with shape (num_grids,).
  """
  y = jnp.pi * density / kappa
  return jnp.where(
      y > epsilon,
      amplitude / (2 * jnp.pi) * (jnp.log(1 + y ** 2) / y - 2 * jnp.arctan(y)),
      amplitude / (2 * jnp.pi) * (-y + y ** 3 / 6))


@tree_util.Partial
def exponential_coulomb_uniform_correlation_density(
    density,
    amplitude=constants.EXPONENTIAL_COULOMB_AMPLITUDE,
    kappa=constants.EXPONENTIAL_COULOMB_KAPPA):
  """Exchange energy density for uniform gas with exponential coulomb.

  Equation 24 in the following paper provides the correlation energy per length
  for 1d uniform gas with exponential coulomb interaction.

  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  y = pi * density / kappa
  correlation energy per length
      = -amplitude * kappa * y ** 2 / (pi ** 2) / (
        alpha + beta * sqrt(y) + gamma * y + delta * sqrt(y ** 3)
        + eta * y ** 2 + sigma * sqrt(y ** 5)
        + nu * pi * kappa ** 2 / amplitude * y ** 3)

  correlation energy density
      = correlation energy per length * pi / (kappa * y)
      = -amplitude * y / pi / (
        alpha + beta * sqrt(y) + gamma * y + delta * sqrt(y ** 3)
        + eta * y ** 2 + sigma * sqrt(y ** 5)
        + nu * pi * kappa ** 2 / amplitude * y ** 3)

  Note the correlation energy density converge to zero at high density limit.

  Args:
    density: Float numpy array with shape (num_grids,).
    amplitude: Float, parameter of exponential Coulomb interaction.
    kappa: Float, parameter of exponential Coulomb interaction.

  Returns:
    Float numpy array with shape (num_grids,).
  """
  y = jnp.pi * density / kappa
  alpha = 2.
  beta = -1.00077
  gamma = 6.26099
  delta = -11.9041
  eta = 9.62614
  sigma = -1.48334
  nu = 1.
  # The derivative of sqrt is not defined at y=0, we use two jnp.where to avoid
  # nan at 0.
  finite_y = jnp.where(y == 0., 1., y)
  out = -amplitude * finite_y / jnp.pi / (
      alpha + beta * jnp.sqrt(finite_y)
      + gamma * finite_y + delta * finite_y ** 1.5
      + eta * finite_y ** 2 + sigma * finite_y ** 2.5
      + nu * jnp.pi * kappa ** 2 / amplitude * finite_y ** 3
      )
  return jnp.where(y == 0., -amplitude * y / jnp.pi / alpha, out)


@tree_util.Partial
def lda_xc_energy_density(density):
  """XC energy density of Local Density Approximation with exponential coulomb.

  One-dimensional mimicking of electronic structure: The case for exponentials.
  Physical Review B 91.23 (2015): 235141.
  https://arxiv.org/pdf/1504.05620.pdf

  Args:
    density: Float numpy array with shape (num_grids,).

  Returns:
    Float numpy array with shape (num_grids,).
  """
  return (
      exponential_coulomb_uniform_exchange_density(density)
      + exponential_coulomb_uniform_correlation_density(density))


def get_lda_xc_energy_density_fn():
  """Gets lda_xc_energy_density() that takes a dummy params.

  Returns:
    lda_xc_energy_density() takes two arguments:
      * density: Float numpy array with shape (num_grids,).
      * params: A dummy argument, not used.
  """
  def lda_xc_energy_density_fn(density, params):
    del params
    return lda_xc_energy_density(density)
  return lda_xc_energy_density_fn
