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

"""Simulation of Keplerian orbits."""

from typing import Mapping, Tuple

import chex
import jax.numpy as jnp
import jaxopt


def generate_canonical_coordinates(
    t,
    simulation_parameters,
    check_convergence = False):
  """Generates positions and momentums in polar coordinates."""

  def eccentric_anomaly_to_time(eccentric_anomaly):
    """Maps eccentricity to time."""
    mean_anomaly = eccentric_anomaly - e * jnp.sin(eccentric_anomaly)
    return period * mean_anomaly / (2 * jnp.pi) + t0  # pytype: disable=bad-return-type  # jnp-type

  def fixed_point_func(eccentric_anomaly):
    """Defines a function f such that its fixed point E satisfies eccentricity_to_time(E) = t."""
    return e * jnp.sin(eccentric_anomaly) + (2 * jnp.pi) * (t - t0) / period  # pytype: disable=bad-return-type  # jax-types

  t0, a, m, e, k = (simulation_parameters['t0'], simulation_parameters['a'],
                    simulation_parameters['m'], simulation_parameters['e'],
                    simulation_parameters['k'])

  # First, compute the eccentric_anomaly at this instant.
  period = (2 * jnp.pi * jnp.power(a, 1.5)) / jnp.sqrt(k)
  solver = jaxopt.FixedPointIteration(
      fixed_point_func, maxiter=20, verbose=False)
  eccentric_anomaly_init = t
  eccentric_anomaly = solver.run(eccentric_anomaly_init).params

  # Checks that the solver has converged.
  if check_convergence:
    assert jnp.allclose(eccentric_anomaly_to_time(eccentric_anomaly), t)

  # Then, compute the position in polar coordinates.
  r = a * (1 - e * jnp.cos(eccentric_anomaly))
  phi = 2 * jnp.arctan2(
      jnp.sqrt(1 + e) * jnp.sin(eccentric_anomaly / 2),
      jnp.sqrt(1 - e) * jnp.cos(eccentric_anomaly / 2))

  # Finally, compute the radial and angular momentum.
  f = 2 * jnp.pi * a / (period * jnp.sqrt(1 - (e**2)))
  v_r = f * (e * jnp.sin(phi))
  v_phi = f * (1 + e * jnp.cos(phi))
  p_r = m * v_r
  p_phi = m * r * v_phi

  # Bundle everything up.
  position = jnp.asarray([r, phi])
  momentum = jnp.asarray([p_r, p_phi])
  return position, momentum


def compute_angular_momentum(
    position, momentum,
    simulation_parameters):
  """Computes the angular momentum at these coordinates."""
  del position, simulation_parameters
  p_phi = momentum[1]
  return p_phi  # pytype: disable=bad-return-type  # numpy-scalars


def compute_hamiltonian(position, momentum,
                        simulation_parameters):
  """Computes the Hamiltonian at these coordinates."""
  m, k = simulation_parameters['m'], simulation_parameters['k']
  r = position[0]
  p_r, p_phi = momentum
  return (p_r**2) / (2 * m) + (p_phi**2) / (2 * m * (r**2)) - k / r


def polar_to_cartesian(
    position, momentum,
    simulation_parameters):
  """Converts positions and momentums from polar to Cartesian coordinates."""
  m = simulation_parameters['m']
  r, phi = position
  p_r, p_phi = momentum
  v_r = p_r / m
  v_phi = p_phi / (m * r)
  position_cartesian = jnp.asarray([r * jnp.cos(phi), r * jnp.sin(phi)])
  velocity_cartesian = jnp.asarray([
      v_r * jnp.cos(phi) - r * v_phi * jnp.sin(phi),
      v_r * jnp.sin(phi) + r * v_phi * jnp.cos(phi)
  ])
  momentum_cartesian = velocity_cartesian * m
  return position_cartesian, momentum_cartesian
