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

"""Tests for wildfire_simulator."""
from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import random

from wildfire_perc_sim import utils
from wildfire_perc_sim import wildfire_simulator


class WildfireSimulatorTest(absltest.TestCase):

  def test_fire_active(self):
    lit = jnp.zeros((1, 10, 10))
    self.assertFalse(wildfire_simulator.fire_active(lit))
    # Fire at one point
    lit_fire_pt = lit.at[0, 3, 3].set(1)
    self.assertTrue(wildfire_simulator.fire_active(lit_fire_pt))
    # Fire has exhausted if it reaches the walls/edge of the field
    lit_fire_edge = lit.at[0, 0, :].set(1)
    self.assertFalse(wildfire_simulator.fire_active(lit_fire_edge))

  def test_simulation(self):
    prng = random.PRNGKey(0)
    field_shape = (128, 128)
    neighborhood_size = 5
    batch_size = 2
    batched_field_shape = (batch_size,) + field_shape

    prng, key = random.split(prng)
    terrain = random.normal(key, batched_field_shape)

    prng, key = random.split(prng)
    wind = random.normal(key, batched_field_shape + (2,))

    prng, key = random.split(prng)
    slope_alpha = random.normal(key, (batch_size, 1))

    prng, key = random.split(prng)
    wind_alpha = random.normal(key, (batch_size, 1))

    prng, key = random.split(prng)
    density = random.normal(key, batched_field_shape)
    nominal_ignition_heat = jnp.array([3.0])

    prng, key = random.split(prng)
    moisture = random.normal(key, batched_field_shape)
    prng, key = random.split(prng)
    moisture_alpha = random.normal(key, (batch_size, 1))

    simprop = wildfire_simulator.SimulatorProperties.create(
        neighborhood_size=neighborhood_size,
        boundary_condition=utils.BoundaryCondition.INFINITE,
        nominal_ignition_heat=nominal_ignition_heat,
        burn_duration=jnp.array([5.0]))
    simparams = wildfire_simulator.SimulatorParameters(
        slope_alpha=slope_alpha,
        wind_alpha=wind_alpha,
        moisture_alpha=moisture_alpha)
    fieldprop = wildfire_simulator.FieldProperties(
        moisture=moisture, terrain=terrain, wind=wind, density=density)

    # Testing batched kernel generation
    dynamic_kernel = wildfire_simulator.parameterized_generate_dynamic_kernel(
        simprop, fieldprop, simparams)

    self.assertEqual(dynamic_kernel.shape,
                     (batch_size, *field_shape, *simprop.base_kernel.shape))

    ignition_heat = wildfire_simulator.get_ignition_heat(
        simprop, fieldprop, simparams)

    self.assertEqual(ignition_heat.shape, batched_field_shape)

    prng, key = random.split(prng)
    lit_source = random.randint(key, (batch_size, *field_shape), 0, 2)
    lit_source = utils.set_border(lit_source, 0, 16)
    bstate = wildfire_simulator.start_fire(lit_source, ignition_heat)

    self.assertEqual(bstate.lit.shape, batched_field_shape)
    self.assertEqual(bstate.heat.shape, batched_field_shape)
    self.assertEqual(bstate.fire.shape, batched_field_shape)

    count = 0

    while wildfire_simulator.fire_active(bstate.lit) and count < 10:
      bstate = wildfire_simulator.burn_step(bstate, simprop, dynamic_kernel,
                                            ignition_heat)

      self.assertEqual(bstate.lit.shape, batched_field_shape)
      self.assertEqual(bstate.heat.shape, batched_field_shape)
      self.assertEqual(bstate.fire.shape, batched_field_shape)
      self.assertEqual(bstate.burnt.shape, batched_field_shape)

      count += 1

  def test_differentiable_simulation(self):
    prng = random.PRNGKey(0)
    field_shape = (128, 128)
    neighborhood_size = 5
    batch_size = 2
    batched_field_shape = (batch_size,) + field_shape

    prng, key = random.split(prng)
    hidden_state = random.normal(key, batched_field_shape + (6,))

    terrain = hidden_state[Ellipsis, 0]
    wind = hidden_state[Ellipsis, 1:3]
    moisture = jnp.abs(hidden_state[Ellipsis, 3])
    density = jnp.abs(hidden_state[Ellipsis, 4])

    prng, key1, key2, key3 = random.split(prng, 4)
    slope_alpha = jnp.abs(random.normal(key1, (batch_size, 1)))
    wind_alpha = jnp.abs(random.normal(key2, (batch_size, 1)))
    moisture_alpha = jnp.abs(random.normal(key3, (batch_size, 1)))

    nominal_ignition_heat = 3.0

    simprop = wildfire_simulator.SimulatorProperties.create(
        neighborhood_size=neighborhood_size,
        boundary_condition=utils.BoundaryCondition.INFINITE,
        nominal_ignition_heat=nominal_ignition_heat,
        burn_duration=jnp.array([5.0]),
        sigmoid_coefficient=15.0)
    simparams = wildfire_simulator.SimulatorParameters(
        slope_alpha=slope_alpha,
        wind_alpha=wind_alpha,
        moisture_alpha=moisture_alpha)
    fieldprop = wildfire_simulator.FieldProperties(
        moisture=moisture, terrain=terrain, wind=wind, density=density)

    ignition_heat = wildfire_simulator.get_ignition_heat(
        simprop, fieldprop, simparams, True)

    prng, key = random.split(prng)
    lit_source = random.randint(key, (batch_size, *field_shape), 0, 2)
    lit_source = utils.set_border(lit_source, 0, 16)
    bstate = wildfire_simulator.start_fire(lit_source, ignition_heat)
    bstate = bstate.replace(
        lit=bstate.lit.astype('float32'),
        fire=bstate.fire.astype('float32'),
        burnt=bstate.burnt.astype('float32'),
        heat=bstate.heat.astype('float32'))

    def loss_function(
        bstate,
        simparams,
        fieldprop):
      ignition_heat = wildfire_simulator.get_ignition_heat(
          simprop, fieldprop, simparams, True)

      dynamic_kernel = wildfire_simulator.parameterized_generate_dynamic_kernel(
          simprop, fieldprop, simparams)

      bstate = wildfire_simulator.approximate_burn_step(bstate, simprop,
                                                        dynamic_kernel,
                                                        ignition_heat)

      return (bstate.lit.mean() + bstate.heat.mean() + bstate.fire.mean() +
              bstate.burnt.mean())

    grad_fn = jax.grad(loss_function, argnums=(0, 1, 2))
    dbstate, dsimparams, dfieldprop = grad_fn(bstate, simparams, fieldprop)

    def isfinite(x):
      return jnp.all(jnp.isfinite(x))

    self.assertTrue(isfinite(dbstate.lit))
    self.assertTrue(isfinite(dbstate.heat))
    self.assertTrue(isfinite(dbstate.fire))
    self.assertTrue(isfinite(dbstate.burnt))

    self.assertTrue(isfinite(dsimparams.slope_alpha))
    self.assertTrue(isfinite(dsimparams.wind_alpha))
    self.assertTrue(isfinite(dsimparams.moisture_alpha))

    self.assertTrue(isfinite(dfieldprop.terrain))
    self.assertTrue(isfinite(dfieldprop.density))
    self.assertTrue(isfinite(dfieldprop.moisture))
    self.assertTrue(isfinite(dfieldprop.wind))


if __name__ == '__main__':
  absltest.main()
