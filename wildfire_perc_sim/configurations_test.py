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

"""Tests for configurations."""
from absl.testing import absltest

from jax import numpy as jnp
from jax import random

from wildfire_perc_sim import configurations


class ConfigurationsTest(absltest.TestCase):
  field_shape = (110, 110)
  batched_field_shape = (3, 110, 110)

  def test_density_random_normal(self):
    prng = random.PRNGKey(0)
    for fs in (self.field_shape, self.batched_field_shape):
      density = configurations.density_random_normal(prng, fs, 1.0, 0.25)

      self.assertEqual(density.shape, fs)
      self.assertTrue((density >= 0).all())

  def test_density_bool(self):
    prng = random.PRNGKey(0)
    for fs in (self.field_shape, self.batched_field_shape):
      density = configurations.density_bool(prng, fs, 0.25)

      self.assertEqual(density.shape, fs)
      self.assertTrue(jnp.logical_or(density == 1, density == 0).all())

  def test_density_patchy(self):
    prng = random.PRNGKey(0)
    density = configurations.density_patchy(prng, self.field_shape, 10.0, 0.75,
                                            0.25, 1.0, 0.25)

    self.assertEqual(density.shape, self.field_shape)

  def test_moisture_random_normal(self):
    prng = random.PRNGKey(0)
    for fs in (self.field_shape, self.batched_field_shape):
      moisture = configurations.moisture_random_normal(prng, fs, 1.0, 0.25)

      self.assertEqual(moisture.shape, fs)
      self.assertTrue((moisture >= 0).all())

  def test_wind_uniform(self):
    prng = random.PRNGKey(0)
    for fs in (self.field_shape, self.batched_field_shape):
      if len(fs) == 2:
        wind_components = random.normal(prng, (2,))
      else:
        wind_components = random.normal(prng, (fs[0], 2))
      wind = configurations.wind_uniform(fs, wind_components)

      self.assertEqual(wind.shape, fs + (2,))

  def test_location_random(self):
    prng = random.PRNGKey(0)
    locations = configurations.location_random(prng, ((0, 100), (10, 80)), 10,
                                               None)
    self.assertEqual(locations.shape, (10, 2))

    init_value = jnp.ones_like(locations[:, 0])
    for val in (locations[:, 0] >= 0, locations[:, 0] <= 100,
                locations[:, 1] >= 10, locations[:, 1] <= 80):
      init_value = jnp.logical_and(init_value, val)
    self.assertTrue(init_value.all())

    locations = configurations.location_random(prng, ((0, 100), (10, 80)), 10,
                                               3)
    self.assertEqual(locations.shape, (3, 10, 2))

    init_value = jnp.ones_like(locations[:, :, 0])
    for val in (locations[:, :, 0] >= 0, locations[:, :, 0] <= 100,
                locations[:, :, 1] >= 10, locations[:, :, 1] <= 80):
      init_value = jnp.logical_and(init_value, val)
    self.assertTrue(init_value.all())

  def test_lit_from_pts(self):
    prng = random.PRNGKey(0)
    for fs in (self.field_shape, self.batched_field_shape):
      core_fs = fs if len(fs) == 2 else fs[1:]
      locations = configurations.location_random(
          prng, ((core_fs[0] + 15, core_fs[1] - 15),
                 (core_fs[0] + 25, core_fs[1] - 5)), 10,
          None if len(fs) == 2 else fs[0])

      lit = configurations.lit_from_pts(fs, locations)

      self.assertEqual(lit.shape, fs)

  def test_terrain_slope(self):
    terrain = configurations.terrain_slope(self.field_shape, jnp.ones((1,)),
                                           jnp.ones((1,)))
    self.assertEqual(terrain.shape, self.field_shape)

    terrain = configurations.terrain_slope(self.field_shape,
                                           jnp.ones(self.field_shape),
                                           jnp.ones(self.field_shape))
    self.assertEqual(terrain.shape, self.field_shape)

  def test_terrain_ds(self):
    prng = random.PRNGKey(0)
    terrain = configurations.terrain_ds(prng, self.field_shape, 10.0, 0.5)

    self.assertEqual(terrain.shape, self.field_shape)


if __name__ == '__main__':
  absltest.main()
