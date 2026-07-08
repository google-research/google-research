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

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Unit tests for camera module and simulator builders."""

# pylint: disable=invalid-name,g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax.numpy as jnp
from jax_rtm import AtmosphereState
from jax_rtm import camera
from jax_rtm import GeometryState
from jax_rtm import SurfaceState


class CameraTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.n_levels = 5
    self.height = 2
    self.width = 2
    self.n_pixels = self.height * self.width

    self.p_prof = jnp.linspace(10000.0, 100000.0, self.n_levels)
    self.t_prof = jnp.linspace(220.0, 290.0, self.n_levels)
    self.q_prof = jnp.linspace(1e-5, 1e-2, self.n_levels)
    self.clwc = jnp.zeros(self.n_levels)
    self.ciwc = jnp.linspace(0.0, 1e-5, self.n_levels)
    self.cswc = jnp.zeros(self.n_levels)
    self.crwc = jnp.zeros(self.n_levels)
    self.tau_ray = jnp.zeros(self.n_levels)
    self.omega_ray = jnp.zeros(self.n_levels)

    self.atmosphere_batch = AtmosphereState(
        p_prof=jnp.tile(self.p_prof, (self.n_pixels, 1)),
        T_prof=jnp.tile(self.t_prof, (self.n_pixels, 1)),
        q_prof=jnp.tile(self.q_prof, (self.n_pixels, 1)),
        clwc=jnp.tile(self.clwc, (self.n_pixels, 1)),
        ciwc_nat=jnp.tile(self.ciwc, (self.n_pixels, 1)),
        cswc=jnp.tile(self.cswc, (self.n_pixels, 1)),
        crwc=jnp.tile(self.crwc, (self.n_pixels, 1)),
        tau_ray_prof=jnp.tile(self.tau_ray, (self.n_pixels, 1)),
        omega_ray_prof=jnp.tile(self.omega_ray, (self.n_pixels, 1)),
    )

    self.surface_batch = SurfaceState(
        is_land=jnp.array([True, False, True, False]),
        sd=jnp.zeros(self.n_pixels),
        skt=jnp.array([290.0, 292.0, 288.0, 291.0]),
        lat=jnp.array([10.0, 20.0, 30.0, 40.0]),
        u10=jnp.zeros(self.n_pixels),
        v10=jnp.zeros(self.n_pixels),
    )

    self.geometry_batch = GeometryState(
        mu_0=jnp.ones(self.n_pixels) * 0.5,
        mu_view=jnp.ones(self.n_pixels) * 0.8,
    )

  def test_load_configs(self):
    # Test dictionary input override
    custom_params = {
        "satellite": "goes",
        "t_land_bias": -1.0,
        "t_water_bias": -0.5,
        "ciwc_multiplier": 4.0,
    }
    _, micro_params, t_land, t_water = camera._load_configs(
        custom_params
    )  # pylint: disable=protected-access
    self.assertEqual(t_land, -1.0)
    self.assertEqual(t_water, -0.5)
    self.assertEqual(micro_params.ciwc_multiplier, 4.0)

    # Test invalid input type
    with self.assertRaises(TypeError):
      camera._load_configs(123)  # pylint: disable=protected-access

  def test_get_batch_simulator(self):
    batch_sim = camera.get_batch_simulator(
        {}, n_streams=4, num_doubling_steps=2, mu_0=0.5
    )
    b84, b103, b123, tau1 = batch_sim(
        self.atmosphere_batch, self.surface_batch, self.geometry_batch
    )
    self.assertEqual(b84.shape, (self.n_pixels,))
    self.assertEqual(b103.shape, (self.n_pixels,))
    self.assertEqual(b123.shape, (self.n_pixels,))
    self.assertTrue(jnp.all(jnp.isfinite(b84)))
    self.assertTrue(jnp.all(jnp.isfinite(tau1)))

  def test_ash_rgb_compositor(self):
    b84 = jnp.array([280.0, 285.0])
    b103 = jnp.array([282.0, 287.0])
    b123 = jnp.array([281.0, 286.0])
    rgb = camera.ash_rgb_compositor(b84, b103, b123)
    self.assertEqual(rgb.shape, (2, 3))
    self.assertTrue(jnp.all(jnp.isfinite(rgb)))
    self.assertTrue(jnp.all(rgb >= 0.0))
    self.assertTrue(jnp.all(rgb <= 1.0))


if __name__ == "__main__":
  absltest.main()
