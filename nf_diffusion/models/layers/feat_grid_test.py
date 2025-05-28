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

"""Tests for feat_grid."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from nf_diffusion.models.layers import feat_grid


class FeatGridTest(absltest.TestCase):

  def test_init_feat_grid(self):
    grid = feat_grid.FeatureGrid(128, res=[32, 33])
    x = jnp.zeros((1, 1024, 2))
    rng = jax.random.PRNGKey(0)
    key, rng = jax.random.split(rng)
    params = grid.init(key, x)
    self.assertEqual(params["params"]["features"].shape, (1, 32, 33, 128))

  def test_feat_grid_forward_shape(self):
    grid = feat_grid.FeatureGrid(128, res=[32, 64])
    x = jnp.zeros((3, 1024, 2))
    rng = jax.random.PRNGKey(0)
    key, rng = jax.random.split(rng)
    params = grid.init(key, x)
    y, _ = grid.apply(params, x)
    self.assertEqual(y.shape, (3, 1024, 128))


if __name__ == "__main__":
  absltest.main()
