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

"""Tests for grid_utils."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import grid_utils
from jax import random
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np


# The number of coordinates to test.
NUM_TEST_COORDS = 100


class GridUtilsTest(chex.TestCase, parameterized.TestCase):

  def wrap_fn(self, fn):
    """Wraps the given function with checkify and chex.variant."""
    wrapped_fn = checkify.checkify(self.variant(fn))
    return lambda *args, **kwargs: wrapped_fn(*args, **kwargs)[1]

  @chex.variants(with_device=True)
  @parameterized.product(
      fill_value=[-42.1234, 0.0, 100.1],
      grid_size=[2, 3, 100],
      num_features=[1, 3],
  )
  def test_trilerp_grid_all_same(self, fill_value, grid_size, num_features):
    """Test trilerp with a grid containing the same value everywhere."""
    rng = random.PRNGKey(0)
    shape = (grid_size, grid_size, grid_size, num_features)
    grid = jnp.full(shape, fill_value)
    # We add 0.5 because grid_utils.trilerp expects pixel centers.
    coordinates = (
        random.uniform(
            rng, (NUM_TEST_COORDS, 3), minval=0, maxval=grid_size - 1
        )
        + 0.5
    )
    trilerp = functools.partial(grid_utils.trilerp, datastructure='grid')
    results = self.wrap_fn(trilerp)(grid, coordinates)
    np.testing.assert_allclose(results, fill_value, rtol=1e-6)

  @chex.variants(with_device=True)
  @parameterized.product(
      fill_value=[-42.1234, 0.0, 100.1],
      grid_size=[2, 3, 100],
      hash_map_size=[1, 128, 4096],
      num_features=[1, 3],
  )
  def test_trilerp_hash_all_same(
      self, fill_value, grid_size, hash_map_size, num_features
  ):
    """Test trilerp with a hash grid containing the same value everywhere."""
    rng = random.PRNGKey(0)
    shape = (hash_map_size, num_features)
    hash_grid = jnp.full(shape, fill_value)
    # We add 0.5 because grid_utils.trilerp expects pixel centers.
    coordinates = (
        random.uniform(
            rng, (NUM_TEST_COORDS, 3), minval=0, maxval=grid_size - 1
        )
        + 0.5
    )
    trilerp = functools.partial(grid_utils.trilerp, datastructure='hash')
    results = self.wrap_fn(trilerp)(hash_grid, coordinates)
    np.testing.assert_allclose(results, fill_value, rtol=1e-6)

  def test_trilerp_invalid_datastructure(self):
    """Tests that an invalid value for `datastructure` raises an error."""
    with self.assertRaisesRegex(ValueError, 'datastructure'):
      grid_utils.trilerp(
          jnp.zeros((1,)),
          jnp.zeros((3,)),
          datastructure='this_is_invalid',
      )


if __name__ == '__main__':
  absltest.main()
