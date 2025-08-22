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

"""Tests for utils."""
from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import random

from wildfire_perc_sim import utils


class UtilsTest(absltest.TestCase):

  def test_radius_tensor(self):
    kernel_dim = 3
    radius = jnp.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                        [[1, 0, -1], [1, 0, -1], [1, 0, -1]]])
    radius_computed = utils.radius_tensor(kernel_dim)
    self.assertTrue(jnp.all(radius == radius_computed))

  def test_pad_tensor_3d(self):
    arr = jnp.zeros((10, 10, 3))
    for bc in (utils.BoundaryCondition.INFINITE,
               utils.BoundaryCondition.PERIODIC,
               utils.BoundaryCondition.LATERAL_PERIODIC):
      arr_padded = utils.pad_tensor_3d(arr, (5, 5), bc)
      self.assertIsInstance(arr_padded, jnp.ndarray)
      self.assertEqual(arr_padded.shape, (14, 14, 3))

  def test_set_border(self):
    arr = jnp.zeros((10, 10))
    arr_border = utils.set_border(arr, 1.0, 2)
    self.assertTrue(jnp.all(arr_border[:2, :] == 1))
    self.assertTrue(jnp.all(arr_border[-2:, :] == 1))
    self.assertTrue(jnp.all(arr_border[:, :2] == 1))
    self.assertTrue(jnp.all(arr_border[:, -2:] == 1))
    # Ensure original array is not modified
    self.assertTrue(jnp.all(arr == 0))

  def test_termial(self):
    n = 10
    res = 0
    for i in range(1, n + 1):
      res += i
    self.assertEqual(res, utils.termial(n))

  def test_get_stencil(self):
    for neighborhood_size in range(1, 4):
      neighborhood = utils.get_stencil(neighborhood_size)
      self.assertTrue(
          jnp.all(jnp.linalg.norm(neighborhood, axis=-1) <= neighborhood_size))

  def test_reparameterize(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    mean = jnp.zeros((16, 128))
    logvar = jnp.zeros((16, 128))
    sample = utils.reparameterize(key, mean, logvar)
    self.assertEqual(sample.shape, mean.shape)

  def test_apply_percolation_convolution(self):
    prng = random.PRNGKey(0)
    prng, key1, key2 = random.split(prng, 3)

    state = random.normal(key1, (4, 110, 110, 5))
    kernel1 = random.normal(key2, (3, 4, 3, 3, 5, 16))
    kernel2 = jnp.prod(kernel1, axis=0, keepdims=False)

    state_updated1 = utils.apply_percolation_convolution(kernel1, state)
    state_updated2 = utils.apply_percolation_convolution(kernel2, state)

    self.assertEqual(state_updated1.shape, (*state.shape[:3], 16))
    self.assertEqual(state_updated2.shape, (*state.shape[:3], 16))
    self.assertTrue((state_updated1 == state_updated2).all())

  def test_restructure_sequence_data(self):
    data_shape = (3, 2, 1)
    seq_data = [jnp.ones((4, *data_shape)) for _ in range(5)]
    updated_seq_data = utils.restructure_sequence_data(seq_data)

    self.assertLen(updated_seq_data, 4)
    for s in updated_seq_data:
      self.assertEqual(s.shape, (len(seq_data),) + data_shape)

  def test_restructure_distributed_sequence_data(self):
    data_shape = (3, 2, 1)
    seq_data = [jnp.ones((5, 4, *data_shape)) for _ in range(6)]
    updated_seq_data = utils.restructure_distributed_sequence_data(seq_data)

    self.assertLen(updated_seq_data, 4)
    for s in updated_seq_data:
      self.assertEqual(s.shape, (len(seq_data), 5) + data_shape)

  def test_sigmoid(self):
    # Numerical Stability Tests
    self.assertAlmostEqual(utils.sigmoid(jnp.array([1e4]), 15.0), 1.0)
    self.assertAlmostEqual(utils.sigmoid(jnp.array([0.0]), 15.0), 0.5)
    self.assertAlmostEqual(utils.sigmoid(jnp.array([-1e4]), 15.0), 0.0)

    self.assertTrue(
        jnp.all(
            jnp.logical_not(
                jnp.isnan(
                    jax.grad(lambda x: jnp.sum(utils.sigmoid(x, 15.0)))(
                        jnp.array([1e4, 0.0, 1e-4]))))))

  def test_prepend_dict_keys(self):
    d = {'a': 1, 'b': 2, 'c': 3}
    d_new = utils.prepend_dict_keys(d, 'train/')

    self.assertLen(d_new, 3)
    self.assertLen(d, 3)
    for new_key in ('train/a', 'train/b', 'train/c'):
      self.assertIn(new_key, d_new)
    for old_key in ('a', 'b', 'c'):
      self.assertIn(old_key, d)


if __name__ == '__main__':
  absltest.main()
