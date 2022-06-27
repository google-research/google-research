# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for jax_dft.np_utils."""

from absl.testing import absltest
from jax.config import config
import jax.numpy as jnp
import numpy as np

from jax_dft import np_utils


# Set the default dtype as float64
config.update('jax_enable_x64', True)


class OnpUtilsTest(absltest.TestCase):

  def test_flatten(self):
    (tree, shapes), vec = np_utils.flatten(
        [(jnp.array([1, 2, 3]), (jnp.array([4, 5]))), jnp.array([99])])
    self.assertIsInstance(vec, np.ndarray)
    np.testing.assert_allclose(vec, [1., 2., 3., 4., 5., 99.])
    self.assertEqual(shapes, [(3,), (2,), (1,)])

    # unflatten should convert 1d array back to pytree.
    params = np_utils.unflatten((tree, shapes), vec)
    self.assertIsInstance(params[0][0], np.ndarray)
    np.testing.assert_allclose(params[0][0], [1., 2., 3.])
    self.assertIsInstance(params[0][1], np.ndarray)
    np.testing.assert_allclose(params[0][1], [4., 5.])
    self.assertIsInstance(params[1], np.ndarray)
    np.testing.assert_allclose(params[1], [99.])

  def test_get_exact_h_atom_density(self):
    grids = np.linspace(-10, 10, 1001)
    dx = 0.02
    displacements = np.array([
        grids,
        grids - 2,
    ])
    density = np_utils._get_exact_h_atom_density(displacements, dx)
    self.assertIsInstance(density, np.ndarray)
    self.assertEqual(density.shape, (2, 1001))
    np.testing.assert_allclose(np.sum(density, axis=1) * dx, [1, 1])
    self.assertAlmostEqual(density[0][501], 0.40758, places=4)
    self.assertAlmostEqual(density[1][601], 0.40758, places=4)

  def test_get_exact_h_atom_density_wrong_shape(self):
    grids = np.linspace(-10, 10, 1001)
    dx = 0.02
    with self.assertRaisesRegex(
        ValueError, 'displacements is expected to have ndim=2, but got 1'):
      np_utils._get_exact_h_atom_density(grids, dx)

  def test_spherical_superposition_density(self):
    density = np_utils.spherical_superposition_density(
        grids=np.linspace(-10, 10, 1001),
        locations=np.array([0, 2]),
        nuclear_charges=np.array([1, 2]))
    self.assertIsInstance(density, np.ndarray)
    self.assertEqual(density.shape, (1001,))
    np.testing.assert_allclose(np.sum(density) * 0.02, 3)


if __name__ == '__main__':
  absltest.main()
