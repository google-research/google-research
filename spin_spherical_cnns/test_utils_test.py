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

"""Tests spin_spherical_cnns.test_utils."""

import functools
from absl.testing import parameterized
from flax import linen as nn
import jax.numpy as jnp
import tensorflow as tf

from spin_spherical_cnns import spin_spherical_harmonics
from spin_spherical_cnns import test_utils


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=[4, 8, 16], spins=(0, 1, 2))


class TestUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(1, 4, 4, 2, 3)],
                            [(2, 8, 8, 1, 2)])
  def test_get_rotated_pair_shapes(self, shape):
    transformer = _get_transformer()
    *_, num_spins, _ = shape
    spins = jnp.arange(num_spins)
    pair = test_utils.get_rotated_pair(transformer,
                                       shape,
                                       spins,
                                       1.0, 2.0, 3.0)
    self.assertEqual(pair.sphere.shape, shape)
    self.assertEqual(pair.rotated_sphere.shape, shape)

  @parameterized.parameters(1, 3, 10)
  def test_get_rotated_pair_azimuthal_rotation(self, shift):
    """Check that azimuthal rotation corresponds to horizontal shift."""
    resolution = 16
    transformer = _get_transformer()

    spins = (0, 1)
    shape = (2, resolution, resolution, len(spins), 2)
    # Convert shift to azimuthal rotation angle.
    gamma = shift * 2 * jnp.pi / resolution
    # sympy returns nans for wigner-ds when beta==0, hence the 1e-8 here.
    beta = 1e-8
    pair = test_utils.get_rotated_pair(transformer,
                                       shape,
                                       spins,
                                       0.0, beta, gamma)
    shifted_sphere = jnp.roll(pair.sphere, shift, axis=2)

    self.assertAllClose(shifted_sphere, pair.rotated_sphere)

  def test_apply_model_to_rotated_pairs_with_simple_model(self):
    transformer = _get_transformer()
    resolution = 8
    spins = (0, 1)

    # We use a dummy model that doubles its inputs. Outputs of
    # `apply_model_to_rotated_pairs` must be equal and double of the input
    # rotated coefficients.
    class Double(nn.Module):

      @nn.compact
      def __call__(self, inputs):
        return 2 * inputs

    output_1, output_2, pair = test_utils.apply_model_to_rotated_pairs(
        transformer, Double(), resolution, spins)

    self.assertAllClose(2 * pair.rotated_coefficients, output_1)
    self.assertAllClose(2 * pair.rotated_coefficients, output_2)


if __name__ == "__main__":
  tf.test.main()
