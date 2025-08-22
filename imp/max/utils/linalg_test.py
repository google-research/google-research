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

"""Tests for linalg helper functions."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from imp.max.utils import linalg

jax.config.update('jax_threefry_partitionable', False)


class ProjectionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'mxn_m_smaller',
          'array_shape': (4, 5),
          'rank': 2,
      }, {
          'testcase_name': 'mxn_m_bigger',
          'array_shape': (6, 5),
          'rank': 2,
      }, {
          'testcase_name': 'mxn_m_smaller_rank_equal_or_larger',
          'array_shape': (4, 5),
          'rank': 4,
      }, {
          'testcase_name': 'mxn_m_bigger_rank_equal_or_larger',
          'array_shape': (6, 5),
          'rank': 5,
      },
  )
  def test_2d_svd_projection(self, array_shape, rank):
    key = jax.random.key(0)
    array = jax.random.uniform(key, array_shape, dtype=jnp.float32)

    @jax.jit
    def _run_projection(array):
      projection_state = linalg.svd_projector(array, rank=rank)
      projected_array = linalg.project_array(
          array=array,
          projection_state=projection_state,
          back_projection=False,
      )
      reconstructed_array = linalg.project_array(
          array=projected_array,
          projection_state=projection_state,
          back_projection=True,
      )
      return projected_array, reconstructed_array, projection_state

    # Run projection under jit
    (projected_array,
     reconstructed_array,
     projection_state) = _run_projection(array)

    # Derive expected shapes and values
    m, n = array.shape
    if min(m, n) <= rank:
      expected_projector_shape = ()
      expected_projected_shape = array_shape
    elif m <= n:
      expected_projector_shape = (m, rank)
      expected_projected_shape = (rank, n)
    else:
      expected_projector_shape = (rank, n)
      expected_projected_shape = (m, rank)
    expected_reconstruction_norm_error = {
        ((4, 5), 2): 0.87980354,
        ((6, 5), 2): 0.77656335,
        ((4, 5), 4): 0.,
        ((6, 5), 5): 0.,
    }

    # Assertions
    if expected_projector_shape:
      chex.assert_shape(projection_state.projector, expected_projector_shape)
    else:
      self.assertIsInstance(
          projection_state, linalg.EmptyLowRankProjectionState)
    chex.assert_shape(projected_array, expected_projected_shape)
    chex.assert_shape(reconstructed_array, array.shape)
    self.assertEqual(
        jnp.linalg.norm(reconstructed_array - array),
        expected_reconstruction_norm_error[(array_shape, rank)]
    )


if __name__ == '__main__':
  absltest.main()
