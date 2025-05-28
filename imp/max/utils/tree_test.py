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
import chex
import jax
import jax.numpy as jnp

from imp.max.utils import linalg
from imp.max.utils import tree


class ProjectionTest(absltest.TestCase):

  def test_svd_projection(self):
    rank = 2

    def _random_creator(shape, k):
      return jax.random.normal(jax.random.key(k), shape, dtype=jnp.float32)

    @jax.jit
    def _run_projection(array_tree):
      lr_state_tree = tree.tree_low_rank_projector(array_tree=array_tree,
                                                   rank=rank,
                                                   method='svd')
      lr_array_tree = tree.tree_project_array(
          array_tree=array_tree,
          projection_state_tree=lr_state_tree,
          back_projection=False,
      )
      hr_array_tree = tree.tree_project_array(
          array_tree=lr_array_tree,
          projection_state_tree=lr_state_tree,
          back_projection=True,
      )
      return lr_array_tree, hr_array_tree, lr_state_tree

    # Construct a tree that covers all common situations
    array_tree = {
        'key_1': {
            'kernel': _random_creator((4, 5), 0),
        },
        'key_2': {
            'kernel': _random_creator((6, 5), 1),
        },
        'key_3': {
            'bias': _random_creator((4,), 2),
            'kernel': _random_creator((2, 5), 3),
        },
        'key_4': {
            'kernel': _random_creator((4, 2), 4),
        },
    }
    # Run projection under jit
    (projected_array_tree,
     reconstructed_array_tree,
     projection_state_tree) = _run_projection(array_tree)

    # Assertions
    expected_params_tree = jax.tree.map(
        lambda x: linalg._derive_projection_params(x, rank), array_tree)

    def _assert_shapes(array,
                       projection_state,
                       projected_array,
                       reconstructed_array,
                       expected_params):
      expected_projector_shape = expected_params.projector_shape
      expected_projected_shape = expected_params.projected_shape
      if expected_projector_shape:
        chex.assert_shape(projection_state.projector, expected_projector_shape)
      else:
        self.assertIsInstance(
            projection_state, linalg.EmptyLowRankProjectionState)
      chex.assert_shape(projected_array, expected_projected_shape)
      chex.assert_shape(reconstructed_array, array.shape)

    jax.tree.map(_assert_shapes,
                 array_tree,
                 projection_state_tree,
                 projected_array_tree,
                 reconstructed_array_tree,
                 expected_params_tree)


if __name__ == '__main__':
  absltest.main()
