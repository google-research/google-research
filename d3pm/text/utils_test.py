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

"""Tests for utils.py."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from d3pm.text import types
from d3pm.text import utils


class UtilsTest(absltest.TestCase):

  def test_build_batch(self):
    batch = {'inputs': jnp.zeros((100, 10)), 'targets': jnp.zeros((100, 5, 5))}
    shapes = jax.eval_shape(lambda: batch)

    info = types.DatasetInfo(features=['inputs', 'targets'], shapes=shapes)
    batch_reconstructed = utils.build_batch_from_info(
        info, prune_batch_dim=False)

    chex.assert_trees_all_close(batch, batch_reconstructed)

  def test_dataset_info(self):
    batch = {'inputs': jnp.zeros((100, 10)), 'targets': jnp.zeros((100, 5, 5))}
    info = utils.get_dataset_info_from_batch(batch=batch)

    batch_reconstructed = utils.build_batch_from_info(
        info, prune_batch_dim=False)

    chex.assert_trees_all_close(batch, batch_reconstructed)


if __name__ == '__main__':
  absltest.main()
