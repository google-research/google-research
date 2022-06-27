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

"""Tests for tree_utils."""

from absl.testing import absltest

import jax.numpy as jnp

from jaxsel import tree_utils


class TreeUtilsTest(absltest.TestCase):

  def test_global_norm(self):
    tree = (jnp.ones(10), jnp.ones(2) * 5)
    tree_norm = tree_utils.global_norm(tree)

    # The desired norm is the norm of the vector of all entries in tree.
    true_norm = jnp.linalg.norm(jnp.ones(12).at[-2:].set(5))

    assert jnp.allclose(tree_norm, true_norm)


if __name__ == '__main__':
  absltest.main()
