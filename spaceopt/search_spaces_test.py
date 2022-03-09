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

"""Tests for search_spaces."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from spaceopt import search_spaces


class SearchSpacesTest(absltest.TestCase):

  def setUp(self):

    super(SearchSpacesTest, self).setUp()
    self.search_space_base = jnp.array([[-5, 10], [0, 15]])
    self.reduce_rate = 1 / 2
    self.key = jax.random.PRNGKey(0)

  def test_check_volume(self):
    """Test that the generated search space has the right volume ratio."""
    new_search_space = search_spaces.generate_search_space_reduce_vol(
        self.key, self.search_space_base, self.reduce_rate)
    base_vol = search_spaces.eval_vol(self.search_space_base)
    new_vol = search_spaces.eval_vol(new_search_space)
    self.assertEqual(new_vol / base_vol, self.reduce_rate)


if __name__ == '__main__':
  absltest.main()
