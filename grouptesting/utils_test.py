# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tests for the utils functions."""

from absl.testing import absltest
import jax
import jax.numpy as np
import jax.test_util

from grouptesting import utils


class UtilsTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def test_unique(self):
    arr = np.array(
        [[True, True, True], [True, True, False], [True, True, False]])
    self.assertEqual(utils.unique(self.rng, arr), 2)

  def test_select_from_sizes(self):
    sizes = np.array([1, 4, 8, 2])
    prior = np.array([0.1])
    self.assertArraysAllClose(
        utils.select_from_sizes(prior, sizes),
        prior[0] * np.ones_like(sizes), check_dtypes=True)

    prior = np.array([0.4, 0.2, 0.1])
    expected = np.array([0.4, 0.1, 0.1, 0.2])
    self.assertArraysAllClose(
        utils.select_from_sizes(prior, sizes), expected, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
